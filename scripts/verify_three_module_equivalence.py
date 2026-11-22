from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.io import write_json
from pvfs.iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from pvfs.iter5_methods import run_iter5_episode
from pvfs.metrics import aggregate_episodes, evaluate_episode
from pvfs.modules import ThreeModulePipeline, get_b0_32_config, get_m3_pruned_config
from tests.test_three_module_equivalence import _trace_legacy_iter5_intermediates


def verify_full_dev_equivalence(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run full 120-episode L vs R equivalence audit on locked 79 Dev images."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    _, lbl_dir = resolve_data_dirs(None, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id, mask_by_id = {}, {}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in all_ids}

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    report: dict[str, Any] = {
        "schema": "pv-2m-dinov3sat-three-module-equivalence-v1",
        "device": run_device,
        "variants": {},
    }

    for cfg in (get_m3_pruned_config(), get_b0_32_config()):
        pipe = ThreeModulePipeline(cfg)
        v_report: dict[str, Any] = {"by_k": {}, "all_passed": True}

        for k in sorted(ep_by_k.keys()):
            max_bias_diff = 0.0
            max_temp_diff = 0.0
            max_tmid_diff = 0.0
            max_s_sem_diff = 0.0
            max_p_sem_diff = 0.0
            max_s_res_diff = 0.0
            max_p_res_diff = 0.0
            max_p512_diff = 0.0
            max_s32_diff = 0.0
            total_mask_flips = 0
            total_mfinal_mcoarse_diffs = 0
            n_query_checks = 0
            self_rescue_triggers = 0

            ep_L_evals, ep_R_evals = [], []

            for ep in ep_by_k[k]:
                sup_ids = ep["support_ids"]
                q_ids = ep["query_ids"]
                q_recs = [records_by_id[qid] for qid in q_ids]

                p512_L, s32_L = run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids, cfg)
                trace_L = _trace_legacy_iter5_intermediates(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids, cfg)

                out_R = pipe.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
                p512_R, s32_R = out_R.to_legacy_outputs()

                max_bias_diff = max(max_bias_diff, abs(out_R.calibration.bias - trace_L["bias"]))
                max_temp_diff = max(max_temp_diff, abs(out_R.calibration.temp - trace_L["temp"]))
                if trace_L["mid_veto_ref"] is not None and out_R.calibration.mid_veto_ref is not None:
                    max_tmid_diff = max(max_tmid_diff, abs(out_R.calibration.mid_veto_ref - trace_L["mid_veto_ref"]))

                for qid in q_ids:
                    n_query_checks += 1
                    qL = trace_L["per_q"][qid]
                    sem_R = out_R.semantic_by_id[qid]
                    rec_R = out_R.recovery_by_id[qid]
                    bnd_R = out_R.boundary_by_id[qid]

                    if rec_R.diagnostics.get("self_rescue_triggered", False):
                        self_rescue_triggers += 1

                    max_s_sem_diff = max(max_s_sem_diff, float((sem_R.s_sem - qL["s_sem"]).abs().max().item()))
                    max_p_sem_diff = max(max_p_sem_diff, float((sem_R.p_sem - qL["p_sem"]).abs().max().item()))
                    max_s_res_diff = max(max_s_res_diff, float((rec_R.s_res - qL["s_res"]).abs().max().item()))
                    max_p_res_diff = max(max_p_res_diff, float((rec_R.p_res - qL["p_res"]).abs().max().item()))
                    max_p512_diff = max(max_p512_diff, float(np.max(np.abs(p512_R[qid] - p512_L[qid]))))
                    max_s32_diff = max(max_s32_diff, float(np.max(np.abs(s32_R[qid] - s32_L[qid]))))

                    m_L = p512_L[qid] >= 0.50
                    total_mask_flips += int(np.logical_xor(bnd_R.m_coarse, m_L).sum())
                    total_mfinal_mcoarse_diffs += int(np.logical_xor(bnd_R.m_final, bnd_R.m_coarse).sum())

                ep_L_evals.append(evaluate_episode(q_recs, p512_L, s32_L))
                ep_R_evals.append(evaluate_episode(q_recs, p512_R, s32_R))

            agg_L = aggregate_episodes(ep_L_evals)
            agg_R = aggregate_episodes(ep_R_evals)

            passed_k = bool(
                total_mask_flips == 0
                and total_mfinal_mcoarse_diffs == 0
                and max_p512_diff == 0.0
                and max_s32_diff == 0.0
                and max_s_sem_diff == 0.0
                and max_s_res_diff == 0.0
            )
            v_report["all_passed"] = v_report["all_passed"] and passed_k
            v_report["by_k"][str(k)] = {
                "episodes": len(ep_by_k[k]),
                "query_checks": n_query_checks,
                "self_rescue_triggers": self_rescue_triggers,
                "max_bias_diff": max_bias_diff,
                "max_temp_diff": max_temp_diff,
                "max_tmid_diff": max_tmid_diff,
                "max_s_sem_diff": max_s_sem_diff,
                "max_p_sem_diff": max_p_sem_diff,
                "max_s_res_diff": max_s_res_diff,
                "max_p_res_diff": max_p_res_diff,
                "max_probs_512_diff": max_p512_diff,
                "max_scores_32_diff": max_s32_diff,
                "total_mask_flips": total_mask_flips,
                "total_mfinal_mcoarse_diffs": total_mfinal_mcoarse_diffs,
                "exact_bitwise_pass": passed_k,
                "macro_L": agg_L["macro"],
                "macro_R": agg_R["macro"],
            }
            print(
                f"[{cfg.name} K={k}] checks={n_query_checks} max_p512_diff={max_p512_diff:.2e} "
                f"max_s32_diff={max_s32_diff:.2e} mask_flips={total_mask_flips} "
                f"Dice_L={agg_L['macro']['dice']:.4f} Dice_R={agg_R['macro']['dice']:.4f} "
                f"BF1@2_R={agg_R['macro']['boundary_f1']:.4f} NegFP_R={agg_R['macro']['neg_fp']:.4f} PASS={passed_k}",
                flush=True,
            )

        report["variants"][cfg.name] = v_report

    report["elapsed_sec"] = float(time.perf_counter() - t0)
    write_json(output_dir / "equivalence_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact L vs R equivalence across all 120 Dev episodes.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "three_module_equiv")
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    verify_full_dev_equivalence(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        output_dir=args.output_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
