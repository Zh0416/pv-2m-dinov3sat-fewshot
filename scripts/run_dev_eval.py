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

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import write_json
from pvfs.metrics import aggregate_binary_episodes, compare_methods, evaluate_binary_episode
from pvfs.modules import DeformableBoundaryConfig, ThreeModulePipeline, get_b0_32_config, get_clean_core_config


def run_dev_evaluation(
    *,
    split_dir: Path,
    cache_pt: Path,
    deform_config_json: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run paired 120-episode PV-100scence Dev evaluation on B0_32_Ref, CleanCore_M3Off, and CleanCore_M3DeformGrad."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id: dict[str, dict[str, Any]] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    rgb_by_id: dict[str, np.ndarray] = {}
    for row in dev_records:
        rid = row["id"]
        rgb_by_id[rid] = read_verified_rgb(img_dir, row)
        mask = read_verified_mask(lbl_dir, row)
        mask_by_id[rid] = mask
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks

    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in all_ids}

    def_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")
    variants = (
        ("B0_32_Ref", ThreeModulePipeline(get_b0_32_config())),
        ("CleanCore_M3Off", ThreeModulePipeline(get_clean_core_config())),
        ("CleanCore_M3DeformGrad", ThreeModulePipeline(get_clean_core_config(), refiner_cfg=def_cfg)),
    )

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    methods_res: dict[str, dict[str, Any]] = {v: {} for v, _ in variants}
    for k in sorted(ep_by_k.keys()):
        ep_evals: dict[str, list[dict]] = {v: [] for v, _ in variants}
        for ep in ep_by_k[k]:
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]
            for vname, pipe in variants:
                out = pipe.run_episode(
                    caches,
                    tokens_by_key,
                    mask_by_id,
                    aff_by_id,
                    sup_ids,
                    q_ids,
                    rgb_by_id=(rgb_by_id if vname == "CleanCore_M3DeformGrad" else None),
                    tau_grad_per_px=0.20,
                )
                probs_512, scores_32 = out.to_legacy_outputs()
                masks_512 = out.final_masks_512()
                ep_evals[vname].append(evaluate_binary_episode(q_recs, masks_512, probs_512, scores_32))

        for vname, _ in variants:
            methods_res[vname][str(k)] = aggregate_binary_episodes(ep_evals[vname], episodes_per_fold=10)
            m = methods_res[vname][str(k)]["macro"]
            print(
                f"[Dev K={k}] {vname:22s}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} NegFP={m['neg_fp']:.4f}",
                flush=True,
            )

    comparisons = {
        cname: {str(k): compare_methods(methods_res[cand][str(k)], methods_res[ref][str(k)]) for k in sorted(ep_by_k.keys())}
        for cname, cand, ref in (
            ("CleanCore_M3Off_vs_B0_32_Ref", "CleanCore_M3Off", "B0_32_Ref"),
            ("CleanCore_M3DeformGrad_vs_CleanCore_M3Off", "CleanCore_M3DeformGrad", "CleanCore_M3Off"),
            ("CleanCore_M3DeformGrad_vs_B0_32_Ref", "CleanCore_M3DeformGrad", "B0_32_Ref"),
        )
    }
    summary = {
        "schema": "pv-dinov3sat-clean-core-dev-v1",
        "device": run_device,
        "elapsed_sec": float(time.perf_counter() - t0),
        "methods": methods_res,
        "comparisons": comparisons,
    }
    write_json(output_dir / "clean_core_dev_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run converged CleanCore 120-episode Dev evaluation.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--deform-config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "clean_core_dev")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_dev_evaluation(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        deform_config_json=args.deform_config_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
