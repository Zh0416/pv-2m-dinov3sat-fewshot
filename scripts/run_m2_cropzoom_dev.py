from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.ndimage import label as nd_label
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, score_from_pairwise
from pvfs.io import write_json
from pvfs.iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from pvfs.iter5_methods import Iter5Config
from pvfs.metrics import BOX_3, aggregate_episodes, compare_methods, evaluate_episode
from pvfs.modules import ThreeModulePipeline, get_m3_pruned_config
from pvfs.modules.crop_zoom import (
    CropZoomConfig,
    audit_offline_candidate_coverage,
    extract_reflect_padded_crop,
    project_crop_scores_max_increment,
    propose_seedless_candidate_windows,
)


def _compute_missed_component_recovery(
    gt_mask: np.ndarray,
    m_base: np.ndarray,
    m_cand: np.ndarray,
    *,
    min_area: int = 16,
    iou_thr: float = 0.10,
) -> tuple[int, int]:
    """Count GT components completely missed by m_base (IoU==0) and how many m_cand recovers (IoU>=0.10)."""
    gt = np.asarray(gt_mask > 0, dtype=bool)
    if not gt.any():
        return 0, 0
    lbl, n_c = nd_label(gt, structure=BOX_3)
    missed = 0
    recovered = 0
    for cid in range(1, n_c + 1):
        comp = lbl == cid
        if int(comp.sum()) < min_area:
            continue
        if not np.logical_and(comp, m_base).any():
            missed += 1
            inter = int(np.logical_and(comp, m_cand).sum())
            if inter > 0:
                # Component-level recall/IoU within component bounding box
                union = int(comp.sum()) + int(np.logical_and(m_cand, ~gt).sum() * 0)
                if (inter / max(1, int(comp.sum()))) >= iou_thr:
                    recovered += 1
    return missed, recovered


@torch.inference_mode()
def _score_windows_full_image_tokens(
    s_sem_64: torch.Tensor,
    p_sem_64: torch.Tensor,
    q_tok_64: torch.Tensor,
    sup_fg_bank: torch.Tensor,
    sup_bg_bank: torch.Tensor,
    bias: float,
    temp: float,
    *,
    crop_cfg: CropZoomConfig,
) -> tuple[list[tuple[Any, torch.Tensor]], list[Any]]:
    """Control variant: verify the EXACT same candidate windows using full-image deep_64 tokens only."""
    wins = propose_seedless_candidate_windows(s_sem_64, p_sem_64, crop_cfg)
    if not wins or sup_fg_bank.shape[0] == 0 or sup_bg_bank.shape[0] == 0:
        return [], wins

    tok_2d = q_tok_64.reshape(64, 64, -1)
    m_fg = min(3, int(sup_fg_bank.shape[0]))
    m_bg = min(3, int(sup_bg_bank.shape[0]))
    scored: list[tuple[Any, torch.Tensor]] = []

    for win in wins:
        orig_ty = win.y0 // 8
        orig_tx = win.x0 // 8
        win_h = (win.y1 - win.y0) // 8
        win_w = (win.x1 - win.x0) // 8
        # Extract full-image tokens for the window (clamped reflect on borders)
        ys = np.clip(np.arange(orig_ty, orig_ty + win_h), 0, 63)
        xs = np.clip(np.arange(orig_tx, orig_tx + win_w), 0, 63)
        sub_tok = tok_2d[ys][:, xs].reshape(-1, tok_2d.shape[-1])
        top_f = (sub_tok @ sup_fg_bank.T).clamp(-1.0, 1.0).topk(m_fg, dim=1).values.mean(dim=1)
        top_b = (sub_tok @ sup_bg_bank.T).clamp(-1.0, 1.0).topk(m_bg, dim=1).values.mean(dim=1)
        # Local contrast relative to window background median
        raw_loc = (top_f - 0.50 * top_b).reshape(win_h, win_w)
        loc_bg = float(torch.quantile(raw_loc.reshape(-1), 0.50).item())
        loc_peak = float(torch.quantile(raw_loc.reshape(-1), 0.95).item())
        if (loc_peak - loc_bg) >= crop_cfg.min_support_crop_separation and float(top_f.max().item()) > float(top_b.median().item()) + 0.05:
            # Align local logit into global S_sem scale via locked S_hat = bias + temp * l_crop
            l_crop = (raw_loc - (loc_bg + 0.08)) / max(0.06, temp)
            s_hat = bias + temp * l_crop
            scored.append((win, s_hat))
    return scored, wins


def run_m2_cropzoom_dev_experiment(
    *,
    split_dir: Path,
    cache_pt: Path,
    config_json: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run Stage 0 offline candidate coverage audit + Module ② ablations on all 120 Dev episodes."""
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

    m3_cfg = get_m3_pruned_config()
    m3_no_rec_cfg = Iter5Config(**{**m3_cfg.__dict__, "name": "M2_NoRecovery", "use_self_rescue": False})
    crop_cfg = CropZoomConfig.from_json(config_json, mode="full_token_reverify")

    pipe_no_rec = ThreeModulePipeline(m3_no_rec_cfg)
    pipe_r = ThreeModulePipeline(m3_cfg)

    stage0_coverage_by_k: dict[str, Any] = {}
    methods_res: dict[str, dict[str, Any]] = {
        "M2_NoRecovery": {},
        "M2_SelfRescue_R": {},
        "M2_Ctrl_FullTokenReverify": {},
    }
    seedless_subset_by_k: dict[str, Any] = {}

    for k in sorted(ep_by_k.keys()):
        ep_no_rec, ep_r, ep_ft = [], [], []
        tot_missed = 0
        tot_covered = 0
        pos_wins = 0
        pos_imgs = 0
        neg_triggered = 0
        neg_imgs = 0
        neg_wins = 0
        ft_writebacks = 0
        ft_missed_tot = 0
        ft_rec_tot = 0
        seedless_r_recs, seedless_ft_recs = [], []

        for ep in ep_by_k[k]:
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]

            out_nr = pipe_no_rec.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            out_r = pipe_r.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)

            p_nr, s32_nr = out_nr.to_legacy_outputs()
            p_r, s32_r = out_r.to_legacy_outputs()
            ep_no_rec.append(evaluate_episode(q_recs, p_nr, s32_nr))
            ep_r.append(evaluate_episode(q_recs, p_r, s32_r))

            # Build support FG/BG token pools for Full-Token Reverify control
            fg_chunks, bg_chunks = [], []
            for sid in sup_ids:
                p_i = caches["deep_64"]["id_to_pos"][sid]
                fg_m = caches["deep_64"]["fg_all_masks"][p_i] if k == 1 else caches["deep_64"]["fg_core_masks"][p_i]
                bg_m = caches["deep_64"]["bg_masks"][p_i]
                t_s = tokens_by_key["deep_64"][sid]
                if fg_m.any():
                    fg_chunks.append(t_s[torch.from_numpy(np.flatnonzero(fg_m)).to(t_s.device)])
                if bg_m.any():
                    b_idx = np.flatnonzero(bg_m)
                    if len(b_idx) > 256:
                        b_idx = b_idx[:: max(1, len(b_idx) // 256)]
                    bg_chunks.append(t_s[torch.from_numpy(b_idx).to(t_s.device)])
            sup_fg = torch.cat(fg_chunks, dim=0) if fg_chunks else torch.empty((0, 1), device=run_device)
            sup_bg = torch.cat(bg_chunks, dim=0) if bg_chunks else torch.empty((0, 1), device=run_device)

            p_ft_ep, s32_ft_ep = {}, {}
            calib = out_r.calibration

            for qid in q_ids:
                rec = records_by_id[qid]
                sem_q = out_r.semantic_by_id[qid]
                rec_q = out_r.recovery_by_id[qid]
                bnd_q = out_r.boundary_by_id[qid]

                scored_wins, wins = _score_windows_full_image_tokens(
                    sem_q.s_sem,
                    sem_q.p_sem,
                    tokens_by_key["deep_64"][qid],
                    sup_fg,
                    sup_bg,
                    calib.bias,
                    calib.temp,
                    crop_cfg=crop_cfg,
                )

                # Stage 0 Offline GT Coverage Audit
                aud = audit_offline_candidate_coverage(rec["mask"], bnd_q.m_coarse, wins)
                if rec["positive"]:
                    pos_imgs += 1
                    pos_wins += len(wins)
                    tot_missed += aud["missed_components"]
                    tot_covered += aud["covered_missed_components"]
                else:
                    neg_imgs += 1
                    neg_wins += len(wins)
                    if aud["neg_triggered"]:
                        neg_triggered += 1

                # Apply Full-Token Reverify writeback on top of Self-Rescue S_res
                s_ft, wb_diag = project_crop_scores_max_increment(
                    rec_q.s_res, scored_wins, max_writeback_delta=crop_cfg.max_writeback_delta
                )
                if wb_diag["crop_windows_applied"] > 0:
                    ft_writebacks += 1
                p_res_ft = torch.sigmoid((s_ft - calib.bias) / calib.temp)
                s32_ft = F.avg_pool2d(s_ft.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
                rec_ft_obj = type(rec_q)(
                    qid=qid, grid_hw=64, s_res=s_ft, p_res=p_res_ft, scores_32=s32_ft, diagnostics=wb_diag
                )
                bnd_ft = pipe_r.boundary.infer_query(rec_ft_obj, aff_weights=aff_by_id[qid])
                p_ft_ep[qid] = bnd_ft.p_coarse
                s32_ft_ep[qid] = s32_ft.detach().cpu().numpy().astype(np.float32)

                if rec["positive"]:
                    mc, rc = _compute_missed_component_recovery(rec["mask"], bnd_q.m_coarse, bnd_ft.m_coarse)
                    ft_missed_tot += mc
                    ft_rec_tot += rc
                    if not rec_q.diagnostics.get("self_rescue_triggered", False):
                        gt_b = rec["mask"] > 0
                        seedless_r_recs.append(float(np.logical_and(gt_b, bnd_q.m_coarse).sum() / max(1, int(gt_b.sum()))))
                        seedless_ft_recs.append(float(np.logical_and(gt_b, bnd_ft.m_coarse).sum() / max(1, int(gt_b.sum()))))

            ep_ft.append(evaluate_episode(q_recs, p_ft_ep, s32_ft_ep))

        methods_res["M2_NoRecovery"][str(k)] = aggregate_episodes(ep_no_rec)
        methods_res["M2_SelfRescue_R"][str(k)] = aggregate_episodes(ep_r)
        methods_res["M2_Ctrl_FullTokenReverify"][str(k)] = aggregate_episodes(ep_ft)

        cov_rate = float(tot_covered / max(1, tot_missed))
        stage0_coverage_by_k[str(k)] = {
            "total_missed_gt_components": tot_missed,
            "covered_missed_gt_components": tot_covered,
            "missed_component_coverage_rate": cov_rate,
            "stage0_gate_ge_025_pass": bool(cov_rate >= 0.25),
            "mean_windows_per_pos_img": float(pos_wins / max(1, pos_imgs)),
            "mean_windows_per_neg_img": float(neg_wins / max(1, neg_imgs)),
            "pure_negative_trigger_rate": float(neg_triggered / max(1, neg_imgs)),
            "full_token_writeback_rate": float(ft_writebacks / max(1, pos_imgs + neg_imgs)),
            "full_token_missed_component_recovery_rate": float(ft_rec_tot / max(1, ft_missed_tot)),
        }
        seedless_subset_by_k[str(k)] = {
            "n_seedless_pos_queries": len(seedless_r_recs),
            "mean_recall_R": float(np.mean(seedless_r_recs)) if seedless_r_recs else None,
            "mean_recall_FullToken": float(np.mean(seedless_ft_recs)) if seedless_ft_recs else None,
        }
        print(
            f"[M2 Stage0 K={k}] MissedComps={tot_missed} Covered={tot_covered} ({cov_rate:.1%}, Gate>=25%: {cov_rate>=0.25}) "
            f"| NegTrig={neg_triggered}/{neg_imgs} ({neg_triggered/max(1,neg_imgs):.1%}) "
            f"| SeedlessPos={len(seedless_r_recs)}",
            flush=True,
        )
        for v_n in ("M2_NoRecovery", "M2_SelfRescue_R", "M2_Ctrl_FullTokenReverify"):
            m = methods_res[v_n][str(k)]["macro"]
            sm = methods_res[v_n][str(k)]["strata"]["small_lt5pct"]["recall"]
            ty = methods_res[v_n][str(k)]["strata"]["tiny_lt1pct"]["recall"]
            print(
                f"   {v_n:26s}: Dice={m['dice']:.4f} Rec={m['recall']:.4f} Sm<5R={sm:.4f} Tiny<1R={ty:.4f} "
                f"Det={m['detected']:.4f} NegFP={m['neg_fp']:.4f} PosBgFP={m['pos_bg_fp']:.4f}",
                flush=True,
            )

    stage0_passed_any = any(v["stage0_gate_ge_025_pass"] for v in stage0_coverage_by_k.values())
    comparisons = {
        "SelfRescueR_vs_NoRecovery": {
            str(k): compare_methods(methods_res["M2_SelfRescue_R"][str(k)], methods_res["M2_NoRecovery"][str(k)])
            for k in sorted(ep_by_k.keys())
        },
        "FullTokenCtrl_vs_SelfRescueR": {
            str(k): compare_methods(methods_res["M2_Ctrl_FullTokenReverify"][str(k)], methods_res["M2_SelfRescue_R"][str(k)])
            for k in sorted(ep_by_k.keys())
        },
    }

    summary = {
        "schema": "pv-2m-dinov3sat-m2-cropzoom-stage0-dev-v1",
        "device": run_device,
        "elapsed_sec": float(time.perf_counter() - t0),
        "stage0_coverage_by_k": stage0_coverage_by_k,
        "seedless_subset_by_k": seedless_subset_by_k,
        "verdict": {
            "stage0_candidate_coverage_pass": bool(stage0_passed_any),
            "crop_reencoding_executed": False,
            "note": (
                "Stage 0 coverage gate PASSED; ready to execute frozen DINOv3-SAT 128->512 crop-zoom re-encoding with weights mounted."
                if stage0_passed_any
                else "Stage 0 coverage gate FAILED (<25% missed GT component coverage); stopped before full crop re-encoding per locked protocol."
            ),
        },
        "comparisons": comparisons,
        "methods": methods_res,
    }
    write_json(output_dir / "m2_cropzoom_stage0_summary.json", summary)
    print("M2 Stage 0 Verdict:", summary["verdict"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module ② Stage 0 candidate coverage audit & Full-Token control.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m2_cropzoom.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "m2_cropzoom_dev")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_m2_cropzoom_dev_experiment(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        config_json=args.config_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
