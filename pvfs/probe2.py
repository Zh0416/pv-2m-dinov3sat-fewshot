from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices
from .io import write_json
from .metrics import aggregate_episodes, compare_methods, evaluate_episode

LAYER_GRIDS: dict[str, int] = {
    "early_32": 32,
    "mid_32": 32,
    "semi_32": 32,
    "deep_32": 32,
    "mid_64": 64,
    "deep_64": 64,
}


def _to_tokens_norm(feat_map: torch.Tensor, device: str) -> torch.Tensor:
    t = feat_map.to(device=device, dtype=torch.float32)
    t = F.normalize(t, dim=0)
    return t.flatten(1).T.contiguous()


def _ex_neigh_contrast_on_support(
    tok: torch.Tensor,
    mask_512: np.ndarray,
    grid_hw: int,
    *,
    core_purity: float,
    alpha_bg: float = 0.50,
    top_m: int = 3,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute ex-neighborhood contrast (s_pos - alpha * s_neg) on support bg and support fg_all."""
    radius = 2 if grid_hw == 64 else 1
    min_bg = 64 if grid_hw == 64 else 16
    fg_c, bg_m, _ = select_fg_bg_indices(mask_512, grid_hw, core_purity)
    fg_a, _, _ = select_fg_bg_indices(mask_512, grid_hw, 0.50)

    fg_t = tok[torch.from_numpy(np.flatnonzero(fg_c)).to(tok.device)]
    bg_flat = np.flatnonzero(bg_m)
    if len(bg_flat) < min_bg:
        return None, None
    bg_idx = torch.from_numpy(bg_flat).to(tok.device)
    bg_t = tok[bg_idx]
    yx_b = torch.stack((bg_idx // grid_hw, bg_idx % grid_hw), dim=1)

    s_pos_b = (bg_t @ fg_t.T).clamp(-1.0, 1.0).topk(min(top_m, fg_t.shape[0]), dim=1).values.mean(dim=1)
    near_b = ((yx_b[:, 0:1] - yx_b[:, 0:1].T).abs() <= radius) & ((yx_b[:, 1:2] - yx_b[:, 1:2].T).abs() <= radius)
    vc_b = int((~near_b).sum(dim=1).min().item())
    if vc_b < 1:
        return None, None
    s_neg_b = (bg_t @ bg_t.T).clamp(-1.0, 1.0).masked_fill(near_b, -2.0).topk(min(top_m, vc_b), dim=1).values.mean(dim=1)
    c_bg = s_pos_b - alpha_bg * s_neg_b

    # Exclude local radius around fg token against full image fg & bg so fg calibration stays cross-region
    fga_flat = np.flatnonzero(fg_a)
    fga_idx = torch.from_numpy(fga_flat).to(tok.device)
    fga_t = tok[fga_idx]
    yx_f = torch.stack((fga_idx // grid_hw, fga_idx % grid_hw), dim=1)
    fg_all_idx = torch.from_numpy(np.flatnonzero(fg_c)).to(tok.device)
    yx_fc = torch.stack((fg_all_idx // grid_hw, fg_all_idx % grid_hw), dim=1)
    near_ff = ((yx_f[:, 0:1] - yx_fc[:, 0:1].T).abs() <= radius) & ((yx_f[:, 1:2] - yx_fc[:, 1:2].T).abs() <= radius)
    vc_ff = int((~near_ff).sum(dim=1).min().item())
    if vc_ff < 1:
        return c_bg, None
    s_pos_f = (fga_t @ fg_t.T).clamp(-1.0, 1.0).masked_fill(near_ff, -2.0).topk(min(top_m, vc_ff), dim=1).values.mean(dim=1)
    s_neg_f = (fga_t @ bg_t.T).clamp(-1.0, 1.0).topk(min(top_m, bg_t.shape[0]), dim=1).values.mean(dim=1)
    return c_bg, (s_pos_f - alpha_bg * s_neg_f)


@torch.inference_mode()
def run_probe2(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    _, lbl_dir = resolve_data_dirs(image_dir, label_dir)
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
    deep_mid_tokens: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in LAYER_GRIDS.items():
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        if key in ("deep_64", "mid_64", "deep_32"):
            deep_mid_tokens[key] = toks
        else:
            del toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    ep_by_k = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    # Probe 1: Multi-layer diagnostic stats across K=3 on key target/confounder IDs
    diag_ids = [
        "正样本/PV-73.tif", "正样本/PV-55.tif", "正样本/PV-49.tif", "正样本/PV-03.tif",
        "正样本/PV-26.tif", "正样本/PV-10.tif", "正样本/PV-30.tif", "正样本/PV-80.tif",
        "正样本/PV-21.tif", "正样本/PV-12.tif", "正样本/PV-38.tif",
        "负样本/PV-93.tif", "负样本/PV-100.tif", "负样本/PV-96.tif", "负样本/PV-86.tif",
    ]
    diag_accum = {did: {k: defaultdict(list) for k in LAYER_GRIDS} for did in diag_ids}
    diag_self_contrast = {did: defaultdict(list) for did in diag_ids}
    for ep in ep_by_k[3]:
        sids = ep["support_ids"]
        for qid in ep["query_ids"]:
            if qid not in diag_accum:
                continue
            rec = records_by_id[qid]
            m64 = (
                torch.from_numpy(rec["mask"].astype(np.float32)).reshape(64, 8, 64, 8).mean(dim=(1, 3)).ravel().to(run_device)
            )
            m32 = (
                torch.from_numpy(rec["mask"].astype(np.float32)).reshape(32, 16, 32, 16).mean(dim=(1, 3)).ravel().to(run_device)
            )
            for key, ghw in LAYER_GRIDS.items():
                sc, sp, sn = score_from_pairwise(caches[key], qid, sids, use_core=True, shot_mode="balanced_70_30", alpha_bg=0.5)
                mg = m64 if ghw == 64 else m32
                if rec["positive"]:
                    fg_m = mg >= 0.5
                    bg_m = mg == 0.0
                    if fg_m.any():
                        diag_accum[qid][key]["fg_spos"].append(float(sp[fg_m].mean().item()))
                        diag_accum[qid][key]["fg_sneg"].append(float(sn[fg_m].mean().item()))
                        diag_accum[qid][key]["fg_c50"].append(float(sc[fg_m].mean().item()))
                        diag_accum[qid][key]["fg_q95"].append(float(torch.quantile(sc[fg_m], 0.95).item()))
                    if bg_m.any():
                        diag_accum[qid][key]["bg_q98"].append(float(torch.quantile(sc[bg_m], 0.98).item()))
                        diag_accum[qid][key]["bg_sneg_at_q98"].append(float(sn[bg_m][sc[bg_m] >= torch.quantile(sc[bg_m], 0.95)].mean().item()))
                else:
                    q95 = torch.quantile(sc, 0.95)
                    top_m = sc >= q95
                    diag_accum[qid][key]["neg_q98"].append(float(torch.quantile(sc, 0.98).item()))
                    diag_accum[qid][key]["neg_q95"].append(float(q95.item()))
                    diag_accum[qid][key]["neg_spos_top"].append(float(sp[top_m].mean().item()))
                    diag_accum[qid][key]["neg_sneg_top"].append(float(sn[top_m].mean().item()))
                if key == "deep_64":
                    diag_self_contrast[qid]["q99"].append(float(torch.quantile(sc, 0.99).item()))
                    diag_self_contrast[qid]["q95"].append(float(torch.quantile(sc, 0.95).item()))
                    diag_self_contrast[qid]["q50"].append(float(torch.quantile(sc, 0.50).item()))
                    diag_self_contrast[qid]["q99_minus_q50"].append(
                        float((torch.quantile(sc, 0.99) - torch.quantile(sc, 0.50)).item())
                    )

    layer_profile = {
        did: {k: {m_k: float(np.mean(v)) for m_k, v in stats.items()} for k, stats in by_k.items() if stats}
        for did, by_k in diag_accum.items()
    }
    self_contrast_profile = {
        did: {m_k: float(np.mean(v)) for m_k, v in stats.items()}
        for did, stats in diag_self_contrast.items()
    }

    # Probe 2: Calibration + Shot-Pool fast sweep on M1_Full base score
    configs = [
        ("M1_Ref_fast", "balanced_70_30", True, 0.45, 0.95, False, 0.075),
        ("CalFix_30_70", "balanced_70_30", False, 0.30, 0.95, False, 0.080),
        ("CalFix_20_80", "balanced_70_30", False, 0.20, 0.94, False, 0.085),
        ("CalFix_Top2_25_75", "top2_soft", False, 0.25, 0.94, True, 0.085),
        ("CalFix_Top2_20_80", "top2_soft", False, 0.20, 0.92, True, 0.090),
    ]

    sweep_results: dict[str, dict] = {c[0]: {} for c in configs}
    for cname, shot_mode, fg_calib_core, w_fg, bg_q, k1_use_all_fg, k1_margin in configs:
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ep in eps:
                sids = ep["support_ids"]
                # mid veto ref
                mid_vals = []
                if k >= 2:
                    for idx, sid in enumerate(sids):
                        sm, _, _ = score_from_pairwise(
                            caches["mid_64"], sid, sids, use_core=True, shot_mode=shot_mode, alpha_bg=0.5, exclude_pos_idx=idx
                        )
                        bm = caches["mid_64"]["bg_masks"][caches["mid_64"]["id_to_pos"][sid]]
                        if bm.any():
                            mid_vals.append(sm[torch.from_numpy(np.flatnonzero(bm)).to(sm.device)])
                    mid_ref = float(torch.quantile(torch.cat(mid_vals), 0.85).item()) if mid_vals else 0.20
                else:
                    c_bg_m, _ = _ex_neigh_contrast_on_support(deep_mid_tokens["mid_64"][sids[0]], mask_by_id[sids[0]], 64, core_purity=0.85)
                    mid_ref = float(torch.quantile(c_bg_m, 0.85).item()) if c_bg_m is not None else 0.20

                def raw_comb(qid_in: str, ex_idx: int | None) -> tuple[torch.Tensor, torch.Tensor]:
                    s64, _, _ = score_from_pairwise(caches["deep_64"], qid_in, sids, use_core=True, shot_mode=shot_mode, alpha_bg=0.5, exclude_pos_idx=ex_idx)
                    s32, _, _ = score_from_pairwise(caches["deep_32"], qid_in, sids, use_core=True, shot_mode=shot_mode, alpha_bg=0.5, exclude_pos_idx=ex_idx)
                    sm, _, _ = score_from_pairwise(caches["mid_64"], qid_in, sids, use_core=True, shot_mode=shot_mode, alpha_bg=0.5, exclude_pos_idx=ex_idx)
                    s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
                    comb = 0.70 * s64 + 0.30 * s32_up + 0.35 * sm - 0.40 * F.softplus((mid_ref - sm) * 10.0) / 10.0
                    comb_32 = F.avg_pool2d(comb.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
                    return comb, comb_32

                if k >= 2:
                    fg_meds, bg_list = [], []
                    for idx, sid in enumerate(sids):
                        c64, _ = raw_comb(sid, idx)
                        p_i = caches["deep_64"]["id_to_pos"][sid]
                        fg_mask = caches["deep_64"]["fg_core_masks"][p_i] if fg_calib_core else caches["deep_64"]["fg_all_masks"][p_i]
                        bg_mask = caches["deep_64"]["bg_masks"][p_i]
                        fg_meds.append(float(torch.median(c64[torch.from_numpy(np.flatnonzero(fg_mask)).to(c64.device)]).item()))
                        if bg_mask.any():
                            bg_list.append(c64[torch.from_numpy(np.flatnonzero(bg_mask)).to(c64.device)])
                    mu_fg = float(np.mean(fg_meds))
                    mu_bg = float(torch.quantile(torch.cat(bg_list), bg_q).item())
                    bias = w_fg * mu_fg + (1.0 - w_fg) * mu_bg if mu_fg > mu_bg + 0.02 else mu_bg + 0.05
                    temp = max((mu_fg - mu_bg) / (2.0 * math.log(4.0)), 0.04)
                else:
                    c64_b, _ = _ex_neigh_contrast_on_support(deep_mid_tokens["deep_64"][sids[0]], mask_by_id[sids[0]], 64, core_purity=0.85)
                    c32_b, _ = _ex_neigh_contrast_on_support(deep_mid_tokens["deep_32"][sids[0]], mask_by_id[sids[0]], 32, core_purity=0.85)
                    cm_b, _ = _ex_neigh_contrast_on_support(deep_mid_tokens["mid_64"][sids[0]], mask_by_id[sids[0]], 64, core_purity=0.85)
                    if c64_b is not None and c32_b is not None:
                        _, bg64_m, _ = select_fg_bg_indices(mask_by_id[sids[0]], 64, 0.85)
                        _, bg32_m, _ = select_fg_bg_indices(mask_by_id[sids[0]], 32, 0.85)
                        bg32_flat = np.flatnonzero(bg32_m)
                        c32_full = torch.zeros((32, 32), device=c64_b.device)
                        c32_full[torch.from_numpy(bg32_flat // 32).to(c64_b.device), torch.from_numpy(bg32_flat % 32).to(c64_b.device)] = c32_b
                        bg64_flat = np.flatnonzero(bg64_m)
                        c32_at_64 = c32_full[torch.from_numpy(bg64_flat // 64).to(c64_b.device) // 2, torch.from_numpy(bg64_flat % 64).to(c64_b.device) // 2]
                        bg_s = 0.70 * c64_b + 0.30 * c32_at_64
                        if cm_b is not None:
                            bg_s = bg_s + 0.35 * cm_b - 0.40 * F.softplus((mid_ref - cm_b) * 10.0) / 10.0
                        qval = float(torch.quantile(bg_s, bg_q).item())
                        iqr = float((torch.quantile(bg_s, 0.75) - torch.quantile(bg_s, 0.25)).item())
                        bias = qval + (k1_margin * (iqr / 0.08) if k1_use_all_fg else k1_margin)
                        # clamp for numerical safety
                        if k1_use_all_fg:
                            bias = qval + min(max(k1_margin * max(iqr / 0.12, 0.85), 0.06), 0.12)
                    else:
                        bias = 0.30
                    temp = 0.06

                p512_d, s32_d = {}, {}
                for qid in ep["query_ids"]:
                    c64, c32 = raw_comb(qid, None)
                    p64 = torch.sigmoid((c64.reshape(1, 1, 64, 64) - bias) / temp)
                    p512_d[qid] = F.interpolate(p64, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
                    s32_d[qid] = c32.cpu().numpy().astype(np.float32)
                ep_evals.append(evaluate_episode([records_by_id[q] for q in ep["query_ids"]], p512_d, s32_d))
            agg = aggregate_episodes(ep_evals, episodes_per_fold=episodes_lock["episodes_per_fold"])
            sweep_results[cname][str(k)] = agg
            m = agg["macro"]
            print(f"[Probe2] {cname:18s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} Rec={m['recall']:.4f} Prec={m['precision']:.4f} NegFP={m['neg_fp']:.4f} PosBg={m['pos_bg_fp']:.4f}", flush=True)

    summary = {
        "layer_profile_k3": layer_profile,
        "self_contrast_k3": self_contrast_profile,
        "sweep_macro": {
            cname: {k: agg["macro"] for k, agg in by_k.items()}
            for cname, by_k in sweep_results.items()
        },
        "sweep_comparisons": {
            f"{c}_vs_M1_Ref_fast": {
                k: compare_methods(sweep_results[c][k], sweep_results["M1_Ref_fast"][k])
                for k in ("1", "3", "5")
            }
            for c in sweep_results if c != "M1_Ref_fast"
        },
    }
    write_json(output_dir / "probe2_summary.json", summary)
    return summary
