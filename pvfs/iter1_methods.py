from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .io import write_json
from .metrics import aggregate_episodes, compare_methods, evaluate_episode


@dataclass(frozen=True)
class Iter1Config:
    name: str
    use_32: bool = True
    use_64: bool = True
    w_64: float = 0.70
    core_purity: float = 0.85
    use_core_balance: bool = True
    use_mid_layer: bool = True
    lambda_mid: float = 0.35
    beta_veto: float = 0.40
    use_calib_v1: bool = True
    top_m: int = 3
    alpha_bg: float = 0.50


ITER1_VARIANTS: tuple[Iter1Config, ...] = (
    Iter1Config(
        name="B0_32_Ref",
        use_32=True,
        use_64=False,
        w_64=0.0,
        core_purity=0.50,
        use_core_balance=False,
        use_mid_layer=False,
        use_calib_v1=False,
    ),
    Iter1Config(
        name="M1_Scale",
        use_32=True,
        use_64=True,
        w_64=0.70,
        core_purity=0.50,
        use_core_balance=False,
        use_mid_layer=False,
        use_calib_v1=False,
    ),
    Iter1Config(
        name="M1_Scale_Core",
        use_32=True,
        use_64=True,
        w_64=0.70,
        core_purity=0.85,
        use_core_balance=True,
        use_mid_layer=False,
        use_calib_v1=False,
    ),
    Iter1Config(
        name="M1_Scale_Core_Calib",
        use_32=True,
        use_64=True,
        w_64=0.70,
        core_purity=0.85,
        use_core_balance=True,
        use_mid_layer=False,
        use_calib_v1=True,
    ),
    Iter1Config(
        name="M1_Full",
        use_32=True,
        use_64=True,
        w_64=0.70,
        core_purity=0.85,
        use_core_balance=True,
        use_mid_layer=True,
        lambda_mid=0.35,
        beta_veto=0.40,
        use_calib_v1=True,
    ),
    Iter1Config(
        name="M1_NoCore_LOO",
        use_32=True,
        use_64=True,
        w_64=0.70,
        core_purity=0.50,
        use_core_balance=False,
        use_mid_layer=True,
        lambda_mid=0.35,
        beta_veto=0.40,
        use_calib_v1=True,
    ),
)


def _to_tokens(feat_map: torch.Tensor, device: str) -> torch.Tensor:
    t = feat_map.to(device=device, dtype=torch.float32)
    t = F.normalize(t, dim=0)
    return t.flatten(1).T.contiguous()


def _select_fg_bg_indices(mask_512: np.ndarray, grid_hw: int, core_purity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_px = 512 // grid_hw
    y_grid = mask_512.astype(np.float32).reshape(grid_hw, patch_px, grid_hw, patch_px).mean(axis=(1, 3)).ravel()
    fg_mask = y_grid >= core_purity
    if int(fg_mask.sum()) < 8:
        fg_mask = y_grid >= 0.50
    if not fg_mask.any():
        cands = np.flatnonzero(y_grid > 0.0)
        if len(cands) == 0:
            raise ValueError("Support image has no positive PV pixels")
        count = min(len(cands), max(1, int(math.ceil(float(y_grid.sum())))))
        fg_mask[cands[np.argsort(-y_grid[cands], kind="stable")[:count]]] = True
    bg_mask = y_grid == 0.0
    return fg_mask, bg_mask, y_grid


def _build_bank_for_scale(
    tokens_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    grid_hw: int,
    core_purity: float,
) -> dict:
    fg_per_img, bg_list, bg_src, bg_yx, fg_masks = [], [], [], [], []
    for idx, sid in enumerate(support_ids):
        tok = tokens_by_id[sid]
        fg_mask, bg_mask, _ = _select_fg_bg_indices(mask_by_id[sid], grid_hw, core_purity)
        fg_idx_t = torch.from_numpy(np.flatnonzero(fg_mask)).to(tok.device)
        fg_per_img.append(tok[fg_idx_t])
        fg_masks.append(fg_mask)
        if bg_mask.any():
            bg_flat = np.flatnonzero(bg_mask)
            bg_idx_t = torch.from_numpy(bg_flat).to(tok.device)
            bg_list.append(tok[bg_idx_t])
            bg_src.append(torch.full((bg_idx_t.numel(),), idx, dtype=torch.long, device=tok.device))
            bg_yx.append(
                torch.stack(
                    (torch.from_numpy(bg_flat // grid_hw), torch.from_numpy(bg_flat % grid_hw)),
                    dim=1,
                ).to(tok.device)
            )
    bg = torch.cat(bg_list, dim=0) if bg_list else fg_per_img[0].new_empty((0, fg_per_img[0].shape[1]))
    bg_s = torch.cat(bg_src, dim=0) if bg_src else torch.empty((0,), dtype=torch.long, device=fg_per_img[0].device)
    bg_c = torch.cat(bg_yx, dim=0) if bg_yx else torch.empty((0, 2), dtype=torch.long, device=fg_per_img[0].device)
    return {
        "fg_per_img": fg_per_img,
        "bg": bg,
        "bg_src": bg_s,
        "bg_yx": bg_c,
        "fg_masks": fg_masks,
        "grid_hw": grid_hw,
    }


def _score_bank(
    q_tok: torch.Tensor,
    fg_list: list[torch.Tensor],
    bg_bank: torch.Tensor,
    *,
    top_m: int,
    alpha_bg: float,
    balanced: bool,
) -> torch.Tensor:
    if balanced and len(fg_list) > 1:
        per_shot = []
        for fg_s in fg_list:
            sim = (q_tok @ fg_s.T).clamp(-1.0, 1.0)
            m = min(top_m, fg_s.shape[0])
            per_shot.append(sim.topk(m, dim=1).values.mean(dim=1))
        stacked = torch.stack(per_shot, dim=1)  # [N_q, K]
        s_pos = 0.70 * stacked.max(dim=1).values + 0.30 * stacked.mean(dim=1)
    else:
        fg_all = torch.cat(fg_list, dim=0)
        sim = (q_tok @ fg_all.T).clamp(-1.0, 1.0)
        s_pos = sim.topk(min(top_m, fg_all.shape[0]), dim=1).values.mean(dim=1)

    if bg_bank.shape[0] > 0 and alpha_bg > 0.0:
        sim_bg = (q_tok @ bg_bank.T).clamp(-1.0, 1.0)
        s_neg = sim_bg.topk(min(top_m, bg_bank.shape[0]), dim=1).values.mean(dim=1)
        return s_pos - alpha_bg * s_neg
    return s_pos


def _raw_score_image(
    sid_or_qid: str,
    feats_tokens: dict[str, dict[str, torch.Tensor]],
    banks: dict[str, dict],
    cfg: Iter1Config,
    *,
    exclude_idx: int | None = None,
    mid_veto_ref: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute raw map (grid_hw x grid_hw) and optional mid_64 map; supports LOO over support idx."""
    def get_fg_bg(bank: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        if exclude_idx is None:
            return bank["fg_per_img"], bank["bg"]
        fgs = [t for i, t in enumerate(bank["fg_per_img"]) if i != exclude_idx]
        bgs = bank["bg"][bank["bg_src"] != exclude_idx]
        return fgs, bgs

    out_hw = 64 if cfg.use_64 else 32
    score_64 = None
    mid_64_map = None

    if cfg.use_64:
        fgs64, bgs64 = get_fg_bg(banks["deep_64"])
        s64 = _score_bank(
            feats_tokens["deep_64"][sid_or_qid],
            fgs64,
            bgs64,
            top_m=cfg.top_m,
            alpha_bg=cfg.alpha_bg,
            balanced=cfg.use_core_balance,
        ).reshape(64, 64)
        score_64 = s64

    if cfg.use_32:
        fgs32, bgs32 = get_fg_bg(banks["deep_32"])
        s32 = _score_bank(
            feats_tokens["deep_32"][sid_or_qid],
            fgs32,
            bgs32,
            top_m=cfg.top_m,
            alpha_bg=cfg.alpha_bg,
            balanced=cfg.use_core_balance,
        ).reshape(32, 32)
        if cfg.use_64:
            s32_up = F.interpolate(s32[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
            comb = cfg.w_64 * score_64 + (1.0 - cfg.w_64) * s32_up
        else:
            comb = s32
    else:
        comb = score_64

    if cfg.use_mid_layer:
        fgsm, bgsm = get_fg_bg(banks["mid_64"])
        smid = _score_bank(
            feats_tokens["mid_64"][sid_or_qid],
            fgsm,
            bgsm,
            top_m=cfg.top_m,
            alpha_bg=cfg.alpha_bg,
            balanced=cfg.use_core_balance,
        ).reshape(64, 64)
        mid_64_map = smid
        if mid_veto_ref is not None:
            comb = comb + cfg.lambda_mid * smid - cfg.beta_veto * F.softplus((mid_veto_ref - smid) * 10.0) / 10.0

    s32_for_cam = comb if out_hw == 32 else F.avg_pool2d(comb[None, None], 2, stride=2)[0, 0]
    return comb, s32_for_cam, (mid_64_map if mid_64_map is not None else comb)


@torch.inference_mode()
def _calibrate_iter1(
    feats_tokens: dict[str, dict[str, torch.Tensor]],
    banks: dict[str, dict],
    support_ids: list[str],
    cfg: Iter1Config,
) -> tuple[float, float, float | None]:
    k = len(support_ids)
    ref_key = "deep_64" if cfg.use_64 else "deep_32"
    ref_bank = banks[ref_key]
    grid_hw = ref_bank["grid_hw"]

    # Step 1: estimate mid_veto_ref purely from support background if mid_layer is enabled
    mid_veto_ref = None
    if cfg.use_mid_layer:
        mid_bg_vals = []
        if k >= 2:
            for idx, sid in enumerate(support_ids):
                fgs_m = [t for i, t in enumerate(banks["mid_64"]["fg_per_img"]) if i != idx]
                bgs_m = banks["mid_64"]["bg"][banks["mid_64"]["bg_src"] != idx]
                sm = _score_bank(
                    feats_tokens["mid_64"][sid], fgs_m, bgs_m,
                    top_m=cfg.top_m, alpha_bg=cfg.alpha_bg, balanced=cfg.use_core_balance,
                )
                bm = banks["mid_64"]["bg_src"] == idx
                if bm.any():
                    yx = banks["mid_64"]["bg_yx"][bm]
                    mid_bg_vals.append(sm[yx[:, 0] * 64 + yx[:, 1]])
            if mid_bg_vals:
                mid_veto_ref = float(torch.quantile(torch.cat(mid_bg_vals), 0.85).item())
        if mid_veto_ref is None:
            bg_m = banks["mid_64"]["bg"]
            fg_m = torch.cat(banks["mid_64"]["fg_per_img"], dim=0)
            yx_m = banks["mid_64"]["bg_yx"]
            if bg_m.shape[0] >= 64:
                s_pos = (bg_m @ fg_m.T).clamp(-1.0, 1.0).topk(min(cfg.top_m, fg_m.shape[0]), dim=1).values.mean(dim=1)
                dy = (yx_m[:, 0:1] - yx_m[:, 0:1].T).abs()
                dx = (yx_m[:, 1:2] - yx_m[:, 1:2].T).abs()
                near = (dy <= 2) & (dx <= 2)
                vc = int((~near).sum(dim=1).min().item())
                if vc >= 1:
                    s_neg = (bg_m @ bg_m.T).clamp(-1.0, 1.0).masked_fill(near, -2.0).topk(min(cfg.top_m, vc), dim=1).values.mean(dim=1)
                    mid_veto_ref = float(torch.quantile(s_pos - cfg.alpha_bg * s_neg, 0.85).item())
            if mid_veto_ref is None:
                mid_veto_ref = 0.20

    # Step 2: calibrate final combined score
    if k >= 2:
        fg_medians, bg_all = [], []
        for idx, sid in enumerate(support_ids):
            comb, _, _ = _raw_score_image(
                sid, feats_tokens, banks, cfg, exclude_idx=idx, mid_veto_ref=mid_veto_ref
            )
            flat = comb.reshape(-1)
            fg_m = torch.from_numpy(ref_bank["fg_masks"][idx]).to(flat.device)
            fg_medians.append(float(torch.median(flat[fg_m]).item()))
            bm = ref_bank["bg_src"] == idx
            if bm.any():
                yx = ref_bank["bg_yx"][bm]
                bg_all.append(flat[yx[:, 0] * grid_hw + yx[:, 1]])
        if fg_medians and bg_all:
            bg_cat = torch.cat(bg_all)
            if cfg.use_calib_v1:
                mu_fg = float(np.mean(fg_medians))
                mu_bg = float(torch.quantile(bg_cat, 0.95).item())
                if mu_fg > mu_bg + 0.02:
                    bias = 0.45 * mu_fg + 0.55 * mu_bg
                    temp = max((mu_fg - mu_bg) / (2.0 * math.log(4.0)), 0.04)
                    return bias, temp, mid_veto_ref
                return mu_bg + 0.05, 0.06, mid_veto_ref
            else:
                fg_cat = torch.cat([
                    _raw_score_image(sid, feats_tokens, banks, cfg, exclude_idx=idx, mid_veto_ref=mid_veto_ref)[0].reshape(-1)[
                        torch.from_numpy(ref_bank["fg_masks"][idx]).to(ref_bank["bg"].device)
                    ]
                    for idx, sid in enumerate(support_ids)
                ])
                mu_fg = float(torch.median(fg_cat).item())
                mu_bg = float(torch.quantile(bg_cat, 0.90).item())
                if mu_fg > mu_bg + 0.02:
                    bias = 0.5 * (mu_fg + mu_bg)
                    temp = max((mu_fg - mu_bg) / (2.0 * math.log(4.0)), 0.04)
                    return bias, temp, mid_veto_ref
                return mu_bg + 0.05, 0.06, mid_veto_ref

    # K == 1: ex-neighborhood background distribution on the single support image
    radius = 2 if grid_hw == 64 else 1
    min_bg = 64 if grid_hw == 64 else 16
    bg_vals_list = []

    def ex_bg_contrast(bank: dict) -> torch.Tensor | None:
        bg, yx = bank["bg"], bank["bg_yx"]
        if bg.shape[0] < min_bg:
            return None
        fg = torch.cat(bank["fg_per_img"], dim=0)
        s_pos = (bg @ fg.T).clamp(-1.0, 1.0).topk(min(cfg.top_m, fg.shape[0]), dim=1).values.mean(dim=1)
        dy = (yx[:, 0:1] - yx[:, 0:1].T).abs()
        dx = (yx[:, 1:2] - yx[:, 1:2].T).abs()
        near = (dy <= radius) & (dx <= radius)
        vc = int((~near).sum(dim=1).min().item())
        if vc < 1:
            return None
        s_neg = (bg @ bg.T).clamp(-1.0, 1.0).masked_fill(near, -2.0).topk(min(cfg.top_m, vc), dim=1).values.mean(dim=1)
        return s_pos - cfg.alpha_bg * s_neg

    c64 = ex_bg_contrast(banks["deep_64"]) if cfg.use_64 else None
    c32 = ex_bg_contrast(banks["deep_32"]) if cfg.use_32 else None
    cmid = ex_bg_contrast(banks["mid_64"]) if cfg.use_mid_layer else None

    if c64 is not None and c32 is not None:
        # align c32 to 64x64 background coordinates via nearest/index mapping
        yx64 = banks["deep_64"]["bg_yx"]
        c32_full = torch.zeros((32, 32), device=c64.device)
        yx32 = banks["deep_32"]["bg_yx"]
        c32_full[yx32[:, 0], yx32[:, 1]] = c32
        c32_at_64 = c32_full[yx64[:, 0] // 2, yx64[:, 1] // 2]
        bg_score = cfg.w_64 * c64 + (1.0 - cfg.w_64) * c32_at_64
    elif c64 is not None:
        bg_score = c64
    elif c32 is not None:
        bg_score = c32
    else:
        return 0.30, 0.06, mid_veto_ref

    if cfg.use_mid_layer and cmid is not None and mid_veto_ref is not None:
        bg_score = bg_score + cfg.lambda_mid * cmid - cfg.beta_veto * F.softplus((mid_veto_ref - cmid) * 10.0) / 10.0

    q_level = 0.95 if cfg.use_calib_v1 else 0.90
    margin = 0.075 if cfg.use_calib_v1 else 0.080
    q_val = float(torch.quantile(bg_score, q_level).item())
    return q_val + margin, 0.06, mid_veto_ref


@torch.inference_mode()
def run_iter1_episode(
    feats_tokens: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter1Config,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    banks: dict[str, dict] = {}
    if cfg.use_64:
        banks["deep_64"] = _build_bank_for_scale(feats_tokens["deep_64"], mask_by_id, support_ids, 64, cfg.core_purity)
    if cfg.use_32:
        banks["deep_32"] = _build_bank_for_scale(feats_tokens["deep_32"], mask_by_id, support_ids, 32, cfg.core_purity)
    if cfg.use_mid_layer:
        banks["mid_64"] = _build_bank_for_scale(feats_tokens["mid_64"], mask_by_id, support_ids, 64, cfg.core_purity)

    bias, temp, mid_veto_ref = _calibrate_iter1(feats_tokens, banks, support_ids, cfg)

    probs_512, scores_32 = {}, {}
    for qid in query_ids:
        comb, s32_cam, _ = _raw_score_image(qid, feats_tokens, banks, cfg, exclude_idx=None, mid_veto_ref=mid_veto_ref)
        p_grid = torch.sigmoid((comb - bias) / temp)
        p512 = F.interpolate(p_grid[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        probs_512[qid] = p512.cpu().numpy().astype(np.float32)
        scores_32[qid] = s32_cam.cpu().numpy().astype(np.float32)
    return probs_512, scores_32


def run_iter1_ablations(
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
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    records_by_id, mask_by_id = {}, {}
    feats_tokens: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "deep_64": {}, "mid_64": {}}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask
        for key in ("deep_32", "deep_64", "mid_64"):
            feats_tokens[key][rid] = _to_tokens(feats[rid][key], run_device)

    ep_by_k = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    results: dict[str, dict] = {cfg.name: {} for cfg in ITER1_VARIANTS}
    for cfg in ITER1_VARIANTS:
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ep in eps:
                q_recs = [records_by_id[qid] for qid in ep["query_ids"]]
                p512, s32 = run_iter1_episode(feats_tokens, mask_by_id, ep["support_ids"], ep["query_ids"], cfg)
                ep_evals.append(evaluate_episode(q_recs, p512, s32, threshold=0.5))
            agg = aggregate_episodes(ep_evals, episodes_per_fold=episodes_lock["episodes_per_fold"])
            results[cfg.name][str(k)] = agg
            m = agg["macro"]
            sd = agg["strata"].get("small_lt5pct", {}).get("dice", 0.0)
            sr = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"[Iter1] {cfg.name:20s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"NegFP={m['neg_fp']:.4f} SmallD={sd:.4f} SmallR={sr:.4f}",
                flush=True,
            )

    comparisons = {}
    pairs = [
        ("M1_Scale", "B0_32_Ref"),
        ("M1_Scale_Core", "M1_Scale"),
        ("M1_Scale_Core_Calib", "M1_Scale_Core"),
        ("M1_Full", "M1_Scale_Core_Calib"),
        ("M1_Full", "M1_NoCore_LOO"),
        ("M1_Full", "B0_32_Ref"),
        ("M1_Scale_Core_Calib", "B0_32_Ref"),
    ]
    for cand, ref in pairs:
        cname = f"{cand}_vs_{ref}"
        comparisons[cname] = {
            str(k): compare_methods(results[cand][str(k)], results[ref][str(k)])
            for k in sorted(ep_by_k)
        }

    # Summarize worst residual FN / FP on M1_Full and M1_Scale_Core_Calib
    error_taxonomy = {}
    for variant in ("M1_Scale_Core_Calib", "M1_Full"):
        error_taxonomy[variant] = {}
        for k_str in ("1", "3", "5"):
            imgs = list(results[variant][k_str]["images"].values())
            pos_imgs = [v for v in imgs if v["positive"]]
            neg_imgs = [v for v in imgs if not v["positive"]]
            error_taxonomy[variant][k_str] = {
                "worst_fn": sorted(pos_imgs, key=lambda v: v["recall"])[:8],
                "worst_pos_bg_fp": sorted(pos_imgs, key=lambda v: -v["pos_bg_fp"])[:8],
                "worst_neg_fp": sorted(neg_imgs, key=lambda v: -v["neg_fp"])[:8],
            }

    payload = {
        "iteration": "iter1_multiscale_core_layer",
        "methods": results,
        "comparisons": comparisons,
        "error_taxonomy": error_taxonomy,
    }
    write_json(output_dir / "iter1_summary.json", payload)
    return payload
