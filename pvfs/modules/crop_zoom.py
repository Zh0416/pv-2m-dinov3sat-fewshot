from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label as nd_label, median_filter
import torch
from torch.nn import functional as F

from ..metrics import BOX_3


@dataclass(frozen=True)
class CropZoomConfig:
    enabled: bool = False
    mode: str = "off"  # "off" | "full_token_reverify" | "crop_zoom_only" | "self_rescue_plus_crop_zoom"
    local_contrast_kernel: int = 9
    min_local_prominence: float = 0.08
    max_existing_p_sem_at_peak: float = 0.65
    nms_min_center_dist_tokens: int = 6
    max_windows_per_image: int = 4
    crop_size_px: int = 128
    crop_upsample_size_px: int = 512
    core_writeback_margin_px: int = 16
    min_support_crop_separation: float = 0.05
    max_writeback_delta: float = 0.30

    @classmethod
    def from_json(cls, path: Path | str, *, mode: str = "self_rescue_plus_crop_zoom") -> "CropZoomConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cp = raw.get("candidate_window_proposal", {})
        cr = raw.get("crop_reencoding_and_calibration", {})
        return cls(
            enabled=(mode != "off"),
            mode=mode,
            local_contrast_kernel=int(cp.get("local_contrast_kernel_tokens", 9)),
            min_local_prominence=float(cp.get("min_local_prominence", 0.08)),
            max_existing_p_sem_at_peak=float(cp.get("max_existing_p_sem_at_peak", 0.65)),
            nms_min_center_dist_tokens=int(cp.get("nms_min_center_dist_tokens", 6)),
            max_windows_per_image=int(cp.get("max_windows_per_image", 4)),
            crop_size_px=int(cp.get("crop_size_px", 128)),
            crop_upsample_size_px=int(cp.get("crop_upsample_size_px", 512)),
            core_writeback_margin_px=int(cp.get("core_writeback_margin_px", 16)),
            min_support_crop_separation=float(cr.get("min_support_crop_separation", 0.05)),
            max_writeback_delta=float(cr.get("max_writeback_delta", 0.30)),
        )


@dataclass(frozen=True)
class CandidateWindow:
    win_id: int
    center_tok_y: int
    center_tok_x: int
    prominence: float
    p_sem_at_peak: float
    # Pixel box in [0, 512] (may exceed [0, 512] conceptually; handled via reflect pad on extraction)
    y0: int
    y1: int
    x0: int
    x1: int
    # Valid core writeback box clamped strictly inside [0, 512] and aligned to 64-grid tokens
    core_tok_y0: int
    core_tok_y1: int
    core_tok_x0: int
    core_tok_x1: int


def propose_seedless_candidate_windows(
    s_sem_64: np.ndarray | torch.Tensor,
    p_sem_64: np.ndarray | torch.Tensor,
    cfg: CropZoomConfig = CropZoomConfig(),
) -> list[CandidateWindow]:
    """Propose up to max_windows_per_image local-prominence candidate windows on unthresholded S_sem.

    Operates independently of whether strong FG seeds exist (does NOT require Self-Rescue seeds).
    """
    s_np = s_sem_64.detach().cpu().numpy().reshape(64, 64).astype(np.float32) if isinstance(s_sem_64, torch.Tensor) else np.asarray(s_sem_64, dtype=np.float32).reshape(64, 64)
    p_np = p_sem_64.detach().cpu().numpy().reshape(64, 64).astype(np.float32) if isinstance(p_sem_64, torch.Tensor) else np.asarray(p_sem_64, dtype=np.float32).reshape(64, 64)

    loc_med = median_filter(s_np, size=cfg.local_contrast_kernel, mode="reflect")
    prom = s_np - loc_med

    cand_mask = (prom >= cfg.min_local_prominence) & (p_np <= cfg.max_existing_p_sem_at_peak)
    ys, xs = np.nonzero(cand_mask)
    if len(ys) == 0:
        return []

    scores = prom[ys, xs]
    # Deterministic descending sort by local prominence, tie-broken by row then col
    order = np.lexsort((xs, ys, -scores))
    half = cfg.crop_size_px // 2
    margin_tok = cfg.core_writeback_margin_px // 8
    half_tok = half // 8

    windows: list[CandidateWindow] = []
    for idx in order:
        cy = int(ys[idx])
        cx = int(xs[idx])
        if any(max(abs(cy - w.center_tok_y), abs(cx - w.center_tok_x)) < cfg.nms_min_center_dist_tokens for w in windows):
            continue
        py_center = cy * 8 + 4
        px_center = cx * 8 + 4
        y0, y1 = py_center - half, py_center + half
        x0, x1 = px_center - half, px_center + half

        # Core writeback token range strictly inside [0, 64]
        ct_y0 = max(0, cy - half_tok + margin_tok)
        ct_y1 = min(64, cy + half_tok - margin_tok)
        ct_x0 = max(0, cx - half_tok + margin_tok)
        ct_x1 = min(64, cx + half_tok - margin_tok)
        if ct_y1 <= ct_y0 or ct_x1 <= ct_x0:
            continue

        windows.append(
            CandidateWindow(
                win_id=len(windows),
                center_tok_y=cy,
                center_tok_x=cx,
                prominence=float(scores[idx]),
                p_sem_at_peak=float(p_np[cy, cx]),
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                core_tok_y0=ct_y0,
                core_tok_y1=ct_y1,
                core_tok_x0=ct_x0,
                core_tok_x1=ct_x1,
            )
        )
        if len(windows) >= cfg.max_windows_per_image:
            break
    return windows


def extract_reflect_padded_crop(
    img_hwc: np.ndarray,
    win: CandidateWindow,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract [crop_size, crop_size, C] RGB crop with reflect padding for border-adjacent windows.

    Returns (crop_arr, valid_in_crop_slice=(vy0, vy1, vx0, vx1)) marking original in-bounds pixels.
    """
    h, w = img_hwc.shape[:2]
    pad_top = max(0, -win.y0)
    pad_bot = max(0, win.y1 - h)
    pad_left = max(0, -win.x0)
    pad_right = max(0, win.x1 - w)

    if img_hwc.ndim == 3:
        padded = np.pad(img_hwc, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)), mode="reflect")
    else:
        padded = np.pad(img_hwc, ((pad_top, pad_bot), (pad_left, pad_right)), mode="reflect")

    sy0 = win.y0 + pad_top
    sy1 = win.y1 + pad_top
    sx0 = win.x0 + pad_left
    sx1 = win.x1 + pad_left
    crop = padded[sy0:sy1, sx0:sx1]

    crop_h = win.y1 - win.y0
    crop_w = win.x1 - win.x0
    valid_box = (pad_top, crop_h - pad_bot, pad_left, crop_w - pad_right)
    return crop, valid_box


def project_crop_scores_max_increment(
    s_res_64: torch.Tensor,
    window_score_maps: list[tuple[CandidateWindow, torch.Tensor]],
    *,
    max_writeback_delta: float = 0.30,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project verified crop-aligned scores S_hat_crop ([crop_h_tok, crop_w_tok]) onto 64x64 grid.

    Takes max-increment (never additive stacking) across overlapping windows, strictly within
    each window's in-bounds core region.
    """
    base_2d = s_res_64.reshape(64, 64)
    best_inc = torch.zeros_like(base_2d)
    applied_windows = 0

    for win, s_hat_crop_2d in window_score_maps:
        # s_hat_crop_2d covers the full 128x128 crop window (16x16 in 64-grid token coordinates)
        win_tok_h = (win.y1 - win.y0) // 8
        win_tok_w = (win.x1 - win.x0) // 8
        if s_hat_crop_2d.shape != (win_tok_h, win_tok_w):
            s_hat_16 = F.interpolate(
                s_hat_crop_2d.reshape(1, 1, s_hat_crop_2d.shape[0], s_hat_crop_2d.shape[1]),
                size=(win_tok_h, win_tok_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        else:
            s_hat_16 = s_hat_crop_2d

        # Origin of window in 64-grid token coordinates (integer floor of win.y0/8, win.x0/8)
        orig_ty = win.y0 // 8
        orig_tx = win.x0 // 8

        # Map core_tok box into both full 64x64 grid and local win_tok grid
        gy0, gy1 = win.core_tok_y0, win.core_tok_y1
        gx0, gx1 = win.core_tok_x0, win.core_tok_x1
        ly0, ly1 = gy0 - orig_ty, gy1 - orig_ty
        lx0, lx1 = gx0 - orig_tx, gx1 - orig_tx

        # Clamp strictly to valid local indices
        c_ly0 = max(0, ly0)
        c_ly1 = min(win_tok_h, ly1)
        c_lx0 = max(0, lx0)
        c_lx1 = min(win_tok_w, lx1)
        if c_ly1 <= c_ly0 or c_lx1 <= c_lx0:
            continue

        c_gy0 = orig_ty + c_ly0
        c_gy1 = orig_ty + c_ly1
        c_gx0 = orig_tx + c_lx0
        c_gx1 = orig_tx + c_lx1

        inc = F.relu(s_hat_16[c_ly0:c_ly1, c_lx0:c_lx1] - base_2d[c_gy0:c_gy1, c_gx0:c_gx1]).clamp(0.0, max_writeback_delta)
        if float(inc.max().item()) > 0.0:
            applied_windows += 1
        best_inc[c_gy0:c_gy1, c_gx0:c_gx1] = torch.maximum(best_inc[c_gy0:c_gy1, c_gx0:c_gx1], inc)

    return (base_2d + best_inc).reshape(-1), {
        "crop_windows_applied": applied_windows,
        "crop_max_increment": float(best_inc.max().item()),
        "crop_nonzero_tokens": int((best_inc > 0.0).sum().item()),
    }


def audit_offline_candidate_coverage(
    gt_mask_512: np.ndarray,
    m_coarse_512: np.ndarray,
    windows: list[CandidateWindow],
    *,
    min_missed_area_px: int = 16,
    min_cover_frac: float = 0.50,
) -> dict[str, Any]:
    """Offline Stage 0 GT audit: measure how many completely missed GT components are covered by candidate windows."""
    gt = np.asarray(gt_mask_512 > 0, dtype=bool)
    m0 = np.asarray(m_coarse_512, dtype=bool)

    if not gt.any():
        return {
            "positive": False,
            "missed_components": 0,
            "covered_missed_components": 0,
            "windows_triggered": len(windows),
            "neg_triggered": len(windows) > 0,
        }

    # Union of core boxes in 512x512 pixel coordinates
    core_union = np.zeros((512, 512), dtype=bool)
    for w in windows:
        core_union[w.core_tok_y0 * 8 : w.core_tok_y1 * 8, w.core_tok_x0 * 8 : w.core_tok_x1 * 8] = True

    lbl, n_comp = nd_label(gt, structure=BOX_3)
    missed = 0
    covered = 0
    for cid in range(1, n_comp + 1):
        comp = lbl == cid
        area = int(comp.sum())
        if area < min_missed_area_px:
            continue
        if not np.logical_and(comp, m0).any():
            missed += 1
            cov_frac = float(np.logical_and(comp, core_union).sum() / area)
            if cov_frac >= min_cover_frac:
                covered += 1

    return {
        "positive": True,
        "missed_components": missed,
        "covered_missed_components": covered,
        "windows_triggered": len(windows),
        "neg_triggered": False,
    }
