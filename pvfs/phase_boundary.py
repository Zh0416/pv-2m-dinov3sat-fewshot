from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label as nd_label,
    uniform_filter,
)

from .metrics import BOX_3, DISK_2, _disk_structure

DISK_1 = _disk_structure(1)
DISK_4 = _disk_structure(4)
DISK_8 = _disk_structure(8)
CROSS_3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass(frozen=True)
class PhaseBoundaryConfig:
    coarse_threshold: float = 0.5
    max_shift_px: int = 8
    target_segment_arc_len_px: int = 16
    min_segment_arc_len_px: int = 8
    corner_split_angle_deg: float = 45.0
    min_component_area_px: int = 64
    min_opposite_shore_clearance_px: float = 2.0
    min_inter_component_clearance_px: float = 2.0
    min_valid_column_coverage: float = 0.75
    delta_gt_global_bias: float = 0.0

    trend_radius_px: int = 8
    trend_sigma_spatial_px: float = 6.0
    trend_sigma_range: float = 0.12

    interior_erosion_px: int = 8
    fit_holdout_block_size_px: int = 12
    fit_holdout_guard_gap_px: int = 2
    min_fit_pixels: int = 48
    min_holdout_pixels: int = 32
    min_span_periods: float = 2.0
    default_period_min_px: float = 2.8
    default_period_max_px: float = 12.0
    hard_sampling_period_min_px: float = 2.5
    hard_sampling_period_max_px: float = 14.0
    small_target_area_thr_px: int = 1024
    fft_grid_size: int = 64
    top_candidate_frequencies: int = 12
    max_normal_matrix_cond: float = 50.0
    min_holdout_r2_floor: float = 0.05
    min_fit_holdout_phase_cos: float = 0.50
    min_holdout_energy: float = 1e-6
    noise_var_floor: float = 1e-4

    null_quantile_alpha: float = 0.05
    support_block_size_px: int = 40
    support_boundary_margin_px: int = 8
    min_support_null_blocks: int = 4
    k1_forbid_outward: bool = True
    p_floor_quantile: float = 0.25
    p_floor_min: float = 0.20
    p_floor_max: float = 0.48

    block_arc_len_px_E: int = 4
    num_phase_grid_E: int = 16

    fallback_gain_per_px: dict[str, float] = field(
        default_factory=lambda: {
            "B_GradAlign": 0.035,
            "C_TexVar": 0.25,
            "D_BilateralTerm": 0.08,
            "E_FreqAmp_NoPhase": 0.22,
            "F_FullPhase": 0.22,
            "G_PhaseCounterfactual": 0.22,
            "H_NoSemGuard": 0.22,
        }
    )

    @classmethod
    def from_json(cls, path: Path | str) -> "PhaseBoundaryConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        g = raw.get("geometry", {})
        s = raw.get("signal", {})
        c = raw.get("carrier", {})
        cal = raw.get("calibration", {})
        pe = raw.get("phase_marginalization_E", {})
        p_clamp = cal.get("p_floor_clamp", [0.20, 0.48])
        return cls(
            coarse_threshold=float(g.get("coarse_threshold", 0.5)),
            max_shift_px=int(g.get("max_shift_px", 8)),
            target_segment_arc_len_px=int(g.get("target_segment_arc_len_px", 16)),
            min_segment_arc_len_px=int(g.get("min_segment_arc_len_px", 8)),
            corner_split_angle_deg=float(g.get("corner_split_angle_deg", 45.0)),
            min_component_area_px=int(g.get("min_component_area_px", 64)),
            min_opposite_shore_clearance_px=float(g.get("min_opposite_shore_clearance_px", 2.0)),
            min_inter_component_clearance_px=float(g.get("min_inter_component_clearance_px", 2.0)),
            min_valid_column_coverage=float(g.get("min_valid_column_coverage", 0.75)),
            delta_gt_global_bias=float(g.get("delta_gt_global_bias", 0.0)),
            trend_radius_px=int(s.get("trend_radius_px", 8)),
            trend_sigma_spatial_px=float(s.get("trend_sigma_spatial_px", 6.0)),
            trend_sigma_range=float(s.get("trend_sigma_range", 0.12)),
            interior_erosion_px=int(c.get("interior_erosion_px", 8)),
            fit_holdout_block_size_px=int(c.get("fit_holdout_block_size_px", 12)),
            fit_holdout_guard_gap_px=int(c.get("fit_holdout_guard_gap_px", 2)),
            min_fit_pixels=int(c.get("min_fit_pixels", 48)),
            min_holdout_pixels=int(c.get("min_holdout_pixels", 32)),
            min_span_periods=float(c.get("min_span_periods", 2.0)),
            default_period_min_px=float(c.get("default_period_min_px", 2.8)),
            default_period_max_px=float(c.get("default_period_max_px", 12.0)),
            hard_sampling_period_min_px=float(c.get("hard_sampling_period_min_px", 2.5)),
            hard_sampling_period_max_px=float(c.get("hard_sampling_period_max_px", 14.0)),
            small_target_area_thr_px=int(c.get("small_target_area_thr_px", 1024)),
            fft_grid_size=int(c.get("fft_grid_size", 64)),
            top_candidate_frequencies=int(c.get("top_candidate_frequencies", 12)),
            max_normal_matrix_cond=float(c.get("max_normal_matrix_cond", 50.0)),
            min_holdout_r2_floor=float(c.get("min_holdout_r2_floor", 0.05)),
            min_fit_holdout_phase_cos=float(c.get("min_fit_holdout_phase_cos", 0.50)),
            min_holdout_energy=float(c.get("min_holdout_energy", 1e-6)),
            noise_var_floor=float(c.get("noise_var_floor", 1e-4)),
            null_quantile_alpha=float(cal.get("null_quantile_alpha", 0.05)),
            support_block_size_px=int(cal.get("support_block_size_px", 40)),
            support_boundary_margin_px=int(cal.get("support_boundary_margin_px", 8)),
            min_support_null_blocks=int(cal.get("min_support_null_blocks", 4)),
            k1_forbid_outward=(cal.get("k1_outward_expansion_policy", "forbid_outward_when_k_eq_1") == "forbid_outward_when_k_eq_1"),
            p_floor_quantile=float(cal.get("p_floor_quantile", 0.25)),
            p_floor_min=float(p_clamp[0]),
            p_floor_max=float(p_clamp[1]),
            block_arc_len_px_E=int(pe.get("block_arc_len_px", 4)),
            num_phase_grid_E=int(pe.get("num_phase_grid", 16)),
            fallback_gain_per_px=dict(cal.get("fallback_gain_per_px", cls().fallback_gain_per_px)),
        )


def boundary_f1_at_radius(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    radius: int = 2,
    t_b: np.ndarray | None = None,
    t_dil: np.ndarray | None = None,
) -> float:
    """Boundary contour F1 at pixel tolerance radius (1px or 2px).

    Strictly prevents cross-radius GT dilation cache reuse by validating radius.
    """
    truth_b = np.asarray(truth, dtype=bool)
    pred_b = np.asarray(pred, dtype=bool)
    if t_b is None:
        t_b = truth_b ^ binary_erosion(truth_b, structure=BOX_3, border_value=1)
    p_b = pred_b ^ binary_erosion(pred_b, structure=BOX_3, border_value=1)
    if not t_b.any() and not p_b.any():
        return 1.0
    if not t_b.any() or not p_b.any():
        return 0.0
    disk = DISK_1 if radius == 1 else (DISK_2 if radius == 2 else _disk_structure(radius))
    if t_dil is None:
        t_dil = binary_dilation(t_b, structure=disk)
    p_dil = binary_dilation(p_b, structure=disk)
    prec = float(np.logical_and(p_b, t_dil).sum() / p_b.sum())
    rec = float(np.logical_and(t_b, p_dil).sum() / t_b.sum())
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0


def compute_oracle_pm8px_mask(
    mask_gt: np.ndarray,
    prob_512: np.ndarray,
    *,
    threshold: float = 0.5,
    max_shift_px: int = 8,
) -> dict[str, Any]:
    """Compute +/-8px coarse-contour oracle replacement mask and error decomposition."""
    gt = np.asarray(mask_gt > 0, dtype=bool)
    m0 = np.asarray(prob_512 >= threshold, dtype=bool)
    err = gt ^ m0
    total_err = int(err.sum())

    c_b = m0 ^ binary_erosion(m0, structure=BOX_3, border_value=1)
    disk = DISK_8 if max_shift_px == 8 else _disk_structure(max_shift_px)
    if c_b.any():
        band = binary_dilation(c_b, structure=disk)
    else:
        band = np.zeros_like(m0, dtype=bool)

    oracle_mask = m0.copy()
    oracle_mask[band] = gt[band]

    fixable_err = int(np.logical_and(err, band).sum())
    unfixed_err = total_err - fixable_err

    # Separate whole-component missed GT objects and whole-component spurious FPs
    whole_missed_px = 0
    if gt.any():
        gt_lbl, n_gt = nd_label(gt, structure=BOX_3)
        for cid in range(1, n_gt + 1):
            comp = gt_lbl == cid
            if not np.logical_and(comp, m0).any():
                whole_missed_px += int(comp.sum())

    whole_spurious_px = 0
    if m0.any():
        pr_lbl, n_pr = nd_label(m0, structure=BOX_3)
        for cid in range(1, n_pr + 1):
            comp = pr_lbl == cid
            if not np.logical_and(comp, gt).any():
                whole_spurious_px += int(comp.sum())

    return {
        "oracle_mask": oracle_mask,
        "band_mask": band,
        "total_error_px": total_err,
        "pm8_fixable_error_px": fixable_err,
        "pm8_unfixed_error_px": unfixed_err,
        "pm8_fixable_fraction": float(fixable_err / total_err) if total_err > 0 else 1.0,
        "whole_missed_fn_px": whole_missed_px,
        "whole_spurious_fp_px": whole_spurious_px,
        "whole_missed_fn_fraction": float(whole_missed_px / total_err) if total_err > 0 else 0.0,
        "whole_spurious_fp_fraction": float(whole_spurious_px / total_err) if total_err > 0 else 0.0,
    }


def compute_luma_and_detrended_residual(
    rgb: np.ndarray,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute luminance I(x), boundary-safe local trend m(x), and residual r(x)=I(x)-m(x).

    Uses a deterministic bilateral low-pass filter that is completely independent of
    candidate displacement d and avoids cross-boundary stripe leakage.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 3:
        if arr.max() > 1.5:
            arr = arr / 255.0
        luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    else:
        luma = (arr / 255.0 if arr.max() > 1.5 else arr.copy()).astype(np.float32)

    rad = cfg.trend_radius_px
    sig_s2 = 2.0 * (cfg.trend_sigma_spatial_px ** 2)
    sig_r2 = 2.0 * (cfg.trend_sigma_range ** 2)

    padded = np.pad(luma, rad, mode="reflect")
    h, w = luma.shape
    num = np.zeros_like(luma, dtype=np.float32)
    den = np.zeros_like(luma, dtype=np.float32)

    # Stride-2 symmetric bilateral taps + center tap give exact isotropic low-pass
    # while running in <6 ms per 512x512 image on CPU.
    offsets = sorted(set(range(-rad, rad + 1, 2)) | {0, -1, 1})
    for dy in offsets:
        for dx in offsets:
            w_sp = math.exp(-(dx * dx + dy * dy) / sig_s2)
            shifted = padded[rad + dy : rad + dy + h, rad + dx : rad + dx + w]
            diff = shifted - luma
            w_tot = w_sp * np.exp(-(diff * diff) / sig_r2)
            num += w_tot * shifted
            den += w_tot

    trend = num / np.maximum(den, 1e-8)
    residual = luma - trend
    return luma, trend, residual


def compute_scharr_gradient_magnitude(luma: np.ndarray) -> np.ndarray:
    """Compute normalized Scharr gradient magnitude ||nabla I(x)||."""
    p = np.pad(luma.astype(np.float32), 1, mode="reflect")
    gx = (
        3.0 * (p[:-2, 2:] - p[:-2, :-2])
        + 10.0 * (p[1:-1, 2:] - p[1:-1, :-2])
        + 3.0 * (p[2:, 2:] - p[2:, :-2])
    ) / 32.0
    gy = (
        3.0 * (p[2:, :-2] - p[:-2, :-2])
        + 10.0 * (p[2:, 1:-1] - p[:-2, 1:-1])
        + 3.0 * (p[2:, 2:] - p[:-2, 2:])
    ) / 32.0
    mag = np.sqrt(gx * gx + gy * gy)
    scale = float(np.quantile(mag, 0.98))
    if scale > 1e-5:
        mag = np.clip(mag / scale, 0.0, 2.0)
    return mag.astype(np.float32)


def _bilinear_sample_2d(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample img at continuous (x_col, y_row) coordinates with integer pixel centers."""
    h, w = img.shape
    valid = (xs >= 0.0) & (xs <= w - 1.0) & (ys >= 0.0) & (ys <= h - 1.0)
    xc = np.clip(xs, 0.0, w - 1.0001)
    yc = np.clip(ys, 0.0, h - 1.0001)
    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (xc - x0).astype(np.float32)
    wy = (yc - y0).astype(np.float32)
    val = (
        (1.0 - wy) * ((1.0 - wx) * img[y0, x0] + wx * img[y0, x1])
        + wy * ((1.0 - wx) * img[y1, x0] + wx * img[y1, x1])
    )
    return val.astype(np.float32), valid


def check_topology_preserved(
    m_old: np.ndarray,
    m_new: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    n_fg_old: int | None = None,
    n_bg_old: int | None = None,
) -> bool:
    """Verify foreground (8-conn) and background/hole (4-conn) component counts are unchanged.

     forbidding new components, deleted targets, new holes, or merged targets.
    """
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        sub_old = m_old[y0:y1, x0:x1]
        sub_new = m_new[y0:y1, x0:x1]
        _, l_fg_old = nd_label(sub_old, structure=BOX_3)
        _, l_fg_new = nd_label(sub_new, structure=BOX_3)
        if l_fg_old != l_fg_new:
            return False
        _, l_bg_old = nd_label(~sub_old, structure=CROSS_3)
        _, l_bg_new = nd_label(~sub_new, structure=CROSS_3)
        if l_bg_old != l_bg_new:
            return False
        return True

    if n_fg_old is None:
        _, n_fg_old = nd_label(m_old, structure=BOX_3)
    if n_bg_old is None:
        _, n_bg_old = nd_label(~m_old, structure=CROSS_3)
    _, n_fg_new = nd_label(m_new, structure=BOX_3)
    if n_fg_new != n_fg_old:
        return False
    _, n_bg_new = nd_label(~m_new, structure=CROSS_3)
    return n_bg_new == n_bg_old


def _order_boundary_cycles(boundary_mask: np.ndarray) -> list[np.ndarray]:
    """Trace ordered 8-connected boundary pixel loops as arrays of shape [L, 2] in (x, y) order."""
    ys, xs = np.nonzero(boundary_mask)
    if len(xs) == 0:
        return []
    coords = np.stack([xs, ys], axis=1).astype(np.int32)
    coord_map: dict[tuple[int, int], int] = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
    visited = np.zeros(len(coords), dtype=bool)
    # Clockwise 8-neighbor offsets (dx, dy)
    nbrs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    cycles: list[np.ndarray] = []

    for start_idx in range(len(coords)):
        if visited[start_idx]:
            continue
        chain = [start_idx]
        visited[start_idx] = True
        cur = start_idx
        while True:
            cx, cy = int(coords[cur, 0]), int(coords[cur, 1])
            nxt = -1
            # Prefer 4-neighbors first to follow tight contour perimeter, then diagonals
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]:
                j = coord_map.get((cx + dx, cy + dy), -1)
                if j >= 0 and not visited[j]:
                    nxt = j
                    break
            if nxt < 0:
                break
            visited[nxt] = True
            chain.append(nxt)
            cur = nxt
        if len(chain) >= 4:
            cycles.append(coords[chain])
    return cycles


@dataclass
class ContourSegmentCorridor:
    seg_id: int
    comp_id: int
    boundary_xy: np.ndarray  # [L_s, 2] integer (x, y)
    arc_len: int
    normal_xy: np.ndarray  # [2] unit outward normal (nx, ny)
    tangent_xy: np.ndarray  # [2] unit tangent (tx, ty)
    corridor_y: np.ndarray  # [N_v] row indices in V_s
    corridor_x: np.ndarray  # [N_v] col indices in V_s
    corridor_u: np.ndarray  # [N_v] integer layer in {-8..-1} U {+1..+8}
    corridor_col: np.ndarray  # [N_v] nearest boundary point index in 0..L_s-1
    corridor_in_m0: np.ndarray  # [N_v] bool whether pixel is inside M_0
    candidate_shifts: list[int]  # admissible shifts d in {-8..8}, always includes 0
    add_indices_by_d: dict[int, np.ndarray]  # indices into V_s for A_s^+(d)
    rem_indices_by_d: dict[int, np.ndarray]  # indices into V_s for A_s^-(d)
    layer_indices_by_d: dict[int, np.ndarray]  # boundary layer indices in V_s for grad align B
    local_bbox: tuple[int, int, int, int]  # (y0, y1, x0, x1) padded bbox for fast topology check


@dataclass
class ComponentCarrier:
    comp_id: int
    valid: bool
    fallback_reason: str
    area_px: int
    fit_mask: np.ndarray | None = None
    holdout_mask: np.ndarray | None = None
    origin_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    omega_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    period_vec_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    wavelength_px: float = 0.0
    a_cos: float = 0.0
    b_sin: float = 0.0
    amplitude: float = 0.0
    phase_rad: float = 0.0
    holdout_r2: float = 0.0
    phase_cos_fh: float = 0.0
    sigma2: float = 1e-3
    fit_var: float = 1e-3


def _split_cycle_into_segments(
    cycle_xy: np.ndarray,
    comp_mask: np.ndarray,
    dt_in: np.ndarray,
    dt_out: np.ndarray,
    cfg: PhaseBoundaryConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split an ordered boundary chain into segments at corners >45 deg and arc length L_seg."""
    n_pts = len(cycle_xy)
    if n_pts < cfg.min_segment_arc_len_px:
        return []

    h, w = comp_mask.shape
    signed_dt = dt_out - dt_in  # increases outward (from inside <0 to outside >0)

    # Compute local outward gradient of signed distance field at each boundary point
    normals = np.zeros((n_pts, 2), dtype=np.float32)
    for i, (x, y) in enumerate(cycle_xy):
        x0, x1 = max(0, x - 2), min(w - 1, x + 2)
        y0, y1 = max(0, y - 2), min(h - 1, y + 2)
        gx = float(signed_dt[y, x1] - signed_dt[y, x0])
        gy = float(signed_dt[y1, x] - signed_dt[y0, x])
        norm = math.hypot(gx, gy)
        if norm > 1e-6:
            normals[i, 0] = gx / norm
            normals[i, 1] = gy / norm
        else:
            normals[i, 0] = 1.0
            normals[i, 1] = 0.0

    # Smooth point normals with 3-point moving window along chain
    sm_normals = normals.copy()
    if n_pts >= 5:
        for i in range(n_pts):
            v = normals[(i - 1) % n_pts] + normals[i] + normals[(i + 1) % n_pts]
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                sm_normals[i] = v / nv

    cos_thr = math.cos(math.radians(cfg.corner_split_angle_deg))
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    cur_indices: list[int] = [0]
    anchor_n = sm_normals[0].copy()

    for i in range(1, n_pts):
        dot = float(np.dot(sm_normals[i], anchor_n))
        if dot < cos_thr or len(cur_indices) >= cfg.target_segment_arc_len_px:
            if len(cur_indices) >= cfg.min_segment_arc_len_px:
                pts = cycle_xy[cur_indices]
                n_mean = sm_normals[cur_indices].mean(axis=0)
                nn = float(np.linalg.norm(n_mean))
                if nn > 1e-6:
                    segments.append((pts, (n_mean / nn).astype(np.float32)))
            cur_indices = [i]
            anchor_n = sm_normals[i].copy()
        else:
            cur_indices.append(i)
            n_mean = sm_normals[cur_indices].mean(axis=0)
            nn = float(np.linalg.norm(n_mean))
            if nn > 1e-6:
                anchor_n = n_mean / nn

    if len(cur_indices) >= cfg.min_segment_arc_len_px:
        pts = cycle_xy[cur_indices]
        n_mean = sm_normals[cur_indices].mean(axis=0)
        nn = float(np.linalg.norm(n_mean))
        if nn > 1e-6:
            segments.append((pts, (n_mean / nn).astype(np.float32)))
    elif segments and len(cur_indices) > 0:
        # Merge short trailing tail if compatible in normal orientation
        prev_pts, prev_n = segments[-1]
        tail_n = sm_normals[cur_indices].mean(axis=0)
        if float(np.dot(prev_n, tail_n)) >= cos_thr and (len(prev_pts) + len(cur_indices)) <= int(1.5 * cfg.target_segment_arc_len_px):
            merged_pts = np.concatenate([prev_pts, cycle_xy[cur_indices]], axis=0)
            n_mean = prev_n * len(prev_pts) + tail_n * len(cur_indices)
            nn = float(np.linalg.norm(n_mean))
            if nn > 1e-6:
                segments[-1] = (merged_pts, (n_mean / nn).astype(np.float32))
    return segments


def build_component_geometry_and_corridors(
    m0: np.ndarray,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
) -> tuple[np.ndarray, int, dict[int, list[ContourSegmentCorridor]], dict[str, int]]:
    """Extract 8-connected components and build fixed Voronoi-partitioned +/-8px corridors V_s."""
    m0_b = np.asarray(m0, dtype=bool)
    h, w = m0_b.shape
    comp_labels, num_fg = nd_label(m0_b, structure=BOX_3)
    corridors_by_comp: dict[int, list[ContourSegmentCorridor]] = {}
    stats = {
        "total_components": int(num_fg),
        "small_skipped_components": 0,
        "total_segments": 0,
        "total_arc_len_px": 0,
    }
    if num_fg == 0:
        return comp_labels, 0, corridors_by_comp, stats

    max_d = cfg.max_shift_px
    seg_global_id = 0

    for cid in range(1, num_fg + 1):
        comp = comp_labels == cid
        area = int(comp.sum())
        if area < cfg.min_component_area_px:
            stats["small_skipped_components"] += 1
            corridors_by_comp[cid] = []
            continue

        # Boundary pixels of component cid (4-adjacent to background)
        c_ys, c_xs = np.nonzero(comp)
        pad_c = max_d + 6
        cy0 = max(0, int(c_ys.min()) - pad_c)
        cy1 = min(h, int(c_ys.max()) + pad_c + 1)
        cx0 = max(0, int(c_xs.min()) - pad_c)
        cx1 = min(w, int(c_xs.max()) + pad_c + 1)

        sub_comp = comp[cy0:cy1, cx0:cx1]
        sub_m0 = m0_b[cy0:cy1, cx0:cx1]
        eroded_sub = binary_erosion(sub_comp, structure=CROSS_3, border_value=0)
        b_mask_sub = sub_comp & (~eroded_sub)
        if not b_mask_sub.any():
            corridors_by_comp[cid] = []
            continue

        dt_in_sub = distance_transform_edt(sub_comp).astype(np.float32)
        dt_out_sub = distance_transform_edt(~sub_comp).astype(np.float32)

        other_fg_sub = sub_m0 & (~sub_comp)
        if other_fg_sub.any():
            dt_other_sub = distance_transform_edt(~other_fg_sub).astype(np.float32)
        else:
            dt_other_sub = np.full(sub_comp.shape, 999.0, dtype=np.float32)

        cycles_sub = _order_boundary_cycles(b_mask_sub)
        raw_segs: list[tuple[np.ndarray, np.ndarray]] = []
        for cyc_sub in cycles_sub:
            segs_sub = _split_cycle_into_segments(cyc_sub, sub_comp, dt_in_sub, dt_out_sub, cfg)
            for pts_sub, n_vec in segs_sub:
                pts_full = pts_sub.copy()
                pts_full[:, 0] += cx0
                pts_full[:, 1] += cy0
                raw_segs.append((pts_full, n_vec))

        if not raw_segs:
            corridors_by_comp[cid] = []
            continue

        # Assign every boundary pixel in raw_segs a segment index and column index inside sub-box
        sh, sw = sub_comp.shape
        b_seg_id = np.full((sh, sw), -1, dtype=np.int32)
        b_col_id = np.full((sh, sw), -1, dtype=np.int32)
        seg_pts_list: list[np.ndarray] = []
        seg_norm_list: list[np.ndarray] = []

        for s_idx, (pts, n_vec) in enumerate(raw_segs):
            seg_pts_list.append(pts)
            seg_norm_list.append(n_vec)
            for col_i, (px, py) in enumerate(pts):
                b_seg_id[py - cy0, px - cx0] = s_idx
                b_col_id[py - cy0, px - cx0] = col_i

        active_b = b_seg_id >= 0
        if not active_b.any():
            corridors_by_comp[cid] = []
            continue

        # Nearest active boundary pixel via exact EDT indices inside sub-box
        dist_to_b, (near_y_sub, near_x_sub) = distance_transform_edt(~active_b, return_indices=True)
        ribbon = (dist_to_b <= (max_d + 0.75)) & (sub_comp | (~sub_m0))
        ribbon = ribbon & (sub_comp | (dt_other_sub > cfg.min_inter_component_clearance_px))

        rib_y_sub, rib_x_sub = np.nonzero(ribbon)
        rib_ny_sub = near_y_sub[rib_y_sub, rib_x_sub]
        rib_nx_sub = near_x_sub[rib_y_sub, rib_x_sub]
        rib_seg = b_seg_id[rib_ny_sub, rib_nx_sub]
        rib_col = b_col_id[rib_ny_sub, rib_nx_sub]

        rib_y = rib_y_sub + cy0
        rib_x = rib_x_sub + cx0
        rib_ny = rib_ny_sub + cy0
        rib_nx = rib_nx_sub + cx0

        comp_corridors: list[ContourSegmentCorridor] = []
        n_segs_c = len(seg_pts_list)

        all_b_y_sub, all_b_x_sub = np.nonzero(active_b)
        all_b_seg = b_seg_id[all_b_y_sub, all_b_x_sub]
        all_b_y = all_b_y_sub + cy0
        all_b_x = all_b_x_sub + cx0

        # Group ribbon pixels by segment index in O(N) via argsort + split
        valid_rib = rib_seg >= 0
        rib_y = rib_y[valid_rib]
        rib_x = rib_x[valid_rib]
        rib_ny = rib_ny[valid_rib]
        rib_nx = rib_nx[valid_rib]
        rib_col = rib_col[valid_rib]
        rib_seg = rib_seg[valid_rib]

        order_rib = np.argsort(rib_seg, kind="mergesort")
        rib_y = rib_y[order_rib]
        rib_x = rib_x[order_rib]
        rib_ny = rib_ny[order_rib]
        rib_nx = rib_nx[order_rib]
        rib_col = rib_col[order_rib]
        rib_seg_sorted = rib_seg[order_rib]
        uniq_s, split_idx = np.unique(rib_seg_sorted, return_index=True)
        slices_by_s = {
            int(s_val): slice(int(split_idx[i]), int(split_idx[i + 1]) if (i + 1) < len(split_idx) else len(rib_seg_sorted))
            for i, s_val in enumerate(uniq_s)
        }

        for s_idx in range(n_segs_c):
            sl = slices_by_s.get(s_idx)
            if sl is None:
                continue
            pts = seg_pts_list[s_idx]
            n_vec = seg_norm_list[s_idx]
            t_vec = np.array([-n_vec[1], n_vec[0]], dtype=np.float32)
            l_s = len(pts)

            vy = rib_y[sl]
            vx = rib_x[sl]
            vny = rib_ny[sl]
            vnx = rib_nx[sl]
            vcol = rib_col[sl]
            in_m0 = comp[vy, vx]

            # Project displacement from nearest boundary pixel onto fixed outward normal n_vec
            dx = (vx - vnx).astype(np.float32)
            dy = (vy - vny).astype(np.float32)
            proj_n = dx * n_vec[0] + dy * n_vec[1]

            # Filter pixels with coherent normal orientation
            valid_dir = np.where(in_m0, proj_n <= 0.35, proj_n >= 0.25)
            if not np.any(valid_dir):
                continue
            vy = vy[valid_dir]
            vx = vx[valid_dir]
            vcol = vcol[valid_dir]
            in_m0 = in_m0[valid_dir]
            proj_n = proj_n[valid_dir]

            # Convert to discrete signed layer u_s in {-max_d..-1} (inside) and {+1..+max_d} (outside)
            u_layer = np.zeros(len(vy), dtype=np.int32)
            out_m = ~in_m0
            in_m = in_m0
            u_layer[out_m] = np.clip(np.round(proj_n[out_m]).astype(np.int32), 1, max_d)
            u_layer[in_m] = -np.clip(np.round(-proj_n[in_m]).astype(np.int32) + 1, 1, max_d)

            # Local padded bounding box (pad = max_d + 4 >= max(|u|) + clearance)
            pad = max_d + 4
            by0 = max(0, int(vy.min()) - pad)
            by1 = min(h, int(vy.max()) + pad + 1)
            bx0 = max(0, int(vx.min()) - pad)
            bx1 = min(w, int(vx.max()) + pad + 1)

            # Opposite-shore check for interior pixels restricted to local_bbox (exact same result, 50x faster)
            non_adj = (
                (np.abs(all_b_seg - s_idx) > 1)
                & (np.abs(all_b_seg - s_idx) < (n_segs_c - 1))
                & (all_b_y >= by0)
                & (all_b_y < by1)
                & (all_b_x >= bx0)
                & (all_b_x < bx1)
            )
            if np.any(non_adj) and np.any(in_m):
                opp_y = all_b_y[non_adj]
                opp_x = all_b_x[non_adj]
                in_idx = np.flatnonzero(in_m)
                iy = vy[in_idx][:, None]
                ix = vx[in_idx][:, None]
                min_d_opp = np.sqrt(np.min((iy - opp_y[None, :]) ** 2 + (ix - opp_x[None, :]) ** 2, axis=1))
                safe_in = min_d_opp > (np.abs(u_layer[in_idx]) + cfg.min_opposite_shore_clearance_px)
                keep_mask = np.ones(len(vy), dtype=bool)
                keep_mask[in_idx[~safe_in]] = False
                vy = vy[keep_mask]
                vx = vx[keep_mask]
                vcol = vcol[keep_mask]
                in_m0 = in_m0[keep_mask]
                u_layer = u_layer[keep_mask]

            if len(vy) == 0:
                continue

            local_bbox = (by0, by1, bx0, bx1)
            sub_m0_box = m0_b[by0:by1, bx0:bx1]
            _, box_fg_old = nd_label(sub_m0_box, structure=BOX_3)
            _, box_bg_old = nd_label(~sub_m0_box, structure=CROSS_3)
            vy_rel = vy - by0
            vx_rel = vx - bx0

            min_cols = max(2, int(math.ceil(cfg.min_valid_column_coverage * l_s)))
            candidate_shifts = [0]
            add_indices_by_d: dict[int, np.ndarray] = {0: np.empty(0, dtype=np.int32)}
            rem_indices_by_d: dict[int, np.ndarray] = {0: np.empty(0, dtype=np.int32)}
            base_b_idx = np.flatnonzero(u_layer == -1).astype(np.int32)
            layer_indices_by_d: dict[int, np.ndarray] = {0: base_b_idx}

            # Check outward shifts d = +1 .. +max_d
            for d in range(1, max_d + 1):
                layer_d = np.flatnonzero(u_layer == d).astype(np.int32)
                if len(layer_d) == 0:
                    break
                cum_add = np.flatnonzero((u_layer >= 1) & (u_layer <= d)).astype(np.int32)
                n_cov_cols = len(np.unique(vcol[cum_add]))
                if n_cov_cols < min_cols:
                    break
                candidate_shifts.append(d)
                add_indices_by_d[d] = cum_add
                rem_indices_by_d[d] = np.empty(0, dtype=np.int32)
                layer_indices_by_d[d] = layer_d

            # Verify topology at maximum outward shift; if violated, prune from largest d downward
            while any(d > 0 for d in candidate_shifts):
                d_max_out = max(d for d in candidate_shifts if d > 0)
                sub_cand = sub_m0_box.copy()
                cum_add = add_indices_by_d[d_max_out]
                sub_cand[vy_rel[cum_add], vx_rel[cum_add]] = True
                if nd_label(sub_cand, structure=BOX_3)[1] == box_fg_old and nd_label(~sub_cand, structure=CROSS_3)[1] == box_bg_old:
                    break
                candidate_shifts.remove(d_max_out)
                del add_indices_by_d[d_max_out]
                del rem_indices_by_d[d_max_out]
                del layer_indices_by_d[d_max_out]

            # Check inward shifts d = -1 .. -max_d
            for d_abs in range(1, max_d + 1):
                d = -d_abs
                layer_d = np.flatnonzero(u_layer == d).astype(np.int32)
                if len(layer_d) == 0:
                    break
                cum_rem = np.flatnonzero((u_layer >= d) & (u_layer <= -1)).astype(np.int32)
                n_cov_cols = len(np.unique(vcol[cum_rem]))
                if n_cov_cols < min_cols:
                    break
                candidate_shifts.append(d)
                add_indices_by_d[d] = np.empty(0, dtype=np.int32)
                rem_indices_by_d[d] = cum_rem
                nxt_layer = np.flatnonzero(u_layer == (d - 1)).astype(np.int32)
                layer_indices_by_d[d] = nxt_layer if len(nxt_layer) > 0 else layer_d

            # Verify topology at maximum inward shift; if violated, prune from most negative d upward
            while any(d < 0 for d in candidate_shifts):
                d_min_in = min(d for d in candidate_shifts if d < 0)
                sub_cand = sub_m0_box.copy()
                cum_rem = rem_indices_by_d[d_min_in]
                sub_cand[vy_rel[cum_rem], vx_rel[cum_rem]] = False
                if nd_label(sub_cand, structure=BOX_3)[1] == box_fg_old and nd_label(~sub_cand, structure=CROSS_3)[1] == box_bg_old:
                    break
                candidate_shifts.remove(d_min_in)
                del add_indices_by_d[d_min_in]
                del rem_indices_by_d[d_min_in]
                del layer_indices_by_d[d_min_in]

            candidate_shifts.sort()
            comp_corridors.append(
                ContourSegmentCorridor(
                    seg_id=seg_global_id,
                    comp_id=cid,
                    boundary_xy=pts,
                    arc_len=l_s,
                    normal_xy=n_vec,
                    tangent_xy=t_vec,
                    corridor_y=vy.astype(np.int32),
                    corridor_x=vx.astype(np.int32),
                    corridor_u=u_layer.astype(np.int32),
                    corridor_col=vcol.astype(np.int32),
                    corridor_in_m0=in_m0,
                    candidate_shifts=candidate_shifts,
                    add_indices_by_d=add_indices_by_d,
                    rem_indices_by_d=rem_indices_by_d,
                    layer_indices_by_d=layer_indices_by_d,
                    local_bbox=local_bbox,
                )
            )
            seg_global_id += 1
            stats["total_segments"] += 1
            stats["total_arc_len_px"] += l_s

        corridors_by_comp[cid] = comp_corridors

    return comp_labels, int(num_fg), corridors_by_comp, stats


def _partition_interior_fit_holdout(
    interior_mask: np.ndarray,
    cfg: PhaseBoundaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Split eroded interior I_c into spatially separated fit block F and holdout block H."""
    ys, xs = np.nonzero(interior_mask)
    if len(xs) < (cfg.min_fit_pixels + cfg.min_holdout_pixels):
        empty = np.zeros_like(interior_mask, dtype=bool)
        return empty, empty

    bsize = cfg.fit_holdout_block_size_px
    gap = cfg.fit_holdout_guard_gap_px
    y0, x0 = int(ys.min()), int(xs.min())
    by = (ys - y0) // bsize
    bx = (xs - x0) // bsize
    ry = (ys - y0) % bsize
    rx = (xs - x0) % bsize

    # Guard gap at block borders prevents adjacent pixels from straddling F and H
    in_core = (ry >= gap) & (ry < (bsize - gap)) & (rx >= gap) & (rx < (bsize - gap))
    parity = ((by + bx) % 2) == 0

    fit_m = np.zeros_like(interior_mask, dtype=bool)
    hold_m = np.zeros_like(interior_mask, dtype=bool)
    fit_idx = in_core & parity
    hold_idx = in_core & (~parity)
    fit_m[ys[fit_idx], xs[fit_idx]] = True
    hold_m[ys[hold_idx], xs[hold_idx]] = True

    if int(fit_m.sum()) >= cfg.min_fit_pixels and int(hold_m.sum()) >= cfg.min_holdout_pixels:
        return fit_m, hold_m

    # Fallback for narrow/compact interiors: split along principal major axis with 2px guard gap
    coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1).astype(np.float32)
    cov = (coords.T @ coords) / max(1, len(coords))
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    proj = coords @ major
    med = float(np.median(proj))

    fit_m.fill(False)
    hold_m.fill(False)
    # Use 60% for fit and 40% for holdout separated by guard gap
    split_pt = float(np.quantile(proj, 0.58))
    f_sel = proj <= (split_pt - 0.5 * gap)
    h_sel = proj >= (split_pt + 0.5 * gap)
    fit_m[ys[f_sel], xs[f_sel]] = True
    hold_m[ys[h_sel], xs[h_sel]] = True
    return fit_m, hold_m


def _propose_candidate_wavevectors(
    residual: np.ndarray,
    fit_mask: np.ndarray,
    period_min_px: float,
    period_max_px: float,
    cfg: PhaseBoundaryConfig,
) -> list[np.ndarray]:
    """Propose candidate wavevectors omega in R^2 from F's 2D spectrum + structure tensor orientation."""
    ys, xs = np.nonzero(fit_mask)
    if len(xs) < cfg.min_fit_pixels:
        return []

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch_r = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    patch_m = fit_mask[y0:y1, x0:x1]
    vals = residual[ys, xs]
    patch_r[patch_m] = vals - float(vals.mean())

    n_fft = max(cfg.fft_grid_size, 1 << int(math.ceil(math.log2(max(y1 - y0, x1 - x0, 32)))))
    spec = np.fft.rfft2(patch_r, s=(n_fft, n_fft))
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    fy = np.fft.fftfreq(n_fft, d=1.0) * (2.0 * math.pi)  # rad/px
    fx = np.fft.rfftfreq(n_fft, d=1.0) * (2.0 * math.pi)
    gy, gx = np.meshgrid(fy, fx, indexing="ij")
    w_mag = np.sqrt(gx * gx + gy * gy)

    w_min = 2.0 * math.pi / period_max_px
    w_max = 2.0 * math.pi / period_min_px
    ring = (w_mag >= w_min) & (w_mag <= w_max) & (gx >= 0.0)
    if not np.any(ring):
        return []

    masked_p = np.where(ring, power, -1.0)
    flat_order = np.argsort(-masked_p.ravel())
    candidates: list[np.ndarray] = []

    for idx in flat_order:
        if masked_p.ravel()[idx] <= 0.0:
            break
        iy, ix = divmod(int(idx), masked_p.shape[1])
        w_vec = np.array([float(gx[iy, ix]), float(gy[iy, ix])], dtype=np.float32)
        # Enforce minimum separation in frequency space to diversify candidates
        if all(float(np.linalg.norm(w_vec - c)) > 0.08 and float(np.linalg.norm(w_vec + c)) > 0.08 for c in candidates):
            candidates.append(w_vec)
            if len(candidates) >= (cfg.top_candidate_frequencies - 3):
                break

    # Supplement with structure-tensor gradient orientation hint on F (only direction, never contour normal)
    p_res = np.pad(residual, 1, mode="reflect")
    rx = 0.5 * (p_res[1:-1, 2:] - p_res[1:-1, :-2])[ys, xs]
    ry = 0.5 * (p_res[2:, 1:-1] - p_res[:-2, 1:-1])[ys, xs]
    jxx = float(np.mean(rx * rx))
    jyy = float(np.mean(ry * ry))
    jxy = float(np.mean(rx * ry))
    if (jxx + jyy) > 1e-8:
        theta_st = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
        dir_st = np.array([math.cos(theta_st), math.sin(theta_st)], dtype=np.float32)
        for lam in (0.3 * period_min_px + 0.7 * period_max_px, 0.5 * (period_min_px + period_max_px), 0.7 * period_min_px + 0.3 * period_max_px):
            if period_min_px <= lam <= period_max_px:
                candidates.append(dir_st * float(2.0 * math.pi / lam))

    return candidates[: cfg.top_candidate_frequencies]


def _fit_carrier_least_squares(
    coords_xy: np.ndarray,
    vals: np.ndarray,
    origin_xy: np.ndarray,
    omega_xy: np.ndarray,
    max_cond: float,
) -> tuple[bool, float, float, float]:
    """Solve weighted least-squares (a, b) for r_hat(x) = a cos(w^T(x-x0)) + b sin(w^T(x-x0))."""
    phase = (coords_xy - origin_xy[None, :]) @ omega_xy
    c = np.cos(phase).astype(np.float64)
    s = np.sin(phase).astype(np.float64)
    m00 = float(np.dot(c, c))
    m01 = float(np.dot(c, s))
    m11 = float(np.dot(s, s))
    mat = np.array([[m00, m01], [m01, m11]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(mat)
    if eigvals[0] <= 1e-6:
        return False, 0.0, 0.0, float("inf")
    cond = float(eigvals[1] / eigvals[0])
    if cond > max_cond:
        return False, 0.0, 0.0, cond
    rhs = np.array([float(np.dot(c, vals)), float(np.dot(s, vals))], dtype=np.float64)
    ab = np.linalg.solve(mat, rhs)
    return True, float(ab[0]), float(ab[1]), cond


def estimate_and_freeze_component_carrier(
    comp_id: int,
    comp_mask: np.ndarray,
    corridor_union_mask: np.ndarray,
    residual: np.ndarray,
    *,
    period_range_px: tuple[float, float] | None = None,
    holdout_r2_floor: float | None = None,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
) -> ComponentCarrier:
    """Estimate, validate on spatially separated holdout H, and freeze interior carrier (omega, a, b)."""
    area = int(comp_mask.sum())
    if area < cfg.min_component_area_px:
        return ComponentCarrier(comp_id=comp_id, valid=False, fallback_reason="component_too_small", area_px=area)

    # Erode by interior_erosion_px (8px) and strictly exclude any corridor pixel in U_s V_s
    disk_e = DISK_8 if cfg.interior_erosion_px == 8 else _disk_structure(cfg.interior_erosion_px)
    interior = binary_erosion(comp_mask, structure=disk_e, border_value=0) & (~corridor_union_mask)
    if int(interior.sum()) < (cfg.min_fit_pixels + cfg.min_holdout_pixels):
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="insufficient_eroded_interior",
            area_px=area,
        )

    fit_mask, hold_mask = _partition_interior_fit_holdout(interior, cfg)
    n_fit = int(fit_mask.sum())
    n_hold = int(hold_mask.sum())
    if n_fit < cfg.min_fit_pixels or n_hold < cfg.min_holdout_pixels:
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="insufficient_fit_or_holdout_split",
            area_px=area,
            fit_mask=fit_mask,
            holdout_mask=hold_mask,
        )

    f_y, f_x = np.nonzero(fit_mask)
    h_y, h_x = np.nonzero(hold_mask)
    f_xy = np.stack([f_x, f_y], axis=1).astype(np.float32)
    h_xy = np.stack([h_x, h_y], axis=1).astype(np.float32)
    r_f = residual[f_y, f_x].astype(np.float32)
    r_h = residual[h_y, h_x].astype(np.float32)

    hold_energy = float(np.dot(r_h, r_h))
    fit_var = float(np.var(r_f))
    if hold_energy < cfg.min_holdout_energy or fit_var < (0.25 * cfg.noise_var_floor):
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="low_interior_signal_energy",
            area_px=area,
            fit_mask=fit_mask,
            holdout_mask=hold_mask,
        )

    p_min, p_max = period_range_px if period_range_px is not None else (cfg.default_period_min_px, cfg.default_period_max_px)
    p_min = max(cfg.hard_sampling_period_min_px, float(p_min))
    p_max = min(cfg.hard_sampling_period_max_px, float(p_max))
    if p_min >= p_max:
        p_min, p_max = cfg.default_period_min_px, cfg.default_period_max_px

    origin_xy = f_xy.mean(axis=0).astype(np.float32)
    candidates = _propose_candidate_wavevectors(residual, fit_mask, p_min, p_max, cfg)
    if not candidates:
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="no_candidate_frequency_in_band",
            area_px=area,
            fit_mask=fit_mask,
            holdout_mask=hold_mask,
        )

    w_min = 2.0 * math.pi / p_max
    w_max = 2.0 * math.pi / p_min

    # Local sub-grid refinement of top-4 FFT/ST candidates on F only (never touching H!)
    # Use deterministic stride cap (<=1024 pts) during grid search, then fit final (a,b) on full F.
    sub_stride = max(1, len(f_xy) // 1024)
    f_xy_search = f_xy[::sub_stride]
    r_f_search = r_f[::sub_stride]

    refined_candidates: list[tuple[np.ndarray, float, float]] = []
    for w0 in candidates[:6]:
        best_w = w0.copy()
        best_f_sse = float("inf")
        for step in (0.035, 0.010):
            cur_center = best_w.copy()
            for dwx in (-step, 0.0, step):
                for dwy in (-step, 0.0, step):
                    w_try = cur_center + np.array([dwx, dwy], dtype=np.float32)
                    wm = float(np.linalg.norm(w_try))
                    if wm < w_min or wm > w_max:
                        continue
                    lam = 2.0 * math.pi / wm
                    w_unit = w_try / wm
                    span = float(np.ptp(f_xy_search @ w_unit))
                    if span < (cfg.min_span_periods * lam):
                        continue
                    ok, a_val, b_val, _ = _fit_carrier_least_squares(
                        f_xy_search, r_f_search, origin_xy, w_try, cfg.max_normal_matrix_cond
                    )
                    if not ok:
                        continue
                    ph = (f_xy_search - origin_xy[None, :]) @ w_try
                    pred_f = a_val * np.cos(ph) + b_val * np.sin(ph)
                    sse = float(np.sum((r_f_search - pred_f) ** 2))
                    if sse < best_f_sse:
                        best_f_sse = sse
                        best_w = w_try
        if math.isfinite(best_f_sse):
            ok_full, a_full, b_full, _ = _fit_carrier_least_squares(
                f_xy, r_f, origin_xy, best_w, cfg.max_normal_matrix_cond
            )
            if ok_full:
                refined_candidates.append((best_w, a_full, b_full))

    if not refined_candidates:
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="ill_conditioned_or_insufficient_period_span",
            area_px=area,
            fit_mask=fit_mask,
            holdout_mask=hold_mask,
        )

    # Evaluate on spatially separated holdout H
    best_s_h = -float("inf")
    best_tuple: tuple[np.ndarray, float, float, float] | None = None

    for w_vec, a_f, b_f in refined_candidates:
        ph_h = (h_xy - origin_xy[None, :]) @ w_vec
        pred_h = a_f * np.cos(ph_h) + b_f * np.sin(ph_h)
        s_h = 1.0 - float(np.sum((r_h - pred_h) ** 2)) / hold_energy
        if s_h > best_s_h:
            # Verify phase consistency when fitted independently on H
            ok_h, a_h, b_h, _ = _fit_carrier_least_squares(
                h_xy, r_h, origin_xy, w_vec, cfg.max_normal_matrix_cond
            )
            if ok_h:
                amp_f = math.hypot(a_f, b_f)
                amp_h = math.hypot(a_h, b_h)
                if amp_f > 1e-6 and amp_h > 1e-6:
                    cos_fh = (a_f * a_h + b_f * b_h) / (amp_f * amp_h)
                else:
                    cos_fh = -1.0
            else:
                cos_fh = -1.0
            best_s_h = s_h
            best_tuple = (w_vec, a_f, b_f, float(cos_fh))

    r2_min = max(cfg.min_holdout_r2_floor, float(holdout_r2_floor or 0.0))
    if best_tuple is None or best_s_h < r2_min:
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="holdout_r2_below_threshold",
            area_px=area,
            fit_mask=fit_mask,
            holdout_mask=hold_mask,
            holdout_r2=float(best_s_h) if math.isfinite(best_s_h) else -1.0,
        )

    w_star, a_star, b_star, cos_fh = best_tuple
    if cos_fh < cfg.min_fit_holdout_phase_cos:
        return ComponentCarrier(
            comp_id=comp_id,
            valid=False,
            fallback_reason="fit_holdout_phase_inconsistent",
            area_px=area,
            fit_mask=fit_mask,
            holdout_mask=hold_mask,
            holdout_r2=float(best_s_h),
            phase_cos_fh=float(cos_fh),
        )

    # Estimate robust noise variance sigma_s^2 from holdout residuals via MAD
    ph_h = (h_xy - origin_xy[None, :]) @ w_star
    res_h = r_h - (a_star * np.cos(ph_h) + b_star * np.sin(ph_h))
    mad = float(np.median(np.abs(res_h - np.median(res_h))))
    sigma_mad = 1.4826 * mad
    sigma2 = max(sigma_mad * sigma_mad, cfg.noise_var_floor)

    w_norm2 = float(np.dot(w_star, w_star))
    period_vec = (2.0 * math.pi * w_star / max(w_norm2, 1e-8)).astype(np.float32)
    wavelength = float(2.0 * math.pi / math.sqrt(max(w_norm2, 1e-8)))
    amp = math.hypot(a_star, b_star)
    # r_hat(x) = a cos(theta) + b sin(theta) = A cos(theta + phi0) where phi0 = atan2(-b, a)
    phase_rad = math.atan2(-b_star, a_star)

    return ComponentCarrier(
        comp_id=comp_id,
        valid=True,
        fallback_reason="none",
        area_px=area,
        fit_mask=fit_mask,
        holdout_mask=hold_mask,
        origin_xy=origin_xy,
        omega_xy=w_star.astype(np.float32),
        period_vec_xy=period_vec,
        wavelength_px=wavelength,
        a_cos=float(a_star),
        b_sin=float(b_star),
        amplitude=float(amp),
        phase_rad=float(phase_rad),
        holdout_r2=float(best_s_h),
        phase_cos_fh=float(cos_fh),
        sigma2=float(sigma2),
        fit_var=float(fit_var),
    )


def select_best_shift_with_tiebreak(
    gains_by_d: dict[int, float],
    threshold: float,
    *,
    tol: float = 1e-12,
) -> tuple[int, float]:
    """Select optimal displacement d* with deterministic tie-breaking:

    1. Maximum G_s(d);
    2. Prefer d = 0 on ties;
    3. Prefer smaller |d|;
    4. Prefer inward (d < 0) before outward (d > 0).
    Only moves if G_s(d*) > threshold.
    """
    best_d = 0
    best_g = float(gains_by_d.get(0, 0.0))

    # Order candidates by tie-break priority so strict '>' implements exact tie-breaking
    ordered_ds = sorted(
        gains_by_d.keys(),
        key=lambda d: (0 if d == 0 else 1, abs(d), 0 if d < 0 else 1),
    )
    for d in ordered_ds:
        g = float(gains_by_d[d])
        if g > best_g + tol:
            best_g = g
            best_d = d

    if best_d != 0 and best_g > threshold:
        return best_d, best_g
    return 0, 0.0


def score_segment_candidates(
    seg: ContourSegmentCorridor,
    variant: str,
    *,
    residual: np.ndarray,
    grad_mag: np.ndarray,
    local_var_map: np.ndarray,
    prob_512: np.ndarray,
    carrier: ComponentCarrier | None,
    p_floor: float | None,
    allow_outward: bool,
    gamma_offset_rad: float = 0.0,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
) -> dict[str, Any]:
    """Score all admissible candidates d in D_s for one contour segment under variant B..H."""
    l_s = float(max(1, seg.arc_len))
    vy = seg.corridor_y
    vx = seg.corridor_x
    v_xy = np.stack([vx, vy], axis=1).astype(np.float32)
    r_v = residual[vy, vx].astype(np.float32)

    needs_carrier = variant in (
        "D_BilateralTerm",
        "E_FreqAmp_NoPhase",
        "F_FullPhase",
        "G_PhaseCounterfactual",
        "H_NoSemGuard",
    )
    if needs_carrier and (carrier is None or not carrier.valid):
        return {
            "evaluable": False,
            "reason": carrier.fallback_reason if carrier is not None else "no_carrier",
            "gains_by_d": {0: 0.0},
            "evidence_v": None,
        }

    use_sem_guard = variant != "H_NoSemGuard"

    def _passes_outward_guard(d: int) -> bool:
        if d <= 0:
            return True
        if not allow_outward:
            return False
        if not use_sem_guard or p_floor is None:
            return True
        add_idx = seg.add_indices_by_d[d]
        if len(add_idx) == 0:
            return False
        mean_p0 = float(prob_512[vy[add_idx], vx[add_idx]].mean())
        return mean_p0 >= p_floor

    admissible_ds = [d for d in seg.candidate_shifts if _passes_outward_guard(d)]
    if 0 not in admissible_ds:
        admissible_ds.append(0)
    admissible_ds.sort()

    gains_by_d: dict[int, float] = {0: 0.0}
    evidence_v: np.ndarray | None = None

    if variant == "B_GradAlign":
        g_v = grad_mag[vy, vx].astype(np.float32)
        base_idx = seg.layer_indices_by_d[0]
        base_g = float(g_v[base_idx].sum()) if len(base_idx) > 0 else 0.0
        for d in admissible_ds:
            if d == 0:
                continue
            l_idx = seg.layer_indices_by_d.get(d, np.empty(0, dtype=np.int32))
            if len(l_idx) == 0:
                continue
            gains_by_d[d] = (float(g_v[l_idx].sum()) - base_g) / l_s
        evidence_v = g_v

    elif variant == "C_TexVar":
        v_v = local_var_map[vy, vx].astype(np.float32)
        in_v = v_v[seg.corridor_in_m0]
        out_v = v_v[~seg.corridor_in_m0]
        if len(in_v) < 4 or len(out_v) < 4:
            return {"evaluable": False, "reason": "insufficient_corridor_sides", "gains_by_d": {0: 0.0}, "evidence_v": None}
        mu_in = float(carrier.fit_var) if (carrier is not None and carrier.valid) else float(np.median(in_v))
        mu_out = float(np.median(out_v))
        var_scale = max(float(np.var(v_v)), 1e-6)
        e_v = (((v_v - mu_out) ** 2 - (v_v - mu_in) ** 2) / (2.0 * var_scale)).astype(np.float32)
        for d in admissible_ds:
            if d == 0:
                continue
            add_idx = seg.add_indices_by_d[d]
            rem_idx = seg.rem_indices_by_d[d]
            g_val = (float(e_v[add_idx].sum()) if len(add_idx) else 0.0) - (float(e_v[rem_idx].sum()) if len(rem_idx) else 0.0)
            gains_by_d[d] = g_val / l_s
        evidence_v = e_v

    elif variant == "D_BilateralTerm":
        assert carrier is not None
        v_half = 0.5 * carrier.period_vec_xy
        r_minus, ok_m = _bilinear_sample_2d(residual, v_xy[:, 0] - v_half[0], v_xy[:, 1] - v_half[1])
        r_plus, ok_p = _bilinear_sample_2d(residual, v_xy[:, 0] + v_half[0], v_xy[:, 1] + v_half[1])
        r_zero = r_v
        u_Minus = seg.corridor_u.astype(np.float32) - 0.5 * float(np.dot(carrier.period_vec_xy, seg.normal_xy))
        u_Plus = seg.corridor_u.astype(np.float32) + 0.5 * float(np.dot(carrier.period_vec_xy, seg.normal_xy))
        u_Zero = seg.corridor_u.astype(np.float32)
        both_in_domain = ok_m & ok_p & (u_Minus >= -cfg.max_shift_px - 0.5) & (u_Plus <= cfg.max_shift_px + 0.5)

        def _weighted_corr(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 4:
                return 0.0
            a0 = a - float(a.mean())
            b0 = b - float(b.mean())
            den = math.sqrt(float(np.dot(a0, a0) * np.dot(b0, b0))) + 1e-6
            return float(np.dot(a0, b0)) / den

        def _eval_E(d: int) -> float | None:
            # Boundary after shift d sits between layer u <= d and u > d
            thr = float(d) if d >= 0 else float(d - 1)
            in_side = both_in_domain & (u_Zero <= thr) & (u_Minus <= thr) & (u_Plus <= thr)
            out_side = both_in_domain & (u_Zero > thr) & (u_Minus > thr) & (u_Plus > thr)
            if int(in_side.sum()) < 4 or int(out_side.sum()) < 4:
                return None
            r_in = _weighted_corr(r_minus[in_side], r_plus[in_side]) - 0.5 * (
                _weighted_corr(r_minus[in_side], r_zero[in_side]) + _weighted_corr(r_zero[in_side], r_plus[in_side])
            )
            r_out = _weighted_corr(r_minus[out_side], r_plus[out_side]) - 0.5 * (
                _weighted_corr(r_minus[out_side], r_zero[out_side]) + _weighted_corr(r_zero[out_side], r_plus[out_side])
            )
            return float(r_in - r_out)

        e0 = _eval_E(0)
        if e0 is None:
            return {"evaluable": False, "reason": "bilateral_corridor_too_narrow", "gains_by_d": {0: 0.0}, "evidence_v": None}
        for d in admissible_ds:
            if d == 0:
                continue
            ed = _eval_E(d)
            if ed is not None:
                gains_by_d[d] = float(ed - e0)

    elif variant == "E_FreqAmp_NoPhase":
        assert carrier is not None
        amp = carrier.amplitude
        inv_2sig2 = 1.0 / (2.0 * carrier.sigma2)
        base_theta = (v_xy - carrier.origin_xy[None, :]) @ carrier.omega_xy  # [N_v]
        phis = np.linspace(0.0, 2.0 * math.pi, cfg.num_phase_grid_E, endpoint=False, dtype=np.float32)
        # e_phi: [N_v, N_phi]
        r_hat_phi = amp * np.cos(base_theta[:, None] + phis[None, :])
        e_phi = ((r_v[:, None] ** 2 - (r_v[:, None] - r_hat_phi) ** 2) * inv_2sig2).astype(np.float64)

        b_len = max(1, cfg.block_arc_len_px_E)
        col_blk = seg.corridor_col // b_len
        u_blk = np.where(seg.corridor_u < 0, (-1 - seg.corridor_u) // b_len, 100 + (seg.corridor_u - 1) // b_len)
        block_ids = col_blk * 1000 + u_blk
        unique_blocks, inv_b = np.unique(block_ids, return_inverse=True)
        n_b = len(unique_blocks)
        n_v = len(r_v)
        log_n_phi = math.log(float(cfg.num_phase_grid_E))

        # Vectorized block indicator matrix [n_b, n_v] for instant logsumexp across all d
        blk_ind = np.zeros((n_b, n_v), dtype=np.float64)
        blk_ind[inv_b, np.arange(n_v)] = 1.0

        def _l_amp_vec(in_mask_v: np.ndarray) -> float:
            active = blk_ind * in_mask_v[None, :].astype(np.float64)
            has_px = active.sum(axis=1) > 0.5
            if not np.any(has_px):
                return 0.0
            s_phi = active[has_px] @ e_phi  # [n_active_b, n_phi]
            m_val = s_phi.max(axis=1, keepdims=True)
            lse = m_val[:, 0] + np.log(np.exp(s_phi - m_val).sum(axis=1)) - log_n_phi
            return float(lse.sum())

        base_m = seg.corridor_in_m0.copy()
        l0 = _l_amp_vec(base_m)
        for d in admissible_ds:
            if d == 0:
                continue
            cand_m = base_m.copy()
            if d > 0:
                cand_m[seg.add_indices_by_d[d]] = True
            else:
                cand_m[seg.rem_indices_by_d[d]] = False
            gains_by_d[d] = (_l_amp_vec(cand_m) - l0) / l_s
        evidence_v = e_phi.mean(axis=1).astype(np.float32)

    elif variant in ("F_FullPhase", "G_PhaseCounterfactual", "H_NoSemGuard"):
        assert carrier is not None
        inv_2sig2 = 1.0 / (2.0 * carrier.sigma2)
        base_theta = (v_xy - carrier.origin_xy[None, :]) @ carrier.omega_xy
        if variant == "G_PhaseCounterfactual":
            r_hat = carrier.amplitude * np.cos(base_theta + carrier.phase_rad + gamma_offset_rad)
        else:
            r_hat = carrier.a_cos * np.cos(base_theta) + carrier.b_sin * np.sin(base_theta)
        e_v = ((r_v * r_v - (r_v - r_hat) ** 2) * inv_2sig2).astype(np.float32)
        for d in admissible_ds:
            if d == 0:
                continue
            add_idx = seg.add_indices_by_d[d]
            rem_idx = seg.rem_indices_by_d[d]
            g_val = (float(e_v[add_idx].sum()) if len(add_idx) else 0.0) - (float(e_v[rem_idx].sum()) if len(rem_idx) else 0.0)
            gains_by_d[d] = g_val / l_s
        evidence_v = e_v

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return {
        "evaluable": True,
        "reason": "none",
        "gains_by_d": gains_by_d,
        "evidence_v": evidence_v,
    }


@dataclass
class SupportCalibrationBundle:
    k_shots: int
    allow_outward: bool
    p_floor: float | None
    p_floor_fallback: bool
    period_range_px: tuple[float, float]
    holdout_r2_floor: float
    tau_by_variant: dict[str, float]
    null_block_counts: dict[str, int]
    tau_fallback_by_variant: dict[str, bool]


def precompute_image_signals(
    rgb: np.ndarray,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
) -> dict[str, np.ndarray]:
    """Precompute image-only deterministic signals (luma, trend, residual, grad_mag, local_var)."""
    luma, trend, residual = compute_luma_and_detrended_residual(rgb, cfg)
    grad_mag = compute_scharr_gradient_magnitude(luma)
    local_var = uniform_filter(residual * residual, size=5).astype(np.float32)
    return {
        "luma": luma,
        "trend": trend,
        "residual": residual,
        "grad_mag": grad_mag,
        "local_var": local_var,
    }


def precompute_support_image_null_item(
    rgb: np.ndarray,
    mask_gt: np.ndarray,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
    *,
    signals: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Precompute GT carrier wavelengths and pure-block null-movement gains for one support image."""
    gt = np.asarray(mask_gt > 0, dtype=bool)
    sig = signals if signals is not None else precompute_image_signals(rgb, cfg)
    luma = sig["luma"]
    res = sig["residual"]
    grad_m = sig["grad_mag"]
    var_m = sig["local_var"]
    p_dummy = np.ones_like(luma, dtype=np.float32)

    sup_wavelengths: list[float] = []
    sup_r2s: list[float] = []
    variants = (
        "B_GradAlign",
        "C_TexVar",
        "D_BilateralTerm",
        "E_FreqAmp_NoPhase",
        "F_FullPhase",
        "G_PhaseCounterfactual",
        "H_NoSemGuard",
    )
    null_gains: dict[str, list[float]] = {v: [] for v in variants}

    if gt.any():
        gt_lbl, n_gt = nd_label(gt, structure=BOX_3)
        empty_corr = np.zeros_like(gt, dtype=bool)
        for cid in range(1, n_gt + 1):
            comp = gt_lbl == cid
            c_info = estimate_and_freeze_component_carrier(cid, comp, empty_corr, res, cfg=cfg)
            if c_info.valid:
                sup_wavelengths.append(c_info.wavelength_px)
                sup_r2s.append(c_info.holdout_r2)

    bsize = cfg.support_block_size_px
    margin = cfg.support_boundary_margin_px
    disk_m = DISK_8 if margin == 8 else _disk_structure(margin)
    pure_fg = binary_erosion(gt, structure=disk_m, border_value=0) if gt.any() else np.zeros_like(gt, dtype=bool)
    pure_bg = binary_erosion(~gt, structure=disk_m, border_value=0)

    # Prebuild the template 40x40 half-plane geometry once
    pseudo_m0 = np.zeros((bsize, bsize), dtype=bool)
    pseudo_m0[:, : (bsize // 2 + 4)] = True
    _, _, sub_corrs, _ = build_component_geometry_and_corridors(pseudo_m0, cfg)
    segs = sub_corrs.get(1, [])
    center_segs = [
        s
        for s in segs
        if np.all((s.boundary_xy[:, 0] >= 12) & (s.boundary_xy[:, 0] <= bsize - 12))
        and np.all((s.boundary_xy[:, 1] >= 8) & (s.boundary_xy[:, 1] <= bsize - 8))
    ]
    if center_segs:
        seg = center_segs[0]
        corr_u = np.zeros((bsize, bsize), dtype=bool)
        corr_u[seg.corridor_y, seg.corridor_x] = True

        h, w = gt.shape
        for y0 in range(0, h - bsize + 1, bsize):
            for x0 in range(0, w - bsize + 1, bsize):
                blk_fg = pure_fg[y0 : y0 + bsize, x0 : x0 + bsize]
                blk_bg = pure_bg[y0 : y0 + bsize, x0 : x0 + bsize]
                if not (bool(blk_fg.all()) or bool(blk_bg.all())):
                    continue
                sub_res = res[y0 : y0 + bsize, x0 : x0 + bsize]
                sub_grad = grad_m[y0 : y0 + bsize, x0 : x0 + bsize]
                sub_var = var_m[y0 : y0 + bsize, x0 : x0 + bsize]
                sub_p = p_dummy[y0 : y0 + bsize, x0 : x0 + bsize]
                sub_carrier = estimate_and_freeze_component_carrier(1, pseudo_m0, corr_u, sub_res, cfg=cfg)
                for v_name in variants:
                    sc = score_segment_candidates(
                        seg,
                        v_name,
                        residual=sub_res,
                        grad_mag=sub_grad,
                        local_var_map=sub_var,
                        prob_512=sub_p,
                        carrier=sub_carrier,
                        p_floor=0.0,
                        allow_outward=True,
                        gamma_offset_rad=math.pi if v_name == "G_PhaseCounterfactual" else 0.0,
                        cfg=cfg,
                    )
                    if sc["evaluable"]:
                        max_g = max(float(g) for d, g in sc["gains_by_d"].items() if d != 0) if len(sc["gains_by_d"]) > 1 else 0.0
                        null_gains[v_name].append(max(0.0, max_g))

    inner_band = (gt & (~binary_erosion(gt, structure=DISK_8, border_value=0))) if gt.any() else np.zeros_like(gt, dtype=bool)
    if gt.any() and not inner_band.any():
        inner_band = gt

    return {
        "wavelengths": sup_wavelengths,
        "r2s": sup_r2s,
        "null_gains": null_gains,
        "inner_band": inner_band,
    }


def calibrate_support_thresholds(
    support_ids: list[str],
    rgb_by_id: dict[str, np.ndarray],
    mask_by_id: dict[str, np.ndarray],
    loo_probs_512_by_id: dict[str, np.ndarray] | None,
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
    *,
    precomputed_support_items: dict[str, dict[str, Any]] | None = None,
) -> SupportCalibrationBundle:
    """Calibrate support-set period range, LOO semantic floor p_floor, and null-movement quantile tau^(m)."""
    k = len(support_ids)

    # 1. Outward semantic floor p_floor from genuine leave-one-support-image-out predictions
    if k == 1 and cfg.k1_forbid_outward:
        allow_outward = False
        p_floor = None
        p_floor_fallback = True
    elif loo_probs_512_by_id is not None and k >= 2:
        allow_outward = True
        missed_vals = []
        edge_vals = []
        for sid in support_ids:
            if sid not in loo_probs_512_by_id:
                continue
            p_loo = loo_probs_512_by_id[sid]
            if precomputed_support_items is not None and sid in precomputed_support_items:
                inner_band = precomputed_support_items[sid]["inner_band"]
            else:
                gt = np.asarray(mask_by_id[sid] > 0, dtype=bool)
                if not gt.any():
                    continue
                inner_band = gt & (~binary_erosion(gt, structure=DISK_8, border_value=0))
                if not inner_band.any():
                    inner_band = gt
            if not inner_band.any():
                continue
            missed = inner_band & (p_loo < cfg.coarse_threshold)
            if int(missed.sum()) >= 16:
                missed_vals.append(p_loo[missed])
            edge_vals.append(p_loo[inner_band])
        pool = missed_vals if missed_vals else edge_vals
        if pool:
            cat = np.concatenate(pool)
            q_val = float(np.quantile(cat, cfg.p_floor_quantile))
            p_floor = float(np.clip(q_val, cfg.p_floor_min, cfg.p_floor_max))
            p_floor_fallback = False
        else:
            p_floor = 0.35
            p_floor_fallback = True
    else:
        allow_outward = False
        p_floor = None
        p_floor_fallback = True

    # 2. Support-set period library and null-movement spatial blocks
    sup_wavelengths: list[float] = []
    sup_r2s: list[float] = []
    variants = (
        "B_GradAlign",
        "C_TexVar",
        "D_BilateralTerm",
        "E_FreqAmp_NoPhase",
        "F_FullPhase",
        "G_PhaseCounterfactual",
        "H_NoSemGuard",
    )
    null_gains: dict[str, list[float]] = {v: [] for v in variants}

    for sid in support_ids:
        if precomputed_support_items is not None and sid in precomputed_support_items:
            item = precomputed_support_items[sid]
        else:
            item = precompute_support_image_null_item(rgb_by_id[sid], mask_by_id[sid], cfg)
        sup_wavelengths.extend(item["wavelengths"])
        sup_r2s.extend(item["r2s"])
        for v_name in variants:
            null_gains[v_name].extend(item["null_gains"][v_name])

    if sup_wavelengths:
        q_lo = float(np.quantile(sup_wavelengths, 0.10)) * 0.80
        q_hi = float(np.quantile(sup_wavelengths, 0.90)) * 1.25
        period_range = (
            max(cfg.hard_sampling_period_min_px, min(cfg.default_period_min_px, q_lo)),
            min(cfg.hard_sampling_period_max_px, max(cfg.default_period_max_px, q_hi)),
        )
    else:
        period_range = (cfg.default_period_min_px, cfg.default_period_max_px)

    holdout_floor = cfg.min_holdout_r2_floor
    if sup_r2s:
        holdout_floor = max(cfg.min_holdout_r2_floor, min(0.12, float(np.quantile(sup_r2s, 0.15)) * 0.5))

    tau_by_variant: dict[str, float] = {}
    null_counts: dict[str, int] = {}
    tau_fallback: dict[str, bool] = {}

    q_level = 1.0 - cfg.null_quantile_alpha
    for v_name in variants:
        vals = null_gains[v_name]
        null_counts[v_name] = len(vals)
        if len(vals) >= cfg.min_support_null_blocks:
            q_tau = float(np.quantile(vals, q_level))
            # Enforce at least 50% of theoretical noise floor so zero-texture background cannot yield tau=0
            tau_by_variant[v_name] = max(q_tau, 0.5 * cfg.fallback_gain_per_px[v_name])
            tau_fallback[v_name] = False
        else:
            tau_by_variant[v_name] = float(cfg.fallback_gain_per_px[v_name])
            tau_fallback[v_name] = True

    return SupportCalibrationBundle(
        k_shots=k,
        allow_outward=allow_outward,
        p_floor=p_floor,
        p_floor_fallback=p_floor_fallback,
        period_range_px=period_range,
        holdout_r2_floor=holdout_floor,
        tau_by_variant=tau_by_variant,
        null_block_counts=null_counts,
        tau_fallback_by_variant=tau_fallback,
    )


def refine_query_image_all_variants(
    rgb: np.ndarray,
    prob_512: np.ndarray,
    calib: SupportCalibrationBundle,
    *,
    variants: tuple[str, ...] = (
        "A_M3_Pruned",
        "B_GradAlign",
        "C_TexVar",
        "D_BilateralTerm",
        "E_FreqAmp_NoPhase",
        "F_FullPhase",
        "G_pi_2",
        "G_pi",
        "G_3pi_2",
        "G_FixedTauF_pi_2",
        "G_FixedTauF_pi",
        "G_FixedTauF_3pi_2",
        "H_NoSemGuard",
    ),
    cfg: PhaseBoundaryConfig = PhaseBoundaryConfig(),
    precomputed_signals: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run shared geometry + carrier estimation once per query image, then evaluate all variants."""
    m0 = np.asarray(prob_512 >= cfg.coarse_threshold, dtype=bool)
    if not m0.any():
        empty_res = {
            v: {
                "mask": m0.copy(),
                "moved_segments": 0,
                "moved_arc_len_px": 0,
                "inward_arc_len_px": 0,
                "outward_arc_len_px": 0,
                "net_area_delta_px": 0,
                "changed_area_px": 0,
                "mean_abs_d_moved": 0.0,
                "segment_records": {},
            }
            for v in variants
        }
        return {
            "m0": m0,
            "total_arc_len_px": 0,
            "carrier_valid_arc_len_px": 0,
            "small_target_arc_len_px": 0,
            "small_target_carrier_valid_arc_len_px": 0,
            "component_fallbacks": {"no_foreground": 1},
            "carriers": {},
            "corridors": {},
            "variants": empty_res,
        }

    if precomputed_signals is not None:
        luma = precomputed_signals["luma"]
        trend = precomputed_signals["trend"]
        residual = precomputed_signals["residual"]
        grad_mag = precomputed_signals["grad_mag"]
        local_var = precomputed_signals["local_var"]
    else:
        luma, trend, residual = compute_luma_and_detrended_residual(rgb, cfg)
        grad_mag = compute_scharr_gradient_magnitude(luma)
        local_var = uniform_filter(residual * residual, size=5).astype(np.float32)

    comp_labels, num_fg, corridors_by_comp, geom_stats = build_component_geometry_and_corridors(m0, cfg)
    _, n_bg_init = nd_label(~m0, structure=CROSS_3)

    carriers: dict[int, ComponentCarrier] = {}
    comp_fallbacks: dict[str, int] = {}
    carrier_valid_arc = 0
    small_arc = 0
    small_valid_arc = 0

    for cid in range(1, num_fg + 1):
        comp_m = comp_labels == cid
        segs = corridors_by_comp.get(cid, [])
        c_arc = sum(s.arc_len for s in segs)
        is_small_comp = int(comp_m.sum()) < cfg.small_target_area_thr_px
        if is_small_comp:
            small_arc += c_arc

        if not segs:
            reason = "component_too_small" if int(comp_m.sum()) < cfg.min_component_area_px else "no_valid_corridor"
            carriers[cid] = ComponentCarrier(comp_id=cid, valid=False, fallback_reason=reason, area_px=int(comp_m.sum()))
            comp_fallbacks[reason] = comp_fallbacks.get(reason, 0) + 1
            continue

        corr_union = np.zeros_like(m0, dtype=bool)
        for s in segs:
            corr_union[s.corridor_y, s.corridor_x] = True

        c_info = estimate_and_freeze_component_carrier(
            cid,
            comp_m,
            corr_union,
            residual,
            period_range_px=calib.period_range_px,
            holdout_r2_floor=calib.holdout_r2_floor,
            cfg=cfg,
        )
        carriers[cid] = c_info
        if c_info.valid:
            carrier_valid_arc += c_arc
            if is_small_comp:
                small_valid_arc += c_arc
        else:
            comp_fallbacks[c_info.fallback_reason] = comp_fallbacks.get(c_info.fallback_reason, 0) + 1

    all_segs: list[ContourSegmentCorridor] = []
    for cid in sorted(corridors_by_comp.keys()):
        all_segs.extend(corridors_by_comp[cid])

    variant_outputs: dict[str, Any] = {}
    gamma_map = {
        "G_pi_2": 0.5 * math.pi,
        "G_pi": math.pi,
        "G_3pi_2": 1.5 * math.pi,
        "G_FixedTauF_pi_2": 0.5 * math.pi,
        "G_FixedTauF_pi": math.pi,
        "G_FixedTauF_3pi_2": 1.5 * math.pi,
    }
    score_cache: dict[tuple[int, str, float], dict[str, Any]] = {}

    for v_name in variants:
        if v_name == "A_M3_Pruned":
            variant_outputs[v_name] = {
                "mask": m0.copy(),
                "moved_segments": 0,
                "moved_arc_len_px": 0,
                "inward_arc_len_px": 0,
                "outward_arc_len_px": 0,
                "net_area_delta_px": 0,
                "changed_area_px": 0,
                "mean_abs_d_moved": 0.0,
                "segment_records": {},
            }
            continue

        if v_name.startswith("G_FixedTauF_"):
            base_v = "G_PhaseCounterfactual"
            tau = calib.tau_by_variant["F_FullPhase"]
        elif v_name.startswith("G_"):
            base_v = "G_PhaseCounterfactual"
            tau = calib.tau_by_variant["G_PhaseCounterfactual"]
        else:
            base_v = v_name
            tau = calib.tau_by_variant[v_name]

        gamma = gamma_map.get(v_name, 0.0)
        seg_records: dict[int, dict[str, Any]] = {}
        proposals: list[tuple[int, float, int, ContourSegmentCorridor]] = []

        for seg in all_segs:
            c_info = carriers.get(seg.comp_id)
            cache_key = (seg.seg_id, base_v, gamma)
            sc = score_cache.get(cache_key)
            if sc is None:
                sc = score_segment_candidates(
                    seg,
                    base_v,
                    residual=residual,
                    grad_mag=grad_mag,
                    local_var_map=local_var,
                    prob_512=prob_512,
                    carrier=c_info,
                    p_floor=calib.p_floor,
                    allow_outward=calib.allow_outward,
                    gamma_offset_rad=gamma,
                    cfg=cfg,
                )
                score_cache[cache_key] = sc
            if not sc["evaluable"]:
                seg_records[seg.seg_id] = {
                    "evaluable": False,
                    "reason": sc["reason"],
                    "d_star": 0,
                    "gain_star": 0.0,
                    "margin": -tau,
                    "arc_len": seg.arc_len,
                    "gains_by_d": {0: 0.0},
                    "applied_d": 0,
                }
                continue

            d_star, g_star = select_best_shift_with_tiebreak(sc["gains_by_d"], tau)
            best_nonzero_d, best_nonzero_g = select_best_shift_with_tiebreak(sc["gains_by_d"], -1e9)
            margin = float(best_nonzero_g - tau)
            seg_records[seg.seg_id] = {
                "evaluable": True,
                "reason": "moved" if d_star != 0 else "below_tau_threshold",
                "d_star": int(d_star),
                "gain_star": float(g_star),
                "best_unthresholded_d": int(best_nonzero_d),
                "best_unthresholded_gain": float(best_nonzero_g),
                "margin": margin,
                "arc_len": seg.arc_len,
                "gains_by_d": sc["gains_by_d"],
                "applied_d": 0,
            }
            if d_star != 0:
                proposals.append((seg.comp_id, -margin, seg.seg_id, seg))

        # Apply proposals in fixed deterministic order: component ID asc, gain margin desc, seg_id asc
        proposals.sort(key=lambda item: (item[0], item[1], item[2]))
        m_cur = m0.copy()
        moved_segs = 0
        moved_arc = 0
        inward_arc = 0
        outward_arc = 0
        abs_ds: list[int] = []

        # Fast-path: try applying all proposals of each component in batch; if topology holds (disjoint Voronoi corridors
        # with opposite-shore and inter-component clearance), accept all; otherwise resolve sequentially segment-by-segment.
        props_by_comp: dict[int, list[tuple[int, float, int, ContourSegmentCorridor]]] = defaultdict(list)
        for p_item in proposals:
            props_by_comp[p_item[0]].append(p_item)

        for cid_p, c_props in props_by_comp.items():
            cy0 = min(p[3].local_bbox[0] for p in c_props)
            cy1 = max(p[3].local_bbox[1] for p in c_props)
            cx0 = min(p[3].local_bbox[2] for p in c_props)
            cx1 = max(p[3].local_bbox[3] for p in c_props)
            sub_cur_c = m_cur[cy0:cy1, cx0:cx1]
            _, fg_old_c = nd_label(sub_cur_c, structure=BOX_3)
            _, bg_old_c = nd_label(~sub_cur_c, structure=CROSS_3)
            sub_batch = sub_cur_c.copy()
            for _, _, sid_i, seg in c_props:
                d_val = int(seg_records[sid_i]["d_star"])
                if d_val > 0:
                    add_idx = seg.add_indices_by_d[d_val]
                    sub_batch[seg.corridor_y[add_idx] - cy0, seg.corridor_x[add_idx] - cx0] = True
                else:
                    rem_idx = seg.rem_indices_by_d[d_val]
                    sub_batch[seg.corridor_y[rem_idx] - cy0, seg.corridor_x[rem_idx] - cx0] = False

            if nd_label(sub_batch, structure=BOX_3)[1] == fg_old_c and nd_label(~sub_batch, structure=CROSS_3)[1] == bg_old_c:
                m_cur[cy0:cy1, cx0:cx1] = sub_batch
                for _, _, sid_i, seg in c_props:
                    d_val = int(seg_records[sid_i]["d_star"])
                    seg_records[sid_i]["applied_d"] = d_val
                    moved_segs += 1
                    moved_arc += seg.arc_len
                    if d_val < 0:
                        inward_arc += seg.arc_len
                    else:
                        outward_arc += seg.arc_len
                    abs_ds.append(abs(d_val))
            else:
                for _, _, sid_i, seg in c_props:
                    d_val = int(seg_records[sid_i]["d_star"])
                    by0, by1, bx0, bx1 = seg.local_bbox
                    sub_cur = m_cur[by0:by1, bx0:bx1]
                    _, fg_old_b = nd_label(sub_cur, structure=BOX_3)
                    _, bg_old_b = nd_label(~sub_cur, structure=CROSS_3)
                    sub_try = sub_cur.copy()
                    if d_val > 0:
                        add_idx = seg.add_indices_by_d[d_val]
                        sub_try[seg.corridor_y[add_idx] - by0, seg.corridor_x[add_idx] - bx0] = True
                    else:
                        rem_idx = seg.rem_indices_by_d[d_val]
                        sub_try[seg.corridor_y[rem_idx] - by0, seg.corridor_x[rem_idx] - bx0] = False
                    if nd_label(sub_try, structure=BOX_3)[1] == fg_old_b and nd_label(~sub_try, structure=CROSS_3)[1] == bg_old_b:
                        m_cur[by0:by1, bx0:bx1] = sub_try
                        seg_records[sid_i]["applied_d"] = d_val
                        moved_segs += 1
                        moved_arc += seg.arc_len
                        if d_val < 0:
                            inward_arc += seg.arc_len
                        else:
                            outward_arc += seg.arc_len
                        abs_ds.append(abs(d_val))
                    else:
                        seg_records[sid_i]["reason"] = "multi_segment_topology_conflict"

        # Final global topology check safety net
        if moved_segs > 0 and not check_topology_preserved(m0, m_cur, n_fg_old=num_fg, n_bg_old=n_bg_init):
            m_cur = m0.copy()
            moved_segs = 0
            moved_arc = 0
            inward_arc = 0
            outward_arc = 0
            abs_ds = []
            for r in seg_records.values():
                if r.get("applied_d", 0) != 0:
                    r["applied_d"] = 0
                    r["reason"] = "global_topology_safety_revert"

        variant_outputs[v_name] = {
            "mask": m_cur,
            "moved_segments": moved_segs,
            "moved_arc_len_px": moved_arc,
            "inward_arc_len_px": inward_arc,
            "outward_arc_len_px": outward_arc,
            "net_area_delta_px": int(m_cur.sum()) - int(m0.sum()),
            "changed_area_px": int(np.logical_xor(m_cur, m0).sum()),
            "mean_abs_d_moved": float(np.mean(abs_ds)) if abs_ds else 0.0,
            "segment_records": seg_records,
        }

    return {
        "m0": m0,
        "luma": luma,
        "trend": trend,
        "residual": residual,
        "total_arc_len_px": int(geom_stats["total_arc_len_px"]),
        "carrier_valid_arc_len_px": int(carrier_valid_arc),
        "small_target_arc_len_px": int(small_arc),
        "small_target_carrier_valid_arc_len_px": int(small_valid_arc),
        "component_fallbacks": comp_fallbacks,
        "carriers": carriers,
        "corridors": {s.seg_id: s for s in all_segs},
        "variants": variant_outputs,
    }
