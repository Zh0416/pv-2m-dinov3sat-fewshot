from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label as nd_label

from ..metrics import BOX_3

CROSS_3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _compute_luma(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [3, H, W] or [H, W, 3] uint8/float image to [H, W] float32 luminance in [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() > 1.5:
            arr = arr / 255.0
        return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    return (arr / 255.0 if arr.max() > 1.5 else arr.copy()).astype(np.float32)


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
    """Verify foreground (8-conn) and background/hole (4-conn) component counts are unchanged."""
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        sub_old = m_old[y0:y1, x0:x1]
        sub_new = m_new[y0:y1, x0:x1]
        if nd_label(sub_old, structure=BOX_3)[1] != nd_label(sub_new, structure=BOX_3)[1]:
            return False
        return nd_label(~sub_old, structure=CROSS_3)[1] == nd_label(~sub_new, structure=CROSS_3)[1]

    if n_fg_old is None:
        _, n_fg_old = nd_label(m_old, structure=BOX_3)
    if n_bg_old is None:
        _, n_bg_old = nd_label(~m_old, structure=CROSS_3)
    if nd_label(m_new, structure=BOX_3)[1] != n_fg_old:
        return False
    return nd_label(~m_new, structure=CROSS_3)[1] == n_bg_old


def _order_boundary_cycles(boundary_mask: np.ndarray) -> list[np.ndarray]:
    """Trace ordered 8-connected boundary pixel loops as arrays of shape [L, 2] in (x, y) order."""
    ys, xs = np.nonzero(boundary_mask)
    if len(xs) == 0:
        return []
    coords = np.stack([xs, ys], axis=1).astype(np.int32)
    coord_map: dict[tuple[int, int], int] = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
    visited = np.zeros(len(coords), dtype=bool)
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


@dataclass(frozen=True)
class DeformableBoundaryConfig:
    enabled: bool = False
    mode: str = "off"  # "off" | "deformable_grad" | "deformable_gt_oracle_offline"
    max_shift_r: int = 8
    candidate_shifts: tuple[int, ...] = (-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8)
    control_point_spacing_px: int = 8
    high_curvature_angle_deg: float = 35.0
    high_curvature_spacing_px: int = 6
    corner_lock_angle_deg: float = 65.0
    max_adjacent_slope_diff: int = 4
    min_component_area_px: int = 64
    min_opposite_shore_clearance_px: float = 2.0
    min_inter_component_clearance_px: float = 2.0
    fallback_tau_grad_per_px: float = 0.035

    @classmethod
    def from_json(cls, path: Path | str, *, mode: str = "deformable_grad") -> "DeformableBoundaryConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        g = raw.get("deformable_geometry", {})
        return cls(
            enabled=(mode != "off"),
            mode=mode,
            max_shift_r=int(g.get("max_shift_r_px", 8)),
            candidate_shifts=tuple(int(x) for x in g.get("candidate_shifts", [-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8])),
            control_point_spacing_px=int(g.get("control_point_spacing_px", 8)),
            high_curvature_angle_deg=float(g.get("high_curvature_refine_angle_deg", 35.0)),
            high_curvature_spacing_px=int(g.get("high_curvature_spacing_px", 6)),
            max_adjacent_slope_diff=int(g.get("max_adjacent_slope_diff_px", 4)),
            min_opposite_shore_clearance_px=float(g.get("min_opposite_shore_clearance_px", 2.0)),
            min_inter_component_clearance_px=float(g.get("min_inter_component_clearance_px", 2.0)),
        )


@dataclass
class DeformableArc:
    arc_id: int
    start_cp: int
    end_cp: int
    pts_xy: np.ndarray  # [L_i, 2] float32 original contour points
    normals_xy: np.ndarray  # [L_i, 2] float32 unit outward normals (endpoint normals match shared cp normals)
    u_params: np.ndarray  # [L_i] float32 in [0, 1]
    corridor_y: np.ndarray  # [N_v] int32 disjoint Voronoi pixels assigned to this arc
    corridor_x: np.ndarray  # [N_v] int32
    corridor_u_param: np.ndarray  # [N_v] float32 in [0, 1] parameter of nearest boundary pt
    corridor_signed_dist: np.ndarray  # [N_v] float32 signed normal offset (<=0 inside M0, >0 outside M0)
    corridor_in_m0: np.ndarray  # [N_v] bool
    max_inward_safe: float  # opposite-shore safe inward limit (positive scalar <= max_shift_r)
    max_outward_safe: float  # inter-component / border safe outward limit (<= max_shift_r)


@dataclass
class DeformableLoop:
    comp_id: int
    loop_id: int
    is_closed: bool
    n_cps: int
    cp_locked_zero: np.ndarray  # [n_cps] bool (True for sharp corners lacking a single reliable normal)
    arcs: list[DeformableArc]
    local_bbox: tuple[int, int, int, int]
    perimeter_px: int


def _extract_loop_normals(cycle_xy: np.ndarray, signed_dt: np.ndarray) -> np.ndarray:
    """Compute smoothed unit outward normals along an ordered boundary pixel chain."""
    h, w = signed_dt.shape
    n_pts = len(cycle_xy)
    normals = np.zeros((n_pts, 2), dtype=np.float32)
    for i, (x, y) in enumerate(cycle_xy):
        x0, x1 = max(0, int(x) - 2), min(w - 1, int(x) + 2)
        y0, y1 = max(0, int(y) - 2), min(h - 1, int(y) + 2)
        gx = float(signed_dt[y, x1] - signed_dt[y, x0])
        gy = float(signed_dt[y1, x] - signed_dt[y0, x])
        nv = math.hypot(gx, gy)
        if nv > 1e-6:
            normals[i, 0] = gx / nv
            normals[i, 1] = gy / nv
        else:
            normals[i, 0] = 1.0
            normals[i, 1] = 0.0
    if n_pts >= 5:
        sm = normals.copy()
        for i in range(n_pts):
            v = normals[(i - 1) % n_pts] + normals[i] + normals[(i + 1) % n_pts]
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                sm[i] = v / nv
        return sm
    return normals


def _place_control_points(
    cycle_xy: np.ndarray,
    normals: np.ndarray,
    cfg: DeformableBoundaryConfig,
) -> tuple[list[int], np.ndarray]:
    """Place control point indices along closed cycle, refining at high curvature and locking sharp corners."""
    n_pts = len(cycle_xy)
    if n_pts < 12:
        return [], np.zeros(0, dtype=bool)

    cos_high = math.cos(math.radians(cfg.high_curvature_angle_deg))
    cos_lock = math.cos(math.radians(cfg.corner_lock_angle_deg))

    cp_indices = [0]
    last_idx = 0
    for i in range(1, n_pts):
        dist = i - last_idx
        n_prev = normals[(i - 2) % n_pts]
        n_next = normals[(i + 2) % n_pts]
        dot = float(np.dot(n_prev, n_next))
        step = cfg.high_curvature_spacing_px if dot < cos_high else cfg.control_point_spacing_px
        if dist >= step and (n_pts - i) >= cfg.high_curvature_spacing_px:
            cp_indices.append(i)
            last_idx = i

    if len(cp_indices) < 3:
        return [], np.zeros(0, dtype=bool)

    locked = np.zeros(len(cp_indices), dtype=bool)
    for k_i, idx in enumerate(cp_indices):
        n_prev = normals[(idx - 2) % n_pts]
        n_next = normals[(idx + 2) % n_pts]
        # Lock sharp corners lacking a single reliable normal to 0 (prevents vertex splitting or folding)
        if float(np.dot(n_prev, n_next)) < cos_lock:
            locked[k_i] = True
    return cp_indices, locked


def build_deformable_loops(
    m0: np.ndarray,
    cfg: DeformableBoundaryConfig = DeformableBoundaryConfig(),
) -> tuple[np.ndarray, int, list[DeformableLoop]]:
    """Build deformable multi-control-point contour loops with strictly disjoint Voronoi arc corridors."""
    m0_b = np.asarray(m0, dtype=bool)
    h, w = m0_b.shape
    comp_labels, num_fg = nd_label(m0_b, structure=BOX_3)
    loops: list[DeformableLoop] = []
    if num_fg == 0:
        return comp_labels, 0, loops

    max_r = cfg.max_shift_r
    loop_counter = 0

    for cid in range(1, num_fg + 1):
        comp = comp_labels == cid
        if int(comp.sum()) < cfg.min_component_area_px:
            continue

        c_ys, c_xs = np.nonzero(comp)
        pad_c = max_r + 6
        cy0 = max(0, int(c_ys.min()) - pad_c)
        cy1 = min(h, int(c_ys.max()) + pad_c + 1)
        cx0 = max(0, int(c_xs.min()) - pad_c)
        cx1 = min(w, int(c_xs.max()) + pad_c + 1)

        sub_comp = comp[cy0:cy1, cx0:cx1]
        sub_m0 = m0_b[cy0:cy1, cx0:cx1]
        b_mask_sub = sub_comp & (~binary_erosion(sub_comp, structure=CROSS_3, border_value=0))
        if not b_mask_sub.any():
            continue

        dt_in_sub = distance_transform_edt(sub_comp).astype(np.float32)
        dt_out_sub = distance_transform_edt(~sub_comp).astype(np.float32)
        signed_dt_sub = dt_out_sub - dt_in_sub

        other_fg_sub = sub_m0 & (~sub_comp)
        dt_other_sub = distance_transform_edt(~other_fg_sub).astype(np.float32) if other_fg_sub.any() else np.full(sub_comp.shape, 999.0, dtype=np.float32)

        cycles_sub = _order_boundary_cycles(b_mask_sub)
        sh, sw = sub_comp.shape

        # Assign every boundary pixel in valid cycles to (loop_idx, arc_idx, u_param)
        for cyc_sub in cycles_sub:
            if len(cyc_sub) < 16:
                continue
            # Check whether cycle is truly closed (end 8-adjacent to start)
            dx_end = abs(int(cyc_sub[0, 0]) - int(cyc_sub[-1, 0]))
            dy_end = abs(int(cyc_sub[0, 1]) - int(cyc_sub[-1, 1]))
            is_closed = max(dx_end, dy_end) <= 2

            normals = _extract_loop_normals(cyc_sub, signed_dt_sub)
            cp_indices, cp_locked = _place_control_points(cyc_sub, normals, cfg)
            n_cps = len(cp_indices)
            if n_cps < 3:
                continue

            # Build arc index slices covering every pixel of cyc_sub
            n_arcs = n_cps if is_closed else (n_cps - 1)
            arc_slices: list[np.ndarray] = []
            for a_i in range(n_arcs):
                s_idx = cp_indices[a_i]
                e_idx = cp_indices[(a_i + 1) % n_cps]
                if e_idx > s_idx:
                    idxs = np.arange(s_idx, e_idx + 1, dtype=np.int32)
                else:
                    idxs = np.concatenate([np.arange(s_idx, len(cyc_sub), dtype=np.int32), np.arange(0, e_idx + 1, dtype=np.int32)])
                arc_slices.append(idxs)

            # Label boundary pixels of THIS loop for disjoint Voronoi corridor assignment
            b_arc_map = np.full((sh, sw), -1, dtype=np.int32)
            b_u_map = np.zeros((sh, sw), dtype=np.float32)
            b_nx_map = np.zeros((sh, sw), dtype=np.float32)
            b_ny_map = np.zeros((sh, sw), dtype=np.float32)

            for a_i, idxs in enumerate(arc_slices):
                l_a = len(idxs)
                us = np.linspace(0.0, 1.0, l_a, dtype=np.float32)
                # Shared endpoint normals: interpolate linearly between cp start/end normals and renormalize
                n_start = normals[idxs[0]]
                n_end = normals[idxs[-1]]
                for k_pt, pt_i in enumerate(idxs):
                    # Avoid overwriting start control point of arc 0 with last arc's end except deterministically
                    px, py = int(cyc_sub[pt_i, 0]), int(cyc_sub[pt_i, 1])
                    if b_arc_map[py, px] == -1:
                        b_arc_map[py, px] = a_i
                        b_u_map[py, px] = us[k_pt]
                        nv = (1.0 - us[k_pt]) * n_start + us[k_pt] * n_end
                        nn = float(np.linalg.norm(nv))
                        nv = nv / nn if nn > 1e-6 else normals[pt_i]
                        b_nx_map[py, px] = nv[0]
                        b_ny_map[py, px] = nv[1]

            active_b = b_arc_map >= 0
            if not active_b.any():
                continue

            dist_b, (near_y_s, near_x_s) = distance_transform_edt(~active_b, return_indices=True)
            ribbon = (dist_b <= (max_r + 0.75)) & (sub_comp | (~sub_m0))
            ribbon = ribbon & (sub_comp | (dt_other_sub > cfg.min_inter_component_clearance_px))

            ry_s, rx_s = np.nonzero(ribbon)
            ny_s = near_y_s[ry_s, rx_s]
            nx_s = near_x_s[ry_s, rx_s]
            r_arc = b_arc_map[ny_s, nx_s]
            r_u = b_u_map[ny_s, nx_s]
            r_nx = b_nx_map[ny_s, nx_s]
            r_ny = b_ny_map[ny_s, nx_s]
            r_in = sub_comp[ry_s, rx_s]
            r_sdist = (rx_s - nx_s).astype(np.float32) * r_nx + (ry_s - ny_s).astype(np.float32) * r_ny

            # Consistent side sign: inside M0 pixels have signed_dist <= 0, outside > 0
            # Map boundary pixels themselves (sdist==0, in_m0==True) to -0.5 so d=-1 removes them cleanly
            r_sdist = np.where(r_in, np.minimum(r_sdist - 0.5, -0.01), np.maximum(r_sdist, 0.01))

            order = np.argsort(r_arc, kind="mergesort")
            ry_s, rx_s = ry_s[order], rx_s[order]
            r_arc, r_u, r_in, r_sdist = r_arc[order], r_u[order], r_in[order], r_sdist[order]
            uniq_a, split_idx = np.unique(r_arc, return_index=True)
            slices_a = {
                int(av): slice(int(split_idx[i]), int(split_idx[i + 1]) if (i + 1) < len(split_idx) else len(r_arc))
                for i, av in enumerate(uniq_a)
            }

            arcs: list[DeformableArc] = []
            for a_i, idxs in enumerate(arc_slices):
                l_a = len(idxs)
                us = np.linspace(0.0, 1.0, l_a, dtype=np.float32)
                pts_sub = cyc_sub[idxs].astype(np.float32)
                pts_full = pts_sub.copy()
                pts_full[:, 0] += cx0
                pts_full[:, 1] += cy0

                # Interpolated unit normals guaranteeing exact match at shared endpoints
                n_s = normals[idxs[0]]
                n_e = normals[idxs[-1]]
                n_arr = (1.0 - us[:, None]) * n_s[None, :] + us[:, None] * n_e[None, :]
                n_norms = np.linalg.norm(n_arr, axis=1, keepdims=True)
                n_arr = np.where(n_norms > 1e-6, n_arr / np.maximum(n_norms, 1e-6), normals[idxs])

                sl = slices_a.get(a_i)
                if sl is None:
                    vy = np.empty(0, dtype=np.int32)
                    vx = np.empty(0, dtype=np.int32)
                    vu = np.empty(0, dtype=np.float32)
                    vsd = np.empty(0, dtype=np.float32)
                    vin = np.empty(0, dtype=bool)
                else:
                    vy = (ry_s[sl] + cy0).astype(np.int32)
                    vx = (rx_s[sl] + cx0).astype(np.int32)
                    vu = r_u[sl].astype(np.float32)
                    vsd = r_sdist[sl].astype(np.float32)
                    vin = r_in[sl]

                # Estimate safe inward limit from medial thickness & safe outward limit from dt_other / borders
                in_pts_dt = dt_in_sub[pts_sub[:, 1].astype(np.int32), pts_sub[:, 0].astype(np.int32)]
                out_other_dt = dt_other_sub[pts_sub[:, 1].astype(np.int32), pts_sub[:, 0].astype(np.int32)]
                # Check distance to image border along outward normal
                border_clearance = np.minimum(
                    np.minimum(pts_full[:, 0], w - 1 - pts_full[:, 0]),
                    np.minimum(pts_full[:, 1], h - 1 - pts_full[:, 1]),
                )
                max_out_safe = float(min(max_r, max(0.0, float(out_other_dt.min()) - cfg.min_inter_component_clearance_px), float(border_clearance.min())))
                # Opposite-shore limit: inside corridor max inward depth cannot exceed local interior thickness
                max_in_depth = float(np.max(-vsd[vin])) if np.any(vin) else 0.0
                max_in_safe = float(min(max_r, max(0.0, max_in_depth)))

                arcs.append(
                    DeformableArc(
                        arc_id=a_i,
                        start_cp=a_i,
                        end_cp=(a_i + 1) % n_cps,
                        pts_xy=pts_full,
                        normals_xy=n_arr.astype(np.float32),
                        u_params=us,
                        corridor_y=vy,
                        corridor_x=vx,
                        corridor_u_param=vu,
                        corridor_signed_dist=vsd,
                        corridor_in_m0=vin,
                        max_inward_safe=max_in_safe,
                        max_outward_safe=max_out_safe,
                    )
                )

            loops.append(
                DeformableLoop(
                    comp_id=cid,
                    loop_id=loop_counter,
                    is_closed=is_closed,
                    n_cps=n_cps,
                    cp_locked_zero=cp_locked,
                    arcs=arcs,
                    local_bbox=(cy0, cy1, cx0, cx1),
                    perimeter_px=len(cyc_sub),
                )
            )
            loop_counter += 1

    return comp_labels, int(num_fg), loops


def _arc_changed_indices(
    arc: DeformableArc,
    a: int,
    b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint (add_idx, rem_idx) into arc's Voronoi corridor for endpoint shifts (a, b).

    Strictly returns empty arrays when (a, b) == (0, 0) (exact zero-shift preservation).
    """
    if a == 0 and b == 0 or len(arc.corridor_y) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    delta_u = (1.0 - arc.corridor_u_param) * float(a) + arc.corridor_u_param * float(b)
    add_mask = (~arc.corridor_in_m0) & (delta_u > 0.0) & (arc.corridor_signed_dist <= delta_u)
    rem_mask = arc.corridor_in_m0 & (delta_u < 0.0) & (arc.corridor_signed_dist >= delta_u)
    return np.flatnonzero(add_mask).astype(np.int32), np.flatnonzero(rem_mask).astype(np.int32)


def solve_deformable_loop_dp(
    loop: DeformableLoop,
    *,
    grad_mag: np.ndarray | None,
    prob_512: np.ndarray | None,
    gt_mask_offline: np.ndarray | None,
    allow_outward: bool,
    p_floor: float | None,
    cfg: DeformableBoundaryConfig,
) -> tuple[list[int], float, bool]:
    """Solve exact dynamic programming over shared control-point shifts d_0..d_{N-1}.

    When gt_mask_offline is supplied (Stage 0 offline geometry reference ONLY), optimizes
    exact GT pixel utility U_i(a,b); otherwise optimizes fixed-weight Scharr gradient gain
    Delta_G_i(a,b) = G_i(a,b) - G_i(0,0) with semantic outward guard.
    """
    shifts = cfg.candidate_shifts
    n_s = len(shifts)
    zero_idx = shifts.index(0)
    n_cps = loop.n_cps
    n_arcs = len(loop.arcs)
    neg_inf = -1e18

    # Precompute arc transition score tables score_tbl[arc_i, s_a, s_b]
    score_tbl = np.full((n_arcs, n_s, n_s), neg_inf, dtype=np.float64)

    for a_i, arc in enumerate(loop.arcs):
        start_locked = bool(loop.cp_locked_zero[arc.start_cp])
        end_locked = bool(loop.cp_locked_zero[arc.end_cp])

        base_g = 0.0
        if grad_mag is not None and gt_mask_offline is None:
            g0, ok0 = _bilinear_sample_2d(grad_mag, arc.pts_xy[:, 0], arc.pts_xy[:, 1])
            base_g = float(g0[ok0].sum())

        for ia, a in enumerate(shifts):
            if start_locked and a != 0:
                continue
            if a > 0 and (not allow_outward or a > arc.max_outward_safe + 0.25):
                continue
            if a < 0 and (-a > arc.max_inward_safe + 0.25):
                continue
            for ib, b in enumerate(shifts):
                if end_locked and b != 0:
                    continue
                if abs(b - a) > cfg.max_adjacent_slope_diff:
                    continue
                if b > 0 and (not allow_outward or b > arc.max_outward_safe + 0.25):
                    continue
                if b < 0 and (-b > arc.max_inward_safe + 0.25):
                    continue

                if a == 0 and b == 0:
                    score_tbl[a_i, ia, ib] = 0.0
                    continue

                add_idx, rem_idx = _arc_changed_indices(arc, a, b)

                if gt_mask_offline is not None:
                    # Offline Stage 0 GT reachable pixel utility U_i(a, b)
                    u_val = 0.0
                    if len(add_idx) > 0:
                        y_add = gt_mask_offline[arc.corridor_y[add_idx], arc.corridor_x[add_idx]]
                        u_val += float(np.sum(np.where(y_add, 1.0, -1.0)))
                    if len(rem_idx) > 0:
                        y_rem = gt_mask_offline[arc.corridor_y[rem_idx], arc.corridor_x[rem_idx]]
                        u_val -= float(np.sum(np.where(y_rem, 1.0, -1.0)))
                    score_tbl[a_i, ia, ib] = u_val
                else:
                    assert grad_mag is not None
                    # Outward semantic guard
                    if len(add_idx) > 0 and prob_512 is not None and p_floor is not None:
                        mean_p = float(prob_512[arc.corridor_y[add_idx], arc.corridor_x[add_idx]].mean())
                        if mean_p < p_floor:
                            continue
                    # Sample Scharr gradient at displaced curve with fixed original arc weights
                    disp = ((1.0 - arc.u_params) * float(a) + arc.u_params * float(b))[:, None]
                    new_xy = arc.pts_xy + disp * arc.normals_xy
                    gd, okd = _bilinear_sample_2d(grad_mag, new_xy[:, 0], new_xy[:, 1])
                    if not np.all(okd):
                        continue
                    score_tbl[a_i, ia, ib] = float(gd.sum()) - base_g

    # Exact DP with tie-breaking penalty favoring zero shift and smaller total |d|
    tie_eps = 1e-7
    shift_abs = np.array([abs(s) for s in shifts], dtype=np.float64)

    best_total_score = 0.0
    best_path = [0] * n_cps

    if loop.is_closed:
        start_states = [zero_idx] if loop.cp_locked_zero[0] else list(range(n_s))
        for s0 in start_states:
            dp = np.full((n_cps, n_s), neg_inf, dtype=np.float64)
            parent = np.full((n_cps, n_s), -1, dtype=np.int32)
            dp[0, s0] = -tie_eps * shift_abs[s0]

            for step in range(1, n_cps):
                trans = score_tbl[step - 1]  # [n_s, n_s]
                cand = dp[step - 1][:, None] + trans - tie_eps * shift_abs[None, :]
                best_prev = np.argmax(cand, axis=0)
                best_val = cand[best_prev, np.arange(n_s)]
                dp[step] = np.where(best_val > neg_inf * 0.5, best_val, neg_inf)
                parent[step] = best_prev

            # Close loop via final arc (n_cps - 1 -> 0) back to s0
            close_vals = dp[n_cps - 1] + score_tbl[n_cps - 1, :, s0]
            last_s = int(np.argmax(close_vals))
            raw_total = float(close_vals[last_s])
            if raw_total > best_total_score + 1e-9:
                # Backtrack
                path_idx = [0] * n_cps
                cur = last_s
                for step in range(n_cps - 1, 0, -1):
                    path_idx[step] = cur
                    cur = int(parent[step, cur])
                path_idx[0] = s0
                # Recompute exact untied score
                exact_sc = sum(
                    float(score_tbl[i, path_idx[i], path_idx[(i + 1) % n_cps]])
                    for i in range(n_arcs)
                )
                if exact_sc > best_total_score + 1e-9:
                    best_total_score = exact_sc
                    best_path = [shifts[idx] for idx in path_idx]
    else:
        dp = np.full((n_cps, n_s), neg_inf, dtype=np.float64)
        parent = np.full((n_cps, n_s), -1, dtype=np.int32)
        for s0 in range(n_s):
            if not loop.cp_locked_zero[0] or s0 == zero_idx:
                dp[0, s0] = -tie_eps * shift_abs[s0]
        for step in range(1, n_cps):
            trans = score_tbl[step - 1]
            cand = dp[step - 1][:, None] + trans - tie_eps * shift_abs[None, :]
            best_prev = np.argmax(cand, axis=0)
            best_val = cand[best_prev, np.arange(n_s)]
            dp[step] = np.where(best_val > neg_inf * 0.5, best_val, neg_inf)
            parent[step] = best_prev
        last_s = int(np.argmax(dp[n_cps - 1]))
        if float(dp[n_cps - 1, last_s]) > 1e-9:
            path_idx = [0] * n_cps
            cur = last_s
            for step in range(n_cps - 1, 0, -1):
                path_idx[step] = cur
                cur = int(parent[step, cur])
            path_idx[0] = cur
            exact_sc = sum(float(score_tbl[i, path_idx[i], path_idx[i + 1]]) for i in range(n_arcs))
            if exact_sc > 1e-9:
                best_total_score = exact_sc
                best_path = [shifts[idx] for idx in path_idx]

    moved = any(d != 0 for d in best_path)
    return best_path, best_total_score, moved


def refine_mask_deformable_grad(
    m0: np.ndarray,
    rgb_512: np.ndarray | None,
    prob_512: np.ndarray | None,
    *,
    allow_outward: bool = True,
    p_floor: float | None = 0.28,
    tau_grad_per_px: float = 0.035,
    gt_mask_offline: np.ndarray | None = None,
    cfg: DeformableBoundaryConfig = DeformableBoundaryConfig(),
) -> dict[str, Any]:
    """Refine binary mask M_coarse -> M_final via deformable multi-control-point DP + topology verification."""
    m0_b = np.asarray(m0, dtype=bool)
    if not m0_b.any() or not cfg.enabled:
        return {
            "mask": m0_b.copy(),
            "moved_loops": 0,
            "total_loops": 0,
            "moved_arc_len_px": 0,
            "total_arc_len_px": 0,
            "mean_abs_d_moved": 0.0,
            "topology_reverts": 0,
            "net_area_delta_px": 0,
        }

    grad_mag = None
    if gt_mask_offline is None:
        if rgb_512 is None:
            raise ValueError("rgb_512 is required for deformable gradient refinement")
        luma = _compute_luma(rgb_512)
        grad_mag = compute_scharr_gradient_magnitude(luma)

    _, num_fg_init, loops = build_deformable_loops(m0_b, cfg)
    _, num_bg_init = nd_label(~m0_b, structure=CROSS_3)

    m_cur = m0_b.copy()
    moved_loops = 0
    total_arc_len = 0
    moved_arc_len = 0
    abs_ds: list[float] = []
    topo_reverts = 0

    for loop in loops:
        total_arc_len += loop.perimeter_px
        path, total_gain, has_move = solve_deformable_loop_dp(
            loop,
            grad_mag=grad_mag,
            prob_512=prob_512,
            gt_mask_offline=gt_mask_offline,
            allow_outward=allow_outward,
            p_floor=p_floor,
            cfg=cfg,
        )
        if not has_move:
            continue
        if gt_mask_offline is None:
            gain_per_px = total_gain / max(1, loop.perimeter_px)
            if gain_per_px <= tau_grad_per_px:
                continue

        # Apply loop displacement inside local bbox and verify exact topology preservation
        cy0, cy1, cx0, cx1 = loop.local_bbox
        sub_cur = m_cur[cy0:cy1, cx0:cx1]
        _, fg_old = nd_label(sub_cur, structure=BOX_3)
        _, bg_old = nd_label(~sub_cur, structure=CROSS_3)
        sub_try = sub_cur.copy()

        loop_moved_arc = 0
        loop_abs_d: list[float] = []
        for arc in loop.arcs:
            a = path[arc.start_cp]
            b = path[arc.end_cp]
            if a == 0 and b == 0:
                continue
            add_idx, rem_idx = _arc_changed_indices(arc, a, b)
            if len(add_idx) > 0:
                sub_try[arc.corridor_y[add_idx] - cy0, arc.corridor_x[add_idx] - cx0] = True
            if len(rem_idx) > 0:
                sub_try[arc.corridor_y[rem_idx] - cy0, arc.corridor_x[rem_idx] - cx0] = False
            loop_moved_arc += len(arc.pts_xy)
            loop_abs_d.append(0.5 * (abs(a) + abs(b)))

        if nd_label(sub_try, structure=BOX_3)[1] == fg_old and nd_label(~sub_try, structure=CROSS_3)[1] == bg_old:
            m_cur[cy0:cy1, cx0:cx1] = sub_try
            moved_loops += 1
            moved_arc_len += loop_moved_arc
            abs_ds.extend(loop_abs_d)
        else:
            topo_reverts += 1

    # Final global topology safety net
    if moved_loops > 0 and not check_topology_preserved(m0_b, m_cur, n_fg_old=num_fg_init, n_bg_old=num_bg_init):
        m_cur = m0_b.copy()
        moved_loops = 0
        moved_arc_len = 0
        abs_ds = []
        topo_reverts += 1

    return {
        "mask": m_cur,
        "moved_loops": moved_loops,
        "total_loops": len(loops),
        "moved_arc_len_px": moved_arc_len,
        "total_arc_len_px": total_arc_len,
        "mean_abs_d_moved": float(np.mean(abs_ds)) if abs_ds else 0.0,
        "topology_reverts": topo_reverts,
        "net_area_delta_px": int(m_cur.sum()) - int(m0_b.sum()),
    }
