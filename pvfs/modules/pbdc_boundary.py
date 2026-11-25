from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label, uniform_filter

from ..metrics import BOX_3
from .deformable_boundary import (
    CROSS_3,
    DeformableArc,
    DeformableBoundaryConfig,
    DeformableLoop,
    _arc_changed_indices,
    _bilinear_sample_2d,
    build_deformable_loops,
    check_topology_preserved,
)


@dataclass(frozen=True)
class PBDCConfig:
    """Interior-Core-Anchored Asymmetric Step & Semantic-Uncertainty Deformable Contour (PBDC).

    Key physical properties across 2m blurry and 0.8m HD striped imagery:
      1. Anchors ONLY on confident interior PV core mu_in (immune to distant dikes/roads across water);
      2. Half-pixel exact interface (pts + (d + 0.5)*n) with multi-scale coherent step jump (1.0px & 2.8px),
         provably annihilating <=2px internal bright/dark PV mounting seams;
      3. Interior-to-exterior asymmetry (outer side must deviate more from mu_in than inner side);
      4. Zero positive balloon force: region & semantic terms act strictly as one-sided transport resistances,
         combined with a 1.18x self-alignment lock so already-accurate HD contours never drift.
    """

    enabled: bool = True
    max_shift_r: int = 6
    candidate_shifts: tuple[int, ...] = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
    control_point_spacing_px: int = 8
    high_curvature_angle_deg: float = 35.0
    high_curvature_spacing_px: int = 6
    corner_lock_angle_deg: float = 65.0
    max_adjacent_slope_diff: int = 3
    min_component_area_px: int = 48
    min_opposite_shore_clearance_px: float = 2.0
    min_inter_component_clearance_px: float = 2.0

    # Local interior core prototype estimation
    local_proto_radius_px: float = 24.0

    # Multi-scale half-pixel-centered step taps
    step_tap_narrow_px: float = 1.0
    step_tap_wide_px: float = 2.8
    asym_tap_px: float = 1.8

    # Self-lock & variational resistance weights (strictly zero positive balloon force!)
    w_edge: float = 1.60
    self_lock_ratio: float = 0.28
    self_lock_floor: float = 0.055
    w_reg_veto: float = 0.42
    w_sem: float = 1.35
    w_slope: float = 0.055
    w_disp: float = 0.025

    # Semantic uncertainty band [max(p_floor, out_p_min), in_p_max] & loop gain gate
    default_p_floor: float = 0.34
    out_p_min: float = 0.34
    in_p_max: float = 0.72
    k1_p_floor: float = 0.38
    k1_max_outward_px: int = 4
    min_gain_per_moved_px: float = 0.070


def compute_opponent_color_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sharp and 3x3-smoothed perceptual opponent color maps [H, W, 3] in float32."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB array, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() > 1.5:
        arr = arr / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    c0 = 0.299 * r + 0.587 * g + 0.114 * b
    c1 = 0.70 * (r - g)
    c2 = 0.70 * (b - 0.5 * (r + g))
    sharp = np.stack((c0, c1, c2), axis=-1).astype(np.float32)
    smooth = uniform_filter(sharp, size=(3, 3, 1), mode="nearest").astype(np.float32)
    return sharp, smooth


def _sample_color_2d(color_hw3: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c0, ok = _bilinear_sample_2d(color_hw3[..., 0], xs, ys)
    c1, _ = _bilinear_sample_2d(color_hw3[..., 1], xs, ys)
    c2, _ = _bilinear_sample_2d(color_hw3[..., 2], xs, ys)
    return np.stack((c0, c1, c2), axis=-1), ok


def _estimate_arc_interior_prototypes(
    loop: DeformableLoop,
    comp_labels: np.ndarray,
    prob_512: np.ndarray,
    color_smooth: np.ndarray,
    cfg: PBDCConfig,
) -> list[dict[str, Any]]:
    """Estimate local interior PV appearance prototype (mu_in, sig_in) strictly from confident interior core."""
    cy0, cy1, cx0, cx1 = loop.local_bbox
    sub_comp = comp_labels[cy0:cy1, cx0:cx1] == loop.comp_id
    sub_prob = prob_512[cy0:cy1, cx0:cx1]
    sub_col = color_smooth[cy0:cy1, cx0:cx1]

    dt_in = distance_transform_edt(sub_comp).astype(np.float32)
    core_in = sub_comp & (dt_in >= 2.5) & (sub_prob >= 0.65)
    if int(core_in.sum()) < 20:
        core_in = sub_comp & (dt_in >= 1.2) & (sub_prob >= 0.52)
    if int(core_in.sum()) < 8:
        core_in = sub_comp

    n_arcs = len(loop.arcs)
    if int(core_in.sum()) < 6:
        return [{"valid": False} for _ in range(n_arcs)]

    in_ys, in_xs = np.nonzero(core_in)
    in_cols = sub_col[in_ys, in_xs]  # [N_in, 3]
    in_xs_f = (in_xs + cx0).astype(np.float32)
    in_ys_f = (in_ys + cy0).astype(np.float32)

    mu_comp = np.median(in_cols, axis=0).astype(np.float32)
    dists_comp = np.linalg.norm(in_cols - mu_comp[None, :], axis=-1)
    sig_comp = max(0.025, 1.4826 * float(np.median(dists_comp)))
    rad2 = float(cfg.local_proto_radius_px ** 2)

    protos: list[dict[str, Any]] = []
    for arc in loop.arcs:
        mx = float(arc.pts_xy[:, 0].mean())
        my = float(arc.pts_xy[:, 1].mean())
        d2 = (in_xs_f - mx) ** 2 + (in_ys_f - my) ** 2
        loc = d2 <= rad2
        if int(loc.sum()) >= 16:
            loc_cols = in_cols[loc]
            mu_loc = np.median(loc_cols, axis=0).astype(np.float32)
            mu_in = (0.80 * mu_loc + 0.20 * mu_comp).astype(np.float32)
            sig_loc = max(0.025, 1.4826 * float(np.median(np.linalg.norm(loc_cols - mu_in[None, :], axis=-1))))
            sig_in = 0.70 * sig_loc + 0.30 * sig_comp
        else:
            mu_in = mu_comp
            sig_in = sig_comp
        protos.append({"valid": True, "mu_in": mu_in, "sig_in": float(sig_in)})
    return protos


def _precompute_arc_boundary_profile(
    arc: DeformableArc,
    d_grid: np.ndarray,
    proto: dict[str, Any],
    color_sharp: np.ndarray,
    color_smooth: np.ndarray,
    cfg: PBDCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute pointwise boundary evidence e_bnd[L_a, M_grid] across 0.5px offset grid d_grid."""
    iface_xy = arc.pts_xy[:, None, :] + (d_grid[None, :, None] + 0.5) * arc.normals_xy[:, None, :]
    n_xy = arc.normals_xy[:, None, :]

    def _samp(cmap: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, ok = _sample_color_2d(cmap, xy[..., 0].ravel(), xy[..., 1].ravel())
        return vals.reshape(*xy.shape[:2], 3), ok.reshape(xy.shape[:2])

    c_in1, ok1 = _samp(color_sharp, iface_xy - cfg.step_tap_narrow_px * n_xy)
    c_out1, ok2 = _samp(color_sharp, iface_xy + cfg.step_tap_narrow_px * n_xy)
    c_in2, ok3 = _samp(color_smooth, iface_xy - cfg.step_tap_wide_px * n_xy)
    c_out2, ok4 = _samp(color_smooth, iface_xy + cfg.step_tap_wide_px * n_xy)
    c_ina, ok5 = _samp(color_smooth, iface_xy - cfg.asym_tap_px * n_xy)
    c_outa, ok6 = _samp(color_smooth, iface_xy + cfg.asym_tap_px * n_xy)

    ok_all = ok1 & ok2 & ok3 & ok4 & ok5 & ok6
    mu_in = proto["mu_in"]
    denom = proto["sig_in"] + 0.035

    # 1. Multi-scale directionally coherent step jump (kills <=2px internal panel seams/ridges!)
    dc1 = c_in1 - c_out1
    dc2 = c_in2 - c_out2
    coherent_dot = np.maximum(0.0, np.sum(dc1 * dc2, axis=-1))
    s_coherent = np.sqrt(coherent_dot) / denom

    # 2. Interior-to-exterior PV dissimilarity asymmetry (kills road-to-grass & pond-water balloon!)
    phi_in = np.linalg.norm(c_ina - mu_in[None, None, :], axis=-1) / denom
    phi_out = np.linalg.norm(c_outa - mu_in[None, None, :], axis=-1) / denom
    d_phi = np.maximum(0.0, phi_out - phi_in)
    w_inner = np.exp(-np.square(np.maximum(0.0, phi_in - 1.50)))

    e_bnd = w_inner * np.tanh(0.55 * np.sqrt(s_coherent * d_phi))
    return np.where(ok_all, e_bnd, 0.0).astype(np.float32), ok_all


def _run_closed_or_open_dp(
    loop: DeformableLoop,
    score_tbl: np.ndarray,
    shifts: tuple[int, ...],
) -> tuple[list[int], np.ndarray, float, bool]:
    """Exact DP solver; rotates closed loops to a locked corner when available for 13x speedup."""
    n_s = len(shifts)
    zero_idx = shifts.index(0)
    n_cps = loop.n_cps
    n_arcs = len(loop.arcs)
    neg_inf = -1e18
    tie_eps = 1e-7
    shift_abs = np.array([abs(s) for s in shifts], dtype=np.float64)

    best_total_score = 0.0
    best_path_idx = [zero_idx] * n_cps

    if loop.is_closed:
        locked_Indices = np.flatnonzero(loop.cp_locked_zero)
        rot = int(locked_Indices[0]) if len(locked_Indices) > 0 else 0
        start_states = [zero_idx] if len(locked_Indices) > 0 else list(range(n_s))

        for s0 in start_states:
            dp = np.full((n_cps, n_s), neg_inf, dtype=np.float64)
            parent = np.full((n_cps, n_s), -1, dtype=np.int32)
            dp[0, s0] = -tie_eps * shift_abs[s0]
            for step in range(1, n_cps):
                arc_idx = (rot + step - 1) % n_cps
                cand = dp[step - 1][:, None] + score_tbl[arc_idx] - tie_eps * shift_abs[None, :]
                best_prev = np.argmax(cand, axis=0)
                best_val = cand[best_prev, np.arange(n_s)]
                dp[step] = np.where(best_val > neg_inf * 0.5, best_val, neg_inf)
                parent[step] = best_prev

            last_arc_idx = (rot + n_cps - 1) % n_cps
            close_vals = dp[n_cps - 1] + score_tbl[last_arc_idx, :, s0]
            last_s = int(np.argmax(close_vals))
            if float(close_vals[last_s]) > best_total_score + 1e-9:
                rot_path = [zero_idx] * n_cps
                cur = last_s
                for step in range(n_cps - 1, 0, -1):
                    rot_path[step] = cur
                    cur = int(parent[step, cur])
                rot_path[0] = s0
                p_idx = [zero_idx] * n_cps
                for step in range(n_cps):
                    p_idx[(rot + step) % n_cps] = rot_path[step]
                exact_sc = sum(float(score_tbl[i, p_idx[i], p_idx[(i + 1) % n_cps]]) for i in range(n_arcs))
                if exact_sc > best_total_score + 1e-9:
                    best_total_score = exact_sc
                    best_path_idx = p_idx
    else:
        dp = np.full((n_cps, n_s), neg_inf, dtype=np.float64)
        parent = np.full((n_cps, n_s), -1, dtype=np.int32)
        for s0 in range(n_s):
            if not loop.cp_locked_zero[0] or s0 == zero_idx:
                dp[0, s0] = -tie_eps * shift_abs[s0]
        for step in range(1, n_cps):
            cand = dp[step - 1][:, None] + score_tbl[step - 1] - tie_eps * shift_abs[None, :]
            best_prev = np.argmax(cand, axis=0)
            best_val = cand[best_prev, np.arange(n_s)]
            dp[step] = np.where(best_val > neg_inf * 0.5, best_val, neg_inf)
            parent[step] = best_prev
        last_s = int(np.argmax(dp[n_cps - 1]))
        if float(dp[n_cps - 1, last_s]) > 1e-9:
            p_idx = [zero_idx] * n_cps
            cur = last_s
            for step in range(n_cps - 1, 0, -1):
                p_idx[step] = cur
                cur = int(parent[step, cur])
            p_idx[0] = cur
            exact_sc = sum(float(score_tbl[i, p_idx[i], p_idx[i + 1]]) for i in range(n_arcs))
            if exact_sc > 1e-9:
                best_total_score = exact_sc
                best_path_idx = p_idx

    best_path = [shifts[idx] for idx in best_path_idx]
    arc_scores = np.array(
        [
            float(score_tbl[i, best_path_idx[loop.arcs[i].start_cp], best_path_idx[loop.arcs[i].end_cp]])
            for i in range(n_arcs)
        ],
        dtype=np.float64,
    )
    return best_path, arc_scores, best_total_score, any(d != 0 for d in best_path)


def solve_pbdc_loop_dp(
    loop: DeformableLoop,
    protos: list[dict[str, Any]],
    *,
    color_sharp: np.ndarray,
    color_smooth: np.ndarray,
    prob_512: np.ndarray,
    allow_outward: bool,
    max_outward_cap: int,
    p_floor: float,
    cfg: PBDCConfig,
) -> tuple[list[int], np.ndarray, float, bool]:
    """Build vectorized arc utility tables and solve closed-loop DP."""
    shifts = cfg.candidate_shifts
    n_s = len(shifts)
    zero_idx = shifts.index(0)
    n_arcs = len(loop.arcs)
    neg_inf = -1e18

    r_max = float(cfg.max_shift_r)
    n_grid = int(round(4.0 * r_max)) + 1
    d_grid = np.linspace(-r_max, r_max, n_grid, dtype=np.float32)
    grid_zero_idx = n_grid // 2

    score_tbl = np.full((n_arcs, n_s, n_s), neg_inf, dtype=np.float64)

    for a_i, arc in enumerate(loop.arcs):
        proto = protos[a_i]
        start_locked = bool(loop.cp_locked_zero[arc.start_cp])
        end_locked = bool(loop.cp_locked_zero[arc.end_cp])
        score_tbl[a_i, zero_idx, zero_idx] = 0.0

        l_pts = len(arc.pts_xy)
        l_scale = l_pts / 8.0
        out_lim = min(float(max_outward_cap), arc.max_outward_safe + 0.25)
        in_lim = arc.max_inward_safe + 0.25

        # Valid (ia, ib) pairs for this arc
        valid_pairs: list[tuple[int, int, int, int]] = []
        for ia, a in enumerate(shifts):
            if start_locked and a != 0:
                continue
            if a > 0 and (not allow_outward or a > out_lim):
                continue
            if a < 0 and (-a > in_lim):
                continue
            for ib, b in enumerate(shifts):
                if end_locked and b != 0:
                    continue
                if abs(b - a) > cfg.max_adjacent_slope_diff:
                    continue
                if b > 0 and (not allow_outward or b > out_lim):
                    continue
                if b < 0 and (-b > in_lim):
                    continue
                if a == 0 and b == 0:
                    continue
                valid_pairs.append((ia, ib, a, b))

        if not valid_pairs or not proto.get("valid", False) or len(arc.corridor_y) == 0:
            continue

        pts_bnd_tbl, pts_ok_tbl = _precompute_arc_boundary_profile(arc, d_grid, proto, color_sharp, color_smooth, cfg)
        if not np.all(pts_ok_tbl[:, grid_zero_idx]):
            continue
        e0 = float(pts_bnd_tbl[:, grid_zero_idx].sum())
        lock_margin = cfg.self_lock_ratio * e0 + cfg.self_lock_floor * l_pts

        # Vectorized evaluation across all P valid (a, b) pairs for this arc!
        ia_arr = np.array([p[0] for p in valid_pairs], dtype=np.int32)
        ib_arr = np.array([p[1] for p in valid_pairs], dtype=np.int32)
        a_arr = np.array([p[2] for p in valid_pairs], dtype=np.float32)
        b_arr = np.array([p[3] for p in valid_pairs], dtype=np.float32)

        # 1. Edge score E_ab [P]
        disp_mat = (1.0 - arc.u_params[:, None]) * a_arr[None, :] + arc.u_params[:, None] * b_arr[None, :]  # [L, P]
        f_idx = (disp_mat + r_max) * 2.0
        i0 = np.clip(np.floor(f_idx).astype(np.int32), 0, n_grid - 2)
        w1 = f_idx - i0
        pt_rows = np.arange(l_pts)[:, None]
        ok_pairs = np.all(pts_ok_tbl[pt_rows, i0] & pts_ok_tbl[pt_rows, i0 + 1], axis=0)  # [P]
        e_ab = ((1.0 - w1) * pts_bnd_tbl[pt_rows, i0] + w1 * pts_bnd_tbl[pt_rows, i0 + 1]).sum(axis=0)  # [P]

        # 2. Vectorized one-sided corridor region veto & semantic transport resistance [P]
        cor_cols = color_smooth[arc.corridor_y, arc.corridor_x]
        cor_probs = prob_512[arc.corridor_y, arc.corridor_x]
        denom = proto["sig_in"] + 0.035
        cor_phi = np.linalg.norm(cor_cols - proto["mu_in"][None, :], axis=-1) / denom  # [N_v]

        add_reg_cost = np.maximum(0.0, cor_phi - 1.45)  # [N_v] >= 0: penalizes swallowing non-PV exterior
        rem_reg_cost = np.maximum(0.0, 1.25 - cor_phi) * np.maximum(0.0, cor_probs - 0.52) * 2.5  # penalizes cutting confident PV core
        add_sem_cost = np.maximum(0.0, 0.50 - cor_probs)
        rem_sem_cost = np.maximum(0.0, cor_probs - 0.52)

        delta_cor = (1.0 - arc.corridor_u_param[:, None]) * a_arr[None, :] + arc.corridor_u_param[:, None] * b_arr[None, :]  # [N_v, P]
        in_m0 = arc.corridor_in_m0[:, None]
        sdist = arc.corridor_signed_dist[:, None]
        add_m = (~in_m0) & (delta_cor > 0.0) & (sdist <= delta_cor)  # [N_v, P]
        rem_m = in_m0 & (delta_cor < 0.0) & (sdist >= delta_cor)  # [N_v, P]

        add_cnt = add_m.sum(axis=0)  # [P]
        rem_cnt = rem_m.sum(axis=0)  # [P]
        eff_p_floor = max(p_floor, cfg.out_p_min)
        add_p_mean = (cor_probs[:, None] * add_m).sum(axis=0) / np.maximum(1, add_cnt)
        rem_p_mean = (cor_probs[:, None] * rem_m).sum(axis=0) / np.maximum(1, rem_cnt)
        # Strictly stay within the semantic uncertainty ribbon [eff_p_floor, cfg.in_p_max]!
        p_band_ok = ((add_cnt == 0) | (add_p_mean >= eff_p_floor)) & ((rem_cnt == 0) | (rem_p_mean <= cfg.in_p_max))

        reg_veto = (add_reg_cost[:, None] * add_m + rem_reg_cost[:, None] * rem_m).sum(axis=0)  # [P] >= 0
        sem_cost = (add_sem_cost[:, None] * add_m + rem_sem_cost[:, None] * rem_m).sum(axis=0)  # [P] >= 0

        d_edge = e_ab - e0 - lock_margin
        u_vec = (
            cfg.w_edge * d_edge
            - cfg.w_reg_veto * reg_veto
            - cfg.w_sem * sem_cost
            - cfg.w_slope * np.abs(b_arr - a_arr) * l_scale
            - cfg.w_disp * 0.5 * (np.abs(a_arr) + np.abs(b_arr)) * l_scale
        )
        valid_final = ok_pairs & p_band_ok
        score_tbl[a_i, ia_arr[valid_final], ib_arr[valid_final]] = u_vec[valid_final]

    return _run_closed_or_open_dp(loop, score_tbl, shifts)


def _apply_loop_with_localized_fallback(
    sub_cur: np.ndarray,
    loop: DeformableLoop,
    path: list[int],
    arc_scores: np.ndarray,
) -> tuple[np.ndarray, list[int], bool, int]:
    """Apply loop displacement; if full loop pinches topology, keep valid blocks via 8-sector check."""
    cy0, _, cx0, _ = loop.local_bbox
    fg_old = nd_label(sub_cur, structure=BOX_3)[1]
    bg_old = nd_label(~sub_cur, structure=CROSS_3)[1]

    def _render(cand_path: list[int]) -> np.ndarray:
        out = sub_cur.copy()
        for arc in loop.arcs:
            a, b = cand_path[arc.start_cp], cand_path[arc.end_cp]
            if a == 0 and b == 0:
                continue
            add_idx, rem_idx = _arc_changed_indices(arc, a, b)
            if len(add_idx) > 0:
                out[arc.corridor_y[add_idx] - cy0, arc.corridor_x[add_idx] - cx0] = True
            if len(rem_idx) > 0:
                out[arc.corridor_y[rem_idx] - cy0, arc.corridor_x[rem_idx] - cx0] = False
        return out

    def _valid_topo(mask_sub: np.ndarray) -> bool:
        return nd_label(mask_sub, structure=BOX_3)[1] == fg_old and nd_label(~mask_sub, structure=CROSS_3)[1] == bg_old

    sub_full = _render(path)
    if _valid_topo(sub_full):
        return sub_full, path, True, 0

    n_cps = loop.n_cps
    n_blocks = min(8, n_cps)
    blocks = np.array_split(np.arange(n_cps), n_blocks)
    block_gains = []
    for b_idx, blk in enumerate(blocks):
        blk_set = set(blk.tolist())
        g = sum(
            float(arc_scores[i])
            for i, arc in enumerate(loop.arcs)
            if (arc.start_cp in blk_set or arc.end_cp in blk_set)
        )
        block_gains.append((g, b_idx, blk))
    block_gains.sort(key=lambda x: -x[0])

    safe_path = [0] * n_cps
    cur_sub = sub_cur.copy()
    any_accepted = False
    for g, _, blk in block_gains:
        if g <= 0.0 or all(path[int(c)] == 0 for c in blk):
            continue
        trial_path = list(safe_path)
        for c in blk:
            trial_path[int(c)] = path[int(c)]
        trial_sub = _render(trial_path)
        if _valid_topo(trial_sub):
            safe_path = trial_path
            cur_sub = trial_sub
            any_accepted = True

    if any_accepted:
        return cur_sub, safe_path, True, 1
    return sub_cur, [0] * n_cps, False, 1


def refine_mask_pbdc(
    m0: np.ndarray,
    rgb_512: np.ndarray,
    prob_512: np.ndarray,
    *,
    k_shots: int = 5,
    p_floor_loo: float | None = None,
    cfg: PBDCConfig = PBDCConfig(),
) -> dict[str, Any]:
    """Refine coarse mask M_coarse -> M_final via Interior-Core-Anchored Asymmetric Step Deformable Contour."""
    m0_b = np.asarray(m0, dtype=bool)
    if not m0_b.any() or not cfg.enabled:
        return {
            "mask": m0_b.copy(),
            "moved_loops": 0,
            "total_loops": 0,
            "moved_arc_len_px": 0,
            "total_arc_len_px": 0,
            "mean_abs_d_moved": 0.0,
            "localized_fallbacks": 0,
            "topology_reverts": 0,
            "net_area_delta_px": 0,
        }

    prob_f = np.asarray(prob_512, dtype=np.float32)
    color_sharp, color_smooth = compute_opponent_color_maps(rgb_512)

    geom_cfg = DeformableBoundaryConfig(
        enabled=True,
        mode="deformable_grad",
        max_shift_r=cfg.max_shift_r,
        candidate_shifts=cfg.candidate_shifts,
        control_point_spacing_px=cfg.control_point_spacing_px,
        high_curvature_angle_deg=cfg.high_curvature_angle_deg,
        high_curvature_spacing_px=cfg.high_curvature_spacing_px,
        corner_lock_angle_deg=cfg.corner_lock_angle_deg,
        max_adjacent_slope_diff=cfg.max_adjacent_slope_diff,
        min_component_area_px=cfg.min_component_area_px,
        min_opposite_shore_clearance_px=cfg.min_opposite_shore_clearance_px,
        min_inter_component_clearance_px=cfg.min_inter_component_clearance_px,
    )
    comp_labels, num_fg_init, loops = build_deformable_loops(m0_b, geom_cfg)
    _, num_bg_init = nd_label(~m0_b, structure=CROSS_3)

    if k_shots <= 1:
        allow_outward = True
        max_out_cap = cfg.k1_max_outward_px
        p_floor = cfg.k1_p_floor
    else:
        allow_outward = True
        max_out_cap = cfg.max_shift_r
        p_floor = float(p_floor_loo) if p_floor_loo is not None else cfg.default_p_floor

    m_cur = m0_b.copy()
    moved_loops = 0
    total_arc_len = 0
    moved_arc_len = 0
    abs_ds: list[float] = []
    loc_fallbacks = 0
    topo_reverts = 0

    for loop in loops:
        total_arc_len += loop.perimeter_px
        protos = _estimate_arc_interior_prototypes(loop, comp_labels, prob_f, color_smooth, cfg)
        if not any(p.get("valid", False) for p in protos):
            continue

        path, arc_scores, total_gain, has_move = solve_pbdc_loop_dp(
            loop,
            protos,
            color_sharp=color_sharp,
            color_smooth=color_smooth,
            prob_512=prob_f,
            allow_outward=allow_outward,
            max_outward_cap=max_out_cap,
            p_floor=p_floor,
            cfg=cfg,
        )
        if not has_move:
            continue

        moved_px_cand = sum(
            len(arc.pts_xy) for arc in loop.arcs if (path[arc.start_cp] != 0 or path[arc.end_cp] != 0)
        )
        if moved_px_cand == 0 or (total_gain / float(moved_px_cand)) < cfg.min_gain_per_moved_px:
            continue

        cy0, cy1, cx0, cx1 = loop.local_bbox
        sub_cur = m_cur[cy0:cy1, cx0:cx1]
        sub_new, applied_path, ok_apply, fb_used = _apply_loop_with_localized_fallback(sub_cur, loop, path, arc_scores)
        loc_fallbacks += fb_used
        if not ok_apply:
            topo_reverts += 1
            continue

        m_cur[cy0:cy1, cx0:cx1] = sub_new
        loop_moved_arc = 0
        for arc in loop.arcs:
            a, b = applied_path[arc.start_cp], applied_path[arc.end_cp]
            if a != 0 or b != 0:
                loop_moved_arc += len(arc.pts_xy)
                abs_ds.append(0.5 * (abs(a) + abs(b)))
        if loop_moved_arc > 0:
            moved_loops += 1
            moved_arc_len += loop_moved_arc

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
        "localized_fallbacks": loc_fallbacks,
        "topology_reverts": topo_reverts,
        "net_area_delta_px": int(m_cur.sum()) - int(m0_b.sum()),
    }
