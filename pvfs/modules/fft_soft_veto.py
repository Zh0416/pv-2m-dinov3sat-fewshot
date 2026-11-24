from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sp_fft


@dataclass(frozen=True)
class FFTSoftVetoConfig:
    """Locked configuration for support-driven adaptive-band FFT soft veto (no fixed-band fallback)."""

    enabled: bool = False
    window_size: int = 32
    grid_hw: int = 64
    pv_cov_min: float = 0.75
    bg_cov_max: float = 0.0
    safe_r_min: float = 4.0
    safe_r_max: float = 15.5
    band_gain_frac: float = 0.25
    min_band_width: float = 3.0
    energy_quantile: float = 0.01
    energy_floor_abs: float = 1e-8
    q_cand: float = 0.85
    min_pv_cand: int = 16
    min_bg_cand: int = 16
    num_bins: int = 6
    min_nonempty_bins: int = 3
    min_bin_total: int = 5
    min_bin_bg: int = 3
    tau: float = 0.40
    gamma: float = 0.05
    cap: float = 0.05
    k1_nonoverlap_grid_dist: int = 4  # 32px window / 8px grid stride -> |dr|>=4 or |dc|>=4 guarantees zero pixel overlap

    @classmethod
    def from_json(cls, path: Path | str, *, enabled: bool = True) -> "FFTSoftVetoConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fw = raw.get("fft_window", {})
        bs = raw.get("band_selection", {})
        cc = raw.get("candidate_calibration", {})
        pn = raw.get("penalty", {})
        al = raw.get("anti_leakage", {})
        return cls(
            enabled=enabled,
            window_size=int(fw.get("window_size", 32)),
            grid_hw=int(fw.get("grid_hw", 64)),
            pv_cov_min=float(bs.get("pv_cov_min", 0.75)),
            bg_cov_max=float(bs.get("bg_cov_max", 0.0)),
            safe_r_min=float(bs.get("safe_r_min", 4.0)),
            safe_r_max=float(bs.get("safe_r_max", 15.5)),
            band_gain_frac=float(bs.get("band_gain_frac", 0.25)),
            min_band_width=float(bs.get("min_band_width", 3.0)),
            energy_quantile=float(fw.get("energy_quantile", 0.01)),
            energy_floor_abs=float(fw.get("energy_floor_abs", 1e-8)),
            q_cand=float(cc.get("q_cand", 0.85)),
            min_pv_cand=int(cc.get("min_pv_cand", 16)),
            min_bg_cand=int(cc.get("min_bg_cand", 16)),
            num_bins=int(cc.get("num_bins", 6)),
            min_nonempty_bins=int(cc.get("min_nonempty_bins", 3)),
            min_bin_total=int(cc.get("min_bin_total", 5)),
            min_bin_bg=int(cc.get("min_bin_bg", 3)),
            tau=float(pn.get("tau", 0.40)),
            gamma=float(pn.get("gamma", 0.05)),
            cap=float(pn.get("cap", 0.05)),
            k1_nonoverlap_grid_dist=int(al.get("k1_nonoverlap_grid_dist", 4)),
        )


def fft_radius(size: int = 32) -> np.ndarray:
    f = sp_fft.fftfreq(size) * size
    return np.hypot(f[:, None], f[None, :])


def grid_centers(shape: tuple[int, int], grid_hw: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rows = ((np.arange(grid_hw) + 0.5) * shape[0] / grid_hw).astype(int)
    cols = ((np.arange(grid_hw) + 0.5) * shape[1] / grid_hw).astype(int)
    return rows, cols


def _hann_2d(size: int = 32) -> tuple[np.ndarray, float]:
    h1 = np.hanning(size).astype(np.float32)
    h2 = np.outer(h1, h1).astype(np.float32)
    norm = float(size**2 * np.square(h2).sum())
    return h2, norm


@dataclass
class ImageFFTPrecompute:
    """Per-image cached local 32x32 Hann FFT spectra aligned to the 64x64 DINO grid.

    Notice: `window_cov` is NEVER attached to query images; support masks are passed separately
    to `compute_window_mask_coverage` strictly during support calibration.
    """

    image_id: str
    window_size: int
    grid_hw: int
    rows: np.ndarray  # [64] int pixel centers
    cols: np.ndarray  # [64] int pixel centers
    interior: np.ndarray  # [64, 64] bool (entire 32x32 window strictly inside image & valid pixels)
    power: np.ndarray  # [64, 64, 32, 32] float32 non-DC FFT power (P(0)=0)
    total_power: np.ndarray  # [64, 64] float32 sum_{u!=0} P(u)
    norm_energy: np.ndarray  # [64, 64] float32 total_power / (32^2 * sum(H^2))
    radial_profile: np.ndarray  # [64, 64, 15] float32 L2-normalized log1p(p95) radial shell profile
    _peak_share_cache: dict[tuple[float, float], np.ndarray] = field(default_factory=dict, repr=False)


def precompute_image_fft(
    image_id: str,
    rgb_512: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    window_size: int = 32,
    grid_hw: int = 64,
) -> ImageFFTPrecompute:
    """Compute exact 32x32 Hann-windowed 2D FFT spectra and 15-shell radial profiles on the 64x64 grid.

    Accepts RGB in either CHW [3, H, W] or HWC [H, W, 3] uint8 / float format.
    """
    arr = np.asarray(rgb_512)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {arr.shape}")

    rgb_f = arr.astype(np.float32)
    if rgb_f.max() > 1.5:
        rgb_f = rgb_f / 255.0
    # Standard BT.601 RGB->Gray matching cv2.cvtColor(RGB2GRAY)
    gray = (0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]).astype(np.float32)
    h, w = gray.shape
    half = window_size // 2
    rows, cols = grid_centers((h, w), grid_hw=grid_hw)

    if valid_mask is None:
        valid_px = np.ones((h, w), dtype=bool)
    else:
        valid_px = np.asarray(valid_mask, dtype=bool)
        if valid_px.shape != (h, w):
            raise ValueError(f"valid_mask shape {valid_px.shape} != image shape {(h, w)}")

    # Strictly interior check: full [r-half, r+half) x [c-half, c+half) inside [0, h) x [0, w) and all pixels valid
    interior = np.zeros((grid_hw, grid_hw), dtype=bool)
    valid_r_idx = np.flatnonzero((rows >= half) & (rows + half <= h))
    valid_c_idx = np.flatnonzero((cols >= half) & (cols + half <= w))
    if len(valid_r_idx) > 0 and len(valid_c_idx) > 0:
        vp_views = sliding_window_view(valid_px, (window_size, window_size))
        r_starts = rows[valid_r_idx] - half
        c_starts = cols[valid_c_idx] - half
        sub_valid = vp_views[r_starts[:, None], c_starts[None, :]].all(axis=(-2, -1))
        interior[np.ix_(valid_r_idx, valid_c_idx)] = sub_valid

    hann, norm = _hann_2d(window_size)
    hann_sum = float(hann.sum())
    padded = np.pad(gray, half, mode="reflect")
    win_views = sliding_window_view(padded, (window_size, window_size))

    rad = fft_radius(window_size)
    bins_grid = np.rint(rad).astype(int)
    shells = [bins_grid == b for b in range(1, window_size // 2)]

    power = np.zeros((grid_hw, grid_hw, window_size, window_size), dtype=np.float32)
    total_power = np.zeros((grid_hw, grid_hw), dtype=np.float32)
    radial_profile = np.zeros((grid_hw, grid_hw, len(shells)), dtype=np.float32)

    for start in range(0, grid_hw, 8):
        stop = min(start + 8, grid_hw)
        windows = win_views[rows[start:stop, None], cols[None, :]]  # [B, 64, 32, 32]
        mean = np.sum(windows * hann, axis=(-2, -1), keepdims=True) / hann_sum
        p = (np.abs(sp_fft.fft2((windows - mean) * hann)) ** 2).astype(np.float32)
        p[..., 0, 0] = 0.0
        tot = p.sum(axis=(-2, -1))
        rad_vals = np.stack([np.percentile(p[..., sh], 95, axis=-1) for sh in shells], axis=-1).astype(np.float32)
        rad_vals = np.log1p(rad_vals)
        norms = np.linalg.norm(rad_vals, axis=-1, keepdims=True)
        rad_norm = rad_vals / np.maximum(norms, 1e-8)

        power[start:stop] = p
        total_power[start:stop] = tot
        radial_profile[start:stop] = rad_norm

    norm_energy = (total_power / norm).astype(np.float32)
    return ImageFFTPrecompute(
        image_id=image_id,
        window_size=window_size,
        grid_hw=grid_hw,
        rows=rows,
        cols=cols,
        interior=interior,
        power=power,
        total_power=total_power,
        norm_energy=norm_energy,
        radial_profile=radial_profile,
    )


def compute_window_mask_coverage(
    mask_512: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    interior: np.ndarray,
    *,
    window_size: int = 32,
) -> np.ndarray:
    """Compute exact PV coverage fraction over the ENTIRE 32x32 FFT window for each 64x64 grid position."""
    m = (np.asarray(mask_512) > 0).astype(np.float32)
    h, w = m.shape
    half = window_size // 2
    grid_hw = len(rows)
    cov = np.full((grid_hw, grid_hw), np.nan, dtype=np.float32)

    valid_r = np.flatnonzero((rows >= half) & (rows + half <= h))
    valid_c = np.flatnonzero((cols >= half) & (cols + half <= w))
    if len(valid_r) > 0 and len(valid_c) > 0:
        m_views = sliding_window_view(m, (window_size, window_size))
        r_starts = rows[valid_r] - half
        c_starts = cols[valid_c] - half
        sub_cov = m_views[r_starts[:, None], c_starts[None, :]].mean(axis=(-2, -1))
        cov[np.ix_(valid_r, valid_c)] = sub_cov
    cov[~interior] = np.nan
    return cov


def compute_adaptive_peak_share(
    pre: ImageFFTPrecompute,
    band_mask: np.ndarray,
    *,
    band_key: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute adaptive_peak_share matching physical_feature_visualization.py::band_responses.

    Finds the brightest in-band bin, unions the 3x3 neighborhoods around the peak and its conjugate
    (deduplicated near Nyquist), sums ONLY in-band power, and divides by total non-DC power.
    """
    if band_key is not None and band_key in pre._peak_share_cache:
        return pre._peak_share_cache[band_key]

    size = pre.window_size
    power_b = pre.power * band_mask  # [64, 64, 32, 32]
    offsets = np.arange(-1, 2)
    dr, dc = (a.ravel() for a in np.meshgrid(offsets, offsets, indexing="ij"))

    brightest = power_b.reshape(pre.grid_hw, pre.grid_hw, -1).argmax(axis=-1)
    pr, pc = (a[..., None] for a in np.divmod(brightest, size))
    ir, ic = np.ogrid[: pre.grid_hw, : pre.grid_hw]
    ar, ac = (pr + dr) % size, (pc + dc) % size
    br, bc = (-pr + dr) % size, (-pc + dc) % size
    delta_r = (br - pr + size // 2) % size - size // 2
    delta_c = (bc - pc + size // 2) % size - size // 2
    unique = (np.abs(delta_r) > 1) | (np.abs(delta_c) > 1)

    peak = power_b[ir[..., None], ic[..., None], ar, ac].sum(axis=-1)
    peak += (power_b[ir[..., None], ic[..., None], br, bc] * unique).sum(axis=-1)
    share = np.clip(peak / np.maximum(pre.total_power, 1e-8), 0.0, 1.0).astype(np.float32)

    if band_key is not None:
        if len(pre._peak_share_cache) > 16:
            pre._peak_share_cache.clear()
        pre._peak_share_cache[band_key] = share
    return share


@dataclass(frozen=True)
class BandSelectionResult:
    valid: bool
    reason: str
    low: float | None = None
    high: float | None = None
    peak_r: int | None = None
    peak_score: float | None = None
    band_mask: np.ndarray | None = None


def select_adaptive_band(
    pv_profiles: np.ndarray,
    bg_profiles: np.ndarray,
    cfg: FFTSoftVetoConfig,
) -> BandSelectionResult:
    """Select continuous radial band from support PV vs pure-BG radial spectra (ZERO fixed-band fallback)."""
    if pv_profiles.shape[0] == 0:
        return BandSelectionResult(valid=False, reason="no_pv_band_windows")
    if bg_profiles.shape[0] == 0:
        return BandSelectionResult(valid=False, reason="no_bg_band_windows")

    bins = np.arange(1, cfg.window_size // 2, dtype=np.float32)
    valid_r = (bins >= cfg.safe_r_min) & (bins <= cfg.safe_r_max)
    fg = pv_profiles.mean(axis=0)
    bg = bg_profiles.mean(axis=0)
    gain = fg - bg
    score = np.where(valid_r, fg * np.maximum(0.0, gain), np.nan)
    if not np.isfinite(score).any():
        return BandSelectionResult(valid=False, reason="no_valid_radius_bins")

    best_idx = int(np.nanargmax(score))
    best_score = float(score[best_idx])
    if not np.isfinite(best_score) or best_score <= 0.0 or gain[best_idx] <= 0.0:
        return BandSelectionResult(valid=False, reason="no_positive_contrast_peak", peak_score=best_score)

    selected = bins[valid_r & (gain >= cfg.band_gain_frac * gain[best_idx])]
    low = max(cfg.safe_r_min, float(selected.min()) - 0.5)
    high = min(cfg.safe_r_max, float(selected.max()) + 0.5)
    if high - low < cfg.min_band_width:
        low = float(np.clip((low + high) / 2.0 - cfg.min_band_width / 2.0, cfg.safe_r_min, cfg.safe_r_max - cfg.min_band_width))
        high = low + cfg.min_band_width

    rad = fft_radius(cfg.window_size)
    band_mask = (rad >= low) & (rad <= high)
    if not band_mask.any():
        return BandSelectionResult(valid=False, reason="empty_band_mask")

    return BandSelectionResult(
        valid=True,
        reason="ok",
        low=round(float(low), 4),
        high=round(float(high), 4),
        peak_r=int(bins[best_idx]),
        peak_score=best_score,
        band_mask=band_mask,
    )


@dataclass(frozen=True)
class FFTSoftVetoReference:
    """Episode-level (or LOO-level) support-calibrated adaptive FFT soft-veto reference."""

    active: bool
    reason: str
    band_selected: bool = False
    band_low: float | None = None
    band_high: float | None = None
    band_peak_r: int | None = None
    band_mask: np.ndarray | None = None
    energy_thr: float = 1e-8
    t_cand: float = 0.0
    a_min: float = 0.0
    a_max: float = 1.0
    edges: np.ndarray | None = None  # [B'+1] sorted unique edges
    n_pv_bins: np.ndarray | None = None  # [B'] int counts
    n_bg_bins: np.ndarray | None = None  # [B'] int counts
    e_bg_bins: np.ndarray | None = None  # [B'] float32 add-1 smoothed log-likelihood ratio
    bin_eligible: np.ndarray | None = None  # [B'] bool (total >= 5 and bg >= 3)
    n_pv_band_windows: int = 0
    n_bg_band_windows: int = 0
    n_pv_cand_windows: int = 0
    n_bg_cand_windows: int = 0
    num_merged_bins: int = 0
    num_nonempty_bins: int = 0
    num_eligible_bins: int = 0


def fit_fft_soft_veto_reference(
    pre_list: list[ImageFFTPrecompute],
    cov_list: list[np.ndarray],
    s_deep_list: list[np.ndarray],
    cfg: FFTSoftVetoConfig,
    *,
    exclude_mask_list: list[np.ndarray] | None = None,
) -> FFTSoftVetoReference:
    """Fit adaptive band, energy floor, t_cand, and 6-bin smoothed E_bg reference strictly from support data."""
    if not cfg.enabled:
        return FFTSoftVetoReference(active=False, reason="disabled")

    n_sup = len(pre_list)
    if n_sup == 0:
        return FFTSoftVetoReference(active=False, reason="no_support_images")

    # 1. Energy threshold Q_0.01 over valid interior support windows
    e_vals = []
    valid_interior_list: list[np.ndarray] = []
    for i in range(n_sup):
        vi = pre_list[i].interior.copy()
        if exclude_mask_list is not None:
            vi &= ~exclude_mask_list[i]
        valid_interior_list.append(vi)
        if vi.any():
            e_vals.append(pre_list[i].norm_energy[vi])
    if not e_vals:
        return FFTSoftVetoReference(active=False, reason="no_valid_interior_support_windows")
    all_e = np.concatenate(e_vals)
    energy_thr = max(cfg.energy_floor_abs, float(np.quantile(all_e, cfg.energy_quantile)))

    # 2. Whole-window PV (>=0.75) and pure-BG (==0) masks for band selection & candidate calibration
    pv_profs, bg_profs = [], []
    pv_masks: list[np.ndarray] = []
    bg_masks: list[np.ndarray] = []
    for i in range(n_sup):
        ev = valid_interior_list[i] & (pre_list[i].norm_energy >= energy_thr)
        cov = cov_list[i]
        finite_s = np.isfinite(s_deep_list[i])
        pv_m = ev & np.isfinite(cov) & (cov >= cfg.pv_cov_min) & finite_s
        bg_m = ev & np.isfinite(cov) & (cov <= cfg.bg_cov_max) & finite_s
        pv_masks.append(pv_m)
        bg_masks.append(bg_m)
        if pv_m.any():
            pv_profs.append(pre_list[i].radial_profile[pv_m])
        if bg_m.any():
            bg_profs.append(pre_list[i].radial_profile[bg_m])

    n_pv_band = sum(int(m.sum()) for m in pv_masks)
    n_bg_band = sum(int(m.sum()) for m in bg_masks)
    if n_pv_band == 0 or n_bg_band == 0:
        return FFTSoftVetoReference(
            active=False,
            reason="insufficient_band_support_windows",
            energy_thr=energy_thr,
            n_pv_band_windows=n_pv_band,
            n_bg_band_windows=n_bg_band,
        )

    band_res = select_adaptive_band(np.concatenate(pv_profs, axis=0), np.concatenate(bg_profs, axis=0), cfg)
    if not band_res.valid or band_res.band_mask is None:
        return FFTSoftVetoReference(
            active=False,
            reason=band_res.reason,
            band_selected=False,
            energy_thr=energy_thr,
            n_pv_band_windows=n_pv_band,
            n_bg_band_windows=n_bg_band,
        )

    # 3. Candidate threshold t_cand = Q_{0.85} of valid pure-BG support windows' deep scores
    bg_deep_all = np.concatenate([s_deep_list[i][bg_masks[i]] for i in range(n_sup) if bg_masks[i].any()])
    t_cand = float(np.quantile(bg_deep_all, cfg.q_cand))

    # 4. Compute adaptive_peak_share on support windows with S_deep >= t_cand
    band_key = (float(band_res.low), float(band_res.high))
    a_pv_list, a_bg_list = [], []
    for i in range(n_sup):
        pv_c = pv_masks[i] & (s_deep_list[i] >= t_cand)
        bg_c = bg_masks[i] & (s_deep_list[i] >= t_cand)
        if pv_c.any() or bg_c.any():
            a_map = compute_adaptive_peak_share(pre_list[i], band_res.band_mask, band_key=band_key)
            if pv_c.any():
                a_pv_list.append(a_map[pv_c])
            if bg_c.any():
                a_bg_list.append(a_map[bg_c])

    a_pv = np.concatenate(a_pv_list) if a_pv_list else np.empty((0,), dtype=np.float32)
    a_bg = np.concatenate(a_bg_list) if a_bg_list else np.empty((0,), dtype=np.float32)
    n_pv_c = int(a_pv.shape[0])
    n_bg_c = int(a_bg.shape[0])

    if n_pv_c < cfg.min_pv_cand or n_bg_c < cfg.min_bg_cand:
        reason = "insufficient_pv_candidates" if n_pv_c < cfg.min_pv_cand else "insufficient_bg_candidates"
        return FFTSoftVetoReference(
            active=False,
            reason=reason,
            band_selected=True,
            band_low=band_res.low,
            band_high=band_res.high,
            band_peak_r=band_res.peak_r,
            band_mask=band_res.band_mask,
            energy_thr=energy_thr,
            t_cand=t_cand,
            n_pv_band_windows=n_pv_band,
            n_bg_band_windows=n_bg_band,
            n_pv_cand_windows=n_pv_c,
            n_bg_cand_windows=n_bg_c,
        )

    # 5. Equal-frequency 6-bin histogram on merged candidate responses, merging duplicate edges
    a_all = np.concatenate((a_pv, a_bg))
    a_min = float(a_all.min())
    a_max = float(a_all.max())
    raw_edges = np.quantile(a_all, np.linspace(0.0, 1.0, cfg.num_bins + 1))
    edges = np.unique(raw_edges).astype(np.float64)
    b_prime = len(edges) - 1
    if b_prime < cfg.min_nonempty_bins:
        return FFTSoftVetoReference(
            active=False,
            reason="fewer_than_3_nonempty_bins",
            band_selected=True,
            band_low=band_res.low,
            band_high=band_res.high,
            band_peak_r=band_res.peak_r,
            band_mask=band_res.band_mask,
            energy_thr=energy_thr,
            t_cand=t_cand,
            a_min=a_min,
            a_max=a_max,
            n_pv_band_windows=n_pv_band,
            n_bg_band_windows=n_bg_band,
            n_pv_cand_windows=n_pv_c,
            n_bg_cand_windows=n_bg_c,
            num_merged_bins=max(0, b_prime),
        )

    # Assign bins: edges[0] <= v <= edges[-1]; inner boundaries edges[1:-1]
    inner = edges[1:-1]
    j_pv = np.searchsorted(inner, a_pv, side="right")
    j_bg = np.searchsorted(inner, a_bg, side="right")
    n_pv_bins = np.bincount(j_pv, minlength=b_prime).astype(np.int64)
    n_bg_bins = np.bincount(j_bg, minlength=b_prime).astype(np.int64)
    nonempty = int(((n_pv_bins + n_bg_bins) > 0).sum())
    if nonempty < cfg.min_nonempty_bins:
        return FFTSoftVetoReference(
            active=False,
            reason="fewer_than_3_nonempty_bins",
            band_selected=True,
            band_low=band_res.low,
            band_high=band_res.high,
            band_peak_r=band_res.peak_r,
            band_mask=band_res.band_mask,
            energy_thr=energy_thr,
            t_cand=t_cand,
            a_min=a_min,
            a_max=a_max,
            edges=edges,
            n_pv_bins=n_pv_bins,
            n_bg_bins=n_bg_bins,
            n_pv_band_windows=n_pv_band,
            n_bg_band_windows=n_bg_band,
            n_pv_cand_windows=n_pv_c,
            n_bg_cand_windows=n_bg_c,
            num_merged_bins=b_prime,
            num_nonempty_bins=nonempty,
        )

    # Add-1 smoothed class-conditional probabilities and log-likelihood ratio E_bg
    p_bg = (n_bg_bins + 1.0) / (float(n_bg_c) + float(b_prime))
    p_pv = (n_pv_bins + 1.0) / (float(n_pv_c) + float(b_prime))
    e_bg_bins = np.log(p_bg / p_pv).astype(np.float32)
    bin_eligible = ((n_pv_bins + n_bg_bins) >= cfg.min_bin_total) & (n_bg_bins >= cfg.min_bin_bg)

    return FFTSoftVetoReference(
        active=True,
        reason="ok",
        band_selected=True,
        band_low=band_res.low,
        band_high=band_res.high,
        band_peak_r=band_res.peak_r,
        band_mask=band_res.band_mask,
        energy_thr=energy_thr,
        t_cand=t_cand,
        a_min=a_min,
        a_max=a_max,
        edges=edges,
        n_pv_bins=n_pv_bins,
        n_bg_bins=n_bg_bins,
        e_bg_bins=e_bg_bins,
        bin_eligible=bin_eligible,
        n_pv_band_windows=n_pv_band,
        n_bg_band_windows=n_bg_band,
        n_pv_cand_windows=n_pv_c,
        n_bg_cand_windows=n_bg_c,
        num_merged_bins=b_prime,
        num_nonempty_bins=nonempty,
        num_eligible_bins=int(bin_eligible.sum()),
    )


def evaluate_fft_soft_veto_map(
    pre: ImageFFTPrecompute,
    s_deep_64: np.ndarray,
    ref: FFTSoftVetoReference,
    cfg: FFTSoftVetoConfig,
) -> dict[str, Any]:
    """Apply support-calibrated adaptive FFT soft veto to a [64, 64] deep score map."""
    hw = pre.grid_hw
    s_d = np.asarray(s_deep_64, dtype=np.float32).reshape(hw, hw)
    zeros = np.zeros((hw, hw), dtype=np.float32)
    if not (ref.active and ref.band_mask is not None and ref.edges is not None and ref.e_bg_bins is not None and ref.bin_eligible is not None):
        return {
            "delta_fft": zeros.reshape(-1),
            "g_mask": np.zeros((hw, hw), dtype=bool),
            "cand_mask": np.zeros((hw, hw), dtype=bool),
            "e_bg_map": zeros,
            "diagnostics": {
                "cand_tokens": 0,
                "g_active_tokens": 0,
                "penalized_tokens": 0,
                "penalty_rate_all": 0.0,
                "penalty_rate_cand": 0.0,
                "mean_penalty_all": 0.0,
                "mean_penalty_penalized": 0.0,
                "max_penalty": 0.0,
                "mean_e_bg_g_active": 0.0,
            },
        }

    band_key = (float(ref.band_low), float(ref.band_high))
    a_map = compute_adaptive_peak_share(pre, ref.band_mask, band_key=band_key)

    cand_mask = np.isfinite(s_d) & (s_d >= ref.t_cand)
    in_range = (a_map >= ref.a_min) & (a_map <= ref.a_max)
    inner = ref.edges[1:-1]
    j_map = np.searchsorted(inner, a_map, side="right")
    j_safe = np.clip(j_map, 0, len(ref.e_bg_bins) - 1)

    g_mask = (
        cand_mask
        & pre.interior
        & (pre.norm_energy >= ref.energy_thr)
        & in_range
        & ref.bin_eligible[j_safe]
    )
    e_bg_map = np.where(g_mask, ref.e_bg_bins[j_safe], 0.0).astype(np.float32)
    raw_pen = cfg.gamma * np.maximum(0.0, e_bg_map - cfg.tau)
    delta_fft = np.where(g_mask, np.minimum(cfg.cap, raw_pen), 0.0).astype(np.float32)

    pen_mask = delta_fft > 0.0
    n_cand = int(cand_mask.sum())
    n_g = int(g_mask.sum())
    n_pen = int(pen_mask.sum())
    return {
        "delta_fft": delta_fft.reshape(-1),
        "g_mask": g_mask,
        "cand_mask": cand_mask,
        "e_bg_map": e_bg_map,
        "diagnostics": {
            "cand_tokens": n_cand,
            "g_active_tokens": n_g,
            "penalized_tokens": n_pen,
            "penalty_rate_all": float(n_pen / float(hw * hw)),
            "penalty_rate_cand": float(n_pen / float(n_cand)) if n_cand > 0 else 0.0,
            "mean_penalty_all": float(delta_fft.mean()),
            "mean_penalty_penalized": float(delta_fft[pen_mask].mean()) if n_pen > 0 else 0.0,
            "max_penalty": float(delta_fft.max()) if n_pen > 0 else 0.0,
            "mean_e_bg_g_active": float(e_bg_map[g_mask].mean()) if n_g > 0 else 0.0,
        },
    }


def evaluate_k1_nonoverlap_bg_penalties(
    pre: ImageFFTPrecompute,
    cov_64: np.ndarray,
    s_deep_ex_64: np.ndarray,
    bg_flat_indices: np.ndarray,
    cfg: FFTSoftVetoConfig,
) -> np.ndarray:
    """Compute exact 1-shot calibration Delta_FFT on support BG tokens with zero FFT window overlap.

    For each scored background window at grid coordinate (r_x, c_x), EVERY support window with
    Chebyshev grid distance < k1_nonoverlap_grid_dist (4 -> 32px overlap) is excluded from:
      - energy floor Q_0.01,
      - adaptive band selection,
      - t_cand Q_0.85, and
      - 6-bin E_bg histogram counts.
    """
    n_bg = len(bg_flat_indices)
    out = np.zeros(n_bg, dtype=np.float32)
    if n_bg == 0 or not cfg.enabled:
        return out

    hw = pre.grid_hw
    dist = int(cfg.k1_nonoverlap_grid_dist)
    gr, gc = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")

    # Fast exact lower bound on t_cand: removing <= (2*dist-1)^2 = 49 windows out of N_bg_valid
    # can shift q85 by at most 49 rank positions downward; any token strictly below that rank
    # can NEVER satisfy s_deep >= t_cand_ex, so its penalty is guaranteed 0!
    base_bg = (
        pre.interior
        & (pre.norm_energy >= cfg.energy_floor_abs)
        & np.isfinite(cov_64)
        & (cov_64 <= cfg.bg_cov_max)
        & np.isfinite(s_deep_ex_64)
    )
    bg_scores_sorted = np.sort(s_deep_ex_64[base_bg])
    if len(bg_scores_sorted) == 0:
        return out
    max_removed = (2 * dist - 1) ** 2 + int(np.ceil(cfg.energy_quantile * int(pre.interior.sum()))) + 4
    safe_idx = max(0, int(np.floor(cfg.q_cand * len(bg_scores_sorted))) - max_removed)
    safe_s_floor = float(bg_scores_sorted[safe_idx])

    for pos, flat_i in enumerate(bg_flat_indices):
        rx = int(flat_i // hw)
        cx = int(flat_i % hw)
        if not pre.interior[rx, cx]:
            continue
        sx = float(s_deep_ex_64[rx, cx])
        if not np.isfinite(sx) or sx < safe_s_floor:
            continue

        # Strictly exclude all support windows whose 32x32 pixel box overlaps window (rx, cx)
        excl = (np.abs(gr - rx) < dist) & (np.abs(gc - cx) < dist)
        ref_ex = fit_fft_soft_veto_reference([pre], [cov_64], [s_deep_ex_64], cfg, exclude_mask_list=[excl])
        if not ref_ex.active or ref_ex.band_mask is None or ref_ex.edges is None or ref_ex.e_bg_bins is None or ref_ex.bin_eligible is None:
            continue
        if sx < ref_ex.t_cand or pre.norm_energy[rx, cx] < ref_ex.energy_thr:
            continue
        band_key = (float(ref_ex.band_low), float(ref_ex.band_high))
        a_val = float(compute_adaptive_peak_share(pre, ref_ex.band_mask, band_key=band_key)[rx, cx])
        if a_val < ref_ex.a_min or a_val > ref_ex.a_max:
            continue
        j = int(np.searchsorted(ref_ex.edges[1:-1], a_val, side="right"))
        j = min(max(j, 0), len(ref_ex.e_bg_bins) - 1)
        if not bool(ref_ex.bin_eligible[j]):
            continue
        e_bg = float(ref_ex.e_bg_bins[j])
        pen = min(cfg.cap, cfg.gamma * max(0.0, e_bg - cfg.tau))
        out[pos] = float(pen)

    return out
