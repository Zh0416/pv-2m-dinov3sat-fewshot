from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

# Static FFT bin tables cached per (device, window_size) to avoid recomputing trig grids.
_FFT_BIN_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_GABOR_BANK_CACHE: dict[str, torch.Tensor] = {}


def _get_fft_masks(device: torch.device, win: int = 32, n_theta: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), win)
    if key in _FFT_BIN_CACHE:
        return _FFT_BIN_CACHE[key]

    fy = torch.fft.fftfreq(win, d=1.0, device=device) * win
    fx = torch.fft.rfftfreq(win, d=1.0, device=device) * win
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(gx * gx + gy * gy)

    radial_bands = torch.stack(
        (
            (radius >= 1.0) & (radius < 3.0),
            (radius >= 3.0) & (radius < 9.0),
            (radius >= 9.0) & (radius < 15.0),
        ),
        dim=0,
    ).float()  # [3, win, win//2 + 1]

    # Map angles into [0, pi) and mask middle-to-high periodic band r in [2.5, 14.0]
    ang = torch.atan2(gy, gx) % math.pi
    ring = (radius >= 2.5) & (radius <= 14.0)
    ang_idx = torch.clamp((ang / math.pi * n_theta).long(), 0, n_theta - 1)
    sector_masks = torch.stack([(ring & (ang_idx == m)).float() for m in range(n_theta)], dim=0)  # [8, win, win//2+1]
    _FFT_BIN_CACHE[key] = (radial_bands, sector_masks)
    return radial_bands, sector_masks


def _get_gabor_bank(device: torch.device) -> torch.Tensor:
    """Build 12 quadrature (24 cos/sin) oriented Gabor filters for 2m/sub-meter PV grids."""
    key = str(device)
    if key in _GABOR_BANK_CACHE:
        return _GABOR_BANK_CACHE[key]

    ksize = 15
    half = ksize // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=torch.float32),
        torch.arange(-half, half + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    kernels = []
    # Scales cover wavelengths ~3.2px, 5.5px, 9.0px across 4 orientations
    for sigma, lambd in ((1.4, 3.2), (2.4, 5.5), (3.8, 9.0)):
        env = torch.exp(-(xs * xs + ys * ys) / (2.0 * sigma * sigma))
        for theta in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
            x_rot = xs * math.cos(theta) + ys * math.sin(theta)
            phase = 2.0 * math.pi * x_rot / lambd
            kc = env * torch.cos(phase)
            ks = env * torch.sin(phase)
            kc = kc - kc.mean()
            ks = ks - ks.mean()
            kc = kc / (kc.abs().sum() + 1e-6)
            ks = ks / (ks.abs().sum() + 1e-6)
            kernels.extend((kc, ks))
    bank = torch.stack(kernels, dim=0)[:, None, :, :]  # [24, 1, 15, 15]
    _GABOR_BANK_CACHE[key] = bank
    return bank


@torch.inference_mode()
def extract_freq_pyramid(rgb_uint8: np.ndarray, device: torch.device | str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 8-channel 512x512 + 64x64 explicit 2D-FFT/Gabor/Structure features for one RGB image.

    Channels (8):
      0..2: Normalized 2D-FFT radial bandpass energy fractions (low, PV-period, fine).
      3:    Directional orthogonal spectral periodicity C_spec in [0, 1].
      4:    Multi-scale quadrature Gabor mean amplitude (rescaled [0, 1]).
      5:    Gabor orientation anisotropy (max - min) / (max + min) in [0, 1].
      6:    Structure-tensor directional coherence A_coh in [0, 1].
      7:    Normalized Scharr gradient edge magnitude in [0, 1].
    """
    dev = torch.device(device if (str(device) == "cuda" and torch.cuda.is_available()) else "cpu")
    if isinstance(rgb_uint8, torch.Tensor):
        rgb_t = rgb_uint8.to(dev).float()
    else:
        rgb_t = torch.from_numpy(rgb_uint8.copy()).to(dev).float()
    if rgb_t.max() > 1.5:
        rgb_t = rgb_t / 255.0
    if rgb_t.ndim == 3:
        rgb_t = rgb_t[None]  # [1, 3, 512, 512]

    luma = 0.299 * rgb_t[:, 0:1] + 0.587 * rgb_t[:, 1:2] + 0.114 * rgb_t[:, 2:3]  # [1, 1, 512, 512]

    # 1. Local sliding-window 2D FFT spectrum on 32x32 Hann windows (stride 8 -> 61x61, padded to 64x64)
    win, stride = 32, 8
    padded = F.pad(luma, (12, 12, 12, 12), mode="reflect")  # 536 -> (536-32)/8 + 1 = 64 exact!
    patches = padded.unfold(2, win, stride).unfold(3, win, stride).squeeze(0).squeeze(0)  # [64, 64, 32, 32]
    patches = patches - patches.mean(dim=(-2, -1), keepdim=True)
    var = patches.var(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-5
    hann_1d = torch.hann_window(win, periodic=False, device=dev)
    hann_2d = hann_1d[:, None] * hann_1d[None, :]
    win_patches = patches * hann_2d[None, None]
    spec = torch.fft.rfft2(win_patches)
    power = (spec.real * spec.real + spec.imag * spec.imag) / ((win * win) * var)  # [64, 64, 32, 17]

    radial_masks, sector_masks = _get_fft_masks(dev, win=win, n_theta=8)
    band_e = torch.einsum("hwxy,kxy->khw", power, radial_masks)  # [3, 64, 64]
    band_sum = band_e.sum(dim=0, keepdim=True) + 1e-6
    band_frac = band_e / band_sum  # [3, 64, 64]

    sec_e = torch.einsum("hwxy,mxy->mhw", power, sector_masks)  # [8, 64, 64]
    sec_prob = sec_e / (sec_e.sum(dim=0, keepdim=True) + 1e-6)
    entropy = -(sec_prob * torch.log(sec_prob.clamp_min(1e-6))).sum(dim=0) / math.log(8.0)  # [64, 64]
    orth_e = sec_e[:4] + sec_e[4:]  # [4, 64, 64] orthogonal direction pairs
    peak_ratio = orth_e.max(dim=0).values / (orth_e.mean(dim=0) + 1e-6)  # in [1, 4]
    c_spec_64 = ((peak_ratio - 1.0) / 3.0).clamp(0.0, 1.0) * (1.0 - entropy).clamp(0.0, 1.0)

    # 2. Pixel-resolution 512x512 quadrature Gabor response
    bank = _get_gabor_bank(dev)
    g_out = F.conv2d(luma, bank, padding=7).reshape(1, 3, 4, 2, 512, 512)
    g_amp = torch.sqrt(g_out[:, :, :, 0] ** 2 + g_out[:, :, :, 1] ** 2 + 1e-8)  # [1, 3, 4, 512, 512]
    g_mean_ori = g_amp.mean(dim=1).squeeze(0)  # [4, 512, 512]
    g_amp_mean = g_mean_ori.mean(dim=0)
    g_scale = torch.quantile(g_amp_mean.reshape(-1)[::16], 0.99).clamp_min(1e-4)
    gabor_mean_512 = (g_amp_mean / g_scale).clamp(0.0, 1.5)
    gabor_ani_512 = (
        (g_mean_ori.max(dim=0).values - g_mean_ori.min(dim=0).values)
        / (g_mean_ori.max(dim=0).values + g_mean_ori.min(dim=0).values + 1e-5)
    ).clamp(0.0, 1.0)

    # 3. Scharr structure tensor coherence + edge magnitude at 512x512
    kx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], device=dev) / 16.0
    ky = kx.T
    gx = F.conv2d(luma, kx[None, None], padding=1)
    gy = F.conv2d(luma, ky[None, None], padding=1)
    jxx = F.avg_pool2d(gx * gx, kernel_size=5, stride=1, padding=2)
    jyy = F.avg_pool2d(gy * gy, kernel_size=5, stride=1, padding=2)
    jxy = F.avg_pool2d(gx * gy, kernel_size=5, stride=1, padding=2)
    trace = jxx + jyy + 1e-6
    coh_512 = (torch.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy + 1e-10) / trace).squeeze(0).squeeze(0).clamp(0.0, 1.0)

    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(0).squeeze(0)
    grad_scale = torch.quantile(grad_mag.reshape(-1)[::16], 0.99).clamp_min(1e-4)
    edge_512 = (grad_mag / grad_scale).clamp(0.0, 1.5)

    fft_4_64 = torch.cat((band_frac, c_spec_64[None]), dim=0)  # [4, 64, 64]
    fft_4_512 = F.interpolate(fft_4_64[None], size=(512, 512), mode="bilinear", align_corners=False)[0]
    fine_4_512 = torch.stack((gabor_mean_512, gabor_ani_512, coh_512, edge_512), dim=0)  # [4, 512, 512]
    freq_512 = torch.cat((fft_4_512, fine_4_512), dim=0).cpu()
    freq_64 = F.avg_pool2d(freq_512[None], kernel_size=8, stride=8)[0]  # [8, 64, 64]
    freq_64[:4] = fft_4_64.cpu()
    return freq_512.half(), freq_64.float()


def compute_support_freq_alignment(
    freq_64_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    query_ids: list[str],
) -> dict[str, torch.Tensor]:
    """Compute support-calibrated frequency-structure match z_freq ([64, 64]) per query image."""
    fg_rad, bg_rad = [], []
    fg_struct, bg_struct = [], []

    for sid in support_ids:
        f64 = freq_64_by_id[sid]  # [8, 64, 64] on target device
        m64 = torch.from_numpy((mask_by_id[sid].astype(np.float32).reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.50)).to(f64.device)
        b64 = torch.from_numpy((mask_by_id[sid].astype(np.float32).reshape(64, 8, 64, 8).mean(axis=(1, 3)) == 0.0)).to(f64.device)
        if m64.any():
            fg_rad.append(f64[:3, m64].mean(dim=1))
            fg_struct.append(f64[[3, 5, 6], :][:, m64])
        if b64.any():
            bg_rad.append(f64[:3, b64].mean(dim=1))
            bg_struct.append(f64[[3, 5, 6], :][:, b64])

    if fg_rad and bg_rad:
        diff_rad = torch.stack(fg_rad).mean(dim=0) - torch.stack(bg_rad).mean(dim=0)
        w_rad = diff_rad / (diff_rad.norm() + 1e-6)
    else:
        w_rad = torch.tensor([-0.2, 0.9, 0.4], device=next(iter(freq_64_by_id.values())).device)

    def _raw_freq(f64: torch.Tensor) -> torch.Tensor:
        rad_score = (f64[:3] * w_rad[:, None, None]).sum(dim=0)  # support-adaptive FFT bandpass match
        struct_score = 0.45 * f64[3] + 0.30 * f64[5] + 0.25 * f64[6]  # C_spec + Gabor ani + Coh
        return 0.45 * rad_score + 0.55 * struct_score

    # Anchor scale against support fg/bg raw frequency distribution
    fg_vals, bg_vals = [], []
    for sid in support_ids:
        rf = _raw_freq(freq_64_by_id[sid])
        m64 = torch.from_numpy((mask_by_id[sid].astype(np.float32).reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.50)).to(rf.device)
        b64 = torch.from_numpy((mask_by_id[sid].astype(np.float32).reshape(64, 8, 64, 8).mean(axis=(1, 3)) == 0.0)).to(rf.device)
        if m64.any():
            fg_vals.append(rf[m64])
        if b64.any():
            bg_vals.append(rf[b64])

    mu_fg = float(torch.median(torch.cat(fg_vals)).item()) if fg_vals else 0.40
    mu_bg = float(torch.quantile(torch.cat(bg_vals), 0.70).item()) if bg_vals else 0.25
    mid_f = 0.5 * (mu_fg + mu_bg)
    scale_f = max(0.08, mu_fg - mu_bg)

    out = {}
    for qid in query_ids:
        rf = _raw_freq(freq_64_by_id[qid])
        out[qid] = ((rf - mid_f) / scale_f).clamp(-2.5, 2.5)
    return out


def apply_ofg_neg_gate_and_rescue(
    z_sem_64: torch.Tensor,
    z_freq_64: torch.Tensor,
    *,
    w_freq_rescue: float = 0.35,
    w_freq_veto: float = 0.35,
    scene_gate_thr: float = 0.35,
    scene_gate_temp: float = 0.30,
) -> torch.Tensor:
    """Apply Module 2 conjunction spectral-semantic verification & negative-scene attenuation."""
    z_s = z_sem_64.reshape(-1)
    z_f = z_freq_64.reshape(-1)

    # Asymmetric local update: rescue oriented periodic PV regions; gently veto negative-freq clutter
    pos_rescue = w_freq_rescue * F.relu(z_f - 0.05) * torch.sigmoid((z_s + 0.65) / 0.45)
    neg_veto = w_freq_veto * F.relu(-z_f - 0.25)
    z_comb = z_s + pos_rescue - neg_veto

    # Dual-domain conjunction check on top-12 peak tokens (spares real PV while attenuating weak non-periodic FPs)
    top_idx = z_s.topk(min(12, z_s.numel())).indices
    sem_peak = float(z_s[top_idx[: min(8, top_idx.numel())]].mean().item())
    freq_cons = float(z_f[top_idx].mean().clamp(-1.5, 1.5).item())

    sem_gate = 1.0 / (1.0 + math.exp(-(sem_peak - scene_gate_thr) / max(1e-4, scene_gate_temp)))
    freq_gate = 1.0 / (1.0 + math.exp(-(freq_cons - (-0.25)) / 0.25))
    joint_gate = sem_gate * (0.45 + 0.55 * freq_gate)
    if joint_gate < 0.80:
        pos_mask = z_comb > 0.0
        z_comb = torch.where(pos_mask, z_comb * joint_gate - (1.0 - joint_gate) * 0.55, z_comb)
    return z_comb.reshape(z_sem_64.shape)
