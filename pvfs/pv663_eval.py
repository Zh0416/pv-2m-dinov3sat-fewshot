from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import importlib
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
import torch
from torch.nn import functional as F

from .data import area_target_32
from .dinov3_backbone import SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from .fast_bank import select_fg_bg_indices
from .io import read_json, write_json
from .iter3_methods import _to_tokens_norm
from .metrics import BOX_3, DISK_2, average_precision, paired_site_bootstrap
from .modules import ThreeModulePipeline, get_b0_32_config, get_m3_pruned_config
from .modules.deformable_boundary import DeformableBoundaryConfig, refine_mask_deformable_grad
from .phase_boundary import DISK_1, DISK_8
from .phase_boundary_eval import evaluate_binary_mask_512

# Module-level shared read-only state for Linux fork workers (zero-copy RGB lookup)
_FORK_RGB_512_BY_ID: dict[str, np.ndarray] = {}
_FORK_DEF_CFG: DeformableBoundaryConfig | None = None
_FORK_ALLOW_OUTWARD: bool = False
_FORK_P_FLOOR: float | None = None
_FORK_TAU_DEFORM: float = 0.20


def _deform_worker_task(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray, dict[str, float]]:
    qid, m0, p0_f16 = args
    p0 = p0_f16.astype(np.float32)
    rgb_hwc = _FORK_RGB_512_BY_ID[qid]
    assert _FORK_DEF_CFG is not None
    out = refine_mask_deformable_grad(
        m0,
        rgb_hwc,
        p0,
        allow_outward=_FORK_ALLOW_OUTWARD,
        p_floor=_FORK_P_FLOOR,
        tau_grad_per_px=_FORK_TAU_DEFORM,
        cfg=_FORK_DEF_CFG,
    )
    diag = {
        "total_arc": float(out["total_arc_len_px"]),
        "moved_arc": float(out["moved_arc_len_px"]),
        "net_area": float(out["net_area_delta_px"]),
        "abs_d_wsum": float(out["mean_abs_d_moved"] * out["moved_arc_len_px"]),
        "topo_reverts": float(out["topology_reverts"]),
    }
    return qid, out["mask"], diag


def resolve_pv663_dirs(
    image_dir: Path | None = None,
    mask_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Locate pv663 images/ and mask/ directories cleanly across Kaggle and local disks."""
    probe_img = "PV08_312621_1197085.bmp"
    probe_msk = "PV08_312621_1197085_label.bmp"

    if image_dir is not None:
        m_dir = mask_dir if mask_dir is not None else image_dir.parent / "mask"
        if (image_dir / probe_img).is_file() and (m_dir / probe_msk).is_file():
            return image_dir, m_dir
        raise FileNotFoundError(f"Configured pv663 dirs missing probe files: {image_dir}, {m_dir}")

    candidates = [
        Path("/kaggle/input/datasets/zh0416/pv-dinov3sat-663-v1/pv663/images"),
        Path("/kaggle/input/pv-dinov3sat-663-v1/pv663/images"),
        Path(r"E:\PV-DINOV3SAT\datasets\images"),
    ]
    for img_root in candidates:
        msk_root = img_root.parent / "mask"
        if (img_root / probe_img).is_file() and (msk_root / probe_msk).is_file():
            return img_root, msk_root

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = [
            p.parent
            for p in kaggle_input.rglob(probe_img)
            if p.is_file() and (p.parent.parent / "mask" / probe_msk).is_file()
        ]
        if len(matches) == 1:
            return matches[0], matches[0].parent / "mask"

    raise FileNotFoundError("Could not locate pv663 images/ and mask/ directories")


def load_pv663_pair(
    image_dir: Path,
    mask_dir: Path,
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load native 1024x1024 RGB CHW, 512x512 RGB HWC, and binary 512x512 mask (handling label value 12)."""
    img_path = image_dir / row["id"]
    mask_name = Path(row["mask_rel"]).name if "mask_rel" in row else f"{Path(row['id']).stem}_label.bmp"
    mask_path = mask_dir / mask_name

    pil_img = Image.open(img_path).convert("RGB")
    if pil_img.size != (1024, 1024):
        pil_img = pil_img.resize((1024, 1024), Image.Resampling.BILINEAR)
    rgb_1024_hwc = np.asarray(pil_img, dtype=np.uint8)
    rgb_1024_chw = np.transpose(rgb_1024_hwc, (2, 0, 1)).copy()
    rgb_512_hwc = np.asarray(pil_img.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)

    pil_msk = Image.open(mask_path)
    arr = np.asarray(pil_msk)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (1024, 1024):
        pil_msk = Image.fromarray(arr).resize((1024, 1024), Image.Resampling.NEAREST)
        arr = np.asarray(pil_msk)
    mask_1024 = arr > 0
    mask_512 = (mask_1024.astype(np.float32).reshape(512, 2, 512, 2).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
    if mask_1024.any() and not mask_512.any():
        mask_512 = mask_1024.reshape(512, 2, 512, 2).any(axis=(1, 3)).astype(np.uint8)
    return rgb_1024_chw, rgb_512_hwc, mask_512


def _feat_file(feat_dir: Path, image_id: str) -> Path:
    return feat_dir / f"{Path(image_id).stem}.pt"


def extract_pv663_features_to_dir(
    records: list[dict[str, Any]],
    *,
    image_dir: Path,
    mask_dir: Path,
    feat_dir: Path,
    source_dir: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract frozen DINOv3-SAT deep_32 (512) + mid_64/deep_64 (native 1024) to disk-backed fp16 files."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    rgb_512_by_id: dict[str, np.ndarray] = {}
    mask_512_by_id: dict[str, np.ndarray] = {}

    missing_recs: list[dict[str, Any]] = []
    for row in records:
        rid = row["id"]
        if _feat_file(feat_dir, rid).is_file():
            _, rgb_512, mask_512 = load_pv663_pair(image_dir, mask_dir, row)
            rgb_512_by_id[rid] = rgb_512
            mask_512_by_id[rid] = mask_512
        else:
            missing_recs.append(row)

    if not missing_recs:
        print(f"[PV663 Cache] All {len(records)} feature files already cached in {feat_dir}", flush=True)
        return rgb_512_by_id, mask_512_by_id

    weights = resolve_dinov3_weights(weights_path)
    source = prepare_dinov3_source(source_dir)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    backbones = importlib.import_module("dinov3.hub.backbones")

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = backbones.dinov3_vitl16(pretrained=True, weights=str(weights.resolve()))
    model.to(run_device).eval().requires_grad_(False)

    mean_t = torch.tensor(SAT_MEAN, device=run_device, dtype=torch.float32)[None, :, None, None]
    std_t = torch.tensor(SAT_STD, device=run_device, dtype=torch.float32)[None, :, None, None]
    layers_32 = [20, 21, 22, 23]
    layers_64 = [12, 13, 14, 15, 20, 21, 22, 23]

    t0 = time.perf_counter()
    with torch.inference_mode():
        for idx, row in enumerate(missing_recs, 1):
            rid = row["id"]
            rgb_1024_chw, rgb_512_hwc, mask_512 = load_pv663_pair(image_dir, mask_dir, row)
            rgb_512_by_id[rid] = rgb_512_hwc
            mask_512_by_id[rid] = mask_512

            x1024 = (torch.from_numpy(rgb_1024_chw)[None].to(run_device).float() / 255.0 - mean_t) / std_t
            x512 = F.interpolate(x1024, size=(512, 512), mode="bilinear", align_corners=False)

            outs_32 = model.get_intermediate_layers(x512, n=layers_32, reshape=True, norm=True)
            d32 = F.normalize(torch.stack([t[0].float() for t in outs_32], dim=0).mean(dim=0), dim=0)

            outs_64 = model.get_intermediate_layers(x1024, n=layers_64, reshape=True, norm=True)
            m64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4)], dim=0).mean(dim=0), dim=0)
            d64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4, 8)], dim=0).mean(dim=0), dim=0)

            payload = {"deep_32": d32.half().cpu(), "mid_64": m64.half().cpu(), "deep_64": d64.half().cpu()}
            tmp = _feat_file(feat_dir, rid).with_suffix(".tmp.pt")
            torch.save(payload, tmp)
            tmp.replace(_feat_file(feat_dir, rid))

            if idx % 50 == 0 or idx == len(missing_recs):
                elapsed = time.perf_counter() - t0
                rate = idx / max(1e-6, elapsed)
                eta = (len(missing_recs) - idx) / max(1e-6, rate)
                print(
                    f"[PV663 Extract] {idx}/{len(missing_recs)} ({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m)",
                    flush=True,
                )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rgb_512_by_id, mask_512_by_id


class LazyDiskTokenMap:
    """Mapping interface yielding normalized [N_q, C] float32 tokens on target device with tiny memory footprint."""

    def __init__(
        self,
        feat_dir: Path,
        key: str,
        device: str,
        preload_ids: list[str] | None = None,
    ) -> None:
        self.feat_dir = feat_dir
        self.key = key
        self.device = device
        self._pinned: dict[str, torch.Tensor] = {}
        self._last_id: str | None = None
        self._last_tok: torch.Tensor | None = None
        if preload_ids:
            for rid in preload_ids:
                self._pinned[rid] = self._load_one(rid)

    def _load_one(self, rid: str) -> torch.Tensor:
        d = torch.load(_feat_file(self.feat_dir, rid), map_location="cpu", weights_only=False)
        return _to_tokens_norm(d[self.key], self.device)

    def __contains__(self, rid: object) -> bool:
        if not isinstance(rid, str):
            return False
        return rid in self._pinned or _feat_file(self.feat_dir, rid).is_file()

    def __getitem__(self, rid: str) -> torch.Tensor:
        if rid in self._pinned:
            return self._pinned[rid]
        if self._last_id == rid and self._last_tok is not None:
            return self._last_tok
        tok = self._load_one(rid)
        self._last_id = rid
        self._last_tok = tok
        return tok


@torch.inference_mode()
def build_pairwise_topk_cache_streaming(
    feat_dir: Path,
    key: str,
    mask_by_id: dict[str, np.ndarray],
    all_ids: list[str],
    pos_ids: list[str],
    *,
    grid_hw: int,
    core_purity: float = 0.85,
    top_m: int = 3,
    device: str = "cuda",
    chunk_imgs: int = 8,
) -> dict[str, Any]:
    """Memory-safe streaming equivalent to pvfs.fast_bank.build_pairwise_topk_cache.

    Produces identical fp16 top-m similarity caches without stacking all 663 query images on GPU at once.
    """
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    mm_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    n_q = grid_hw * grid_hw
    n_pos = len(pos_ids)
    n_all = len(all_ids)
    id_to_all = {rid: i for i, rid in enumerate(all_ids)}
    id_to_pos = {rid: i for i, rid in enumerate(pos_ids)}

    fg_core_top = torch.full((n_pos, n_all, n_q, top_m), -2.0, dtype=torch.float16, device=dev)
    fg_all_top = torch.full((n_pos, n_all, n_q, top_m), -2.0, dtype=torch.float16, device=dev)
    bg_top = torch.full((n_pos, n_all, n_q, top_m), -2.0, dtype=torch.float16, device=dev)

    # Pre-extract support banks (only n_pos ~ 58 images, tiny!)
    fg_core_masks: list[np.ndarray] = []
    fg_all_masks: list[np.ndarray] = []
    bg_masks: list[np.ndarray] = []
    fg_core_banks: list[torch.Tensor | None] = []
    fg_all_banks: list[torch.Tensor | None] = []
    bg_banks: list[torch.Tensor | None] = []

    for sid in pos_ids:
        d = torch.load(_feat_file(feat_dir, sid), map_location="cpu", weights_only=False)
        tok_s = _to_tokens_norm(d[key], str(dev)).to(mm_dtype)
        m512 = mask_by_id[sid]
        fg_c, bg_m, _ = select_fg_bg_indices(m512, grid_hw, core_purity)
        fg_a, _, _ = select_fg_bg_indices(m512, grid_hw, 0.50)
        fg_core_masks.append(fg_c)
        fg_all_masks.append(fg_a)
        bg_masks.append(bg_m)

        def _bank(mask_arr: np.ndarray) -> torch.Tensor | None:
            idx = np.flatnonzero(mask_arr)
            if len(idx) == 0:
                return None
            return tok_s[torch.from_numpy(idx).to(dev)]

        fg_core_banks.append(_bank(fg_c))
        fg_all_banks.append(_bank(fg_a))
        bg_banks.append(_bank(bg_m) if bg_m.any() else None)

    def _fill_chunk(q_mm: torch.Tensor, bank: torch.Tensor | None, dst: torch.Tensor, s_i: int, start: int, end: int) -> None:
        if bank is None:
            return
        m_take = min(top_m, bank.shape[0])
        sim = (q_mm @ bank.T).float().clamp(-1.0, 1.0)
        vals = sim.topk(m_take, dim=1).values.reshape(end - start, n_q, m_take)
        if m_take < top_m:
            pad = vals.new_full((end - start, n_q, top_m - m_take), -2.0)
            vals = torch.cat((vals, pad), dim=2)
        dst[s_i, start:end] = vals.to(torch.float16)

    for start in range(0, n_all, chunk_imgs):
        end = min(n_all, start + chunk_imgs)
        q_list = []
        for rid in all_ids[start:end]:
            d = torch.load(_feat_file(feat_dir, rid), map_location="cpu", weights_only=False)
            q_list.append(_to_tokens_norm(d[key], str(dev)).to(mm_dtype))
        q_mm = torch.stack(q_list, dim=0).reshape((end - start) * n_q, -1)
        for s_i in range(n_pos):
            _fill_chunk(q_mm, fg_core_banks[s_i], fg_core_top, s_i, start, end)
            _fill_chunk(q_mm, fg_all_banks[s_i], fg_all_top, s_i, start, end)
            _fill_chunk(q_mm, bg_banks[s_i], bg_top, s_i, start, end)

    return {
        "grid_hw": grid_hw,
        "top_m": top_m,
        "id_to_all": id_to_all,
        "id_to_pos": id_to_pos,
        "fg_core_top": fg_core_top,
        "fg_all_top": fg_all_top,
        "bg_top": bg_top,
        "fg_core_masks": fg_core_masks,
        "fg_all_masks": fg_all_masks,
        "bg_masks": bg_masks,
    }


def _calibrate_p_floor_loo(
    support_ids: list[str],
    mask_by_id: dict[str, np.ndarray],
    loo_probs: dict[str, np.ndarray] | None,
) -> tuple[bool, float | None]:
    """Exact same outward semantic floor calibration rule as Module ③ locked Dev experiment."""
    if len(support_ids) == 1 or loo_probs is None:
        return False, None
    missed_vals, edge_vals = [], []
    for sid in support_ids:
        gt = np.asarray(mask_by_id[sid] > 0, dtype=bool)
        if not gt.any() or sid not in loo_probs:
            continue
        p_loo = loo_probs[sid]
        inner = gt & (~binary_erosion(gt, structure=DISK_8, border_value=0))
        if not inner.any():
            inner = gt
        missed = inner & (p_loo < 0.50)
        if int(missed.sum()) >= 16:
            missed_vals.append(p_loo[missed])
        edge_vals.append(p_loo[inner])
    pool = missed_vals if missed_vals else edge_vals
    if not pool:
        return True, 0.35
    q = float(np.quantile(np.concatenate(pool), 0.25))
    return True, float(np.clip(q, 0.20, 0.48))


def evaluate_pv663_episode(
    query_records: list[dict[str, Any]],
    masks_512: dict[str, np.ndarray],
    probs_512_ref: dict[str, np.ndarray],
    scores_32_ref: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate one episode on PV663 query images and aggregate to site rows."""
    per_image = {}
    by_site: dict[str, list[tuple[dict, dict, np.ndarray, np.ndarray]]] = defaultdict(list)
    for rec in query_records:
        qid = rec["id"]
        if rec["positive"] and "_gt_b" not in rec:
            truth = rec["mask"] > 0
            rec["_gt_b"] = truth ^ binary_erosion(truth, structure=BOX_3, border_value=1)
            rec["_gt_dil_1px"] = binary_dilation(rec["_gt_b"], structure=DISK_1)
            rec["_gt_dil_2px"] = binary_dilation(rec["_gt_b"], structure=DISK_2)
        im_m = evaluate_binary_mask_512(
            rec["mask"],
            masks_512[qid],
            probs_512_ref[qid],
            t_b=rec.get("_gt_b"),
            t_dil_1px=rec.get("_gt_dil_1px"),
            t_dil_2px=rec.get("_gt_dil_2px"),
        )
        im_m["id"] = qid
        im_m["site"] = rec["site"]
        im_m["target_fraction"] = float(rec["target_fraction"])
        per_image[qid] = im_m
        by_site[rec["site"]].append((rec, im_m, probs_512_ref[qid], scores_32_ref[qid]))

    site_rows = []
    for site, items in sorted(by_site.items()):
        pos_items = [x for x in items if x[0]["positive"]]
        row: dict[str, Any] = {
            "site": site,
            "positive_images": len(pos_items),
            "negative_images": len(items) - len(pos_items),
        }
        if pos_items:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected"):
                row[key] = float(np.mean([x[1][key] for x in pos_items]))
            row["ap_512"] = average_precision(
                np.concatenate([(x[0]["mask"] > 0).ravel() for x in pos_items]),
                np.concatenate([x[2].ravel() for x in pos_items]),
            )
            row["ap_32"] = average_precision(
                np.concatenate([(x[0]["y32"] >= 0.25).ravel() for x in pos_items]),
                np.concatenate([x[3].ravel() for x in pos_items]),
            )
        site_rows.append(row)
    return {"per_image": per_image, "site_rows": site_rows}


def aggregate_pv663_episodes(episode_results: list[dict[str, Any]], expected_episodes: int) -> dict[str, Any]:
    """Aggregate episodes to independent 5km geographic site level, then macro + 4 size strata."""
    site_ep_vals: dict[str, list[dict]] = defaultdict(list)
    img_ep_vals: dict[str, list[dict]] = defaultdict(list)
    for ep_res in episode_results:
        for sr in ep_res["site_rows"]:
            site_ep_vals[sr["site"]].append(sr)
        for qid, im in ep_res["per_image"].items():
            img_ep_vals[qid].append(im)

    pos_keys = ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected", "ap_512", "ap_32")
    site_summary: dict[str, dict[str, Any]] = {}
    for site, rows in sorted(site_ep_vals.items()):
        if len(rows) != expected_episodes:
            raise ValueError(f"Site {site} appeared in {len(rows)} episodes, expected {expected_episodes}")
        entry: dict[str, Any] = {
            "site": site,
            "positive_images": rows[0]["positive_images"],
            "negative_images": rows[0]["negative_images"],
        }
        for key in pos_keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        site_summary[site] = entry

    pos_sites = [v for v in site_summary.values() if v["positive_images"] > 0]

    image_summary: dict[str, dict[str, Any]] = {}
    for qid, rows in sorted(img_ep_vals.items()):
        r0 = rows[0]
        item: dict[str, Any] = {
            "id": qid,
            "site": r0["site"],
            "positive": r0["positive"],
            "target_fraction": r0["target_fraction"],
        }
        for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected"):
            item[key] = float(np.mean([r[key] for r in rows]))
        image_summary[qid] = item

    pos_imgs = list(image_summary.values())
    strata: dict[str, dict[str, Any]] = {}
    for label, cond in (
        ("tiny_lt1pct", lambda v: v["target_fraction"] < 0.01),
        ("small_lt5pct", lambda v: v["target_fraction"] < 0.05),
        ("mid_5_15pct", lambda v: 0.05 <= v["target_fraction"] < 0.15),
        ("large_ge15pct", lambda v: v["target_fraction"] >= 0.15),
    ):
        sub = [v for v in pos_imgs if cond(v)]
        if sub:
            strata[label] = {
                "count": len(sub),
                "dice": float(np.mean([v["dice"] for v in sub])),
                "iou": float(np.mean([v["iou"] for v in sub])),
                "recall": float(np.mean([v["recall"] for v in sub])),
                "precision": float(np.mean([v["precision"] for v in sub])),
                "bf1_1px": float(np.mean([v["bf1_1px"] for v in sub])),
                "boundary_f1": float(np.mean([v["boundary_f1"] for v in sub])),
                "detected": float(np.mean([v["detected"] for v in sub])),
            }

    macro = {
        "n_pos_sites": len(pos_sites),
        "n_query_images": len(pos_imgs),
        **{k: float(np.mean([v[k] for v in pos_sites])) for k in pos_keys},
    }
    return {
        "macro": macro,
        "strata": strata,
        "sites": site_summary,
        "images": image_summary,
    }


def compare_pv663_methods(
    cand: dict[str, Any],
    ref: dict[str, Any],
    *,
    n_boot: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Paired site-level bootstrap comparison across all 116 independent PV663 query sites."""
    site_ids = sorted(s for s, v in cand["sites"].items() if v["positive_images"] > 0)
    out: dict[str, Any] = {}
    for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "bf1_1px", "boundary_f1", "detected", "ap_512"):
        c_map = {s: cand["sites"][s][key] for s in site_ids}
        r_map = {s: ref["sites"][s][key] for s in site_ids}
        diffs = [c_map[s] - r_map[s] for s in site_ids]
        boot = paired_site_bootstrap(c_map, r_map, seed=seed, draws=n_boot)
        out[key] = {
            "mean_delta": boot["mean_delta"],
            "ci95": boot["ci95"],
            "improved_sites": int(sum(1 for d in diffs if (d < 0 if key == "pos_bg_fp" else d > 0))),
            "total_sites": len(site_ids),
        }
    out["small_lt5pct_recall_delta"] = float(
        cand["strata"].get("small_lt5pct", {}).get("recall", 0.0)
        - ref["strata"].get("small_lt5pct", {}).get("recall", 0.0)
    )
    out["small_lt5pct_dice_delta"] = float(
        cand["strata"].get("small_lt5pct", {}).get("dice", 0.0)
        - ref["strata"].get("small_lt5pct", {}).get("dice", 0.0)
    )
    return out


def _save_pv663_panel(
    out_dir: Path,
    rgb_512_by_id: dict[str, np.ndarray],
    mask_512_by_id: dict[str, np.ndarray],
    panel_votes: dict[str, dict[str, np.ndarray]],
    query_records: list[dict[str, Any]],
) -> Path | None:
    """Save a 4-row representative K=5 visual comparison panel across size strata."""
    strata_pick = []
    for cond in (
        lambda r: r["target_fraction"] < 0.01,
        lambda r: 0.01 <= r["target_fraction"] < 0.05,
        lambda r: 0.05 <= r["target_fraction"] < 0.15,
        lambda r: r["target_fraction"] >= 0.15,
    ):
        cands = sorted([r for r in query_records if cond(r)], key=lambda x: x["target_fraction"])
        if cands:
            strata_pick.append(cands[len(cands) // 2]["id"])
    if not strata_pick:
        return None

    cols = ("GT", "B0_32_Ref", "M3_Off_R", "Stage3_R_DeformGrad")
    short = ("GT", "B0_32_Ref", "M3_Pruned (R)", "Best (R+DeformGrad)")
    tile = 320
    canvas = Image.new("RGB", (tile * len(cols), tile * len(strata_pick)), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)

    for ri, qid in enumerate(strata_pick):
        rgb = rgb_512_by_id[qid].astype(np.float32)
        gt = mask_512_by_id[qid] > 0
        for ci, (cname, lbl) in enumerate(zip(cols, short)):
            base = rgb.copy()
            if cname == "GT":
                base[gt] = 0.50 * base[gt] + 0.50 * np.array([35, 220, 95], dtype=np.float32)
            else:
                pred = panel_votes[cname][qid] >= 0.5
                tp, fp, fn = gt & pred, (~gt) & pred, gt & (~pred)
                base[tp] = 0.50 * base[tp] + 0.50 * np.array([35, 220, 95], dtype=np.float32)
                base[fp] = 0.45 * base[fp] + 0.55 * np.array([240, 55, 55], dtype=np.float32)
                base[fn] = 0.45 * base[fn] + 0.55 * np.array([250, 205, 45], dtype=np.float32)
            im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.BILINEAR)
            canvas.paste(im, (ci * tile, ri * tile))
            draw.text((ci * tile + 6, ri * tile + 4), f"{qid} | {lbl}", fill=(255, 255, 255))

    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panels_dir / "k5_pv663_compare.png"
    canvas.save(panel_path)
    return panel_path


def format_pv663_report(summary: dict[str, Any]) -> str:
    sp = summary["split_summary"]
    lines = [
        "=== PV663 (zh0416/pv-dinov3sat-663-v1) 1/3/5-SHOT EVALUATION REPORT ===",
        f"Total={sp['total_images']} images / {sp['total_sites']} sites (5km) | "
        f"Support Pool={sp['support_pool']['total']} imgs ({sp['support_pool_sites']} sites) | "
        f"Query Set={sp['query_set']['total']} imgs ({sp['query_set_sites']} sites) | "
        f"Episodes/K={summary['episodes_per_k']} | Elapsed={summary['elapsed_sec']/60:.1f} min",
        "",
        "=== MACRO SITE-LEVEL METRICS (116 INDEPENDENT QUERY SITES) ===",
    ]
    for vname in ("B0_32_Ref", "M3_Off_R", "Stage3_R_DeformGrad"):
        lines.append(f"\n--- {vname} ---")
        for k in ("1", "3", "5"):
            if k not in summary["methods"][vname]:
                continue
            agg = summary["methods"][vname][k]
            m = agg["macro"]
            st = agg["strata"]
            t1 = st.get("tiny_lt1pct", {}).get("recall", 0.0)
            s5r = st.get("small_lt5pct", {}).get("recall", 0.0)
            s5d = st.get("small_lt5pct", {}).get("dice", 0.0)
            lines.append(
                f"  K={k}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} PosBgFP={m['pos_bg_fp']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} Det={m['detected']:.4f} | "
                f"Sm<5D={s5d:.4f} Sm<5R={s5r:.4f} Tiny<1R={t1:.4f}"
            )

    lines.append("\n=== PAIRED SITE-LEVEL BOOTSTRAP COMPARISONS (10,000 RESAMPLES) ===")
    for cname, by_k in summary["comparisons"].items():
        lines.append(f"\n[{cname}]")
        for k in ("1", "3", "5"):
            if k not in by_k:
                continue
            c = by_k[k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} "
                f"({c['dice']['improved_sites']}/{c['dice']['total_sites']} sites) | "
                f"dIoU={c['iou']['mean_delta']:+.4f} | dAP512={c['ap_512']['mean_delta']:+.4f} | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f} | dPosBgFP={c['pos_bg_fp']['mean_delta']:+.4f}"
            )
    return "\n".join(lines) + "\n"


def run_pv663_evaluation_from_features(
    *,
    split_payload: dict[str, Any],
    episodes_payload: dict[str, Any],
    feat_dir: Path,
    rgb_512_by_id: dict[str, np.ndarray],
    mask_512_by_id: dict[str, np.ndarray],
    deform_config_json: Path,
    output_dir: Path,
    device: str = "cuda",
    num_workers: int = 4,
    tau_deform: float = 0.20,
    n_boot: int = 10000,
) -> dict[str, Any]:
    """Run B0_32_Ref, M3_Off_R, and Stage3_R_DeformGrad across locked PV663 1/3/5-shot episodes."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    rec_by_id = {r["id"]: r for r in split_payload["records"]}
    unique_sup_ids: list[str] = list(episodes_payload["unique_support_ids"])
    query_ids: list[str] = list(episodes_payload["query_ids"])
    needed_ids = sorted(set(unique_sup_ids) | set(query_ids))

    records_eval: dict[str, dict[str, Any]] = {}
    for rid in needed_ids:
        mask = mask_512_by_id[rid]
        records_eval[rid] = {**rec_by_id[rid], "mask": mask, "y32": area_target_32(mask)}

    print(
        f"[PV663] Building streaming top-3 pairwise caches for {len(needed_ids)} imgs x {len(unique_sup_ids)} supports on {run_device}...",
        flush=True,
    )
    caches: dict[str, dict[str, Any]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        caches[key] = build_pairwise_topk_cache_streaming(
            feat_dir,
            key,
            mask_512_by_id,
            needed_ids,
            unique_sup_ids,
            grid_hw=ghw,
            core_purity=0.85,
            top_m=3,
            device=run_device,
        )

    tokens_by_key = {
        key: LazyDiskTokenMap(feat_dir, key, run_device, preload_ids=unique_sup_ids)
        for key in ("deep_32", "deep_64", "mid_64")
    }

    pipe_b0 = ThreeModulePipeline(get_b0_32_config())
    pipe_r = ThreeModulePipeline(get_m3_pruned_config())
    def_cfg = DeformableBoundaryConfig.from_json(deform_config_json, mode="deformable_grad")

    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes_payload["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    variants = ("B0_32_Ref", "M3_Off_R", "Stage3_R_DeformGrad")
    methods_res: dict[str, dict[str, Any]] = {v: {} for v in variants}
    deform_diag_by_k: dict[str, dict[str, Any]] = {}
    panel_votes: dict[str, dict[str, np.ndarray]] = {
        v: defaultdict(lambda: np.zeros((512, 512), dtype=np.float32)) for v in variants
    }
    query_recs = [records_eval[qid] for qid in query_ids]

    use_fork_pool = (num_workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FORK_RGB_512_BY_ID, _FORK_DEF_CFG, _FORK_ALLOW_OUTWARD, _FORK_P_FLOOR, _FORK_TAU_DEFORM
    _FORK_RGB_512_BY_ID = rgb_512_by_id
    _FORK_DEF_CFG = def_cfg
    _FORK_TAU_DEFORM = float(tau_deform)

    for k in sorted(ep_by_k.keys()):
        k_t0 = time.perf_counter()
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
        move_acc: dict[str, float] = defaultdict(float)

        for ep_idx, ep in enumerate(eps, 1):
            sup_ids = ep["support_ids"]
            # 1. Baseline B0_32_Ref
            out_b0 = pipe_b0.run_episode(caches, tokens_by_key, mask_512_by_id, None, sup_ids, query_ids)
            p_b0, s32_b0 = out_b0.to_legacy_outputs()
            m_b0 = {qid: out_b0.boundary_by_id[qid].m_coarse for qid in query_ids}

            # 2. Best upstream R (M3_Pruned_Ref)
            out_r = pipe_r.run_episode(caches, tokens_by_key, mask_512_by_id, None, sup_ids, query_ids)
            p_r, s32_r = out_r.to_legacy_outputs()
            m_r = {qid: out_r.boundary_by_id[qid].m_coarse for qid in query_ids}

            # 3. Module ③ LOO semantic floor calibration + Deformable DP Grad refinement
            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for s_i in sup_ids:
                    rem = [s for s in sup_ids if s != s_i]
                    o_l = pipe_r.run_episode(caches, tokens_by_key, mask_512_by_id, None, rem, [s_i])
                    loo_probs[s_i] = o_l.boundary_by_id[s_i].p_coarse
            allow_out, p_floor = _calibrate_p_floor_loo(sup_ids, mask_512_by_id, loo_probs)

            m_def: dict[str, np.ndarray] = {}
            if use_fork_pool:
                _FORK_ALLOW_OUTWARD = allow_out
                _FORK_P_FLOOR = p_floor
                tasks = [(qid, m_r[qid], p_r[qid].astype(np.float16)) for qid in query_ids]
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
                    for qid, mask_out, diag in ex.map(_deform_worker_task, tasks, chunksize=8):
                        m_def[qid] = mask_out
                        for dk, dv in diag.items():
                            move_acc[dk] += dv
            else:
                for qid in query_ids:
                    d_out = refine_mask_deformable_grad(
                        m_r[qid],
                        rgb_512_by_id[qid],
                        p_r[qid],
                        allow_outward=allow_out,
                        p_floor=p_floor,
                        tau_grad_per_px=tau_deform,
                        cfg=def_cfg,
                    )
                    m_def[qid] = d_out["mask"]
                    move_acc["total_arc"] += float(d_out["total_arc_len_px"])
                    move_acc["moved_arc"] += float(d_out["moved_arc_len_px"])
                    move_acc["net_area"] += float(d_out["net_area_delta_px"])
                    move_acc["abs_d_wsum"] += float(d_out["mean_abs_d_moved"] * d_out["moved_arc_len_px"])
                    move_acc["topo_reverts"] += float(d_out["topology_reverts"])

            ep_evals["B0_32_Ref"].append(evaluate_pv663_episode(query_recs, m_b0, p_b0, s32_b0))
            ep_evals["M3_Off_R"].append(evaluate_pv663_episode(query_recs, m_r, p_r, s32_r))
            ep_evals["Stage3_R_DeformGrad"].append(evaluate_pv663_episode(query_recs, m_def, p_r, s32_r))

            if k == 5:
                w = 1.0 / len(eps)
                for qid in query_ids:
                    panel_votes["B0_32_Ref"][qid] += m_b0[qid].astype(np.float32) * w
                    panel_votes["M3_Off_R"][qid] += m_r[qid].astype(np.float32) * w
                    panel_votes["Stage3_R_DeformGrad"][qid] += m_def[qid].astype(np.float32) * w

            print(
                f"[PV663 K={k}] Episode {ep_idx}/{len(eps)} done ({(time.perf_counter()-k_t0)/60:.1f}m) "
                f"| allow_out={allow_out} p_floor={p_floor}",
                flush=True,
            )

        for v in variants:
            methods_res[v][str(k)] = aggregate_pv663_episodes(ep_evals[v], expected_episodes=len(eps))
        deform_diag_by_k[str(k)] = {
            "moved_arc_fraction": float(move_acc["moved_arc"] / max(1.0, move_acc["total_arc"])),
            "mean_abs_d_px": float(move_acc["abs_d_wsum"] / max(1.0, move_acc["moved_arc"])),
            "mean_net_area_delta_px": float(move_acc["net_area"] / max(1, len(eps) * len(query_ids))),
            "topology_reverts": int(move_acc["topo_reverts"]),
        }
        for v in variants:
            m = methods_res[v][str(k)]["macro"]
            sm = methods_res[v][str(k)]["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            print(
                f"[PV663 K={k} SUMMARY] {v:20s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"AP512={m['ap_512']:.4f} BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} Sm<5R={sm:.4f}",
                flush=True,
            )

    comparisons = {}
    for cname, cand_k, ref_k in (
        ("Stage3_R_DeformGrad_vs_B0_32_Ref", "Stage3_R_DeformGrad", "B0_32_Ref"),
        ("Stage3_R_DeformGrad_vs_M3_Off_R", "Stage3_R_DeformGrad", "M3_Off_R"),
        ("M3_Off_R_vs_B0_32_Ref", "M3_Off_R", "B0_32_Ref"),
    ):
        comparisons[cname] = {
            str(k): compare_pv663_methods(methods_res[cand_k][str(k)], methods_res[ref_k][str(k)], n_boot=n_boot)
            for k in sorted(ep_by_k.keys())
        }

    if 5 in ep_by_k:
        _save_pv663_panel(output_dir, rgb_512_by_id, mask_512_by_id, panel_votes, query_recs)

    summary = {
        "schema": "pv-2m-dinov3sat-pv663-eval-v1",
        "dataset": "zh0416/pv-dinov3sat-663-v1",
        "device": run_device,
        "elapsed_sec": float(time.perf_counter() - t0),
        "episodes_per_k": int(episodes_payload["episodes_per_k"]),
        "split_summary": split_payload["summary"],
        "min_support_query_distance_km": split_payload["min_support_query_distance_km"],
        "deform_diagnostics_by_k": deform_diag_by_k,
        "comparisons": comparisons,
        "methods": methods_res,
    }
    write_json(output_dir / "pv663_eval_summary.json", summary)
    report_txt = format_pv663_report(summary)
    (output_dir / "pv663_report.txt").write_text(report_txt, encoding="utf-8")
    print("\n" + report_txt, flush=True)
    return summary


def run_pv663_benchmark(
    *,
    split_json: Path,
    episodes_json: Path,
    deform_config_json: Path,
    source_dir: Path,
    feat_cache_dir: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    mask_dir: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    """End-to-end PV663 1/3/5-shot benchmark: feature extraction + streaming evaluation + report."""
    split_payload = read_json(split_json)
    episodes_payload = read_json(episodes_json)
    img_dir, msk_dir = resolve_pv663_dirs(image_dir, mask_dir)

    needed_set = set(episodes_payload["unique_support_ids"]) | set(episodes_payload["query_ids"])
    needed_recs = [r for r in split_payload["records"] if r["id"] in needed_set]

    rgb_512_by_id, mask_512_by_id = extract_pv663_features_to_dir(
        needed_recs,
        image_dir=img_dir,
        mask_dir=msk_dir,
        feat_dir=feat_cache_dir,
        source_dir=source_dir,
        weights_path=weights_path,
        device=device,
    )
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))
    return run_pv663_evaluation_from_features(
        split_payload=split_payload,
        episodes_payload=episodes_payload,
        feat_dir=feat_cache_dir,
        rgb_512_by_id=rgb_512_by_id,
        mask_512_by_id=mask_512_by_id,
        deform_config_json=deform_config_json,
        output_dir=output_dir,
        device=device,
        num_workers=workers,
    )
