from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_baselines.finetune_models import DINOv3KShotFTRunner, EndToEndSegKShotRunner
from external_baselines.matcher_dinov3sat import MatcherDINOv3Runner
from external_baselines.persam_f import PerSAMFRunner
from external_baselines.run_pv100_ext_all import _save_ext_grid, _save_ext_pos_zoom_panel
from external_baselines.sam_utils import LazySamEmbedMap, extract_sam_embeddings_to_dir, load_sam_vit_l
from pvfs.data import area_target_32
from pvfs.dinov3_backbone import SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from pvfs.fast_bank import compute_dino_affinity_weights
from pvfs.io import write_json
from pvfs.metrics import evaluate_binary_mask_512
from pvfs.modules import PBDCConfig, ThreeModulePipeline, get_b0_32_config, get_clean_core_config, refine_mask_pbdc
from pvfs.pv663_eval import (
    LazyDiskTokenMap,
    aggregate_pv663_episodes,
    build_pairwise_topk_cache_streaming,
    compare_pv663_methods,
    evaluate_pv663_episode,
)

# Shared fork worker state for Ours (v3.1-PBDC-Pure)
_FW_RGB_BY_ID: dict[str, np.ndarray] = {}
_FW_PBDC_CFG: PBDCConfig | None = None
_FW_K_SHOTS: int = 5


def _pbdc_worker(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray]:
    qid, m0, p0_f16 = args
    assert _FW_PBDC_CFG is not None
    out = refine_mask_pbdc(m0, _FW_RGB_BY_ID[qid], p0_f16.astype(np.float32), k_shots=_FW_K_SHOTS, cfg=_FW_PBDC_CFG)
    return qid, out["mask"]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(min(1.0, a)))


def resolve_pv01_root(explicit_root: Path | None = None) -> Path:
    if explicit_root is not None and (explicit_root / "PV01_Rooftop_Brick").is_dir():
        return explicit_root
    candidates = [
        Path("/kaggle/input/datasets/louisdelignac/pv-segmentation-from-satellite-and-aerial-imagery/PV01"),
        Path("/kaggle/input/pv-segmentation-from-satellite-and-aerial-imagery/PV01"),
    ]
    for c in candidates:
        if (c / "PV01_Rooftop_Brick").is_dir():
            return c
    ki = Path("/kaggle/input")
    if ki.is_dir():
        for p in ki.rglob("PV01_Rooftop_Brick"):
            if p.is_dir():
                return p.parent
    raise FileNotFoundError("Could not locate PV01 dataset directory containing PV01_Rooftop_Brick")


def build_pv01_split_and_episodes(
    pv01_root: Path,
    cluster_km: float = 0.25,
    episodes_per_k: int = 5,
    seed: int = 20261127,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Scan all 645 PV01 (0.1m UAV) pairs, upsample 256->512, cluster by geo distance, and lock 20%/80% split."""
    subdirs = ("PV01_Rooftop_Brick", "PV01_Rooftop_FlatConcrete", "PV01_Rooftop_SteelTile")
    raw_items: list[dict[str, Any]] = []
    rgb_512_by_id: dict[str, np.ndarray] = {}
    mask_512_by_id: dict[str, np.ndarray] = {}

    for sd in subdirs:
        dpath = pv01_root / sd
        for img_path in sorted(dpath.glob("*.bmp")):
            if img_path.stem.endswith("_label"):
                continue
            lbl_path = dpath / f"{img_path.stem}_label.bmp"
            if not lbl_path.is_file():
                continue
            m = re.search(r"PV01_(\d+)_(\d+)", img_path.stem)
            if not m:
                continue
            lat = int(m.group(1)) / 10000.0
            lon = int(m.group(2)) / 10000.0
            roof = sd.replace("PV01_Rooftop_", "")
            rid = f"{roof}_{img_path.stem}"

            rgb_pil = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            msk_pil = Image.open(lbl_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
            rgb_arr = np.asarray(rgb_pil, dtype=np.uint8)
            msk_arr = (np.asarray(msk_pil, dtype=np.uint8) > 0).astype(np.uint8)
            tf = float(msk_arr.mean())
            if tf <= 0.0:
                continue  # skip any degenerate empty mask if present

            rgb_512_by_id[rid] = rgb_arr
            mask_512_by_id[rid] = msk_arr
            raw_items.append(
                {
                    "id": rid,
                    "roof_type": sd.replace("PV01_Rooftop_", ""),
                    "lat": lat,
                    "lon": lon,
                    "positive": True,
                    "target_fraction": tf,
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                }
            )

    n = len(raw_items)
    assert n > 0, f"No PV01 pairs found under {pv01_root}"

    # Union-Find geographic clustering at cluster_km (0.25 km = ~10x tile width for 0.1m 256px UAV)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        la1, lo1 = raw_items[i]["lat"], raw_items[i]["lon"]
        for j in range(i + 1, n):
            if _haversine_km(la1, lo1, raw_items[j]["lat"], raw_items[j]["lon"]) <= cluster_km:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    comp_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, item in enumerate(raw_items):
        comp_map[_find(i)].append(item)

    sites_info: list[dict[str, Any]] = []
    for c_idx, (_, items) in enumerate(sorted(comp_map.items(), key=lambda kv: (-len(kv[1]), kv[1][0]["id"]))):
        site_name = f"PV01_SITE_{c_idx:02d}"
        dom_roof = Counter(x["roof_type"] for x in items).most_common(1)[0][0]
        items_sorted = sorted(items, key=lambda x: (x["target_fraction"], x["id"]))
        rep = items_sorted[len(items_sorted) // 2]
        sites_info.append(
            {
                "site": site_name,
                "dom_roof": dom_roof,
                "count": len(items),
                "mean_tf": float(np.mean([x["target_fraction"] for x in items])),
                "items": items_sorted,
                "rep": rep,
            }
        )

    # Stratified 20% support sites / 80% query sites across the 3 rooftop types.
    # Avoid putting the single giant site (>150 imgs) in support so query stays ~80% of images.
    by_roof: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sites_info:
        by_roof[s["dom_roof"]].append(s)

    sup_site_names: set[str] = set()
    for roof, slist in sorted(by_roof.items()):
        # Sort by count ascending so moderate-sized sites serve as support and large clusters stay in query
        slist_sorted = sorted(slist, key=lambda x: (x["count"], x["site"]))
        n_sup = max(2, round(len(slist_sorted) * 0.25))
        n_sup = min(n_sup, max(1, len(slist_sorted) - 2))
        # Pick evenly spaced indices from the sorted list (skipping the single largest site)
        cand = slist_sorted[:-1] if len(slist_sorted) > 2 else slist_sorted
        step = len(cand) / float(n_sup)
        for j in range(n_sup):
            sup_site_names.add(cand[min(len(cand) - 1, int(j * step + step * 0.5))]["site"])

    records: list[dict[str, Any]] = []
    sup_reps: list[dict[str, Any]] = []
    sup_coords: list[tuple[float, float]] = []
    qry_coords: list[tuple[float, float]] = []

    for s in sites_info:
        part = "support_pool" if s["site"] in sup_site_names else "query_set"
        for it in s["items"]:
            rec = {
                "id": it["id"],
                "site": s["site"],
                "roof_type": it["roof_type"],
                "partition": part,
                "positive": True,
                "target_fraction": it["target_fraction"],
                "lat": it["lat"],
                "lon": it["lon"],
            }
            records.append(rec)
            if part == "support_pool":
                sup_coords.append((it["lat"], it["lon"]))
            else:
                qry_coords.append((it["lat"], it["lon"]))
        if part == "support_pool":
            sup_reps.append({**s["rep"], "site": s["site"]})

    min_dist_km = min(
        _haversine_km(la1, lo1, la2, lo2)
        for la1, lo1 in sup_coords
        for la2, lo2 in qry_coords
    )
    assert min_dist_km > cluster_km * 0.99, f"Geo leakage detected: min_dist={min_dist_km:.4f} km"
    assert len(sup_reps) >= 5, f"Need >=5 support sites for K=5, got {len(sup_reps)}"

    # Deterministic episodes: 5 episodes per K (K=1, 3, 5), each drawing K distinct support sites
    rng = np.random.default_rng(seed)
    sup_reps.sort(key=lambda x: x["id"])
    episodes: list[dict[str, Any]] = []
    for k in (1, 3, 5):
        for ep_i in range(episodes_per_k):
            perm = sorted(int(i) for i in rng.choice(len(sup_reps), size=k, replace=False))
            picked = [sup_reps[i] for i in perm]
            episodes.append(
                {
                    "k": k,
                    "episode": ep_i,
                    "support_ids": [x["id"] for x in picked],
                    "support_sites": [x["site"] for x in picked],
                }
            )

    split_lock = {
        "schema": "pv01-ext-all7-lock-v1",
        "cluster_km": cluster_km,
        "min_dist_km": float(min_dist_km),
        "total_images": len(records),
        "support_sites": len(sup_site_names),
        "support_images": sum(1 for r in records if r["partition"] == "support_pool"),
        "query_sites": len(sites_info) - len(sup_site_names),
        "query_images": sum(1 for r in records if r["partition"] == "query_set"),
        "episodes_per_k": episodes_per_k,
        "records": sorted(records, key=lambda x: x["id"]),
        "episodes": episodes,
    }
    return split_lock, rgb_512_by_id, mask_512_by_id


@torch.no_grad()
def extract_dinov3_to_dir(
    ids: list[str],
    rgb_512_by_id: dict[str, np.ndarray],
    *,
    feat_dir: Path,
    source_dir: Path,
    device: str,
) -> None:
    feat_dir.mkdir(parents=True, exist_ok=True)
    need = [rid for rid in ids if not (feat_dir / f"{rid}.pt").is_file()]
    if not need:
        return
    src = prepare_dinov3_source(source_dir)
    weights = resolve_dinov3_weights()
    model = torch.hub.load(str(src), "dinov3_vitl16", source="local", weights=str(weights))
    model = model.to(device).eval()

    mean = torch.tensor(SAT_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(SAT_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    t0 = time.perf_counter()
    for idx, rid in enumerate(need, 1):
        rgb = rgb_512_by_id[rid]
        x512 = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
        x512 = (x512 - mean) / std
        x1024 = F.interpolate(x512, size=(1024, 1024), mode="bilinear", align_corners=False)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            o32 = model.get_intermediate_layers(x512, n=[23], reshape=True, norm=True)[0][0]
            o64 = model.get_intermediate_layers(x1024, n=[15, 23], reshape=True, norm=True)
            mid64 = o64[0][0]
            deep64 = o64[1][0]

        torch.save(
            {
                "deep_32": o32.detach().to(torch.float16).cpu(),
                "mid_64": mid64.detach().to(torch.float16).cpu(),
                "deep_64": deep64.detach().to(torch.float16).cpu(),
            },
            feat_dir / f"{rid}.pt",
        )
        if idx % 100 == 0 or idx == len(need):
            print(f"[PV01 DINOv3 Extract] {idx}/{len(need)} ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_pv01_ext_report(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "=== PV01 (0.1m UAV Rooftop) EXTERNAL MODEL COMPARISON (7 Complete Models) ===",
        f"Split: {meta['support_sites']} support sites ({meta['support_images']} imgs, {meta['unique_support_used']} used) / "
        f"{meta['query_sites']} query sites ({meta['query_images']} imgs) | min_dist={meta['min_dist_km']:.2f} km",
        f"Episodes/K: {meta['episodes_per_k']} | Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
    ]
    order = (
        "A0_B0_32_Ref",
        "PerSAM_F_SAM_ViTL",
        "UNet_ResNet50_FT",
        "SegFormer_MiTB2_FT",
        "DINOv3SAT_ConvHead_FT",
        "Matcher_DINOv3SAT_SAM_ViTL",
        "Ours_v31_PBDC_Pure",
    )
    for k in ("1", "3", "5"):
        lines.append(f"\n--- K={k} ({meta['query_sites']} geo sites, {meta['query_images']} query imgs x {meta['episodes_per_k']} eps) ---")
        for mname in order:
            res = summary["methods"][mname][k]
            m = res["macro"]
            sm_r = res["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            lines.append(
                f"  {mname:28s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} Sm<5R={sm_r:.4f}"
            )

    if summary.get("k5_by_roof"):
        lines.append("\n--- K=5 Per-Rooftop-Material Breakdown (Image Macro Dice / BF1@2) ---")
        for roof, rdict in sorted(summary["k5_by_roof"].items()):
            row_s = " | ".join(
                f"{m.split('_')[0]}={rdict[m]['dice']:.4f}/{rdict[m]['bf1_2']:.4f}"
                for m in order
                if m in rdict
            )
            lines.append(f"  {roof:14s} (n={rdict['count']:3d}): {row_s}")

    lines.append(f"\n--- Paired Site Bootstrap Deltas ({meta['query_sites']} Geo Sites, Ours vs External / Baseline) ---")
    for mname in order[:-1]:
        cname = f"Ours_vs_{mname}"
        lines.append(f"[{cname}]")
        for k in ("1", "3", "5"):
            c = summary["comparisons"][cname][k]
            lines.append(
                f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} "
                f"({c['dice']['improved_sites']}/{c['dice']['total_sites']}) | "
                f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]} | "
                f"dSm<5R={c['small_lt5pct_recall_delta']:+.4f}"
            )
    return "\n".join(lines) + "\n"


def run_pv01_ext_all(
    *,
    pv01_root: Path | None,
    source_dir: Path,
    dino_feat_dir: Path,
    sam_cache_dir: Path,
    sam_embed_dir: Path,
    output_dir: Path,
    cluster_km: float = 0.25,
    episodes_per_k: int = 5,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    root = resolve_pv01_root(pv01_root)
    print(f"[PV01 Ext] Scanning & clustering PV01 at {root} (cluster_km={cluster_km})...", flush=True)
    sp, rgb_512_by_id, mask_512_by_id = build_pv01_split_and_episodes(
        root, cluster_km=cluster_km, episodes_per_k=episodes_per_k
    )
    write_json(output_dir / "pv01_split_and_episodes_lock.json", sp)
    print(
        f"[PV01 Ext] Locked split: {sp['support_sites']} support sites ({sp['support_images']} imgs) / "
        f"{sp['query_sites']} query sites ({sp['query_images']} imgs), min_dist={sp['min_dist_km']:.2f} km",
        flush=True,
    )

    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in sp["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    rec_by_id = {r["id"]: r for r in sp["records"]}
    query_recs_raw = [r for r in sp["records"] if r["partition"] == "query_set"]
    query_ids = [r["id"] for r in query_recs_raw]
    unique_sup_ids = sorted({sid for eps in ep_by_k.values() for ep in eps for sid in ep["support_ids"]})
    used_ids = unique_sup_ids + query_ids

    query_records = [
        {**r, "mask": mask_512_by_id[r["id"]], "y32": area_target_32(mask_512_by_id[r["id"]])}
        for r in query_recs_raw
    ]

    # 1. Extract DINOv3-SAT features + SAM ViT-L embeddings for used_ids
    extract_dinov3_to_dir(used_ids, rgb_512_by_id, feat_dir=dino_feat_dir, source_dir=source_dir, device=run_device)
    sam = load_sam_vit_l(sam_cache_dir, device=run_device)
    extract_sam_embeddings_to_dir(sam, rgb_512_by_id, used_ids, sam_embed_dir, device=run_device)
    sam_embeds = LazySamEmbedMap(sam_embed_dir, device=run_device, preload_ids=unique_sup_ids)

    # 2. Build DINOv3-SAT streaming pairwise caches + affinity weights
    tokens_by_key = {
        k: LazyDiskTokenMap(dino_feat_dir, k, run_device, preload_ids=unique_sup_ids)
        for k in ("deep_32", "deep_64", "mid_64")
    }
    caches = {
        k: build_pairwise_topk_cache_streaming(
            dino_feat_dir, k, mask_512_by_id, used_ids, unique_sup_ids, grid_hw=ghw, core_purity=0.85, top_m=3, device=run_device
        )
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }
    core_cfg = get_clean_core_config()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in query_ids}

    pipe_a0 = ThreeModulePipeline(get_b0_32_config(), refiner_cfg=None)
    pipe_a3 = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    pbdc_cfg = PBDCConfig(enabled=True)
    persam_runner = PerSAMFRunner(sam, ft_steps=100, lr=0.05)
    matcher_runner = MatcherDINOv3Runner(sam, num_clusters=8, merge_margin=0.05)
    dino_ft = DINOv3KShotFTRunner(steps=120, lr=3e-3, wd=1e-2)
    unet_ft = EndToEndSegKShotRunner("unet_resnet50", steps=80, lr=3e-4, wd=1e-2, device=run_device)
    segf_ft = EndToEndSegKShotRunner("segformer_mit_b2", steps=80, lr=3e-4, wd=1e-2, device=run_device)

    model_names = (
        "A0_B0_32_Ref",
        "PerSAM_F_SAM_ViTL",
        "Matcher_DINOv3SAT_SAM_ViTL",
        "UNet_ResNet50_FT",
        "SegFormer_MiTB2_FT",
        "DINOv3SAT_ConvHead_FT",
        "Ours_v31_PBDC_Pure",
    )
    methods_res: dict[str, dict[str, Any]] = defaultdict(dict)
    vis_k5_ep0: dict[str, dict[str, np.ndarray]] = {}
    vis_k5_prob_ours: dict[str, np.ndarray] = {}

    use_fork = (workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    global _FW_RGB_BY_ID, _FW_PBDC_CFG, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_512_by_id
    _FW_PBDC_CFG = pbdc_cfg

    for k in (1, 3, 5):
        t_k = time.perf_counter()
        eps = ep_by_k[k]
        ep_evals: dict[str, list[dict]] = {m: [] for m in model_names}

        for ep_idx, ep in enumerate(eps, 1):
            sup_ids = ep["support_ids"]
            seed_off = k * 100 + ep_idx

            # 1. A0
            out_a0 = pipe_a0.run_episode(caches, tokens_by_key, mask_512_by_id, aff_by_id, sup_ids, query_ids)
            p_a0, s32_a0 = out_a0.to_legacy_outputs()
            m_a0 = out_a0.coarse_masks_512()
            ep_evals["A0_B0_32_Ref"].append(evaluate_pv663_episode(query_records, m_a0, p_a0, s32_a0))

            # 2. PerSAM-F
            ps_st = persam_runner.fit_episode(sam_embeds, mask_512_by_id, sup_ids)
            m_ps, p_ps, s32_ps = {}, {}, {}
            for qid in query_ids:
                m_ps[qid], p_ps[qid], s32_ps[qid] = persam_runner.infer_query(sam_embeds[qid], ps_st)
            ep_evals["PerSAM_F_SAM_ViTL"].append(evaluate_pv663_episode(query_records, m_ps, p_ps, s32_ps))

            # 3. Matcher (DINOv3-SAT + SAM)
            mt_st = matcher_runner.fit_episode(tokens_by_key["deep_64"], mask_512_by_id, sup_ids)
            m_mt, p_mt, s32_mt = {}, {}, {}
            for qid in query_ids:
                m_mt[qid], p_mt[qid], s32_mt[qid] = matcher_runner.infer_query(tokens_by_key["deep_64"][qid], sam_embeds[qid], mt_st)
            ep_evals["Matcher_DINOv3SAT_SAM_ViTL"].append(evaluate_pv663_episode(query_records, m_mt, p_mt, s32_mt))

            # 4. DINOv3-SAT + ConvHead FT
            head = dino_ft.fit_episode(tokens_by_key, mask_512_by_id, sup_ids, ep_seed_offset=seed_off)
            m_df, p_df, s32_df = dino_ft.infer_queries(head, tokens_by_key, query_ids)
            ep_evals["DINOv3SAT_ConvHead_FT"].append(evaluate_pv663_episode(query_records, m_df, p_df, s32_df))
            del head

            # 5. U-Net (ResNet50) FT
            u_mod = unet_ft.fit_episode(rgb_512_by_id, mask_512_by_id, sup_ids, ep_seed_offset=seed_off)
            m_un, p_un, s32_un = unet_ft.infer_queries(u_mod, rgb_512_by_id, query_ids)
            ep_evals["UNet_ResNet50_FT"].append(evaluate_pv663_episode(query_records, m_un, p_un, s32_un))
            del u_mod

            # 6. SegFormer (MiT-B2) FT
            s_mod = segf_ft.fit_episode(rgb_512_by_id, mask_512_by_id, sup_ids, ep_seed_offset=seed_off)
            m_sf, p_sf, s32_sf = segf_ft.infer_queries(s_mod, rgb_512_by_id, query_ids)
            ep_evals["SegFormer_MiTB2_FT"].append(evaluate_pv663_episode(query_records, m_sf, p_sf, s32_sf))
            del s_mod

            # 7. Ours (v3.1-PBDC-Pure)
            out_a3 = pipe_a3.run_episode(caches, tokens_by_key, mask_512_by_id, aff_by_id, sup_ids, query_ids)
            p_a3, s32_a3 = out_a3.to_legacy_outputs()
            m_a3 = out_a3.coarse_masks_512()
            _FW_K_SHOTS = k
            tasks = [(qid, m_a3[qid], p_a3[qid].astype(np.float16)) for qid in query_ids]
            m_ours: dict[str, np.ndarray] = {}
            if use_fork:
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    for qid, m_ref in ex.map(_pbdc_worker, tasks, chunksize=8):
                        m_ours[qid] = m_ref
            else:
                for task in tasks:
                    qid, m_ref = _pbdc_worker(task)
                    m_ours[qid] = m_ref
            ep_evals["Ours_v31_PBDC_Pure"].append(evaluate_pv663_episode(query_records, m_ours, p_a3, s32_a3))

            if k == 5 and ep_idx == 1:
                vis_k5_ep0 = {"A0": m_a0, "Matcher": m_mt, "Ours": m_ours}
                vis_k5_prob_ours = p_a3

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[PV01 Ext K={k}] Ep {ep_idx}/{len(eps)} done ({(time.perf_counter()-t_k)/60:.1f}m)", flush=True)

        for mname in model_names:
            methods_res[mname][str(k)] = aggregate_pv663_episodes(ep_evals[mname], expected_episodes=len(eps))
            m = methods_res[mname][str(k)]["macro"]
            print(f"  -> [K={k}] {mname:28s}: Dice={m['dice']:.4f} BF1@2={m['boundary_f1']:.4f}", flush=True)

    # Per-rooftop-material breakdown at K=5
    k5_by_roof: dict[str, dict[str, Any]] = {}
    for roof in ("Brick", "FlatConcrete", "SteelTile"):
        rids = [r["id"] for r in query_recs_raw if r["roof_type"] == roof]
        if not rids:
            continue
        entry: dict[str, Any] = {"count": len(rids)}
        for mname in model_names:
            imgs = methods_res[mname]["5"]["images"]
            entry[mname] = {
                "dice": float(np.mean([imgs[rid]["dice"] for rid in rids])),
                "bf1_2": float(np.mean([imgs[rid]["boundary_f1"] for rid in rids])),
            }
        k5_by_roof[roof] = entry

    comparisons = {
        f"Ours_vs_{mname}": {
            str(k): compare_pv663_methods(methods_res["Ours_v31_PBDC_Pure"][str(k)], methods_res[mname][str(k)])
            for k in (1, 3, 5)
        }
        for mname in model_names
        if mname != "Ours_v31_PBDC_Pure"
    }

    summary = {
        "schema": "pv01-ext-all7-v1",
        "elapsed_sec": float(time.perf_counter() - t0),
        "meta": {
            "support_sites": sp["support_sites"],
            "support_images": sp["support_images"],
            "unique_support_used": len(unique_sup_ids),
            "query_sites": sp["query_sites"],
            "query_images": sp["query_images"],
            "min_dist_km": sp["min_dist_km"],
            "episodes_per_k": episodes_per_k,
        },
        "methods": dict(methods_res),
        "k5_by_roof": k5_by_roof,
        "comparisons": comparisons,
    }
    write_json(output_dir / "pv01_ext_all7_summary.json", summary)
    report = format_pv01_ext_report(summary)
    (output_dir / "pv01_ext_all7_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)

    # Render 6 diverse rooftop comparison panels (2 Brick + 2 FlatConcrete + 2 SteelTile) + 4x zoom panels
    print("[PV01 Ext] Rendering 6 rooftop comparison panels (A0 vs DINOv3-SAT+SAM vs Ours)...", flush=True)
    vis_dir = output_dir / "visuals"
    zoom_dir = vis_dir / "zoom"
    zoom_dir.mkdir(parents=True, exist_ok=True)

    chosen: list[dict[str, Any]] = []
    for roof in ("Brick", "FlatConcrete", "SteelTile"):
        r_recs = [r for r in query_recs_raw if r["roof_type"] == roof]
        scored = []
        for r in r_recs:
            rid = r["id"]
            gt = mask_512_by_id[rid]
            ms = {"A0": vis_k5_ep0["A0"][rid], "Matcher": vis_k5_ep0["Matcher"][rid], "Ours": vis_k5_ep0["Ours"][rid]}
            mets = {km: evaluate_binary_mask_512(gt, m, vis_k5_prob_ours[rid]) for km, m in ms.items()}
            scored.append(
                {
                    "id": rid,
                    "short_id": rid,
                    "positive": True,
                    "target_pct": float((gt > 0).mean()) * 100.0,
                    "rgb": rgb_512_by_id[rid],
                    "gt": gt,
                    "masks": ms,
                    "metrics": mets,
                    "gain": mets["Ours"]["dice"] - mets["Matcher"]["dice"],
                }
            )
        scored.sort(key=lambda x: -x["gain"])
        # Pick 1 where Ours shines most + 1 median-area typical scene
        if scored:
            chosen.append(scored[0])
        if len(scored) > 2:
            mid = sorted(scored[1:], key=lambda x: abs(x["target_pct"] - 12.0))[0]
            chosen.append(mid)

    _save_ext_grid(
        vis_dir / "summary_pv01_ext_6scenes_grid.png",
        "PV01 (0.1m UAV Rooftop, K=5) Comparison: A0 vs DINOv3-SAT+SAM vs Ours (v3.1-PBDC)",
        chosen[:6],
    )
    for idx, sc in enumerate(chosen[:6], 1):
        _save_ext_pos_zoom_panel(zoom_dir / f"pv01_{idx:02d}_{sc['short_id']}.png", sc)

    print(f"[PV01 Ext] Complete in {(time.perf_counter()-t0)/60:.1f} min!", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PV01 (0.1m UAV) 7-model external comparison + visuals.")
    parser.add_argument("--pv01-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--dino-feat-dir", type=Path, default=Path("/tmp/pv01_ext_dino_feats"))
    parser.add_argument("--sam-cache-dir", type=Path, default=Path("/tmp/sam_weights"))
    parser.add_argument("--sam-embed-dir", type=Path, default=Path("/tmp/pv01_ext_sam_embeds"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "external_baselines" / "results" / "pv01_all7")
    parser.add_argument("--cluster-km", type=float, default=0.25)
    parser.add_argument("--episodes-per-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run_pv01_ext_all(
        pv01_root=args.pv01_root,
        source_dir=args.source_dir,
        dino_feat_dir=args.dino_feat_dir,
        sam_cache_dir=args.sam_cache_dir,
        sam_embed_dir=args.sam_embed_dir,
        output_dir=args.output_dir,
        cluster_km=args.cluster_km,
        episodes_per_k=args.episodes_per_k,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
