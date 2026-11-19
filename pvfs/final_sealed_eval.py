from __future__ import annotations

from collections import defaultdict
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import rasterio
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, resolve_data_dirs
from .dinov3_backbone import SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from .fast_bank import build_pairwise_topk_cache
from .io import read_json, sha256_file, write_json
from .iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from .iter5_methods import ITER5_VARIANTS, Iter5Config, run_iter5_episode
from .metrics import evaluate_episode, paired_site_bootstrap

SEALED_VARIANT_NAMES = ("B0_32_Ref", "M1_Full_Ref", "M5_BalCal", "M5_TightRescue", "M3_Pruned_Ref")
SEALED_VARIANTS: tuple[Iter5Config, ...] = tuple(
    cfg for cfg in ITER5_VARIANTS if cfg.name in SEALED_VARIANT_NAMES
)


def load_sealed_test_records(split_dir: Path) -> tuple[dict, dict, list[dict]]:
    """Load the 21 strictly sealed test records for ONE-SHOT post-lock evaluation."""
    manifest = read_json(split_dir / "dataset_manifest.json")
    split = read_json(split_dir / "split_lock.json")
    if split.get("status") != "locked_after_manual_qc" or not manifest.get("manual_qc_confirmed"):
        raise PermissionError("Locked manual QC split is required")

    assignment = split["assignment"]
    by_id = {row["id"]: row for row in manifest["images"]}
    if set(by_id) != set(assignment):
        raise ValueError("Manifest and split IDs mismatch")

    test_records = []
    for image_id in sorted(by_id):
        info = assignment[image_id]
        if info["partition"] != "test":
            continue
        if info.get("fold") is not None:
            raise ValueError(f"Expected fold=None for sealed test {image_id}, got {info}")
        row = by_id[image_id]
        test_records.append(
            {
                "id": image_id,
                "site": info["site"],
                "fold": None,
                "partition": "test",
                "positive": bool(row["positive"]),
                "target_fraction": float(row["target_fraction"]),
                "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"],
                "meters_per_pixel": list(row["meters_per_pixel"]),
            }
        )
    if len(test_records) != 21:
        raise ValueError(f"Expected 21 sealed test images, found {len(test_records)}")
    return manifest, split, test_records


def read_sealed_rgb_and_mask(image_dir: Path, label_dir: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read and SHA-256 verify a sealed test RGB image and binary PV mask."""
    if row.get("partition") != "test":
        raise PermissionError("read_sealed_rgb_and_mask is only for sealed test records")
    img_path = image_dir / row["id"]
    lbl_path = label_dir / row["id"]
    if sha256_file(img_path) != row["image_sha256"]:
        raise ValueError(f"Test image SHA-256 mismatch for {row['id']}")
    if sha256_file(lbl_path) != row["label_sha256"]:
        raise ValueError(f"Test label SHA-256 mismatch for {row['id']}")
    with rasterio.open(img_path) as src:
        rgb = src.read()
    with rasterio.open(lbl_path) as src:
        mask = src.read(1)
    if rgb.shape != (3, 512, 512) or rgb.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB shape/dtype for {row['id']}: {rgb.shape}, {rgb.dtype}")
    if mask.shape != (512, 512):
        raise ValueError(f"Unexpected mask shape for {row['id']}: {mask.shape}")
    if not np.array_equal(np.unique(mask), np.unique(mask.astype(np.uint8))) or not set(np.unique(mask).tolist()).issubset({0, 1}):
        raise ValueError(f"Mask for {row['id']} is not binary 0/1")
    return rgb, mask.astype(np.uint8)


def build_sealed_test_episodes(
    dev_records: list[dict],
    test_records: list[dict],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    num_episodes: int = 40,
    seed_base: int = 20261200,
) -> dict[str, list[dict]]:
    """Build deterministic K-shot episodes sampling dev positive sites to query all 21 sealed test images."""
    pos_by_site: dict[str, list[dict]] = defaultdict(list)
    for r in dev_records:
        if r["partition"] != "dev":
            raise PermissionError("Support pool must contain strictly dev records")
        if r["positive"]:
            pos_by_site[r["site"]].append(r)

    sites = sorted(pos_by_site)
    query_ids = [r["id"] for r in test_records]
    test_sites = {r["site"] for r in test_records}
    if test_sites & set(sites):
        raise ValueError(f"Site leakage between dev support and sealed test: {test_sites & set(sites)}")

    episodes_by_k: dict[str, list[dict]] = {}
    for k in ks:
        eps = []
        for ep in range(num_episodes):
            rng = random.Random(seed_base + 1000 * k + ep)
            chosen_sites = rng.sample(sites, k)
            support_ids = [rng.choice(pos_by_site[s])["id"] for s in chosen_sites]
            eps.append(
                {
                    "k": int(k),
                    "episode": int(ep),
                    "support_sites": chosen_sites,
                    "support_ids": support_ids,
                    "query_ids": query_ids,
                }
            )
        episodes_by_k[str(k)] = eps
    return episodes_by_k


def extract_sealed_test_features(
    test_records: list[dict],
    *,
    image_dir: Path,
    label_dir: Path,
    source_dir: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract pure frozen DINOv3-SAT deep_32, mid_64, deep_64 features + verified masks for 21 test images."""
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
    layers_32 = (20, 21, 22, 23)
    layers_64 = (12, 13, 14, 15, 20, 21, 22, 23)

    feats: dict[str, dict[str, torch.Tensor]] = {}
    rgbs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for idx, row in enumerate(test_records, 1):
            rgb, mask = read_sealed_rgb_and_mask(image_dir, label_dir, row)
            rgbs[row["id"]] = rgb
            masks[row["id"]] = mask

            x512 = (torch.from_numpy(rgb.copy())[None].to(run_device).float() / 255.0 - mean_t) / std_t
            outs_32 = model.get_intermediate_layers(x512, n=list(layers_32), reshape=True, norm=True)
            d32 = F.normalize(torch.stack([t[0].float() for t in outs_32], dim=0).mean(dim=0), dim=0)

            x1024 = F.interpolate(x512, size=(1024, 1024), mode="bilinear", align_corners=False)
            outs_64 = model.get_intermediate_layers(x1024, n=list(layers_64), reshape=True, norm=True)
            m64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4)], dim=0).mean(dim=0), dim=0)
            d64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4, 8)], dim=0).mean(dim=0), dim=0)

            feats[row["id"]] = {
                "deep_32": d32.half().cpu(),
                "mid_64": m64.half().cpu(),
                "deep_64": d64.half().cpu(),
            }
            if idx % 7 == 0 or idx == len(test_records):
                print(f"[SealedTest] Extracted frozen DINOv3-SAT features {idx}/{len(test_records)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats, rgbs, masks


def aggregate_sealed_episodes(episode_results: list[dict], expected_episodes: int = 40) -> dict:
    """Aggregate 40 test episodes at the independent 19-site unit (15 pos sites, 4 neg sites)."""
    site_ep_vals: dict[str, list[dict]] = defaultdict(list)
    img_ep_vals: dict[str, list[dict]] = defaultdict(list)
    for ep_res in episode_results:
        for sr in ep_res["site_rows"]:
            site_ep_vals[sr["site"]].append(sr)
        for qid, im in ep_res["per_image"].items():
            img_ep_vals[qid].append(im)

    site_summary: dict[str, dict] = {}
    pos_keys = ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected", "ap_512", "ap_32")
    neg_keys = ("neg_fp", "neg_clean", "neg_max_prob")

    for site, rows in sorted(site_ep_vals.items()):
        if len(rows) != expected_episodes:
            raise ValueError(f"Test site {site} appeared in {len(rows)} episodes, expected {expected_episodes}")
        entry = {
            "site": site,
            "fold": None,
            "positive_images": rows[0]["positive_images"],
            "negative_images": rows[0]["negative_images"],
        }
        for key in pos_keys:
            vals = [r[key] for r in rows if r[key] is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        for key in neg_keys:
            vals = [r[key] for r in rows if r[key] is not None]
            entry[key] = float(np.mean(vals)) if vals else None
        site_summary[site] = entry

    pos_sites = [v for v in site_summary.values() if v["positive_images"] > 0]
    neg_sites = [v for v in site_summary.values() if v["negative_images"] > 0]

    image_summary = {}
    for qid, rows in sorted(img_ep_vals.items()):
        r0 = rows[0]
        item = {
            "id": qid,
            "site": r0["site"],
            "positive": r0["positive"],
            "target_fraction": r0["target_fraction"],
            "max_prob_mean": float(np.mean([r["max_prob"] for r in rows])),
        }
        if r0["positive"]:
            for key in ("dice", "iou", "recall", "precision", "pos_bg_fp", "boundary_f1", "detected"):
                item[key] = float(np.mean([r[key] for r in rows]))
        else:
            for key in ("neg_fp", "neg_clean"):
                item[key] = float(np.mean([r[key] for r in rows]))
        image_summary[qid] = item

    pos_imgs = [v for v in image_summary.values() if v["positive"]]
    neg_imgs = [v for v in image_summary.values() if not v["positive"]]
    strata = {}
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
                "recall": float(np.mean([v["recall"] for v in sub])),
                "dice": float(np.mean([v["dice"] for v in sub])),
                "precision": float(np.mean([v["precision"] for v in sub])),
                "detected": float(np.mean([v["detected"] for v in sub])),
            }

    macro = {
        "n_pos_sites": len(pos_sites),
        "n_neg_sites": len(neg_sites),
        **{k: float(np.mean([v[k] for v in pos_sites])) for k in pos_keys},
        "neg_fp": float(np.mean([v["neg_fp"] for v in neg_sites])),
        "neg_clean": float(np.mean([v["neg_clean"] for v in neg_sites])),
        "neg_max_fp": float(max(v["neg_fp"] for v in neg_imgs)) if neg_imgs else 0.0,
        "neg_worst_id": max(neg_imgs, key=lambda v: v["neg_fp"])["id"] if neg_imgs else None,
    }
    return {"macro": macro, "strata": strata, "sites": site_summary, "images": image_summary}


def compare_sealed_methods(cand_agg: dict, ref_agg: dict) -> dict:
    """Compare candidate vs reference across the 15 positive and 4 negative sealed test sites."""
    c_sites = cand_agg["sites"]
    r_sites = ref_agg["sites"]
    pos_keys = sorted(s for s, v in c_sites.items() if v["positive_images"] > 0)
    neg_keys = sorted(s for s, v in c_sites.items() if v["negative_images"] > 0)

    def boot(keys: list[str], metric: str) -> dict:
        return paired_site_bootstrap({s: c_sites[s][metric] for s in keys}, {s: r_sites[s][metric] for s in keys})

    dice_b = boot(pos_keys, "dice")
    rec_b = boot(pos_keys, "recall")
    prec_b = boot(pos_keys, "precision")
    ap512_b = boot(pos_keys, "ap_512")
    bf1_b = boot(pos_keys, "boundary_f1")
    neg_fp_b = boot(neg_keys, "neg_fp")
    ref_neg = max(1e-8, ref_agg["macro"]["neg_fp"])
    return {
        "dice": dice_b,
        "recall": rec_b,
        "precision": prec_b,
        "ap_512": ap512_b,
        "boundary_f1": bf1_b,
        "neg_fp": {**neg_fp_b, "relative_reduction": float((ref_agg["macro"]["neg_fp"] - cand_agg["macro"]["neg_fp"]) / ref_neg)},
        "small_lt5pct_recall_delta": float(
            cand_agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            - ref_agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
        ),
    }


def _save_sealed_panels(
    out_dir: Path,
    rgb_by_id: dict[str, np.ndarray],
    mask_by_id: dict[str, np.ndarray],
    panel_probs: dict[str, dict[str, np.ndarray]],
    test_records: list[dict],
) -> None:
    from PIL import Image, ImageDraw

    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    pos_recs = sorted([r for r in test_records if r["positive"]], key=lambda x: x["target_fraction"])
    chosen_pos = [pos_recs[0]["id"], pos_recs[len(pos_recs) // 3]["id"], pos_recs[(2 * len(pos_recs)) // 3]["id"], pos_recs[-1]["id"]]
    chosen_neg = sorted([r["id"] for r in test_records if not r["positive"]])

    cols = ("GT", "B0_32_Ref", "M5_BalCal", "M3_Pruned_Ref")
    for fname, qids in (("k5_sealed_pos_compare.png", chosen_pos), ("k5_sealed_neg_compare.png", chosen_neg)):
        tile = 320
        canvas = Image.new("RGB", (tile * len(cols), tile * len(qids)), (18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        for ri, qid in enumerate(qids):
            rgb = rgb_by_id[qid].transpose(1, 2, 0).astype(np.float32)
            gt = mask_by_id[qid] > 0
            for ci, cname in enumerate(cols):
                base = rgb.copy()
                if cname == "GT":
                    base[gt] = 0.55 * base[gt] + 0.45 * np.array([30, 220, 110], dtype=np.float32)
                else:
                    pred = panel_probs[cname][qid] >= 0.5
                    tp = gt & pred
                    fp = (~gt) & pred
                    fn = gt & (~pred)
                    base[tp] = 0.55 * base[tp] + 0.45 * np.array([30, 220, 110], dtype=np.float32)
                    base[fp] = 0.50 * base[fp] + 0.50 * np.array([240, 60, 55], dtype=np.float32)
                    base[fn] = 0.50 * base[fn] + 0.50 * np.array([245, 200, 40], dtype=np.float32)
                im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.BILINEAR)
                canvas.paste(im, (ci * tile, ri * tile))
                draw.text((ci * tile + 8, ri * tile + 6), f"{qid} | {cname}", fill=(255, 255, 255))
        canvas.save(panels_dir / fname)


def run_final_sealed_eval(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    source_dir: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    _, _, test_records = load_sealed_test_records(split_dir)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    dev_payload = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    dev_feats = dev_payload["features"]

    test_feats, test_rgbs, test_masks = extract_sealed_test_features(
        test_records,
        image_dir=img_dir,
        label_dir=lbl_dir,
        source_dir=source_dir,
        weights_path=weights_path,
        device=run_device,
    )

    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "mid_64": {}, "deep_64": {}}
    mask_by_id: dict[str, np.ndarray] = {}
    aff_by_id: dict[str, torch.Tensor] = {}

    for r in dev_records:
        mask_by_id[r["id"]] = read_verified_mask(lbl_dir, r)
        for key in ("deep_32", "mid_64", "deep_64"):
            tokens_by_key[key][r["id"]] = _to_tokens_norm(dev_feats[r["id"]][key], run_device)
    for r in test_records:
        mask_by_id[r["id"]] = test_masks[r["id"]]
        for key in ("deep_32", "mid_64", "deep_64"):
            tokens_by_key[key][r["id"]] = _to_tokens_norm(test_feats[r["id"]][key], run_device)
        aff_by_id[r["id"]] = compute_dino_affinity_weights(tokens_by_key["deep_64"][r["id"]], tau=0.12)

    all_ids = [r["id"] for r in dev_records] + [r["id"] for r in test_records]
    dev_pos_ids = [r["id"] for r in dev_records if r["positive"]]
    caches = {
        key: build_pairwise_topk_cache(tokens_by_key[key], mask_by_id, all_ids, dev_pos_ids, grid_hw=(64 if "64" in key else 32))
        for key in ("deep_32", "mid_64", "deep_64")
    }

    test_by_id = {r["id"]: r for r in test_records}
    query_items = [
        {**test_by_id[r["id"]], "mask": mask_by_id[r["id"]], "y32": area_target_32(mask_by_id[r["id"]])}
        for r in test_records
    ]

    episodes_by_k = build_sealed_test_episodes(dev_records, test_records, ks=(1, 3, 5), num_episodes=40)
    methods_summary: dict[str, dict] = {cfg.name: {} for cfg in SEALED_VARIANTS}
    panel_probs: dict[str, dict[str, np.ndarray]] = {}

    for cfg in SEALED_VARIANTS:
        print(f"[SealedTest] Evaluating locked variant {cfg.name}...", flush=True)
        for k_str in ("1", "3", "5"):
            ep_results = []
            prob_accum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((512, 512), dtype=np.float32))
            for ep in episodes_by_k[k_str]:
                p512_map, s32_map = run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"], cfg)
                ep_results.append(evaluate_episode(query_items, p512_map, s32_map))
                if k_str == "5" and cfg.name in ("B0_32_Ref", "M5_BalCal", "M3_Pruned_Ref"):
                    for qid, p in p512_map.items():
                        prob_accum[qid] += p / len(episodes_by_k[k_str])
            methods_summary[cfg.name][k_str] = aggregate_sealed_episodes(ep_results, expected_episodes=40)
            if k_str == "5" and cfg.name in ("B0_32_Ref", "M5_BalCal", "M3_Pruned_Ref"):
                panel_probs[cfg.name] = dict(prob_accum)

    _save_sealed_panels(output_dir, test_rgbs, test_masks, panel_probs, test_records)
    comparisons = {}
    for cname, rname in (
        ("M5_BalCal_vs_B0_32_Ref", "B0_32_Ref"),
        ("M5_BalCal_vs_M1_Full_Ref", "M1_Full_Ref"),
        ("M5_TightRescue_vs_B0_32_Ref", "B0_32_Ref"),
        ("M3_Pruned_Ref_vs_B0_32_Ref", "B0_32_Ref"),
    ):
        comparisons[cname] = {
            k_str: compare_sealed_methods(methods_summary[cname.split("_vs_")[0]][k_str], methods_summary[rname][k_str])
            for k_str in ("1", "3", "5")
        }

    summary = {
        "schema": "pv-2m-dinov3sat-sealed-test-v1",
        "note": "One-shot evaluation of pre-locked dev configurations on the 21 sealed test images (15 pos sites, 4 neg sites), 40 dev-support episodes per K.",
        "methods": methods_summary,
        "comparisons": comparisons,
    }
    write_json(output_dir / "sealed_test_summary.json", summary)
    return summary
