from __future__ import annotations

from collections import defaultdict
import importlib
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .dinov3_backbone import SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from .fast_bank import build_pairwise_topk_cache
from .final_sealed_eval import (
    aggregate_sealed_episodes,
    build_sealed_test_episodes,
    compare_sealed_methods,
    extract_sealed_test_features,
    load_sealed_test_records,
)
from .freq_features import apply_ofg_neg_gate_and_rescue, compute_support_freq_alignment, extract_freq_pyramid
from .io import write_json
from .iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from .iter5_methods import run_iter5_episode
from .metrics import evaluate_episode, paired_site_bootstrap
from .v2_decoder import SCFBDecoder, build_decoder_input_512, compute_support_rgb_stats
from .v2_pipeline import V2_VARIANTS, V2Variant, compute_mscp_episode_maps


def resolve_ext185_dirs(root_dir: Path | None = None) -> tuple[Path, Path]:
    """Locate PV-512-MERGED-185 images/ and labels/ directories."""
    probe = Path("PV-000001.bmp")
    if root_dir is not None:
        img_d, lbl_d = root_dir / "images", root_dir / "labels"
        if (img_d / probe).is_file() and (lbl_d / probe).is_file():
            return img_d, lbl_d
        raise FileNotFoundError(f"Missing {probe} under {root_dir}")

    for cand in (
        Path("/kaggle/input/datasets/zh0416/pv-512-merged-185/PV-512-MERGED-185"),
        Path("/kaggle/input/pv-512-merged-185/PV-512-MERGED-185"),
    ):
        if (cand / "images" / probe).is_file() and (cand / "labels" / probe).is_file():
            return cand / "images", cand / "labels"

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = [p for p in kaggle_input.rglob("PV-512-MERGED-185") if (p / "images" / probe).is_file()]
        if len(matches) == 1:
            return matches[0] / "images", matches[0] / "labels"
    raise FileNotFoundError("Could not locate PV-512-MERGED-185 dataset")


def load_ext185_records(img_dir: Path, lbl_dir: Path) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load all 185 BMP images & binary masks (0=bg, 255=PV) from PV-512-MERGED-185."""
    records = []
    rgbs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for p in sorted(img_dir.glob("*.bmp")):
        lp = lbl_dir / p.name
        if not lp.is_file():
            raise FileNotFoundError(f"Missing label for {p.name} in {lbl_dir}")
        rgb_hwc = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        lbl_hw = np.array(Image.open(lp).convert("L"), dtype=np.uint8)
        if rgb_hwc.shape[:2] != (512, 512) or lbl_hw.shape != (512, 512):
            raise ValueError(f"Unexpected shape in {p.name}: {rgb_hwc.shape}, {lbl_hw.shape}")
        mask = (lbl_hw > 127).astype(np.uint8)
        frac = float(mask.mean())
        rid = f"ext185/{p.name}"
        rgbs[rid] = rgb_hwc.transpose(2, 0, 1).copy()
        masks[rid] = mask
        records.append(
            {
                "id": rid,
                "site": rid,
                "fold": None,
                "partition": "ext185",
                "positive": bool(mask.any()),
                "target_fraction": frac,
            }
        )
    if not records:
        raise ValueError(f"No BMP files found in {img_dir}")
    return records, rgbs, masks


def extract_dinov3_features_generic(
    records: list[dict],
    rgb_by_id: dict[str, np.ndarray],
    *,
    source_dir: Path,
    weights_path: Path | None = None,
    device: str = "cuda",
    tag: str = "Ext185",
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract pure frozen DINOv3-SAT deep_32, mid_64, deep_64 features cleanly for arbitrary RGB records."""
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
    with torch.inference_mode():
        for idx, row in enumerate(records, 1):
            rgb = rgb_by_id[row["id"]]
            x512 = (torch.from_numpy(rgb.copy())[None].to(run_device).float() / 255.0 - mean_t) / std_t
            outs_32 = model.get_intermediate_layers(x512, n=list(layers_32), reshape=True, norm=True)
            d32 = F.normalize(torch.stack([t[0].float() for t in outs_32], dim=0).mean(dim=0), dim=0)

            x1024 = F.interpolate(x512, size=(1024, 1024), mode="bilinear", align_corners=False)
            outs_64 = model.get_intermediate_layers(x1024, n=list(layers_64), reshape=True, norm=True)
            m64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4)], dim=0).mean(dim=0), dim=0)
            d64 = F.normalize(torch.stack([outs_64[i][0].float() for i in range(4, 8)], dim=0).mean(dim=0), dim=0)

            feats[row["id"]] = {"deep_32": d32.half().cpu(), "mid_64": m64.half().cpu(), "deep_64": d64.half().cpu()}
            if idx % 37 == 0 or idx == len(records):
                print(f"[{tag}] Extracted DINOv3-SAT features {idx}/{len(records)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


@torch.inference_mode()
def run_v2_episode_ensemble(
    variant: V2Variant,
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor],
    rgb_by_id: dict[str, torch.Tensor],
    freq512_by_id: dict[str, torch.Tensor],
    freq64_by_id: dict[str, torch.Tensor],
    support_ids: list[str],
    query_ids: list[str],
    *,
    decoders: list[SCFBDecoder] | None = None,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run one V2 episode using locked 4-fold ensemble SCFB decoders when variant.kind=='decoder'."""
    if variant.kind == "iter5_ref":
        assert variant.ref_cfg is not None
        return run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, support_ids, query_ids, variant.ref_cfg)

    mscp = compute_mscp_episode_maps(caches, tokens_by_key, mask_by_id, aff_by_id, support_ids, query_ids)
    zf_map = compute_support_freq_alignment(freq64_by_id, mask_by_id, support_ids, query_ids)

    if variant.kind == "freq_only":
        probs_512, scores_32 = {}, {}
        for qid in query_ids:
            z_comb = apply_ofg_neg_gate_and_rescue(mscp[qid]["z_sem"], zf_map[qid])
            p512 = torch.sigmoid(F.interpolate(z_comb[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0])
            s32 = F.avg_pool2d(z_comb[None, None], kernel_size=2, stride=2)[0, 0]
            probs_512[qid] = p512.cpu().numpy().astype(np.float32)
            scores_32[qid] = s32.cpu().numpy().astype(np.float32)
        return probs_512, scores_32

    assert decoders
    fg_rgb, bg_rgb = compute_support_rgb_stats(rgb_by_id, mask_by_id, support_ids, device)
    probs_512, scores_32 = {}, {}
    use_amp = device.type == "cuda"
    batch_size = 8
    for start in range(0, len(query_ids), batch_size):
        chunk_ids = query_ids[start : start + batch_size]
        xb = torch.stack(
            [
                build_decoder_input_512(
                    mscp[qid]["sem_6"],
                    zf_map[qid],
                    rgb_by_id[qid].to(device),
                    freq512_by_id[qid].to(device),
                    fg_rgb,
                    bg_rgb,
                    disable_freq=variant.disable_freq,
                )
                for qid in chunk_ids
            ],
            dim=0,
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits_sum = sum(dec(xb).float().squeeze(1) for dec in decoders) / float(len(decoders))
        probs = torch.sigmoid(logits_sum)
        s32_b = F.avg_pool2d(logits_sum[:, None], kernel_size=16, stride=16).squeeze(1)
        for bi, qid in enumerate(chunk_ids):
            probs_512[qid] = probs[bi].cpu().numpy().astype(np.float32)
            scores_32[qid] = s32_b[bi].cpu().numpy().astype(np.float32)
    return probs_512, scores_32


def aggregate_generic_episodes(episode_results: list[dict]) -> dict:
    """Aggregate episodes with variable query counts (handles zero-negative or negative-present pools cleanly)."""
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
        entry = {"site": site, "positive_images": rows[0]["positive_images"], "negative_images": rows[0]["negative_images"]}
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
        item = {"id": qid, "site": r0["site"], "positive": r0["positive"], "target_fraction": r0["target_fraction"]}
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
            }

    macro = {
        "n_pos_sites": len(pos_sites),
        "n_neg_sites": len(neg_sites),
        **{k: float(np.mean([v[k] for v in pos_sites])) for k in pos_keys},
        "neg_fp": float(np.mean([v["neg_fp"] for v in neg_sites])) if neg_sites else 0.0,
        "neg_clean": float(np.mean([v["neg_clean"] for v in neg_sites])) if neg_sites else 1.0,
        "neg_max_fp": float(max(v["neg_fp"] for v in neg_imgs)) if neg_imgs else 0.0,
    }
    return {"macro": macro, "strata": strata, "sites": site_summary, "images": image_summary}


def build_ext185_episodes(
    dev_records: list[dict],
    ext_records: list[dict],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    num_episodes: int = 20,
    seed_base: int = 20261250,
) -> dict[str, dict[str, list[dict]]]:
    """Build deterministic Cross-Dataset (dev->ext185) and In-Domain (ext185->ext185) episodes."""
    dev_pos_by_site: dict[str, list[str]] = defaultdict(list)
    for r in dev_records:
        if r["positive"]:
            dev_pos_by_site[r["site"]].append(r["id"])
    dev_sites = sorted(dev_pos_by_site)
    ext_all_ids = [r["id"] for r in ext_records]
    ext_pos_ids = [r["id"] for r in ext_records if r["positive"]]

    out: dict[str, dict[str, list[dict]]] = {"cross_dataset": {}, "in_domain": {}}
    for k in ks:
        cross_eps, indom_eps = [], []
        for ep in range(num_episodes):
            rng_c = random.Random(seed_base + 1000 * k + ep)
            chosen_sites = rng_c.sample(dev_sites, k)
            cross_eps.append(
                {
                    "support_ids": [rng_c.choice(dev_pos_by_site[s]) for s in chosen_sites],
                    "query_ids": ext_all_ids,
                }
            )
            rng_i = random.Random(seed_base + 50000 + 1000 * k + ep)
            sup_i = rng_i.sample(ext_pos_ids, k)
            sup_set = set(sup_i)
            indom_eps.append(
                {
                    "support_ids": sup_i,
                    "query_ids": [qid for qid in ext_all_ids if qid not in sup_set],
                }
            )
        out["cross_dataset"][str(k)] = cross_eps
        out["in_domain"][str(k)] = indom_eps
    return out


def _save_ext_panels(
    out_dir: Path,
    fname: str,
    rgb_u8_by_id: dict[str, np.ndarray],
    mask_by_id: dict[str, np.ndarray],
    panel_probs: dict[str, dict[str, np.ndarray]],
    chosen_ids: list[str],
) -> None:
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    cols = ("GT", "Ref_M5_BalCal", "V2_FreqOnly_NoDecoder", "V2_Full_WithDecoder")
    labels = ("GT", "M5_BalCal", "V2_FreqOnly", "V2_FullDecoder")
    tile = 320
    canvas = Image.new("RGB", (tile * len(cols), tile * len(chosen_ids)), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    for ri, qid in enumerate(chosen_ids):
        rgb = rgb_u8_by_id[qid].transpose(1, 2, 0).astype(np.float32)
        gt = mask_by_id[qid] > 0
        for ci, (cname, lbl) in enumerate(zip(cols, labels)):
            base = rgb.copy()
            if cname == "GT":
                base[gt] = 0.50 * base[gt] + 0.50 * np.array([35, 220, 95], dtype=np.float32)
            else:
                pred = panel_probs[cname][qid] >= 0.5
                tp, fp, fn = gt & pred, (~gt) & pred, gt & (~pred)
                base[tp] = 0.50 * base[tp] + 0.50 * np.array([35, 220, 95], dtype=np.float32)
                base[fp] = 0.45 * base[fp] + 0.55 * np.array([240, 55, 55], dtype=np.float32)
                base[fn] = 0.45 * base[fn] + 0.55 * np.array([250, 205, 45], dtype=np.float32)
            im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((tile, tile), Image.Resampling.BILINEAR)
            canvas.paste(im, (ci * tile, ri * tile))
            draw.text((ci * tile + 6, ri * tile + 4), f"{qid} | {lbl}", fill=(255, 255, 255))
    canvas.save(panels_dir / fname)


def run_v2_sealed_and_ext185(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    decoders_pt: Path,
    source_dir: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    ext185_root: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> dict:
    """Evaluate locked V2 decoders (trained only on dev) on Sealed 21 AND External PV-512-MERGED-185."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 1. Load 4-fold trained decoders cleanly without retraining
    dec_ckpt = torch.load(decoders_pt, map_location="cpu", weights_only=False)
    dec_ensembles: dict[str, list[SCFBDecoder]] = {}
    for vname in ("V2_Full_WithDecoder", "V2_Decoder_NoFreq_LOO"):
        models = []
        for f in range(4):
            m = SCFBDecoder(in_ch=17).to(run_device)
            m.load_state_dict(dec_ckpt[vname][f])
            m.eval()
            models.append(m)
        dec_ensembles[vname] = models

    # 2. Load dev features & sealed test + external 185 records
    _, _, dev_records = load_dev_records(split_dir)
    _, _, test_records = load_sealed_test_records(split_dir)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)
    ext_img_dir, ext_lbl_dir = resolve_ext185_dirs(ext185_root)
    ext_records, ext_rgbs_u8, ext_masks = load_ext185_records(ext_img_dir, ext_lbl_dir)

    dev_payload = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    dev_feats = dev_payload["features"]
    test_feats, test_rgbs_u8, test_masks = extract_sealed_test_features(
        test_records, image_dir=img_dir, label_dir=lbl_dir, source_dir=source_dir, weights_path=weights_path, device=str(run_device)
    )
    ext_feats = extract_dinov3_features_generic(
        ext_records, ext_rgbs_u8, source_dir=source_dir, weights_path=weights_path, device=str(run_device), tag="Ext185"
    )

    # Assemble combined feature/frequency maps
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "mid_64": {}, "deep_64": {}}
    mask_by_id: dict[str, np.ndarray] = {}
    rgb_u8_all: dict[str, np.ndarray] = {}
    rgb_t_all: dict[str, torch.Tensor] = {}
    freq512_by_id: dict[str, torch.Tensor] = {}
    freq64_by_id: dict[str, torch.Tensor] = {}
    aff_by_id: dict[str, torch.Tensor] = {}

    def _register(rid: str, rgb_u8: np.ndarray, mask: np.ndarray, feat_entry: dict) -> None:
        rgb_u8_all[rid] = rgb_u8
        rgb_t_all[rid] = (torch.from_numpy(rgb_u8.copy()).float() / 255.0).half()
        f512, f64 = extract_freq_pyramid(rgb_u8, device=run_device)
        freq512_by_id[rid] = f512
        freq64_by_id[rid] = f64.to(run_device)
        mask_by_id[rid] = mask
        for key in ("deep_32", "mid_64", "deep_64"):
            tokens_by_key[key][rid] = _to_tokens_norm(feat_entry[key], run_device)
        aff_by_id[rid] = compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12)

    for r in dev_records:
        _register(r["id"], read_verified_rgb(img_dir, r), read_verified_mask(lbl_dir, r), dev_feats[r["id"]])
    for r in test_records:
        _register(r["id"], test_rgbs_u8[r["id"]], test_masks[r["id"]], test_feats[r["id"]])
    for r in ext_records:
        _register(r["id"], ext_rgbs_u8[r["id"]], ext_masks[r["id"]], ext_feats[r["id"]])

    # Pairwise cache allowing support from EITHER dev positive OR ext185 positive
    all_ids = [r["id"] for r in (*dev_records, *test_records, *ext_records)]
    pos_support_pool = [r["id"] for r in (*dev_records, *ext_records) if r["positive"]]
    caches = {
        key: build_pairwise_topk_cache(tokens_by_key[key], mask_by_id, all_ids, pos_support_pool, grid_hw=(64 if "64" in key else 32))
        for key in ("deep_32", "mid_64", "deep_64")
    }

    # --- PART A: SEALED TEST 21 (40 episodes per K) ---
    sealed_eps = build_sealed_test_episodes(dev_records, test_records, ks=(1, 3, 5), num_episodes=40)
    test_items = {r["id"]: {**r, "mask": mask_by_id[r["id"]], "y32": area_target_32(mask_by_id[r["id"]])} for r in test_records}
    sealed_methods: dict[str, dict] = {v.name: {} for v in V2_VARIANTS}
    sealed_panels: dict[str, dict[str, np.ndarray]] = {}
    panel_cols = ("Ref_M5_BalCal", "V2_FreqOnly_NoDecoder", "V2_Full_WithDecoder")

    for var in V2_VARIANTS:
        print(f"[V2 Sealed21] Evaluating {var.name}...", flush=True)
        decs = dec_ensembles[var.name] if var.kind == "decoder" else None
        for k_str in ("1", "3", "5"):
            ep_res = []
            p_acc: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((512, 512), dtype=np.float32))
            for ep in sealed_eps[k_str]:
                p512_map, s32_map = run_v2_episode_ensemble(
                    var, caches, tokens_by_key, mask_by_id, aff_by_id, rgb_t_all, freq512_by_id, freq64_by_id,
                    ep["support_ids"], ep["query_ids"], decoders=decs, device=run_device,
                )
                ep_res.append(evaluate_episode([test_items[q] for q in ep["query_ids"]], p512_map, s32_map))
                if k_str == "5" and var.name in panel_cols:
                    for qid, p in p512_map.items():
                        p_acc[qid] += p / len(sealed_eps[k_str])
            sealed_methods[var.name][k_str] = aggregate_sealed_episodes(ep_res, expected_episodes=40)
            if k_str == "5" and var.name in panel_cols:
                sealed_panels[var.name] = dict(p_acc)

    pos_t = sorted([r for r in test_records if r["positive"]], key=lambda x: x["target_fraction"])
    neg_t = sorted([r["id"] for r in test_records if not r["positive"]])
    _save_ext_panels(output_dir, "k5_v2_sealed_pos.png", rgb_u8_all, mask_by_id, sealed_panels, [pos_t[0]["id"], pos_t[5]["id"], pos_t[11]["id"], pos_t[-1]["id"]])
    _save_ext_panels(output_dir, "k5_v2_sealed_neg.png", rgb_u8_all, mask_by_id, sealed_panels, neg_t)

    sealed_cmp = {}
    for cname, rname in (
        ("V2_Full_WithDecoder_vs_Ref_M5_BalCal", "Ref_M5_BalCal"),
        ("V2_Full_WithDecoder_vs_Ref_M3_Pruned", "Ref_M3_Pruned"),
        ("V2_Full_WithDecoder_vs_V2_FreqOnly_NoDecoder", "V2_FreqOnly_NoDecoder"),
        ("V2_Full_WithDecoder_vs_V2_Decoder_NoFreq_LOO", "V2_Decoder_NoFreq_LOO"),
        ("V2_FreqOnly_NoDecoder_vs_Ref_M5_BalCal", "Ref_M5_BalCal"),
    ):
        cand = cname.split("_vs_")[0]
        sealed_cmp[cname] = {k_str: compare_sealed_methods(sealed_methods[cand][k_str], sealed_methods[rname][k_str]) for k_str in ("1", "3", "5")}

    # --- PART B: EXTERNAL DATASET PV-512-MERGED-185 (20 episodes per K, Cross-Dataset & In-Domain) ---
    ext_eps = build_ext185_episodes(dev_records, ext_records, ks=(1, 3, 5), num_episodes=20)
    ext_items = {r["id"]: {**r, "mask": mask_by_id[r["id"]], "y32": area_target_32(mask_by_id[r["id"]])} for r in ext_records}
    ext_results: dict[str, dict] = {"cross_dataset": {v.name: {} for v in V2_VARIANTS}, "in_domain": {v.name: {} for v in V2_VARIANTS}}
    ext_panels: dict[str, dict[str, np.ndarray]] = {}

    for proto in ("cross_dataset", "in_domain"):
        for var in V2_VARIANTS:
            print(f"[Ext185 {proto}] Evaluating {var.name}...", flush=True)
            decs = dec_ensembles[var.name] if var.kind == "decoder" else None
            for k_str in ("1", "3", "5"):
                ep_res = []
                p_acc: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((512, 512), dtype=np.float32))
                for ep in ext_eps[proto][k_str]:
                    p512_map, s32_map = run_v2_episode_ensemble(
                        var, caches, tokens_by_key, mask_by_id, aff_by_id, rgb_t_all, freq512_by_id, freq64_by_id,
                        ep["support_ids"], ep["query_ids"], decoders=decs, device=run_device,
                    )
                    ep_res.append(evaluate_episode([ext_items[q] for q in ep["query_ids"]], p512_map, s32_map))
                    if proto == "cross_dataset" and k_str == "5" and var.name in panel_cols:
                        for qid, p in p512_map.items():
                            p_acc[qid] += p / len(ext_eps[proto][k_str])
                ext_results[proto][var.name][k_str] = aggregate_generic_episodes(ep_res)
                if proto == "cross_dataset" and k_str == "5" and var.name in panel_cols:
                    ext_panels[var.name] = dict(p_acc)

    ext_pos = sorted([r for r in ext_records if r["positive"]], key=lambda x: x["target_fraction"])
    ext_neg = [r["id"] for r in ext_records if not r["positive"]]
    chosen_ext = [ext_pos[0]["id"], ext_pos[len(ext_pos) // 2]["id"], ext_pos[-1]["id"]]
    if ext_neg:
        chosen_ext.append(ext_neg[0])
    elif len(ext_pos) > 3:
        chosen_ext.append(ext_pos[len(ext_pos) // 4]["id"])
    _save_ext_panels(output_dir, "k5_ext185_cross_compare.png", rgb_u8_all, mask_by_id, ext_panels, chosen_ext)

    summary = {
        "schema": "pv-2m-dinov3sat-v2-sealed-and-ext185-v1",
        "ext185_stats": {
            "total": len(ext_records),
            "positive": sum(1 for r in ext_records if r["positive"]),
            "negative": sum(1 for r in ext_records if not r["positive"]),
            "small_lt5pct": sum(1 for r in ext_records if r["positive"] and r["target_fraction"] < 0.05),
        },
        "sealed21": {"methods": sealed_methods, "comparisons": sealed_cmp},
        "ext185": ext_results,
    }
    write_json(output_dir / "v2_sealed_and_ext185_summary.json", summary)
    return summary
