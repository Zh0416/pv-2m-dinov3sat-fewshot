from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .baseline import calibrate_b0, normalize_dino_tokens, score_raw_tokens
from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .dinov3_backbone import PINNED_COMMIT, SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from .episodes import load_and_verify_episodes
from .io import sha256_file, write_json
from .metrics import aggregate_episodes, evaluate_episode

LAYER_GROUPS_32 = {
    "early_32": (8, 9, 10, 11),
    "mid_32": (12, 13, 14, 15),
    "semi_32": (16, 17, 18, 19),
    "deep_32": (20, 21, 22, 23),
}

LAYER_GROUPS_64 = {
    "mid_64": (12, 13, 14, 15),
    "deep_64": (20, 21, 22, 23),
}


def build_multiscale_dev_cache(
    *,
    split_dir: Path,
    output_dir: Path,
    source_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Extract and save normalized float16 multi-layer & multi-scale features for the 79 dev images ONLY."""
    import importlib
    import sys

    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    img_dir, _ = resolve_data_dirs(image_dir, label_dir)

    weights = resolve_dinov3_weights(weights_path)
    source = prepare_dinov3_source(source_dir)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    backbones = importlib.import_module("dinov3.hub.backbones")

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = backbones.dinov3_vitl16(pretrained=True, weights=str(weights.resolve()))
    model.to(run_device).eval().requires_grad_(False)

    all_layers = sorted({l for grp in (*LAYER_GROUPS_32.values(), *LAYER_GROUPS_64.values()) for l in grp})
    layer_to_pos = {l: i for i, l in enumerate(all_layers)}

    mean_t = torch.tensor(SAT_MEAN, device=run_device, dtype=torch.float32)[None, :, None, None]
    std_t = torch.tensor(SAT_STD, device=run_device, dtype=torch.float32)[None, :, None, None]

    cache_payload: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for idx, row in enumerate(dev_records, 1):
            if row["partition"] != "dev":
                raise PermissionError("Refusing to cache sealed test image")
            rgb = read_verified_rgb(img_dir, row)
            x512 = (torch.from_numpy(rgb.copy())[None].to(run_device).float() / 255.0 - mean_t) / std_t
            outs_32 = model.get_intermediate_layers(x512, n=all_layers, reshape=True, norm=True)

            entry: dict[str, torch.Tensor] = {}
            for name, grp in LAYER_GROUPS_32.items():
                stacked = torch.stack([outs_32[layer_to_pos[l]][0].float() for l in grp], dim=0).mean(dim=0)
                normed = F.normalize(stacked, dim=0)
                if not torch.isfinite(normed).all():
                    raise RuntimeError(f"Non-finite 512 feature for {row['id']} / {name}")
                entry[name] = normed.half().cpu()

            x1024 = F.interpolate(x512, size=(1024, 1024), mode="bilinear", align_corners=False)
            layers_64 = sorted({l for grp in LAYER_GROUPS_64.values() for l in grp})
            pos_64 = {l: i for i, l in enumerate(layers_64)}
            outs_64 = model.get_intermediate_layers(x1024, n=layers_64, reshape=True, norm=True)
            for name, grp in LAYER_GROUPS_64.items():
                stacked = torch.stack([outs_64[pos_64[l]][0].float() for l in grp], dim=0).mean(dim=0)
                normed = F.normalize(stacked, dim=0)
                if not torch.isfinite(normed).all():
                    raise RuntimeError(f"Non-finite 1024 feature for {row['id']} / {name}")
                entry[name] = normed.half().cpu()

            cache_payload[row["id"]] = entry
            if idx % 20 == 0 or idx == len(dev_records):
                print(f"[Cache] Extracted 512+1024 multi-layer features {idx}/{len(dev_records)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_pt = output_dir / "dev_features_v1.pt"
    tmp_pt = output_dir / "dev_features_v1.tmp.pt"
    torch.save(
        {
            "schema": "pvfs-multiscale-cache-v1",
            "commit": PINNED_COMMIT,
            "weight_sha256": sha256_file(weights),
            "features": cache_payload,
        },
        tmp_pt,
    )
    tmp_pt.replace(out_pt)
    print(f"[Cache] Saved {out_pt} ({out_pt.stat().st_size / (1024**3):.3f} GiB)", flush=True)
    return out_pt


@torch.inference_mode()
def _build_pool_generic(
    tokens_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    support_ids: list[str],
    grid_hw: int,
    *,
    purity: float = 0.5,
) -> dict:
    patch_px = 512 // grid_hw
    fg_list, bg_list = [], []
    fg_src, bg_src, bg_yx = [], [], []
    fg_masks = []
    for idx, sid in enumerate(support_ids):
        m = mask_by_id[sid].astype(np.float32)
        y_grid = m.reshape(grid_hw, patch_px, grid_hw, patch_px).mean(axis=(1, 3)).ravel()
        tok = tokens_by_id[sid]
        fg_mask = y_grid >= purity
        if not fg_mask.any():
            cands = np.flatnonzero(y_grid > 0)
            count = min(len(cands), max(1, int(np.ceil(float(y_grid.sum())))))
            fg_mask[cands[np.argsort(-y_grid[cands], kind="stable")[:count]]] = True
        bg_mask = y_grid == 0.0
        fg_idx_t = torch.from_numpy(np.flatnonzero(fg_mask)).to(tok.device)
        fg_list.append(tok[fg_idx_t])
        fg_src.append(torch.full((fg_idx_t.numel(),), idx, dtype=torch.long, device=tok.device))
        fg_masks.append(fg_mask)
        if bg_mask.any():
            bg_flat = np.flatnonzero(bg_mask)
            bg_idx_t = torch.from_numpy(bg_flat).to(tok.device)
            bg_list.append(tok[bg_idx_t])
            bg_src.append(torch.full((bg_idx_t.numel(),), idx, dtype=torch.long, device=tok.device))
            bg_yx.append(torch.stack((torch.from_numpy(bg_flat // grid_hw), torch.from_numpy(bg_flat % grid_hw)), dim=1).to(tok.device))

    fg = torch.cat(fg_list, dim=0)
    fg_s = torch.cat(fg_src, dim=0)
    bg = torch.cat(bg_list, dim=0) if bg_list else fg.new_empty((0, fg.shape[1]))
    bg_s = torch.cat(bg_src, dim=0) if bg_src else fg_s.new_empty((0,))
    bg_c = torch.cat(bg_yx, dim=0) if bg_yx else fg_s.new_empty((0, 2))
    return {"fg": fg, "fg_src": fg_s, "bg": bg, "bg_src": bg_s, "bg_yx": bg_c, "fg_masks_32": fg_masks, "grid_hw": grid_hw}


@torch.inference_mode()
def run_layer_scale_probe(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict:
    """Evaluate single-layer and 64x64 RoPE contrast baselines across K=1,3,5."""
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    _, lbl_dir = resolve_data_dirs(None, label_dir)
    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    records_by_id, mask_by_id = {}, {}
    for row in dev_records:
        mask = read_verified_mask(lbl_dir, row)
        records_by_id[row["id"]] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[row["id"]] = mask

    ep_by_k = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    probe_keys = ("early_32", "mid_32", "semi_32", "deep_32", "mid_64", "deep_64")
    results: dict[str, dict] = {}
    for key in probe_keys:
        grid_hw = 64 if key.endswith("_64") else 32
        toks = {
            rid: F.normalize(feats[rid][key].to(device=run_device, dtype=torch.float32), dim=0).flatten(1).T.contiguous()
            for rid in records_by_id
        }
        results[key] = {}
        for k, eps in sorted(ep_by_k.items()):
            ep_evals = []
            for ep in eps:
                pool = _build_pool_generic(toks, mask_by_id, ep["support_ids"], grid_hw, purity=0.5)
                # reuse B0 calibration logic adapted to generic grid_hw
                if k >= 2:
                    fg_all, bg_all = [], []
                    for idx, sid in enumerate(ep["support_ids"]):
                        o_fg = pool["fg"][pool["fg_src"] != idx]
                        o_bg = pool["bg"][pool["bg_src"] != idx]
                        if o_fg.shape[0] == 0:
                            continue
                        raw, _, _ = score_raw_tokens(toks[sid], o_fg, o_bg, top_m=3, alpha_bg=0.5)
                        fg_all.append(raw[torch.from_numpy(pool["fg_masks_32"][idx]).to(raw.device)])
                        bm = pool["bg_src"] == idx
                        if bm.any():
                            yx = pool["bg_yx"][bm]
                            bg_all.append(raw[yx[:, 0] * grid_hw + yx[:, 1]])
                    fg_cat, bg_cat = torch.cat(fg_all), torch.cat(bg_all)
                    mu_fg, mu_bg = float(fg_cat.median().item()), float(torch.quantile(bg_cat, 0.90).item())
                    if mu_fg > mu_bg + 0.02:
                        bias = 0.5 * (mu_fg + mu_bg)
                        temp = max((mu_fg - mu_bg) / (2.0 * np.log(4.0)), 0.04)
                    else:
                        bias, temp = mu_bg + 0.05, 0.06
                else:
                    # For 1-shot at 64x64, use ex-neighborhood radius 2 (equivalent 32m exclusion!)
                    radius = 2 if grid_hw == 64 else 1
                    min_bg = 64 if grid_hw == 64 else 16
                    bg, fg, yx = pool["bg"], pool["fg"], pool["bg_yx"]
                    if bg.shape[0] >= min_bg:
                        sim_fg = (bg @ fg.T).clamp(-1.0, 1.0)
                        s_pos_bg = sim_fg.topk(min(3, fg.shape[0]), dim=1).values.mean(dim=1)
                        dy = (yx[:, 0:1] - yx[:, 0:1].T).abs()
                        dx = (yx[:, 1:2] - yx[:, 1:2].T).abs()
                        near = (dy <= radius) & (dx <= radius)
                        valid_cnt = int((~near).sum(dim=1).min().item())
                        if valid_cnt >= 1:
                            sim_bg = (bg @ bg.T).clamp(-1.0, 1.0).masked_fill(near, -2.0)
                            s_neg_bg = sim_bg.topk(min(3, valid_cnt), dim=1).values.mean(dim=1)
                            q90 = float(torch.quantile(s_pos_bg - 0.5 * s_neg_bg, 0.90).item())
                            bias, temp = q90 + 0.08, 0.06
                        else:
                            bias, temp = 0.30, 0.06
                    else:
                        bias, temp = 0.30, 0.06

                probs_512, scores_32 = {}, {}
                q_recs = [records_by_id[qid] for qid in ep["query_ids"]]
                for qid in ep["query_ids"]:
                    raw, _, _ = score_raw_tokens(toks[qid], pool["fg"], pool["bg"], top_m=3, alpha_bg=0.5)
                    raw_g = raw.reshape(grid_hw, grid_hw)
                    p_g = torch.sigmoid((raw_g - bias) / temp)
                    p512 = F.interpolate(p_g[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
                    probs_512[qid] = p512.cpu().numpy().astype(np.float32)
                    if grid_hw == 32:
                        scores_32[qid] = raw_g.cpu().numpy().astype(np.float32)
                    else:
                        scores_32[qid] = F.avg_pool2d(raw_g[None, None], 2, stride=2)[0, 0].cpu().numpy().astype(np.float32)
                ep_evals.append(evaluate_episode(q_recs, probs_512, scores_32, threshold=0.5))

            agg = aggregate_episodes(ep_evals, episodes_per_fold=episodes_lock["episodes_per_fold"])
            m = agg["macro"]
            small_d = agg["strata"].get("small_lt5pct", {}).get("dice", 0.0)
            small_r = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
            results[key][str(k)] = {
                "macro": m,
                "strata": agg["strata"],
                "pv73_rec": agg["images"].get("正样本/PV-73.tif", {}).get("recall"),
                "pv93_fp": agg["images"].get("负样本/PV-93.tif", {}).get("neg_fp"),
                "pv80_bgfp": agg["images"].get("正样本/PV-80.tif", {}).get("pos_bg_fp"),
                "pv21_bgfp": agg["images"].get("正样本/PV-21.tif", {}).get("pos_bg_fp"),
            }
            print(
                f"[Probe] {key:8s} K={k}: Dice={m['dice']:.4f} AP512={m['ap_512']:.4f} "
                f"Rec={m['recall']:.4f} Prec={m['precision']:.4f} BF1={m['boundary_f1']:.4f} "
                f"NegFP={m['neg_fp']:.4f} SmallD={small_d:.4f} SmallR={small_r:.4f} "
                f"PV93={results[key][str(k)]['pv93_fp']:.3f} PV80bg={results[key][str(k)]['pv80_bgfp']:.3f} PV73r={results[key][str(k)]['pv73_rec']:.3f}",
                flush=True,
            )

    write_json(output_dir / "layer_scale_probe.json", results)
    return results
