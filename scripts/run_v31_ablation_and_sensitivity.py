from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache, compute_dino_affinity_weights, to_tokens_norm
from pvfs.io import read_json, write_json
from pvfs.metrics import aggregate_binary_episodes, compare_methods, evaluate_binary_episode
from pvfs.modules import PBDCConfig, ThreeModulePipeline, get_clean_core_config, refine_mask_pbdc
from pvfs.pv663_eval import (
    LazyDiskTokenMap,
    aggregate_pv663_episodes,
    build_pairwise_topk_cache_streaming,
    compare_pv663_methods,
    evaluate_pv663_episode,
    extract_pv663_features_to_dir,
    resolve_pv663_dirs,
)

# Shared read-only Linux fork worker state for parallel PBDC
_FW_RGB_BY_ID: dict[str, np.ndarray] = {}
_FW_PBDC_SPECS: list[tuple[str, PBDCConfig]] = []
_FW_K_SHOTS: int = 5


def _pbdc_multi_worker(args: tuple[str, np.ndarray, np.ndarray]) -> tuple[str, dict[str, np.ndarray]]:
    qid, m0, p0_f16 = args
    p0 = p0_f16.astype(np.float32)
    rgb = _FW_RGB_BY_ID[qid]
    out_masks: dict[str, np.ndarray] = {}
    for name, cfg in _FW_PBDC_SPECS:
        out_masks[name] = refine_mask_pbdc(m0, rgb, p0, k_shots=_FW_K_SHOTS, cfg=cfg)["mask"]
    return qid, out_masks


def _run_pbdc_batch(
    q_ids: list[str],
    m_a3: dict[str, np.ndarray],
    p_a3: dict[str, np.ndarray],
    specs: list[tuple[str, PBDCConfig]],
    rgb_by_id: dict[str, np.ndarray],
    k_shots: int,
    workers: int,
) -> dict[str, dict[str, np.ndarray]]:
    global _FW_RGB_BY_ID, _FW_PBDC_SPECS, _FW_K_SHOTS
    _FW_RGB_BY_ID = rgb_by_id
    _FW_PBDC_SPECS = specs
    _FW_K_SHOTS = k_shots

    tasks = [(qid, m_a3[qid], p_a3[qid].astype(np.float16)) for qid in q_ids]
    res: dict[str, dict[str, np.ndarray]] = {name: {} for name, _ in specs}
    use_fork = (workers > 1) and (sys.platform != "win32") and ("fork" in multiprocessing.get_all_start_methods())
    if use_fork:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for qid, mdict in ex.map(_pbdc_multi_worker, tasks, chunksize=8):
                for name, m in mdict.items():
                    res[name][qid] = m
    else:
        for task in tasks:
            qid, mdict = _pbdc_multi_worker(task)
            for name, m in mdict.items():
                res[name][qid] = m
    return res


def _pbdc_v31_ablation_specs() -> list[tuple[str, PBDCConfig]]:
    return [
        ("A4_v31_PBDC_Pure", PBDCConfig(enabled=True)),
        ("I3_v31_wo_AsymCore", PBDCConfig(enabled=True, ablate_undirected_grad=True)),
        ("I4_v31_wo_SemPull", PBDCConfig(enabled=True, w_sem=0.0)),
        ("I5_v31_wo_SectorFB", PBDCConfig(enabled=True, ablate_whole_loop_fallback=True)),
    ]


SENSITIVITY_SPECS: list[dict[str, Any]] = [
    {
        "key": "lambda_mid",
        "title": "M1: Mid-Layer Weight ($w_{\\mathrm{mid}}$)",
        "xlabel": "$w_{\\mathrm{mid}}$",
        "default": 0.35,
        "values": [0.0, 0.15, 0.25, 0.35, 0.50, 0.70],
        "kind": "core",
    },
    {
        "key": "w_fg_calib",
        "title": "M1: Calibration Weight ($\\alpha_{\\mathrm{cal}}$)",
        "xlabel": "$\\alpha_{\\mathrm{cal}}$",
        "default": 0.38,
        "values": [0.25, 0.32, 0.38, 0.45, 0.55],
        "kind": "core",
    },
    {
        "key": "qsc_weight",
        "title": "M2: Self-Rescue Gain ($\\beta_{\\mathrm{pos}}$)",
        "xlabel": "$\\beta_{\\mathrm{pos}}$",
        "default": 0.40,
        "values": [0.0, 0.20, 0.30, 0.40, 0.55, 0.70],
        "kind": "core",
    },
    {
        "key": "aff_tau",
        "title": "M3A: Affinity Temp ($\\tau_{\\mathrm{aff}}$)",
        "xlabel": "$\\tau_{\\mathrm{aff}}$",
        "default": 0.10,
        "values": [0.05, 0.08, 0.10, 0.15, 0.20],
        "kind": "aff",
    },
    {
        "key": "w_sem",
        "title": "M3B: PBDC Sem Pull-Back ($\\lambda_{\\mathrm{sem}}$)",
        "xlabel": "$\\lambda_{\\mathrm{sem}}$",
        "default": 0.85,
        "values": [0.0, 0.40, 0.65, 0.85, 1.10, 1.50],
        "kind": "pbdc",
    },
    {
        "key": "max_shift_r",
        "title": "M3B: PBDC Search Radius ($d_{\\max}$, px)",
        "xlabel": "$d_{\\max}$ (px)",
        "default": 6,
        "values": [1, 2, 3, 4, 6, 8],
        "kind": "pbdc",
    },
]


def _make_pbdc_cfg(key: str, val: Any) -> PBDCConfig:
    if key == "w_sem":
        return PBDCConfig(enabled=True, w_sem=float(val))
    if key == "max_shift_r":
        r = int(val)
        return PBDCConfig(
            enabled=True,
            max_shift_r=r,
            candidate_shifts=tuple(range(-r, r + 1)),
            k1_max_outward_px=min(4, r),
        )
    raise ValueError(f"Unknown PBDC param: {key}")


def _eval_pv100_all(
    split_dir: Path,
    dev_cache_pt: Path,
    episodes_per_fold: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    print("[PV100 2m] Loading locked dev split & cached DINOv3-SAT features...", flush=True)
    _, _, dev_recs = load_dev_records(split_dir)
    ep_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_recs)
    img_dir, lbl_dir = resolve_data_dirs(None, None)
    rgb_hwc_by_id: dict[str, np.ndarray] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    rec_by_id: dict[str, dict[str, Any]] = {}
    for row in dev_recs:
        rid = row["id"]
        rgb_chw = read_verified_rgb(img_dir, row)
        rgb_hwc_by_id[rid] = np.transpose(rgb_chw, (1, 2, 0)).copy()
        mask = read_verified_mask(lbl_dir, row)
        mask_by_id[rid] = mask
        rec_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}

    all_ids = [r["id"] for r in dev_recs]
    pos_ids = [r["id"] for r in dev_recs if r["positive"]]
    bundle = torch.load(dev_cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    tokens_by_key = {
        k: {rid: to_tokens_norm(feats[rid][k], device) for rid in all_ids}
        for k in ("deep_32", "deep_64", "mid_64")
    }
    caches = {
        k: build_pairwise_topk_cache(tokens_by_key[k], mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }

    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in ep_lock["episodes"]:
        if int(ep["episode"]) < episodes_per_fold:
            ep_by_k[int(ep["k"])].append(ep)

    core_cfg = get_clean_core_config()
    aff_default = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in all_ids}
    pipe_default = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    pbdc_specs = _pbdc_v31_ablation_specs()

    # Part 1: v3.1 Aligned Internal Ablation (K=1, 3: A3 + A4_v31; K=5: A3 + A4_v31 + I3/I4/I5)
    abl_methods: dict[str, dict[str, Any]] = defaultdict(dict)
    for k in (1, 3, 5):
        eps = ep_by_k[k]
        ev_map: dict[str, list[dict]] = defaultdict(list)
        k_specs = pbdc_specs if k == 5 else [pbdc_specs[0]]
        for ep in eps:
            q_ids = ep["query_ids"]
            q_recs = [rec_by_id[q] for q in q_ids]
            out_a3 = pipe_default.run_episode(caches, tokens_by_key, mask_by_id, aff_default, ep["support_ids"], q_ids)
            p_a3, s32_a3 = out_a3.to_legacy_outputs()
            m_a3 = out_a3.coarse_masks_512()
            ev_map["A3_Plus_M3A_Aff"].append(evaluate_binary_episode(q_recs, m_a3, p_a3, s32_a3))
            refined = _run_pbdc_batch(q_ids, m_a3, p_a3, k_specs, rgb_hwc_by_id, k, workers)
            for name, _ in k_specs:
                ev_map[name].append(evaluate_binary_episode(q_recs, refined[name], p_a3, s32_a3))
        for mname, evs in ev_map.items():
            abl_methods[mname][str(k)] = aggregate_binary_episodes(evs, episodes_per_fold=episodes_per_fold)
        print(f"[PV100 2m] v3.1 Ablation K={k} done ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    abl_comp: dict[str, dict[str, Any]] = {
        "v31_vs_A3_Plus_M3A_Aff": {
            str(k): compare_methods(abl_methods["A4_v31_PBDC_Pure"][str(k)], abl_methods["A3_Plus_M3A_Aff"][str(k)])
            for k in (1, 3, 5)
        }
    }
    for ref in ("I3_v31_wo_AsymCore", "I4_v31_wo_SemPull", "I5_v31_wo_SectorFB"):
        abl_comp[f"v31_vs_{ref}"] = {
            "5": compare_methods(abl_methods["A4_v31_PBDC_Pure"]["5"], abl_methods[ref]["5"])
        }

    # Part 2: 6-Parameter Sensitivity Sweep (K=5, 2 eps/fold = all 50 sites covered twice)
    sens_res: dict[str, list[dict[str, Any]]] = {}
    pbdc_default_spec = [("v31", PBDCConfig(enabled=True))]
    sens_ep_per_fold = 2
    eps5_sens = [ep for ep in ep_by_k[5] if int(ep["episode"]) < sens_ep_per_fold]

    k5_sens_a3: list[tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    ev_def_sens = []
    for ep in eps5_sens:
        q_ids = ep["query_ids"]
        q_recs = [rec_by_id[q] for q in q_ids]
        out_a3 = pipe_default.run_episode(caches, tokens_by_key, mask_by_id, aff_default, ep["support_ids"], q_ids)
        p_a3, s32_a3 = out_a3.to_legacy_outputs()
        m_a3 = out_a3.coarse_masks_512()
        k5_sens_a3.append((q_recs, m_a3, p_a3, s32_a3))
        ref_def = _run_pbdc_batch(q_ids, m_a3, p_a3, pbdc_default_spec, rgb_hwc_by_id, 5, workers)["v31"]
        ev_def_sens.append(evaluate_binary_episode(q_recs, ref_def, p_a3, s32_a3))
    def_macro_sens = aggregate_binary_episodes(ev_def_sens, episodes_per_fold=sens_ep_per_fold)["macro"]

    for spec in SENSITIVITY_SPECS:
        key = spec["key"]
        rows: list[dict[str, Any]] = []
        if spec["kind"] == "pbdc":
            non_def = [v for v in spec["values"] if v != spec["default"]]
            sweep_specs = [(str(v), _make_pbdc_cfg(key, v)) for v in non_def]
            ev_by_v: dict[str, list[dict]] = defaultdict(list)
            for q_recs, m_a3, p_a3, s32_a3 in k5_sens_a3:
                q_ids = [r["id"] for r in q_recs]
                refined = _run_pbdc_batch(q_ids, m_a3, p_a3, sweep_specs, rgb_hwc_by_id, 5, workers)
                for sv, _ in sweep_specs:
                    ev_by_v[sv].append(evaluate_binary_episode(q_recs, refined[sv], p_a3, s32_a3))
            for v in spec["values"]:
                m = def_macro_sens if v == spec["default"] else aggregate_binary_episodes(ev_by_v[str(v)], episodes_per_fold=sens_ep_per_fold)["macro"]
                rows.append({"value": v, "dice": m["dice"], "iou": m["iou"], "bf1_1px": m["bf1_1px"], "bf1_2px": m["boundary_f1"], "neg_fp": m["neg_fp"]})
        else:
            for v in spec["values"]:
                if v == spec["default"]:
                    m = def_macro_sens
                    rows.append({"value": v, "dice": m["dice"], "iou": m["iou"], "bf1_1px": m["bf1_1px"], "bf1_2px": m["boundary_f1"], "neg_fp": m["neg_fp"]})
                    continue
                if spec["kind"] == "aff":
                    cfg_v = replace(core_cfg, aff_tau=float(v))
                    aff_v = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=float(v)) for rid in all_ids}
                else:
                    cfg_v = replace(core_cfg, **{key: float(v)})
                    aff_v = aff_default
                pipe_v = ThreeModulePipeline(cfg_v, refiner_cfg=None)
                evs = []
                for ep in eps5_sens:
                    q_ids = ep["query_ids"]
                    q_recs = [rec_by_id[q] for q in q_ids]
                    out_v = pipe_v.run_episode(caches, tokens_by_key, mask_by_id, aff_v, ep["support_ids"], q_ids)
                    p_v, s32_v = out_v.to_legacy_outputs()
                    m_v = out_v.coarse_masks_512()
                    ref_v = _run_pbdc_batch(q_ids, m_v, p_v, pbdc_default_spec, rgb_hwc_by_id, 5, workers)["v31"]
                    evs.append(evaluate_binary_episode(q_recs, ref_v, p_v, s32_v))
                agg = aggregate_binary_episodes(evs, episodes_per_fold=sens_ep_per_fold)
                m = agg["macro"]
                rows.append({"value": v, "dice": m["dice"], "iou": m["iou"], "bf1_1px": m["bf1_1px"], "bf1_2px": m["boundary_f1"], "neg_fp": m["neg_fp"]})
        sens_res[key] = rows
        print(f"[PV100 2m] Sensitivity {key} done ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    return {"ablation": {"methods": dict(abl_methods), "comparisons": abl_comp}, "sensitivity_k5": sens_res}


def _eval_pv663_all(
    split_json: Path,
    episodes_json: Path,
    source_dir: Path,
    dino_feat_dir: Path,
    episodes_per_k: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    sp = read_json(split_json)
    ep_lock = read_json(episodes_json)
    ep_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ep in ep_lock["episodes"]:
        k = int(ep["k"])
        if len(ep_by_k[k]) < episodes_per_k:
            ep_by_k[k].append(ep)

    rec_by_id = {r["id"]: r for r in sp["records"]}
    query_recs_raw = [r for r in sp["records"] if r["partition"] == "query_set"]
    query_ids = [r["id"] for r in query_recs_raw]
    unique_sup_ids = sorted({sid for eps in ep_by_k.values() for ep in eps for sid in ep["support_ids"]})
    used_recs = [rec_by_id[rid] for rid in unique_sup_ids] + query_recs_raw
    used_ids = [r["id"] for r in used_recs]

    img_dir, msk_dir = resolve_pv663_dirs(None, None)
    print(f"[PV663 0.8m] Extracting/loading DINOv3-SAT features for {len(used_recs)} images...", flush=True)
    rgb_512_by_id, mask_512_by_id = extract_pv663_features_to_dir(
        used_recs, image_dir=img_dir, mask_dir=msk_dir, feat_dir=dino_feat_dir, source_dir=source_dir, device=device
    )
    query_records = [{**r, "mask": mask_512_by_id[r["id"]], "y32": area_target_32(mask_512_by_id[r["id"]])} for r in query_recs_raw]

    tokens_by_key = {
        k: LazyDiskTokenMap(dino_feat_dir, k, device, preload_ids=unique_sup_ids)
        for k in ("deep_32", "deep_64", "mid_64")
    }
    caches = {
        k: build_pairwise_topk_cache_streaming(
            dino_feat_dir, k, mask_512_by_id, used_ids, unique_sup_ids, grid_hw=ghw, core_purity=0.85, top_m=3, device=device
        )
        for k, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64))
    }

    core_cfg = get_clean_core_config()
    aff_default = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=core_cfg.aff_tau) for rid in query_ids}
    pipe_default = ThreeModulePipeline(core_cfg, refiner_cfg=None)
    pbdc_specs = _pbdc_v31_ablation_specs()

    # Part 1: v3.1 Aligned Internal Ablation (K=1 full 5 eps: A3 + A4_v31; K=5 3 eps: A3 + A4_v31 + I3/I4/I5)
    abl_methods: dict[str, dict[str, Any]] = defaultdict(dict)
    k5_sens_a3: tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
    k5_sens_def_ev: dict[str, Any] | None = None

    for k, k_eps_limit, k_specs in (
        (1, episodes_per_k, [pbdc_specs[0]]),
        (5, 3, pbdc_specs),
    ):
        eps = ep_by_k[k][:k_eps_limit]
        ev_map: dict[str, list[dict]] = defaultdict(list)
        for ep_i, ep in enumerate(eps):
            out_a3 = pipe_default.run_episode(caches, tokens_by_key, mask_512_by_id, aff_default, ep["support_ids"], query_ids)
            p_a3, s32_a3 = out_a3.to_legacy_outputs()
            m_a3 = out_a3.coarse_masks_512()
            ev_map["A3_Plus_M3A_Aff"].append(evaluate_pv663_episode(query_records, m_a3, p_a3, s32_a3))
            refined = _run_pbdc_batch(query_ids, m_a3, p_a3, k_specs, rgb_512_by_id, k, workers)
            for name, _ in k_specs:
                ev_item = evaluate_pv663_episode(query_records, refined[name], p_a3, s32_a3)
                ev_map[name].append(ev_item)
                if k == 5 and ep_i == 0 and name == "A4_v31_PBDC_Pure":
                    k5_sens_a3 = (m_a3, p_a3, s32_a3)
                    k5_sens_def_ev = ev_item
        for mname, evs in ev_map.items():
            abl_methods[mname][str(k)] = aggregate_pv663_episodes(evs, expected_episodes=len(eps))
        print(f"[PV663 0.8m] v3.1 Ablation K={k} ({len(eps)} eps) done ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    abl_comp: dict[str, dict[str, Any]] = {
        "v31_vs_A3_Plus_M3A_Aff": {
            str(k): compare_pv663_methods(abl_methods["A4_v31_PBDC_Pure"][str(k)], abl_methods["A3_Plus_M3A_Aff"][str(k)])
            for k in (1, 5)
        }
    }
    for ref in ("I3_v31_wo_AsymCore", "I4_v31_wo_SemPull", "I5_v31_wo_SectorFB"):
        abl_comp[f"v31_vs_{ref}"] = {
            "5": compare_pv663_methods(abl_methods["A4_v31_PBDC_Pure"]["5"], abl_methods[ref]["5"])
        }

    # Part 2: 6-Parameter Sensitivity Sweep (K=5, Ep0 covering ALL 116 geo sites & 510 query imgs)
    assert k5_sens_a3 is not None and k5_sens_def_ev is not None
    def_macro_sens = aggregate_pv663_episodes([k5_sens_def_ev], expected_episodes=1)["macro"]
    sens_res: dict[str, list[dict[str, Any]]] = {}
    pbdc_default_spec = [("v31", PBDCConfig(enabled=True))]
    ep5_0 = ep_by_k[5][0]

    for spec in SENSITIVITY_SPECS:
        key = spec["key"]
        rows: list[dict[str, Any]] = []
        if spec["kind"] == "pbdc":
            non_def = [v for v in spec["values"] if v != spec["default"]]
            sweep_specs = [(str(v), _make_pbdc_cfg(key, v)) for v in non_def]
            m_a3, p_a3, s32_a3 = k5_sens_a3
            refined = _run_pbdc_batch(query_ids, m_a3, p_a3, sweep_specs, rgb_512_by_id, 5, workers)
            for v in spec["values"]:
                if v == spec["default"]:
                    m = def_macro_sens
                else:
                    ev_v = evaluate_pv663_episode(query_records, refined[str(v)], p_a3, s32_a3)
                    m = aggregate_pv663_episodes([ev_v], expected_episodes=1)["macro"]
                rows.append({"value": v, "dice": m["dice"], "iou": m["iou"], "bf1_1px": m["bf1_1px"], "bf1_2px": m["boundary_f1"]})
        else:
            for v in spec["values"]:
                if v == spec["default"]:
                    m = def_macro_sens
                    rows.append({"value": v, "dice": m["dice"], "iou": m["iou"], "bf1_1px": m["bf1_1px"], "bf1_2px": m["boundary_f1"]})
                    continue
                if spec["kind"] == "aff":
                    cfg_v = replace(core_cfg, aff_tau=float(v))
                    aff_v = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=float(v)) for rid in query_ids}
                else:
                    cfg_v = replace(core_cfg, **{key: float(v)})
                    aff_v = aff_default
                pipe_v = ThreeModulePipeline(cfg_v, refiner_cfg=None)
                out_v = pipe_v.run_episode(caches, tokens_by_key, mask_512_by_id, aff_v, ep5_0["support_ids"], query_ids)
                p_v, s32_v = out_v.to_legacy_outputs()
                m_v = out_v.coarse_masks_512()
                ref_v = _run_pbdc_batch(query_ids, m_v, p_v, pbdc_default_spec, rgb_512_by_id, 5, workers)["v31"]
                ev_v = evaluate_pv663_episode(query_records, ref_v, p_v, s32_v)
                m = aggregate_pv663_episodes([ev_v], expected_episodes=1)["macro"]
                rows.append({"value": v, "dice": m["dice"], "iou": m["iou"], "bf1_1px": m["bf1_1px"], "bf1_2px": m["boundary_f1"]})
        sens_res[key] = rows
        print(f"[PV663 0.8m] Sensitivity {key} done ({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    return {"ablation": {"methods": dict(abl_methods), "comparisons": abl_comp}, "sensitivity_k5": sens_res}


def plot_dual_res_sensitivity(pv100_sens: dict[str, list[dict]], pv663_sens: dict[str, list[dict]], out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), dpi=300)
    axes_flat = axes.ravel()

    for idx, spec in enumerate(SENSITIVITY_SPECS):
        ax = axes_flat[idx]
        key = spec["key"]
        r2 = pv100_sens[key]
        r08 = pv663_sens[key]
        xs = [row["value"] for row in r2]

        d2 = [row["dice"] * 100.0 for row in r2]
        b2 = [row["bf1_2px"] * 100.0 for row in r2]
        d08 = [row["dice"] * 100.0 for row in r08]
        b08 = [row["bf1_2px"] * 100.0 for row in r08]

        ax.plot(xs, d08, "-o", color="#1f77b4", lw=2.0, ms=5, label="0.8m Dice")
        ax.plot(xs, d2, "-s", color="#2ca02c", lw=2.0, ms=5, label="2.0m Dice")
        ax.plot(xs, b08, "--o", color="#d62728", lw=1.8, ms=4.5, alpha=0.85, label="0.8m BF1@2")
        ax.plot(xs, b2, "--s", color="#ff7f0e", lw=1.8, ms=4.5, alpha=0.85, label="2.0m BF1@2")

        dv = spec["default"]
        ax.axvline(dv, color="#333333", ls=":", lw=1.4, label=f"Locked ({dv})")
        ax.set_title(spec["title"], fontweight="bold")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel("Score (%)")
        ax.grid(True, ls="--", alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, ncol=2)

    fig.suptitle(
        "Dual-Resolution Hyperparameter Sensitivity Analysis (K=5, One-At-A-Time, Locked v3.1-PBDC-Pure)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "sensitivity_dual_res_2x3.png", dpi=300)
    fig.savefig(out_dir / "sensitivity_dual_res_2x3.pdf")
    plt.close(fig)


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "=== v3.1-PBDC-Pure ALIGNED INTERNAL ABLATION + 6-PARAM DUAL-RESOLUTION SENSITIVITY ===",
        f"Total Elapsed: {summary['elapsed_sec']/60:.1f} min",
    ]
    for ds_label, ds_key in (("PV-100scence (2m, 50 Dev Sites)", "pv100_2m"), ("PV663 (0.8m, 116 Geo Sites)", "pv663_08m")):
        ds = summary[ds_key]
        lines.append(f"\n================ {ds_label}: v3.1 INTERNAL ABLATION ================")
        for k in ("1", "3", "5"):
            present = [
                mname
                for mname in ("A3_Plus_M3A_Aff", "A4_v31_PBDC_Pure", "I3_v31_wo_AsymCore", "I4_v31_wo_SemPull", "I5_v31_wo_SectorFB")
                if k in ds["ablation"]["methods"].get(mname, {})
            ]
            if not present:
                continue
            lines.append(f"--- K={k} ---")
            for mname in present:
                m = ds["ablation"]["methods"][mname][k]["macro"]
                neg_s = f" NegFP={m['neg_fp']:.4f}" if "neg_fp" in m else ""
                lines.append(
                    f"  {mname:20s}: Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
                    f"BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                    f"Rec={m['recall']:.4f} Prec={m['precision']:.4f}{neg_s}"
                )
        lines.append("--- Paired Site Bootstrap Deltas (A4_v31 vs Ablated, positive = mechanism helps) ---")
        for cname, cdict in ds["ablation"]["comparisons"].items():
            lines.append(f"[{cname}]")
            for k in sorted(cdict.keys(), key=int):
                c = cdict[k]
                bf1_1_s = ""
                if "bf1_1px" in c:
                    bf1_1_s = f"dBF1@1={c['bf1_1px']['mean_delta']:+.4f} CI{[round(x,4) for x in c['bf1_1px']['ci95']]} | "
                lines.append(
                    f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} | "
                    f"{bf1_1_s}"
                    f"dBF1@2={c['boundary_f1']['mean_delta']:+.4f} CI{[round(x,4) for x in c['boundary_f1']['ci95']]}"
                )

    lines.append("\n================ DUAL-RESOLUTION SENSITIVITY SUMMARY (K=5) ================")
    for spec in SENSITIVITY_SPECS:
        key = spec["key"]
        dv = spec["default"]
        lines.append(f"\n[{key}] (locked default = {dv})")
        r2 = summary["pv100_2m"]["sensitivity_k5"][key]
        r08 = summary["pv663_08m"]["sensitivity_k5"][key]
        for a, b in zip(r2, r08):
            mark = " <-- LOCKED" if a["value"] == dv else ""
            lines.append(
                f"  val={str(a['value']):>5s} | 2m: Dice={a['dice']:.4f} BF1@2={a['bf1_2px']:.4f} NegFP={a['neg_fp']:.4f} | "
                f"0.8m: Dice={b['dice']:.4f} BF1@2={b['bf1_2px']:.4f}{mark}"
            )
    return "\n".join(lines) + "\n"


def run_all(
    *,
    split_dir: Path,
    dev_cache_pt: Path,
    pv663_split_json: Path,
    pv663_episodes_json: Path,
    source_dir: Path,
    pv663_dino_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else min(4, max(1, os.cpu_count() or 1))
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    res_2m = _eval_pv100_all(split_dir, dev_cache_pt, episodes_per_fold=4, device=run_device, workers=workers)
    res_08m = _eval_pv663_all(
        pv663_split_json, pv663_episodes_json, source_dir, pv663_dino_dir, episodes_per_k=5, device=run_device, workers=workers
    )

    summary = {
        "schema": "v31-ablation-sensitivity-dual-res-v1",
        "elapsed_sec": float(time.perf_counter() - t0),
        "pv100_2m": res_2m,
        "pv663_08m": res_08m,
    }
    write_json(output_dir / "v31_ablation_and_sensitivity_summary.json", summary)
    report = format_report(summary)
    (output_dir / "v31_ablation_and_sensitivity_report.txt").write_text(report, encoding="utf-8")
    plot_dual_res_sensitivity(res_2m["sensitivity_k5"], res_08m["sensitivity_k5"], output_dir)
    print("\n" + report, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3.1 aligned internal ablation + 6-param dual-res sensitivity.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--dev-cache-pt", type=Path, required=True)
    parser.add_argument("--pv663-split-json", type=Path, default=REPO_ROOT / "configs" / "pv663_split_lock.json")
    parser.add_argument("--pv663-episodes-json", type=Path, default=REPO_ROOT / "configs" / "pv663_episodes_lock.json")
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--pv663-dino-dir", type=Path, default=Path("/tmp/pv663_v31_dino_feats"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "v31_ablation_sensitivity")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run_all(
        split_dir=args.split_dir,
        dev_cache_pt=args.dev_cache_pt,
        pv663_split_json=args.pv663_split_json,
        pv663_episodes_json=args.pv663_episodes_json,
        source_dir=args.source_dir,
        pv663_dino_dir=args.pv663_dino_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
