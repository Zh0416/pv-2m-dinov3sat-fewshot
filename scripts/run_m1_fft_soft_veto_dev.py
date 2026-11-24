from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from pvfs.episodes import load_and_verify_episodes
from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.io import write_json
from pvfs.iter3_methods import _to_tokens_norm, compute_dino_affinity_weights
from pvfs.iter5_methods import run_iter5_episode
from pvfs.metrics import aggregate_episodes, compare_methods, evaluate_episode
from pvfs.modules import ThreeModulePipeline, get_m3_pruned_config
from pvfs.modules.fft_soft_veto import FFTSoftVetoConfig, precompute_image_fft
from scripts.run_m1_hardneg_dev import _evaluate_threshold_sweep_for_equal_recall, _interp_at_target_recall

DIAG_IMAGE_IDS = (
    "负样本/PV-93.tif",
    "负样本/PV-100.tif",
    "负样本/PV-96.tif",
    "负样本/PV-86.tif",
    "负样本/PV-92.tif",
    "正样本/PV-80.tif",
    "正样本/PV-55.tif",
    "正样本/PV-49.tif",
    "正样本/PV-73.tif",
)


def _m1_p512_and_s32(sem_out: Any) -> tuple[np.ndarray, np.ndarray]:
    """Upsample Module ① continuous P_sem [4096] -> [512, 512] and pool S_sem -> [32, 32]."""
    hw = sem_out.grid_hw
    p512 = (
        F.interpolate(sem_out.p_sem.reshape(1, 1, hw, hw), size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if hw == 64:
        s32 = F.avg_pool2d(sem_out.s_sem.reshape(1, 1, 64, 64), 2, stride=2)[0, 0].detach().cpu().numpy().astype(np.float32)
    else:
        s32 = sem_out.s_sem.reshape(32, 32).detach().cpu().numpy().astype(np.float32)
    return p512, s32


def format_m1_fft_soft_veto_report(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "=== MODULE 1: SUPPORT-ADAPTIVE FFT SOFT-VETO DEV REPORT (79 IMAGES / 120 PAIRED EPISODES) ===",
        f"Device={summary['device']} | Elapsed={summary['elapsed_sec']/60:.1f} min | "
        f"R_Equiv_MaxDiff={summary['smoke_checks']['r_equiv_max_diff']:.2e} | "
        f"InactiveFallback_MaxDiff={summary['smoke_checks']['inactive_fallback_max_diff']:.2e}",
    ]

    for stage_key, stage_label in (("final", "FINAL OUTPUT (After Self-Rescue + Affinity)"), ("module1", "MODULE 1 OUTPUT (P_sem Upsampled to 512)")):
        lines.append(f"\n================ {stage_label} ================")
        for vname in ("R", "R_NoMidVeto", "R_FFTSoftVeto"):
            lines.append(f"\n--- {vname} ---")
            for k in ("1", "3", "5"):
                agg = summary[stage_key]["methods"][vname][k]
                m = agg["macro"]
                s5r = agg["strata"].get("small_lt5pct", {}).get("recall", 0.0)
                t1r = agg["strata"].get("tiny_lt1pct", {}).get("recall", 0.0)
                eq_r = summary[stage_key]["equal_recall_vs_R"][vname][k]
                eq_n = summary[stage_key]["equal_recall_vs_NoMid"][vname][k]
                lines.append(
                    f"  K={k}: AP512={m['ap_512']:.4f} Dice={m['dice']:.4f} Rec={m['recall']:.4f} Prec={m['precision']:.4f} "
                    f"NegFP={m['neg_fp']:.4f} PosBgFP={m['pos_bg_fp']:.4f} Sm<5R={s5r:.4f} Tiny<1R={t1r:.4f} | "
                    f"EqRec(R)[NegFP={eq_r['neg_site_fp_at_recall']:.4f}, BgFP={eq_r['pos_bg_fp_at_recall']:.4f}, thr={eq_r['matched_threshold']:.3f}] | "
                    f"EqRec(NoMid)[NegFP={eq_n['neg_site_fp_at_recall']:.4f}, BgFP={eq_n['pos_bg_fp_at_recall']:.4f}]"
                )

        lines.append(f"\n--- Paired Site Bootstrap Comparisons ({stage_key}) ---")
        for cname, by_k in summary[stage_key]["comparisons"].items():
            lines.append(f"[{cname}]")
            cand_v, ref_v = cname.split("_vs_", 1)
            for k in ("1", "3", "5"):
                c = by_k[k]
                d_bgfp = (
                    summary[stage_key]["methods"][cand_v][k]["macro"]["pos_bg_fp"]
                    - summary[stage_key]["methods"][ref_v][k]["macro"]["pos_bg_fp"]
                )
                lines.append(
                    f"  K={k}: dDice={c['dice']['mean_delta']:+.4f} CI{[round(x,4) for x in c['dice']['ci95']]} (f={c['dice']['positive_folds']}/4) | "
                    f"dAP512={c['ap_512']['mean_delta']:+.4f} | dRec={c['recall']['mean_delta']:+.4f} | "
                    f"dNegFP={c['neg_fp']['mean_delta']:+.4f} CI{[round(x,4) for x in c['neg_fp']['ci95']]} (f={c['neg_fp']['improved_folds']}/4) | "
                    f"dPosBgFP={d_bgfp:+.4f} | dSm<5R={c['small_lt5pct_recall_delta']:+.4f}"
                )

    lines.append("\n================ FFT VETO & SELF-RESCUE DIAGNOSTICS ================")
    for k in ("1", "3", "5"):
        d = summary["fft_diagnostics"][k]
        sr = summary["self_rescue_diagnostics"][k]
        lines.append(
            f"  K={k}: BandSel={d['band_selected_rate']:.1%} ActiveEp={d['active_episode_rate']:.1%} "
            f"Reasons={d['fallback_reasons']} | "
            f"PenRate(all={d['penalty_rate_all']:.4f}, cand={d['penalty_rate_cand']:.3f}, "
            f"pos_q={d['penalty_rate_pos_queries']:.4f}, neg_q={d['penalty_rate_neg_queries']:.4f}) "
            f"MeanPen(penalized)={d['mean_penalty_penalized']:.4f} | "
            f"SelfRescue[R: pos={sr['R']['pos_rate']:.3f} neg={sr['R']['neg_rate']:.3f} | "
            f"NoMid: pos={sr['R_NoMidVeto']['pos_rate']:.3f} neg={sr['R_NoMidVeto']['neg_rate']:.3f} | "
            f"FFT: pos={sr['R_FFTSoftVeto']['pos_rate']:.3f} neg={sr['R_FFTSoftVeto']['neg_rate']:.3f}]"
        )

    lines.append("\n================ PER-IMAGE HARD CASE DIAGNOSTICS (FINAL) ================")
    for k in ("1", "3", "5"):
        lines.append(f"\n[K={k}]")
        for qid in DIAG_IMAGE_IDS:
            row = summary["hard_case_diagnostics"][k].get(qid)
            if not row:
                continue
            if not row["positive"]:
                lines.append(
                    f"  {qid:18s}: NegFP [R={row['R']['final_neg_fp']:.4f} | NoMid={row['R_NoMidVeto']['final_neg_fp']:.4f} | FFT={row['R_FFTSoftVeto']['final_neg_fp']:.4f}] "
                    f"M1_NegFP [R={row['R']['m1_neg_fp']:.4f} | NoMid={row['R_NoMidVeto']['m1_neg_fp']:.4f} | FFT={row['R_FFTSoftVeto']['m1_neg_fp']:.4f}] "
                    f"PenRate={row['fft_diag']['penalty_rate_all']:.4f} CandPenRate={row['fft_diag']['penalty_rate_cand']:.3f} MeanEbg={row['fft_diag']['mean_e_bg_g_active']:+.3f}"
                )
            else:
                lines.append(
                    f"  {qid:18s}: Dice [R={row['R']['final_dice']:.4f} | NoMid={row['R_NoMidVeto']['final_dice']:.4f} | FFT={row['R_FFTSoftVeto']['final_dice']:.4f}] "
                    f"Rec [R={row['R']['final_recall']:.4f} | NoMid={row['R_NoMidVeto']['final_recall']:.4f} | FFT={row['R_FFTSoftVeto']['final_recall']:.4f}] "
                    f"BgFP [R={row['R']['final_pos_bg_fp']:.4f} | NoMid={row['R_NoMidVeto']['final_pos_bg_fp']:.4f} | FFT={row['R_FFTSoftVeto']['final_pos_bg_fp']:.4f}] "
                    f"PenRate={row['fft_diag']['penalty_rate_all']:.4f}"
                )

    return "\n".join(lines) + "\n"


def run_m1_fft_soft_veto_dev_experiment(
    *,
    split_dir: Path,
    cache_pt: Path,
    config_json: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    missing_feat = [rid for rid in all_ids if rid not in feats]
    if missing_feat:
        raise KeyError(f"dev_features_v1.pt missing {len(missing_feat)} dev IDs, e.g. {missing_feat[:3]}")

    # Step 1: Verify image/mask SHA-256 alignment and precompute 32x32 Hann FFT spectra once
    print(f"[M1-FFT] Verifying 79 dev images/masks & precomputing 32x32 Hann FFT spectra...", flush=True)
    records_by_id: dict[str, dict[str, Any]] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    fft_pre_by_id: dict[str, Any] = {}
    for row in dev_records:
        rid = row["id"]
        rgb_chw = read_verified_rgb(img_dir, row)
        mask = read_verified_mask(lbl_dir, row)
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask
        fft_pre_by_id[rid] = precompute_image_fft(rid, rgb_chw, window_size=32, grid_hw=64)
        # Alignment assertion: 64x64 grid centers and 60x60 interior mask on 512x512
        assert fft_pre_by_id[rid].rows.shape == (64,) and int(fft_pre_by_id[rid].interior.sum()) == 3600

    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
        if run_device == "cuda":
            torch.cuda.empty_cache()

    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in all_ids}

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    m3_cfg = get_m3_pruned_config()
    no_mid_cfg = replace(m3_cfg, name="R_NoMidVeto", beta_veto=0.0)
    fft_cfg = FFTSoftVetoConfig.from_json(config_json, enabled=True)
    fft_impossible_cfg = replace(fft_cfg, min_pv_cand=999999)  # forces episode fallback for smoke check

    pipe_r = ThreeModulePipeline(m3_cfg)
    pipe_nomid = ThreeModulePipeline(no_mid_cfg)
    pipe_fft = ThreeModulePipeline(m3_cfg, fft_veto_cfg=fft_cfg)
    pipe_fft_fallback = ThreeModulePipeline(m3_cfg, fft_veto_cfg=fft_impossible_cfg)

    # Step 2: Smoke checks (R exact match to legacy run_iter5_episode + inactive FFT exact match to R_NoMidVeto)
    r_equiv_max_diff = 0.0
    fallback_max_diff = 0.0
    for k_chk in (1, 3, 5):
        ep0 = ep_by_k[k_chk][0]
        pL, _ = run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep0["support_ids"], ep0["query_ids"], m3_cfg)
        outR = pipe_r.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep0["support_ids"], ep0["query_ids"])
        pR, _ = outR.to_legacy_outputs()
        outNM = pipe_nomid.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep0["support_ids"], ep0["query_ids"])
        pNM, _ = outNM.to_legacy_outputs()
        outFB = pipe_fft_fallback.run_episode(
            caches, tokens_by_key, mask_by_id, aff_by_id, ep0["support_ids"], ep0["query_ids"], fft_pre_by_id=fft_pre_by_id
        )
        pFB, _ = outFB.to_legacy_outputs()
        assert not outFB.calibration.fft_veto_active
        assert abs(outFB.calibration.bias - outNM.calibration.bias) == 0.0
        assert abs(outFB.calibration.temp - outNM.calibration.temp) == 0.0
        for qid in ep0["query_ids"]:
            r_equiv_max_diff = max(r_equiv_max_diff, float(np.max(np.abs(pR[qid] - pL[qid]))))
            fallback_max_diff = max(fallback_max_diff, float(np.max(np.abs(pFB[qid] - pNM[qid]))))
    if r_equiv_max_diff > 1e-7 or fallback_max_diff > 1e-7:
        raise RuntimeError(f"Smoke check failed: r_equiv={r_equiv_max_diff}, fallback={fallback_max_diff}")
    print(f"[M1-FFT] Smoke checks PASSED (r_equiv={r_equiv_max_diff:.2e}, fallback_vs_NoMid={fallback_max_diff:.2e})", flush=True)

    # Step 3: Full 120-episode paired evaluation across R, R_NoMidVeto, R_FFTSoftVeto
    variants = (
        ("R", pipe_r),
        ("R_NoMidVeto", pipe_nomid),
        ("R_FFTSoftVeto", pipe_fft),
    )
    thresholds = np.linspace(0.05, 0.95, 181)

    stage_methods: dict[str, dict[str, dict[str, Any]]] = {
        "final": {v: {} for v, _ in variants},
        "module1": {v: {} for v, _ in variants},
    }
    stage_eq_r: dict[str, dict[str, dict[str, Any]]] = {
        "final": {v: {} for v, _ in variants},
        "module1": {v: {} for v, _ in variants},
    }
    stage_eq_nomid: dict[str, dict[str, dict[str, Any]]] = {
        "final": {v: {} for v, _ in variants},
        "module1": {v: {} for v, _ in variants},
    }
    fft_diag_by_k: dict[str, dict[str, Any]] = {}
    sr_diag_by_k: dict[str, dict[str, Any]] = {}
    hard_case_by_k: dict[str, dict[str, Any]] = {}

    for k in sorted(ep_by_k.keys()):
        eps = ep_by_k[k]
        final_evals: dict[str, list[dict]] = {v: [] for v, _ in variants}
        m1_evals: dict[str, list[dict]] = {v: [] for v, _ in variants}
        final_sweep_data: dict[str, list[tuple[list[dict], dict[str, np.ndarray]]]] = {v: [] for v, _ in variants}
        m1_sweep_data: dict[str, list[tuple[list[dict], dict[str, np.ndarray]]]] = {v: [] for v, _ in variants}

        sr_counts: dict[str, dict[str, int]] = {
            v: {"pos_trig": 0, "pos_tot": 0, "neg_trig": 0, "neg_tot": 0} for v, _ in variants
        }
        band_sel_n = 0
        active_ep_n = 0
        reasons: Counter[str] = Counter()
        band_lows: list[float] = []
        band_highs: list[float] = []
        band_peaks: list[int] = []
        fft_q_stats: dict[str, list[float]] = defaultdict(list)
        per_img_fft_diag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for ep in eps:
            sup_ids = ep["support_ids"]
            q_ids = ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]

            for vname, pipe in variants:
                out = pipe.run_episode(
                    caches,
                    tokens_by_key,
                    mask_by_id,
                    aff_by_id,
                    sup_ids,
                    q_ids,
                    fft_pre_by_id=(fft_pre_by_id if vname == "R_FFTSoftVeto" else None),
                )
                p_fin, s32_fin = out.to_legacy_outputs()
                p_m1: dict[str, np.ndarray] = {}
                s32_m1: dict[str, np.ndarray] = {}
                for qid in q_ids:
                    p_m1[qid], s32_m1[qid] = _m1_p512_and_s32(out.semantic_by_id[qid])
                    trig = bool(out.recovery_by_id[qid].diagnostics.get("self_rescue_triggered", False))
                    if records_by_id[qid]["positive"]:
                        sr_counts[vname]["pos_tot"] += 1
                        sr_counts[vname]["pos_trig"] += int(trig)
                    else:
                        sr_counts[vname]["neg_tot"] += 1
                        sr_counts[vname]["neg_trig"] += int(trig)

                final_evals[vname].append(evaluate_episode(q_recs, p_fin, s32_fin))
                m1_evals[vname].append(evaluate_episode(q_recs, p_m1, s32_m1))
                final_sweep_data[vname].append((q_recs, p_fin))
                m1_sweep_data[vname].append((q_recs, p_m1))

                if vname == "R_FFTSoftVeto":
                    cd = out.calibration.diagnostics
                    band_sel_n += int(bool(cd.get("fft_veto_band_selected", False)))
                    active_ep_n += int(bool(cd.get("fft_veto_active", False)))
                    reasons[str(cd.get("fft_veto_reason", "unknown"))] += 1
                    if cd.get("fft_veto_band_low") is not None:
                        band_lows.append(float(cd["fft_veto_band_low"]))
                        band_highs.append(float(cd["fft_veto_band_high"]))
                        band_peaks.append(int(cd["fft_veto_band_peak_r"]))
                    for qid in q_ids:
                        qd = out.semantic_by_id[qid].diagnostics
                        fft_q_stats["penalty_rate_all"].append(float(qd["penalty_rate_all"]))
                        fft_q_stats["penalty_rate_cand"].append(float(qd["penalty_rate_cand"]))
                        if qd["penalized_tokens"] > 0:
                            fft_q_stats["mean_penalty_penalized"].append(float(qd["mean_penalty_penalized"]))
                        if records_by_id[qid]["positive"]:
                            fft_q_stats["penalty_rate_pos_queries"].append(float(qd["penalty_rate_all"]))
                        else:
                            fft_q_stats["penalty_rate_neg_queries"].append(float(qd["penalty_rate_all"]))
                        if qid in DIAG_IMAGE_IDS:
                            for dk in ("penalty_rate_all", "penalty_rate_cand", "mean_penalty_all", "mean_e_bg_g_active"):
                                per_img_fft_diag[qid][dk].append(float(qd[dk]))

        for vname, _ in variants:
            stage_methods["final"][vname][str(k)] = aggregate_episodes(final_evals[vname], episodes_per_fold=10)
            stage_methods["module1"][vname][str(k)] = aggregate_episodes(m1_evals[vname], episodes_per_fold=10)

        # Equal-recall threshold sweeps for both final and module1 outputs
        for stage_key, sw_data in (("final", final_sweep_data), ("module1", m1_sweep_data)):
            r_rec_target = stage_methods[stage_key]["R"][str(k)]["macro"]["recall"]
            nm_rec_target = stage_methods[stage_key]["R_NoMidVeto"][str(k)]["macro"]["recall"]
            for vname, _ in variants:
                curve = _evaluate_threshold_sweep_for_equal_recall(sw_data[vname], thresholds)
                stage_eq_r[stage_key][vname][str(k)] = _interp_at_target_recall(curve, r_rec_target)
                stage_eq_nomid[stage_key][vname][str(k)] = _interp_at_target_recall(curve, nm_rec_target)

        fft_diag_by_k[str(k)] = {
            "episodes": len(eps),
            "band_selected_count": band_sel_n,
            "band_selected_rate": float(band_sel_n / max(1, len(eps))),
            "active_episode_count": active_ep_n,
            "active_episode_rate": float(active_ep_n / max(1, len(eps))),
            "fallback_reasons": dict(reasons),
            "band_low_mean": float(np.mean(band_lows)) if band_lows else None,
            "band_high_mean": float(np.mean(band_highs)) if band_highs else None,
            "band_peak_r_counts": dict(Counter(band_peaks)),
            "penalty_rate_all": float(np.mean(fft_q_stats["penalty_rate_all"])) if fft_q_stats["penalty_rate_all"] else 0.0,
            "penalty_rate_cand": float(np.mean(fft_q_stats["penalty_rate_cand"])) if fft_q_stats["penalty_rate_cand"] else 0.0,
            "penalty_rate_pos_queries": float(np.mean(fft_q_stats["penalty_rate_pos_queries"])) if fft_q_stats["penalty_rate_pos_queries"] else 0.0,
            "penalty_rate_neg_queries": float(np.mean(fft_q_stats["penalty_rate_neg_queries"])) if fft_q_stats["penalty_rate_neg_queries"] else 0.0,
            "mean_penalty_penalized": float(np.mean(fft_q_stats["mean_penalty_penalized"])) if fft_q_stats["mean_penalty_penalized"] else 0.0,
        }
        sr_diag_by_k[str(k)] = {
            vname: {
                "pos_rate": float(sr_counts[vname]["pos_trig"] / max(1, sr_counts[vname]["pos_tot"])),
                "neg_rate": float(sr_counts[vname]["neg_trig"] / max(1, sr_counts[vname]["neg_tot"])),
            }
            for vname, _ in variants
        }

        hc: dict[str, Any] = {}
        for qid in DIAG_IMAGE_IDS:
            if qid not in records_by_id:
                continue
            pos = bool(records_by_id[qid]["positive"])
            entry: dict[str, Any] = {"positive": pos}
            for vname, _ in variants:
                fin_im = stage_methods["final"][vname][str(k)]["images"][qid]
                m1_im = stage_methods["module1"][vname][str(k)]["images"][qid]
                if pos:
                    entry[vname] = {
                        "final_dice": fin_im["dice"],
                        "final_recall": fin_im["recall"],
                        "final_pos_bg_fp": fin_im["pos_bg_fp"],
                        "m1_dice": m1_im["dice"],
                        "m1_recall": m1_im["recall"],
                        "m1_pos_bg_fp": m1_im["pos_bg_fp"],
                    }
                else:
                    entry[vname] = {
                        "final_neg_fp": fin_im["neg_fp"],
                        "m1_neg_fp": m1_im["neg_fp"],
                    }
            entry["fft_diag"] = {
                dk: float(np.mean(per_img_fft_diag[qid][dk])) if per_img_fft_diag[qid][dk] else 0.0
                for dk in ("penalty_rate_all", "penalty_rate_cand", "mean_penalty_all", "mean_e_bg_g_active")
            }
            hc[qid] = entry
        hard_case_by_k[str(k)] = hc

        for vname, _ in variants:
            mf = stage_methods["final"][vname][str(k)]["macro"]
            eqr = stage_eq_r["final"][vname][str(k)]
            print(
                f"[M1-FFT K={k}] {vname:15s}: Dice={mf['dice']:.4f} AP512={mf['ap_512']:.4f} "
                f"Rec={mf['recall']:.4f} NegFP={mf['neg_fp']:.4f} EqRecNegFP={eqr['neg_site_fp_at_recall']:.4f}",
                flush=True,
            )

    stage_comparisons: dict[str, dict[str, dict[str, Any]]] = {"final": {}, "module1": {}}
    for stage_key in ("final", "module1"):
        for cname, cand_v, ref_v in (
            ("R_FFTSoftVeto_vs_R_NoMidVeto", "R_FFTSoftVeto", "R_NoMidVeto"),
            ("R_FFTSoftVeto_vs_R", "R_FFTSoftVeto", "R"),
            ("R_NoMidVeto_vs_R", "R_NoMidVeto", "R"),
        ):
            stage_comparisons[stage_key][cname] = {
                k: compare_methods(stage_methods[stage_key][cand_v][k], stage_methods[stage_key][ref_v][k])
                for k in ("1", "3", "5")
            }

    summary = {
        "schema": "pv-2m-dinov3sat-m1-fft-soft-veto-dev-v1",
        "device": run_device,
        "elapsed_sec": float(time.perf_counter() - t0),
        "config": fft_cfg.__dict__,
        "smoke_checks": {
            "r_equiv_max_diff": r_equiv_max_diff,
            "inactive_fallback_max_diff": fallback_max_diff,
        },
        "final": {
            "methods": stage_methods["final"],
            "comparisons": stage_comparisons["final"],
            "equal_recall_vs_R": stage_eq_r["final"],
            "equal_recall_vs_NoMid": stage_eq_nomid["final"],
        },
        "module1": {
            "methods": stage_methods["module1"],
            "comparisons": stage_comparisons["module1"],
            "equal_recall_vs_R": stage_eq_r["module1"],
            "equal_recall_vs_NoMid": stage_eq_nomid["module1"],
        },
        "fft_diagnostics": fft_diag_by_k,
        "self_rescue_diagnostics": sr_diag_by_k,
        "hard_case_diagnostics": hard_case_by_k,
    }
    write_json(output_dir / "m1_fft_soft_veto_summary.json", summary)
    report_txt = format_m1_fft_soft_veto_report(summary)
    (output_dir / "m1_fft_soft_veto_report.txt").write_text(report_txt, encoding="utf-8")
    print("\n" + report_txt, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 1 support-adaptive FFT soft-veto paired Dev experiment.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, default=REPO_ROOT / "configs" / "m1_fft_soft_veto_locked.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "m1_fft_soft_veto_dev")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_m1_fft_soft_veto_dev_experiment(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        config_json=args.config_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
