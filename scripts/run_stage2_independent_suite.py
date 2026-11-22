from __future__ import annotations

import argparse
from collections import defaultdict
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
from pvfs.iter5_methods import Iter5Config, run_iter5_episode
from pvfs.metrics import aggregate_episodes, compare_methods, evaluate_episode
from pvfs.modules import ThreeModulePipeline, get_b0_32_config, get_m3_pruned_config
from pvfs.modules.crop_zoom import (
    CropZoomConfig,
    audit_offline_candidate_coverage,
    project_crop_scores_max_increment,
)
from pvfs.modules.deformable_boundary import DeformableBoundaryConfig, refine_mask_deformable_grad
from pvfs.modules.hard_negative import HardNegConfig
from pvfs.phase_boundary import (
    PhaseBoundaryConfig,
    SupportCalibrationBundle,
    compute_oracle_pm8px_mask,
    precompute_image_signals,
    refine_query_image_all_variants,
)
from pvfs.phase_boundary_eval import aggregate_binary_episodes, compare_boundary_methods, evaluate_binary_episode
from scripts.run_m1_hardneg_dev import _evaluate_threshold_sweep_for_equal_recall, _interp_at_target_recall
from scripts.run_m2_cropzoom_dev import _compute_missed_component_recovery, _score_windows_full_image_tokens
from scripts.run_m3_deformgrad_dev import _calibrate_p_floor_loo, _rigid_gt_reachable_ref


def run_stage2_independent_suite(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build shared frozen DINOv3-SAT pairwise caches ONCE, then run strictly independent:
      0) 120-episode L vs R Equivalence Audit
      1) Module ① (R vs RandomBG vs HardNegH + Equal-Recall curves)
      2) Module ② (Stage 0 Coverage Audit + NoRecovery vs SelfRescue_R vs FullTokenCtrl)
      3) Module ③ (Stage 0 Offline GT Ref + Off_R vs Rigid_Grad vs Deformable_Grad)
    Never merges Module ①/②/③ changes together before individual verification.
    """
    t_all = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)

    bundle = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = bundle["features"]
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    records_by_id, mask_by_id, rgb_by_id = {}, {}, {}
    for row in dev_records:
        rid = row["id"]
        mask = read_verified_mask(lbl_dir, row)
        rgb = np.transpose(read_verified_rgb(img_dir, row), (1, 2, 0))
        records_by_id[rid] = {**row, "mask": mask, "y32": area_target_32(mask)}
        mask_by_id[rid] = mask
        rgb_by_id[rid] = rgb

    t_cache0 = time.perf_counter()
    print(f"Building shared pairwise caches on device={run_device}...", flush=True)
    caches: dict[str, dict] = {}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for key, ghw in (("deep_32", 32), ("deep_64", 64), ("mid_64", 64)):
        toks = {rid: _to_tokens_norm(feats[rid][key], run_device) for rid in all_ids}
        caches[key] = build_pairwise_topk_cache(toks, mask_by_id, all_ids, pos_ids, grid_hw=ghw, core_purity=0.85, top_m=3)
        tokens_by_key[key] = toks
        if run_device == "cuda":
            torch.cuda.empty_cache()
    aff_by_id = {rid: compute_dino_affinity_weights(tokens_by_key["deep_64"][rid], tau=0.12) for rid in all_ids}
    cache_sec = time.perf_counter() - t_cache0
    print(f"Shared caches ready in {cache_sec:.1f}s.", flush=True)

    ep_by_k: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes_lock["episodes"]:
        ep_by_k[int(ep["k"])].append(ep)

    m3_cfg = get_m3_pruned_config()
    b0_cfg = get_b0_32_config()

    # =========================================================================
    # 0. FULL 120-EPISODE L vs R EQUIVALENCE AUDIT
    # =========================================================================
    t_eq0 = time.perf_counter()
    equiv_audit: dict[str, Any] = {}
    for e_cfg in (m3_cfg, b0_cfg):
        pipe_eq = ThreeModulePipeline(e_cfg)
        e_by_k: dict[str, Any] = {}
        all_ok = True
        for k in sorted(ep_by_k.keys()):
            max_p_diff = 0.0
            max_s_diff = 0.0
            mask_flips = 0
            n_q = 0
            for ep in ep_by_k[k]:
                pL, sL = run_iter5_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"], e_cfg)
                outR = pipe_eq.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"])
                pR, sR = outR.to_legacy_outputs()
                for qid in ep["query_ids"]:
                    n_q += 1
                    max_p_diff = max(max_p_diff, float(np.max(np.abs(pR[qid] - pL[qid]))))
                    max_s_diff = max(max_s_diff, float(np.max(np.abs(sR[qid] - sL[qid]))))
                    mask_flips += int(np.logical_xor(outR.boundary_by_id[qid].m_coarse, pL[qid] >= 0.50).sum())
                    mask_flips += int(np.logical_xor(outR.boundary_by_id[qid].m_final, outR.boundary_by_id[qid].m_coarse).sum())
            ok_k = bool(mask_flips == 0 and max_p_diff == 0.0 and max_s_diff == 0.0)
            all_ok = all_ok and ok_k
            e_by_k[str(k)] = {
                "query_checks": n_q,
                "max_probs_512_diff": max_p_diff,
                "max_scores_32_diff": max_s_diff,
                "mask_flips": mask_flips,
                "exact_pass": ok_k,
            }
        equiv_audit[e_cfg.name] = {"all_passed": all_ok, "by_k": e_by_k}
        print(f"[0. Equiv L vs R] {e_cfg.name}: all_passed={all_ok}", flush=True)
    write_json(output_dir / "equivalence_summary.json", {"elapsed_sec": time.perf_counter() - t_eq0, "variants": equiv_audit})

    # =========================================================================
    # 1. INDEPENDENT EXPERIMENT: MODULE ① (Hard-Negative Reverification H)
    # =========================================================================
    t_m1_0 = time.perf_counter()
    m1_cfg_path = REPO_ROOT / "configs" / "three_module_exp_m1_hardneg.json"
    m1_specs = (
        ("M1_Base_R", HardNegConfig(enabled=False, mode="off")),
        ("M1_Ctrl_EqualBudgetRandomBG", HardNegConfig.from_json(m1_cfg_path, mode="random_bg")),
        ("M1_Cand_HardNegH", HardNegConfig.from_json(m1_cfg_path, mode="hard_neg_H")),
    )
    m1_pipes = {name: ThreeModulePipeline(m3_cfg, hard_neg_cfg=hcfg) for name, hcfg in m1_specs}
    m1_methods: dict[str, dict[str, Any]] = {name: {} for name, _ in m1_specs}
    m1_diag: dict[str, Any] = {}
    m1_eq_rec: dict[str, Any] = {}
    thr_grid = np.linspace(0.05, 0.95, 37)

    # Also cache R episode outputs per (k, ep_idx) so M2 and M3 share exact R baseline without recomputing!
    r_episode_cache: dict[tuple[int, int], Any] = {}

    for k in sorted(ep_by_k.keys()):
        ep_evals_v: dict[str, list[dict]] = {name: [] for name, _ in m1_specs}
        ep_probs_v: dict[str, list[tuple[list[dict], dict[str, np.ndarray]]]] = {name: [] for name, _ in m1_specs}
        dk: dict[str, Any] = {}
        for name, _ in m1_specs:
            pipe = m1_pipes[name]
            active_eps = 0
            bank_sizes, t_hs, mean_pens = [], [], []
            abstain_reasons: dict[str, int] = defaultdict(int)
            for ep_idx, ep in enumerate(ep_by_k[k]):
                q_recs = [records_by_id[qid] for qid in ep["query_ids"]]
                out = pipe.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, ep["support_ids"], ep["query_ids"])
                if name == "M1_Base_R":
                    r_episode_cache[(k, ep_idx)] = out
                p512, s32 = out.to_legacy_outputs()
                ep_evals_v[name].append(evaluate_episode(q_recs, p512, s32))
                ep_probs_v[name].append((q_recs, p512))
                cd = out.calibration.diagnostics
                if out.calibration.hard_neg_active:
                    active_eps += 1
                    bank_sizes.append(int(cd.get("hard_neg_bank_tokens", 0)))
                    if out.calibration.hard_neg_t_h is not None:
                        t_hs.append(float(out.calibration.hard_neg_t_h))
                else:
                    abstain_reasons[str(cd.get("hard_neg_reason", "disabled"))] += 1
                for qid in ep["query_ids"]:
                    mean_pens.append(float(out.semantic_by_id[qid].diagnostics.get("hard_neg_mean_penalty", 0.0)))

            m1_methods[name][str(k)] = aggregate_episodes(ep_evals_v[name])
            dk[name] = {
                "active_episodes": active_eps,
                "active_rate": float(active_eps / 40.0),
                "mean_bank_tokens": float(np.mean(bank_sizes)) if bank_sizes else 0.0,
                "mean_t_h": float(np.mean(t_hs)) if t_hs else None,
                "mean_query_penalty": float(np.mean(mean_pens)) if mean_pens else 0.0,
                "abstain_reasons": dict(abstain_reasons),
            }
            m = m1_methods[name][str(k)]["macro"]
            print(
                f"[1. M1 K={k}] {name:28s}: Dice={m['dice']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} "
                f"NegFP={m['neg_fp']:.4f} NegMax={m['neg_max_fp']:.4f} PosBgFP={m['pos_bg_fp']:.4f} AP512={m['ap_512']:.4f} Act={active_eps}/40",
                flush=True,
            )
        m1_diag[str(k)] = dk
        base_rec = m1_methods["M1_Base_R"][str(k)]["macro"]["recall"]
        eq_k: dict[str, Any] = {"baseline_R_recall_at_050": base_rec, "variants": {}}
        for name, _ in m1_specs:
            curve = _evaluate_threshold_sweep_for_equal_recall(ep_probs_v[name], thr_grid)
            eq_k["variants"][name] = {
                "at_base_recall": _interp_at_target_recall(curve, base_rec),
                "at_base_recall_minus_002": _interp_at_target_recall(curve, max(0.05, base_rec - 0.02)),
            }
        m1_eq_rec[str(k)] = eq_k

    m1_cmps = {
        "HardNegH_vs_BaseR": {k: compare_methods(m1_methods["M1_Cand_HardNegH"][k], m1_methods["M1_Base_R"][k]) for k in ("1", "3", "5")},
        "RandomBG_vs_BaseR": {k: compare_methods(m1_methods["M1_Ctrl_EqualBudgetRandomBG"][k], m1_methods["M1_Base_R"][k]) for k in ("1", "3", "5")},
        "HardNegH_vs_RandomBG": {k: compare_methods(m1_methods["M1_Cand_HardNegH"][k], m1_methods["M1_Ctrl_EqualBudgetRandomBG"][k]) for k in ("1", "3", "5")},
    }
    m1_gate_ks = [k for k in ("1", "3", "5") if m1_cmps["HardNegH_vs_BaseR"][k]["gates"]["Gate_FP"]]
    m1_eq_ks = [
        k
        for k in ("1", "3", "5")
        if m1_eq_rec[k]["variants"]["M1_Cand_HardNegH"]["at_base_recall"]["neg_site_fp_at_recall"]
        < m1_eq_rec[k]["variants"]["M1_Base_R"]["at_base_recall"]["neg_site_fp_at_recall"] - 0.001
    ]
    m1_summary = {
        "schema": "pv-2m-dinov3sat-m1-hardneg-dev-v1",
        "elapsed_sec": time.perf_counter() - t_m1_0,
        "verdict": {
            "gate_fp_passed_ks": m1_gate_ks,
            "gate_fp_overall_pass": bool(len(m1_gate_ks) >= 2 and ("1" in m1_gate_ks or "3" in m1_gate_ks)),
            "equal_recall_neg_fp_improved_ks": m1_eq_ks,
            "module_1_accepted": bool(len(m1_gate_ks) >= 2 and ("1" in m1_gate_ks or "3" in m1_gate_ks) and len(m1_eq_ks) >= 2),
        },
        "diagnostics_by_k": m1_diag,
        "equal_recall_by_k": m1_eq_rec,
        "comparisons": m1_cmps,
        "methods": m1_methods,
    }
    write_json(output_dir / "m1_hardneg_summary.json", m1_summary)
    print(f"[1. M1 DONE in {m1_summary['elapsed_sec']:.1f}s] Verdict: {m1_summary['verdict']}", flush=True)

    # =========================================================================
    # 2. INDEPENDENT EXPERIMENT: MODULE ② (Stage 0 Coverage Audit + Full-Token Control)
    # =========================================================================
    t_m2_0 = time.perf_counter()
    m2_cfg_path = REPO_ROOT / "configs" / "three_module_exp_m2_cropzoom.json"
    crop_cfg = CropZoomConfig.from_json(m2_cfg_path, mode="full_token_reverify")
    m3_no_rec_cfg = Iter5Config(**{**m3_cfg.__dict__, "name": "M2_NoRecovery", "use_self_rescue": False})
    pipe_nr = ThreeModulePipeline(m3_no_rec_cfg)
    pipe_r = m1_pipes["M1_Base_R"]

    m2_methods: dict[str, dict[str, Any]] = {"M2_NoRecovery": {}, "M2_SelfRescue_R": {}, "M2_Ctrl_FullTokenReverify": {}}
    m2_stage0: dict[str, Any] = {}
    m2_seedless: dict[str, Any] = {}

    for k in sorted(ep_by_k.keys()):
        ep_nr, ep_ft = [], []
        tot_missed = tot_covered = pos_wins = pos_imgs = neg_trig = neg_imgs = neg_wins = ft_wb = ft_m = ft_rec = 0
        sl_r, sl_ft = [], []

        for ep_idx, ep in enumerate(ep_by_k[k]):
            sup_ids, q_ids = ep["support_ids"], ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]
            out_r = r_episode_cache[(k, ep_idx)]
            out_nr = pipe_nr.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            p_nr, s32_nr = out_nr.to_legacy_outputs()
            ep_nr.append(evaluate_episode(q_recs, p_nr, s32_nr))

            fg_chunks, bg_chunks = [], []
            for sid in sup_ids:
                p_i = caches["deep_64"]["id_to_pos"][sid]
                fg_m = caches["deep_64"]["fg_all_masks"][p_i] if k == 1 else caches["deep_64"]["fg_core_masks"][p_i]
                bg_m = caches["deep_64"]["bg_masks"][p_i]
                t_s = tokens_by_key["deep_64"][sid]
                if fg_m.any():
                    fg_chunks.append(t_s[torch.from_numpy(np.flatnonzero(fg_m)).to(t_s.device)])
                if bg_m.any():
                    b_idx = np.flatnonzero(bg_m)
                    if len(b_idx) > 256:
                        b_idx = b_idx[:: max(1, len(b_idx) // 256)]
                    bg_chunks.append(t_s[torch.from_numpy(b_idx).to(t_s.device)])
            sup_fg = torch.cat(fg_chunks, dim=0) if fg_chunks else torch.empty((0, 1), device=run_device)
            sup_bg = torch.cat(bg_chunks, dim=0) if bg_chunks else torch.empty((0, 1), device=run_device)

            p_ft_ep, s32_ft_ep = {}, {}
            calib = out_r.calibration
            for qid in q_ids:
                rec = records_by_id[qid]
                sem_q, rec_q, bnd_q = out_r.semantic_by_id[qid], out_r.recovery_by_id[qid], out_r.boundary_by_id[qid]
                scored_w, wins = _score_windows_full_image_tokens(
                    sem_q.s_sem, sem_q.p_sem, tokens_by_key["deep_64"][qid], sup_fg, sup_bg, calib.bias, calib.temp, crop_cfg=crop_cfg
                )
                aud = audit_offline_candidate_coverage(rec["mask"], bnd_q.m_coarse, wins)
                if rec["positive"]:
                    pos_imgs += 1
                    pos_wins += len(wins)
                    tot_missed += aud["missed_components"]
                    tot_covered += aud["covered_missed_components"]
                else:
                    neg_imgs += 1
                    neg_wins += len(wins)
                    if aud["neg_triggered"]:
                        neg_trig += 1

                s_ft, wb_d = project_crop_scores_max_increment(rec_q.s_res, scored_w, max_writeback_delta=crop_cfg.max_writeback_delta)
                if wb_d["crop_windows_applied"] > 0:
                    ft_wb += 1
                p_res_ft = torch.sigmoid((s_ft - calib.bias) / calib.temp)
                s32_ft = F.avg_pool2d(s_ft.reshape(1, 1, 64, 64), 2, stride=2)[0, 0]
                rec_ft = type(rec_q)(qid=qid, grid_hw=64, s_res=s_ft, p_res=p_res_ft, scores_32=s32_ft, diagnostics=wb_d)
                bnd_ft = pipe_r.boundary.infer_query(rec_ft, aff_weights=aff_by_id[qid])
                p_ft_ep[qid] = bnd_ft.p_coarse
                s32_ft_ep[qid] = s32_ft.detach().cpu().numpy().astype(np.float32)

                if rec["positive"]:
                    mc, rc = _compute_missed_component_recovery(rec["mask"], bnd_q.m_coarse, bnd_ft.m_coarse)
                    ft_m += mc
                    ft_rec += rc
                    if not rec_q.diagnostics.get("self_rescue_triggered", False):
                        gt_b = rec["mask"] > 0
                        sl_r.append(float(np.logical_and(gt_b, bnd_q.m_coarse).sum() / max(1, int(gt_b.sum()))))
                        sl_ft.append(float(np.logical_and(gt_b, bnd_ft.m_coarse).sum() / max(1, int(gt_b.sum()))))
            ep_ft.append(evaluate_episode(q_recs, p_ft_ep, s32_ft_ep))

        m2_methods["M2_NoRecovery"][str(k)] = aggregate_episodes(ep_nr)
        m2_methods["M2_SelfRescue_R"][str(k)] = m1_methods["M1_Base_R"][str(k)]
        m2_methods["M2_Ctrl_FullTokenReverify"][str(k)] = aggregate_episodes(ep_ft)
        cov_r = float(tot_covered / max(1, tot_missed))
        m2_stage0[str(k)] = {
            "missed_gt_components": tot_missed,
            "covered_missed_gt_components": tot_covered,
            "coverage_rate": cov_r,
            "stage0_gate_ge_025_pass": bool(cov_r >= 0.25),
            "mean_wins_pos": float(pos_wins / max(1, pos_imgs)),
            "mean_wins_neg": float(neg_wins / max(1, neg_imgs)),
            "pure_neg_trigger_rate": float(neg_trig / max(1, neg_imgs)),
            "full_token_writeback_rate": float(ft_wb / max(1, pos_imgs + neg_imgs)),
            "full_token_missed_comp_recovery_rate": float(ft_rec / max(1, ft_m)),
        }
        m2_seedless[str(k)] = {
            "n_seedless_pos_queries": len(sl_r),
            "mean_recall_R": float(np.mean(sl_r)) if sl_r else None,
            "mean_recall_FullToken": float(np.mean(sl_ft)) if sl_ft else None,
        }
        print(
            f"[2. M2 K={k}] Stage0 Cov={tot_covered}/{tot_missed} ({cov_r:.1%}) NegTrig={neg_trig}/{neg_imgs} "
            f"| FT Sm<5R={m2_methods['M2_Ctrl_FullTokenReverify'][str(k)]['strata']['small_lt5pct']['recall']:.4f} "
            f"(R={m2_methods['M2_SelfRescue_R'][str(k)]['strata']['small_lt5pct']['recall']:.4f}) "
            f"NegFP={m2_methods['M2_Ctrl_FullTokenReverify'][str(k)]['macro']['neg_fp']:.4f}",
            flush=True,
        )

    m2_cmps = {
        "SelfRescueR_vs_NoRecovery": {k: compare_methods(m2_methods["M2_SelfRescue_R"][k], m2_methods["M2_NoRecovery"][k]) for k in ("1", "3", "5")},
        "FullTokenCtrl_vs_SelfRescueR": {k: compare_methods(m2_methods["M2_Ctrl_FullTokenReverify"][k], m2_methods["M2_SelfRescue_R"][k]) for k in ("1", "3", "5")},
    }
    m2_summary = {
        "schema": "pv-2m-dinov3sat-m2-cropzoom-stage0-dev-v1",
        "elapsed_sec": time.perf_counter() - t_m2_0,
        "stage0_coverage_by_k": m2_stage0,
        "seedless_subset_by_k": m2_seedless,
        "verdict": {
            "stage0_coverage_pass": bool(any(v["stage0_gate_ge_025_pass"] for v in m2_stage0.values())),
            "full_token_gate_fn_ks": [k for k in ("1", "3", "5") if m2_cmps["FullTokenCtrl_vs_SelfRescueR"][k]["gates"]["Gate_FN"]],
        },
        "comparisons": m2_cmps,
        "methods": m2_methods,
    }
    write_json(output_dir / "m2_cropzoom_stage0_summary.json", m2_summary)
    print(f"[2. M2 DONE in {m2_summary['elapsed_sec']:.1f}s] Verdict: {m2_summary['verdict']}", flush=True)

    # =========================================================================
    # 3. INDEPENDENT EXPERIMENT: MODULE ③ (Stage 0 Offline GT Ref + Rigid vs Deformable Grad)
    # =========================================================================
    t_m3_0 = time.perf_counter()
    m3_cfg_path = REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json"
    pb_cfg = PhaseBoundaryConfig()
    def_cfg = DeformableBoundaryConfig.from_json(m3_cfg_path, mode="deformable_grad")
    signals_by_id = {rid: precompute_image_signals(rgb_by_id[rid], pb_cfg) for rid in all_ids}

    m3_vars = ("M3_Off_R", "M3_Rigid_Grad", "M3_Deformable_Grad", "Offline_Rigid_GT_Ref", "Offline_Deform_GT_Ref", "Offline_Loose_PM8px")
    m3_methods: dict[str, dict[str, Any]] = {v: {} for v in m3_vars}
    m3_diag: dict[str, Any] = {}

    for k in sorted(ep_by_k.keys()):
        ep_evals: dict[str, list[dict]] = {v: [] for v in m3_vars}
        move_acc: dict[str, dict[str, float]] = {"M3_Rigid_Grad": defaultdict(float), "M3_Deformable_Grad": defaultdict(float)}

        for ep_idx, ep in enumerate(ep_by_k[k]):
            sup_ids, q_ids = ep["support_ids"], ep["query_ids"]
            q_recs = [records_by_id[qid] for qid in q_ids]
            out_r = r_episode_cache[(k, ep_idx)]
            p_r, s32_r = out_r.to_legacy_outputs()

            loo_probs = None
            if k >= 2:
                loo_probs = {}
                for s_i in sup_ids:
                    rem = [s for s in sup_ids if s != s_i]
                    o_l = pipe_r.run_episode(caches, tokens_by_key, mask_by_id, aff_by_id, rem, [s_i])
                    loo_probs[s_i] = o_l.boundary_by_id[s_i].p_coarse

            allow_out, p_floor = _calibrate_p_floor_loo(sup_ids, mask_by_id, loo_probs)
            rigid_calib = SupportCalibrationBundle(
                k_shots=k,
                allow_outward=allow_out,
                p_floor=p_floor,
                p_floor_fallback=(p_floor is None),
                period_range_px=(2.8, 12.0),
                holdout_r2_floor=0.05,
                tau_by_variant={vn: (0.63 if vn == "B_GradAlign" else 1e9) for vn in pb_cfg.fallback_gain_per_px},
                null_block_counts={vn: 10 for vn in pb_cfg.fallback_gain_per_px},
                tau_fallback_by_variant={vn: False for vn in pb_cfg.fallback_gain_per_px},
            )

            masks_v: dict[str, dict[str, np.ndarray]] = {v: {} for v in m3_vars}
            for qid in q_ids:
                m0 = out_r.boundary_by_id[qid].m_coarse
                p0 = p_r[qid]
                gt = records_by_id[qid]["mask"]
                masks_v["M3_Off_R"][qid] = m0

                r_out = refine_query_image_all_variants(
                    rgb_by_id[qid], p0, rigid_calib, variants=("B_GradAlign",), cfg=pb_cfg, precomputed_signals=signals_by_id[qid]
                )
                bi = r_out["variants"]["B_GradAlign"]
                masks_v["M3_Rigid_Grad"][qid] = bi["mask"]
                move_acc["M3_Rigid_Grad"]["total_arc"] += r_out["total_arc_len_px"]
                move_acc["M3_Rigid_Grad"]["moved_arc"] += bi["moved_arc_len_px"]
                move_acc["M3_Rigid_Grad"]["net_area"] += bi["net_area_delta_px"]
                move_acc["M3_Rigid_Grad"]["abs_d_wsum"] += bi["mean_abs_d_moved"] * bi["moved_arc_len_px"]

                d_out = refine_mask_deformable_grad(
                    m0, rgb_by_id[qid], p0, allow_outward=allow_out, p_floor=p_floor, tau_grad_per_px=0.20, cfg=def_cfg
                )
                masks_v["M3_Deformable_Grad"][qid] = d_out["mask"]
                move_acc["M3_Deformable_Grad"]["total_arc"] += d_out["total_arc_len_px"]
                move_acc["M3_Deformable_Grad"]["moved_arc"] += d_out["moved_arc_len_px"]
                move_acc["M3_Deformable_Grad"]["net_area"] += d_out["net_area_delta_px"]
                move_acc["M3_Deformable_Grad"]["abs_d_wsum"] += d_out["mean_abs_d_moved"] * d_out["moved_arc_len_px"]
                move_acc["M3_Deformable_Grad"]["topo_reverts"] += d_out["topology_reverts"]

                if k == 5:
                    masks_v["Offline_Rigid_GT_Ref"][qid] = _rigid_gt_reachable_ref(m0, gt, pb_cfg)
                    masks_v["Offline_Deform_GT_Ref"][qid] = refine_mask_deformable_grad(
                        m0, None, p0, allow_outward=True, p_floor=None, gt_mask_offline=(gt > 0), cfg=def_cfg
                    )["mask"]
                    masks_v["Offline_Loose_PM8px"][qid] = compute_oracle_pm8px_mask(gt, p0)["oracle_mask"]
                else:
                    masks_v["Offline_Rigid_GT_Ref"][qid] = m0
                    masks_v["Offline_Deform_GT_Ref"][qid] = m0
                    masks_v["Offline_Loose_PM8px"][qid] = compute_oracle_pm8px_mask(gt, p0)["oracle_mask"]

            for v in m3_vars:
                ep_evals[v].append(evaluate_binary_episode(q_recs, masks_v[v], p_r, s32_r))

        for v in m3_vars:
            m3_methods[v][str(k)] = aggregate_binary_episodes(ep_evals[v])
        m3_diag[str(k)] = {
            vn: {
                "moved_arc_fraction": float(st["moved_arc"] / max(1.0, st["total_arc"])),
                "mean_abs_d_px": float(st["abs_d_wsum"] / max(1.0, st["moved_arc"])),
                "mean_net_area_delta_px": float(st["net_area"] / 760.0),
                "topology_reverts": int(st.get("topo_reverts", 0)),
            }
            for vn, st in move_acc.items()
        }
        for v in ("M3_Off_R", "M3_Rigid_Grad", "M3_Deformable_Grad", "Offline_Rigid_GT_Ref", "Offline_Deform_GT_Ref"):
            if k != 5 and v.startswith("Offline_"):
                continue
            m = m3_methods[v][str(k)]["macro"]
            sm = m3_methods[v][str(k)]["strata"]["small_lt5pct"]["recall"]
            print(
                f"[3. M3 K={k}] {v:22s}: BF1@1={m['bf1_1px']:.4f} BF1@2={m['boundary_f1']:.4f} "
                f"Dice={m['dice']:.4f} Sm<5R={sm:.4f} NegFP={m['neg_fp']:.4f}",
                flush=True,
            )

    m3_cmps = {
        "DeformGrad_vs_OffR": {k: compare_boundary_methods(m3_methods["M3_Deformable_Grad"][k], m3_methods["M3_Off_R"][k]) for k in ("1", "3", "5")},
        "RigidGrad_vs_OffR": {k: compare_boundary_methods(m3_methods["M3_Rigid_Grad"][k], m3_methods["M3_Off_R"][k]) for k in ("1", "3", "5")},
        "DeformGrad_vs_RigidGrad": {k: compare_boundary_methods(m3_methods["M3_Deformable_Grad"][k], m3_methods["M3_Rigid_Grad"][k]) for k in ("1", "3", "5")},
        "Offline_DeformGTRef_vs_RigidGTRef_5shot": compare_boundary_methods(m3_methods["Offline_Deform_GT_Ref"]["5"], m3_methods["Offline_Rigid_GT_Ref"]["5"]),
    }
    c5 = m3_cmps["DeformGrad_vs_OffR"]["5"]
    guard_ok = all(
        m3_cmps["DeformGrad_vs_OffR"][k]["dice"]["mean_delta"] >= -0.005
        and m3_cmps["DeformGrad_vs_OffR"][k]["small_lt5pct_recall_delta"] >= -0.010
        and m3_cmps["DeformGrad_vs_OffR"][k]["neg_fp"]["mean_delta"] <= 0.002
        for k in ("1", "3", "5")
    )
    m3_pass = bool(
        c5["boundary_f1"]["mean_delta"] >= 0.015
        and c5["boundary_f1"]["ci95"][0] > 0.0
        and c5["boundary_f1"]["positive_folds"] >= 3
        and c5["bf1_1px"]["mean_delta"] > 0.0
        and guard_ok
    )
    m3_summary = {
        "schema": "pv-2m-dinov3sat-m3-deformgrad-dev-v1",
        "elapsed_sec": time.perf_counter() - t_m3_0,
        "verdict": {
            "m3_deformable_grad_pass": m3_pass,
            "bf1_2px_delta_5shot": c5["boundary_f1"]["mean_delta"],
            "bf1_2px_ci95_5shot": c5["boundary_f1"]["ci95"],
            "bf1_2px_positive_folds_5shot": c5["boundary_f1"]["positive_folds"],
            "bf1_1px_delta_5shot": c5["bf1_1px"]["mean_delta"],
            "all_k_guardrails_pass": guard_ok,
        },
        "diagnostics_by_k": m3_diag,
        "comparisons": m3_cmps,
        "methods": m3_methods,
    }
    write_json(output_dir / "m3_deformgrad_summary.json", m3_summary)
    print(f"[3. M3 DONE in {m3_summary['elapsed_sec']:.1f}s] Verdict: {m3_summary['verdict']}", flush=True)

    master = {
        "schema": "pv-2m-dinov3sat-stage2-independent-suite-v1",
        "device": run_device,
        "cache_build_sec": cache_sec,
        "total_elapsed_sec": time.perf_counter() - t_all,
        "verdicts": {
            "0_equiv_120ep": bool(all(v["all_passed"] for v in equiv_audit.values())),
            "1_m1_hardneg": m1_summary["verdict"],
            "2_m2_cropzoom_stage0": m2_summary["verdict"],
            "3_m3_deformgrad": m3_summary["verdict"],
        },
    }
    write_json(output_dir / "stage2_master_summary.json", master)
    print("STAGE 2 INDEPENDENT SUITE COMPLETE:", master["verdicts"], flush=True)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared-cache independent Stage 1 Equiv + Stage 2 M1/M2/M3 Dev suite.")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "configs" / "locked_split")
    parser.add_argument("--cache-pt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "stage2_independent")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_stage2_independent_suite(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
