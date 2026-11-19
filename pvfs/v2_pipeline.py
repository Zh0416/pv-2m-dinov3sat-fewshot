from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import area_target_32, load_dev_records, read_verified_mask, read_verified_rgb, resolve_data_dirs
from .episodes import load_and_verify_episodes
from .fast_bank import build_pairwise_topk_cache, score_from_pairwise, select_fg_bg_indices
from .freq_features import apply_ofg_neg_gate_and_rescue, compute_support_freq_alignment, extract_freq_pyramid
from .io import write_json
from .iter3_methods import _ex_neigh_bg_contrast, _to_tokens_norm, compute_dino_affinity_weights, propagate_affinity
from .iter5_methods import ITER5_VARIANTS, Iter5Config, _apply_mid_verified_rescue, run_iter5_episode
from .metrics import aggregate_episodes, compare_methods, evaluate_episode
from .v2_decoder import SCFBDecoder, build_decoder_input_512, compute_support_rgb_stats, scfb_episodic_loss

# Locked Module-1 semantic backbone config shared cleanly across V2 ablations
V2_SEMANTIC_CFG = Iter5Config(
    name="V2_MSCP_Core",
    shot_mode="balanced_70_30",
    w_fg_calib=0.42,
    use_self_rescue=True,
    seed_prob_thr=0.70,
    min_seeds=5,
    seed_mid_margin=0.01,
    qsc_weight=0.45,
    use_affinity=True,
)


@dataclass(frozen=True)
class V2Variant:
    name: str
    kind: str  # "iter5_ref" | "freq_only" | "decoder"
    disable_freq: bool = False
    ref_cfg: Iter5Config | None = None


V2_VARIANTS: tuple[V2Variant, ...] = (
    V2Variant("Ref_M5_BalCal", kind="iter5_ref", ref_cfg=next(c for c in ITER5_VARIANTS if c.name == "M5_BalCal")),
    V2Variant("Ref_M3_Pruned", kind="iter5_ref", ref_cfg=next(c for c in ITER5_VARIANTS if c.name == "M3_Pruned_Ref")),
    V2Variant("V2_FreqOnly_NoDecoder", kind="freq_only"),
    V2Variant("V2_Decoder_NoFreq_LOO", kind="decoder", disable_freq=True),
    V2Variant("V2_Full_WithDecoder", kind="decoder", disable_freq=False),
)


@torch.inference_mode()
def compute_mscp_episode_maps(
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor] | None,
    support_ids: list[str],
    query_ids: list[str],
    cfg: Iter5Config = V2_SEMANTIC_CFG,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run Module 1 (MSCP) to produce z_sem ([64,64]) and 6-channel semantic map ([6,64,64])."""
    k = len(support_ids)
    use_core = cfg.k1_use_core if k == 1 else cfg.kge2_use_core
    core_purity = 0.85 if use_core else 0.50

    mid_veto_ref = 0.20
    if k >= 2:
        mid_vals = []
        for idx, sid in enumerate(support_ids):
            sm, _, _ = score_from_pairwise(caches["mid_64"], sid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=idx)
            bm = caches["mid_64"]["bg_masks"][caches["mid_64"]["id_to_pos"][sid]]
            if bm.any():
                mid_vals.append(sm[torch.from_numpy(np.flatnonzero(bm)).to(sm.device)])
        if mid_vals:
            mid_veto_ref = float(torch.quantile(torch.cat(mid_vals), 0.85).item())
    else:
        c_bg_m = _ex_neigh_bg_contrast(tokens_by_key["mid_64"][support_ids[0]], mask_by_id[support_ids[0]], 64, core_purity=core_purity)
        if c_bg_m is not None:
            mid_veto_ref = float(torch.quantile(c_bg_m, 0.85).item())

    def _raw_all(qid: str, ex_idx: int | None):
        s32, _, _ = score_from_pairwise(caches["deep_32"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx)
        s64, sp64, sn64 = score_from_pairwise(caches["deep_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx)
        sm, _, _ = score_from_pairwise(caches["mid_64"], qid, support_ids, use_core=use_core, shot_mode=cfg.shot_mode, exclude_pos_idx=ex_idx)
        s32_up = F.interpolate(s32.reshape(1, 1, 32, 32), size=(64, 64), mode="bilinear", align_corners=False).reshape(-1)
        comb = cfg.w_s64 * s64 + (1.0 - cfg.w_s64) * s32_up + cfg.lambda_mid * sm - cfg.beta_veto * F.softplus((mid_veto_ref - sm) * 10.0) / 10.0
        return comb, s64, s32_up, sm, sp64, sn64

    if k >= 2:
        fg_items, bg_list = [], []
        for idx, sid in enumerate(support_ids):
            comb_s, *_ = _raw_all(sid, idx)
            p_i = caches["deep_64"]["id_to_pos"][sid]
            fg_m = caches["deep_64"]["fg_all_masks"][p_i]
            bg_m = caches["deep_64"]["bg_masks"][p_i]
            fg_items.append(float(torch.median(comb_s[torch.from_numpy(np.flatnonzero(fg_m)).to(comb_s.device)]).item()))
            if bg_m.any():
                bg_list.append(comb_s[torch.from_numpy(np.flatnonzero(bg_m)).to(comb_s.device)])
        if bg_list:
            mu_fg = float(np.mean(fg_items))
            mu_bg = float(torch.quantile(torch.cat(bg_list), cfg.bg_q_calib).item())
            if mu_fg > mu_bg + 0.02:
                bias = cfg.w_fg_calib * mu_fg + (1.0 - cfg.w_fg_calib) * mu_bg
                temp = max((mu_fg - mu_bg) / (2.0 * float(np.log(4.0))), 0.04)
            else:
                bias, temp = mu_bg + 0.05, 0.06
        else:
            bias, temp = 0.30, 0.06
    else:
        sid0 = support_ids[0]
        c32_b = _ex_neigh_bg_contrast(tokens_by_key["deep_32"][sid0], mask_by_id[sid0], 32, core_purity=core_purity)
        c64_b = _ex_neigh_bg_contrast(tokens_by_key["deep_64"][sid0], mask_by_id[sid0], 64, core_purity=core_purity)
        cm_b = _ex_neigh_bg_contrast(tokens_by_key["mid_64"][sid0], mask_by_id[sid0], 64, core_purity=core_purity)
        if c64_b is not None and c32_b is not None and cm_b is not None:
            _, bg64_m, _ = select_fg_bg_indices(mask_by_id[sid0], 64, core_purity)
            _, bg32_m, _ = select_fg_bg_indices(mask_by_id[sid0], 32, core_purity)
            bg32_flat = np.flatnonzero(bg32_m)
            c32_full = torch.zeros((32, 32), device=c64_b.device)
            c32_full[torch.from_numpy(bg32_flat // 32).to(c64_b.device), torch.from_numpy(bg32_flat % 32).to(c64_b.device)] = c32_b
            bg64_t = torch.from_numpy(np.flatnonzero(bg64_m)).to(c64_b.device)
            c32_at_64 = c32_full[(bg64_t // 64) // 2, (bg64_t % 64) // 2]
            bg_s = cfg.w_s64 * c64_b + (1.0 - cfg.w_s64) * c32_at_64 + cfg.lambda_mid * cm_b - cfg.beta_veto * F.softplus((mid_veto_ref - cm_b) * 10.0) / 10.0
            bias = float(torch.quantile(bg_s, cfg.bg_q_calib).item()) + cfg.k1_margin
        else:
            bias = 0.30
        temp = 0.06

    out: dict[str, dict[str, torch.Tensor]] = {}
    for qid in query_ids:
        comb, s64, s32_up, sm, sp64, sn64 = _raw_all(qid, None)
        p_base = torch.sigmoid((comb - bias) / temp)
        comb_fin = _apply_mid_verified_rescue(comb, p_base, sm, mid_veto_ref, tokens_by_key["deep_64"][qid], cfg) if cfg.use_self_rescue else comb
        p_grid = torch.sigmoid((comb_fin - bias) / temp)
        if cfg.use_affinity:
            w_aff = aff_by_id[qid] if aff_by_id is not None else compute_dino_affinity_weights(tokens_by_key["deep_64"][qid], tau=cfg.aff_tau)
            p_grid = propagate_affinity(p_grid.reshape(64, 64), w_aff, steps=cfg.aff_steps, eta=cfg.aff_eta)
        z_sem = torch.logit(p_grid.clamp(1e-4, 1.0 - 1e-4)).reshape(64, 64)
        sem_6 = torch.stack(
            (
                z_sem,
                s64.reshape(64, 64),
                s32_up.reshape(64, 64),
                (sm - mid_veto_ref).reshape(64, 64),
                sp64.reshape(64, 64),
                sn64.reshape(64, 64),
            ),
            dim=0,
        )
        out[qid] = {"z_sem": z_sem, "sem_6": sem_6}
    return out


def train_fold_scfb_decoder(
    *,
    train_records: list[dict],
    caches: dict[str, dict],
    tokens_by_key: dict[str, dict[str, torch.Tensor]],
    mask_by_id: dict[str, np.ndarray],
    aff_by_id: dict[str, torch.Tensor],
    rgb_by_id: dict[str, torch.Tensor],
    freq512_by_id: dict[str, torch.Tensor],
    freq64_by_id: dict[str, torch.Tensor],
    device: torch.device,
    disable_freq: bool = False,
    steps: int = 240,
    seed: int = 20261125,
) -> SCFBDecoder:
    """Train SCFB-Decoder strictly on training fold records (site-disjoint from val/test)."""
    pos_by_site: dict[str, list[str]] = defaultdict(list)
    neg_ids: list[str] = []
    for r in train_records:
        if r["positive"]:
            pos_by_site[r["site"]].append(r["id"])
        else:
            neg_ids.append(r["id"])
    pos_sites = sorted(pos_by_site)
    pos_all = [r["id"] for r in train_records if r["positive"]]
    site_of = {r["id"]: r["site"] for r in train_records}

    model = SCFBDecoder(in_ch=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=8e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(seed)

    model.train()
    for step in range(steps):
        k_tr = (1, 3, 5)[step % 3]
        chosen_sites = rng.sample(pos_sites, min(k_tr, len(pos_sites)))
        sup_ids = [rng.choice(pos_by_site[s]) for s in chosen_sites]
        sup_site_set = set(chosen_sites)

        cand_pos = [qid for qid in pos_all if site_of[qid] not in sup_site_set]
        if not cand_pos:
            cand_pos = [qid for qid in pos_all if qid not in set(sup_ids)]
        q_pos = rng.sample(cand_pos, min(3, len(cand_pos)))
        q_neg = rng.sample(neg_ids, min(1, len(neg_ids))) if neg_ids else []
        q_ids = q_pos + q_neg

        with torch.inference_mode():
            mscp = compute_mscp_episode_maps(caches, tokens_by_key, mask_by_id, aff_by_id, sup_ids, q_ids)
            zf_map = compute_support_freq_alignment(freq64_by_id, mask_by_id, sup_ids, q_ids)
            fg_rgb, bg_rgb = compute_support_rgb_stats(rgb_by_id, mask_by_id, sup_ids, device)
            xs, ys = [], []
            for qid in q_ids:
                xs.append(
                    build_decoder_input_512(
                        mscp[qid]["sem_6"],
                        zf_map[qid],
                        rgb_by_id[qid].to(device),
                        freq512_by_id[qid].to(device),
                        fg_rgb,
                        bg_rgb,
                        disable_freq=disable_freq,
                    )
                )
                ys.append(torch.from_numpy((mask_by_id[qid] > 0).astype(np.float32)).to(device)[None])
            xb = torch.stack(xs, dim=0)
            yb = torch.stack(ys, dim=0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = scfb_episodic_loss(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    return model


@torch.inference_mode()
def run_v2_episode(
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
    decoder: SCFBDecoder | None = None,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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

    assert decoder is not None
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
            logits = decoder(xb).float().squeeze(1)  # [B, 512, 512]
        probs = torch.sigmoid(logits)
        s32_b = F.avg_pool2d(logits[:, None], kernel_size=16, stride=16).squeeze(1)
        for bi, qid in enumerate(chunk_ids):
            probs_512[qid] = probs[bi].cpu().numpy().astype(np.float32)
            scores_32[qid] = s32_b[bi].cpu().numpy().astype(np.float32)
    return probs_512, scores_32


def _save_v2_panels(
    out_dir: Path,
    rgb_by_id: dict[str, torch.Tensor],
    mask_by_id: dict[str, np.ndarray],
    panel_probs: dict[str, dict[str, np.ndarray]],
) -> None:
    from PIL import Image, ImageDraw

    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    fn_targets = ("正样本/PV-55.tif", "正样本/PV-49.tif", "正样本/PV-73.tif", "正样本/PV-10.tif")
    fp_targets = ("正样本/PV-80.tif", "负样本/PV-93.tif", "负样本/PV-100.tif", "负样本/PV-96.tif")
    cols = ("GT", "Ref_M5_BalCal", "V2_FreqOnly_NoDecoder", "V2_Full_WithDecoder")
    short_cols = ("GT", "M5_BalCal", "V2_FreqOnly", "V2_FullDecoder")

    for fname, qids in (("k5_v2_fn_compare.png", fn_targets), ("k5_v2_fp_compare.png", fp_targets)):
        q_valid = [q for q in qids if q in rgb_by_id]
        if not q_valid:
            continue
        tile = 320
        canvas = Image.new("RGB", (tile * len(cols), tile * len(q_valid)), (18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        for ri, qid in enumerate(q_valid):
            rgb = (rgb_by_id[qid].float().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.float32)
            gt = mask_by_id[qid] > 0
            for ci, (cname, lbl) in enumerate(zip(cols, short_cols)):
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


def run_v2_dev_ablations(
    *,
    split_dir: Path,
    cache_pt: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    device: str = "cuda",
    train_steps: int = 240,
) -> dict:
    """Run strict 4-fold site-disjoint episodic training & ablations for FreqBoundPV-FS V2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, dev_records = load_dev_records(split_dir)
    img_dir, lbl_dir = resolve_data_dirs(image_dir, label_dir)
    episodes_lock = load_and_verify_episodes(split_dir, split_dir / "episodes_lock.json", dev_records)

    run_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    payload = torch.load(cache_pt, map_location="cpu", weights_only=False)
    feats = payload["features"]

    dev_by_id = {r["id"]: r for r in dev_records}
    tokens_by_key: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "mid_64": {}, "deep_64": {}}
    mask_by_id: dict[str, np.ndarray] = {}
    rgb_by_id: dict[str, torch.Tensor] = {}
    freq512_by_id: dict[str, torch.Tensor] = {}
    freq64_by_id: dict[str, torch.Tensor] = {}
    aff_by_id: dict[str, torch.Tensor] = {}

    for r in dev_records:
        qid = r["id"]
        rgb_np = read_verified_rgb(img_dir, r)
        rgb_t = torch.from_numpy(rgb_np.copy()).float() / 255.0
        rgb_by_id[qid] = rgb_t.half()  # compact fp16 [3, 512, 512]
        f512, f64 = extract_freq_pyramid(rgb_np, device=run_device)
        freq512_by_id[qid] = f512
        freq64_by_id[qid] = f64.to(run_device)
        mask_by_id[qid] = read_verified_mask(lbl_dir, r)
        for key in ("deep_32", "mid_64", "deep_64"):
            tokens_by_key[key][qid] = _to_tokens_norm(feats[qid][key], run_device)
        aff_by_id[qid] = compute_dino_affinity_weights(tokens_by_key["deep_64"][qid], tau=0.12)

    all_ids = [r["id"] for r in dev_records]
    pos_ids = [r["id"] for r in dev_records if r["positive"]]
    caches = {
        key: build_pairwise_topk_cache(tokens_by_key[key], mask_by_id, all_ids, pos_ids, grid_hw=(64 if "64" in key else 32))
        for key in ("deep_32", "mid_64", "deep_64")
    }

    # Train fold-isolated SCFB decoders (Full and NoFreq_LOO) strictly on fold != f
    fold_decoders: dict[str, dict[int, SCFBDecoder]] = {"V2_Decoder_NoFreq_LOO": {}, "V2_Full_WithDecoder": {}}
    for f in range(4):
        tr_recs = [r for r in dev_records if r["fold"] != f]
        print(f"[V2] Training fold {f} SCFB decoders on {len(tr_recs)} training-fold images...", flush=True)
        fold_decoders["V2_Full_WithDecoder"][f] = train_fold_scfb_decoder(
            train_records=tr_recs,
            caches=caches,
            tokens_by_key=tokens_by_key,
            mask_by_id=mask_by_id,
            aff_by_id=aff_by_id,
            rgb_by_id=rgb_by_id,
            freq512_by_id=freq512_by_id,
            freq64_by_id=freq64_by_id,
            device=run_device,
            disable_freq=False,
            steps=train_steps,
            seed=20261125 + 100 * f,
        )
        fold_decoders["V2_Decoder_NoFreq_LOO"][f] = train_fold_scfb_decoder(
            train_records=tr_recs,
            caches=caches,
            tokens_by_key=tokens_by_key,
            mask_by_id=mask_by_id,
            aff_by_id=aff_by_id,
            rgb_by_id=rgb_by_id,
            freq512_by_id=freq512_by_id,
            freq64_by_id=freq64_by_id,
            device=run_device,
            disable_freq=True,
            steps=train_steps,
            seed=20261125 + 100 * f,
        )

    # Save 4-fold trained state_dicts for zero-shot Sealed + External evaluation without retraining drift
    torch.save(
        {
            "V2_Full_WithDecoder": {f: m.state_dict() for f, m in fold_decoders["V2_Full_WithDecoder"].items()},
            "V2_Decoder_NoFreq_LOO": {f: m.state_dict() for f, m in fold_decoders["V2_Decoder_NoFreq_LOO"].items()},
        },
        output_dir / "v2_fold_decoders.pt",
    )

    methods_summary: dict[str, dict] = {v.name: {} for v in V2_VARIANTS}
    panel_probs: dict[str, dict[str, np.ndarray]] = {}
    panel_names = ("Ref_M5_BalCal", "V2_FreqOnly_NoDecoder", "V2_Full_WithDecoder")

    for var in V2_VARIANTS:
        print(f"[V2] Evaluating {var.name} across K={{1,3,5}}...", flush=True)
        for k_str in ("1", "3", "5"):
            ep_results = []
            prob_accum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((512, 512), dtype=np.float32))
            for ep in episodes_lock["episodes_by_k"][k_str]:
                f = int(ep["fold"])
                dec = fold_decoders[var.name][f] if var.kind == "decoder" else None
                p512_map, s32_map = run_v2_episode(
                    var,
                    caches,
                    tokens_by_key,
                    mask_by_id,
                    aff_by_id,
                    rgb_by_id,
                    freq512_by_id,
                    freq64_by_id,
                    ep["support_ids"],
                    ep["query_ids"],
                    decoder=dec,
                    device=run_device,
                )
                q_items = [{**dev_by_id[q], "mask": mask_by_id[q], "y32": area_target_32(mask_by_id[q])} for q in ep["query_ids"]]
                ep_results.append(evaluate_episode(q_items, p512_map, s32_map))
                if k_str == "5" and var.name in panel_names:
                    for qid, p in p512_map.items():
                        prob_accum[qid] += p / 10.0
            methods_summary[var.name][k_str] = aggregate_episodes(ep_results, episodes_per_fold=10)
            if k_str == "5" and var.name in panel_names:
                panel_probs[var.name] = dict(prob_accum)

    _save_v2_panels(output_dir, rgb_by_id, mask_by_id, panel_probs)

    comparisons = {}
    for cname, rname in (
        ("V2_FreqOnly_vs_M5_BalCal", "Ref_M5_BalCal"),
        ("V2_Full_vs_M5_BalCal", "Ref_M5_BalCal"),
        ("V2_Full_vs_M3_Pruned", "Ref_M3_Pruned"),
        ("V2_Full_vs_FreqOnly", "V2_FreqOnly_NoDecoder"),
        ("V2_Full_vs_NoFreq_LOO", "V2_Decoder_NoFreq_LOO"),
    ):
        cand_key = {
            "V2_FreqOnly_vs_M5_BalCal": "V2_FreqOnly_NoDecoder",
            "V2_Full_vs_M5_BalCal": "V2_Full_WithDecoder",
            "V2_Full_vs_M3_Pruned": "V2_Full_WithDecoder",
            "V2_Full_vs_FreqOnly": "V2_Full_WithDecoder",
            "V2_Full_vs_NoFreq_LOO": "V2_Full_WithDecoder",
        }[cname]
        comparisons[cname] = {k_str: compare_methods(methods_summary[cand_key][k_str], methods_summary[rname][k_str]) for k_str in ("1", "3", "5")}

    summary = {"schema": "pv-2m-dinov3sat-v2-dev-v1", "methods": methods_summary, "comparisons": comparisons}
    write_json(output_dir / "v2_dev_summary.json", summary)
    return summary
