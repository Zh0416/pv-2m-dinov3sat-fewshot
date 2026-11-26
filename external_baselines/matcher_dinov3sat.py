from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .sam_utils import decode_sam_masks


def bidirectional_matching_dinov3(
    q_tok_64: torch.Tensor,
    sup_fg_toks: torch.Tensor,
    sup_all_toks: torch.Tensor,
    sup_is_fg: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Matcher (ICLR 2024) Bidirectional Matching (BM) on frozen DINOv3-SAT 64x64 tokens.

    1. Forward match: each support foreground token matches its most similar query token;
    2. Reverse match: each matched query token matches back against ALL support tokens (fg + bg);
       keep only query tokens whose reverse match lands inside the support foreground!
    Returns (matched_q_indices [M], matched_scores [M], fg_sim_map_64 [64, 64]).
    """
    # q_tok_64: [4096, C], sup_fg_toks: [N_fg, C], sup_all_toks: [N_all, C]
    sim_fg_q = (sup_fg_toks @ q_tok_64.T).clamp(-1.0, 1.0)  # [N_fg, 4096]
    fg_sim_per_q = sim_fg_q.max(dim=0).values  # [4096]

    # Forward matches from support FG -> query
    fwd_q_idx = torch.unique(sim_fg_q.argmax(dim=1))  # [M_fwd]
    if fwd_q_idx.numel() == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), fg_sim_per_q.reshape(64, 64)

    # Reverse matches from candidate query tokens -> all support tokens (fg + bg)
    q_cands = q_tok_64[fwd_q_idx]  # [M_fwd, C]
    rev_sim = (q_cands @ sup_all_toks.T).clamp(-1.0, 1.0)  # [M_fwd, N_all]
    rev_best = rev_sim.argmax(dim=1)
    keep = sup_is_fg[rev_best]

    kept_idx = fwd_q_idx[keep]
    if kept_idx.numel() == 0:
        # Graceful fallback to top-4 highest foreground-similarity tokens if strict BM is empty
        kept_idx = torch.topk(fg_sim_per_q, k=4).indices

    kept_scores = fg_sim_per_q[kept_idx]
    return (
        kept_idx.detach().cpu().numpy().astype(np.int64),
        kept_scores.detach().cpu().numpy().astype(np.float32),
        fg_sim_per_q.reshape(64, 64),
    )


def _kmeans_2d(pts_xy: np.ndarray, k: int, n_iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic lightweight 2D K-Means for Matcher's Robust Prompt Sampler (RPS)."""
    n = len(pts_xy)
    k_eff = min(k, n)
    if k_eff <= 1:
        return np.zeros(n, dtype=np.int32), pts_xy.mean(axis=0, keepdims=True)
    # Deterministic evenly spaced initialization along sorted coordinates
    order = np.lexsort((pts_xy[:, 1], pts_xy[:, 0]))
    init_idx = order[np.linspace(0, n - 1, k_eff).round().astype(int)]
    centers = pts_xy[init_idx].astype(np.float32).copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iters):
        d2 = ((pts_xy[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k_eff):
            m = labels == c
            if m.any():
                centers[c] = pts_xy[m].mean(axis=0)
    return labels, centers


class MatcherDINOv3Runner:
    """Faithful Matcher (ICLR 2024) with frozen DINOv3-SAT ViT-L/16 replacing DINOv2 + SAM ViT-L decoder.

    Implements the 3 core stages of Liu et al. (ICLR 2024):
      1. Bidirectional Matching (BM) on frozen DINOv3-SAT 64x64 features;
      2. Robust Prompt Sampler (RPS): K-Means clustering on matched query centers to generate
         diverse positive/negative point prompts covering multiple PV instances;
      3. Instance-Level Mask Merging (IMM): decodes SAM ViT-L candidates per cluster and merges
         masks that pass DINOv3-SAT prototype-consistency scoring.
    """

    def __init__(
        self,
        sam: Any,
        *,
        num_clusters: int = 8,
        max_fg_bank: int = 512,
        max_bg_bank: int = 1024,
        merge_margin: float = 0.05,
    ) -> None:
        self.sam = sam
        self.num_clusters = int(num_clusters)
        self.max_fg_bank = int(max_fg_bank)
        self.max_bg_bank = int(max_bg_bank)
        self.merge_margin = float(merge_margin)

    @torch.inference_mode()
    def fit_episode(
        self,
        tokens_deep64: Any,
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
    ) -> dict[str, Any]:
        fg_list, bg_list = [], []
        for sid in support_ids:
            tok = tokens_deep64[sid]  # [4096, C]
            m512 = mask_by_id[sid].astype(np.float32)
            y64 = m512.reshape(64, 8, 64, 8).mean(axis=(1, 3)).ravel()
            fg_idx = np.flatnonzero(y64 >= 0.50)
            if len(fg_idx) == 0:
                fg_idx = np.flatnonzero(y64 > 0.0)
            bg_idx = np.flatnonzero(y64 == 0.0)
            if len(fg_idx) > 0:
                fg_list.append(tok[torch.from_numpy(fg_idx).to(tok.device)])
            if len(bg_idx) > 0:
                bg_list.append(tok[torch.from_numpy(bg_idx).to(tok.device)])

        sup_fg = torch.cat(fg_list, dim=0)
        if sup_fg.shape[0] > self.max_fg_bank:
            step = max(1, sup_fg.shape[0] // self.max_fg_bank)
            sup_fg = sup_fg[::step][: self.max_fg_bank]

        if bg_list:
            sup_bg = torch.cat(bg_list, dim=0)
            if sup_bg.shape[0] > self.max_bg_bank:
                step = max(1, sup_bg.shape[0] // self.max_bg_bank)
                sup_bg = sup_bg[::step][: self.max_bg_bank]
            sup_all = torch.cat((sup_fg, sup_bg), dim=0)
            sup_is_fg = torch.cat(
                (
                    torch.ones(sup_fg.shape[0], dtype=torch.bool, device=sup_fg.device),
                    torch.zeros(sup_bg.shape[0], dtype=torch.bool, device=sup_fg.device),
                ),
                dim=0,
            )
            bg_proto = F.normalize(sup_bg.mean(dim=0), dim=0)
        else:
            sup_all = sup_fg
            sup_is_fg = torch.ones(sup_fg.shape[0], dtype=torch.bool, device=sup_fg.device)
            bg_proto = None

        fg_proto = F.normalize(sup_fg.mean(dim=0), dim=0)
        return {
            "sup_fg": sup_fg,
            "sup_all": sup_all,
            "sup_is_fg": sup_is_fg,
            "fg_proto": fg_proto,
            "bg_proto": bg_proto,
        }

    @torch.inference_mode()
    def infer_query(
        self,
        q_tok_64: torch.Tensor,
        sam_emb_64: torch.Tensor,
        ep_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mask_512 bool, prob_512 float32, score_32 float32) for one query image."""
        sup_fg = ep_state["sup_fg"]
        sup_all = ep_state["sup_all"]
        sup_is_fg = ep_state["sup_is_fg"]
        fg_proto = ep_state["fg_proto"]
        bg_proto = ep_state["bg_proto"]

        # Stage 1: Bidirectional Matching on frozen DINOv3-SAT 64x64 tokens
        matched_idx, matched_scores, fg_sim_64 = bidirectional_matching_dinov3(
            q_tok_64, sup_fg, sup_all, sup_is_fg
        )

        # Prototype contrast map for Instance-Level Mask Merging (IMM)
        q_fg_sim = (q_tok_64 @ fg_proto).reshape(64, 64)
        if bg_proto is not None:
            q_bg_sim = (q_tok_64 @ bg_proto).reshape(64, 64)
            contrast_64 = q_fg_sim - q_bg_sim
        else:
            contrast_64 = q_fg_sim

        # Stage 2: Robust Prompt Sampler (K-Means spatial clustering + hard negative anchors)
        ys = (matched_idx // 64).astype(np.float32) * 8.0 + 4.0
        xs = (matched_idx % 64).astype(np.float32) * 8.0 + 4.0
        pts_xy = np.stack((xs, ys), axis=1)

        # 2 confident background negative points (lowest foreground similarity)
        neg_indices = torch.topk(fg_sim_64.reshape(-1), k=2, largest=False).indices.cpu().numpy()
        neg_pts = np.stack(((neg_indices % 64) * 8.0 + 4.0, (neg_indices // 64) * 8.0 + 4.0), axis=1).astype(np.float32)

        cluster_labels, _ = _kmeans_2d(pts_xy, k=self.num_clusters)
        n_clusters = int(cluster_labels.max()) + 1

        # Stage 3: Decode per-cluster SAM ViT-L candidate masks & merge via DINOv3-SAT consistency
        merged_prob = torch.zeros((512, 512), dtype=torch.float32, device=q_tok_64.device)
        best_fallback_prob: torch.Tensor | None = None
        best_fallback_score = -1e9

        for c in range(n_clusters):
            cm = cluster_labels == c
            if not cm.any():
                continue
            c_pts = pts_xy[cm]
            c_sc = matched_scores[cm]
            # Pick up to 2 highest-confidence matched points in this spatial cluster
            top_k = min(2, len(c_pts))
            top_local = np.argsort(-c_sc)[:top_k]
            pos_pts = c_pts[top_local]

            all_pts = np.concatenate((pos_pts, neg_pts), axis=0)
            all_lbl = np.concatenate((np.ones(len(pos_pts), dtype=np.int64), np.zeros(len(neg_pts), dtype=np.int64)))

            low_res, iou_preds = decode_sam_masks(
                self.sam, sam_emb_64, point_coords_512=all_pts, point_labels=all_lbl, multimask_output=True
            )  # [1, 3, 256, 256], [1, 3]

            # Evaluate all 3 SAM scale masks by DINOv3-SAT foreground-vs-background contrast + SAM IoU
            best_m_idx = 0
            best_m_score = -1e9
            for m_i in range(low_res.shape[1]):
                m64 = F.interpolate(low_res[:, m_i : m_i + 1], size=(64, 64), mode="bilinear", align_corners=False)[0, 0] > 0.0
                area = int(m64.sum().item())
                if area < 1 or area > 3800:
                    continue
                sem_sc = float(contrast_64[m64].mean().item())
                tot_sc = sem_sc + 0.15 * float(iou_preds[0, m_i].item())
                if tot_sc > best_m_score:
                    best_m_score = tot_sc
                    best_m_idx = m_i

            cand_logits_512 = F.interpolate(
                low_res[:, best_m_idx : best_m_idx + 1], size=(512, 512), mode="bilinear", align_corners=False
            )[0, 0]
            cand_prob = torch.sigmoid(cand_logits_512)
            cand_m64 = F.interpolate(
                low_res[:, best_m_idx : best_m_idx + 1], size=(64, 64), mode="bilinear", align_corners=False
            )[0, 0] > 0.0
            cand_sem = float(contrast_64[cand_m64].mean().item()) if cand_m64.any() else -1.0

            if cand_sem > best_fallback_score:
                best_fallback_score = cand_sem
                best_fallback_prob = cand_prob

            if cand_sem >= self.merge_margin:
                merged_prob = torch.maximum(merged_prob, cand_prob)

        if float(merged_prob.max().item()) < 0.50 and best_fallback_prob is not None:
            merged_prob = best_fallback_prob

        prob_512 = merged_prob.detach().cpu().numpy().astype(np.float32)
        mask_512 = prob_512 >= 0.50
        score_32 = F.avg_pool2d(contrast_64[None, None], 2, stride=2)[0, 0].detach().cpu().numpy().astype(np.float32)
        return mask_512, prob_512, score_32
