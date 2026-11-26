from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .sam_utils import decode_sam_masks


def compute_persam_sim_and_points(
    sam_emb_64: torch.Tensor,
    target_proto: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute PerSAM 64x64 cosine similarity map + top-1 positive & top-1 negative point in 512x512 coords."""
    # sam_emb_64: [1, 256, 64, 64], target_proto: [256] L2-normalized
    feat = F.normalize(sam_emb_64[0], dim=0)  # [256, 64, 64]
    sim = torch.einsum("chw,c->hw", feat, target_proto)  # [64, 64]
    flat = sim.reshape(-1)
    pos_idx = int(torch.argmax(flat).item())
    neg_idx = int(torch.argmin(flat).item())

    py, px = divmod(pos_idx, 64)
    ny, nx = divmod(neg_idx, 64)
    # Map 64x64 token centers to 512x512 pixel coordinates (stride 8, center +4)
    pts_512 = np.array([[px * 8.0 + 4.0, py * 8.0 + 4.0], [nx * 8.0 + 4.0, ny * 8.0 + 4.0]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int64)
    return sim, pts_512, labels


def _mask_bbox_512(mask_512: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask_512)
    if len(ys) == 0:
        return None
    return np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum()
    return 1.0 - (2.0 * inter + 1.0) / (probs.sum() + target.sum() + 1.0)


class PerSAMFRunner:
    """Faithful PerSAM-F (ICLR 2024) implementation supporting K-shot episodes on cached SAM ViT-L embeddings.

    Follows Zhang et al. (ICLR 2024):
      1. Support-masked L2-normalized SAM image embedding prototype T_F;
      2. Scale-weight fine-tuning (PerSAM-F): learns 3 scalar weights combining SAM's 3 multi-scale masks
         on the K support images (100 Adam steps, Dice + BCE loss, zero SAM backbone updates);
      3. Two-step cascaded decoding on each query image:
         Step 1: (pos_pt, neg_pt) -> weighted multi-scale coarse mask;
         Step 2: (pos_pt, neg_pt, Step-1 bbox, Step-1 low-res logit) -> final refined mask.
    """

    def __init__(self, sam: Any, *, ft_steps: int = 100, lr: float = 0.05) -> None:
        self.sam = sam
        self.ft_steps = int(ft_steps)
        self.lr = float(lr)

    def fit_episode(
        self,
        sam_embeds: Any,
        mask_by_id: dict[str, np.ndarray],
        support_ids: list[str],
    ) -> dict[str, Any]:
        # 1. Build multi-shot target visual prototype T_F
        protos = []
        sup_embs = []
        for sid in support_ids:
            emb = sam_embeds[sid]  # [1, 256, 64, 64]
            sup_embs.append(emb)
            m512 = torch.from_numpy((mask_by_id[sid] > 0).astype(np.float32)).to(emb.device)[None, None]
            m64 = F.interpolate(m512, size=(64, 64), mode="area")[0, 0] > 0.25
            if not m64.any():
                m64 = F.interpolate(m512, size=(64, 64), mode="bilinear", align_corners=False)[0, 0] > 0.05
            feat_norm = F.normalize(emb[0], dim=0)  # [256, 64, 64]
            p_s = feat_norm[:, m64].mean(dim=1)
            protos.append(F.normalize(p_s, dim=0))
        target_proto = F.normalize(torch.stack(protos, dim=0).mean(dim=0), dim=0)

        # 2. Precompute the 3 multi-scale low-res mask logits on each support image for PerSAM-F weight tuning
        sup_logits_list = []
        sup_targets_list = []
        for sid, emb in zip(support_ids, sup_embs):
            _, pts_512, labels = compute_persam_sim_and_points(emb, target_proto)
            low_res, _ = decode_sam_masks(
                self.sam, emb, point_coords_512=pts_512, point_labels=labels, multimask_output=True
            )  # [1, 3, 256, 256]
            gt_256 = F.interpolate(
                torch.from_numpy((mask_by_id[sid] > 0).astype(np.float32)).to(emb.device)[None, None],
                size=(256, 256),
                mode="nearest",
            )[0, 0]
            sup_logits_list.append(low_res[0].detach().clone())  # [3, 256, 256]
            sup_targets_list.append(gt_256.detach().clone())

        # 3. Optimize only the 3 PerSAM-F scale weights (all SAM parameters remain strictly frozen)
        dev = target_proto.device
        weights = torch.nn.Parameter(torch.ones(3, device=dev) / 3.0)
        opt = torch.optim.Adam([weights], lr=self.lr)
        for _ in range(self.ft_steps):
            opt.zero_grad()
            w_norm = torch.softmax(weights, dim=0)
            loss = torch.zeros((), device=dev)
            for l3, gt256 in zip(sup_logits_list, sup_targets_list):
                comb = (w_norm[:, None, None] * l3).sum(dim=0)
                loss = loss + _dice_loss(comb, gt256) + F.binary_cross_entropy_with_logits(comb, gt256)
            loss = loss / len(sup_logits_list)
            loss.backward()
            opt.step()

        final_w = torch.softmax(weights.detach(), dim=0)
        return {"target_proto": target_proto, "scale_weights": final_w}

    @torch.inference_mode()
    def infer_query(
        self,
        sam_emb_64: torch.Tensor,
        ep_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mask_512 bool, prob_512 float32, score_32 float32) for one query image."""
        target_proto = ep_state["target_proto"]
        scale_w = ep_state["scale_weights"]  # [3]

        sim_64, pts_512, labels = compute_persam_sim_and_points(sam_emb_64, target_proto)

        # Step 1: Point-prompted multi-scale decoding weighted by PerSAM-F scale weights
        low_res_1, _ = decode_sam_masks(
            self.sam, sam_emb_64, point_coords_512=pts_512, point_labels=labels, multimask_output=True
        )
        comb_1 = (scale_w[:, None, None] * low_res_1[0]).sum(dim=0, keepdim=True)[None]  # [1, 1, 256, 256]
        m512_1 = (F.interpolate(comb_1, size=(512, 512), mode="bilinear", align_corners=False)[0, 0] > 0.0).cpu().numpy()

        # Step 2: Cascaded box + point + low-res mask refinement
        box_1 = _mask_bbox_512(m512_1)
        if box_1 is not None:
            low_res_2, _ = decode_sam_masks(
                self.sam,
                sam_emb_64,
                point_coords_512=pts_512,
                point_labels=labels,
                box_512=box_1,
                mask_input_256=comb_1,
                multimask_output=True,
            )
            comb_2 = (scale_w[:, None, None] * low_res_2[0]).sum(dim=0, keepdim=True)[None]
        else:
            comb_2 = comb_1

        logits_512 = F.interpolate(comb_2, size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        prob_512 = torch.sigmoid(logits_512).cpu().numpy().astype(np.float32)
        mask_512 = (logits_512 > 0.0).cpu().numpy().astype(bool)
        score_32 = F.avg_pool2d(sim_64[None, None], 2, stride=2)[0, 0].cpu().numpy().astype(np.float32)
        return mask_512, prob_512, score_32
