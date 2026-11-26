from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

SAM_VIT_L_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
SAM_VIT_L_NAME = "sam_vit_l_0b3195.pth"


def ensure_segment_anything() -> None:
    if importlib.util.find_spec("segment_anything") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/segment-anything.git"],
            check=True,
        )


def resolve_or_download_sam_vit_l(cache_dir: Path) -> Path:
    """Locate local/Kaggle SAM ViT-L checkpoint or download automatically."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cand in (
        cache_dir / SAM_VIT_L_NAME,
        Path("/kaggle/input/segment-anything/pytorch/vit-l/1/sam_vit_l_0b3195.pth"),
        Path("/kaggle/input/sam-vit-l/sam_vit_l_0b3195.pth"),
    ):
        if cand.is_file() and cand.stat().st_size > 100_000_000:
            return cand

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = list(kaggle_input.rglob(SAM_VIT_L_NAME))
        if matches:
            return matches[0]

    dst = cache_dir / SAM_VIT_L_NAME
    print(f"[SAM] Downloading {SAM_VIT_L_NAME} from official Meta URL to {dst}...", flush=True)
    t0 = time.perf_counter()
    urllib.request.urlretrieve(SAM_VIT_L_URL, str(dst))
    print(f"[SAM] Downloaded {dst.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - t0:.1f}s", flush=True)
    return dst


def load_sam_vit_l(cache_dir: Path, device: str = "cuda") -> Any:
    ensure_segment_anything()
    from segment_anything import sam_model_registry

    ckpt = resolve_or_download_sam_vit_l(cache_dir)
    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    sam = sam_model_registry["vit_l"](checkpoint=str(ckpt))
    sam.to(run_device).eval().requires_grad_(False)
    return sam


def _sam_embed_file(embed_dir: Path, image_id: str) -> Path:
    return embed_dir / f"{Path(image_id).stem}_sam.pt"


@torch.inference_mode()
def extract_sam_embeddings_to_dir(
    sam: Any,
    rgb_512_by_id: dict[str, np.ndarray],
    image_ids: list[str],
    embed_dir: Path,
    device: str = "cuda",
) -> None:
    """Extract frozen SAM ViT-L 64x64x256 image embeddings once and cache to disk in fp16."""
    embed_dir.mkdir(parents=True, exist_ok=True)
    missing = [rid for rid in image_ids if not _sam_embed_file(embed_dir, rid).is_file()]
    if not missing:
        print(f"[SAM Cache] All {len(image_ids)} SAM embeddings already cached in {embed_dir}", flush=True)
        return

    run_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    t0 = time.perf_counter()
    for idx, rid in enumerate(missing, 1):
        rgb = rgb_512_by_id[rid]  # [512, 512, 3] uint8
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(run_device, dtype=torch.float32)
        x1024 = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        x_pre = sam.preprocess(x1024)
        emb = sam.image_encoder(x_pre)[0]  # [256, 64, 64]
        tmp = _sam_embed_file(embed_dir, rid).with_suffix(".tmp.pt")
        torch.save(emb.half().cpu(), tmp)
        tmp.replace(_sam_embed_file(embed_dir, rid))
        if idx % 100 == 0 or idx == len(missing):
            el = time.perf_counter() - t0
            print(f"[SAM Extract] {idx}/{len(missing)} ({el/60:.1f}m elapsed)", flush=True)


class LazySamEmbedMap:
    """Disk-backed lazy loader for SAM [1, 256, 64, 64] float32 embeddings."""

    def __init__(self, embed_dir: Path, device: str = "cuda", preload_ids: list[str] | None = None) -> None:
        self.embed_dir = embed_dir
        self.device = device
        self._pinned: dict[str, torch.Tensor] = {}
        self._last_id: str | None = None
        self._last_emb: torch.Tensor | None = None
        if preload_ids:
            for rid in preload_ids:
                self._pinned[rid] = self._load(rid)

    def _load(self, rid: str) -> torch.Tensor:
        e = torch.load(_sam_embed_file(self.embed_dir, rid), map_location="cpu", weights_only=False)
        return e[None].to(device=self.device, dtype=torch.float32)

    def __getitem__(self, rid: str) -> torch.Tensor:
        if rid in self._pinned:
            return self._pinned[rid]
        if self._last_id == rid and self._last_emb is not None:
            return self._last_emb
        emb = self._load(rid)
        self._last_id = rid
        self._last_emb = emb
        return emb


@torch.inference_mode()
def decode_sam_masks(
    sam: Any,
    image_embedding: torch.Tensor,
    *,
    point_coords_512: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    box_512: np.ndarray | None = None,
    mask_input_256: torch.Tensor | None = None,
    multimask_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode masks directly from cached [1, 256, 64, 64] SAM image embedding.

    Coordinates are given in 512x512 space and scaled x2 to SAM's 1024x1024 input grid.
    Returns (low_res_logits [1, M, 256, 256], iou_preds [1, M]).
    """
    dev = image_embedding.device
    pts = None
    if point_coords_512 is not None and point_labels is not None and len(point_coords_512) > 0:
        coords_1024 = torch.as_tensor(point_coords_512 * 2.0, dtype=torch.float32, device=dev)[None]
        labels_t = torch.as_tensor(point_labels, dtype=torch.int64, device=dev)[None]
        pts = (coords_1024, labels_t)

    boxes_t = None
    if box_512 is not None:
        boxes_t = torch.as_tensor(box_512 * 2.0, dtype=torch.float32, device=dev)[None]

    sparse_emb, dense_emb = sam.prompt_encoder(
        points=pts,
        boxes=boxes_t,
        masks=mask_input_256,
    )
    low_res_masks, iou_preds = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=multimask_output,
    )
    return low_res_masks, iou_preds
