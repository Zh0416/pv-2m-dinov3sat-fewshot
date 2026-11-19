from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_verified_rgb
from .io import read_json, sha256_file, sha256_json, write_json

DINOV3_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
PINNED_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
EXPECTED_WEIGHT_SHA_PREFIX = "eadcf0ff"
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
PREPROCESS_VERSION = "sat493m-uint8-float32-v1"


def resolve_dinov3_weights(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None and explicit_path.is_file():
        return explicit_path
    candidates = [
        Path("/kaggle/input/datasets/zh0416/dinov3-sat-geovit-l-14-336/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
        Path("/kaggle/input/dinov3-sat-geovit-l-14-336/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
        Path("/kaggle/input/datasets/zh0416/pv-dinov3sat-encoder-checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = list(kaggle_input.rglob("dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"))
        if matches:
            return matches[0]
    raise FileNotFoundError("Could not find dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")


def prepare_dinov3_source(source_dir: Path) -> Path:
    local_precloned = Path(r"E:\PV-DINOV3SAT\codex运行文件\dinov3")
    if not source_dir.exists() and local_precloned.is_dir():
        source_dir = local_precloned
    source = source_dir.resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", DINOV3_REPO_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", PINNED_COMMIT], check=True)
    if not (source / ".git").is_dir():
        raise RuntimeError(f"Expected Git repo at {source}")
    rev = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    if rev.returncode != 0 or rev.stdout.strip() != PINNED_COMMIT:
        subprocess.run(["git", "-C", str(source), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", PINNED_COMMIT], check=True)
        rev = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    if rev.stdout.strip() != PINNED_COMMIT:
        raise RuntimeError(f"DINOv3 source commit mismatch: {rev.stdout.strip()} != {PINNED_COMMIT}")
    if not (source / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError("Missing dinov3/hub/backbones.py in source directory")
    return source


class DinoSATExtractor:
    """Pure frozen DINOv3-SAT ViT-L/16 extractor with provenance-checked caching."""

    def __init__(
        self,
        *,
        source_dir: Path,
        weights_path: Path | None = None,
        cache_dir: Path,
        device: str = "cuda",
        layers: tuple[int, ...] = (20, 21, 22, 23),
        input_size: int = 512,
    ):
        self.source_dir = source_dir
        self.weights_path = weights_path
        self.cache_dir = cache_dir
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.layers = tuple(layers)
        self.input_size = int(input_size)
        self.model = None
        self.weight_sha256 = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.signature = {
            "schema": "pvfs-dino-cache-v1",
            "commit": PINNED_COMMIT,
            "weight_prefix": EXPECTED_WEIGHT_SHA_PREFIX,
            "layers": list(self.layers),
            "input_size": self.input_size,
            "mean": list(SAT_MEAN),
            "std": list(SAT_STD),
            "preprocess": PREPROCESS_VERSION,
        }
        sig_path = self.cache_dir / "signature.json"
        if sig_path.is_file():
            if read_json(sig_path) != self.signature:
                raise ValueError(f"Existing cache at {self.cache_dir} has a different signature")
        else:
            write_json(sig_path, self.signature)

    def _load_model(self) -> None:
        weights = resolve_dinov3_weights(self.weights_path)
        digest = sha256_file(weights)
        if not digest.startswith(EXPECTED_WEIGHT_SHA_PREFIX):
            raise ValueError(f"DINOv3 checkpoint SHA-256 mismatch: {digest}")
        self.weight_sha256 = digest
        source = prepare_dinov3_source(self.source_dir)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        backbones = importlib.import_module("dinov3.hub.backbones")
        if Path(backbones.__file__).resolve() != (source / "dinov3" / "hub" / "backbones.py").resolve():
            raise RuntimeError("Imported dinov3 module from unexpected path")
        model = backbones.dinov3_vitl16(pretrained=True, weights=str(weights.resolve()))
        model.to(self.device).eval().requires_grad_(False)
        if len(model.blocks) != 24:
            raise RuntimeError(f"Expected 24 blocks in vit_large, got {len(model.blocks)}")
        self.model = model

    @torch.inference_mode()
    def extract_image_features(self, rgb_uint8: np.ndarray) -> dict[str, np.ndarray]:
        if self.model is None:
            self._load_model()
        x = torch.from_numpy(rgb_uint8.copy())[None].to(self.device).float() / 255.0
        mean = x.new_tensor(SAT_MEAN)[None, :, None, None]
        std = x.new_tensor(SAT_STD)[None, :, None, None]
        x = (x - mean) / std
        if self.input_size != 512:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        outs = self.model.get_intermediate_layers(x, n=list(self.layers), reshape=True, norm=True)
        stacked = torch.stack([t[0].float() for t in outs], dim=0)  # [L, 1024, H_p, W_p]
        last4_mean = stacked.mean(dim=0).cpu().numpy().astype(np.float32)
        return {"last4_mean": last4_mean}

    def get_features(self, image_dir: Path, row: dict) -> dict[str, np.ndarray]:
        if row.get("partition") == "test":
            raise PermissionError("Refusing to extract or load sealed test features")
        stem = sha256_json({"id": row["id"]})[:24]
        npz_path = self.cache_dir / f"{stem}.npz"
        meta_path = self.cache_dir / f"{stem}.json"
        if npz_path.is_file() and meta_path.is_file():
            meta = read_json(meta_path)
            if (
                meta.get("id") == row["id"]
                and meta.get("image_sha256") == row["image_sha256"]
                and meta.get("npz_sha256") == sha256_file(npz_path)
            ):
                with np.load(npz_path, allow_pickle=False) as arc:
                    return {k: arc[k].copy() for k in arc.files}

        rgb = read_verified_rgb(image_dir, row)
        feats = self.extract_image_features(rgb)
        if self.model is not None and not getattr(self, "_determinism_checked", False):
            feats2 = self.extract_image_features(rgb)
            for k in feats:
                if not np.allclose(feats[k], feats2[k], rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f"Non-deterministic DINOv3 output on {row['id']}")
            self._determinism_checked = True

        tmp = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **feats)
        tmp.replace(npz_path)
        write_json(
            meta_path,
            {
                "id": row["id"],
                "image_sha256": row["image_sha256"],
                "npz_sha256": sha256_file(npz_path),
            },
        )
        return feats

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
