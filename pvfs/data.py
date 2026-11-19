from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .io import read_json, sha256_file


def load_dev_records(split_dir: Path) -> tuple[dict, dict, list[dict]]:
    """Load locked manifest/split and return strictly development records only."""
    manifest_path = split_dir / "dataset_manifest.json"
    split_path = split_dir / "split_lock.json"
    manifest = read_json(manifest_path)
    split = read_json(split_path)
    if split.get("status") != "locked_after_manual_qc" or not manifest.get("manual_qc_confirmed"):
        raise PermissionError("Locked manual QC split is required")

    assignment = split["assignment"]
    by_id = {row["id"]: row for row in manifest["images"]}
    if set(by_id) != set(assignment):
        raise ValueError("Manifest and split IDs mismatch")

    dev_records = []
    for image_id in sorted(by_id):
        info = assignment[image_id]
        if info["partition"] == "test":
            continue
        if info["partition"] != "dev" or info["fold"] not in (0, 1, 2, 3):
            raise ValueError(f"Unexpected partition/fold for {image_id}: {info}")
        row = by_id[image_id]
        dev_records.append(
            {
                "id": image_id,
                "site": info["site"],
                "fold": int(info["fold"]),
                "partition": "dev",
                "positive": bool(row["positive"]),
                "target_fraction": float(row["target_fraction"]),
                "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"],
                "meters_per_pixel": list(row["meters_per_pixel"]),
            }
        )
    if len(dev_records) != 79:
        raise ValueError(f"Expected 79 dev images, found {len(dev_records)}")
    return manifest, split, dev_records


def resolve_data_dirs(image_dir: Path | None = None, label_dir: Path | None = None) -> tuple[Path, Path]:
    """Locate paired PV-100scence images/ and labels/ roots cleanly."""
    probe = Path("正样本/PV-01.tif")
    if image_dir is not None and label_dir is not None:
        if (image_dir / probe).is_file() and (label_dir / probe).is_file():
            return image_dir, label_dir
        raise FileNotFoundError(f"Configured data dirs lack {probe}: {image_dir}, {label_dir}")

    candidates = [
        Path(r"E:\PV-多场景\images"),
        Path("/kaggle/input/datasets/zh0416/pv-100scence/PV-多场景/images"),
        Path("/kaggle/input/pv-100scence/PV-多场景/images"),
    ]
    for img_root in candidates:
        lbl_root = img_root.parent / "labels"
        if (img_root / probe).is_file() and (lbl_root / probe).is_file():
            return img_root, lbl_root

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        found = [
            p
            for p in kaggle_input.rglob("images")
            if p.is_dir() and (p / probe).is_file() and (p.parent / "labels" / probe).is_file()
        ]
        if len(found) == 1:
            return found[0], found[0].parent / "labels"

    raise FileNotFoundError("Could not locate PV-100scence images/ and labels/ directories")


def read_verified_rgb(image_dir: Path, row: dict) -> np.ndarray:
    if row.get("partition") == "test":
        raise PermissionError("Refusing to read sealed test image")
    path = image_dir / row["id"]
    if sha256_file(path) != row["image_sha256"]:
        raise ValueError(f"Image SHA-256 mismatch for {row['id']}")
    with rasterio.open(path) as src:
        rgb = src.read()
    if rgb.shape != (3, 512, 512) or rgb.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB shape/dtype for {row['id']}: {rgb.shape}, {rgb.dtype}")
    return rgb


def read_verified_mask(label_dir: Path, row: dict) -> np.ndarray:
    if row.get("partition") == "test":
        raise PermissionError("Refusing to read sealed test label")
    path = label_dir / row["id"]
    if sha256_file(path) != row["label_sha256"]:
        raise ValueError(f"Label SHA-256 mismatch for {row['id']}")
    with rasterio.open(path) as src:
        mask = src.read(1)
    if mask.shape != (512, 512) or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"Unexpected binary mask for {row['id']}")
    return (mask > 0).astype(np.uint8)


def area_target_32(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (512, 512):
        raise ValueError(f"Expected 512x512 mask, got {mask.shape}")
    return mask.astype(np.float32).reshape(32, 16, 32, 16).mean(axis=(1, 3))
