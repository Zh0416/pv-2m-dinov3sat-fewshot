from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from .data import load_dev_records, read_verified_rgb, resolve_data_dirs
from .dinov3_backbone import PINNED_COMMIT, SAT_MEAN, SAT_STD, prepare_dinov3_source, resolve_dinov3_weights
from .io import sha256_file

LAYER_GROUPS_32 = {
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

    layers_32 = sorted({l for grp in LAYER_GROUPS_32.values() for l in grp})
    pos_32 = {l: i for i, l in enumerate(layers_32)}
    layers_64 = sorted({l for grp in LAYER_GROUPS_64.values() for l in grp})
    pos_64 = {l: i for i, l in enumerate(layers_64)}

    mean_t = torch.tensor(SAT_MEAN, device=run_device, dtype=torch.float32)[None, :, None, None]
    std_t = torch.tensor(SAT_STD, device=run_device, dtype=torch.float32)[None, :, None, None]

    cache_payload: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for idx, row in enumerate(dev_records, 1):
            if row["partition"] != "dev":
                raise PermissionError("Refusing to cache sealed test image")
            rgb = read_verified_rgb(img_dir, row)
            x512 = (torch.from_numpy(rgb.copy())[None].to(run_device).float() / 255.0 - mean_t) / std_t
            outs_32 = model.get_intermediate_layers(x512, n=layers_32, reshape=True, norm=True)

            entry: dict[str, torch.Tensor] = {}
            for name, grp in LAYER_GROUPS_32.items():
                stacked = torch.stack([outs_32[pos_32[l]][0].float() for l in grp], dim=0).mean(dim=0)
                normed = F.normalize(stacked, dim=0)
                if not torch.isfinite(normed).all():
                    raise RuntimeError(f"Non-finite 512 feature for {row['id']} / {name}")
                entry[name] = normed.half().cpu()

            x1024 = F.interpolate(x512, size=(1024, 1024), mode="bilinear", align_corners=False)
            outs_64 = model.get_intermediate_layers(x1024, n=layers_64, reshape=True, norm=True)
            for name, grp in LAYER_GROUPS_64.items():
                stacked = torch.stack([outs_64[pos_64[l]][0].float() for l in grp], dim=0).mean(dim=0)
                normed = F.normalize(stacked, dim=0)
                if not torch.isfinite(normed).all():
                    raise RuntimeError(f"Non-finite 1024 feature for {row['id']} / {name}")
                entry[name] = normed.half().cpu()

            cache_payload[row["id"]] = entry
            if idx % 20 == 0 or idx == len(dev_records):
                print(f"[Cache] Extracted 512+1024 features {idx}/{len(dev_records)}", flush=True)

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
