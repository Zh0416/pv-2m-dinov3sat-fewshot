from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
OUT_DIR = Path("/kaggle/working")
SOURCE_DIR = Path("/tmp/dinov3-source")


def ensure_pkg(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

    sys.path.insert(0, str(REPO_DIR))
    from pvfs.cache_builder import build_multiscale_dev_cache, run_layer_scale_probe

    split_dir = REPO_DIR / "configs" / "locked_split"
    cache_pt = build_multiscale_dev_cache(
        split_dir=split_dir,
        output_dir=OUT_DIR,
        source_dir=SOURCE_DIR,
        device="cuda",
    )
    run_layer_scale_probe(
        split_dir=split_dir,
        cache_pt=cache_pt,
        output_dir=OUT_DIR,
        device="cuda",
    )
    print("Cache & Layer/Scale Probe complete!", flush=True)


if __name__ == "__main__":
    main()
