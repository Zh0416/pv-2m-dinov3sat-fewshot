from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
OUT_DIR = Path("/kaggle/working")


def ensure_pkg(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def find_cache_pt() -> Path:
    for p in (
        Path("/kaggle/input/pv-2m-dinov3-sat-dev-cache/dev_features_v1.pt"),
        Path("/kaggle/input/notebooks/zh0416/pv-2m-dinov3-sat-dev-cache/dev_features_v1.pt"),
    ):
        if p.is_file():
            return p
    matches = list(Path("/kaggle/input").rglob("dev_features_v1.pt"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not locate dev_features_v1.pt under /kaggle/input: {matches}")


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

    sys.path.insert(0, str(REPO_DIR))
    from pvfs.probe2 import run_probe2

    cache_pt = find_cache_pt()
    print("Probe 2 using cache:", cache_pt, flush=True)
    run_probe2(
        split_dir=REPO_DIR / "configs" / "locked_split",
        cache_pt=cache_pt,
        output_dir=OUT_DIR,
        device="cuda",
    )
    print("Probe 2 complete!", flush=True)


if __name__ == "__main__":
    main()
