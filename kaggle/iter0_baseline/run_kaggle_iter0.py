from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
OUT_DIR = Path("/kaggle/working")
SOURCE_DIR = Path("/tmp/dinov3-source")
CACHE_DIR = Path("/tmp/dino_cache_iter0")


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
    from pvfs.runner import run_iter0

    summary = run_iter0(
        split_dir=REPO_DIR / "configs" / "locked_split",
        output_dir=OUT_DIR,
        source_dir=SOURCE_DIR,
        cache_dir=CACHE_DIR,
        device="cuda",
    )
    print("Iter0 complete! Device:", summary["device"], flush=True)


if __name__ == "__main__":
    main()
