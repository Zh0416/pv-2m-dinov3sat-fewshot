from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_BRANCH = "clean-core-v1"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
OUT_DIR = Path("/kaggle/working")
SOURCE_DIR = Path("/tmp/dinov3-source")


def ensure_pkg(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    sys.path.insert(0, str(REPO_DIR))
    from pvfs.cache_builder import build_multiscale_dev_cache

    build_multiscale_dev_cache(
        split_dir=REPO_DIR / "configs" / "locked_split",
        output_dir=OUT_DIR,
        source_dir=SOURCE_DIR,
        device="cuda",
    )
    print("CleanCore Dev feature cache extraction complete!", flush=True)


if __name__ == "__main__":
    main()
