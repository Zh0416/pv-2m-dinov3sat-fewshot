from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_BRANCH = "clean-core-v1"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
OUT_DIR = Path("/kaggle/working")


def ensure_pkg(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def find_dev_cache_pt() -> Path:
    for p in (
        Path("/kaggle/input/pv-2m-dinov3-sat-dev-cache/dev_features_v1.pt"),
        Path("/kaggle/input/notebooks/zh0416/pv-2m-dinov3-sat-dev-cache/dev_features_v1.pt"),
    ):
        if p.is_file():
            return p
    matches = list(Path("/kaggle/input").rglob("dev_features_v1.pt"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not locate dev_features_v1.pt: {matches}")


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    sys.path.insert(0, str(REPO_DIR))
    from scripts.render_pv100_all100 import run_pv100_all_100

    run_pv100_all_100(
        split_dir=REPO_DIR / "configs" / "locked_split",
        dev_cache_pt=find_dev_cache_pt(),
        source_dir=Path("/tmp/dinov3-source"),
        out_dir=OUT_DIR,
        device="cuda",
    )
    print("PV-100scence all 100 images rendered!", flush=True)


if __name__ == "__main__":
    main()
