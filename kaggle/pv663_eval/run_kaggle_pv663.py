from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_BRANCH = "phase-boundary-dev"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
SOURCE_DIR = Path("/tmp/dinov3-source")
FEAT_CACHE_DIR = Path("/tmp/pv663_feat_cache")
OUT_DIR = Path("/kaggle/working")


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

    rev = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    print(f"Running PV663 1/3/5-shot Best Model (R + M3_Deformable_Grad) on Kaggle GPU at commit: {rev}", flush=True)

    sys.path.insert(0, str(REPO_DIR))
    from pvfs.pv663_eval import run_pv663_benchmark

    run_pv663_benchmark(
        split_json=REPO_DIR / "configs" / "pv663_split_lock.json",
        episodes_json=REPO_DIR / "configs" / "pv663_episodes_lock.json",
        deform_config_json=REPO_DIR / "configs" / "three_module_exp_m3_deformgrad.json",
        source_dir=SOURCE_DIR,
        feat_cache_dir=FEAT_CACHE_DIR,
        output_dir=OUT_DIR,
        device="cuda",
        num_workers=4,
    )
    print("PV663 1/3/5-shot evaluation complete!", flush=True)


if __name__ == "__main__":
    main()
