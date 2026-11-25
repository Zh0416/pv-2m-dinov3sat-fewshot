from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_BRANCH = "m3-pbdc-slim-params"
REPO_DIR = Path("/tmp/pv-2m-dinov3sat-fewshot")
SOURCE_DIR = Path("/tmp/dinov3-source")
PV663_FEAT_DIR = Path("/tmp/pv663_subset_feats")
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
    raise FileNotFoundError(f"Could not locate dev_features_v1.pt under /kaggle/input: {matches}")


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    rev = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    print(f"Running Module 3 Cross-Resolution (2m + 0.8m) Subset Benchmark at commit: {rev}", flush=True)

    sys.path.insert(0, str(REPO_DIR))
    from scripts.run_m3_dual_subset_eval import run_dual_subset_benchmark

    run_dual_subset_benchmark(
        split_dir=REPO_DIR / "configs" / "locked_split",
        dev_cache_pt=find_dev_cache_pt(),
        pv663_subset_json=REPO_DIR / "configs" / "pv663_subset_lock.json",
        deform_config_json=REPO_DIR / "configs" / "three_module_exp_m3_deformgrad.json",
        source_dir=SOURCE_DIR,
        pv663_feat_dir=PV663_FEAT_DIR,
        output_dir=OUT_DIR,
        device="cuda",
        num_workers=4,
    )
    print("Module 3 Dual-Dataset Subset Benchmark complete!", flush=True)


if __name__ == "__main__":
    main()
