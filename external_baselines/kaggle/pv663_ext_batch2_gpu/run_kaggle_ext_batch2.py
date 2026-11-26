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


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    ensure_pkg("segmentation_models_pytorch", "segmentation-models-pytorch")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    rev = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    print(f"Running PV663 External Batch 2 (U-Net / SegFormer / DINOv3-FT) at commit: {rev}", flush=True)

    sys.path.insert(0, str(REPO_DIR))
    from external_baselines.run_pv663_ext_batch2 import run_pv663_ext_batch2

    run_pv663_ext_batch2(
        split_json=REPO_DIR / "configs" / "pv663_split_lock.json",
        episodes_json=REPO_DIR / "configs" / "pv663_episodes_lock.json",
        ours_ref_json=REPO_DIR / "external_baselines" / "results" / "pv663_batch1" / "ours_v31_site_ref.json",
        source_dir=Path("/tmp/dinov3-source"),
        dino_feat_dir=Path("/tmp/pv663_ext2_dino_feats"),
        output_dir=OUT_DIR,
        episodes_per_k=5,
        device="cuda",
    )
    print("PV663 External Batch 2 complete!", flush=True)


if __name__ == "__main__":
    main()
