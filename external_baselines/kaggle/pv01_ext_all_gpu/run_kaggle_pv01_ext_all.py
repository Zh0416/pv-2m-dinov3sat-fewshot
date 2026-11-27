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
    ensure_pkg("timm", "timm")
    ensure_pkg("transformers", "transformers")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    rev = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    print(f"Running PV01 (0.1m UAV) 7-Model External Comparison + Visuals at commit: {rev}", flush=True)

    sys.path.insert(0, str(REPO_DIR))
    from external_baselines.run_pv01_ext_all import run_pv01_ext_all

    run_pv01_ext_all(
        pv01_root=None,
        source_dir=Path("/tmp/dinov3-source"),
        dino_feat_dir=Path("/tmp/pv01_ext_dino_feats"),
        sam_cache_dir=Path("/tmp/sam_weights"),
        sam_embed_dir=Path("/tmp/pv01_ext_sam_embeds"),
        output_dir=OUT_DIR,
        cluster_km=0.25,
        episodes_per_k=5,
        device="cuda",
        num_workers=4,
    )
    print("PV01 7-Model External Comparison + Visuals complete!", flush=True)


if __name__ == "__main__":
    main()
