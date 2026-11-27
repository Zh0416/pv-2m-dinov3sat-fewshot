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
    ensure_pkg("segmentation_models_pytorch", "segmentation-models-pytorch")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    rev = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    print(f"Running PV-100scence 7-Model External Comparison + Visuals at commit: {rev}", flush=True)

    sys.path.insert(0, str(REPO_DIR))
    from external_baselines.run_pv100_ext_all import run_pv100_ext_all

    run_pv100_ext_all(
        ext_split_json=REPO_DIR / "external_baselines" / "configs" / "pv100_ext_split_lock.json",
        dev_cache_pt=find_dev_cache_pt(),
        source_dir=Path("/tmp/dinov3-source"),
        sam_cache_dir=Path("/tmp/sam_weights"),
        sam_embed_dir=Path("/tmp/pv100_ext_sam_embeds"),
        output_dir=OUT_DIR,
        device="cuda",
        num_workers=4,
    )
    print("PV-100scence 7-Model External Comparison + Visuals complete!", flush=True)


if __name__ == "__main__":
    main()
