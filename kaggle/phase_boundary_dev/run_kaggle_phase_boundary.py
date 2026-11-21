from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/Zh0416/pv-2m-dinov3sat-fewshot.git"
REPO_BRANCH = "phase-boundary-dev"
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
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

    rev = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    print(f"Running Phase Boundary Dev Evaluation at commit: {rev}", flush=True)

    sys.path.insert(0, str(REPO_DIR))
    from pvfs.phase_boundary_eval import run_phase_boundary_dev_evaluation

    cache_pt = find_cache_pt()
    print("Using cached dev features:", cache_pt, flush=True)
    summary = run_phase_boundary_dev_evaluation(
        split_dir=REPO_DIR / "configs" / "locked_split",
        cache_pt=cache_pt,
        output_dir=OUT_DIR,
        config_json=REPO_DIR / "configs" / "phase_boundary_locked.json",
        device="cuda",
    )
    print("Phase Boundary Dev evaluation complete! Verdicts:", summary["verdicts"], flush=True)


if __name__ == "__main__":
    main()
