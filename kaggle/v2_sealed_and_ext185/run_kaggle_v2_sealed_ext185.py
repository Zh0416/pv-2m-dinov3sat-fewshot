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


def find_input_file(filename: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(filename))
    if len(matches) == 1:
        return matches[0]
    for m in matches:
        if filename in m.name:
            return m
    raise FileNotFoundError(f"Could not locate {filename} under /kaggle/input: {matches}")


def main() -> None:
    ensure_pkg("rasterio", "rasterio")
    if (REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

    sys.path.insert(0, str(REPO_DIR))
    from pvfs.v2_sealed_and_ext185 import run_v2_sealed_and_ext185

    dev_cache_pt = find_input_file("dev_features_v1.pt")
    decoders_pt = find_input_file("v2_fold_decoders.pt")
    print("Using dev cache:", dev_cache_pt, flush=True)
    print("Using V2 decoders:", decoders_pt, flush=True)

    run_v2_sealed_and_ext185(
        split_dir=REPO_DIR / "configs" / "locked_split",
        dev_cache_pt=dev_cache_pt,
        decoders_pt=decoders_pt,
        source_dir=Path("/tmp/dinov3_repo"),
        output_dir=OUT_DIR,
        device="cuda",
    )
    print("V2 Sealed 21 + External PV-512-MERGED-185 complete!", flush=True)


if __name__ == "__main__":
    main()
