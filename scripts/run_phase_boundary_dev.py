from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.phase_boundary_eval import run_phase_boundary_dev_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3_Pruned Phase Boundary Refinement Dev evaluation & ablations.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "locked_split",
        help="Path to locked_split directory containing split_lock.json and episodes_lock.json",
    )
    parser.add_argument(
        "--cache-pt",
        type=Path,
        required=True,
        help="Path to dev_features_v1.pt multiscale DINOv3-SAT feature cache",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=REPO_ROOT / "configs" / "phase_boundary_locked.json",
        help="Path to pre-registered phase_boundary_locked.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "phase_boundary_dev",
        help="Directory to save phase_boundary_summary.json and diagnostic panels",
    )
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional override for raw RGB image directory")
    parser.add_argument("--label-dir", type=Path, default=None, help="Optional override for GT mask directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device for baseline M3_Pruned_Ref scoring (cuda or cpu)")
    args = parser.parse_args()

    summary = run_phase_boundary_dev_evaluation(
        split_dir=args.split_dir,
        cache_pt=args.cache_pt,
        output_dir=args.output_dir,
        config_json=args.config_json,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        device=args.device,
    )
    print("Verdicts:", summary["verdicts"], flush=True)


if __name__ == "__main__":
    main()
