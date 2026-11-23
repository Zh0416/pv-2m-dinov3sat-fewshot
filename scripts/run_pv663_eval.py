from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pvfs.pv663_eval import run_pv663_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1/3/5-shot best model (R + M3_Deformable_Grad) evaluation on PV663.")
    parser.add_argument("--split-json", type=Path, default=REPO_ROOT / "configs" / "pv663_split_lock.json")
    parser.add_argument("--episodes-json", type=Path, default=REPO_ROOT / "configs" / "pv663_episodes_lock.json")
    parser.add_argument("--deform-config", type=Path, default=REPO_ROOT / "configs" / "three_module_exp_m3_deformgrad.json")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/dinov3-source"))
    parser.add_argument("--feat-cache-dir", type=Path, default=Path("/tmp/pv663_feat_cache"))
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "pv663_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_pv663_benchmark(
        split_json=args.split_json,
        episodes_json=args.episodes_json,
        deform_config_json=args.deform_config,
        source_dir=args.source_dir,
        feat_cache_dir=args.feat_cache_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        weights_path=args.weights_path,
        device=args.device,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
