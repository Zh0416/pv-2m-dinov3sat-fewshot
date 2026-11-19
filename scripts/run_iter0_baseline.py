from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvfs.runner import run_iter0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("configs/locked_split"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iter0"))
    parser.add_argument("--source-dir", type=Path, default=Path("/kaggle/working/dinov3-source"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/dino_cache"))
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_iter0(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        cache_dir=args.cache_dir,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        weights_path=args.weights_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
