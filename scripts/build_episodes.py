from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvfs.data import load_dev_records
from pvfs.episodes import load_and_verify_episodes, save_episodes_lock


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    split_dir = root / "configs" / "locked_split"
    _, _, dev_records = load_dev_records(split_dir)
    out = split_dir / "episodes_lock.json"
    payload = save_episodes_lock(split_dir, dev_records, out)
    load_and_verify_episodes(split_dir, out, dev_records)
    print(f"Generated and verified {payload['total_episodes']} episodes at {out}")


if __name__ == "__main__":
    main()
