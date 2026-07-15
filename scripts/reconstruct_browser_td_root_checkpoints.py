from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trainer.td.browser_pack_checkpoint import (
    BrowserPackCheckpointError,
    reconstruct_browser_td_root_checkpoints,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct trainer-compatible ValueNet and OpponentModel checkpoints "
            "from a static td-root-search-v1 browser model pack."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the browser pack manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the reconstructed checkpoint pair.",
    )
    parser.add_argument(
        "--value-filename",
        default="value.pt",
        help="Plain output filename for the value checkpoint (default: value.pt).",
    )
    parser.add_argument(
        "--opponent-filename",
        default="opponent.pt",
        help="Plain output filename for the opponent checkpoint (default: opponent.pt).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing checkpoint outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = reconstruct_browser_td_root_checkpoints(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            value_filename=args.value_filename,
            opponent_filename=args.opponent_filename,
            overwrite=args.overwrite,
        )
    except (BrowserPackCheckpointError, OSError, RuntimeError) as error:
        raise SystemExit(f"Browser-pack checkpoint reconstruction failed: {error}") from error
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
