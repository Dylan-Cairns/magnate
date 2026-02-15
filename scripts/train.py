from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.policies import policy_from_name
from trainer.training import collect_training_samples, write_samples_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Magnate training samples through the TS bridge."
    )
    parser.add_argument("--games", type=int, default=20, help="Number of games to sample.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="train",
        help="Seed prefix used to derive deterministic per-game seeds.",
    )
    parser.add_argument(
        "--player-a-policy",
        type=str,
        default="heuristic",
        help="Policy name for PlayerA (random|heuristic).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        default="random",
        help="Policy name for PlayerB (random|heuristic).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/training_samples.jsonl"),
        help="Output JSONL path for decision samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_a = policy_from_name(args.player_a_policy)
    policy_b = policy_from_name(args.player_b_policy)

    with BridgeClient() as client:
        env = MagnateBridgeEnv(client=client)
        samples = collect_training_samples(
            env=env,
            policy_player_a=policy_a,
            policy_player_b=policy_b,
            games=args.games,
            seed_prefix=args.seed_prefix,
        )

    write_samples_jsonl(samples, args.out)
    print(f"Collected {len(samples)} decision samples into {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
