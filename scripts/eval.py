from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import evaluate_matchup
from trainer.policies import policy_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline Magnate policies.")
    parser.add_argument("--games", type=int, default=50, help="Number of games to run.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="eval",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_a = policy_from_name(args.player_a_policy)
    policy_b = policy_from_name(args.player_b_policy)

    with BridgeClient() as client:
        env = MagnateBridgeEnv(client=client)
        summary = evaluate_matchup(
            env=env,
            policy_player_a=policy_a,
            policy_player_b=policy_b,
            games=args.games,
            seed_prefix=args.seed_prefix,
        )

    print(
        json.dumps(
            {
                "games": summary.games,
                "winners": summary.winners,
                "winsByPolicy": summary.wins_by_policy,
                "averageTurn": summary.average_turn,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
