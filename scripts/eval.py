from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        help="Policy name for PlayerA (random|heuristic|bc|ppo).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        default="random",
        help="Policy name for PlayerB (random|heuristic|bc|ppo).",
    )
    parser.add_argument(
        "--player-a-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --player-a-policy=bc or ppo.",
    )
    parser.add_argument(
        "--player-b-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --player-b-policy=bc or ppo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_a = policy_from_name(args.player_a_policy, checkpoint_path=args.player_a_checkpoint)
    policy_b = policy_from_name(args.player_b_policy, checkpoint_path=args.player_b_checkpoint)

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
