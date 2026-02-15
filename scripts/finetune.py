from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.behavior_cloning import (
    BehaviorCloningModel,
    load_behavior_cloning_checkpoint,
    save_behavior_cloning_checkpoint,
)
from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import MatchSummary, evaluate_matchup
from trainer.policies import BehaviorCloningPolicy, policy_from_name
from trainer.reinforcement import ReinforceConfig, fine_tune_with_reinforce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a BC checkpoint via seeded REINFORCE self-play."
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        required=True,
        help="Input BC checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("artifacts/rl_checkpoint.json"),
        help="Output fine-tuned checkpoint path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of self-play episodes.",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="rl",
        help="Seed prefix used for deterministic self-play episodes.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01, help="REINFORCE learning rate.")
    parser.add_argument("--l2", type=float, default=1e-5, help="REINFORCE L2 regularization.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for self-play action selection.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--max-decisions-per-game",
        type=int,
        default=2000,
        help="Safety guard to prevent runaway episodes.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="If >0, run final eval against random and heuristic with this many games each.",
    )
    parser.add_argument(
        "--eval-seed-prefix",
        type=str,
        default="rl-eval",
        help="Seed prefix used for final eval games.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ReinforceConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        l2=args.l2,
        temperature=args.temperature,
        seed=args.seed,
        max_decisions_per_game=args.max_decisions_per_game,
    )

    model = load_behavior_cloning_checkpoint(args.checkpoint_in)
    eval_snapshots: List[Dict[str, object]] = []

    with BridgeClient() as client:
        env = MagnateBridgeEnv(client=client)
        summary = fine_tune_with_reinforce(
            env=env,
            model=model,
            config=config,
            seed_prefix=args.seed_prefix,
        )

        if args.eval_games > 0:
            eval_snapshots = _run_final_evals(
                env=env,
                model=model,
                eval_games=args.eval_games,
                eval_seed_prefix=args.eval_seed_prefix,
            )

    save_behavior_cloning_checkpoint(
        model=model,
        output_path=args.checkpoint_out,
        metadata={
            "sourceCheckpoint": str(args.checkpoint_in),
            "mode": "reinforce-self-play",
            "episodes": args.episodes,
            "learningRate": args.learning_rate,
            "l2": args.l2,
            "temperature": args.temperature,
            "seed": args.seed,
            "maxDecisionsPerGame": args.max_decisions_per_game,
            "seedPrefix": args.seed_prefix,
            "evalGames": args.eval_games,
        },
    )

    print(
        json.dumps(
            {
                "checkpointIn": str(args.checkpoint_in),
                "checkpointOut": str(args.checkpoint_out),
                "episodes": summary.episodes,
                "winners": summary.winners,
                "averageTurn": summary.average_turn,
                "averageDecisions": summary.average_decisions,
                "averageEntropy": summary.average_entropy,
                "evalSnapshots": eval_snapshots,
            },
            indent=2,
        )
    )
    return 0


def _run_final_evals(
    env: MagnateBridgeEnv,
    model: BehaviorCloningModel,
    eval_games: int,
    eval_seed_prefix: str,
) -> List[Dict[str, object]]:
    return [
        _to_snapshot(
            opponent="random",
            summary=evaluate_matchup(
                env=env,
                policy_player_a=BehaviorCloningPolicy(model=model, name="rl-finetuned"),
                policy_player_b=policy_from_name("random"),
                games=eval_games,
                seed_prefix=f"{eval_seed_prefix}-random",
            ),
        ),
        _to_snapshot(
            opponent="heuristic",
            summary=evaluate_matchup(
                env=env,
                policy_player_a=BehaviorCloningPolicy(model=model, name="rl-finetuned"),
                policy_player_b=policy_from_name("heuristic"),
                games=eval_games,
                seed_prefix=f"{eval_seed_prefix}-heuristic",
            ),
        ),
    ]


def _to_snapshot(opponent: str, summary: MatchSummary) -> Dict[str, object]:
    return {
        "opponent": opponent,
        "games": summary.games,
        "winners": dict(summary.winners),
        "winsByPolicy": dict(summary.wins_by_policy),
        "averageTurn": summary.average_turn,
    }


if __name__ == "__main__":
    raise SystemExit(main())
