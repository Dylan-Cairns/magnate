from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from trainer.behavior_cloning import (
    BehaviorCloningModel,
    load_behavior_cloning_checkpoint,
    save_behavior_cloning_checkpoint,
)
from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import MatchSummary, evaluate_matchup
from trainer.policies import BehaviorCloningPolicy, policy_from_name
from trainer.reinforcement import ReinforceConfig, ReinforceEpisodeResult, fine_tune_with_reinforce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a BC checkpoint with stabilized RL (mixed opponents + BC anchor)."
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
        help="Output fine-tuned checkpoint path (best checkpoint when eval is enabled).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of RL episodes.",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="rl",
        help="Seed prefix used for deterministic episodes.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.002, help="REINFORCE learning rate.")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 regularization.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for learner actions.",
    )
    parser.add_argument(
        "--bc-anchor-coeff",
        type=float,
        default=0.02,
        help="Strength of weight regularization toward the source BC checkpoint.",
    )
    parser.add_argument(
        "--self-play-weight",
        type=float,
        default=0.4,
        help="Training opponent mix weight for self-play snapshot.",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=0.4,
        help="Training opponent mix weight for heuristic policy.",
    )
    parser.add_argument(
        "--random-weight",
        type=float,
        default=0.2,
        help="Training opponent mix weight for random policy.",
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
        default=100,
        help="If >0, run eval checkpoints against random and heuristic.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate and consider checkpoint selection every N episodes.",
    )
    parser.add_argument(
        "--eval-seed-prefix",
        type=str,
        default="rl-eval",
        help="Seed prefix used for evaluation games.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=("fixed-holdout", "rolling"),
        default="fixed-holdout",
        help=(
            "Evaluation seed strategy. "
            "'fixed-holdout' keeps the same seed set across checkpoints; "
            "'rolling' uses episode-specific seeds."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_model = load_behavior_cloning_checkpoint(args.checkpoint_in)
    model = source_model.copy()
    best_model = model.copy()
    best_score = -math.inf
    best_episode = 0

    eval_snapshots: List[Dict[str, object]] = []

    config = ReinforceConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        l2=args.l2,
        temperature=args.temperature,
        seed=args.seed,
        max_decisions_per_game=args.max_decisions_per_game,
        self_play_weight=args.self_play_weight,
        heuristic_weight=args.heuristic_weight,
        random_weight=args.random_weight,
        bc_anchor_coeff=args.bc_anchor_coeff,
    )

    with BridgeClient() as client:
        env = MagnateBridgeEnv(client=client)

        def maybe_evaluate(episode_number: int) -> None:
            nonlocal best_model, best_score, best_episode
            if args.eval_games <= 0:
                return
            eval_suffix = (
                "-holdout"
                if args.eval_mode == "fixed-holdout"
                else f"-{episode_number}"
            )
            random_summary = evaluate_matchup(
                env=env,
                policy_player_a=BehaviorCloningPolicy(model=model, name="rl-finetuned"),
                policy_player_b=policy_from_name("random"),
                games=args.eval_games,
                seed_prefix=f"{args.eval_seed_prefix}-random{eval_suffix}",
            )
            heuristic_summary = evaluate_matchup(
                env=env,
                policy_player_a=BehaviorCloningPolicy(model=model, name="rl-finetuned"),
                policy_player_b=policy_from_name("heuristic"),
                games=args.eval_games,
                seed_prefix=f"{args.eval_seed_prefix}-heuristic{eval_suffix}",
            )
            score = _selection_score(random_summary, heuristic_summary)
            snapshot = {
                "episode": episode_number,
                "selectionScore": score,
                "random": _to_snapshot("random", random_summary),
                "heuristic": _to_snapshot("heuristic", heuristic_summary),
            }
            eval_snapshots.append(snapshot)
            if score > best_score:
                best_score = score
                best_model = model.copy()
                best_episode = episode_number

        def on_episode_end(episode_index: int, _result: ReinforceEpisodeResult) -> None:
            if args.eval_games <= 0 or args.eval_every <= 0:
                return
            episode_number = episode_index + 1
            if episode_number % args.eval_every == 0:
                maybe_evaluate(episode_number)

        summary = fine_tune_with_reinforce(
            env=env,
            model=model,
            config=config,
            seed_prefix=args.seed_prefix,
            anchor_model=source_model,
            start_episode_index=0,
            on_episode_end=on_episode_end,
        )

        if args.eval_games > 0:
            final_episode = args.episodes
            if args.eval_every <= 0 or final_episode % args.eval_every != 0:
                maybe_evaluate(final_episode)

    selected_model = best_model if args.eval_games > 0 else model
    selected_episode = best_episode if args.eval_games > 0 else args.episodes

    save_behavior_cloning_checkpoint(
        model=selected_model,
        output_path=args.checkpoint_out,
        metadata={
            "sourceCheckpoint": str(args.checkpoint_in),
            "mode": "reinforce-mixed-opponents",
            "episodes": args.episodes,
            "learningRate": args.learning_rate,
            "l2": args.l2,
            "temperature": args.temperature,
            "bcAnchorCoeff": args.bc_anchor_coeff,
            "seed": args.seed,
            "maxDecisionsPerGame": args.max_decisions_per_game,
            "seedPrefix": args.seed_prefix,
            "opponentMix": {
                "self": args.self_play_weight,
                "heuristic": args.heuristic_weight,
                "random": args.random_weight,
            },
            "evalGames": args.eval_games,
            "evalEvery": args.eval_every,
            "evalMode": args.eval_mode,
            "selectedEpisode": selected_episode,
            "selectionScore": best_score if args.eval_games > 0 else None,
        },
    )

    print(
        json.dumps(
            {
                "checkpointIn": str(args.checkpoint_in),
                "checkpointOut": str(args.checkpoint_out),
                "episodes": summary.episodes,
                "winners": summary.winners,
                "opponentCounts": summary.opponent_counts,
                "averageTurn": summary.average_turn,
                "averageDecisions": summary.average_decisions,
                "averageEntropy": summary.average_entropy,
                "selectedEpisode": selected_episode,
                "selectionScore": best_score if args.eval_games > 0 else None,
                "evalSnapshots": eval_snapshots,
            },
            indent=2,
        )
    )
    return 0


def _selection_score(random_summary: MatchSummary, heuristic_summary: MatchSummary) -> float:
    random_win_rate = _player_a_win_rate(random_summary)
    heuristic_win_rate = _player_a_win_rate(heuristic_summary)
    return (0.7 * heuristic_win_rate) + (0.3 * random_win_rate)


def _player_a_win_rate(summary: MatchSummary) -> float:
    if summary.games <= 0:
        return 0.0
    return summary.winners["PlayerA"] / float(summary.games)


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
