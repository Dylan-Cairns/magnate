from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from trainer.bridge_client import BridgeClient
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import MatchSummary, evaluate_matchup
from trainer.policies import TorchPpoPolicy, policy_from_name
from trainer.ppo_model import (
    CandidateActorCritic,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)
from trainer.ppo_training import PpoConfig, PpoUpdateSummary, train_ppo_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO policy against the TS Magnate bridge environment."
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("artifacts/ppo_checkpoint.pt"),
        help="Output PPO checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        default=None,
        help="Optional checkpoint to continue PPO training from.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Restore optimizer state from checkpoint-in when present.",
    )
    parser.add_argument("--episodes", type=int, default=1024, help="Total training episodes.")
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=32,
        help="Episodes collected per PPO optimization update.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="PPO model hidden layer width.")
    parser.add_argument("--seed-prefix", type=str, default="ppo", help="Episode seed prefix.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio.")
    parser.add_argument("--train-epochs", type=int, default=4, help="PPO epochs per update.")
    parser.add_argument("--minibatch-size", type=int, default=128, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Policy sampling temperature.")
    parser.add_argument(
        "--max-decisions-per-game",
        type=int,
        default=2000,
        help="Safety guard on learner decision count per game.",
    )
    parser.add_argument("--self-play-weight", type=float, default=0.5, help="Self-play opponent weight.")
    parser.add_argument("--heuristic-weight", type=float, default=0.3, help="Heuristic opponent weight.")
    parser.add_argument("--random-weight", type=float, default=0.2, help="Random opponent weight.")

    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="If >0, evaluate checkpoint candidates against random/heuristic.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=5,
        help="Evaluate and consider selection every N PPO updates.",
    )
    parser.add_argument(
        "--eval-seed-prefix",
        type=str,
        default="ppo-eval",
        help="Seed prefix used for evaluation games.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=("fixed-holdout", "rolling"),
        default="fixed-holdout",
        help="Evaluation seed strategy across checkpoint candidates.",
    )
    parser.add_argument(
        "--selection-heuristic-weight",
        type=float,
        default=0.7,
        help="Selection-score weight for heuristic matchup win rate.",
    )
    parser.add_argument(
        "--selection-random-weight",
        type=float,
        default=0.3,
        help="Selection-score weight for random matchup win rate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.checkpoint_in is not None:
        model, loaded_payload = load_ppo_checkpoint(args.checkpoint_in)
        optimizer_state = loaded_payload.get("optimizerStateDict")
    else:
        model = CandidateActorCritic(
            observation_dim=OBSERVATION_DIM,
            action_feature_dim=ACTION_FEATURE_DIM,
            hidden_dim=args.hidden_dim,
        )
        optimizer_state = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.resume_optimizer and isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)

    best_model_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    best_update = 0
    eval_snapshots: List[Dict[str, object]] = []

    selection_weight_total = args.selection_heuristic_weight + args.selection_random_weight
    if selection_weight_total <= 0:
        raise SystemExit("selection weights must sum to > 0.")

    config = PpoConfig(
        total_episodes=args.episodes,
        episodes_per_update=args.episodes_per_update,
        seed=args.seed,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        train_epochs=args.train_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        temperature=args.temperature,
        max_decisions_per_game=args.max_decisions_per_game,
        self_play_weight=args.self_play_weight,
        heuristic_weight=args.heuristic_weight,
        random_weight=args.random_weight,
    )

    with BridgeClient() as client:
        env = MagnateBridgeEnv(client=client)

        def maybe_evaluate(update_number: int) -> None:
            nonlocal best_model_state, best_score, best_update
            if args.eval_games <= 0:
                return

            eval_suffix = (
                "-holdout"
                if args.eval_mode == "fixed-holdout"
                else f"-u{update_number}"
            )

            ppo_policy = TorchPpoPolicy(model=model, name="ppo-training")
            random_summary = evaluate_matchup(
                env=env,
                policy_player_a=ppo_policy,
                policy_player_b=policy_from_name("random"),
                games=args.eval_games,
                seed_prefix=f"{args.eval_seed_prefix}-random{eval_suffix}",
            )
            heuristic_summary = evaluate_matchup(
                env=env,
                policy_player_a=ppo_policy,
                policy_player_b=policy_from_name("heuristic"),
                games=args.eval_games,
                seed_prefix=f"{args.eval_seed_prefix}-heuristic{eval_suffix}",
            )

            score = _selection_score(
                random_summary=random_summary,
                heuristic_summary=heuristic_summary,
                heuristic_weight=args.selection_heuristic_weight,
                random_weight=args.selection_random_weight,
            )
            snapshot = {
                "update": update_number,
                "selectionScore": score,
                "random": _to_snapshot("random", random_summary),
                "heuristic": _to_snapshot("heuristic", heuristic_summary),
            }
            eval_snapshots.append(snapshot)
            if score > best_score:
                best_score = score
                best_model_state = copy.deepcopy(model.state_dict())
                best_update = update_number

        def on_update_end(update_index: int, _summary: PpoUpdateSummary) -> None:
            if args.eval_games <= 0 or args.eval_every_updates <= 0:
                return
            update_number = update_index + 1
            if update_number % args.eval_every_updates == 0:
                maybe_evaluate(update_number)

        optimizer, training_summary = train_ppo_policy(
            env=env,
            model=model,
            config=config,
            seed_prefix=args.seed_prefix,
            start_episode_index=0,
            optimizer=optimizer,
            on_update_end=on_update_end,
        )

        if args.eval_games > 0:
            final_update = training_summary.updates
            if args.eval_every_updates <= 0 or final_update % args.eval_every_updates != 0:
                maybe_evaluate(final_update)

    if args.eval_games > 0 and best_update > 0:
        model.load_state_dict(best_model_state)
        selected_update = best_update
    else:
        selected_update = training_summary.updates

    save_ppo_checkpoint(
        model=model,
        output_path=args.checkpoint_out,
        metadata={
            "sourceCheckpoint": str(args.checkpoint_in) if args.checkpoint_in is not None else None,
            "mode": "ppo",
            "totalEpisodes": args.episodes,
            "episodesPerUpdate": args.episodes_per_update,
            "seedPrefix": args.seed_prefix,
            "seed": args.seed,
            "evalGames": args.eval_games,
            "evalEveryUpdates": args.eval_every_updates,
            "evalMode": args.eval_mode,
            "selectedUpdate": selected_update,
            "selectionScore": best_score if args.eval_games > 0 else None,
            "selectionWeights": {
                "heuristic": args.selection_heuristic_weight,
                "random": args.selection_random_weight,
            },
            "opponentMix": {
                "self": args.self_play_weight,
                "heuristic": args.heuristic_weight,
                "random": args.random_weight,
            },
        },
        optimizer_state_dict=optimizer.state_dict(),
    )

    print(
        json.dumps(
            {
                "checkpointIn": str(args.checkpoint_in) if args.checkpoint_in is not None else None,
                "checkpointOut": str(args.checkpoint_out),
                "episodes": training_summary.total_episodes,
                "updates": training_summary.updates,
                "winners": training_summary.winners,
                "opponentCounts": training_summary.opponent_counts,
                "averageTurn": training_summary.average_turn,
                "averageTransitions": training_summary.average_transitions,
                "selectedUpdate": selected_update,
                "selectionScore": best_score if args.eval_games > 0 else None,
                "evalSnapshots": eval_snapshots,
            },
            indent=2,
        )
    )
    return 0


def _selection_score(
    random_summary: MatchSummary,
    heuristic_summary: MatchSummary,
    heuristic_weight: float,
    random_weight: float,
) -> float:
    weight_total = heuristic_weight + random_weight
    if weight_total <= 0:
        raise ValueError("selection weights must sum to > 0.")
    heuristic_share = heuristic_weight / weight_total
    random_share = random_weight / weight_total
    return (
        (heuristic_share * _player_a_win_rate(heuristic_summary))
        + (random_share * _player_a_win_rate(random_summary))
    )


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
