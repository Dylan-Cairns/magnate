from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import evaluate_matchup
from trainer.policies import MctsConfig, SearchConfig, policy_from_name

DEFAULT_PROGRESS_EVERY_GAMES = 10


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
        help="Policy name for PlayerA (random|heuristic|search|mcts|ppo).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        default="random",
        help="Policy name for PlayerB (random|heuristic|search|mcts|ppo).",
    )
    parser.add_argument(
        "--player-a-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --player-a-policy=ppo.",
    )
    parser.add_argument(
        "--player-b-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --player-b-policy=ppo.",
    )
    parser.add_argument(
        "--search-worlds",
        type=int,
        default=6,
        help="Search policy world samples per decision.",
    )
    parser.add_argument(
        "--search-rollouts",
        type=int,
        default=1,
        help="Search policy rollouts per action per sampled world.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=14,
        help="Search policy rollout depth limit (decision steps after root action).",
    )
    parser.add_argument(
        "--search-max-root-actions",
        type=int,
        default=6,
        help="Search policy candidate root actions kept after heuristic pre-ranking.",
    )
    parser.add_argument(
        "--search-rollout-epsilon",
        type=float,
        default=0.08,
        help="Search rollout epsilon for random exploratory moves.",
    )
    parser.add_argument(
        "--mcts-worlds",
        type=int,
        default=6,
        help="MCTS determinized world samples per decision.",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=96,
        help="MCTS simulations per sampled world.",
    )
    parser.add_argument(
        "--mcts-depth",
        type=int,
        default=24,
        help="MCTS simulation depth limit (decision steps).",
    )
    parser.add_argument(
        "--mcts-max-root-actions",
        type=int,
        default=8,
        help="MCTS root actions kept after heuristic pre-ranking.",
    )
    parser.add_argument(
        "--mcts-c-puct",
        type=float,
        default=1.25,
        help="MCTS PUCT exploration coefficient.",
    )
    parser.add_argument(
        "--search-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for search priors/value/opponent model.",
    )
    parser.add_argument(
        "--mcts-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for MCTS priors/value.",
    )
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used by guidance checkpoints.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON artifact path. "
            "Default: artifacts/evals/<utc-timestamp>-<seed-prefix>-<player-a>-vs-<player-b>.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    search_config = SearchConfig(
        worlds=args.search_worlds,
        rollouts=args.search_rollouts,
        depth=args.search_depth,
        max_root_actions=args.search_max_root_actions,
        rollout_epsilon=args.search_rollout_epsilon,
    )
    mcts_config = MctsConfig(
        worlds=args.mcts_worlds,
        simulations=args.mcts_simulations,
        depth=args.mcts_depth,
        max_root_actions=args.mcts_max_root_actions,
        c_puct=args.mcts_c_puct,
    )
    policy_a = policy_from_name(
        args.player_a_policy,
        checkpoint_path=args.player_a_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=args.search_guidance_checkpoint,
        mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
        guidance_temperature=args.guidance_temperature,
    )
    policy_b = policy_from_name(
        args.player_b_policy,
        checkpoint_path=args.player_b_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=args.search_guidance_checkpoint,
        mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
        guidance_temperature=args.guidance_temperature,
    )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)

            def on_progress(
                completed_games: int,
                total_games: int,
                winners: dict,
                _wins_by_policy: dict,
            ) -> None:
                elapsed_minutes = (time.perf_counter() - started_at) / 60.0
                player_a_wins = int(winners.get("PlayerA", 0))
                win_rate = (
                    (player_a_wins / completed_games) if completed_games > 0 else 0.0
                )
                pct = (
                    (completed_games / total_games * 100.0)
                    if total_games > 0
                    else 100.0
                )
                print(
                    "[eval] progress "
                    f"games={completed_games}/{total_games} "
                    f"pct={pct:.1f}% "
                    f"playerAWinRate={win_rate:.3f} "
                    f"elapsedMin={elapsed_minutes:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            summary = evaluate_matchup(
                env=env,
                policy_player_a=policy_a,
                policy_player_b=policy_b,
                games=args.games,
                seed_prefix=args.seed_prefix,
                progress_every_games=DEFAULT_PROGRESS_EVERY_GAMES,
                on_progress=on_progress,
            )
    finally:
        policy_a.close()
        policy_b.close()

    output_path = args.out or _default_output_path(
        seed_prefix=args.seed_prefix,
        player_a_policy=args.player_a_policy,
        player_b_policy=args.player_b_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": args.games,
            "seedPrefix": args.seed_prefix,
            "playerAPolicy": args.player_a_policy,
            "playerBPolicy": args.player_b_policy,
            "playerACheckpoint": (
                str(args.player_a_checkpoint) if args.player_a_checkpoint else None
            ),
            "playerBCheckpoint": (
                str(args.player_b_checkpoint) if args.player_b_checkpoint else None
            ),
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
            "mcts": {
                "worlds": args.mcts_worlds,
                "simulations": args.mcts_simulations,
                "depth": args.mcts_depth,
                "maxRootActions": args.mcts_max_root_actions,
                "cPuct": args.mcts_c_puct,
            },
            "guidance": {
                "searchCheckpoint": (
                    str(args.search_guidance_checkpoint)
                    if args.search_guidance_checkpoint
                    else None
                ),
                "mctsCheckpoint": (
                    str(args.mcts_guidance_checkpoint)
                    if args.mcts_guidance_checkpoint
                    else None
                ),
                "temperature": args.guidance_temperature,
            },
        },
        "results": {
            "games": summary.games,
            "winners": summary.winners,
            "winsByPolicy": summary.wins_by_policy,
            "averageTurn": summary.average_turn,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "games": payload["results"]["games"],
                "winners": payload["results"]["winners"],
                "winsByPolicy": payload["results"]["winsByPolicy"],
                "averageTurn": payload["results"]["averageTurn"],
            },
            indent=2,
        )
    )
    return 0


def _default_output_path(
    seed_prefix: str, player_a_policy: str, player_b_policy: str
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed_prefix = _slug(seed_prefix)
    safe_player_a = _slug(player_a_policy)
    safe_player_b = _slug(player_b_policy)
    return (
        Path("artifacts/evals")
        / f"{stamp}-{safe_seed_prefix}-{safe_player_a}-vs-{safe_player_b}.json"
    )


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
