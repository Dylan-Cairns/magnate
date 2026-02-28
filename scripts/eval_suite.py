from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.eval_suite import evaluate_side_swapped
from trainer.policies import MctsConfig, SearchConfig, policy_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical side-swapped paired-seed evaluation and report "
            "win rate, Wilson CI, and side-gap."
        )
    )
    parser.add_argument(
        "--games-per-side",
        type=int,
        default=100,
        help="Games with candidate on each seat (total games = 2x).",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="eval-suite",
        help="Shared seed prefix used for both side-swapped legs.",
    )
    parser.add_argument(
        "--candidate-policy",
        type=str,
        default="search",
        help="Candidate policy (random|heuristic|search|mcts|bc|ppo).",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy (random|heuristic|search|mcts|bc|ppo).",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --candidate-policy=bc or ppo.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --opponent-policy=bc or ppo.",
    )
    parser.add_argument("--search-worlds", type=int, default=8)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=16)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.12)
    parser.add_argument("--mcts-worlds", type=int, default=6)
    parser.add_argument("--mcts-simulations", type=int, default=96)
    parser.add_argument("--mcts-depth", type=int, default=24)
    parser.add_argument("--mcts-max-root-actions", type=int, default=8)
    parser.add_argument("--mcts-c-puct", type=float, default=1.25)
    parser.add_argument(
        "--search-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for search.",
    )
    parser.add_argument(
        "--mcts-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for MCTS.",
    )
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for guidance checkpoints.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON artifact path. "
            "Default: artifacts/evals/<utc>-<seed-prefix>-suite-<candidate>-vs-<opponent>.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    candidate = policy_from_name(
        args.candidate_policy,
        checkpoint_path=args.candidate_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=args.search_guidance_checkpoint,
        mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
        guidance_temperature=args.guidance_temperature,
    )
    opponent = policy_from_name(
        args.opponent_policy,
        checkpoint_path=args.opponent_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=args.search_guidance_checkpoint,
        mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
        guidance_temperature=args.guidance_temperature,
    )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)
            summary = evaluate_side_swapped(
                env=env,
                candidate_policy=candidate,
                opponent_policy=opponent,
                games_per_side=args.games_per_side,
                seed_prefix=args.seed_prefix,
            )
    finally:
        candidate.close()
        opponent.close()

    output_path = args.out or _default_output_path(
        seed_prefix=args.seed_prefix,
        candidate_policy=args.candidate_policy,
        opponent_policy=args.opponent_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gamesPerSide": args.games_per_side,
            "seedPrefix": args.seed_prefix,
            "candidatePolicy": args.candidate_policy,
            "opponentPolicy": args.opponent_policy,
            "candidateCheckpoint": (
                str(args.candidate_checkpoint) if args.candidate_checkpoint else None
            ),
            "opponentCheckpoint": (
                str(args.opponent_checkpoint) if args.opponent_checkpoint else None
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
        "results": summary.to_json(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "candidateWinRate": payload["results"]["candidateWinRate"],
                "candidateWinRateCi95": payload["results"]["candidateWinRateCi95"],
                "sideGap": payload["results"]["sideGap"],
                "candidateWins": payload["results"]["candidateWins"],
                "opponentWins": payload["results"]["opponentWins"],
                "draws": payload["results"]["draws"],
                "totalGames": payload["results"]["totalGames"],
            },
            indent=2,
        )
    )
    return 0


def _default_output_path(
    *,
    seed_prefix: str,
    candidate_policy: str,
    opponent_policy: str,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed_prefix = _slug(seed_prefix)
    safe_candidate = _slug(candidate_policy)
    safe_opponent = _slug(opponent_policy)
    return (
        Path("artifacts/evals")
        / f"{stamp}-{safe_seed_prefix}-suite-{safe_candidate}-vs-{safe_opponent}.json"
    )


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
