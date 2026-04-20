from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.policies import (
    SearchConfig,
    TDSearchPolicyConfig,
    TDValuePolicyConfig,
    policy_from_name,
)
from trainer.td import (
    collect_self_play_games,
    flatten_opponent_samples,
    flatten_value_transitions,
    write_opponent_samples_jsonl,
    write_value_transitions_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TD self-play replay artifacts (value transitions + opponent samples)."
    )
    parser.add_argument("--games", type=int, default=200, help="Number of self-play games.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="td-self-play",
        help="Seed prefix used to derive deterministic per-game seeds.",
    )
    parser.add_argument(
        "--player-a-policy",
        type=str,
        required=True,
        help="Policy for PlayerA (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        required=True,
        help="Policy for PlayerB (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument(
        "--transition-cache-limit",
        type=int,
        default=32,
        help="Exact-state transition cache size per policy instance (0 disables).",
    )
    parser.add_argument(
        "--legal-actions-cache-limit",
        type=int,
        default=32,
        help="Exact-state legal-actions cache size per policy instance (0 disables).",
    )
    parser.add_argument(
        "--observation-cache-limit",
        type=int,
        default=32,
        help="Exact-state observation cache size per policy instance (0 disables).",
    )
    parser.add_argument(
        "--td-value-checkpoint",
        type=Path,
        default=None,
        help="Path to TD value checkpoint used when a policy is td-value.",
    )
    parser.add_argument(
        "--td-worlds",
        type=int,
        default=8,
        help="Determinization world samples per decision for td-value policy.",
    )
    parser.add_argument(
        "--td-search-value-checkpoint",
        type=Path,
        default=None,
        help="Path to TD value checkpoint used when a policy is td-search.",
    )
    parser.add_argument(
        "--td-search-opponent-checkpoint",
        type=Path,
        default=None,
        help="TD opponent checkpoint used when a policy is td-search.",
    )
    parser.add_argument(
        "--td-search-opponent-temperature",
        type=float,
        default=1.0,
        help="Opponent policy temperature for td-search (lower is greedier).",
    )
    parser.add_argument(
        "--td-search-sample-opponent-actions",
        action="store_true",
        help="Sample opponent rollout actions from opponent model distribution in td-search.",
    )
    parser.add_argument(
        "--player-a-td-search-value-checkpoint",
        type=Path,
        default=None,
        help="Optional td-search value checkpoint override for PlayerA.",
    )
    parser.add_argument(
        "--player-a-td-search-opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional td-search opponent checkpoint override for PlayerA.",
    )
    parser.add_argument(
        "--player-b-td-search-value-checkpoint",
        type=Path,
        default=None,
        help="Optional td-search value checkpoint override for PlayerB.",
    )
    parser.add_argument(
        "--player-b-td-search-opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional td-search opponent checkpoint override for PlayerB.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/td_replay"),
        help="Output directory for replay artifacts.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="td-self-play",
        help="Run label used in default artifact names.",
    )
    parser.add_argument(
        "--value-out",
        type=Path,
        default=None,
        help="Optional explicit value-transition JSONL path.",
    )
    parser.add_argument(
        "--opponent-out",
        type=Path,
        default=None,
        help="Optional explicit opponent-sample JSONL path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional explicit summary JSON path.",
    )
    parser.add_argument(
        "--progress-every-games",
        type=int,
        default=25,
        help="Print progress every N completed games (0 disables).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime()
    _validate_policy_args(args)
    if args.games <= 0:
        raise SystemExit("--games must be > 0.")

    started_at = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_label = _slug(args.run_label)

    value_out = args.value_out or (args.out_dir / f"{stamp}-{safe_label}.value.jsonl")
    opponent_out = args.opponent_out or (args.out_dir / f"{stamp}-{safe_label}.opponent.jsonl")
    summary_out = args.summary_out or (args.out_dir / f"{stamp}-{safe_label}.summary.json")

    search_config = SearchConfig(
        worlds=args.search_worlds,
        rollouts=args.search_rollouts,
        depth=args.search_depth,
        max_root_actions=args.search_max_root_actions,
        rollout_epsilon=args.search_rollout_epsilon,
        transition_cache_limit=args.transition_cache_limit,
        legal_actions_cache_limit=args.legal_actions_cache_limit,
        observation_cache_limit=args.observation_cache_limit,
    )
    td_value_config = (
        TDValuePolicyConfig(
            checkpoint_path=args.td_value_checkpoint,
            worlds=args.td_worlds,
            transition_cache_limit=args.transition_cache_limit,
            legal_actions_cache_limit=args.legal_actions_cache_limit,
            observation_cache_limit=args.observation_cache_limit,
        )
        if args.td_value_checkpoint is not None
        else None
    )
    player_a_td_search_value_checkpoint = _effective_player_td_search_value_checkpoint(
        args=args, player="a"
    )
    player_a_td_search_opponent_checkpoint = _effective_player_td_search_opponent_checkpoint(
        args=args, player="a"
    )
    player_b_td_search_value_checkpoint = _effective_player_td_search_value_checkpoint(
        args=args, player="b"
    )
    player_b_td_search_opponent_checkpoint = _effective_player_td_search_opponent_checkpoint(
        args=args, player="b"
    )

    td_search_config_a = (
        TDSearchPolicyConfig(
            value_checkpoint_path=player_a_td_search_value_checkpoint,
            opponent_checkpoint_path=player_a_td_search_opponent_checkpoint,
            worlds=args.search_worlds,
            rollouts=args.search_rollouts,
            depth=args.search_depth,
            max_root_actions=args.search_max_root_actions,
            rollout_epsilon=args.search_rollout_epsilon,
            opponent_temperature=args.td_search_opponent_temperature,
            sample_opponent_actions=args.td_search_sample_opponent_actions,
            transition_cache_limit=args.transition_cache_limit,
            legal_actions_cache_limit=args.legal_actions_cache_limit,
            observation_cache_limit=args.observation_cache_limit,
        )
        if args.player_a_policy.strip().lower() == "td-search"
        else None
    )
    td_search_config_b = (
        TDSearchPolicyConfig(
            value_checkpoint_path=player_b_td_search_value_checkpoint,
            opponent_checkpoint_path=player_b_td_search_opponent_checkpoint,
            worlds=args.search_worlds,
            rollouts=args.search_rollouts,
            depth=args.search_depth,
            max_root_actions=args.search_max_root_actions,
            rollout_epsilon=args.search_rollout_epsilon,
            opponent_temperature=args.td_search_opponent_temperature,
            sample_opponent_actions=args.td_search_sample_opponent_actions,
            transition_cache_limit=args.transition_cache_limit,
            legal_actions_cache_limit=args.legal_actions_cache_limit,
            observation_cache_limit=args.observation_cache_limit,
        )
        if args.player_b_policy.strip().lower() == "td-search"
        else None
    )

    policy_a = policy_from_name(
        args.player_a_policy,
        search_config=search_config,
        td_value_config=td_value_config,
        td_search_config=td_search_config_a,
    )
    policy_b = policy_from_name(
        args.player_b_policy,
        search_config=search_config,
        td_value_config=td_value_config,
        td_search_config=td_search_config_b,
    )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)

            def on_progress(completed_games: int, total_games: int, winners: dict) -> None:
                elapsed_minutes = (time.perf_counter() - started_at) / 60.0
                pct = ((completed_games / total_games) * 100.0) if total_games > 0 else 100.0
                print(
                    "[td-self-play] progress "
                    f"games={completed_games}/{total_games} "
                    f"pct={pct:.1f}% "
                    f"winsA={int(winners.get('PlayerA', 0))} "
                    f"winsB={int(winners.get('PlayerB', 0))} "
                    f"draw={int(winners.get('Draw', 0))} "
                    f"elapsedMin={elapsed_minutes:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            episodes = collect_self_play_games(
                env=env,
                policy_player_a=policy_a,
                policy_player_b=policy_b,
                games=args.games,
                seed_prefix=args.seed_prefix,
                progress_every_games=args.progress_every_games,
                on_progress=on_progress if args.progress_every_games > 0 else None,
            )
        cache_stats = {
            "playerA": _policy_cache_stats_payload(policy_a),
            "playerB": _policy_cache_stats_payload(policy_b),
        }
        cache_stats["combined"] = _merge_cache_stats_payloads(
            [
                cache_stats["playerA"],
                cache_stats["playerB"],
            ]
        )
    finally:
        _close_unique(policy_a, policy_b)

    value_transitions = flatten_value_transitions(episodes)
    opponent_samples = flatten_opponent_samples(episodes)
    write_value_transitions_jsonl(value_transitions, value_out)
    write_opponent_samples_jsonl(opponent_samples, opponent_out)

    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    turn_total = 0
    for episode in episodes:
        winners[episode.winner] += 1
        turn_total += int(episode.turns)

    summary_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": args.games,
            "seedPrefix": args.seed_prefix,
            "playerAPolicy": args.player_a_policy,
            "playerBPolicy": args.player_b_policy,
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
            "cache": {
                "transitionLimit": args.transition_cache_limit,
                "legalActionsLimit": args.legal_actions_cache_limit,
                "observationLimit": args.observation_cache_limit,
            },
            "tdValue": {
                "checkpoint": str(args.td_value_checkpoint) if args.td_value_checkpoint else None,
                "worlds": args.td_worlds,
            },
            "tdSearch": {
                "valueCheckpoint": (
                    str(args.td_search_value_checkpoint)
                    if args.td_search_value_checkpoint
                    else None
                ),
                "opponentCheckpoint": (
                    str(args.td_search_opponent_checkpoint)
                    if args.td_search_opponent_checkpoint
                    else None
                ),
                "playerAValueCheckpoint": (
                    str(player_a_td_search_value_checkpoint)
                    if player_a_td_search_value_checkpoint is not None
                    else None
                ),
                "playerAOpponentCheckpoint": (
                    str(player_a_td_search_opponent_checkpoint)
                    if player_a_td_search_opponent_checkpoint is not None
                    else None
                ),
                "playerBValueCheckpoint": (
                    str(player_b_td_search_value_checkpoint)
                    if player_b_td_search_value_checkpoint is not None
                    else None
                ),
                "playerBOpponentCheckpoint": (
                    str(player_b_td_search_opponent_checkpoint)
                    if player_b_td_search_opponent_checkpoint is not None
                    else None
                ),
                "opponentTemperature": args.td_search_opponent_temperature,
                "sampleOpponentActions": args.td_search_sample_opponent_actions,
            },
        },
        "results": {
            "games": len(episodes),
            "winners": winners,
            "averageTurn": (turn_total / float(len(episodes))) if episodes else 0.0,
            "valueTransitions": len(value_transitions),
            "opponentSamples": len(opponent_samples),
            "cacheStats": cache_stats,
        },
        "artifacts": {
            "valueTransitions": str(value_out),
            "opponentSamples": str(opponent_out),
            "summary": str(summary_out),
        },
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "valueTransitionsArtifact": str(value_out),
                "opponentSamplesArtifact": str(opponent_out),
                "summaryArtifact": str(summary_out),
                "games": summary_payload["results"]["games"],
                "winners": summary_payload["results"]["winners"],
                "averageTurn": summary_payload["results"]["averageTurn"],
                "valueTransitions": summary_payload["results"]["valueTransitions"],
                "opponentSamples": summary_payload["results"]["opponentSamples"],
                "cacheStats": summary_payload["results"]["cacheStats"],
            },
            indent=2,
        )
    )
    return 0


def _close_unique(*policies) -> None:
    seen: set[int] = set()
    for policy in policies:
        if policy is None:
            continue
        key = id(policy)
        if key in seen:
            continue
        seen.add(key)
        policy.close()


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


def _require_supported_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.12+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")


def _validate_policy_args(args: argparse.Namespace) -> None:
    if getattr(args, "transition_cache_limit", 0) < 0:
        raise SystemExit("--transition-cache-limit must be >= 0.")
    if getattr(args, "legal_actions_cache_limit", 0) < 0:
        raise SystemExit("--legal-actions-cache-limit must be >= 0.")
    if getattr(args, "observation_cache_limit", 0) < 0:
        raise SystemExit("--observation-cache-limit must be >= 0.")
    player_a_policy = args.player_a_policy.strip().lower()
    player_b_policy = args.player_b_policy.strip().lower()
    policies = {player_a_policy, player_b_policy}
    if "td-value" in policies and args.td_value_checkpoint is None:
        raise SystemExit("--td-value-checkpoint is required when using td-value policy.")
    if player_a_policy == "td-search":
        if _effective_player_td_search_value_checkpoint(args=args, player="a") is None:
            raise SystemExit(
                "PlayerA td-search requires --player-a-td-search-value-checkpoint "
                "or --td-search-value-checkpoint."
            )
        if _effective_player_td_search_opponent_checkpoint(args=args, player="a") is None:
            raise SystemExit(
                "PlayerA td-search requires --player-a-td-search-opponent-checkpoint "
                "or --td-search-opponent-checkpoint."
            )
    if player_b_policy == "td-search":
        if _effective_player_td_search_value_checkpoint(args=args, player="b") is None:
            raise SystemExit(
                "PlayerB td-search requires --player-b-td-search-value-checkpoint "
                "or --td-search-value-checkpoint."
            )
        if _effective_player_td_search_opponent_checkpoint(args=args, player="b") is None:
            raise SystemExit(
                "PlayerB td-search requires --player-b-td-search-opponent-checkpoint "
                "or --td-search-opponent-checkpoint."
            )


def _effective_player_td_search_value_checkpoint(
    *, args: argparse.Namespace, player: str
) -> Path | None:
    if player == "a":
        return args.player_a_td_search_value_checkpoint or args.td_search_value_checkpoint
    if player == "b":
        return args.player_b_td_search_value_checkpoint or args.td_search_value_checkpoint
    raise SystemExit(f"Unknown player key: {player!r}")


def _effective_player_td_search_opponent_checkpoint(
    *, args: argparse.Namespace, player: str
) -> Path | None:
    if player == "a":
        return args.player_a_td_search_opponent_checkpoint or args.td_search_opponent_checkpoint
    if player == "b":
        return args.player_b_td_search_opponent_checkpoint or args.td_search_opponent_checkpoint
    raise SystemExit(f"Unknown player key: {player!r}")


def _policy_cache_stats_payload(policy: object) -> dict[str, object] | None:
    forward_model = getattr(policy, "_forward_model", None)
    if forward_model is None:
        return None
    cache_stats = getattr(forward_model, "cache_stats", None)
    if not callable(cache_stats):
        return None
    raw_stats = cache_stats()
    return {
        "transition": _cache_metric_payload(
            hits=int(getattr(raw_stats, "transition_hits")),
            misses=int(getattr(raw_stats, "transition_misses")),
            entries=int(getattr(raw_stats, "transition_entries")),
        ),
        "legalActions": _cache_metric_payload(
            hits=int(getattr(raw_stats, "legal_actions_hits")),
            misses=int(getattr(raw_stats, "legal_actions_misses")),
            entries=int(getattr(raw_stats, "legal_actions_entries")),
        ),
        "observation": _cache_metric_payload(
            hits=int(getattr(raw_stats, "observation_hits")),
            misses=int(getattr(raw_stats, "observation_misses")),
            entries=int(getattr(raw_stats, "observation_entries")),
        ),
    }


def _merge_cache_stats_payloads(
    payloads: list[dict[str, object] | None],
) -> dict[str, object] | None:
    valid_payloads = [payload for payload in payloads if payload is not None]
    if not valid_payloads:
        return None
    return {
        "transition": _merge_cache_metric_payloads(valid_payloads, "transition"),
        "legalActions": _merge_cache_metric_payloads(valid_payloads, "legalActions"),
        "observation": _merge_cache_metric_payloads(valid_payloads, "observation"),
    }


def _merge_cache_metric_payloads(
    payloads: list[dict[str, object]],
    metric_key: str,
) -> dict[str, object]:
    hits = 0
    misses = 0
    entries = 0
    for payload in payloads:
        metric = payload.get(metric_key)
        if not isinstance(metric, Mapping):
            continue
        hits += int(metric["hits"])
        misses += int(metric["misses"])
        entries += int(metric["entries"])
    return _cache_metric_payload(hits=hits, misses=misses, entries=entries)


def _cache_metric_payload(*, hits: int, misses: int, entries: int) -> dict[str, object]:
    requests = hits + misses
    hit_rate = (hits / float(requests)) if requests > 0 else None
    return {
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "hitRate": hit_rate,
        "entries": entries,
    }


if __name__ == "__main__":
    raise SystemExit(main())
