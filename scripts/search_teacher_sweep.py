from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SearchPreset:
    name: str
    worlds: int
    rollouts: int
    depth: int
    max_root_actions: int
    rollout_epsilon: float


@dataclass(frozen=True)
class LegResult:
    artifact: Path
    games: int
    winners: Dict[str, int]
    search_wins: int
    search_win_rate: float


@dataclass(frozen=True)
class SweepRow:
    preset: SearchPreset
    search_as_player_a: LegResult
    search_as_player_b: LegResult
    combined_games: int
    combined_search_wins: int
    combined_draws: int
    combined_search_win_rate: float
    side_gap: float


PRESETS: Dict[str, SearchPreset] = {
    "t1": SearchPreset(
        name="t1",
        worlds=2,
        rollouts=1,
        depth=8,
        max_root_actions=4,
        rollout_epsilon=0.12,
    ),
    "t2": SearchPreset(
        name="t2",
        worlds=4,
        rollouts=1,
        depth=12,
        max_root_actions=6,
        rollout_epsilon=0.10,
    ),
    "t3": SearchPreset(
        name="t3",
        worlds=6,
        rollouts=1,
        depth=14,
        max_root_actions=6,
        rollout_epsilon=0.08,
    ),
    "t4": SearchPreset(
        name="t4",
        worlds=8,
        rollouts=1,
        depth=16,
        max_root_actions=8,
        rollout_epsilon=0.08,
    ),
    "t5": SearchPreset(
        name="t5",
        worlds=8,
        rollouts=2,
        depth=16,
        max_root_actions=8,
        rollout_epsilon=0.06,
    ),
    "t6": SearchPreset(
        name="t6",
        worlds=10,
        rollouts=1,
        depth=18,
        max_root_actions=10,
        rollout_epsilon=0.06,
    ),
    "t7": SearchPreset(
        name="t7",
        worlds=8,
        rollouts=2,
        depth=14,
        max_root_actions=8,
        rollout_epsilon=0.04,
    ),
    "t8": SearchPreset(
        name="t8",
        worlds=10,
        rollouts=2,
        depth=14,
        max_root_actions=10,
        rollout_epsilon=0.02,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run side-swapped rollout-search teacher sweeps unattended and produce "
            "ranked summary artifacts."
        )
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(".venv/Scripts/python.exe"),
        help="Python executable used for subprocess eval commands.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games per side for each preset (total per preset = 2x games).",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="search-teacher-sweep",
        help="Run label used for artifact names and seed prefixes.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["t2", "t3", "t4", "t5", "t6"],
        help=f"Preset names to run. Available: {', '.join(sorted(PRESETS.keys()))}.",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy used against search (random|heuristic|search|mcts|bc|ppo).",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for opponent policy when using bc/ppo.",
    )
    parser.add_argument(
        "--search-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional guidance checkpoint injected into search during eval.",
    )
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Guidance softmax temperature (used only when guidance checkpoint is set).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root artifact directory.",
    )
    parser.add_argument(
        "--run-manifest-out",
        type=Path,
        default=None,
        help="Optional explicit output path for sweep manifest JSON.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional explicit output path for markdown ranking summary.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print built-in preset definitions and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running evaluations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_presets:
        _print_presets()
        return 0

    presets = _resolve_presets(args.presets)

    started_at = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_label = _slug(args.run_label)
    run_id = f"{stamp}-{safe_label}"

    artifact_root = args.artifact_dir
    eval_dir = artifact_root / "evals"
    sweep_dir = artifact_root / "sweeps"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.run_manifest_out or (sweep_dir / f"{run_id}-manifest.json")
    summary_path = args.summary_out or (sweep_dir / f"{run_id}-summary.md")

    planned_commands: List[Dict[str, object]] = []
    rows: List[SweepRow] = []

    for index, preset in enumerate(presets, start=1):
        search_a_out = eval_dir / f"{run_id}-{preset.name}-searchA.json"
        search_b_out = eval_dir / f"{run_id}-{preset.name}-searchB.json"

        cmd_search_a = _build_eval_command(
            python_bin=str(args.python_bin),
            games=args.games,
            seed_prefix=f"{safe_label}-{preset.name}-search-a",
            search_as_player_a=True,
            opponent_policy=args.opponent_policy,
            opponent_checkpoint=args.opponent_checkpoint,
            search_guidance_checkpoint=args.search_guidance_checkpoint,
            guidance_temperature=args.guidance_temperature,
            preset=preset,
            out_path=search_a_out,
        )
        cmd_search_b = _build_eval_command(
            python_bin=str(args.python_bin),
            games=args.games,
            seed_prefix=f"{safe_label}-{preset.name}-search-b",
            search_as_player_a=False,
            opponent_policy=args.opponent_policy,
            opponent_checkpoint=args.opponent_checkpoint,
            search_guidance_checkpoint=args.search_guidance_checkpoint,
            guidance_temperature=args.guidance_temperature,
            preset=preset,
            out_path=search_b_out,
        )

        planned_commands.append(
            {
                "preset": preset.name,
                "searchAsPlayerA": cmd_search_a,
                "searchAsPlayerB": cmd_search_b,
                "artifacts": {
                    "searchAsPlayerA": str(search_a_out),
                    "searchAsPlayerB": str(search_b_out),
                },
            }
        )

        print(
            f"[sweep] preset {index}/{len(presets)} {preset.name} "
            f"worlds={preset.worlds} rollouts={preset.rollouts} depth={preset.depth} "
            f"maxRoot={preset.max_root_actions} eps={preset.rollout_epsilon:.2f}"
        )

        if args.dry_run:
            print(f"[sweep] dry-run cmd search-as-A: {_join_command(cmd_search_a)}")
            print(f"[sweep] dry-run cmd search-as-B: {_join_command(cmd_search_b)}")
            continue

        result_a = _run_step(
            name=f"{preset.name}:search-as-A",
            command=cmd_search_a,
        )
        if result_a != 0:
            _write_json(
                manifest_path,
                _manifest_payload(
                    run_id=run_id,
                    args=args,
                    presets=presets,
                    started_at_seconds=started_at,
                    status="failed",
                    rows=rows,
                    planned_commands=planned_commands,
                    manifest_path=manifest_path,
                    summary_path=summary_path,
                ),
            )
            return result_a

        result_b = _run_step(
            name=f"{preset.name}:search-as-B",
            command=cmd_search_b,
        )
        if result_b != 0:
            _write_json(
                manifest_path,
                _manifest_payload(
                    run_id=run_id,
                    args=args,
                    presets=presets,
                    started_at_seconds=started_at,
                    status="failed",
                    rows=rows,
                    planned_commands=planned_commands,
                    manifest_path=manifest_path,
                    summary_path=summary_path,
                ),
            )
            return result_b

        leg_a = _read_leg_result(search_a_out, search_player="PlayerA")
        leg_b = _read_leg_result(search_b_out, search_player="PlayerB")
        row = _build_row(
            preset=preset, search_as_player_a=leg_a, search_as_player_b=leg_b
        )
        rows.append(row)

        print(
            "[sweep] finished preset "
            f"{preset.name} combinedWinRate={row.combined_search_win_rate:.3f} "
            f"sideGap={row.side_gap:.3f} "
            f"searchWins={row.combined_search_wins}/{row.combined_games}"
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (-row.combined_search_win_rate, row.side_gap, row.preset.name),
    )

    status = "dry-run" if args.dry_run else "ok"
    payload = _manifest_payload(
        run_id=run_id,
        args=args,
        presets=presets,
        started_at_seconds=started_at,
        status=status,
        rows=ranked_rows,
        planned_commands=planned_commands,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
    _write_json(manifest_path, payload)

    if not args.dry_run:
        _write_summary_markdown(summary_path, run_id=run_id, rows=ranked_rows)
        print(f"[sweep] summary: {summary_path}")
    print(f"[sweep] manifest: {manifest_path}")

    print(
        json.dumps(
            {
                "status": status,
                "runId": run_id,
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "presets": [preset.name for preset in presets],
            },
            indent=2,
        )
    )
    return 0


def _resolve_presets(names: List[str]) -> List[SearchPreset]:
    resolved: List[SearchPreset] = []
    seen = set()
    for name in names:
        key = name.strip().lower()
        if key in seen:
            raise SystemExit(f"Duplicate preset name: {name}")
        preset = PRESETS.get(key)
        if preset is None:
            raise SystemExit(
                f"Unknown preset '{name}'. Available: {', '.join(sorted(PRESETS.keys()))}"
            )
        seen.add(key)
        resolved.append(preset)
    if not resolved:
        raise SystemExit("No presets resolved.")
    return resolved


def _build_eval_command(
    *,
    python_bin: str,
    games: int,
    seed_prefix: str,
    search_as_player_a: bool,
    opponent_policy: str,
    opponent_checkpoint: Path | None,
    search_guidance_checkpoint: Path | None,
    guidance_temperature: float,
    preset: SearchPreset,
    out_path: Path,
) -> List[str]:
    player_a_policy = "search" if search_as_player_a else opponent_policy
    player_b_policy = opponent_policy if search_as_player_a else "search"

    command = [
        python_bin,
        "-m",
        "scripts.eval",
        "--games",
        str(games),
        "--seed-prefix",
        seed_prefix,
        "--player-a-policy",
        player_a_policy,
        "--player-b-policy",
        player_b_policy,
        "--search-worlds",
        str(preset.worlds),
        "--search-rollouts",
        str(preset.rollouts),
        "--search-depth",
        str(preset.depth),
        "--search-max-root-actions",
        str(preset.max_root_actions),
        "--search-rollout-epsilon",
        str(preset.rollout_epsilon),
        "--out",
        str(out_path),
    ]
    if opponent_checkpoint is not None:
        checkpoint_arg = (
            "--player-b-checkpoint" if search_as_player_a else "--player-a-checkpoint"
        )
        command.extend([checkpoint_arg, str(opponent_checkpoint)])
    if search_guidance_checkpoint is not None:
        command.extend(
            [
                "--search-guidance-checkpoint",
                str(search_guidance_checkpoint),
                "--guidance-temperature",
                str(guidance_temperature),
            ]
        )
    return command


def _run_step(*, name: str, command: List[str]) -> int:
    print(f"[sweep] step {name}: {_join_command(command)}")
    started = time.perf_counter()
    completed = subprocess.run(command)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        print(
            f"[sweep] failed step={name} returnCode={completed.returncode} "
            f"elapsedMin={elapsed / 60.0:.1f}"
        )
    else:
        print(f"[sweep] completed step={name} elapsedMin={elapsed / 60.0:.1f}")
    return completed.returncode


def _read_leg_result(path: Path, *, search_player: str) -> LegResult:
    if not path.exists():
        raise SystemExit(f"Eval artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid eval JSON in {path}: {exc}") from exc

    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Missing 'results' object in {path}")

    games_value = results.get("games")
    if not isinstance(games_value, int) or games_value <= 0:
        raise SystemExit(f"Missing positive integer 'results.games' in {path}")

    winners = results.get("winners")
    if not isinstance(winners, dict):
        raise SystemExit(f"Missing 'results.winners' object in {path}")

    normalized_winners = {
        "PlayerA": int(winners.get("PlayerA", 0)),
        "PlayerB": int(winners.get("PlayerB", 0)),
        "Draw": int(winners.get("Draw", 0)),
    }
    if search_player not in ("PlayerA", "PlayerB"):
        raise SystemExit(f"Unexpected search player id: {search_player}")
    search_wins = normalized_winners[search_player]
    search_win_rate = search_wins / games_value
    return LegResult(
        artifact=path,
        games=games_value,
        winners=normalized_winners,
        search_wins=search_wins,
        search_win_rate=search_win_rate,
    )


def _build_row(
    *,
    preset: SearchPreset,
    search_as_player_a: LegResult,
    search_as_player_b: LegResult,
) -> SweepRow:
    combined_games = search_as_player_a.games + search_as_player_b.games
    combined_search_wins = (
        search_as_player_a.search_wins + search_as_player_b.search_wins
    )
    combined_draws = search_as_player_a.winners.get(
        "Draw", 0
    ) + search_as_player_b.winners.get("Draw", 0)
    combined_search_win_rate = (
        combined_search_wins / combined_games if combined_games > 0 else 0.0
    )
    side_gap = abs(
        search_as_player_a.search_win_rate - search_as_player_b.search_win_rate
    )
    return SweepRow(
        preset=preset,
        search_as_player_a=search_as_player_a,
        search_as_player_b=search_as_player_b,
        combined_games=combined_games,
        combined_search_wins=combined_search_wins,
        combined_draws=combined_draws,
        combined_search_win_rate=combined_search_win_rate,
        side_gap=side_gap,
    )


def _manifest_payload(
    *,
    run_id: str,
    args: argparse.Namespace,
    presets: List[SearchPreset],
    started_at_seconds: float,
    status: str,
    rows: List[SweepRow],
    planned_commands: List[Dict[str, object]],
    manifest_path: Path,
    summary_path: Path,
) -> Dict[str, object]:
    return {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "status": status,
        "elapsedSeconds": round(time.perf_counter() - started_at_seconds, 3),
        "config": {
            "pythonBin": str(args.python_bin),
            "gamesPerSide": args.games,
            "runLabel": args.run_label,
            "presets": [preset.name for preset in presets],
            "opponentPolicy": args.opponent_policy,
            "opponentCheckpoint": (
                str(args.opponent_checkpoint) if args.opponent_checkpoint else None
            ),
            "searchGuidanceCheckpoint": (
                str(args.search_guidance_checkpoint)
                if args.search_guidance_checkpoint
                else None
            ),
            "guidanceTemperature": args.guidance_temperature,
            "dryRun": bool(args.dry_run),
        },
        "presetDefinitions": {
            preset.name: {
                "worlds": preset.worlds,
                "rollouts": preset.rollouts,
                "depth": preset.depth,
                "maxRootActions": preset.max_root_actions,
                "rolloutEpsilon": preset.rollout_epsilon,
            }
            for preset in presets
        },
        "results": [
            {
                "preset": row.preset.name,
                "search": {
                    "worlds": row.preset.worlds,
                    "rollouts": row.preset.rollouts,
                    "depth": row.preset.depth,
                    "maxRootActions": row.preset.max_root_actions,
                    "rolloutEpsilon": row.preset.rollout_epsilon,
                },
                "searchAsPlayerA": {
                    "artifact": str(row.search_as_player_a.artifact),
                    "games": row.search_as_player_a.games,
                    "winners": row.search_as_player_a.winners,
                    "searchWins": row.search_as_player_a.search_wins,
                    "searchWinRate": row.search_as_player_a.search_win_rate,
                },
                "searchAsPlayerB": {
                    "artifact": str(row.search_as_player_b.artifact),
                    "games": row.search_as_player_b.games,
                    "winners": row.search_as_player_b.winners,
                    "searchWins": row.search_as_player_b.search_wins,
                    "searchWinRate": row.search_as_player_b.search_win_rate,
                },
                "combined": {
                    "games": row.combined_games,
                    "searchWins": row.combined_search_wins,
                    "draws": row.combined_draws,
                    "searchWinRate": row.combined_search_win_rate,
                    "sideGap": row.side_gap,
                },
            }
            for row in rows
        ],
        "plannedCommands": planned_commands,
        "artifacts": {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
    }


def _write_summary_markdown(path: Path, *, run_id: str, rows: List[SweepRow]) -> None:
    lines = [
        f"# Search Teacher Sweep Summary ({run_id})",
        "",
        "| Rank | Preset | Worlds | Rollouts | Depth | MaxRoot | Epsilon | WinRate | SideGap | SearchWins | Games |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.preset.name} | {row.preset.worlds} | {row.preset.rollouts} | "
            f"{row.preset.depth} | {row.preset.max_root_actions} | {row.preset.rollout_epsilon:.2f} | "
            f"{row.combined_search_win_rate:.3f} | {row.side_gap:.3f} | "
            f"{row.combined_search_wins} | {row.combined_games} |"
        )
    lines.append("")
    lines.append(
        "Ranking order: combined `searchWinRate` descending, then lower `sideGap`, then preset name."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_presets() -> None:
    payload = {
        name: {
            "worlds": preset.worlds,
            "rollouts": preset.rollouts,
            "depth": preset.depth,
            "maxRootActions": preset.max_root_actions,
            "rolloutEpsilon": preset.rollout_epsilon,
        }
        for name, preset in PRESETS.items()
    }
    print(json.dumps(payload, indent=2))


def _join_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    safe = safe.strip("-")
    return safe or "run"


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
