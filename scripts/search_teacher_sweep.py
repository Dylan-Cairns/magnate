from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class SearchPreset:
    preset_id: str
    worlds: int
    rollouts: int
    depth: int
    max_root_actions: int
    rollout_epsilon: float


@dataclass(frozen=True)
class SweepRow:
    preset: SearchPreset
    artifact: Path
    candidate_win_rate: float
    ci_low: float
    ci_high: float
    side_gap: float
    candidate_wins: int
    opponent_wins: int
    draws: int
    total_games: int


PACKS: Dict[str, List[SearchPreset]] = {
    "coarse-v1": [
        SearchPreset("s01", worlds=4, rollouts=1, depth=10, max_root_actions=4, rollout_epsilon=0.08),
        SearchPreset("s02", worlds=4, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.08),
        SearchPreset("s03", worlds=6, rollouts=1, depth=12, max_root_actions=6, rollout_epsilon=0.08),
        SearchPreset("s04", worlds=6, rollouts=1, depth=16, max_root_actions=6, rollout_epsilon=0.08),
        SearchPreset("s05", worlds=8, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.08),
        SearchPreset("s06", worlds=8, rollouts=1, depth=18, max_root_actions=8, rollout_epsilon=0.08),
        SearchPreset("s07", worlds=10, rollouts=1, depth=16, max_root_actions=8, rollout_epsilon=0.08),
        SearchPreset("s08", worlds=10, rollouts=1, depth=20, max_root_actions=10, rollout_epsilon=0.08),
    ],
    "epsilon-v1": [
        SearchPreset("e00", worlds=6, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.00),
        SearchPreset("e04", worlds=6, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.04),
        SearchPreset("e08", worlds=6, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.08),
        SearchPreset("e12", worlds=6, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.12),
    ],
    "rollouts-v1": [
        SearchPreset("r1", worlds=6, rollouts=1, depth=14, max_root_actions=6, rollout_epsilon=0.08),
        SearchPreset("r2", worlds=6, rollouts=2, depth=14, max_root_actions=6, rollout_epsilon=0.08),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a search sweep using canonical side-swapped paired-seed eval_suite. "
            "Each preset emits a single eval_suite artifact with win rate, CI, and side gap."
        )
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(".venv/Scripts/python.exe"),
        help="Python executable used for subprocess eval_suite commands.",
    )
    parser.add_argument(
        "--games-per-side",
        type=int,
        default=60,
        help="Games per side for each preset (total per preset = 2x this value).",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="search-sweep",
        help="Run label used for artifact names and seed prefixes.",
    )
    parser.add_argument(
        "--pack",
        type=str,
        default="coarse-v1",
        choices=tuple(sorted(PACKS.keys())),
        help="Built-in preset pack to run.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of preset ids from the selected pack (for example: s03 s04).",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy used against search (random|heuristic|search|mcts|ppo).",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for opponent policy when using ppo.",
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
        "--list-packs",
        action="store_true",
        help="Print built-in pack definitions and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running evaluations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_packs:
        print(json.dumps(_pack_payload(), indent=2))
        return 0

    presets = _resolve_presets(pack_name=args.pack, selected=args.presets)
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
        eval_out = eval_dir / f"{run_id}-{preset.preset_id}.json"
        command = _build_eval_suite_command(
            python_bin=str(args.python_bin),
            games_per_side=args.games_per_side,
            seed_prefix=f"{safe_label}-{preset.preset_id}",
            opponent_policy=args.opponent_policy,
            opponent_checkpoint=args.opponent_checkpoint,
            search_guidance_checkpoint=args.search_guidance_checkpoint,
            guidance_temperature=args.guidance_temperature,
            preset=preset,
            out_path=eval_out,
        )

        planned_commands.append(
            {
                "preset": preset.preset_id,
                "command": command,
                "artifact": str(eval_out),
            }
        )

        print(
            f"[sweep] preset {index}/{len(presets)} {preset.preset_id} "
            f"worlds={preset.worlds} rollouts={preset.rollouts} depth={preset.depth} "
            f"maxRoot={preset.max_root_actions} eps={preset.rollout_epsilon:.2f}"
        )

        if args.dry_run:
            print(f"[sweep] dry-run cmd: {_join_command(command)}")
            continue

        result = _run_step(name=preset.preset_id, command=command)
        if result != 0:
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
            return result

        row = _read_eval_suite_row(eval_out, preset)
        rows.append(row)
        print(
            "[sweep] finished preset "
            f"{preset.preset_id} winRate={row.candidate_win_rate:.3f} "
            f"ci95=[{row.ci_low:.3f},{row.ci_high:.3f}] "
            f"sideGap={row.side_gap:.3f} "
            f"wins={row.candidate_wins}/{row.total_games}"
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -row.candidate_win_rate,
            row.side_gap,
            (row.ci_high - row.ci_low),
            row.preset.preset_id,
        ),
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
                "pack": args.pack,
                "presets": [preset.preset_id for preset in presets],
            },
            indent=2,
        )
    )
    return 0


def _resolve_presets(pack_name: str, selected: Sequence[str] | None) -> List[SearchPreset]:
    pack = PACKS[pack_name]
    by_id = {preset.preset_id: preset for preset in pack}
    if not selected:
        return list(pack)

    resolved: List[SearchPreset] = []
    seen = set()
    for preset_id in selected:
        key = preset_id.strip()
        if key in seen:
            raise SystemExit(f"Duplicate preset id: {preset_id}")
        preset = by_id.get(key)
        if preset is None:
            available = ", ".join(by_id.keys())
            raise SystemExit(f"Unknown preset id '{preset_id}' for pack {pack_name}. Available: {available}")
        seen.add(key)
        resolved.append(preset)
    return resolved


def _build_eval_suite_command(
    *,
    python_bin: str,
    games_per_side: int,
    seed_prefix: str,
    opponent_policy: str,
    opponent_checkpoint: Path | None,
    search_guidance_checkpoint: Path | None,
    guidance_temperature: float,
    preset: SearchPreset,
    out_path: Path,
) -> List[str]:
    command = [
        python_bin,
        "-m",
        "scripts.eval_suite",
        "--games-per-side",
        str(games_per_side),
        "--seed-prefix",
        seed_prefix,
        "--candidate-policy",
        "search",
        "--opponent-policy",
        opponent_policy,
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
        command.extend(["--opponent-checkpoint", str(opponent_checkpoint)])
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


def _read_eval_suite_row(path: Path, preset: SearchPreset) -> SweepRow:
    if not path.exists():
        raise SystemExit(f"Eval-suite artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid eval-suite JSON in {path}: {exc}") from exc

    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Missing 'results' object in {path}")

    ci = results.get("candidateWinRateCi95")
    if not isinstance(ci, dict):
        raise SystemExit(f"Missing 'results.candidateWinRateCi95' object in {path}")

    return SweepRow(
        preset=preset,
        artifact=path,
        candidate_win_rate=_as_float(results.get("candidateWinRate"), path, "candidateWinRate"),
        ci_low=_as_float(ci.get("low"), path, "candidateWinRateCi95.low"),
        ci_high=_as_float(ci.get("high"), path, "candidateWinRateCi95.high"),
        side_gap=_as_float(results.get("sideGap"), path, "sideGap"),
        candidate_wins=_as_int(results.get("candidateWins"), path, "candidateWins"),
        opponent_wins=_as_int(results.get("opponentWins"), path, "opponentWins"),
        draws=_as_int(results.get("draws"), path, "draws"),
        total_games=_as_int(results.get("totalGames"), path, "totalGames"),
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
            "gamesPerSide": args.games_per_side,
            "runLabel": args.run_label,
            "pack": args.pack,
            "presets": [preset.preset_id for preset in presets],
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
            preset.preset_id: {
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
                "preset": row.preset.preset_id,
                "search": {
                    "worlds": row.preset.worlds,
                    "rollouts": row.preset.rollouts,
                    "depth": row.preset.depth,
                    "maxRootActions": row.preset.max_root_actions,
                    "rolloutEpsilon": row.preset.rollout_epsilon,
                },
                "artifact": str(row.artifact),
                "candidateWinRate": row.candidate_win_rate,
                "candidateWinRateCi95": {"low": row.ci_low, "high": row.ci_high},
                "sideGap": row.side_gap,
                "candidateWins": row.candidate_wins,
                "opponentWins": row.opponent_wins,
                "draws": row.draws,
                "totalGames": row.total_games,
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
        f"# Search Sweep Summary ({run_id})",
        "",
        "| Rank | Preset | Worlds | Rollouts | Depth | MaxRoot | Epsilon | WinRate | CI95 | SideGap | CandidateWins | Games |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.preset.preset_id} | {row.preset.worlds} | {row.preset.rollouts} | "
            f"{row.preset.depth} | {row.preset.max_root_actions} | {row.preset.rollout_epsilon:.2f} | "
            f"{row.candidate_win_rate:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{row.side_gap:.3f} | {row.candidate_wins} | {row.total_games} |"
        )
    lines.append("")
    lines.append(
        "Ranking order: candidate win rate descending, lower side gap, then narrower CI, then preset id."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _pack_payload() -> Dict[str, object]:
    return {
        name: [
            {
                "preset": preset.preset_id,
                "worlds": preset.worlds,
                "rollouts": preset.rollouts,
                "depth": preset.depth,
                "maxRootActions": preset.max_root_actions,
                "rolloutEpsilon": preset.rollout_epsilon,
            }
            for preset in presets
        ]
        for name, presets in PACKS.items()
    }


def _as_float(value: object, path: Path, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise SystemExit(f"Missing numeric field 'results.{label}' in {path}")
    return float(value)


def _as_int(value: object, path: Path, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"Missing integer field 'results.{label}' in {path}")
    return value


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
