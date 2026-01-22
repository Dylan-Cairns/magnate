from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import eval_suite


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateExecutor:
    def __init__(self, *args, **kwargs) -> None:
        self._futures: list[_ImmediateFuture] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, payload):
        future = _ImmediateFuture(fn(payload))
        self._futures.append(future)
        return future


class EvalSuiteControlFlowTests(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["eval_suite.py", *args]):
            return eval_suite.parse_args()

    def _summary(
        self,
        *,
        games_per_side: int,
        candidate_wins_as_a: int,
        candidate_wins_as_b: int,
        draws_as_a: int = 0,
        draws_as_b: int = 0,
        candidate: str = "heuristic",
        opponent: str = "random",
    ) -> dict[str, object]:
        opponent_wins_as_b = games_per_side - candidate_wins_as_a - draws_as_a
        opponent_wins_as_a = games_per_side - candidate_wins_as_b - draws_as_b
        candidate_wins = candidate_wins_as_a + candidate_wins_as_b
        draws = draws_as_a + draws_as_b
        total_games = games_per_side * 2
        opponent_wins = total_games - candidate_wins - draws
        candidate_win_rate = candidate_wins / float(total_games)
        candidate_win_rate_as_a = candidate_wins_as_a / float(games_per_side)
        candidate_win_rate_as_b = candidate_wins_as_b / float(games_per_side)
        return {
            "gamesPerSide": games_per_side,
            "totalGames": total_games,
            "candidate": candidate,
            "opponent": opponent,
            "winners": {
                "PlayerA": candidate_wins_as_a + opponent_wins_as_a,
                "PlayerB": opponent_wins_as_b + candidate_wins_as_b,
                "Draw": draws,
            },
            "candidateWins": candidate_wins,
            "opponentWins": opponent_wins,
            "draws": draws,
            "candidateWinRate": candidate_win_rate,
            "candidateWinRateCi95": {
                "low": max(0.0, candidate_win_rate - 0.05),
                "high": min(1.0, candidate_win_rate + 0.05),
            },
            "candidateWinRateAsPlayerA": candidate_win_rate_as_a,
            "candidateWinRateAsPlayerB": candidate_win_rate_as_b,
            "sideGap": abs(candidate_win_rate_as_a - candidate_win_rate_as_b),
            "averageTurn": 11.0,
            "legs": {
                "candidateAsPlayerA": {
                    "games": games_per_side,
                    "winners": {
                        "PlayerA": candidate_wins_as_a,
                        "PlayerB": opponent_wins_as_b,
                        "Draw": draws_as_a,
                    },
                    "winsBySeat": {
                        "PlayerA": candidate_wins_as_a,
                        "PlayerB": opponent_wins_as_b,
                    },
                    "policyBySeat": {
                        "PlayerA": candidate,
                        "PlayerB": opponent,
                    },
                    "averageTurn": 10.0,
                },
                "candidateAsPlayerB": {
                    "games": games_per_side,
                    "winners": {
                        "PlayerA": opponent_wins_as_a,
                        "PlayerB": candidate_wins_as_b,
                        "Draw": draws_as_b,
                    },
                    "winsBySeat": {
                        "PlayerA": opponent_wins_as_a,
                        "PlayerB": candidate_wins_as_b,
                    },
                    "policyBySeat": {
                        "PlayerA": opponent,
                        "PlayerB": candidate,
                    },
                    "averageTurn": 12.0,
                },
            },
        }

    def test_main_certify_writes_artifact_progress_and_terminal_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "certify.json"
            progress_path = root / "certify.progress.json"
            summary = self._summary(
                games_per_side=2,
                candidate_wins_as_a=2,
                candidate_wins_as_b=1,
            )

            argv = [
                "--mode",
                "certify",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
            ]
            with patch.object(sys, "argv", ["eval_suite.py", *argv]), patch(
                "scripts.eval_suite._require_supported_runtime",
                return_value=None,
            ), patch("scripts.eval_suite._validate_policy_args", return_value=None), patch(
                "scripts.eval_suite._validate_args",
                return_value=None,
            ), patch("scripts.eval_suite._evaluate_results", return_value=summary), patch(
                "builtins.print"
            ) as mocked_print:
                result = eval_suite.main()

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            terminal_report = json.loads(mocked_print.call_args.args[0])

        self.assertEqual(result, 0)
        self.assertEqual(payload["mode"], "certify")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["results"]["candidateWins"], 3)
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["mode"], "certify")
        self.assertEqual(terminal_report["artifact"], str(artifact_path))
        self.assertEqual(terminal_report["status"], "completed")

    def test_main_gate_prints_terminal_summary_from_gate_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "gate.json"
            progress_path = root / "gate.progress.json"
            gate_payload = {
                "status": "accepted",
                "decision": {"state": "accepted", "reason": "sprt_accept"},
                "results": self._summary(
                    games_per_side=25,
                    candidate_wins_as_a=23,
                    candidate_wins_as_b=22,
                ),
            }

            argv = [
                "--mode",
                "gate",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
            ]
            with patch.object(sys, "argv", ["eval_suite.py", *argv]), patch(
                "scripts.eval_suite._require_supported_runtime",
                return_value=None,
            ), patch("scripts.eval_suite._validate_policy_args", return_value=None), patch(
                "scripts.eval_suite._validate_args",
                return_value=None,
            ), patch("scripts.eval_suite._run_gate_mode", return_value=gate_payload), patch(
                "builtins.print"
            ) as mocked_print:
                result = eval_suite.main()

            terminal_report = json.loads(mocked_print.call_args.args[0])

        self.assertEqual(result, 0)
        self.assertEqual(terminal_report["mode"], "gate")
        self.assertEqual(terminal_report["status"], "accepted")
        self.assertEqual(terminal_report["decision"], "accepted")

    def test_run_gate_mode_accepts_and_updates_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "gate.json"
            progress_path = root / "gate.progress.json"
            args = self._parse(
                "--mode",
                "gate",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
                "--gate-batch-games-per-side",
                "25",
                "--gate-max-games-per-side",
                "50",
            )
            summary = self._summary(
                games_per_side=25,
                candidate_wins_as_a=23,
                candidate_wins_as_b=22,
            )

            with patch("scripts.eval_suite._evaluate_results", return_value=summary):
                payload = eval_suite._run_gate_mode(
                    args=args,
                    output_path=artifact_path,
                    progress_path=progress_path,
                )

            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["decision"]["reason"], "sprt_accept")
        self.assertEqual(len(payload["history"]), 1)
        self.assertEqual(progress["status"], "accepted")
        self.assertEqual(progress["decision"]["state"], "accepted")
        self.assertEqual(artifact["status"], "accepted")

    def test_run_gate_mode_rejects_when_side_gap_exceeds_limit_after_accept_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "gate.json"
            progress_path = root / "gate.progress.json"
            args = self._parse(
                "--mode",
                "gate",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
                "--gate-batch-games-per-side",
                "25",
                "--gate-max-games-per-side",
                "50",
            )
            summary = self._summary(
                games_per_side=25,
                candidate_wins_as_a=25,
                candidate_wins_as_b=21,
            )

            with patch("scripts.eval_suite._evaluate_results", return_value=summary):
                payload = eval_suite._run_gate_mode(
                    args=args,
                    output_path=artifact_path,
                    progress_path=progress_path,
                )

        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["decision"]["reason"], "side_gap_exceeded")

    def test_run_gate_mode_returns_existing_terminal_artifact_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "gate.json"
            progress_path = root / "gate.progress.json"
            args = self._parse(
                "--mode",
                "gate",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
            )
            payload = eval_suite._new_gate_payload(args=args, artifact_path=artifact_path)
            payload["status"] = "accepted"
            payload["decision"] = {"state": "accepted", "reason": "sprt_accept", "maxSideGap": 0.08}
            payload["results"] = self._summary(
                games_per_side=25,
                candidate_wins_as_a=20,
                candidate_wins_as_b=20,
            )
            eval_suite._write_json_atomic(artifact_path, payload)

            with patch("scripts.eval_suite._evaluate_results", side_effect=AssertionError("should not rerun")):
                result = eval_suite._run_gate_mode(
                    args=args,
                    output_path=artifact_path,
                    progress_path=progress_path,
                )

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["decision"]["reason"], "sprt_accept")

    def test_run_gate_mode_finalizes_at_cap_without_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "gate.json"
            progress_path = root / "gate.progress.json"
            args = self._parse(
                "--mode",
                "gate",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
                "--gate-max-games-per-side",
                "50",
            )
            payload = eval_suite._new_gate_payload(args=args, artifact_path=artifact_path)
            payload["progress"]["gamesPerSideCompleted"] = 50
            payload["progress"]["gamesPerSideRemaining"] = 0
            eval_suite._write_json_atomic(artifact_path, payload)

            result = eval_suite._run_gate_mode(
                args=args,
                output_path=artifact_path,
                progress_path=progress_path,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["decision"]["reason"], "max_games_reached_no_results")

    def test_validate_gate_resume_payload_rejects_config_mismatch(self) -> None:
        args = self._parse(
            "--mode",
            "gate",
            "--candidate-policy",
            "heuristic",
            "--opponent-policy",
            "random",
        )
        payload = eval_suite._new_gate_payload(args=args, artifact_path=Path("gate.json"))
        payload["config"]["seedPrefix"] = "different"

        with self.assertRaises(SystemExit):
            eval_suite._validate_gate_resume_payload(args=args, payload=payload)

    def test_evaluate_results_single_worker_writes_completed_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "certify.json"
            progress_path = root / "certify.progress.json"
            args = self._parse(
                "--mode",
                "certify",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
            )
            summary = self._summary(
                games_per_side=2,
                candidate_wins_as_a=2,
                candidate_wins_as_b=1,
            )

            with patch("scripts.eval_suite._run_eval_shard", return_value=summary):
                result = eval_suite._evaluate_results(
                    args=args,
                    games_per_side=2,
                    seed_start_index=0,
                    seed_prefix="eval-suite",
                    workers=1,
                    progress_path=progress_path,
                    progress_label="certify",
                )

            progress = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual(result["candidateWins"], 3)
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["completedShards"], 1)
        self.assertEqual(progress["candidateWins"], 3)

    def test_evaluate_results_multi_worker_merges_shards_and_updates_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "certify.json"
            progress_path = root / "certify.progress.json"
            args = self._parse(
                "--mode",
                "certify",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--out",
                str(artifact_path),
                "--progress-out",
                str(progress_path),
                "--progress-log-minutes",
                "0",
            )

            def shard_result(payload: dict[str, object]) -> dict[str, object]:
                games_per_side = int(payload["gamesPerSide"])
                return self._summary(
                    games_per_side=games_per_side,
                    candidate_wins_as_a=games_per_side,
                    candidate_wins_as_b=max(0, games_per_side - 1),
                )

            with patch("scripts.eval_suite.ProcessPoolExecutor", _ImmediateExecutor), patch(
                "scripts.eval_suite.wait",
                side_effect=lambda pending, timeout, return_when: (set(pending), set()),
            ), patch("scripts.eval_suite._evaluate_shard", side_effect=shard_result), patch(
                "builtins.print"
            ):
                result = eval_suite._evaluate_results(
                    args=args,
                    games_per_side=4,
                    seed_start_index=0,
                    seed_prefix="eval-suite",
                    workers=2,
                    progress_path=progress_path,
                    progress_label="certify",
                )

            progress = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual(result["totalGames"], 8)
        self.assertEqual(result["candidateWins"], 6)
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["completedShards"], 2)
        self.assertEqual(progress["workers"], 2)

    def test_resolve_progress_path_prefers_override(self) -> None:
        output_path = Path("artifacts/evals/certify.json")
        override = Path("artifacts/evals/custom.progress.json")
        args = self._parse(
            "--mode",
            "certify",
            "--candidate-policy",
            "heuristic",
            "--opponent-policy",
            "random",
            "--progress-out",
            str(override),
        )

        resolved = eval_suite._resolve_progress_path(args=args, output_path=output_path)

        self.assertEqual(resolved, override)

    def test_write_json_atomic_round_trips_through_read_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.json"
            payload = {"status": "completed", "results": {"candidateWins": 3}}

            eval_suite._write_json_atomic(artifact_path, payload)
            loaded = eval_suite._read_json_object(artifact_path, label="artifact")

            self.assertEqual(loaded, payload)
            self.assertFalse(artifact_path.with_name(f".{artifact_path.name}.tmp").exists())

    def test_read_json_object_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.json"
            artifact_path.write_text("{", encoding="utf-8")

            with self.assertRaises(SystemExit):
                eval_suite._read_json_object(artifact_path, label="artifact")


if __name__ == "__main__":
    unittest.main()
