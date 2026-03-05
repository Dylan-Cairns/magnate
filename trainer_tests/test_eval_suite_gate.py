from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvalSuiteGateSmokeTests(unittest.TestCase):
    def test_gate_mode_runs_to_terminal_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact = Path(tmp_dir) / "gate-smoke.json"
            command = [
                sys.executable,
                "-m",
                "scripts.eval_suite",
                "--mode",
                "gate",
                "--candidate-policy",
                "heuristic",
                "--opponent-policy",
                "random",
                "--workers",
                "1",
                "--seed-prefix",
                "gate-smoke",
                "--gate-batch-games-per-side",
                "1",
                "--gate-max-games-per-side",
                "2",
                "--progress-every-games",
                "0",
                "--progress-log-minutes",
                "0",
                "--out",
                str(artifact),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    "gate mode run failed.\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("mode"), "gate")
            self.assertIn(payload.get("status"), ("accepted", "rejected", "completed"))

            results = payload.get("results")
            self.assertIsInstance(results, dict)
            assert isinstance(results, dict)
            self.assertGreaterEqual(int(results["gamesPerSide"]), 1)
            self.assertGreaterEqual(int(results["totalGames"]), 2)


if __name__ == "__main__":
    unittest.main()

