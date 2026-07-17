from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts import prepare_td_district_symmetry_evaluation as evaluation
from trainer.td import DISTRICT_AUGMENTATION_S4


class TDDistrictSymmetryEvaluationTests(unittest.TestCase):
    def test_manifest_s4_shorthand_matches_canonical_training_summary_mode(self) -> None:
        self.assertEqual(
            evaluation._canonical_augmentation_mode("s4"),
            DISTRICT_AUGMENTATION_S4,
        )
        self.assertEqual(evaluation._canonical_augmentation_mode("none"), "none")

    def test_matched_sampling_traces_are_required_across_arms(self) -> None:
        control = self._run("pilot-a", "continued-control")
        augmented = self._run("pilot-a", "s4-augmented")
        runs = {
            ("pilot-a", "continued-control"): control,
            ("pilot-a", "s4-augmented"): augmented,
        }

        evaluation._verify_matched_sampling_traces(runs)

        runs[("pilot-a", "s4-augmented")] = replace(
            augmented,
            opponent_sampling_trace_sha256="different",
        )
        with self.assertRaisesRegex(SystemExit, "matched opponent sampling trace"):
            evaluation._verify_matched_sampling_traces(runs)

    def test_head_to_head_spec_binds_an_explicit_pack_and_search_budget(self) -> None:
        spec = evaluation._td_root_bot_spec(
            bot_id="candidate",
            pack_id="frozen-pack",
            model_index_path="model-packs-experiments/pilot/index.json",
            full_games={
                "botSpec": {
                    "worlds": 10,
                    "rollouts": 1,
                    "depth": 40,
                    "maxRootActions": 16,
                    "rolloutEpsilon": 0.0,
                    "guidance": {"root": "td", "rollout": "td", "leaf": "td"},
                }
            },
        )

        self.assertEqual(
            spec["modelIndexPath"],
            "model-packs-experiments/pilot/index.json?tdPackId=frozen-pack",
        )
        self.assertEqual(spec["config"]["worlds"], 10)
        self.assertEqual(spec["guidance"]["leaf"], "td")

    def test_training_summary_selects_only_the_frozen_final_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "value-step-0005000.pt"
            checkpoint.write_bytes(b"checkpoint")
            summary = {
                "config": {
                    "steps": 5000,
                    "seed": 17,
                    "districtAugmentation": "s4",
                },
                "provenance": {
                    "experimentManifestSha256": "manifest",
                    "implementationSha256": "implementation",
                    "valueReplayContentSha256": "replay",
                    "warmStartValueSha256": "warm",
                },
                "results": {
                    "valueSamplingTraceSha256": "trace",
                    "checkpoints": [
                        {"step": 1000, "value": "intermediate.pt"},
                        {"step": 5000, "value": str(checkpoint)},
                    ],
                },
            }

            selected, trace = evaluation._validate_training_summary(
                summary=summary,
                model="value",
                expected_step=5000,
                expected_seed=17,
                expected_augmentation="s4",
                expected_source_manifest_sha256="manifest",
                expected_implementation_sha256="implementation",
                expected_replay_sha256="replay",
                expected_warm_sha256="warm",
            )

            self.assertEqual(selected, checkpoint.resolve())
            self.assertEqual(trace, "trace")

    @staticmethod
    def _run(seed_id: str, arm_id: str) -> evaluation.FinalRunArtifacts:
        return evaluation.FinalRunArtifacts(
            seed_id=seed_id,
            arm_id=arm_id,
            value_checkpoint=Path("value.pt"),
            value_sha256="value",
            value_sampling_trace_sha256="value-trace",
            opponent_checkpoint=Path("opponent.pt"),
            opponent_sha256="opponent",
            opponent_sampling_trace_sha256="opponent-trace",
        )


if __name__ == "__main__":
    unittest.main()
