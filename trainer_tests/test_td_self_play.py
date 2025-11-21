from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from trainer.td.self_play import (
    SelfPlayEpisode,
    collect_self_play_episode,
    collect_self_play_games,
    flatten_opponent_samples,
    flatten_value_transitions,
)
from trainer.types import KeyedAction, LegalActionsResult, StateResult


class _StubPolicy:
    def choose_action_key(self, view, legal_actions, rng, state=None) -> str:
        del view
        del rng
        del state
        return legal_actions[0].action_key


class _ScriptedEnv:
    def __init__(self) -> None:
        self._step = 0

    def reset(self, seed: str, first_player: str) -> StateResult:
        del seed
        del first_player
        self._step = 0
        return StateResult(
            state={"turn": 0},
            view={"obsTag": "A0"},
            terminal=False,
        )

    def legal_actions(self) -> LegalActionsResult:
        if self._step == 0:
            return LegalActionsResult(
                actions=[KeyedAction(action_id="trade", action_key="a0", action={})],
                active_player_id="PlayerA",
                phase="ActionWindow",
            )
        if self._step == 1:
            return LegalActionsResult(
                actions=[KeyedAction(action_id="trade", action_key="b0", action={})],
                active_player_id="PlayerB",
                phase="ActionWindow",
            )
        if self._step == 2:
            return LegalActionsResult(
                actions=[KeyedAction(action_id="trade", action_key="a1", action={})],
                active_player_id="PlayerA",
                phase="ActionWindow",
            )
        raise RuntimeError("No legal actions available in terminal state.")

    def step(self, action_key: str) -> StateResult:
        del action_key
        self._step += 1
        if self._step == 1:
            return StateResult(state={"turn": 1}, view={"obsTag": "B0"}, terminal=False)
        if self._step == 2:
            return StateResult(state={"turn": 2}, view={"obsTag": "A1"}, terminal=False)
        return StateResult(
            state={"turn": 3, "finalScore": {"winner": "PlayerA"}},
            view={"obsTag": "T"},
            terminal=True,
        )


class TDSelfPlayTests(unittest.TestCase):
    @patch("trainer.td.self_play.encode_action_candidates")
    @patch("trainer.td.self_play.encode_observation")
    def test_collect_self_play_episode_builds_transitions_and_samples(
        self,
        mock_encode_observation,
        mock_encode_action_candidates,
    ) -> None:
        mock_encode_observation.side_effect = lambda view: {
            "A0": [1.0, 0.0],
            "B0": [0.0, 1.0],
            "A1": [0.5, 0.5],
            "T": [0.0, 0.0],
        }[view["obsTag"]]
        mock_encode_action_candidates.side_effect = lambda actions: [[float(index)] for index, _ in enumerate(actions)]

        env = _ScriptedEnv()
        policies = {"PlayerA": _StubPolicy(), "PlayerB": _StubPolicy()}
        episode = collect_self_play_episode(
            env=env,
            policies=policies,
            seed="seed-1",
            first_player="PlayerA",
            rng=random.Random(1),
        )

        self.assertEqual(episode.winner, "PlayerA")
        self.assertEqual(episode.turns, 3)
        self.assertEqual(len(episode.opponent_samples), 3)
        self.assertEqual(len(episode.value_transitions), 3)

        first_transition = episode.value_transitions[0]
        self.assertFalse(first_transition.done)
        self.assertEqual(first_transition.player_id, "PlayerA")
        self.assertEqual(first_transition.observation, [1.0, 0.0])
        self.assertEqual(first_transition.next_observation, [0.5, 0.5])

        terminal_a = episode.value_transitions[1]
        terminal_b = episode.value_transitions[2]
        self.assertTrue(terminal_a.done)
        self.assertTrue(terminal_b.done)
        self.assertEqual(terminal_a.reward, 1.0)
        self.assertEqual(terminal_b.reward, -1.0)

    @patch("trainer.td.self_play.encode_action_candidates", return_value=[[0.0]])
    @patch("trainer.td.self_play.encode_observation", return_value=[0.0, 0.0])
    def test_collect_self_play_games_alternates_first_player(self, _obs, _acts) -> None:
        env = _ScriptedEnv()
        policies = {"PlayerA": _StubPolicy(), "PlayerB": _StubPolicy()}
        progress_events = []
        episodes = collect_self_play_games(
            env=env,
            policy_player_a=policies["PlayerA"],
            policy_player_b=policies["PlayerB"],
            games=3,
            seed_prefix="batch",
            progress_every_games=2,
            on_progress=lambda completed, total, winners: progress_events.append(
                (completed, total, dict(winners))
            ),
        )
        self.assertEqual([episode.first_player for episode in episodes], ["PlayerA", "PlayerB", "PlayerA"])

        flattened_transitions = flatten_value_transitions(episodes)
        flattened_samples = flatten_opponent_samples(episodes)
        self.assertEqual(len(flattened_transitions), 9)
        self.assertEqual(len(flattened_samples), 9)
        self.assertTrue(all(isinstance(episode, SelfPlayEpisode) for episode in episodes))
        self.assertEqual(progress_events[0][0], 2)
        self.assertEqual(progress_events[-1][0], 3)


if __name__ == "__main__":
    unittest.main()
