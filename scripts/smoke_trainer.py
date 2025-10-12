from __future__ import annotations

import json

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import evaluate_matchup
from trainer.policies import HeuristicPolicy, RandomLegalPolicy


def main() -> int:
    with BridgeClient() as client:
        metadata = client.metadata()
        print(
            "Bridge metadata:",
            json.dumps(
                {
                    "contractName": metadata.get("contractName"),
                    "contractVersion": metadata.get("contractVersion"),
                    "schemaVersion": metadata.get("schemaVersion"),
                }
            ),
        )

        env = MagnateBridgeEnv(client=client)
        summary = evaluate_matchup(
            env=env,
            policy_player_a=HeuristicPolicy(),
            policy_player_b=RandomLegalPolicy(),
            games=4,
            seed_prefix="smoke",
        )
        print(
            "Smoke matchup:",
            json.dumps(
                {
                    "games": summary.games,
                    "winners": summary.winners,
                    "winsByPolicy": summary.wins_by_policy,
                    "averageTurn": summary.average_turn,
                }
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
