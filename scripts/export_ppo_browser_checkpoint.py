from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

PPO_CHECKPOINT_TYPE = "magnate_ppo_policy_v1"
BROWSER_CHECKPOINT_TYPE = "magnate_ppo_browser_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a PyTorch PPO checkpoint into a browser-friendly JSON payload "
            "for UI inference."
        )
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        required=True,
        help="Input .pt checkpoint path from scripts.train_ppo output.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output browser JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.checkpoint_in, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit("Checkpoint payload must be a mapping object.")

    checkpoint_type = str(payload.get("checkpointType", ""))
    if checkpoint_type != PPO_CHECKPOINT_TYPE:
        raise SystemExit(
            f"Unsupported checkpoint type {checkpoint_type!r}; expected {PPO_CHECKPOINT_TYPE!r}."
        )

    state_dict = payload.get("stateDict")
    if not isinstance(state_dict, dict):
        raise SystemExit("Checkpoint is missing stateDict.")

    out_payload: Dict[str, Any] = {
        "checkpointType": BROWSER_CHECKPOINT_TYPE,
        "sourceCheckpointType": checkpoint_type,
        "sourceCheckpoint": str(args.checkpoint_in),
        "observationDim": int(payload.get("observationDim", 0)),
        "actionFeatureDim": int(payload.get("actionFeatureDim", 0)),
        "hiddenDim": int(payload.get("hiddenDim", 0)),
        "weights": {
            "obsEncoder0Weight": _tensor_to_list(state_dict, "obs_encoder.0.weight"),
            "obsEncoder0Bias": _tensor_to_list(state_dict, "obs_encoder.0.bias"),
            "obsEncoder2Weight": _tensor_to_list(state_dict, "obs_encoder.2.weight"),
            "obsEncoder2Bias": _tensor_to_list(state_dict, "obs_encoder.2.bias"),
            "actionEncoder0Weight": _tensor_to_list(state_dict, "action_encoder.0.weight"),
            "actionEncoder0Bias": _tensor_to_list(state_dict, "action_encoder.0.bias"),
            "policyHead0Weight": _tensor_to_list(state_dict, "policy_head.0.weight"),
            "policyHead0Bias": _tensor_to_list(state_dict, "policy_head.0.bias"),
            "policyHead2Weight": _tensor_to_list(state_dict, "policy_head.2.weight"),
            "policyHead2Bias": _tensor_to_list(state_dict, "policy_head.2.bias"),
        },
        "metadata": payload.get("metadata", {}),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload), encoding="utf-8")
    print(json.dumps({"checkpointIn": str(args.checkpoint_in), "out": str(args.out)}, indent=2))
    return 0


def _tensor_to_list(state_dict: Dict[str, Any], key: str) -> Any:
    value = state_dict.get(key)
    if value is None:
        raise SystemExit(f"stateDict is missing key: {key}")
    if not isinstance(value, torch.Tensor):
        raise SystemExit(f"stateDict[{key!r}] must be a tensor.")
    return value.detach().cpu().tolist()


if __name__ == "__main__":
    raise SystemExit(main())
