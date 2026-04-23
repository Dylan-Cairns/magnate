from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, List, Optional, TextIO, cast
from uuid import uuid4

from .bridge_payloads import (
    ActionId,
    BridgeMetadataPayload,
    GameActionPayload,
    GamePhase,
    PlayerId,
    PlayerViewPayload,
    SerializedStatePayload,
)
from .types import KeyedAction, LegalActionsResult, ObservationResult, StateResult

JsonMapping = Mapping[str, object]

_PLAYER_IDS: tuple[PlayerId, ...] = ("PlayerA", "PlayerB")
_GAME_PHASES: tuple[GamePhase, ...] = (
    "StartTurn",
    "TaxCheck",
    "CollectIncome",
    "ActionWindow",
    "DrawCard",
    "GameOver",
)
_ACTION_IDS: tuple[ActionId, ...] = (
    "buy-deed",
    "choose-income-suit",
    "develop-deed",
    "develop-outright",
    "end-turn",
    "sell-card",
    "trade",
)


class BridgeError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}


class BridgeClient:
    """Synchronous NDJSON client for the Magnate Node bridge runtime."""

    def __init__(
        self,
        command: Optional[Iterable[str]] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        self._cwd = Path(cwd) if cwd else Path(__file__).resolve().parents[1]
        self._command = list(command) if command else _default_bridge_command(self._cwd)
        self._lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stderr_tail_lock = threading.Lock()
        self._process = self._start_process()
        self._stderr_thread = self._start_stderr_drain()

    def _start_process(self) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            self._command,
            cwd=str(self._cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Failed to open bridge process pipes.")
        return process

    def _start_stderr_drain(self) -> threading.Thread | None:
        if self._process.stderr is None:
            return None
        thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            daemon=True,
            name="magnate-bridge-stderr",
        )
        thread.start()
        return thread

    def _drain_stderr(self, stream: TextIO) -> None:
        try:
            while True:
                line = stream.readline()
                if line == "":
                    break
                cleaned = line.rstrip("\r\n")
                with self._stderr_tail_lock:
                    self._stderr_tail.append(cleaned)
        except Exception:
            return None

    def _stderr_snapshot(self) -> str:
        with self._stderr_tail_lock:
            if not self._stderr_tail:
                return ""
            return "\n".join(self._stderr_tail).strip()

    def close(self) -> None:
        if self._process.poll() is not None:
            self._close_pipes()
            self._join_stderr_thread()
            return

        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except OSError:
            pass

        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)
        finally:
            self._close_pipes()
            self._join_stderr_thread()

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _close_pipes(self) -> None:
        for stream in (self._process.stdout, self._process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                continue

    def _join_stderr_thread(self) -> None:
        if self._stderr_thread is None:
            return
        self._stderr_thread.join(timeout=0.5)
        self._stderr_thread = None

    def _request(
        self,
        command: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Bridge process is not running.")
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Bridge process streams are unavailable.")

            request_id = f"py-{uuid4().hex}"
            envelope = {
                "requestId": request_id,
                "command": command,
                "payload": dict(payload) if payload is not None else {},
            }

            self._process.stdin.write(json.dumps(envelope) + "\n")
            self._process.stdin.flush()

            line = self._process.stdout.readline()
            if line == "":
                stderr = self._stderr_snapshot()
                raise RuntimeError(
                    "Bridge process terminated unexpectedly while waiting for response. "
                    f"stderr={stderr!r}"
                )

            candidate = line.strip()
            if not candidate:
                raise RuntimeError("Bridge returned an empty response line.")

            try:
                response = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Bridge returned non-JSON response: {candidate!r}") from error

            if not isinstance(response, dict):
                raise RuntimeError(
                    f"Bridge response must be an object, got {type(response).__name__}."
                )

            response_request_id = response.get("requestId")
            if response_request_id != request_id:
                raise RuntimeError(
                    "Bridge response requestId mismatch. "
                    f"expected={request_id!r} actual={response_request_id!r}"
                )

            if response.get("ok") is not True:
                error = response.get("error") or {}
                raise BridgeError(
                    code=str(error.get("code", "UNKNOWN")),
                    message=str(error.get("message", "Unknown bridge error")),
                    details=error.get("details") or {},
                )

            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Bridge result must be an object, got {type(result).__name__}")
            return dict(result)

    def metadata(self) -> BridgeMetadataPayload:
        result = self._request("metadata")
        return _metadata_from_bridge(result)

    def reset(
        self,
        seed: Optional[str] = None,
        first_player: Optional[PlayerId] = None,
        serialized_state: Optional[SerializedStatePayload] = None,
        skip_advance_to_decision: bool = False,
    ) -> StateResult:
        payload: dict[str, object] = {}
        if seed is not None:
            payload["seed"] = seed
        if first_player is not None:
            payload["firstPlayer"] = first_player
        if serialized_state is not None:
            payload["serializedState"] = serialized_state
            payload["skipAdvanceToDecision"] = skip_advance_to_decision

        result = self._request("reset", payload)
        return _state_result_from_bridge(result)

    def legal_actions(self) -> LegalActionsResult:
        result = self._request("legalActions", {})
        return _legal_actions_result_from_bridge(result)

    def observation(
        self,
        viewer_id: Optional[PlayerId] = None,
        include_legal_action_mask: bool = False,
    ) -> ObservationResult:
        payload: dict[str, object] = {"includeLegalActionMask": include_legal_action_mask}
        if viewer_id is not None:
            payload["viewerId"] = viewer_id

        result = self._request("observation", payload)
        return _observation_result_from_bridge(result)

    def step(
        self,
        action_key: Optional[str] = None,
        action: Optional[GameActionPayload] = None,
    ) -> StateResult:
        payload: dict[str, object] = {}
        if action_key is not None:
            payload["actionKey"] = action_key
        if action is not None:
            payload["action"] = action

        result = self._request("step", payload)
        return _state_result_from_bridge(result)

    def serialize(self) -> SerializedStatePayload:
        result = self._request("serialize", {})
        return _serialized_state_from_bridge(result.get("state"), label="serialize.state")


def _metadata_from_bridge(result: JsonMapping) -> BridgeMetadataPayload:
    contract_name = result.get("contractName")
    if contract_name != "magnate_bridge":
        raise RuntimeError(
            f"metadata.contractName must be 'magnate_bridge', got {contract_name!r}."
        )
    contract_version = result.get("contractVersion")
    if contract_version != "v1":
        raise RuntimeError(f"metadata.contractVersion must be 'v1', got {contract_version!r}.")
    return cast(BridgeMetadataPayload, result)


def _state_result_from_bridge(result: JsonMapping) -> StateResult:
    return StateResult(
        state=_serialized_state_from_bridge(result.get("state"), label="stateResult.state"),
        view=_player_view_from_bridge(result.get("view"), label="stateResult.view"),
        terminal=_require_bool(result.get("terminal"), "stateResult.terminal"),
    )


def _legal_actions_result_from_bridge(result: JsonMapping) -> LegalActionsResult:
    raw_actions = result.get("actions")
    if not isinstance(raw_actions, list):
        raise RuntimeError(
            f"legalActions.actions must be a list, got {type(raw_actions).__name__}."
        )

    actions: list[KeyedAction] = []
    for index, entry in enumerate(raw_actions):
        mapping = _require_mapping(entry, f"legalActions.actions[{index}]")
        action = _require_mapping(mapping.get("action"), f"legalActions.actions[{index}].action")
        actions.append(
            KeyedAction(
                action_id=_require_action_id(
                    mapping.get("actionId"),
                    f"legalActions.actions[{index}].actionId",
                ),
                action_key=_require_str(
                    mapping.get("actionKey"),
                    f"legalActions.actions[{index}].actionKey",
                ),
                action=cast(GameActionPayload, action),
            )
        )

    return LegalActionsResult(
        actions=actions,
        active_player_id=_require_player_id(
            result.get("activePlayerId"),
            "legalActions.activePlayerId",
        ),
        phase=_require_phase(result.get("phase"), "legalActions.phase"),
    )


def _observation_result_from_bridge(result: JsonMapping) -> ObservationResult:
    legal_action_mask: list[str] | None = None
    raw_mask = result.get("legalActionMask")
    if raw_mask is not None:
        if not isinstance(raw_mask, list):
            raise RuntimeError(
                f"observation.legalActionMask must be a list, got {type(raw_mask).__name__}."
            )
        legal_action_mask = [
            _require_str(entry, f"observation.legalActionMask[{index}]")
            for index, entry in enumerate(raw_mask)
        ]

    return ObservationResult(
        view=_player_view_from_bridge(result.get("view"), label="observation.view"),
        legal_action_mask=legal_action_mask,
    )


def _serialized_state_from_bridge(value: object, *, label: str) -> SerializedStatePayload:
    return cast(SerializedStatePayload, _require_mapping(value, label))


def _player_view_from_bridge(value: object, *, label: str) -> PlayerViewPayload:
    return cast(PlayerViewPayload, _require_mapping(value, label))


def _require_mapping(value: object, label: str) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object, got {type(value).__name__}.")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string, got {type(value).__name__}.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be a boolean, got {type(value).__name__}.")
    return value


def _require_player_id(value: object, label: str) -> PlayerId:
    if value not in _PLAYER_IDS:
        raise RuntimeError(f"{label} must be PlayerA|PlayerB, got {value!r}.")
    return cast(PlayerId, value)


def _require_phase(value: object, label: str) -> GamePhase:
    if value not in _GAME_PHASES:
        raise RuntimeError(f"{label} must be a valid game phase, got {value!r}.")
    return cast(GamePhase, value)


def _require_action_id(value: object, label: str) -> ActionId:
    if value not in _ACTION_IDS:
        raise RuntimeError(f"{label} must be a valid action id, got {value!r}.")
    return cast(ActionId, value)


def _default_bridge_command(cwd: Path) -> List[str]:
    tsx_executable = "tsx.cmd" if os.name == "nt" else "tsx"
    tsx_path = cwd / "node_modules" / ".bin" / tsx_executable
    return [str(tsx_path), "src/bridge/cli.ts"]
