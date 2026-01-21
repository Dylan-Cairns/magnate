from __future__ import annotations

from collections import deque
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from .types import KeyedAction, LegalActionsResult, ObservationResult, PlayerId, StateResult


class BridgeError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
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

    def _drain_stderr(self, stream: Any) -> None:
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

    def _request(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Bridge process is not running.")
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Bridge process streams are unavailable.")

            request_id = f"py-{uuid4().hex}"
            envelope = {
                "requestId": request_id,
                "command": command,
                "payload": payload if payload is not None else {},
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
            return result

    def metadata(self) -> Dict[str, Any]:
        return self._request("metadata")

    def reset(
        self,
        seed: Optional[str] = None,
        first_player: Optional[PlayerId] = None,
        serialized_state: Optional[Dict[str, Any]] = None,
        skip_advance_to_decision: bool = False,
    ) -> StateResult:
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if first_player is not None:
            payload["firstPlayer"] = first_player
        if serialized_state is not None:
            payload["serializedState"] = serialized_state
            payload["skipAdvanceToDecision"] = skip_advance_to_decision

        result = self._request("reset", payload)
        return StateResult(
            state=result["state"],
            view=result["view"],
            terminal=bool(result["terminal"]),
        )

    def legal_actions(self) -> LegalActionsResult:
        result = self._request("legalActions", {})
        raw_actions = result.get("actions")
        if not isinstance(raw_actions, list):
            raise RuntimeError("Bridge legalActions result is missing 'actions' list.")

        actions: List[KeyedAction] = []
        for entry in raw_actions:
            if not isinstance(entry, dict):
                raise RuntimeError("Each legal action entry must be an object.")
            actions.append(
                KeyedAction(
                    action_id=str(entry["actionId"]),
                    action_key=str(entry["actionKey"]),
                    action=dict(entry["action"]),
                )
            )

        active_player_id = result.get("activePlayerId")
        if active_player_id not in ("PlayerA", "PlayerB"):
            raise RuntimeError(f"Invalid activePlayerId in legalActions response: {active_player_id!r}")

        return LegalActionsResult(
            actions=actions,
            active_player_id=active_player_id,
            phase=str(result.get("phase", "")),
        )

    def observation(
        self,
        viewer_id: Optional[PlayerId] = None,
        include_legal_action_mask: bool = False,
    ) -> ObservationResult:
        payload: Dict[str, Any] = {"includeLegalActionMask": include_legal_action_mask}
        if viewer_id is not None:
            payload["viewerId"] = viewer_id

        result = self._request("observation", payload)
        legal_action_mask = result.get("legalActionMask")
        if legal_action_mask is not None and not isinstance(legal_action_mask, list):
            raise RuntimeError("Bridge observation legalActionMask must be a list when present.")
        return ObservationResult(
            view=result["view"],
            legal_action_mask=legal_action_mask,
        )

    def step(
        self,
        action_key: Optional[str] = None,
        action: Optional[Dict[str, Any]] = None,
    ) -> StateResult:
        payload: Dict[str, Any] = {}
        if action_key is not None:
            payload["actionKey"] = action_key
        if action is not None:
            payload["action"] = action

        result = self._request("step", payload)
        return StateResult(
            state=result["state"],
            view=result["view"],
            terminal=bool(result["terminal"]),
        )

    def serialize(self) -> Dict[str, Any]:
        result = self._request("serialize", {})
        return dict(result["state"])


def _default_bridge_command(cwd: Path) -> List[str]:
    tsx_executable = "tsx.cmd" if os.name == "nt" else "tsx"
    tsx_path = cwd / "node_modules" / ".bin" / tsx_executable
    return [str(tsx_path), "src/bridge/cli.ts"]
