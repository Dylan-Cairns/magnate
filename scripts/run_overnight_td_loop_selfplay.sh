#!/usr/bin/env bash
set -euo pipefail

log_dir="artifacts/logs"
mkdir -p "$log_dir"
run_stamp="$(date -u +%Y%m%d-%H%M%SZ)"
log_path="$log_dir/${run_stamp}-run_overnight_td_loop_selfplay.log"
status_path="${log_path%.log}.status"

exec > >(tee -a "$log_path") 2>&1

cleanup() {
  local exit_code=$?
  local ended_at
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "endedAtUtc=${ended_at}"
    echo "exitCode=${exit_code}"
    echo "logPath=${log_path}"
  } >"$status_path"

  echo "[overnight] endedAtUtc=${ended_at} exitCode=${exit_code}"
  echo "[overnight] statusPath=${status_path}"
  echo "[overnight] disk snapshot:"
  df -h /workspace || true

  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "[overnight] removing pod ${RUNPOD_POD_ID}"
    runpodctl remove pod "$RUNPOD_POD_ID" || true
  else
    echo "[overnight] RUNPOD_POD_ID is not set; skipping pod removal."
  fi
}
trap cleanup EXIT

echo "[overnight] startedAtUtc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[overnight] logPath=${log_path}"
echo "[overnight] statusPath=${status_path}"

latest_promoted_paths="$(
python - <<'PY'
import json
import pathlib
import sys

root = pathlib.Path("artifacts/td_loops")
if not root.exists():
    sys.exit(0)

latest = None
for summary_path in sorted(root.glob("*/loop.summary.json")):
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    promotion = payload.get("promotion") or {}
    if not bool(promotion.get("promoted")):
        continue
    chunks = payload.get("chunks") or []
    if not chunks:
        continue
    checkpoint = (chunks[-1] or {}).get("latestCheckpoint") or {}
    value_path = checkpoint.get("value")
    opponent_path = checkpoint.get("opponent")
    if not isinstance(value_path, str) or not isinstance(opponent_path, str):
        continue
    latest = (summary_path.parent.name, value_path, opponent_path)

if latest is not None:
    print(latest[0])
    print(latest[1])
    print(latest[2])
PY
)"

warm_start_args=()
if [[ -n "$latest_promoted_paths" ]]; then
  mapfile -t promoted_lines <<<"$latest_promoted_paths"
  if [[ "${#promoted_lines[@]}" -eq 3 ]]; then
    promoted_run_id="${promoted_lines[0]}"
    promoted_value_ckpt="${promoted_lines[1]}"
    promoted_opponent_ckpt="${promoted_lines[2]}"
    echo "Using promoted warm start from ${promoted_run_id}"
    warm_start_args+=(
      --train-warm-start-value-checkpoint "$promoted_value_ckpt"
      --train-warm-start-opponent-checkpoint "$promoted_opponent_ckpt"
    )
  fi
else
  echo "No promoted prior loop found; relying on self-play loop default warm-start resolution."
fi

python -m scripts.run_td_loop_selfplay \
  --cloud --cloud-vcpus 8 \
  --run-label td-loop-selfplay-r1-overnight \
  --chunks-per-loop 6 \
  --collect-games 600 \
  --train-steps 10000 \
  "${warm_start_args[@]}" \
  --eval-games-per-side 200 \
  --incumbent-eval-games-per-side 200 \
  --progress-heartbeat-minutes 30 \
  --eval-progress-log-minutes 30
