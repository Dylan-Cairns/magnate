#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  runpodctl remove pod "$RUNPOD_POD_ID"
}
trap cleanup EXIT

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
  echo "No promoted prior loop found; running without warm start."
fi

python -m scripts.run_td_loop \
  --cloud --cloud-vcpus 16 \
  --run-label td-loop-r2-overnight \
  --chunks-per-loop 3 \
  --collect-games 1200 \
  --train-steps 20000 \
  "${warm_start_args[@]}" \
  --eval-games-per-side 200 \
  --eval-opponent-policy search \
  --progress-heartbeat-minutes 30 \
  --eval-progress-log-minutes 30
