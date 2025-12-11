#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  runpodctl remove pod "$RUNPOD_POD_ID"
}
trap cleanup EXIT

python -m scripts.run_td_loop \
  --cloud --cloud-vcpus 16 \
  --run-label td-loop-r2-overnight \
  --chunks-per-loop 3 \
  --collect-games 800 \
  --train-steps 30000 \
  --eval-games-per-side 200 \
  --eval-opponent-policy search \
  --progress-heartbeat-minutes 30 \
  --eval-progress-log-minutes 30
