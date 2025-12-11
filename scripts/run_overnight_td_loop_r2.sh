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
  --train-warm-start-value-checkpoint artifacts/td_loops/20260304-060553Z-td-loop-r2-overnight/chunks/chunk-003/train/checkpoints/20260304-153945Z-20260304-060553z-td-loop-r2-overnight-chunk-003/value-step-0015000.pt \
  --train-warm-start-opponent-checkpoint artifacts/td_loops/20260304-060553Z-td-loop-r2-overnight/chunks/chunk-003/train/checkpoints/20260304-153945Z-20260304-060553z-td-loop-r2-overnight-chunk-003/opponent-step-0015000.pt \
  --eval-games-per-side 200 \
  --eval-opponent-policy search \
  --progress-heartbeat-minutes 30 \
  --eval-progress-log-minutes 30
