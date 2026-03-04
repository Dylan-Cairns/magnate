#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  runpodctl remove pod "$RUNPOD_POD_ID"
}
trap cleanup EXIT

python -m scripts.run_td_loop \
  --cloud --cloud-vcpus 16 \
  --run-label td-loop-r2-overnight \
  --chunks-per-gate 3 \
  --collect-games 1500 \
  --train-steps 15000 \
  --gate-batch-games-per-side 350 \
  --gate-max-games-per-side 350 \
  --certify-games-per-side 200 \
  --certify-opponents search heuristic
