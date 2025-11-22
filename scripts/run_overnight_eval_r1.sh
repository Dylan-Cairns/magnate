#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALUE_CKPT="artifacts/td_loops/20260302-050919Z-td-loop-r1/train/checkpoints/20260302-154402Z-20260302-050919z-td-loop-r1/value-step-0020000.pt"
OPP_CKPT="artifacts/td_loops/20260302-050919Z-td-loop-r1/train/checkpoints/20260302-154402Z-20260302-050919z-td-loop-r1/opponent-step-0020000.pt"

if [[ ! -f "$VALUE_CKPT" ]]; then
  echo "Missing value checkpoint: $VALUE_CKPT" >&2
  exit 1
fi

if [[ ! -f "$OPP_CKPT" ]]; then
  echo "Missing opponent checkpoint: $OPP_CKPT" >&2
  exit 1
fi

mkdir -p artifacts/evals artifacts/logs

python -m scripts.eval_suite \
  --mode certify \
  --games-per-side 200 \
  --workers 6 \
  --seed-prefix reeval-r1-final-vs-search \
  --candidate-policy td-search \
  --opponent-policy search \
  --search-worlds 6 \
  --search-rollouts 1 \
  --search-depth 14 \
  --search-max-root-actions 6 \
  --search-rollout-epsilon 0.04 \
  --td-search-value-checkpoint "$VALUE_CKPT" \
  --td-search-opponent-checkpoint "$OPP_CKPT" \
  --out artifacts/evals/20260303-reeval-r1-final-vs-search-400.json \
  2>&1 | tee artifacts/logs/20260303-reeval-r1-final-vs-search-400.log

python -m scripts.eval_suite \
  --mode certify \
  --games-per-side 200 \
  --workers 6 \
  --seed-prefix reeval-r1-final-vs-heuristic \
  --candidate-policy td-search \
  --opponent-policy heuristic \
  --search-worlds 6 \
  --search-rollouts 1 \
  --search-depth 14 \
  --search-max-root-actions 6 \
  --search-rollout-epsilon 0.04 \
  --td-search-value-checkpoint "$VALUE_CKPT" \
  --td-search-opponent-checkpoint "$OPP_CKPT" \
  --out artifacts/evals/20260303-reeval-r1-final-vs-heuristic-400.json \
  2>&1 | tee artifacts/logs/20260303-reeval-r1-final-vs-heuristic-400.log

echo "Overnight evals completed."
