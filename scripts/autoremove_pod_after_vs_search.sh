#!/usr/bin/env bash
set -euo pipefail

TARGET_ARTIFACT="${TARGET_ARTIFACT:-artifacts/evals/20260303-reeval-r1-final-vs-search-400.json}"
PROCESS_PATTERN="${PROCESS_PATTERN:-python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 6 --seed-prefix reeval-r1-final-vs-search}"
POLL_SECONDS="${POLL_SECONDS:-30}"

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
  echo "RUNPOD_POD_ID is not set." >&2
  exit 1
fi

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl is not installed or not in PATH." >&2
  exit 1
fi

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECONDS" -le 0 ]]; then
  echo "POLL_SECONDS must be a positive integer." >&2
  exit 1
fi

echo "[autoremove] watching artifact: $TARGET_ARTIFACT"
echo "[autoremove] watching process pattern: $PROCESS_PATTERN"
echo "[autoremove] pod id: $RUNPOD_POD_ID"

while :; do
  if [[ -f "$TARGET_ARTIFACT" ]]; then
    echo "[autoremove] artifact found; removing pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID"
    exit 0
  fi

  if ! pgrep -f "$PROCESS_PATTERN" >/dev/null 2>&1; then
    echo "[autoremove] process not found before artifact; removing pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
