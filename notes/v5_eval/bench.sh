#!/usr/bin/env bash
# Run `vllm bench serve` on ShareGPT against an already-running server, and save
# the result JSON tagged with the mode.
#
#   ./bench.sh baseline    # expects serve.sh baseline running
#   ./bench.sh prefetch    # expects serve.sh prefetch running
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env.sh
source "$HERE/common.env.sh"

MODE="${1:-}"
case "$MODE" in
  baseline|prefetch) ;;
  *) echo "usage: $0 <baseline|prefetch>" >&2; exit 2 ;;
esac

if [ ! -f "$SHAREGPT_PATH" ]; then
  echo "ERROR: ShareGPT not found at $SHAREGPT_PATH — run ./download_dataset.sh first." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
RESULT_FILE="bench_${MODE}.json"

extra=()
[ -n "$SHAREGPT_OUTPUT_LEN" ] && extra+=(--sharegpt-output-len "$SHAREGPT_OUTPUT_LEN")

echo "[bench] mode=$MODE rate=$REQUEST_RATE prompts=$NUM_PROMPTS -> $RESULTS_DIR/$RESULT_FILE"

# --backend vllm + sharegpt is exactly what the repo's own serving-tests use
# (hits /v1/completions). --model must match the server's --served-model-name.
vllm bench serve \
  --backend vllm \
  --base-url "http://${HOST}:${PORT}" \
  --model "$SERVED_NAME" \
  --tokenizer "$MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$SHAREGPT_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --seed 0 \
  --save-result \
  --result-dir "$RESULTS_DIR" \
  --result-filename "$RESULT_FILE" \
  "${extra[@]}"

echo "[bench] saved $RESULTS_DIR/$RESULT_FILE"
