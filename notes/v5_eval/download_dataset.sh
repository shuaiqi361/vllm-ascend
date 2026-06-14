#!/usr/bin/env bash
# Fetch the ShareGPT dataset used by `vllm bench serve --dataset-name sharegpt`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env.sh
source "$HERE/common.env.sh"

DIR="$(dirname "$SHAREGPT_PATH")"
if [ -f "$SHAREGPT_PATH" ]; then
  echo "[dataset] already present: $SHAREGPT_PATH"
  exit 0
fi

mkdir -p "$DIR"
echo "[dataset] downloading ShareGPT -> $SHAREGPT_PATH"
# hf-mirror is the source used by the repo's own benchmark scripts (CN-friendly).
URL="https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
if ! wget -O "$SHAREGPT_PATH" "$URL"; then
  echo "[dataset] hf-mirror failed, trying huggingface.co ..." >&2
  wget -O "$SHAREGPT_PATH" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
fi
echo "[dataset] saved $SHAREGPT_PATH"
