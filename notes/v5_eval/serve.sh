#!/usr/bin/env bash
# Launch vllm serve (foreground) for one mode. Run in its own terminal, then use
# bench.sh from another terminal. For an automated end-to-end A/B use run_ab.sh.
#
#   ./serve.sh baseline   # expert offload + LRC cache,            prefetch OFF
#   ./serve.sh prefetch   # expert offload + LRC cache + next-layer prefetch ON
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env.sh
source "$HERE/common.env.sh"

MODE="${1:-}"
case "$MODE" in
  baseline|prefetch) ;;
  *) echo "usage: $0 <baseline|prefetch>" >&2; exit 2 ;;
esac

# Sanity: if MODEL is a local path it must exist (skip check for HF/MS ids).
if [[ "$MODEL" == /* ]] && [ ! -e "$MODEL" ]; then
  echo "ERROR: MODEL path '$MODEL' does not exist. Edit notes/v5_eval/common.env.sh" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/serve_${MODE}.log"
echo "[serve] mode=$MODE  log=$LOG"
echo "[serve] server log is where the [EXPERT-OFFLOAD-CACHE] hit-rate lines appear."

# tee so you see startup live AND keep the log for compare.sh / the analyzer.
launch_vllm "$MODE" 2>&1 | tee "$LOG"
