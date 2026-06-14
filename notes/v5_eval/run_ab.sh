#!/usr/bin/env bash
# End-to-end A/B: baseline (offload+LRC) vs prefetch (offload+LRC+next-layer
# prefetch). For each mode it starts the server, waits, benchmarks ShareGPT,
# stops the server, then prints a side-by-side comparison.
#
#   ./run_ab.sh
#
# Everything is configured via common.env.sh (edit MODEL there first).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env.sh
source "$HERE/common.env.sh"

if [[ "$MODEL" == /* ]] && [ ! -e "$MODEL" ]; then
  echo "ERROR: MODEL path '$MODEL' does not exist. Edit notes/v5_eval/common.env.sh" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

# Make sure the dataset is present before we spend time starting a server.
bash "$HERE/download_dataset.sh"

run_one() {
  local mode="$1"
  local log="$RESULTS_DIR/serve_${mode}.log"
  echo
  echo "==================================================================="
  echo " MODE: $mode"
  echo "==================================================================="

  # Start server in the background; full output to the log (parsed later).
  ( launch_vllm "$mode" >"$log" 2>&1 ) &
  local server_pid=$!

  # Ensure we always tear the server down, even on bench failure / Ctrl-C.
  trap 'stop_server "$server_pid"' RETURN

  if ! wait_for_server; then
    echo "[run] $mode: server did not come up — see $log" >&2
    tail -n 40 "$log" || true
    return 1
  fi

  bash "$HERE/bench.sh" "$mode"

  stop_server "$server_pid"
  trap - RETURN
}

run_one baseline
run_one prefetch

echo
bash "$HERE/compare.sh"
