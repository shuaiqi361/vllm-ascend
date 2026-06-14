#!/usr/bin/env bash
# Print a side-by-side comparison of the baseline vs prefetch runs:
#   - cache hit rate  (from the server log [EXPERT-OFFLOAD-CACHE] lines)
#   - throughput + latency (from the vllm bench serve result JSONs)
#
# Run after serve+bench for both modes (run_ab.sh does this automatically).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env.sh
source "$HERE/common.env.sh"

have_jq=1; command -v jq >/dev/null 2>&1 || have_jq=0

# Final cumulative cache hit rate across all MoE layers, from a server log.
# Each [EXPERT-OFFLOAD-CACHE] line carries cumulative hits=/misses= per layer;
# we keep the last value seen per layer and sum.
hitrate_from_log() {
  local log="$1"
  [ -f "$log" ] || { echo "n/a (no log)"; return; }
  grep -F "[EXPERT-OFFLOAD-CACHE]" "$log" 2>/dev/null | awk '
    {
      layer=""; h=""; m="";
      for (i=1;i<=NF;i++) {
        if ($i ~ /^layer=/)  { split($i,a,"="); layer=a[2] }
        if ($i ~ /^hits=/)   { split($i,a,"="); h=a[2] }
        if ($i ~ /^misses=/) { split($i,a,"="); m=a[2] }
      }
      if (layer!="" && h!="" && m!="") { H[layer]=h; M[layer]=m }
    }
    END {
      th=0; tm=0; n=0;
      for (l in H) { th+=H[l]; tm+=M[l]; n++ }
      if (th+tm > 0) printf "%.4f  (hits=%d misses=%d over %d layers)\n", th/(th+tm), th, tm, n;
      else print "n/a (no cache lines — is cache_policy_enabled + decode traffic present?)";
    }'
}

metric() { # metric <json> <key>
  local f="$1" k="$2"
  [ -f "$f" ] || { echo "n/a"; return; }
  if [ "$have_jq" = 1 ]; then
    jq -r --arg k "$k" 'if has($k) and (.[$k]!=null) then (.[$k]|tostring) else "n/a" end' "$f"
  else
    python3 - "$f" "$k" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); v=d.get(sys.argv[2])
    print("n/a" if v is None else v)
except Exception:
    print("n/a")
PY
  fi
}

BASE_JSON="$RESULTS_DIR/bench_baseline.json"
PREF_JSON="$RESULTS_DIR/bench_prefetch.json"
BASE_LOG="$RESULTS_DIR/serve_baseline.log"
PREF_LOG="$RESULTS_DIR/serve_prefetch.log"

printf '\n=== moe_offload_v5.0 A/B: baseline (offload+LRC) vs prefetch ===\n\n'
printf '%-26s | %-22s | %-22s\n' "metric" "baseline" "prefetch (+next-layer)"
printf -- '---------------------------+------------------------+------------------------\n'
printf '%-26s | %-22s | %-22s\n' "cache hit rate"        "$(hitrate_from_log "$BASE_LOG")" "$(hitrate_from_log "$PREF_LOG")"
printf '%-26s | %-22s | %-22s\n' "req throughput (req/s)" "$(metric "$BASE_JSON" request_throughput)"     "$(metric "$PREF_JSON" request_throughput)"
printf '%-26s | %-22s | %-22s\n' "output tok/s"           "$(metric "$BASE_JSON" output_throughput)"      "$(metric "$PREF_JSON" output_throughput)"
printf '%-26s | %-22s | %-22s\n' "total tok/s"            "$(metric "$BASE_JSON" total_token_throughput)" "$(metric "$PREF_JSON" total_token_throughput)"
printf '%-26s | %-22s | %-22s\n' "mean TTFT (ms)"         "$(metric "$BASE_JSON" mean_ttft_ms)"           "$(metric "$PREF_JSON" mean_ttft_ms)"
printf '%-26s | %-22s | %-22s\n' "median TTFT (ms)"       "$(metric "$BASE_JSON" median_ttft_ms)"         "$(metric "$PREF_JSON" median_ttft_ms)"
printf '%-26s | %-22s | %-22s\n' "mean TPOT (ms)"         "$(metric "$BASE_JSON" mean_tpot_ms)"           "$(metric "$PREF_JSON" mean_tpot_ms)"
printf '%-26s | %-22s | %-22s\n' "mean ITL (ms)"          "$(metric "$BASE_JSON" mean_itl_ms)"            "$(metric "$PREF_JSON" mean_itl_ms)"
printf '%-26s | %-22s | %-22s\n' "mean E2E (ms)"          "$(metric "$BASE_JSON" mean_e2el_ms)"           "$(metric "$PREF_JSON" mean_e2el_ms)"
printf '\nExpected if prefetch helps: higher hit rate, lower TPOT/ITL (decode),\n'
printf 'and equal-or-higher throughput. If the two columns are identical, prefetch\n'
printf 'likely did not run — confirm --enforce-eager and expert_prefetch_enabled=true.\n\n'

# Optional deeper per-layer / per-step breakdown (needs cache_debug_log_updates):
ANALYZER="$REPO_ROOT/tools/analyze_expert_cache_log.py"
if [ -f "$ANALYZER" ]; then
  echo "Deeper breakdown (per-layer/per-step) available via:"
  echo "  python3 $ANALYZER $PREF_LOG"
  echo "  (set expert_offload_config.cache_debug_log_updates=true for [UPDATE-W] detail;"
  echo "   note that verbose per-call logging will skew latency, so use a separate non-perf run.)"
fi
