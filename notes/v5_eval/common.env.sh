#!/usr/bin/env bash
# Shared configuration + helpers for the moe_offload_v5.0 expert-offload A/B eval.
#
# This file is SOURCED by serve.sh / bench.sh / run_ab.sh / compare.sh.
# Override any value by exporting it before you call those scripts, e.g.
#     MODEL=/data/ds-v4-flash-w8a8 TP=4 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ./run_ab.sh
#
# Target: vllm-ascend on Ascend NPU A3, DeepSeek V4-Flash W8A8.

# ----------------------------------------------------------------------------
# MUST EDIT — model location
# ----------------------------------------------------------------------------
# Local weights dir (recommended) or HF/ModelScope id of DeepSeek V4-Flash W8A8.
: "${MODEL:=/path/to/DeepSeek-V4-Flash-w8a8}"
# Name the client will send in requests; kept equal to MODEL by default so the
# server's --served-model-name matches what `vllm bench serve --model` sends.
: "${SERVED_NAME:=${MODEL}}"

# ----------------------------------------------------------------------------
# Hardware / parallelism (Ascend A3)
# ----------------------------------------------------------------------------
# NPUs visible to this run. For TP>1 list them comma-separated, e.g. "0,1,2,3".
: "${ASCEND_RT_VISIBLE_DEVICES:=0}"
: "${TP:=1}"                 # --tensor-parallel-size
: "${ENABLE_EP:=0}"          # 1 -> add --enable-expert-parallel (expert parallel)

# ----------------------------------------------------------------------------
# Server runtime
# ----------------------------------------------------------------------------
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${MAX_MODEL_LEN:=8192}"
# Keep the decode batch small so the per-step expert working set stays BELOW
# num_device_experts — that is the regime where the LRC cache and the prefetch
# actually do something. A huge batch routes to ~all experts and thrashes.
: "${MAX_NUM_SEQS:=16}"
: "${GPU_MEM_UTIL:=0.9}"

# ----------------------------------------------------------------------------
# Expert-offload config (identical for both runs EXCEPT expert_prefetch_enabled)
# ----------------------------------------------------------------------------
# Resident expert slots per MoE layer. MUST be > per-step working set (≈ topk x
# active seqs) AND < total routed experts, or both LRC and prefetch are inert.
: "${NUM_DEVICE_EXPERTS:=32}"
: "${NUM_DEVICE_LAYERS:=2}"      # prefill pool depth (round-robin buffers)
# How often each layer prints the cumulative [EXPERT-OFFLOAD-CACHE] hit-rate line
# (in cache calls per layer). 100 is cheap and does not perturb latency.
: "${CACHE_LOG_INTERVAL:=100}"

# ----------------------------------------------------------------------------
# Benchmark (ShareGPT)
# ----------------------------------------------------------------------------
: "${SHAREGPT_PATH:=${HOME}/.cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
: "${NUM_PROMPTS:=200}"
# Arrival rate. "inf" = all at once (max throughput). Set to 1/2/4 to read TTFT
# and ITL under steady load. Latency + throughput are reported either way.
: "${REQUEST_RATE:=inf}"
# Optional: force a fixed decode length to emphasize the decode phase (where the
# next-layer prefetch helps). Empty = use ShareGPT's natural output lengths.
: "${SHAREGPT_OUTPUT_LEN:=}"

# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RESULTS_DIR:=${_THIS_DIR}/results}"
# Repo root (for the cache-log analyzer tool), two levels up from notes/v5_eval.
: "${REPO_ROOT:=$(cd "${_THIS_DIR}/../.." && pwd)}"

# ----------------------------------------------------------------------------
# A3 / vllm-ascend environment knobs
# ----------------------------------------------------------------------------
export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
# Uncomment if pulling weights from ModelScope instead of a local path / HF:
# export VLLM_USE_MODELSCOPE=True
#
# Multi-node DP only (single-node serve does not need these):
# export HCCL_IF_IP=<this-node-ip>
# export GLOO_SOCKET_IFNAME=eth0
# export TP_SOCKET_IFNAME=eth0
# export HCCL_SOCKET_IFNAME=eth0

# ----------------------------------------------------------------------------
# Helper: build the --additional-config JSON for a given prefetch flag (0|1).
# Both features (offload + LRC cache policy) are always on; ONLY
# expert_prefetch_enabled changes between baseline and prefetch runs, so the
# difference in hit-rate / latency is attributable to prefetch alone.
# ----------------------------------------------------------------------------
build_additional_config() {
  local prefetch="$1" pf
  if [ "$prefetch" = "1" ]; then pf=true; else pf=false; fi
  printf '%s' \
    '{"enable_cpu_binding":true,"expert_offload_config":{' \
    "\"expert_offload\":true," \
    "\"num_device_experts\":${NUM_DEVICE_EXPERTS}," \
    "\"num_device_layers\":${NUM_DEVICE_LAYERS}," \
    "\"cache_policy_enabled\":true," \
    "\"cache_stats_log_interval\":${CACHE_LOG_INTERVAL}," \
    "\"expert_prefetch_enabled\":${pf}}}"
}

# ----------------------------------------------------------------------------
# Helper: launch vllm serve in the FOREGROUND for the given mode.
# mode = baseline | prefetch. Caller is responsible for redirection / tee.
#
# NOTE on --enforce-eager: the next-layer prefetch is triggered from Python
# inside the MoE forward (trigger_next_layer_prefetch) and is explicitly
# SKIPPED while an ACL graph is being captured. Under graph mode, decode steps
# replay captured kernels and that Python never runs, so prefetch would never
# fire. Eager mode is REQUIRED for a valid A/B (and we use it for BOTH runs so
# the comparison is apples-to-apples).
# ----------------------------------------------------------------------------
launch_vllm() {
  local mode="$1" prefetch add_cfg
  case "$mode" in
    baseline) prefetch=0 ;;
    prefetch) prefetch=1 ;;
    *) echo "launch_vllm: mode must be 'baseline' or 'prefetch'" >&2; return 2 ;;
  esac
  add_cfg="$(build_additional_config "$prefetch")"

  local ep_flag=()
  [ "$ENABLE_EP" = "1" ] && ep_flag=(--enable-expert-parallel)

  echo "[launch] mode=$mode  prefetch_enabled=$([ $prefetch = 1 ] && echo true || echo false)"
  echo "[launch] devices=$ASCEND_RT_VISIBLE_DEVICES  TP=$TP  EP=$ENABLE_EP"
  echo "[launch] additional-config: $add_cfg"

  vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    "${ep_flag[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --enforce-eager \
    --additional-config "$add_cfg"
}

# ----------------------------------------------------------------------------
# Helper: block until the server answers (or SERVER_TIMEOUT seconds elapse).
# ----------------------------------------------------------------------------
wait_for_server() {
  local url="http://${HOST}:${PORT}/v1/models"
  local timeout="${SERVER_TIMEOUT:-1800}" start now
  start=$(date +%s)
  echo "[wait] polling $url (timeout ${timeout}s) ..."
  until curl -fsS "$url" >/dev/null 2>&1; do
    now=$(date +%s)
    if [ $((now - start)) -gt "$timeout" ]; then
      echo "[wait] TIMEOUT after ${timeout}s — check the server log." >&2
      return 1
    fi
    sleep 5
  done
  echo "[wait] server is up."
}

# ----------------------------------------------------------------------------
# Helper: stop the server started for this PORT and wait for it to release.
# Conservative: targets vllm processes bound to THIS port only.
# ----------------------------------------------------------------------------
stop_server() {
  local pid="${1:-}"
  echo "[stop] shutting down server on port $PORT ..."
  [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
  # give workers time to release the NPUs
  local i
  for i in $(seq 1 30); do
    curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 || break
    sleep 2
  done
  pkill -9 -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
  sleep 3
  echo "[stop] done."
}
