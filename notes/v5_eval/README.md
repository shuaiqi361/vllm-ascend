# v5.0 Expert-Offload A/B Eval (ShareGPT, DeepSeek V4-Flash W8A8, Ascend A3)

Measures the effect of the **next-layer expert prefetch** (commit `d783c7a2`) on
top of the existing **expert offload + LRC cache** baseline. Same model, same
config, same workload — the **only** variable is `expert_prefetch_enabled`.

## What it compares

| Run | offload | LRC cache | next-layer prefetch |
|---|:--:|:--:|:--:|
| `baseline` | ✅ | ✅ | ❌ |
| `prefetch` | ✅ | ✅ | ✅ |

Expected if prefetch works: **higher cache hit rate**, **lower TPOT/ITL** (decode
latency), and equal-or-higher throughput.

## Critical preconditions (read before running)

1. **Eager mode is mandatory.** The prefetch is triggered from Python inside the
   MoE forward and is skipped during ACL-graph capture; under graph mode the
   decode path replays captured kernels and the trigger never fires. Both runs
   use `--enforce-eager` so prefetch actually executes and the comparison is
   fair. If the two result columns come out identical, this is the first thing
   to check.
2. **`num_device_experts` must exceed the per-step working set** (≈ `topk ×
   active seqs`) and be **less than** the model's total routed experts. With no
   slack slots, both LRC and prefetch are inert. Default is 32; tune in
   `common.env.sh` to your V4-Flash topk/expert count and `--max-num-seqs`.
3. **Hit-rate logging needs `cache_policy_enabled` (LRC) on** — it is, in both
   runs — and decode traffic. The numbers come from the server log's
   `[EXPERT-OFFLOAD-CACHE]` lines.

## Setup

Edit `common.env.sh` — at minimum set `MODEL` to your DeepSeek V4-Flash W8A8
weights, and `ASCEND_RT_VISIBLE_DEVICES` / `TP` to your A3 topology.

```bash
cd notes/v5_eval
chmod +x *.sh
./download_dataset.sh        # one-time ShareGPT fetch
```

## Run it (automated)

```bash
./run_ab.sh                  # baseline serve→bench→stop, then prefetch, then compare
```

Override anything inline:

```bash
MODEL=/data/ds-v4-flash-w8a8 TP=4 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
NUM_DEVICE_EXPERTS=48 REQUEST_RATE=4 NUM_PROMPTS=200 ./run_ab.sh
```

## Run it (manual, two terminals)

```bash
# terminal 1
./serve.sh baseline
# terminal 2 (once it's up)
./bench.sh baseline
# Ctrl-C terminal 1, then repeat with: ./serve.sh prefetch  /  ./bench.sh prefetch
./compare.sh
```

## Outputs (`results/`)

- `serve_{baseline,prefetch}.log` — server logs (contain `[EXPERT-OFFLOAD-CACHE]`)
- `bench_{baseline,prefetch}.json` — vllm bench serve metrics
- `compare.sh` prints the side-by-side table

## Reading the result

- **Cache hit rate**: cumulative `hits / (hits+misses)` summed across MoE layers
  from the last `[EXPERT-OFFLOAD-CACHE]` line per layer. Prefetch should raise it
  because predicted experts are already resident when the reactive path runs.
- **TPOT / ITL**: per-output-token decode latency — the clearest place prefetch
  pays off (it hides H2D copies behind layer L+1's attention).
- **Throughput / TTFT**: prefill-dominated metrics move less; prefetch is
  decode-oriented.

## Knobs (`common.env.sh`)

`MODEL`, `SERVED_NAME`, `ASCEND_RT_VISIBLE_DEVICES`, `TP`, `ENABLE_EP`, `HOST`,
`PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEM_UTIL`, `NUM_DEVICE_EXPERTS`,
`NUM_DEVICE_LAYERS`, `CACHE_LOG_INTERVAL`, `SHAREGPT_PATH`, `NUM_PROMPTS`,
`REQUEST_RATE`, `SHAREGPT_OUTPUT_LEN`, `RESULTS_DIR`.

## Caveats

- These scripts orchestrate runs; they were **not** executed on hardware here.
  Treat the first run as a smoke test — confirm the server starts, ShareGPT
  loads, and `[EXPERT-OFFLOAD-CACHE]` lines appear before trusting the deltas.
- `vllm bench serve --backend vllm` + `--dataset-name sharegpt` mirrors the
  repo's own `benchmarks/tests/serving-tests.json`. Flag names can drift across
  vllm versions; if `vllm bench serve` rejects an option, check `vllm bench
  serve --help` for the pinned version.
- Background prediction runs on a CPU thread doing a D2H + fp32 matmul; on a
  busy host the prefetch may land late and stall the forward (no time budget in
  v5.0). See `../moe-offload-v5.0-prefetch-study-guide.md` §5.7 / §8.
