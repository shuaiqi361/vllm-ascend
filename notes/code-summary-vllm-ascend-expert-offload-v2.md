# vLLM-Ascend Expert Offload — Code Study & Reference (`moe_offload_v2.0`) — **VERSION 2**

> ## 📌 About this V2 (read first)
> This is a **review pass over V1** (`code-summary-vllm-ascend-expert-offload.md`). It is **additive-only**: every V1 sentence is preserved verbatim. Corrections and additions are inserted as **callout blocks** so the V1↔V2 diff is unambiguous.
>
> **Callout legend:**
> - `⚠️ V2 CORRECTION [id]` — the immediately-preceding V1 statement is wrong or imprecise; the corrected version follows. (V1 text is intentionally left in place, not deleted.)
> - `➕ V2 ADDITION [id]` — net-new material (no V1 equivalent).
> - `🟡 V2 NUANCE [id]` — V1 is defensible but understated/incomplete; sharpening follows.
>
> **`(FACT)` vs `(ANALYSIS)` tags** are added to disputed lines: `(FACT)` = directly read from code with a `file:line`; `(ANALYSIS)` = derived/opinion (verify before relying on it).
>
> **Finding IDs** map to the review:
> - **N1** LRC policy is *provably inert* when `num_device_experts ≤ topk` (the reference config). **Headline.**
> - **N4** Storage-slice copy assumes *no NZ tile padding*, not just contiguity.
> - **A③** `clamp(min=0)` shortfall is *unreachable in decode* (defensive, but unguarded).
> - **A④** Prefill path patches **three** fields and has **no `try/finally`** (real robustness bug).
> - **A⑤** Router-score term is silently dropped under ACL-graph capture.
> - **A①** The prefetch "no-op" verdict is right, but V1's *reasoning* for it is imprecise.
> - **(memory/throughput)** `~29 GB`/"doubles host memory" numbers refined; `ndev=topk` throughput consequence made explicit.
>
> **V2 changelog (where each edit lands):** §1 (memory numbers), §3 (throughput at `ndev=topk`), §4.5 (host memory), §6 (prefetch FAQ, ACL-graph FAQ, new "is the policy doing anything?" FAQ, clamp note), §9 (invariant 6 corrected; new invariants 9–10).

**Source:** `https://github.com/LookAround0301/vllm-ascend/tree/moe_offload_v2.0`
**Purpose of this document:** a learning-oriented, reference-grade explanation of the expert-offload feature — what it is, where it fires in the MoE forward, the exact paging/eviction rules, the prefetch story, the engineering tricks, and (importantly) the concepts that are *easy to get wrong*. A companion file, `expert-offload-optimization-ideas.md`, holds the optimization hypotheses; this file is the "how it works," that file is the "how it could be better."

**How to read this (two audiences):**
- *Humans:* read §1→§6 top to bottom; §6 (Concepts & Common Confusions) is the part that resolves the questions almost everyone hits.
- *Claude agents / tooling:* start at §0 (Navigation), use §7 (Glossary) and §9 (Invariants) as ground truth, and treat every `file:line` reference as a jump target. Line numbers are from the uploaded snapshot and may drift; the symbol names are stable anchors.

**Files covered (all read in full unless noted):** `expert_offload_manager.py`, `lrc_policy.py`, `utils.py`, `__init__.py` (offload package); `fused_moe.py`, `moe_comm_method.py`, `moe_mlp.py`, `token_dispatcher.py`, `experts_selector.py`, `prepare_finalize.py`, `moe_runtime_args.py`, `moe_stage_contracts.py`, `moe_stage_params.py`, `comm_utils.py` (fused-MoE pipeline); `ascend_config.py` (config schema); offload slices of `model_runner_v1.py`; `CMakeLists.txt` / `setup.py` (build flags). The forward call site and config schema, previously inferred, are now **confirmed** from `fused_moe.py` / `moe_comm_method.py` / `ascend_config.py`.

**Reference workload (used throughout):** `vllm serve DeepSeek-V2-Lite` with `--enforce_eager`, `--max-num-seqs 1`, `tp=1 dp=1 --enable-expert-parallel`, and `expert_offload_config = {expert_offload: true, num_device_experts: 6, cache_policy_enabled: true}`. Model facts: 64 routed experts, `top_k=6`, 2 shared experts, 27 layers (layer 0 dense, 26 MoE). Per-expert size ≈ 16.5 MB bf16 (`w13 [2048,2816]` + `w2 [1408,2048]`).

> ⚠️ **V2 CORRECTION [N1] — the reference workload is a degenerate corner for the cache.** It sets `num_device_experts (6) == top_k (6)`. As proved in §9 invariant 9, at `ndev ≤ topk` the LRC policy is **inert** (it never makes a discriminating eviction choice) and the resident set is *forced* to equal the current step's `needed` set every step. So benchmarking "the cache policy" with this exact config measures **nothing about the policy** — `cache_policy_enabled: true` and `false` produce bit-identical residency. To exercise the cache at all, use `num_device_experts > top_k` (e.g. 12/18/32). This is the single most important caveat in V2.

---

## 0. Navigation map (for quick jumps)

| Concern | Where |
|---|---|
| Manager lifecycle (CPU buffers, device slots, both paths, policy) | `expert_offload_manager.py` (`ExpertOffloadManager`, ~`:18-803`) |
| Decode pager (the demand-paging hot path) | `expert_offload_manager.py` `update_weights`→`_update_weights` (~`:534-716`) |
| Prefill bulk-load pool | `expert_offload_manager.py` `create_prefill_pool`/`_prefill_load_layer` (~`:341-528`) |
| Eviction policy (hotness, victim choice) | `lrc_policy.py` `LRCExpertCachePolicy` (`observe` ~`:75-117`, `choose_victim` ~`:119-133`, `hotness` ~`:135-143`) |
| Pre-`__init__` device-map shrink + initial log2phy | `utils.py` `init_expert_offload_config`, `init_log2phy_for_offload` |
| Forward hook (calls `update_weights`, selects path) | `fused_moe.py` (~`:186-191`, prefill patch ~`:230-305`) |
| `log2phy` applied to routed ids (logical→slot) | `moe_comm_method.py` `MoECommMethod.fused_experts` (`routed_topk_ids = log2phy[topk_ids]`) |
| Weight-loader interception (CPU mirror + cold-skip) | `fused_moe.py` `_wrap_weight_loader_for_offload` (~`:552-596`) |
| Router: softmax + top-k | `experts_selector.py` `_native_select_experts` (the `scoring_func=="softmax"` branch) |
| Token expand + sort-by-expert (produces `group_list`) | `token_dispatcher.py` (`npu_moe_init_routing`, `active_num = num_tokens*top_k`) |
| Expert FFN compute (gmm1→SwiGLU→gmm2) | `moe_mlp.py` `unified_apply_mlp` (~`:215-237`) |
| Config schema + defaults | `ascend_config.py` `ExpertOffloadConfig` (~`:613-660`) |
| Model-runner wiring (init, register layers, profile skip) | `model_runner_v1.py` (`:300-303`, `_register_offload_layers`, `_skip_prefill`) |
| Batch-memcpy build flag | `CMakeLists.txt` (`CANN_MEMCPY_BATCH_ASYNC`), `setup.py:342-345` |

---

## 1. What the feature is

In one line: **demand paging for MoE experts.** The full set of routed-expert weights lives in pinned CPU RAM; the NPU holds only a small pool of `num_device_experts` slots per layer; experts are copied into slots on demand as routing decisions reveal which are needed.

**Why it exists.** A sparse MoE model needs *all* expert weights resident to serve *any* token, even though each token activates only `top_k`. For DeepSeek-V2-Lite the routed experts are ~90% of the model (64 × 26 layers × 16.5 MB ≈ 29 GB) and most sit idle per token. Offload trades scarce HBM for host→device (H2D) copy bandwidth: resident weight footprint drops from ~29 GB to `num_device_experts × 26 × 16.5 MB` (≈ 2.7 GB at 6 slots), and the freed HBM becomes KV-cache / context length. Activated parameters per token are unchanged; only *resident* parameters drop.

> 🟡 **V2 NUANCE [memory] — the "≈ 29 GB" figure is ~8% high; "2.7 GB" is fine.** `64 × 26 × 16.5 MB = 27,456 MB ≈ 26.8 GiB`, not 29. `6 × 26 × 16.5 MB ≈ 2,574 MB ≈ 2.5 GiB`. The argument is unchanged; just don't quote 29 GB as exact. `(ANALYSIS, arithmetic)`

**Concrete features (and owner):**
- Full CPU mirror, pinned + pre-transposed to device layout, NZ-cast for W8A8 — `create_weights` (~`:94-118`), `load_w13`/`load_w2` (~`:187-211`), NZ recast (~`:156-182`).
- Shrunken device weight tensor (only `num_device_experts` slots; cold experts never loaded to device) — `utils.init_expert_offload_config` + the weight-loader wrapper's cold-skip (`fused_moe.py:591-592`).

> ➕ **V2 ADDITION [completeness] — scale/offset params are cold-skipped too.** V1 cites only the weight cold-skip at `fused_moe.py:591-592`. For W8A8, the wrapper *also* returns `None` for cold `weight_scale`/`weight_offset` experts (`fused_moe.py:575-576, 582-583`), after mirroring them to CPU via `_add_pending_scale`. `(FACT)`

- Decode demand paging — `update_weights`/`_update_weights`.
- Prefill bulk-load pool (`num_device_layers` tensors each sized for *all* experts, round-robin) — `create_prefill_pool`/`_prefill_load_layer`.
- Pluggable eviction policy (LRC hotness) — `lrc_policy.py`.
- W8A8 support (per-expert scale/offset, derived fp32 scale, pending-drain for load-order races).
- Instrumentation (`[UPDATE-W]`, `[EXPERT-OFFLOAD-CACHE]` hit-rate) gated by `cache_debug_log_updates` / `cache_stats_log_interval`.
- Build-time batched memcpy capability (`aclrtMemcpyBatchAsync`, CANN 8.5+) — detected but **not yet used by the Python path** (see optimization C1).

---

## 2. The MoE forward pipeline (and where offload hooks in)

A routed-MoE layer runs six stages. Offload touches exactly two of them (★).

1. **Route** — `select_experts` → `topk_weights`, `topk_ids` (`fused_moe.py:166`). Softmax + top-k live here, in `experts_selector.py`.
2. **★ Offload hook** — `update_weights(layer, topk_ids, log2phy, topk_weights)` pages experts and rewrites `log2phy`; also decides path (decode vs prefill pool) (`fused_moe.py:186-191`).
3. **Prepare** — layout / optional activation quant (`prepare_finalize.py`).
4. **★ Dispatch (apply log2phy + sort)** — `routed_topk_ids = log2phy[topk_ids].clamp(min=0)` (`moe_comm_method.py`), then `npu_moe_init_routing` expands tokens and groups by expert, emitting `group_list` (`token_dispatcher.py`).

> ⚠️ **V2 CORRECTION/NUANCE [A③] — what `.clamp(min=0)` actually does, and its hidden failure mode.** Confirmed at `moe_comm_method.py:142` (comment: `# safety: unmapped -> 0`). An unmapped expert (`log2phy = -1`) is silently rerouted to **physical slot 0** — i.e. computed against *whatever expert occupies slot 0*, a wrong-output, no-crash event. **Good news (V2):** in the decode path this is **unreachable** — `num_tokens ≤ offload_threshold = ndev//topk` ⇒ `|needed| ≤ num_tokens·topk ≤ ndev`, and `choose_victim` returns `None` only when `|needed| > ndev`, so no `-1` survives. **The risk is future-facing:** any change to the path-selection math (e.g. opt-idea A1's `n_distinct` test or G1's split) that mis-bounds capacity would convert into a *silent* miscompute here, because nothing asserts that no `-1` remains. Treat the clamp as defensive-but-unguarded. `(ANALYSIS, proof in body)`

5. **Expert compute** — `npu_grouped_matmul` (gmm1, gate+up) → `npu_swiglu` → `npu_grouped_matmul` (gmm2, down) (`moe_mlp.py:215-237`).
6. **Combine + finalize** — `npu_moe_token_unpermute` scatters expert outputs back and applies `topk_weights`; finalize reduces across ranks if any.

**When it runs:** every MoE layer, every forward step. Paging is **reactive and on the critical path** — layer L's needed experts are unknown until L's router runs, and the load is host-blocked (`load_stream.synchronize()`, ~`:716`) before compute. No cross-layer look-ahead (see §6 "prefetch").

**Comm method in the reference run:** with `tp=1 dp=1`, effective EP size is 1, so only `AllGatherCommImpl` is set up (`moe_comm_method.setup_moe_comm_method` else-branch). Dispatch is `TokenDispatcherWithAllGather`. The MC2 fused dispatch-compute-combine paths are not exercised here.

**Typed contracts:** `moe_stage_contracts.py` / `moe_stage_params.py` / `moe_runtime_args.py` are plumbing — frozen dataclasses carrying data between stages. The only offload-relevant field is `MoERoutingParams.log2phy`.

---

## 3. How offload works end to end (three phases)

### Phase 1 — load time (the setup that makes it cheap)
- **Shrink the device tensor.** Before `super().__init__()`, `init_expert_offload_config` builds `expert_map_offload` (size = total experts, only first `num_device_experts` valid). The upstream layer allocates `w13_weight`/`w2_weight` with only `num_device_experts` slots — no peak-memory spike from materializing all experts on device.
- **Divert experts to CPU.** `_wrap_weight_loader_for_offload` (`fused_moe.py:552-596`) intercepts every expert weight: stores a pinned, transposed (and NZ-cast for W8A8) CPU copy via `load_w13`/`load_w2`, and **if `expert_id >= num_device_experts`, returns `None` so the original loader never writes it to device** (`:591-592`). Experts `0..ndev-1` go to both CPU and device.
- **Result:** full expert set pinned in host RAM; device holds `ndev` slots; `log2phy` initialized identity for `0..ndev-1`, `-1` elsewhere (`init_log2phy_for_offload`).

### Phase 2 — decode demand paging (`num_tokens <= offload_threshold`)
`_update_weights` (~`:565-716`), running on `load_stream` (~`:613`):
1. Stage `topk_ids` and current `log2phy` to pinned CPU (~`:578-579`).
2. `policy.observe(...)` → `needed`; `slot_owner = {slot: eid}`; `need_to_load = needed − resident` (~`:614-632`). **Only misses are loaded; hits cost nothing.**
3. Per miss: `choose_victim` picks coldest evictable resident → overwrite that slot via flat storage-slice H2D copy of `w13`/`w2`(+scales) (~`:658-705`); update `log2phy` in place (old→`-1`, new→slot) (~`:709-711`).
4. `load_stream.synchronize()` (~`:716`); copy `log2phy` back to device (~`:603`).

> ⚠️ **V2 CORRECTION [N1] — step 3 ("`choose_victim` picks coldest evictable resident") does NOT discriminate at `ndev = topk`.** The choice only matters when there are strictly more eligible victims than misses to place, which (single-token decode) happens iff `|needed| < ndev`, i.e. `ndev > topk`. At `ndev = topk` (reference run), `#misses == #eligible-victims` every step, so *all* non-needed residents are evicted regardless of hotness — the resident set is forced to exactly `needed`, and `observe`/`hotness`/`choose_victim` compute a result that is **never used to decide anything**. See §9 invariant 9 for the proof. `(ANALYSIS, proof)`
>
> *Free corollary for the reference config:* `cache_policy_enabled=false` yields identical eviction outcomes with less host overhead (it skips the EMA/router/freq bookkeeping in `lrc_policy.observe`).

### Phase 3 — prefill bulk-load pool (`num_tokens > offload_threshold`)
`_prefill_load_layer` (~`:464-524`) bulk-copies *all* experts of the current layer into `pool_slot = layer_idx % num_device_layers`. The forward swaps in the pool tensors, **temporarily patches the local-expert count to the full total** so `group_list` is sized right, and uses an **identity `log2phy`** (all experts present), restoring after compute (`fused_moe.py:230-305`).

> ⚠️ **V2 CORRECTION [A④] — the prefill patch touches THREE fields and is NOT exception-safe.** V1 says it patches "the local-expert count." Actually it patches three (`fused_moe.py:230-237`, restored `:302-305`): `layer.moe_config.num_local_experts`, `layer.local_num_experts`, **and** `moe_comm_method.token_dispatcher.num_experts_local` — the last living on the **shared comm-method/token-dispatcher singleton**, not the layer. Worse, the patch→compute→restore is **not** wrapped in `try/finally`: if `moe_comm_method.fused_experts` (`:274`) raises, the restore is skipped and all three fields stay set to `ntotal`, corrupting every subsequent decode step on that layer. This is a concrete robustness bug, not just the (largely theoretical) reentrancy concern in §9 invariant 6. `(FACT)`

### The glue: `log2phy`
A `[global_num_experts]` int table: `log2phy[eid] = slot` (or `-1`). The pager writes it; the kernel reads it once (`routed_topk_ids = log2phy[topk_ids]`) so the matmul only ever sees physical slot indices. This single indirection is why paging is invisible to compute **and why no device-to-device weight movement is ever needed** (§6).

### The two paths, and the threshold
`offload_threshold = num_device_experts // topk` (`:38`). It's a capacity guarantee: worst case `num_tokens` tokens touch `num_tokens × topk` distinct experts, which fits in `ndev` slots iff `num_tokens <= ndev/topk`. Above that, paging would thrash → load everything.
> **Reference-run consequence:** `num_device_experts=6`, `topk=6` ⇒ **threshold = 1**. Only single-token decode uses the cache; *every* step with ≥2 tokens (all prefill, all chunked-prefill chunks, batched decode) uses the heavy bulk-load pool. If you benchmark "the cache" with this config you are mostly measuring "the pool." This is optimization idea A1.

> ➕ **V2 ADDITION [N1/throughput] — quantify what `ndev=topk=6` costs at runtime.** At zero cache margin, decode streams `topk` experts from CPU **every step**: `6 × 16.5 MB × 26 layers ≈ 2.6 GB/token` over PCIe just for paging. If step-to-step routing overlap is ~50%, ≈ 3 misses/layer ⇒ `3 × 16.5 MB × 26 ≈ 1.3 GB/token` ⇒ at ~24 GB/s ≈ **~54 ms/token ⇒ ~18 tok/s ceiling**, before any compute. And any `--max-num-seqs > 1` decode (≥2 tokens > threshold 1) falls into the **prefill pool**, which reloads *all 64 experts/layer* = `64 × 16.5 MB × 26 ≈ 27 GB/token` ⇒ ~1 s/token (unusable). Takeaway: the reference config is a **worst case**, not a representative demo; raising `num_device_experts` (opt-idea A2) is the dominant lever and a prerequisite to measuring anything else. `(ANALYSIS, order-of-magnitude)`

---

## 4. Advanced / borrowable patterns (with costs)

1. **Storage-slice memcpy (the best idea).** `layer.w13_weight.data.untyped_storage()[slot*N:(slot+1)*N].copy_(cpu_expert.untyped_storage())` (~`:681-686`). Slot `s` is the contiguous byte range `[s*N,(s+1)*N]` of the contiguous device tensor, so "update expert in slot S" is a flat DMA, and (for W8A8) the pre-NZ-cast CPU mirror means **no format conversion per copy**. *Cost:* assumes byte-identical contiguous layout; a silent miscompute if upstream repacks. No assertion at the copy site (optimization F3).

> ⚠️ **V2 CORRECTION [N4] — the real hidden assumption is "no NZ tile padding," and it is unchecked.** `N = w13_expert_size_bytes` is computed from the **logical** tensor (`nelement × element_size`, `:159-160`), but for W8A8 the device/CPU buffers are `FRACTAL_NZ` (`:171-174`). NZ casting **pads** dims to tile boundaries (16×16 bf16 / 32×16 int8). The slice `[s*N:(s+1)*N]` is byte-correct **only if** per-expert NZ storage equals the logical `nelement × element_size` — i.e. **only if there is no padding**. For DeepSeek-V2-Lite this holds (2048/2816/1408 are tile-aligned), so the code works — *silently*. A model with non-tile-aligned expert dims would get misaligned per-expert copies → garbage weights → wrong output, no crash. The correct guard (cheaper than F3's "contiguity") is a one-time setup assertion: `dev_tensor.untyped_storage().nbytes() == num_experts × expert_size_bytes`. `(ANALYSIS, mechanism)`

2. **LRC retention as caching.** `hotness = recent_weight·freq + ema_weight·ema + router_weight·router_score − age_weight·age` (`lrc_policy.py:135-143`). Exploits local routing consistency so retained hot experts → high hit rate → tiny per-step deltas. *Cost:* pure-Python `tolist()` + per-row loops on the critical path (optimizations D1/D2); only helps the decode path.

> 🟡 **V2 NUANCE [N1] — "LRC retention as caching" only earns its keep at `ndev > topk`.** At `ndev = topk` retention is impossible (zero spare slots; §9 inv. 9), so this pattern contributes nothing in the reference config — the "high hit rate / tiny per-step deltas" story requires spare capacity that the reference config does not have. `(ANALYSIS)`

3. **Threshold-routed dual path.** Clean small-batch/large-batch split. *Cost:* integer-floor threshold collapses to 1 at `topk==ndev` (A1).
4. **Host-function deferral for ACL-graph capture.** Under capture, the data-dependent paging runs as a host callback node (`_launch_host_func`, ~`:594-601`); inline otherwise. *Cost:* still serializes against the `synchronize`; dormant under `--enforce_eager`. (See §6 ACL graph.)
5. **Pinned, pre-transposed, NZ-matched CPU mirror.** Every H2D is a no-transform DMA. *Cost:* doubles host memory (~29 GB bf16); two-phase pending-drain (`_pending_weights`/`_pending_scales`) to handle load-order races.

> ⚠️ **V2 CORRECTION [memory] — "doubles host memory" is misleading.** The pinned CPU mirror holds the full expert set **once** (~27 GiB bf16 / ~13.4 GiB W8A8) — that *is* the full set, not a doubling. A normal (non-offload) load would transiently need the same bytes anyway; offload's distinctive cost is that the mirror is **pinned/non-pageable** and persists, while the *device* footprint shrinks to `ndev` slots. The only true transient "double" is the per-expert `loaded_weight.cpu().clone()` during loading (`:193`) and the per-layer `torch.stack(...).to('npu')` used for NZ recast (`:170-174`), both short-lived. `(FACT/ANALYSIS)`

---

## 5. Configuration & tuning rules (utilize offload vs. available HBM)

Offload frees the HBM the full expert set would occupy; tuning is deciding how to spend it. Knobs live in `ExpertOffloadConfig` (`ascend_config.py:~613-660`).

- **`num_device_experts` (primary dial; default 32, reference run sets 6).** Floor: must be `≥ topk` (else single-token decode can't fit and falls to the pool). Make it a **multiple of `topk`** to widen the cache regime (raises `offload_threshold`). Tune up to the **hit-rate knee** (from `[EXPERT-OFFLOAD-CACHE]` logs). Per-step GEMM cost stays tied to *active* experts, not pool size (empty groups skipped, §6); the cost of a large value is HBM (× 26 layers), not throughput — modulo the empty-group caveat in optimization A2.

> ⚠️ **V2 CORRECTION [N1] — the floor is `> topk`, not `≥ topk`.** At `ndev == topk` single-token decode *fits* (it does not fall to the pool), but the cache has **zero spare slots**: residency is forced to `needed` every step and the policy is inert (§9 inv. 9). To get *any* caching/retention benefit you need `ndev > topk` (strictly). Practically: set `ndev ≥ 2·topk` to have a meaningful hot region. `(ANALYSIS)`

> ➕ **V2 ADDITION [N6] — `num_device_layers=2` (default) currently wastes HBM.** The prefill pool is host-blocked (`load_stream.synchronize()` `:524`) with no overlap, and the round-robin `pool_slot = layer_idx % ndl` provides **no reuse** (a slot is overwritten by a different layer before the same layer revisits it — see opt-idea B1's V2 rebuttal). So `ndl=2` costs a full extra expert set (~1.1 GB) for **zero** current benefit; it is speculative infrastructure for the unimplemented double-buffer (opt-idea B2). Until B2 lands, `ndl=1` is strictly better. `(ANALYSIS)`

- **`num_device_layers` (prefill pool depth; default 2).** Keep small today: each pool layer costs a full expert set (~1.1 GB), and the pool currently reloads every layer every step (optimizations B1/B2), so extra depth buys little until those land.
- **`--gpu-memory-utilization`.** Set high; offload's savings flow to KV cache. If not context-bound, spend freed HBM on `num_device_experts` instead.
- **`--max-num-seqs`.** Fights the decode cache: more concurrent seqs → more tokens/step → more steps pushed above `offload_threshold` into the pool. If you raise it, raise `num_device_experts` in step.
- **`--max-num-batched-tokens` (chunked prefill).** Larger chunks are cheaper *per token* — the pool reloads all experts per layer per pass regardless of chunk size, so fewer/bigger chunks = fewer total reloads. The reference run (133k as one chunk) is already optimal here.
- **W8A8.** Halves per-expert bytes → roughly doubles the slot budget for the same HBM; usually the best single lever.
- **Host pinned RAM (not a VRAM knob).** The full expert set is pinned (~29 GB bf16 / ~14.5 GB W8A8). Non-pageable; must physically fit or load fails.

---

## 6. Key concepts & common confusions (FAQ)

*This section captures the questions that reliably cause confusion. Read it even if you skim the rest.*

**Q. What does "demand paging" mean?** Two OS terms. *Paging* = keep a small working set in fast memory (NPU slots) and the rest in slow storage (CPU RAM), tracked by a page table (`log2phy`). *Demand* = bring a chunk in only when first accessed (a routing "miss" = a page fault), evicting a cold chunk to make room. The decode path is true demand paging (reactive). The prefill path is the opposite — *eager* bulk loading. Mapping: CPU RAM↔disk, NPU slot↔frame, expert↔page, `log2phy`↔page table, miss↔page fault, `choose_victim`↔replacement policy.

**Q. Does `--enforce_eager` disable offload?** No. Two unrelated "eager"s. *Eager loading* (a caching strategy, my metaphor for prefill) ≠ *`--enforce_eager`* (PyTorch execution mode: run ops immediately vs. capture into an ACL graph). Offload is gated only on `expert_offload: true` (`model_runner_v1.py:300-303`), never on graph mode. `--enforce_eager` only flips one branch inside `update_weights`: `if _EXTRA_CTX.capturing: _launch_host_func(...) else: _update_weights(...)` (~`:594-601`). In your run, the `else` (inline, synchronous) branch runs; the host-function path is dormant. Offload works fully either way.

**Q. How do the two ACL-graph modes interact with offload?** *Eager:* each op dispatched immediately; flexible; launch-overhead-bound for decode. *Graph (capture+replay):* record the kernel sequence once, replay as a unit — kills per-op launch overhead, the big win for launch-bound decode. A graph can't contain host-side control flow, so the data-dependent paging is recorded as a **host-function node** (a CPU callback in the timeline). It's safe for replay because the node is **re-executed every replay** (it recomputes which experts to page) and it operates only on **fixed-address buffers** (`topk_ids_h`, `log2phy_h`, the device weight tensor — all allocated once, contents refreshed in place). The graph captures *where* and *in what order*, never *what value* — so nothing goes stale. For perf testing: eager = transparent diagnostic baseline; graph = shipping number; measure both, and verify graph-mode outputs match eager token-for-token before trusting graph numbers (the prefill path mutates and restores `num_local_experts`, optimization F2).

> ⚠️ **V2 CORRECTION [A⑤] — under ACL-graph capture, the LRC router-score term is silently dropped.** `update_weights` populates `topk_weights_h` only when `not _EXTRA_CTX.capturing` (`:572-573`). So under capture, `observe(router_scores=None)` and the `router_weight · router_score` term of `hotness` is never updated — the policy degrades to `freq + ema − age` only. This changes **eviction decisions** (hence hit rate), but **not outputs**, so V1's advice to "verify graph-mode outputs match eager token-for-token" will *not* catch it. If you A/B the router term (opt-idea D3) you must do it in eager mode; graph mode can't see it. `(FACT)`

> ➕ **V2 ADDITION [N1] — Q. When does the cache policy actually do anything?** Only when `num_device_experts > top_k` (single-token decode). `choose_victim` discriminates by hotness only if, for some miss, there are strictly more eligible victims than misses to place — which (proof in §9 inv. 9) requires `|needed| < ndev`, i.e. `ndev > topk`. At `ndev = topk` (reference config) the resident set is *forced* to equal `needed` every step and the policy's output is unused. **Practical consequences:** (a) you cannot measure or tune any policy parameter (`cache_recent_weight`, `cache_ema_*`, `cache_router_weight`, `cache_age_weight`) in the reference config — they have no effect; (b) every policy-side optimization (opt-ideas D1/D2/D3/D4, the hot-cache half of G1, and E1b) is dormant until A2 raises `ndev` above `topk`. A2 is therefore a **hard prerequisite**, not a tuning nicety.

**Q. Where is the softmax? It's not in `moe_mlp.py`.** Right — selection and computation are separate stages. The softmax + top-k are upstream in `experts_selector.py` (`router_logits.softmax(dim=-1)` then `torch.topk(k=top_k)`), producing `topk_ids`/`topk_weights`. `moe_mlp.py` only computes already-chosen experts; it's routing-blind. The `topk_weights` (softmax probs) are applied at the *other* end, in combine, as the weighted sum of expert outputs.

**Q. How does the kernel know which weights belong to which token's experts?** Three steps: (A) `log2phy[topk_ids]` remaps logical expert ids → physical slot indices; (B) dispatch sorts the (expanded) token rows by slot, emitting `group_list` (token count per slot); (C) `npu_grouped_matmul` walks slabs of the contiguous `[ndev, K, N]` weight tensor **positionally**, applying slab `i`'s weights to that slab's `group_list[i]` contiguous token rows. No per-token weight lookup — grouping + positional indexing does it.

**Q. The expert FFN — what are `w13`/`w2`, and is the SwiGLU formula right?** The operation is `out = down( silu(gate·x) ⊙ up·x )` — silu of the gate half, elementwise-times the up half, then down-project. Code specifics that trip people up: (1) this code stores weights **transposed to `[in,out]`** and computes `x @ W` (row-vector), not `W·x` — hence the `.t()` at load. (2) **`w13` fuses gate (`w1`) + up (`w3`)** into one `[H, 2I]` tensor so gmm1 produces both halves in one matmul; `npu_swiglu` splits and does silu(first)*second. (3) Three meanings of "w1" collide: shard-id `w1`=gate; the local var `w1` in `moe_mlp.py` = the *fused* gate+up (=`w13`); and in casual math `w1` often means gate alone. (4) For MoE *routed* experts the intermediate is **narrower** than hidden (`I=1408 < H=2048`) — a bottleneck, not the classic 4× expansion (that's the dense FFN). Code-accurate: `gate_up = x @ w13 [2I]`; `y = silu(gate_up[:I]) * gate_up[I:]`; `out = y @ w2 [H]`.

**Q. How does this FFN relate to GDN (Gated DeltaNet)?** Different transformer slots. The FFN/MoE is the **channel mixer** (position-wise; no cross-token interaction). GDN is a **token mixer** — a linear-attention layer that replaces *self-attention*, maintaining a fixed-size recurrent state instead of a growing KV cache. They coexist (e.g. Qwen3-Next has GDN token-mixing *and* a SwiGLU-MoE FFN). The two "gates" gate different axes: SwiGLU's gate selects *features within a token's channels*; GDN's gate controls *how fast the recurrent memory of past tokens decays*. GDN does not touch the expert FFN.

**Q. With `num_seq=1` decode, how many tokens does an active expert process?** Exactly **one**. The single token is replicated into `top_k` copies (token expansion), one per chosen expert. So 6 routed experts each process 1 token; 58 process 0; the 2 shared experts also process the 1 token. Critically this makes each expert "matmul" a **GEMV** (`[1,H]@[H,2I]`), not a GEMM — memory-bound, dominated by *reading the weight*, not arithmetic. This is the root cause of decode being bandwidth-bound and of offload's copy cost being hard to hide. Larger batches raise per-expert token counts → real GEMMs → better amortization.

**Q. Why does `group_list` have varied/zero counts if there's only one token?** Because `group_list` partitions the **expanded** rows (`num_tokens × top_k`), not tokens. Batch-1: 6 expanded rows → `group_list` is mostly zeros with six `1`s. Varied counts like `[3,0,5,...]` only happen in the many-token (prefill) regime. The sum is always `num_tokens × top_k`, never `num_tokens`.

**Q. Are the 6-slot buffers shared across the 26 layers, or one per layer?** One per layer — **26 independent `[ndev, K, N]` tensors**, on each `AscendFusedMoE` instance. The LRC policy also keeps a separate `LRCLayerState` per layer. It must be per-layer: expert #37 in layer 5 ≠ expert #37 in layer 6 (different weights), and routing is independent per layer. Per-layer buffers are what let a slot's contents *persist and be reused across decode steps* (the cache). The prefill pool is different — `num_device_layers` big buffers time-shared across layers round-robin.

**Q. On a miss with 4 cached + 2 missing, does it overlap compute with the transfer?** No. It loads only the 2 misses, then `load_stream.synchronize()` host-blocks, then computes all 6 in one grouped matmul. No overlap. And overlap would barely help in decode anyway: the 2-miss transfer (~ms over PCIe) dwarfs the cached-expert compute (~tens of µs GEMV) by ~20×. The lever is *not missing* (cache margin) or *prefetching ahead with prediction*, not intra-layer overlap (optimization E2).

> 🟡 **V2 NUANCE [N1] — "4 cached + 2 missing" cannot happen at `ndev = topk`.** With `ndev = topk = 6` and a single token, every step needs 6 and holds 6, so there are **no spare slots** to carry 4 over — the "4 cached" only exists when `ndev > topk` (here you'd need `ndev ≥ 10` to plausibly carry 4 hits + 2 new). The scenario in this Q implicitly assumes a *non-reference* config. `(ANALYSIS)`

**Q. Is `num_device_experts > topk` configurable, and does a cache hit cost a device-to-device copy?** Yes, it's the same `num_device_experts` knob (default 32 > topk); there is **no separate cache-size parameter** (working buffer and cache are one flat pool — optimization G1). A cache hit costs **nothing**: the expert is already in its slot from a prior step, `log2phy` already points to it, the matmul reads it in place. **There is no device-to-device copy anywhere**; the only weight movement is H2D on a miss. Experts never migrate between slots (overwritten in place by the next H2D). `log2phy` is exactly what removes the need for D2D/compaction.

**Q. Grouped GEMM needs contiguous memory — must the active experts be made contiguous?** No. The weight *tensor* `[ndev, K, N]` is contiguous and each slab is a contiguous sub-block — that satisfies the kernel. The *active subset* may be scattered across non-adjacent slabs (e.g. {0,5,6,20,30,31}); this is expressed by a **`group_list` of length `ndev`, zero-padded** at inactive slots. What gets gathered into contiguous order is the **tokens** (cheap, a few KB), sorted by slot to match where the weights already sit — *not* the weights (expensive, ~100 MB). Gather the cheap thing, leave the expensive thing in place. Caveat: the kernel walks all `ndev` group positions; empty groups cost no bandwidth but may carry per-group overhead at very large `ndev` (optimization A2).

**Q. Is there predictive prefetch? Can the LRC stats drive it?** No predictive prefetch exists — only reactive retention. The LRC table *does* track all experts per layer regardless of residency, and eviction never clears an expert's scores, so the signal is available. But naive reuse is a **no-op**: each layer's slots already hold the top-`ndev` experts by LRC hotness, so "prefetch the hot ones" = "they're already resident." Prefetch can only help the *misses* (not-yet-resident experts about to be needed), which same-layer LRC cannot predict by construction; that needs cross-layer routing correlation or a fast drift detector (optimization E1).

> ⚠️ **V2 CORRECTION [A①] — the verdict (naive same-layer prefetch ≈ no-op) is RIGHT, but the stated reason is imprecise.** Two fixes:
> 1. **"slots already hold the top-`ndev` by hotness"** is only an *approximation*, and only for `ndev > topk`. The eviction rule *maintains* this property (it evicts the **coldest**, `lrc_policy.py:128`), so a high-hotness expert can rarely be non-resident in steady state. At `ndev = topk` the statement is *vacuous* — residency is forced to `needed`, hotness is irrelevant (§9 inv. 9).
> 2. **The precise reason prefetch is hard:** the misses are exactly the experts whose hotness was *low when they were evicted* (that's *why* they were evicted) and is *now rising*. A snapshot of current hotness rank therefore **cannot** flag them — only a **drift/derivative** signal (rising EMA/freq) or **cross-layer** correlation can. This is the correct justification for routing prefetch work to E1b and rejecting naive same-layer reuse (E1a). `(ANALYSIS)`

---

## 7. Glossary

- **Routed vs shared expert:** routed = the many specialists the router picks `top_k` of (offloaded); shared = the few generalists every token always uses (always resident).
- **`top_k`:** experts activated per token (6 here).
- **`w13` / `w2`:** fused gate+up projection `[H,2I]` / down projection `[I,H]`, stored transposed (`[in,out]`).
- **SwiGLU:** `silu(gate) ⊙ up` then down-project.
- **`log2phy`:** logical-expert-id → physical-slot table (`-1` = not resident).
- **Slot:** a fixed contiguous slab (index along dim 0) of the device weight tensor; holds one expert's weights.
- **`group_list`:** per-slot token count after dispatch; length = local expert count; sums to `num_tokens × top_k`.
- **Token expansion:** replicating each token into `top_k` rows (one per chosen expert) before dispatch.
- **NZ format:** Ascend tiled matmul-friendly memory layout (`FRACTAL_NZ`); CPU mirror is pre-cast to it.
- **H2D / D2D:** host→device / device→device copy. This feature uses H2D only.
- **Prefill / decode:** prompt processed in bulk (many tokens) / answer generated one token at a time.
- **GEMV vs GEMM:** matrix×vector (one token) vs matrix×matrix (many tokens); batch-1 decode is GEMV (memory-bound).
- **`offload_threshold`:** `num_device_experts // topk`; `num_tokens >` it ⇒ prefill pool, else decode cache.
- **ACL graph:** captured+replayed kernel sequence (torch_npu analog of CUDA graphs).
- ➕ **V2 — cache margin / spare slots:** `ndev − topk` (single-token decode). The number of slots that can *retain* experts beyond the current step's working set. Zero in the reference config ⇒ no caching, inert policy (§9 inv. 9).

---

## 8. Language & programming-model trade-offs

Python orchestrating `torch_npu` streams: great velocity (introspect modules, reshape buffers, index `self.moe_layers`), at the cost of (a) CPU↔NPU sync on the hot path (`topk_ids_h.tolist()` etc., ~`:615-618`) — the `tensor.item()`-style anti-pattern `AGENTS.md` warns about, at list granularity; (b) GIL-bound host callbacks under capture; (c) no static guarantee on the storage-slice copies (a layout mismatch is a silent miscompute). A "functional-core in Python, hot-loop in C++/AscendC" split is plausible (policy + bookkeeping stay Python; ids stop round-tripping; copies coalesce into `aclrtMemcpyBatchAsync`); the branch already straddles this line (memcpy capability lives in `vllm_ascend_C`).

> 🟡 **V2 NUANCE [N1] — the policy host cost is not just slow in the reference config, it is wasted.** Because the policy is inert at `ndev = topk` (§9 inv. 9), the `tolist()` + per-row EMA loops in `observe` (`lrc_policy.py:89-101`) run every decode step and produce a result that is never used. In the reference config the cheapest "optimization" is `cache_policy_enabled=false`. `(ANALYSIS)`

---

## 9. Invariants & assumptions (ground truth for agents)

Preserve these when modifying the code; violating them is a silent-correctness or perf regression.
1. **`log2phy` is the sole logical→physical map** read by the kernel (`moe_comm_method.py`). Any slot reassignment must update `log2phy` before the dispatch reads it.
2. **A cache hit performs zero copies.** Only misses issue H2D. No code path performs device-to-device weight movement; do not introduce compaction.
3. **Device weight tensor is allocated once at `[ndev,K,N]` and never reallocated/compacted.** Experts move by in-place H2D overwrite of fixed slabs. Storage-slice copies assume byte-identical contiguous CPU↔device layout (NZ-matched for W8A8).

> ⚠️ **V2 CORRECTION [N4] — strengthen invariant 3.** "byte-identical contiguous layout" is necessary but not the failure mode people hit. The exact invariant is: **per-expert NZ storage size == logical `nelement × element_size`** (no tile padding). This holds for tile-aligned expert dims and silently breaks otherwise. There is no runtime assertion; add one (see §4.1 V2 / opt-idea F3-v2).

4. **`group_list` length = local expert count** (`ndev` in decode, full total in the prefill pool) and is zero-padded for inactive slots; the grouped matmul indexes slabs positionally.
5. **Decode path is host-blocked** (`load_stream.synchronize()`); compute starts only after all misses land. (Changing this requires threading a `loading` set into `choose_victim`, optimization D4.)
6. **The prefill path temporarily patches `num_local_experts`** to the full total and restores it (`fused_moe.py:230-305`); this is not reentrancy-safe across concurrent layers (optimization F2).

> ⚠️ **V2 CORRECTION [A④] — invariant 6 is incomplete and the restore is not exception-safe.** The patch covers **three** fields — `layer.moe_config.num_local_experts`, `layer.local_num_experts`, and `moe_comm_method.token_dispatcher.num_experts_local` (`:230-237`, restored `:302-305`) — the last on a shared singleton. And there is **no `try/finally`**: an exception in `moe_comm_method.fused_experts` (`:274`) skips the restore and permanently corrupts all three. Guard with `try/finally` and treat the three-field set as one unit. `(FACT)`

7. **Offload activation is independent of graph mode**; the only mode-dependent branch is inline vs `_launch_host_func`.

> ➕ **V2 ADDITION [A⑤] — addendum to invariant 7.** While *activation* is mode-independent, *policy behavior* is not: under capture the router-score term is dropped from hotness (`:572-573`), so eviction (not output) differs between eager and graph. Document this when comparing modes.

8. **Per-layer state is per-layer:** 26 device buffers and 26 `LRCLayerState`s; never share expert weights or stats across layers.

> ➕ **V2 ADDITION [N1] — invariant 9 (NEW, provable): the LRC victim choice only discriminates when `num_device_experts > top_k`.**
> *Proof (single-token decode):* at a miss, eligible victims `= ndev − (needed ∩ residents) = ndev − hits`; misses `= |needed| − hits`. The *choice of which* to evict matters iff we evict a strict subset, i.e. `misses < eligible ⇔ |needed| < ndev`. With `|needed| = topk`, that is `ndev > topk`. At `ndev = topk`, `misses == eligible` ⇒ all non-needed residents are evicted regardless of hotness ⇒ resident set is forced to `needed` ⇒ `cache_policy_enabled` and all policy weights have **no effect** on residency. Corollary: every policy/caching optimization is dormant until `ndev > topk`.

> ➕ **V2 ADDITION [A③] — invariant 10 (NEW): in the decode path, `log2phy` contains no `-1` for any `needed` expert after `_update_weights`.** Guaranteed by `offload_threshold = ndev//topk` (so `|needed| ≤ ndev`). The `clamp(min=0)` at `moe_comm_method.py:142` is a *defensive* fallback for this invariant, not a routine code path; if you change the path-selection math (opt-ideas A1/G1) you must preserve this invariant or the clamp will silently miscompute (route to slot 0). Add an assertion rather than relying on the clamp.

---

## 10. Similar / superior projects worth studying next

- **KTransformers** — reference CPU/GPU MoE offload; *better at* genuine load/compute pipelining and a mature CPU-expert kernel path. Read `operators/experts.py` and the offload injection rules.
- **DeepSpeed-MoE / ZeRO-Inference** — canonical parameter-offload with a real look-ahead prefetch scheduler and pinned-buffer double buffering (the gap here). Read the ZeRO-Inference offload/prefetch engine.
- **MoE-Infinity / Fiddler** — activation-aware *predictive* expert caching/prefetch; directly addresses the E1 gap. Read their activation tracer / placement heuristics.
- **llama.cpp MoE offload (`-ot` / `--n-cpu-moe`)** — robust static per-tensor CPU/GPU placement and a clean memcpy path.

The honest gap: **no predictive prefetch.** "Prefetching" here = LRC retention + prefill-pool pre-staging + capture-mode plumbing (dormant under `--enforce_eager`). `--safetensors-load-strategy prefetch` and `weight_prefetch_config` are *startup* / *attention-weight* prefetch, unrelated to runtime expert paging. If predictive prefetch is the target, read MoE-Infinity / Fiddler / DeepSpeed first.

> 🟡 **V2 NUANCE [N1/A①] — restate the "honest gap" precisely.** It is not merely "no predictive prefetch." It is: (1) in the **reference config the cache itself is inert** (`ndev=topk`), so there is no retention to speak of — fix with A2 first; and (2) *even with margin*, same-layer LRC retention cannot predict misses (they are the rising-but-currently-cold experts, A① above) — that gap needs a drift/cross-layer predictor (E1b). Prioritize A2 (turn the cache on) before any prefetch research.

---

## 11. Next steps
See `expert-offload-optimization-ideas.md` (and its V2) for the prioritized, designed optimization hypotheses and the experiments to run first. The single highest-signal first move: instrument path-share (cache vs pool), per-layer next-step hit rate, and copy counts on the existing `--enforce_eager` inline path — those three numbers tell you where the time goes and whether prefetch is even viable.

> ⚠️ **V2 CORRECTION [N5] — two of those three numbers already exist in the code.** (1) **Next-step hit rate** is already computed and logged: `_record_cache_stats` measures `already_there = needed ∩ on_device` against the resident set left by the prior step (`expert_offload_manager.py:626-631, 742-748`). Set `cache_stats_log_interval=1` and *read* the `[EXPERT-OFFLOAD-CACHE] hit_rate` line — at `ndev=topk` it equals the step-to-step routing overlap (your prefetch ceiling). (2) **Copy count** is already computed as `n_copies` (`:714`) — it's just discarded (decode `update_weights` returns `None`, contradicting its docstring `:547-548`; see opt-idea F1). So the "instrument first" step is mostly *surfacing existing values* + adding X2's miss-novelty probe, not new measurement. `(FACT)`
