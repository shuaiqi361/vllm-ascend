# Expert Offload — Optimization Ideas (triaged, with designs)

**Companion to:** `code-summary-vllm-ascend-expert-offload.md` (read that first for mechanism + line refs).
**Status:** living document. Each idea has a verdict, evidence (`file:line`), a brief design where concrete, and a validate-first step. These are **analysis + design sketches, not implementation diffs** — building any of them is a separate step (the codebase forbids unreviewed perf changes; see `AGENTS.md`).
**Audience:** humans (read the triage table, then the tiers) and Claude agents (treat `file:line` as jump targets; verdicts and "depends-on" are machine-actionable; designs are sketches to expand, not apply verbatim).

**Verdict legend:** ✅ worth doing · 🧪 worth *measuring* first (cheap experiment gates it) · ⚙️ worth doing but conditional on a measurement · ❌ rejected (kept with rationale so it isn't re-proposed).
**Confidence:** 🟢 high · 🟡 plausible · 🔴 low.

---

## Triage table

| ID | Title | Verdict | Conf | Effort | Priority | Depends on |
|----|-------|:------:|:----:|:------:|:--------:|------------|
| X1 | Path-share + hit-rate + copy-count instrumentation | ✅ | 🟢 | XS | **1** | — |
| X2 | Next-step hit rate + miss-novelty probe | ✅ | 🟢 | S | **1** | X1 |
| A2 | Raise `num_device_experts` (tuning, not code) | ✅ | 🟢 | XS | **1** | X1 |
| C1 | Batched H2D via `aclrtMemcpyBatchAsync` | ✅ | 🟢 | M | **2** | X1 |
| A1 | Unique-expert-aware path threshold | ✅ | 🟢 | S–M | **2** | X1 |
| G1 | Split working buffer vs shared hot cache | ✅ | 🟢/🟡 | L | **2** | X1, A1 |
| B1 | Skip redundant prefill-pool reloads | ✅ | 🟢 | S | **2** | X1 |
| D3 | A/B the router-score term | 🧪 | 🟡 | XS | **2** | X1 |
| D2 | Lazy / vectorized EMA in `observe` | ⚙️ | 🟡 | S | 3 | D1 profile |
| D1 | Vectorize policy off the Python hot path | ⚙️ | 🟡 | M | 3 | profile |
| B2 | Double-buffer the prefill pool | ⚙️ | 🟡 | M–L | 3 | B1, copy/compute profile |
| B3 | Skip fp32 scale recompute on unchanged reload | ⚙️ | 🟡 | XS | 3 | B1 |
| C2 | Trim `log2phy` device↔CPU round-trips | ⚙️ | 🟡 | S | 4 | X1 |
| D4 | Thread `loading` set into `choose_victim` | ⚙️ | 🟡 | XS | 4 | B2/async |
| E1b | Cross-layer / drift predictor for prefetch | 🧪 | 🟡 | L | 3 | X2 positive |
| E1a | Naive LRC-reuse prefetch | ❌ | 🟢 | — | — | — |
| E2 | Intra-layer compute/copy overlap (decode) | ❌ | 🟢 | — | — | — |
| F1–F3 | Correctness/robustness checks | ✅ | 🟢 | XS | 2 | — |

> The two ❌ rows are documented in "Rejected (with rationale)" so the reasoning is preserved.

---

## Experiments to run first (cheap, highest signal)

All run on the existing `--enforce_eager` inline path, where `_update_weights` executes synchronously — so plain counters/logs suffice (no graph-capture complications).

### X1 ✅ Path-share + hit-rate + copy-count instrumentation
- **Why:** answers "am I measuring the cache or the pool?" and "where does the time go?" — prerequisite for nearly everything.
- **Design:** in `update_weights` (`expert_offload_manager.py:~550`), increment counters per call: `path = "pool" if num_tokens > offload_threshold else "cache"`; for the cache path, record `len(needed)`, `len(need_to_load)` (misses), `len(needed) - len(need_to_load)` (hits). Aggregate per layer and globally; dump every `cache_stats_log_interval`. Also time each path (wall clock around the call) to get the pool-vs-cache time share.
- **Read:** if pool-share ≫ cache-share, A1/A2 are the priority; if cache hit-rate is low, A2/G1.

### X2 ✅ Next-step hit rate + miss-novelty probe (gates all prefetch work)
- **Why:** decides whether predictive prefetch (E1b) is viable *before* anyone writes it.
- **Design:** per layer, when `update_weights` runs at decode step *t*, compare this step's `needed` set against the slot contents left by step *t−1* (already available as `slot_owner`). Log (a) hit rate = `|needed ∩ resident| / |needed|`; (b) for each miss, its LRC hotness *rank* among non-resident experts (is the miss in LRC's high-rank tail or genuinely novel?).
- **Read:** hit≈100% ⇒ prefetch pointless (push A2/G1). Misses present *and* LRC-rank-predictable ⇒ even simple prefetch could help. Misses present but *not* LRC-predictable ⇒ need a cross-layer/drift predictor (E1b).

### (also cheap) D3 router-score A/B and C1 copy-count — see those entries.

---

## Tier 1 — highest leverage

### A2 ✅🟢 Raise `num_device_experts` (tuning, not code)
- **Evidence:** default 32 (`ascend_config.py:~620`); reference run sets 6, giving zero cache margin and `offload_threshold=1`.
- **Action:** sweep `num_device_experts ∈ {6,12,18,32}` (multiples of `topk=6`), watch hit-rate knee (X1) and device headroom. No code change.
- **Cost-model caveat:** `group_list` length = `num_device_experts`; the grouped matmul walks all slab positions, skipping empty groups (zero bandwidth) but possibly paying per-group overhead. Negligible at 32; verify on hardware before pushing to e.g. 256 — it bounds how far this scales. (See `code-summary` §6 "Grouped GEMM … contiguous?".)

### C1 ✅🟢 Batched H2D via `aclrtMemcpyBatchAsync`
- **Evidence:** decode (`expert_offload_manager.py:681-686`) and prefill (`:486-492`) issue one `untyped_storage().copy_()` per expert. The build already detects `aclrtMemcpyBatchAsync` and defines `CANN_MEMCPY_BATCH_ASYNC` (`CMakeLists.txt`; `setup.py:342-345`), but the Python path never calls it.
- **Why:** batching N copies into one API call removes per-copy launch overhead — significant when each expert is modest-sized and many are loaded (prefill: all 64/layer; decode: the miss set).
- **Design:** expose a custom op (in `vllm_ascend_C`, behind `CANN_MEMCPY_BATCH_ASYNC`) taking parallel arrays of `(src_ptr, dst_ptr, nbytes)` and issuing one `aclrtMemcpyBatchAsync`. Python side: build the descriptor list for the miss set (decode) or the full set (prefill) instead of looping `copy_`; fall back to the current loop when the macro is off. Keep it on `load_stream`.
- **Validate first:** microbench N individual `aclrtMemcpyAsync` vs one batched call at the real expert sizes; count copies/step from X1.

### A1 ✅🟢 Unique-expert-aware path threshold
- **Evidence:** `offload_threshold = num_device_experts // topk` (`expert_offload_manager.py:38`); forward branches at `fused_moe.py:191`. Worst-case bound collapses to 1 at `topk==ndev`, dumping multi-token decode into the pool unnecessarily.
- **Why:** the real constraint is "do the step's *distinct* experts fit in slots," which is usually far below the `num_tokens × topk` worst case (routing overlaps heavily).
- **Design:** compute `n_distinct = unique(topk_ids).numel()` (a cheap op on the small `topk_ids`); take the cache path when `n_distinct <= num_device_experts` (or `<= working_region` under G1), else the pool. Replaces the `num_tokens > threshold` test.
- **Validate first:** log `n_distinct` vs `num_tokens` across batch sizes (shared with X1); confirm how many pool-bound steps would flip to cache.
- **Risk:** must guarantee the chosen experts truly fit before committing to the cache path (else the no-slots fallback fires). Pair with the capacity check already in the pager.

### G1 ✅ (concept 🟢 / effort 🟡) Split working buffer vs shared hot cache
- **Evidence:** one flat pool; LRC treats all slots as evictable (`choose_victim` over `slot_owner.values()`, `expert_offload_manager.py:630-664`); only transient `protected=needed` (`:663`). No structural split.
- **Problem:** working-set capacity and reuse-cache are fused, so the *whole* pool must absorb the worst-case working set (that's what the threshold enforces). Raising `--max-num-seqs` inflates the *entire* `num_device_experts` even though only the working part needed to grow; the hot cache (a property of the layer's routing, not of any sequence) shouldn't scale with concurrency.
- **Design:** partition each layer's `[ndev,K,N]` tensor into `working_region` (size ≈ `max_batch × topk`, or `n_distinct` per step; refreshed each step, **never** reused across steps) + `hot_cache` (fixed size, LRC-governed, batch-independent, shared across sequences). The grouped matmul already addresses arbitrary slots via `log2phy`, so "activate a cached expert" is a `log2phy`/`group_list` relabel, **no D2D**. Threshold becomes `n_distinct <= working_region` (subsumes A1).
- **Validate first:** X1 + A1 instrumentation; measure how small a dedicated working region could be vs the current fused pool; confirm hot-set sharing across concurrent sequences (per-seq vs aggregate hot sets).
- **Risk:** touches slot bookkeeping, `log2phy` layout, threshold branch, and the prefill-pool interaction — correctness-sensitive. Highest architectural payoff here.

### F1–F3 ✅ Correctness/robustness (cheap, do alongside)
- **F1:** `update_weights` decode branch returns `None` though the docstring promises a copy count (`:547-548` vs `:603`). Confirm `fused_moe.py:190` ignores it (it appears to). Make the return consistent.
- **F2:** prefill path mutates+restores `num_local_experts` (`fused_moe.py:230-305`) — not reentrant across concurrent layers/streams. Confirm single-threaded assumption or guard it.
- **F3:** storage-slice copies assume byte-identical contiguous CPU↔device layout (`:681-686`). Add an element-count/contiguity assertion to fail loud instead of silently miscomputing.

---

## Tier 2 — solid, after Tier 1 / measurements

### B1 ✅🟢 Skip redundant prefill-pool reloads
- **Evidence:** `_prefill_load_layer` unconditionally full-overwrites all experts into `pool_slot = layer_idx % ndl` and `synchronize()`s (`:464-524`); `update_weights` runs every layer every step. No "does this slot already hold layer L?" check.
- **Design:** maintain `slot_layer[pool_slot]` and `slot_dirty`; in `_prefill_load_layer`, skip the H2D (and the fp32 recompute, B3) when `slot_layer[pool_slot] == layer_idx`. With `ndl ≥ num_moe_layers` each layer loads once; with small `ndl`, consecutive chunks revisiting a layer skip reload.
- **Validate first:** count redundant reloads/step (X1 extension); confirm whether multi-chunk prefill revisits layers.

### B3 ⚙️🟡 Skip fp32 scale recompute on unchanged reload
- Rider on B1: `_prefill_load_layer` recomputes `w13_weight_scale_fp32` each load (`:516-522`); skip when the slot's layer is unchanged. Only matters if B1 shows redundant reloads.

### D3 🧪🟡 A/B the router-score term
- **Evidence:** `cache_router_weight` defaults 0.3 (`ascend_config.py:~628`) ⇒ ON ⇒ forces `topk_weights → fp32 → .tolist()` host transfer each step (`expert_offload_manager.py:572-575,615`).
- **Experiment:** run with `cache_router_weight=0` vs `0.3`, compare `[EXPERT-OFFLOAD-CACHE] hit_rate`. If hit rate barely moves, drop the term and its transfer. XS effort, pure config.

### D1 ⚙️🟡 Vectorize policy off the Python hot path
- **Evidence:** `_update_weights` does `topk_ids_h.tolist()` (+ `topk_weights_h.tolist()` when router weight on) (`:615-618`); `observe` loops rows×experts (`lrc_policy.py:96-117`); `choose_victim` is an O(resident) `min` with a Python lambda per miss (`:119-133`).
- **Design:** keep ids as tensors/numpy; compute `needed`, hits/misses, and victim selection with vectorized ops; avoid `tolist()`. Conditional on profiling showing host time matters (grows with `num_experts`, `ndev`, `num_tokens`).
- **Validate first:** profile host time of `_update_weights` vs slot/expert count.

### D2 ⚙️🟡 Lazy / vectorized EMA
- **Evidence:** `observe` decays `ema[eid]` for every `eid in range(num_experts)` per token row (`lrc_policy.py:~96-101`) → O(tokens × experts) in Python.
- **Design:** decay-on-read (store `last_decay_step`; apply `ema *= beta**(step-last)` when read), or vectorize the decay once per step instead of per row. Pairs with D1.
- **Validate first:** profile `observe` vs `num_experts` and tokens/step.

---

## Tier 3 — conditional / research-y

### B2 ⚙️🟡 Double-buffer the prefill pool
- **Evidence:** `pool_slot = layer_idx % ndl` is reuse, not pipelining (`:473`); each layer load `synchronize()`s before compute (`:524`); forward awaits it (`fused_moe.py:190→274`).
- **Design:** with `ndl ≥ 2`, issue layer L+1's pool load on `load_stream` during layer L's compute; sync only when L+1 is actually needed (event-based, not full `synchronize`). Combine with B1 to avoid reloading unchanged slots.
- **Validate first:** profile per-layer copy vs compute time; only worthwhile if copy ≈ or > compute, and if `load_stream` truly runs concurrently with compute on this hardware.

### C2 ⚙️🟡 Trim `log2phy` device↔CPU round-trips
- **Evidence:** `update_weights` copies device `log2phy`→pinned CPU, mutates numpy, copies back every layer (`:576-603`).
- **Design:** keep the host map authoritative (it already is via `log2phy_np`) and push only changed entries to device, or keep the map device-side. Likely minor; measure first.

### D4 ⚙️🟡 Thread `loading` set into `choose_victim`
- **Evidence:** `choose_victim` accepts a `loading` param (`lrc_policy.py:119-133`) but the decode caller passes only `protected=needed` (`:660-664`). Harmless while synchronous; required before any async/pipelined paging (else a slot could be evicted mid-load). Enabler for B2/E1b.

### E1b 🧪🟡 Cross-layer / drift predictor for prefetch
- **Evidence & rationale:** see `code-summary` §6 and E1a below — same-layer LRC reuse is a no-op; the only addressable opportunity is predicting *misses* (not-yet-resident experts about to be needed).
- **Possible signals:** (1) cross-layer routing correlation — does layer L's routing predict L+1's? (2) a fast drift detector that leads LRC's slow averages when the hot set shifts. Prefetch the predicted *non-resident* set on `load_stream` during the L→L+1 window (which includes L+1's attention — a longer hide window than one FFN), evicting cold via `choose_victim(..., loading=...)` (needs D4).
- **Gated by X2:** only build if misses are present *and* predictable. If X2 shows misses ≈ 0, do A2/G1 instead.

---

## Rejected (with rationale — kept so they aren't re-proposed)

### E1a ❌🟢 Naive LRC-reuse prefetch
- **Claim:** "use LRC hotness to prefetch layer L+1's hot experts during layer L."
- **Why rejected:** each layer's slots already hold the top-`ndev` experts by LRC hotness (carried from the last time the layer ran). "Prefetch the hot ones" and "they're already resident" are the same statement under the local-routing-consistency assumption that *also* makes the cache work. No information gain ⇒ no better than doing nothing. The LRC table *does* track all experts regardless of residency (eviction never clears scores), so the data exists — but reusing it for same-layer prefetch is a no-op. The real opportunity is E1b (predict *misses*, which same-layer LRC cannot).

### E2 ❌🟢 (decode) Intra-layer compute/copy overlap
- **Claim:** "compute the cached experts while the missing ones transfer over PCIe."
- **Why rejected for decode:** confirmed serial today (loads misses, `load_stream.synchronize()` at `:716`, then computes all slots). But overlap would hide almost nothing: ballpark batch-1 bf16, a 2-miss transfer ≈ 33 MB over PCIe ≈ ~1.4 ms, while the cached-expert compute is HBM-read-bound at ~70 µs — transfer ≈ 20× the overlappable compute, so perfect overlap hides ~5%. You can't hide a ms-scale transfer behind a µs-scale GEMV. Also costs two dispatch/matmul/combine passes + partial-sum accumulation.
- **Where the leverage actually is:** don't miss (A2/G1), prefetch ahead with prediction (E1b), or transfer faster (C1). Revisit only if a future design sustains large per-expert token counts on a cache path (then compute ≳ transfer).

---

## Suggested order of work
1. **X1 + X2** (instrument) — one short pass; tells you everything below's priority.
2. **A2** (config sweep) — free, immediate.
3. **C1** (batched memcpy) — biggest copy-cost win, infra already compiled in.
4. **A1** then **G1** — fix the path-routing and the buffer/cache conflation (do A1 as the stepping stone into G1).
5. **B1 (+B3)** — if X1 shows the prefill pool dominates (likely in the reference config).
6. **D3** (A/B), then **D1/D2** if profiling flags host cost.
7. **E1b** only if **X2** says misses are present and predictable; **B2/D4** if pipelining becomes the target.
