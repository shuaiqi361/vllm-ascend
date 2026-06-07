# Expert Offload — Optimization Ideas (triaged, with designs) — **VERSION 2**

> ## 📌 About this V2 (read first)
> Review pass over V1 (`expert-offload-optimization-ideas.md`), **additive-only**: every V1 line is preserved verbatim; corrections/additions are **callout blocks**. Verdict changes are shown in the triage table as `V1 → V2` with a footnote ID.
>
> **Callout legend:** `⚠️ V2 CORRECTION [id]` (V1 wrong/imprecise) · `➕ V2 ADDITION [id]` (net-new) · `🟡 V2 NUANCE [id]` (understated).
> **`(FACT)`** = read from code (`file:line`); **`(ANALYSIS)`** = derived/opinion. Several V1 "Evidence" lines are actually `(ANALYSIS)` — flagged below.
>
> **The one structural change that dominates everything (read before the table):**
> > 🔴 **HARD GATE [N1].** The reference config sets `num_device_experts (6) == top_k (6)`. At `ndev ≤ topk` the LRC policy is **provably inert** — it never makes a discriminating eviction choice and residency is forced to equal each step's `needed` set (proof: `code-summary` V2 §9 invariant 9). Therefore the **entire policy/caching half of this document — D1, D2, D3, D4, the hot-cache portion of G1, and E1b — is DORMANT and UNMEASURABLE until A2 raises `ndev` above `topk`.** A2 is not "priority-1 tuning"; it is the prerequisite that turns the cache on at all. Do A2 first or none of D*/G1-cache/E1b can even be evaluated.
>
> **V2 verdict changes at a glance:** **B1 ✅→❌** (dead in any memory-saving config), **C1 priority ↓** (bandwidth-bound, ~3% on weights), **A2 → marked prerequisite**, **D3 depends X1 → depends A2**, **F2/F3 scope corrected**, **X1/X2 effort ↓** (mostly already-logged).

**Companion to:** `code-summary-vllm-ascend-expert-offload.md` (read that first for mechanism + line refs).
**Status:** living document. Each idea has a verdict, evidence (`file:line`), a brief design where concrete, and a validate-first step. These are **analysis + design sketches, not implementation diffs** — building any of them is a separate step (the codebase forbids unreviewed perf changes; see `AGENTS.md`).
**Audience:** humans (read the triage table, then the tiers) and Claude agents (treat `file:line` as jump targets; verdicts and "depends-on" are machine-actionable; designs are sketches to expand, not apply verbatim).

**Verdict legend:** ✅ worth doing · 🧪 worth *measuring* first (cheap experiment gates it) · ⚙️ worth doing but conditional on a measurement · ❌ rejected (kept with rationale so it isn't re-proposed).
**Confidence:** 🟢 high · 🟡 plausible · 🔴 low.

---

## Triage table

| ID | Title | Verdict (V1) | V2 verdict | Conf | Effort | Priority | Depends on |
|----|-------|:------:|:------:|:----:|:------:|:--------:|------------|
| X1 | Path-share + hit-rate + copy-count instrumentation | ✅ | ✅ *(scope ↓, see ‡N5)* | 🟢 | XS | **1** | — |
| X2 | Next-step hit rate + miss-novelty probe | ✅ | ✅ *(hit-rate already logged, ‡N5)* | 🟢 | S→**XS** | **1** | X1 |
| A2 | Raise `num_device_experts` (tuning, not code) | ✅ | ✅ **PREREQUISITE ‡N1** | 🟢 | XS | **1** | X1 |
| C1 | Batched H2D via `aclrtMemcpyBatchAsync` | ✅ | ⚙️ *(scope: scales+prefill, ‡N3)* | 🟢→🟡 | M | ~~2~~→**3** | X1 |
| A1 | Unique-expert-aware path threshold | ✅ | ✅ *(critical iff `max-num-seqs>1`, ‡A1)* | 🟢 | S–M | **2** | X1 |
| G1 | Split working buffer vs shared hot cache | ✅ | ✅ *(cache half gated by A2, ‡N1)* | 🟢/🟡 | L | **2** | X1, A1, **A2** |
| B1 | Skip redundant prefill-pool reloads | ✅ | **❌ ‡N2** | 🟢 | S | ~~2~~ | — |
| D3 | A/B the router-score term | 🧪 | 🧪 *(depends **A2**, eager-only, ‡A5)* | 🟡 | XS | **2** | ~~X1~~→**A2** |
| D2 | Lazy / vectorized EMA in `observe` | ⚙️ | ⚙️ *(gated by A2, ‡N1)* | 🟡 | S | 3 | D1 profile, **A2** |
| D1 | Vectorize policy off the Python hot path | ⚙️ | ⚙️ *(gated by A2, ‡N1)* | 🟡 | M | 3 | profile, **A2** |
| B2 | Double-buffer the prefill pool | ⚙️ | ⚙️ *(the real reason `ndl≥2` exists, ‡N2/N6)* | 🟡 | M–L | 3 | ~~B1~~, copy/compute profile |
| B3 | Skip fp32 scale recompute on unchanged reload | ⚙️ | **❌/folded into B2 ‡N2** | 🟡 | XS | 3 | ~~B1~~ |
| C2 | Trim `log2phy` device↔CPU round-trips | ⚙️ | ⚙️ | 🟡 | S | 4 | X1 |
| D4 | Thread `loading` set into `choose_victim` | ⚙️ | ⚙️ | 🟡 | XS | 4 | B2/async |
| E1b | Cross-layer / drift predictor for prefetch | 🧪 | 🧪 *(gated by A2 then X2, ‡N1)* | 🟡 | L | 3 | X2 positive, **A2** |
| E1a | Naive LRC-reuse prefetch | ❌ | ❌ *(right verdict, reasoning fixed ‡A1-claim)* | 🟢 | — | — | — |
| E2 | Intra-layer compute/copy overlap (decode) | ❌ | ❌ *(math confirmed)* | 🟢 | — | — | — |
| F1–F3 | Correctness/robustness checks | ✅ | ✅ *(F2/F3 corrected, +F4 ‡A4/N4/N6)* | 🟢 | XS | 2 | — |

> The two ❌ rows are documented in "Rejected (with rationale)" so the reasoning is preserved.

> ➕ **V2 ADDITION — footnotes for the table changes (old → new + why):**
> - **‡N1 (A2 prerequisite; D1/D2/D3/E1b/G1-cache gated):** policy inert at `ndev≤topk` (proof: code-summary V2 §9 inv. 9). Nothing policy-related is measurable in the reference config.
> - **‡N2 (B1 ✅→❌; B3 folded; B2 reframed):** B1's skip condition can only fire at `ndl ≥ num_moe_layers`, which puts the whole expert set on device = offload disabled. Dead in every memory-saving config. See B1-v2 below. B2 (overlap) is the only legitimate use of `ndl≥2`.
> - **‡N3 (C1 priority ↓, 🟢→🟡):** `aclrtMemcpyBatchAsync` batches descriptors, not bandwidth; per-expert copies are 16.5 MB (bandwidth-bound) ⇒ ~3% win on weights. Real payoff is the many *tiny* scale/offset copies + prefill. See C1-v2.
> - **‡A1 (A1 priority context):** A1 only matters when `--max-num-seqs > 1` (batched decode > threshold). At the reference `max-num-seqs=1` it never triggers. Co-priority-1 with A2 *if* concurrency is a target.
> - **‡A4 (F2 scope):** prefill patch is 3 fields + no `try/finally` (not just `num_local_experts`).
> - **‡N4 (F3 reword):** the assertion should check "no NZ tile padding" (storage size), not generic contiguity.
> - **‡N5 (X1/X2 effort ↓):** next-step hit rate (`_record_cache_stats`) and copy count (`n_copies`) already exist in code; surfacing > measuring.
> - **‡A5 (D3 depends A2, eager-only):** router term is unmeasurable at zero margin (N1) and is dropped under ACL-graph capture (`:572-573`), so A/B it in eager mode with `ndev>topk`.

---

## Experiments to run first (cheap, highest signal)

All run on the existing `--enforce_eager` inline path, where `_update_weights` executes synchronously — so plain counters/logs suffice (no graph-capture complications).

> ⚠️ **V2 CORRECTION [N1] — but first, raise `ndev` above `topk` (A2), or these experiments measure a degenerate corner.** In the reference config the cache is inert; X1's "is it cache or pool?" and X2's "is prefetch viable?" both answer "the cache isn't doing anything" trivially. Run X1/X2 at `ndev ∈ {12,18,32}` to get signal.

### X1 ✅ Path-share + hit-rate + copy-count instrumentation
- **Why:** answers "am I measuring the cache or the pool?" and "where does the time go?" — prerequisite for nearly everything.
- **Design:** in `update_weights` (`expert_offload_manager.py:~550`), increment counters per call: `path = "pool" if num_tokens > offload_threshold else "cache"`; for the cache path, record `len(needed)`, `len(need_to_load)` (misses), `len(needed) - len(need_to_load)` (hits). Aggregate per layer and globally; dump every `cache_stats_log_interval`. Also time each path (wall clock around the call) to get the pool-vs-cache time share.
- **Read:** if pool-share ≫ cache-share, A1/A2 are the priority; if cache hit-rate is low, A2/G1.

> ⚠️ **V2 CORRECTION [N5] — most of X1 already exists; this is plumbing, not new measurement.** `n_copies` is already computed in `_update_weights` (`:714`) but **discarded** (decode `update_weights` returns `None`, contradicting its docstring `:547-548` — that's exactly F1). Hits/misses are already computed as `already_there`/`need_to_load` (`:631-632`) and logged by `_record_cache_stats` (`:742-748`). The only genuinely new bits are **path-share** (cache vs pool counters) and **wall-clock timing**. `(FACT)`

### X2 ✅ Next-step hit rate + miss-novelty probe (gates all prefetch work)
- **Why:** decides whether predictive prefetch (E1b) is viable *before* anyone writes it.
- **Design:** per layer, when `update_weights` runs at decode step *t*, compare this step's `needed` set against the slot contents left by step *t−1* (already available as `slot_owner`). Log (a) hit rate = `|needed ∩ resident| / |needed|`; (b) for each miss, its LRC hotness *rank* among non-resident experts (is the miss in LRC's high-rank tail or genuinely novel?).
- **Read:** hit≈100% ⇒ prefetch pointless (push A2/G1). Misses present *and* LRC-rank-predictable ⇒ even simple prefetch could help. Misses present but *not* LRC-predictable ⇒ need a cross-layer/drift predictor (E1b).

> ⚠️ **V2 CORRECTION [N5/A1-claim] — X2(a) is ALREADY LOGGED; only X2(b) is new, and X2(b)'s "predictable" reading is backwards.** (a) `_record_cache_stats` already computes `already_there = needed ∩ on_device` against the prior step's residents (`:626-631`) and logs `hit_rate` (`:742-748`). Set `cache_stats_log_interval=1` and read it — at `ndev=topk` it *is* the step-to-step routing overlap = prefetch ceiling. **No code for X2(a).** (b) The miss-novelty probe is the only new work — **but note (A①):** misses are by construction the experts whose hotness was *low at eviction*, so a miss being in LRC's *low* rank is *expected* and does **not** mean "unpredictable." The discriminating signal is **rising hotness (drift)** or **cross-layer correlation**, not current LRC rank. Reframe X2(b) to log each miss's *EMA/freq derivative* and *layer-(L−1) co-occurrence*, not its static rank. `(ANALYSIS)`

### (also cheap) D3 router-score A/B and C1 copy-count — see those entries.

---

## Tier 1 — highest leverage

### A2 ✅🟢 Raise `num_device_experts` (tuning, not code)
- **Evidence:** default 32 (`ascend_config.py:~620`); reference run sets 6, giving zero cache margin and `offload_threshold=1`.
- **Action:** sweep `num_device_experts ∈ {6,12,18,32}` (multiples of `topk=6`), watch hit-rate knee (X1) and device headroom. No code change.
- **Cost-model caveat:** `group_list` length = `num_device_experts`; the grouped matmul walks all slab positions, skipping empty groups (zero bandwidth) but possibly paying per-group overhead. Negligible at 32; verify on hardware before pushing to e.g. 256 — it bounds how far this scales. (See `code-summary` §6 "Grouped GEMM … contiguous?".)

> ⚠️ **V2 CORRECTION [N1] — A2 is a PREREQUISITE, not just tuning, and `6` gives not "zero margin" but "inert policy."** `(FACT)` default 32, reference 6. `(ANALYSIS)` at `ndev=6=topk`, spare = `ndev−topk = 0`, so the policy makes no choice (code-summary V2 §9 inv. 9) and *no* policy parameter is measurable. The sweep must **start above `topk`** (drop 6 from the sweep, or keep it only as the "cache off" baseline). Use `{12,18,24,32}`. Also surface the runtime motivation: at `ndev=6` decode streams ~2.6 GB/token over PCIe (code-summary V2 §3 addition) — A2 is the dominant throughput lever.

### C1 ✅🟢 Batched H2D via `aclrtMemcpyBatchAsync`
- **Evidence:** decode (`expert_offload_manager.py:681-686`) and prefill (`:486-492`) issue one `untyped_storage().copy_()` per expert. The build already detects `aclrtMemcpyBatchAsync` and defines `CANN_MEMCPY_BATCH_ASYNC` (`CMakeLists.txt`; `setup.py:342-345`), but the Python path never calls it.
- **Why:** batching N copies into one API call removes per-copy launch overhead — significant when each expert is modest-sized and many are loaded (prefill: all 64/layer; decode: the miss set).
- **Design:** expose a custom op (in `vllm_ascend_C`, behind `CANN_MEMCPY_BATCH_ASYNC`) taking parallel arrays of `(src_ptr, dst_ptr, nbytes)` and issuing one `aclrtMemcpyBatchAsync`. Python side: build the descriptor list for the miss set (decode) or the full set (prefill) instead of looping `copy_`; fall back to the current loop when the macro is off. Keep it on `load_stream`.
- **Validate first:** microbench N individual `aclrtMemcpyAsync` vs one batched call at the real expert sizes; count copies/step from X1.

> ⚠️ **V2 CORRECTION [N3] — the headline benefit is overstated; expert weight copies are bandwidth-bound, not launch-bound.** `aclrtMemcpyBatchAsync` batches *descriptors*, not bandwidth — the DMA still moves the same bytes at the same rate. Per-expert weight copies are **16.5 MB** (bf16): at ~24 GB/s ≈ 660 µs of transfer vs ~5–20 µs launch overhead ⇒ batching saves **~3%** on the weights. So C1 is **not** "the biggest copy-cost win." `(ANALYSIS, bandwidth math)`
>
> **Where C1 *does* pay (scope it here):** the **scale/offset** copies are many *tiny* per-expert ops (`dev_tensor.data[slot].copy_(...)`, `:688-701`) — those are genuinely launch-bound, and so is the **prefill** path's 64-copies-per-layer descriptor churn. Recommend: implement C1 for **scales + prefill descriptors**, expect a small single-digit-% gain, and **demote priority to 3**. The real copy-cost levers are *transfer fewer bytes* (A2 cache margin) and *don't reload unchanged* (B2), not *batch the same bytes*.

### A1 ✅🟢 Unique-expert-aware path threshold
- **Evidence:** `offload_threshold = num_device_experts // topk` (`expert_offload_manager.py:38`); forward branches at `fused_moe.py:191`. Worst-case bound collapses to 1 at `topk==ndev`, dumping multi-token decode into the pool unnecessarily.
- **Why:** the real constraint is "do the step's *distinct* experts fit in slots," which is usually far below the `num_tokens × topk` worst case (routing overlaps heavily).
- **Design:** compute `n_distinct = unique(topk_ids).numel()` (a cheap op on the small `topk_ids`); take the cache path when `n_distinct <= num_device_experts` (or `<= working_region` under G1), else the pool. Replaces the `num_tokens > threshold` test.
- **Validate first:** log `n_distinct` vs `num_tokens` across batch sizes (shared with X1); confirm how many pool-bound steps would flip to cache.
- **Risk:** must guarantee the chosen experts truly fit before committing to the cache path (else the no-slots fallback fires). Pair with the capacity check already in the pager.

> ⚠️ **V2 CORRECTION [A①/A③] — sharpen the priority and the failure mode.** (1) **Priority is conditional:** A1 only fires when steps have `num_tokens > offload_threshold`, i.e. **batched decode with `--max-num-seqs > 1`** (or chunked-prefill micro-batches). At the reference `max-num-seqs=1` decode is always single-token ⇒ A1 never triggers. So A1 is *co-priority-1 with A2 iff concurrency is a target*, otherwise priority-low. (2) **The "Risk" understates the failure mode:** if the `n_distinct` bound is wrong and the pager hits no-slots, the consequence is **not** a loud error — the unmapped experts get `clamp(min=0)`'d to slot 0 (`moe_comm_method.py:142`) ⇒ *silent wrong-expert compute* (code-summary V2 §9 inv. 10). A1 **must** ship with the setup assertion that no `needed` expert maps to `-1` before dispatch. `(ANALYSIS)`

### G1 ✅ (concept 🟢 / effort 🟡) Split working buffer vs shared hot cache
- **Evidence:** one flat pool; LRC treats all slots as evictable (`choose_victim` over `slot_owner.values()`, `expert_offload_manager.py:630-664`); only transient `protected=needed` (`:663`). No structural split.
- **Problem:** working-set capacity and reuse-cache are fused, so the *whole* pool must absorb the worst-case working set (that's what the threshold enforces). Raising `--max-num-seqs` inflates the *entire* `num_device_experts` even though only the working part needed to grow; the hot cache (a property of the layer's routing, not of any sequence) shouldn't scale with concurrency.
- **Design:** partition each layer's `[ndev,K,N]` tensor into `working_region` (size ≈ `max_batch × topk`, or `n_distinct` per step; refreshed each step, **never** reused across steps) + `hot_cache` (fixed size, LRC-governed, batch-independent, shared across sequences). The grouped matmul already addresses arbitrary slots via `log2phy`, so "activate a cached expert" is a `log2phy`/`group_list` relabel, **no D2D**. Threshold becomes `n_distinct <= working_region` (subsumes A1).
- **Validate first:** X1 + A1 instrumentation; measure how small a dedicated working region could be vs the current fused pool; confirm hot-set sharing across concurrent sequences (per-seq vs aggregate hot sets).
- **Risk:** touches slot bookkeeping, `log2phy` layout, threshold branch, and the prefill-pool interaction — correctness-sensitive. Highest architectural payoff here.

> 🟡 **V2 NUANCE [N1] — G1's `hot_cache` is only meaningful once `hot_cache_size > 0` AND the working region doesn't consume all slots.** This is the structural generalization of invariant 9: the hot cache *is* the "spare slots beyond the working set." If you build G1, the LRC policy finally has a region where it discriminates — but only sized `ndev − working_region`. Size that region `> 0` deliberately; it is the *only* place the policy earns its keep. `(ANALYSIS)`

### F1–F3 ✅ Correctness/robustness (cheap, do alongside)
- **F1:** `update_weights` decode branch returns `None` though the docstring promises a copy count (`:547-548` vs `:603`). Confirm `fused_moe.py:190` ignores it (it appears to). Make the return consistent.
- **F2:** prefill path mutates+restores `num_local_experts` (`fused_moe.py:230-305`) — not reentrant across concurrent layers/streams. Confirm single-threaded assumption or guard it.

> ⚠️ **V2 CORRECTION [A④] — F2 is bigger than "reentrancy": it's an exception-safety bug touching 3 fields.** `(FACT)` The patch covers `layer.moe_config.num_local_experts`, `layer.local_num_experts`, **and** `moe_comm_method.token_dispatcher.num_experts_local` (`:230-237`, restored `:302-305`) — the last on a shared singleton. There is **no `try/finally`**, so an exception in `fused_experts` (`:274`) skips the restore and permanently corrupts all three for that layer. Fix = wrap patch→compute→restore in `try/finally` and restore all three. Upgrade F2 to a real bug, not just a "confirm assumption."

- **F3:** storage-slice copies assume byte-identical contiguous CPU↔device layout (`:681-686`). Add an element-count/contiguity assertion to fail loud instead of silently miscomputing.

> ⚠️ **V2 CORRECTION [N4] — F3's assertion should target NZ tile padding, not "contiguity."** `(ANALYSIS)` The actual silent-miscompute trigger is: for W8A8 the buffers are `FRACTAL_NZ` (`:171-174`), and `w13_expert_size_bytes` is computed from the **logical** tensor (`:159-160`). If NZ padding makes per-expert storage ≠ `nelement×element_size`, the slice `[s*N:(s+1)*N]` is misaligned → garbage weights, no crash. DeepSeek-V2-Lite happens to be tile-aligned so it works silently. Correct guard (one-time, at setup): `assert dev_tensor.untyped_storage().nbytes() == num_experts * expert_size_bytes` for both `w13` and `w2`.

> ➕ **V2 ADDITION [N6] — F4 (NEW): `num_device_layers=2` default wastes ~1.1 GB for zero benefit.** The prefill pool is host-blocked (`:524`) with no overlap, and the round-robin gives no reuse (see B1-v2). So the 2nd pool slot buys nothing until B2 (double-buffer) exists. Fix = default `ndl=1` until B2 lands, or gate `ndl≥2` on B2 being implemented. Cheap, frees HBM for A2. `(ANALYSIS)`

> ➕ **V2 ADDITION [A③] — F5 (NEW): assert no `needed` expert is unmapped before dispatch.** The decode capacity invariant (`|needed| ≤ ndev`) currently has *no* runtime guard; the `clamp(min=0)` masks any violation as a slot-0 miscompute. Add `assert (log2phy[topk_ids] >= 0).all()` (or a debug-gated version) on the cache path. This is the safety net that makes A1/G1 safe to land. `(ANALYSIS)`

---

## Tier 2 — solid, after Tier 1 / measurements

### B1 ✅🟢 Skip redundant prefill-pool reloads
- **Evidence:** `_prefill_load_layer` unconditionally full-overwrites all experts into `pool_slot = layer_idx % ndl` and `synchronize()`s (`:464-524`); `update_weights` runs every layer every step. No "does this slot already hold layer L?" check.
- **Design:** maintain `slot_layer[pool_slot]` and `slot_dirty`; in `_prefill_load_layer`, skip the H2D (and the fp32 recompute, B3) when `slot_layer[pool_slot] == layer_idx`. With `ndl ≥ num_moe_layers` each layer loads once; with small `ndl`, consecutive chunks revisiting a layer skip reload.
- **Validate first:** count redundant reloads/step (X1 extension); confirm whether multi-chunk prefill revisits layers.

> ⚠️ **V2 CORRECTION [N2] — B1 is DEAD (✅→❌). The skip can never fire in a memory-saving config.** `(ANALYSIS, mechanism)` Trace the round-robin: a forward processes **all** layers once (L0..L25). With `ndl=2`, slot 0 is reused within one forward by layers {0,2,4,…,24}; the next time L0 runs (next chunk/request), slot 0 last held L24 ⇒ `slot_layer[0] ≠ 0` ⇒ reload. The skip condition `slot_layer[pool_slot] == layer_idx` holds **only if `ndl ≥ num_moe_layers`** — but that means the pool holds *all 26 layers × all 64 experts* = the entire expert set on device = **offload disabled**. The V1 claim "consecutive chunks revisiting a layer skip reload" is wrong: chunks revisit a layer one *full layer-sweep* later, after the slot is overwritten. So B1 saves nothing in any config that actually saves memory. **Reclassify ❌**; the legitimate idea is B2 (overlap), which is the only sensible use of `ndl≥2`. (B3 folds away with B1.)

### B3 ⚙️🟡 Skip fp32 scale recompute on unchanged reload
- Rider on B1: `_prefill_load_layer` recomputes `w13_weight_scale_fp32` each load (`:516-522`); skip when the slot's layer is unchanged. Only matters if B1 shows redundant reloads.

> ⚠️ **V2 CORRECTION [N2] — B3 inherits B1's death.** Its "skip when slot's layer unchanged" condition never holds at `ndl < num_moe_layers` (same proof as B1-v2). **❌ / folded into B2.** If B2 (double-buffer) lands and a slot legitimately retains a layer across the overlap window, revisit then.

### D3 🧪🟡 A/B the router-score term
- **Evidence:** `cache_router_weight` defaults 0.3 (`ascend_config.py:~628`) ⇒ ON ⇒ forces `topk_weights → fp32 → .tolist()` host transfer each step (`expert_offload_manager.py:572-575,615`).
- **Experiment:** run with `cache_router_weight=0` vs `0.3`, compare `[EXPERT-OFFLOAD-CACHE] hit_rate`. If hit rate barely moves, drop the term and its transfer. XS effort, pure config.

> ⚠️ **V2 CORRECTION [N1/A⑤] — D3 is unmeasurable in the reference config and invisible under graph capture; fix the experiment.** `(ANALYSIS)` (1) At `ndev=topk` the policy is inert (N1), so `hit_rate` is identical for `router_weight ∈ {0, 0.3}` — the A/B reads zero by construction. **Run D3 with `ndev>topk` (depends A2), not X1.** (2) The router term is **dropped under ACL-graph capture** (`topk_weights_h` gated on `not capturing`, `:572-573`), so the A/B is only meaningful in **eager** mode. (3) The `.tolist()` cost D3 wants to remove is tiny in single-token decode (6 floats); the term's *value*, not its cost, is the question. Reframe: "does the router term improve hit rate at `ndev>topk`, eager mode?"

### D1 ⚙️🟡 Vectorize policy off the Python hot path
- **Evidence:** `_update_weights` does `topk_ids_h.tolist()` (+ `topk_weights_h.tolist()` when router weight on) (`:615-618`); `observe` loops rows×experts (`lrc_policy.py:96-117`); `choose_victim` is an O(resident) `min` with a Python lambda per miss (`:119-133`).
- **Design:** keep ids as tensors/numpy; compute `needed`, hits/misses, and victim selection with vectorized ops; avoid `tolist()`. Conditional on profiling showing host time matters (grows with `num_experts`, `ndev`, `num_tokens`).
- **Validate first:** profile host time of `_update_weights` vs slot/expert count.

> 🟡 **V2 NUANCE [N1] — in the reference config this host cost is not just unnecessary, it's *pure waste* (the policy output is unused).** So the cheapest "D1" there is `cache_policy_enabled=false`. D1 the optimization only matters once `ndev>topk` makes the policy do real work — gate it on A2. `(ANALYSIS)`

### D2 ⚙️🟡 Lazy / vectorized EMA
- **Evidence:** `observe` decays `ema[eid]` for every `eid in range(num_experts)` per token row (`lrc_policy.py:~96-101`) → O(tokens × experts) in Python.
- **Design:** decay-on-read (store `last_decay_step`; apply `ema *= beta**(step-last)` when read), or vectorize the decay once per step instead of per row. Pairs with D1.
- **Validate first:** profile `observe` vs `num_experts` and tokens/step.

> ➕ **V2 ADDITION [N1] — the EMA is the inherently-dense part; that's why D2 is structurally separate from D1.** D1 (avoid `tolist()`) addresses the *transfer*; D2 addresses the *only loop that touches all experts regardless of how few changed* (`lrc_policy.py:89-91`). Decay-on-read is the right fix. Still gated by A2 (policy inert otherwise). `(FACT/ANALYSIS)`

---

## Tier 3 — conditional / research-y

### B2 ⚙️🟡 Double-buffer the prefill pool
- **Evidence:** `pool_slot = layer_idx % ndl` is reuse, not pipelining (`:473`); each layer load `synchronize()`s before compute (`:524`); forward awaits it (`fused_moe.py:190→274`).
- **Design:** with `ndl ≥ 2`, issue layer L+1's pool load on `load_stream` during layer L's compute; sync only when L+1 is actually needed (event-based, not full `synchronize`). Combine with B1 to avoid reloading unchanged slots.
- **Validate first:** profile per-layer copy vs compute time; only worthwhile if copy ≈ or > compute, and if `load_stream` truly runs concurrently with compute on this hardware.

> ⚠️ **V2 CORRECTION [N2/N6] — B2 is the *only* justification for `ndl≥2`, and it does not "combine with B1" (B1 is dead).** Reframe: today `ndl=2` is wasted HBM (F4); B2 is what would earn it back, by loading L+1 into slot `(L+1)%2` (≠ L's slot) during L's compute. Drop the "combine with B1" clause. B2 is also where D4 (thread `loading` into `choose_victim`) becomes necessary. Promote B2 as the real successor to the dead B1. `(ANALYSIS)`

### C2 ⚙️🟡 Trim `log2phy` device↔CPU round-trips
- **Evidence:** `update_weights` copies device `log2phy`→pinned CPU, mutates numpy, copies back every layer (`:576-603`).
- **Design:** keep the host map authoritative (it already is via `log2phy_np`) and push only changed entries to device, or keep the map device-side. Likely minor; measure first.

### D4 ⚙️🟡 Thread `loading` set into `choose_victim`
- **Evidence:** `choose_victim` accepts a `loading` param (`lrc_policy.py:119-133`) but the decode caller passes only `protected=needed` (`:660-664`). Harmless while synchronous; required before any async/pipelined paging (else a slot could be evicted mid-load). Enabler for B2/E1b.

### E1b 🧪🟡 Cross-layer / drift predictor for prefetch
- **Evidence & rationale:** see `code-summary` §6 and E1a below — same-layer LRC reuse is a no-op; the only addressable opportunity is predicting *misses* (not-yet-resident experts about to be needed).
- **Possible signals:** (1) cross-layer routing correlation — does layer L's routing predict L+1's? (2) a fast drift detector that leads LRC's slow averages when the hot set shifts. Prefetch the predicted *non-resident* set on `load_stream` during the L→L+1 window (which includes L+1's attention — a longer hide window than one FFN), evicting cold via `choose_victim(..., loading=...)` (needs D4).
- **Gated by X2:** only build if misses are present *and* predictable. If X2 shows misses ≈ 0, do A2/G1 instead.

> ⚠️ **V2 CORRECTION [A①] — the precise reason same-layer LRC can't drive prefetch (fixes E1a's stated rationale).** The misses are exactly the experts whose hotness was **low at eviction** (that's why they were evicted) and is **now rising**. A *snapshot* of LRC rank therefore cannot flag them — by construction they sit in the low-rank tail until the step they're needed. The addressable signal is the **derivative** (rising EMA/freq = "drift detector" signal #2) or **cross-layer** co-occurrence (signal #1). So E1b's signal #2 should be specified as *rate-of-change of EMA/freq*, not their level. Also gate E1b on **A2** (need `ndev>topk` for misses/retention to even exist) in addition to X2. `(ANALYSIS)`

---

## Rejected (with rationale — kept so they aren't re-proposed)

### E1a ❌🟢 Naive LRC-reuse prefetch
- **Claim:** "use LRC hotness to prefetch layer L+1's hot experts during layer L."
- **Why rejected:** each layer's slots already hold the top-`ndev` experts by LRC hotness (carried from the last time the layer ran). "Prefetch the hot ones" and "they're already resident" are the same statement under the local-routing-consistency assumption that *also* makes the cache work. No information gain ⇒ no better than doing nothing. The LRC table *does* track all experts regardless of residency (eviction never clears scores), so the data exists — but reusing it for same-layer prefetch is a no-op. The real opportunity is E1b (predict *misses*, which same-layer LRC cannot).

> ⚠️ **V2 CORRECTION [A①/N1] — verdict ❌ stands, but this "Why rejected" mixes a `(FACT)` with a contestable `(ANALYSIS)` presented as fact.**
> - *"each layer's slots already hold the top-`ndev` experts by LRC hotness"* is **`(ANALYSIS)`, not evidence**, and it is only an *approximation* valid for `ndev > topk`. At `ndev = topk` (reference config) it is **vacuous**: residency is forced to `needed` and hotness is irrelevant (code-summary V2 §9 inv. 9). At `ndev > topk` the eviction-of-coldest rule *approximately* maintains it.
> - The *correct, robust* reason naive prefetch fails: **misses are rising-but-currently-cold experts**, which a static hotness snapshot cannot predict (see E1b-v2). This reason holds regardless of `ndev`, so it's the one to keep.
> - Net: keep ❌, but replace the rationale with the drift argument and tag the residency claim as an approximation. `(ANALYSIS)`

### E2 ❌🟢 (decode) Intra-layer compute/copy overlap
- **Claim:** "compute the cached experts while the missing ones transfer over PCIe."
- **Why rejected for decode:** confirmed serial today (loads misses, `load_stream.synchronize()` at `:716`, then computes all slots). But overlap would hide almost nothing: ballpark batch-1 bf16, a 2-miss transfer ≈ 33 MB over PCIe ≈ ~1.4 ms, while the cached-expert compute is HBM-read-bound at ~70 µs — transfer ≈ 20× the overlappable compute, so perfect overlap hides ~5%. You can't hide a ms-scale transfer behind a µs-scale GEMV. Also costs two dispatch/matmul/combine passes + partial-sum accumulation.
- **Where the leverage actually is:** don't miss (A2/G1), prefetch ahead with prediction (E1b), or transfer faster (C1). Revisit only if a future design sustains large per-expert token counts on a cache path (then compute ≳ transfer).

> 🟡 **V2 NUANCE [N3] — E2's math is sound; just drop "transfer faster (C1)" from the leverage list.** C1 saves ~3% on bandwidth-bound transfers (N3), so it is not a real alternative to E2 either. The honest leverage list is: **don't miss (A2/G1)** and **prefetch with prediction (E1b)**. Also note the "2-miss / cached-expert" framing presumes `ndev>topk` (at `ndev=topk` there are no cached survivors — see code-summary V2 §6). `(ANALYSIS)`

---

## Suggested order of work
1. **X1 + X2** (instrument) — one short pass; tells you everything below's priority.
2. **A2** (config sweep) — free, immediate.
3. **C1** (batched memcpy) — biggest copy-cost win, infra already compiled in.
4. **A1** then **G1** — fix the path-routing and the buffer/cache conflation (do A1 as the stepping stone into G1).
5. **B1 (+B3)** — if X1 shows the prefill pool dominates (likely in the reference config).
6. **D3** (A/B), then **D1/D2** if profiling flags host cost.
7. **E1b** only if **X2** says misses are present and predictable; **B2/D4** if pipelining becomes the target.

> ⚠️ **V2 CORRECTION — revised order of work (the V1 order has A2 too late, C1 too early, and B1 is dead).**
> 1. **A2 first** (config sweep `{12,18,24,32}` — *not* 6) — turns the cache on; **prerequisite** for measuring anything policy-related (N1). Free.
> 2. **X1 + X2** at `ndev>topk` — surface the already-computed counters (N5) + add path-share, wall-clock, and the *drift*-based miss probe (not static rank). Read existing `[EXPERT-OFFLOAD-CACHE] hit_rate` with `cache_stats_log_interval=1` before writing code.
> 3. **F1–F5** (correctness) — cheap, ship alongside: F2 `try/finally` (real bug, A④), F3 NZ-size assert (N4), F5 no-`-1` assert (A③), F4 `ndl=1` default (N6).
> 4. **A1** (only if `--max-num-seqs>1` is a target) → **G1** (the architectural fix; its hot-cache is where the policy finally earns its keep).
> 5. **C1** scoped to **scales + prefill descriptors** only — small win (N3), demoted.
> 6. **D3** (eager, `ndev>topk`) → **D1/D2** if profiling flags host cost.
> 7. **B2** (the real successor to the dead B1) + **D4** if pipelining becomes the target; **E1b** only if X2 shows *drift-predictable* misses.
> ~~**B1/B3**~~ removed — dead (N2).
