# moe_offload_v5.0 Study Guide — Proactive Expert Prefetch + LRC Offload

**Branch:** `moe_offload_v5.0` (upstream `LookAround0301/vllm-ascend`)
**Target model for this study:** DeepSeek V4-Flash, W8A8 quantized (int8 weights, per-expert scale/offset, NZ format)
**Date:** 2026-06-12 · status: code-reading study, no hardware validation

---

## 1. Branch anatomy

Three feature commits on top of a recent upstream base (~30 newer PRs than v2.0's base):

| Commit | Content |
|---|---|
| `c4572593` | Base expert offload + DeepSeek V4 support: `ExpertOffloadManager`, decode paging, prefill pool, W8A8 scale/offset handling |
| `098ea3f4` | LRC cache policy (`lrc_policy.py`), cache stats, log-analysis tools, UT |
| `d783c7a2` | **Next-layer predictive prefetch on a background thread** (the new feature) |

Files:
- `vllm_ascend/expert_offload/expert_offload_manager.py` (1123 lines — everything)
- `vllm_ascend/expert_offload/lrc_policy.py` (eviction scoring)
- `vllm_ascend/expert_offload/utils.py` (log2phy init)
- Hooks: `ops/fused_moe/fused_moe.py` (unquantized) and `quantization/methods/w8a8_dynamic.py` (**the path V4-Flash w8a8 takes**), `worker/model_runner_v1.py` (lifecycle)

---

## 2. The three memory tiers (where experts live)

```
┌─────────────────────────────────────────────────────────────────┐
│ CPU (pinned host memory) — source of truth                      │
│   w13_weights_cpu[layer][expert]  int8, NZ byte layout          │
│   w2_weights_cpu[layer][expert]   int8, NZ byte layout          │
│   scale/offset cpu buffers        per expert                    │
│   _gate_weights_cpu[layer]        fp32 gate.weight (prefetch)   │
│   ALL layers × ALL experts                                      │
├─────────────────────────────────────────────────────────────────┤
│ NPU decode pool — per MoE layer, ndev slots (default 32)        │
│   layer.w13_weight [ndev, ...]  ← contents change at runtime    │
│   layer.log2phy [n_experts] → slot id, or -1 if not resident    │
│   Both reactive loads AND prefetched experts land HERE.         │
│   There is NO separate staging buffer for prefetch.             │
├─────────────────────────────────────────────────────────────────┤
│ NPU prefill pool — ndl buffers (default 2), each ALL experts    │
│   Round-robin: model layer i uses pool slot i % ndl             │
│   Full overwrite per layer, identity log2phy                    │
│   Untouched by prefetch (prefetch is decode-only).              │
└─────────────────────────────────────────────────────────────────┘
```

Key numbers (defaults from `ExpertOffloadConfig`):
- `num_device_experts` (ndev) = 32 slots/layer
- `num_device_layers` (ndl) = 2 prefill pool buffers
- `offload_threshold = ndev // topk` tokens — batches above this take the prefill path, at/below take the decode paging path
- W8A8 special: CPU copies are stored **already in NZ byte layout** (done once at load via a temporary NPU round-trip in `process_weights_after_loading`), so H2D paging is a raw `untyped_storage().copy_()` with no format conversion on the hot path.

---

## 3. Lifecycle (model load)

`NPUModelRunner.load_model` → `_register_offload_layers()`:
1. `create_weights()` — allocate CPU buffers, drain weights loaded early, NZ-convert, build LRC policy (if `cache_policy_enabled`)
2. `register_moe_layer()` per layer + `maybe_create_scale_buffers()` (W8A8)
3. `init_device_experts()` — slots pre-filled with experts 0..ndev-1; refresh fp32 scales
4. `create_prefill_pool()` — allocate + NZ-cast the 2×128-expert pool
5. **`register_gate_weights(model)`** *(only if `expert_prefetch_enabled`)* — snapshots fp32 CPU clones of every `DeepseekV4MoE.gate.weight`, in module order. This is the "cross-layer gate" lookup table for prediction.

---

## 4. Baseline pipeline refresher: reactive paging + LRC (decode)

Per MoE layer in `AscendW8A8DynamicFusedMoEMethod.apply()`:

```
attention → gate: router_logits = x_fp32 @ gate.W^T
         → select_experts(...)            # REAL routing: grouped topk,
                                          # e_score_correction_bias, sigmoid/
                                          # softmax per config, renormalize
         → mgr.update_weights(layer, topk_ids, log2phy, topk_weights, x)
              │  (decode branch, ≤ threshold tokens)
              ├─ [NEW] wait for pending prefetch of THIS layer (see §5)
              ├─ D2H: topk_ids, log2phy → pinned host buffers
              ├─ LRC observe(): update freq (32-call sliding window),
              │                 EMA hotness, router score, last_used
              ├─ needed − resident = misses
              ├─ for each miss: LRC choose_victim() = resident expert with
              │     lowest hotness, never one in `needed` (protected)
              │     → H2D copy w13 + w2 + scale + offset + fp32-scale
              │       on load_stream; update host log2phy
              ├─ load_stream.synchronize()
              └─ write log2phy back to device
         → fused_experts(...)             # GMM kernels index slots via log2phy
```

LRC hotness = `1.0·freq_recent + 0.5·EMA + 0.3·router_score − 0.01·age`. It only matters when `ndev > B·topk` — with slack slots, the policy decides *which cold expert to sacrifice*, retaining likely-future-hot ones.

Prefill (tokens > threshold): bulk-overwrite ALL experts of the layer into pool slot `layer % 2` on `load_stream`, run GMM against the pool with identity log2phy. No paging decisions, no LRC, no prefetch.

---

## 5. The new feature: predictive next-layer prefetch

### 5.1 The idea in one sentence

> While layer L+1's attention is running, a background thread predicts which experts layer L+1's router will pick — by feeding **layer L's MoE input** through **layer L+1's gate weight** on the CPU — and copies the predicted-but-absent experts into L+1's decode slots ahead of time, so L+1's reactive `update_weights` finds them already resident.

This is exactly the **cross-layer gate** insight from FATE (see §7): hidden states entering adjacent MoE blocks are highly similar (the residual stream changes slowly), so layer L's gate input is a good proxy for layer L+1's gate input.

### 5.2 When prefetch starts

In `apply()`, **after** `fused_experts()` has been *submitted* to the compute stream (kernels are async — they have not necessarily run yet):

```python
if enable_expert_offload and not use_prefill_pool:
    mgr.trigger_next_layer_prefetch(layer, x)   # x = MoE INPUT of layer L
```

`trigger_next_layer_prefetch` is cheap and non-blocking on the main thread:
1. Skips entirely if `expert_prefetch_enabled` is off, or during ACL graph capture, or for the last MoE layer.
2. Lazily starts the daemon thread `"ExpertPrefetch"` on first call.
3. Records `compute_event` on the **main compute stream** (its position = right after the GMM submission).
4. Creates `_prefetch_layer_done[L+1] = threading.Event()`.
5. Enqueues `(L, x, compute_event)` and **returns immediately**.

### 5.3 What the worker thread does

```
worker loop:
  (L, x, compute_event) = queue.get()
  compute_event.synchronize()        # HOST-blocks until layer L's GMM has
                                     # actually FINISHED on the NPU
  predicted = predict_next_layer_experts(L, x)
  npu_evt   = _do_prefetch(L+1, predicted)     # H2D on _prefetch_stream
  _prefetch_layer_npu_event[L+1] = npu_evt
  _prefetch_layer_done[L+1].set()
```

Note the consequence of `compute_event.synchronize()`: the prefetch does **not** overlap with layer L's own GMM (by design — avoids bandwidth contention and guarantees `x` semantics). What it *does* overlap with is everything between the end of layer L's MoE and the start of layer L+1's MoE: shared-expert add/scaling, all-reduce, residual+norm, **layer L+1's full attention (MLA)**, gate, and `select_experts`. That window **is the lead time** (§5.7).

### 5.4 How it decides which experts to prefetch

`predict_next_layer_experts(L, x)`:

```python
hs_cpu = x.float().cpu()                       # D2H, tiny in decode (1–B tokens)
logits = hs_cpu @ gate_weights_cpu[L+1].T      # fp32 CPU matmul
probs  = logits.softmax(dim=-1)                # SIMPLIFIED routing
topk   = probs.topk(self.topk)                 # plain topk
return union of topk ids over all tokens
```

**Deliberately simplified vs the real router.** The real DeepSeek V4 routing in `select_experts` uses `use_grouped_topk=True`, `e_score_correction_bias` (aux-loss-free bias), `n_group`/`topk_group` group masking, and the config's `scoring_func` (sigmoid for the DeepSeek V3 family). The prediction uses none of these. The docstring acknowledges it: speed over fidelity, misses are caught by the reactive fallback. Expect the prediction's *ranking* to systematically diverge from real routing wherever the correction bias or group mask is decisive — this caps the achievable hit-rate well below FATE's 97% (which used the *true* gate function plus an over-fetch margin).

**DSv4-specific blind spot:** the first `num_hash_layers` MoE layers route by **token-id hash** (`gate.tid2eid` lookup), not by gate scoring. For those layers the gate-based prediction is essentially noise — yet, ironically, hash routing is *perfectly* predictable from `input_ids` alone. Untapped opportunity.

### 5.5 Where prefetched experts go + how eviction works

`_do_prefetch(L+1, predicted)` — runs on the worker thread, copies on the dedicated `_prefetch_stream`:

1. Snapshot `next_layer.log2phy` D2H → build `slot_owner` (slot → resident expert).
2. `need_to_load = predicted − resident`. If empty → done (no event).
3. `protected = predicted` (never evict a predicted expert). If every resident slot is protected → skip entirely.
4. For each expert to load:
   - Victim = `cache_policy.choose_victim(L+1, ...)` — **the same LRC hotness ranking, queried against layer L+1's statistics**, with `protected=predicted, loading=need_to_load`. (Fallback without LRC: any non-protected slot.)
   - Raw-storage H2D copy of w13, **`time.sleep(0.00025)`** (see §8), then w2, then scale/offset tensors, then fp32-scale refresh.
   - Update the host log2phy snapshot + slot_owner.
5. Write the modified log2phy back to `next_layer.log2phy` (device).
6. Record `completion_event` on `_prefetch_stream`, return it.

So prefetched experts live **directly in layer L+1's normal decode slots** (`w13_weight`/`w2_weight`), registered in the **same log2phy** the reactive path and the GMM use. Unified cache — no staging area, no second copy, no promote step. The prefetcher is just "an early client" of the exact same paging machinery, including the LRC victim policy.

Notably, the prefetch **does not call `cache_policy.observe()`** — predictions never pollute LRC's statistics. Only real routing (in `update_weights`) trains the policy. Clean separation: LRC learns from ground truth, prefetch consumes LRC for eviction.

### 5.6 When prefetch stops and on-demand load takes over (the handoff)

At the top of layer L+1's `update_weights` (decode branch):

```python
layer_done = self._prefetch_layer_done.get(L+1)
if layer_done is not None:
    layer_done.wait()            # host: wait for worker to finish submitting
    npu_event.synchronize()      # device: wait for the DMA copies + log2phy
                                 # writeback on _prefetch_stream to complete
```

Only *after* this barrier does the reactive path read log2phy and compute misses. Consequences:

- **Correctness is unconditional.** Whatever the prediction quality, the reactive path sees a consistent log2phy and loads anything still missing. A 0%-accurate predictor degrades performance, never correctness.
- **Hits become free.** `predicted ∩ actual` experts are already resident → counted as cache hits → zero on-demand copies for them.
- **Mispredicted experts are instantly evictable.** The reactive path's `protected` set is the *actual* `needed` set, so a wrongly-prefetched expert is a prime victim candidate the very same step (its LRC hotness is low since observe() never saw it).
- **Late prefetch blocks the main thread.** If the worker hasn't finished by the time L+1 reaches `update_weights`, the forward pass stalls on `layer_done.wait()`. There is no time-budget/abandon mechanism (contrast FATE, §7).

### 5.7 How many experts per layer, and the lead-time budget

**Count:** `|union of predicted top-k over batch tokens| − already resident`, capped by evictable (non-protected) slots. For decode batch B: at most `B × topk` candidates; B=1 → at most `topk` H2D copies, typically fewer since hot experts are already resident (that's LRC's job). There is no over-fetch margin (FATE prefetches *more* than top-k by confidence percentile) and no under-fetch budget (FATE caps at `n = T_window / t_expert`).

**Lead time** (window the prefetch must fit into):

```
prefetch window  =  [layer L GMM completes]  →  [layer L+1 update_weights]
                 =  rest of layer L's MoE epilogue (shared expert, scaling,
                    all-reduce, residual, norm)
                  + layer L+1 attention (MLA, indexer)
                  + layer L+1 gate + select_experts + topk D2H
```

In single-token decode this is on the order of a few hundred µs. The prefetch must fit: D2H of `x` + CPU fp32 matmul `[B,h]×[h,n_experts]` + per-expert H2D (w13+w2 int8 ≈ `3·h·I` bytes each + scales) **+ 0.25 ms hardcoded sleep per expert**. With k experts to load, the sleeps alone contribute `k × 0.25 ms` — for topk-sized misses this can already exceed the window, turning the §5.6 wait into a real stall. The compensating factor: even when late, those copies were copies the reactive path would have done serially anyway; the overlap converts *part* of them into shadow time. Net win requires reasonable prediction accuracy and a not-too-slow worker.

### 5.8 Sync/async design summary

| Mechanism | Domain | Purpose |
|---|---|---|
| `queue.Queue` | host→worker | submit prefetch jobs, main thread never blocks on submit |
| `compute_event` (NPU event) | worker | delay prefetch until layer L's GMM is done (no contention, defined `x`) |
| `_prefetch_stream` | NPU | H2D copies independent of compute stream and reactive `load_stream` |
| `_prefetch_layer_done[i]` (threading.Event) | main↔worker | host-side completion handshake per layer |
| `_prefetch_layer_npu_event[i]` (NPU event) | main | device-side DMA completion; synced at consume |
| `load_stream.synchronize()` | main | reactive loads complete before GMM uses slots |

Three NPU streams total: main compute, `load_stream` (reactive), `_prefetch_stream` (predictive).

---

## 6. Full-picture pseudo-code (decode step, both features on)

```
# ===== per decode step, layers 0..N-1 =====
for L in moe_layers:
    x   = attention_block(L)                     # MLA + norm
    logits = gate_L(x)                           # fp32
    topk_ids, topk_w = select_experts(logits)    # REAL grouped/sigmoid/bias routing

    # ---- update_weights(L) -------------------------------------------
    if pending_prefetch[L]:                      # issued at layer L-1
        wait host_event[L]; sync npu_event[L]    # barrier: slots+log2phy stable

    lrc.observe(L, topk_ids, topk_w)             # train policy on REAL routing
    misses = unique(topk_ids) - resident(L)      # prefetch hits vanish here
    for e in misses:                             # on-demand fallback
        victim = lrc.choose_victim(L, protected=needed)
        H2D copy expert e → slot(victim)  [load_stream]
        log2phy[victim]=-1; log2phy[e]=slot
    sync load_stream; log2phy → device

    # ---- MoE compute --------------------------------------------------
    submit fused_experts GMM (indexes slots via log2phy)   # async!

    # ---- trigger prefetch for L+1 (main thread, ~free) -----------------
    record compute_event on compute stream
    host_event[L+1] = Event()
    prefetch_queue.put((L, x, compute_event))

# ===== background thread, concurrently =====
loop:
    (L, x, ev) = queue.get()
    ev.synchronize()                             # layer L GMM finished
    pred = topk(softmax(x @ gateW[L+1].T))       # CROSS-LAYER GATE (simplified)
    for e in pred - resident(L+1):               # bounded by evictable slots
        victim = lrc.choose_victim(L+1, protected=pred)
        H2D copy expert e → slot  [prefetch_stream]   (+0.25ms sleep)
        update log2phy snapshot
    log2phy snapshot → device
    record npu_event[L+1]; host_event[L+1].set()
```

Timeline view (decode, one token):

```
main:    [attn L][gate L][upd_w L (wait?)][GMM L]·[attn L+1][gate L+1][upd_w L+1]...
                                            └trigger
worker:                                      [sync GMM L][predict L+1][H2D L+1 exp×k]
                                             └──────── lead window ────────┘
```

---

## 7. Does it borrow from FATE? — Yes, the core idea; not the refinements

FATE (*Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate*, [arXiv:2502.12224](https://arxiv.org/abs/2502.12224), WWW'26) is built on the observation that **adjacent layers' gate inputs are >83% cosine-similar**, so layer i's gate input fed to layer i+1's gate predicts i+1's experts cheaply on the CPU. v5.0's `predict_next_layer_experts` is precisely this mechanism — same input choice (gate input of layer L, i.e. the MoE block input `x`), same target (layer L+1's gate weights), same execution venue (CPU, overlapped with GPU/NPU compute), same misprediction story (on-demand fallback).

What v5.0 implements vs what FATE adds on top:

| Aspect | FATE | v5.0 |
|---|---|---|
| Cross-layer gate prediction | ✅ core idea | ✅ **borrowed directly** |
| Gate function used for prediction | the model's real gate scoring | simplified softmax+topk (no bias / groups / sigmoid) |
| Prefetch set sizing | confidence ≥ 75th percentile (over-fetch margin) → 97.15% accuracy | exactly top-k, no margin |
| Time-budgeted prefetch | offline-profiled `n = T_window / t_expert`, never blocks | unbounded; main thread **waits** if late |
| Cache strategy | shallow-favoring (fully cache shallow layers where prediction is weak) + ARC | uniform ndev slots/layer + custom LRC (freq+EMA+router+age) |
| Quantization | INT4 cache, INT2/INT4 popularity-hybrid IO | model is already W8A8; no extra IO tiering |
| Prefill | popularity-ordered expert transfers | separate full-copy pool (all experts, layerwise overwrite) |
| Misprediction | on-demand transfer | on-demand transfer (reactive path unchanged) |

So: **v5.0 = FATE's prediction kernel grafted onto this repo's pre-existing LRC paging machinery**, minus FATE's accuracy (percentile margin, true gate function) and scheduling (time budget) refinements, plus a threading design FATE doesn't need (FATE's prediction lives in the CPU side of a synchronous pipeline; v5.0 uses a daemon thread + dual events because the main thread is the NPU submission thread).

Interesting echo: FATE found *shallow layers predict poorly* and compensates by fully caching them. DSv4-Flash's shallow layers are **hash-routed** (`tid2eid`) — deterministically predictable from token ids, which would be a *better-than-FATE* fix for exactly the layers where the gate trick fails. v5.0 does neither.

---

## 8. Sharp edges & open questions (study findings, not yet verified on HW)

1. **`time.sleep(0.00025)` between w13 and w2 copies** (`_do_prefetch`). Smells like a band-aid for a race/ordering issue under the async copies (or PCIe queue flooding). Adds 0.25 ms × experts to every prefetch; likely the first thing to investigate/remove with a real fix (stream events between copies).
2. **Prediction fidelity** (§5.4): no `e_score_correction_bias`, no group masking, softmax instead of sigmoid scoring. Hit-rate ceiling unknown — *measure before optimizing*. The existing `[UPDATE-W]` cache_hit/cache_miss debug logs + `tools/analyze_expert_cache_log.py` can quantify it directly (run with/without `expert_prefetch_enabled`).
3. **Hash layers**: prediction is meaningless for `layer_idx < num_hash_layers`, yet those are perfectly predictable from `input_ids`. Free accuracy on the table.
4. **No prefetch for layer 0** (nothing triggers it) and **no cross-step prefetch** (last layer's trigger is skipped; the final hidden state could prefetch next step's layer 0).
5. **Prefill branch skips the prefetch barrier**: `update_weights` returns on the prefill path *before* the `layer_done` wait, so a pending prefetch event can linger across a decode→prefill→decode transition and be consumed stale. Probably benign (slots only get *more* populated) but worth a think for mixed batches.
6. **GIL pressure**: the worker's `x.float().cpu()`, CPU matmul, and copy submissions all contend with the main thread's Python between layers. The "true overlap" claim depends on how much of the copy path releases the GIL.
7. **Logging bug**: the `[PREFETCH] ... NO SLOTS` warning passes 3 args to a 2-placeholder format string → raises a logging error exactly when slots run out.
8. **Hardcoded `/6`** in the `[UPDATE-W]` hit-rate debug line (`len(already_there) / 6`) — assumes a fixed needed-set size; wrong for B>1 or different top-k.
9. **EPLB↔offload log2phy collision** (carried over from v2.0 study): offload owns `layer.log2phy` (`init_log2phy_for_offload`); dynamic EPLB writing the same map would race the prefetch thread's snapshot→mutate→writeback. Currently `enable_expert_offload` forces its own log2phy, but there's no guard.
10. **`register_gate_weights` ordering assumption**: gate list is built from `model.modules()` order of `DeepseekV4MoE` wrappers and must align index-for-index with `moe_layers` registration order. Holds today; fragile to model refactors (and to draft/spec-decode layers).

---

## 9. How to enable (config)

```json
"additional_config": {
  "expert_offload_config": {
    "expert_offload": true,
    "num_device_experts": 32,
    "num_device_layers": 2,
    "cache_policy_enabled": true,
    "expert_prefetch_enabled": true
  }
}
```

All three features compose: offload is the base, LRC needs `cache_policy_enabled`, prefetch needs `expert_prefetch_enabled` (default **false**). LRC is also what gives prefetch its eviction brain (`choose_victim` with L+1 stats); without LRC, prefetch falls back to arbitrary non-protected slots. Remember the standing v2.0 finding: with `ndev == B·topk` there are no slack slots and both LRC and prefetch are inert — `num_device_experts` must exceed the per-step working set to matter.

---

## 10. Pointers

- Reactive path: `expert_offload_manager.py:569` (`update_weights`) → `:656` (`_update_weights`)
- Prefetch: `:773` (`predict_next_layer_experts`), `:828` (`_prefetch_worker`), `:872` (`_do_prefetch`), `:995` (`trigger_next_layer_prefetch`)
- Forward hook (w8a8): `quantization/methods/w8a8_dynamic.py:305` (update) / `:385` (trigger)
- Lifecycle: `worker/model_runner_v1.py:3700` (`_register_offload_layers`)
- LRC: `expert_offload/lrc_policy.py`
- Prior study notes (v2.0 mechanism, shared-VRAM, EPLB collision): `notes/` on branch `moe_offload_v2.0`
- FATE paper: https://arxiv.org/abs/2502.12224
