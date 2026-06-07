# vLLM-Ascend — Co-Design Surfaces for Expert Offload & Proactive Expert Prefetching (`moe_offload_v2.0`)

> **What this note is.** A map of the *other* subsystems you must understand and co-design with before extending the expert-offload feature or building **proactive expert prefetching**. It is the "what else touches this" companion to:
> - `code-summary-vllm-ascend-expert-offload.md` (+ `-v2.md`) — how expert offload works.
> - `code-summary-vllm-ascend-kv-cache.md` — how the KV cache works and shares HBM with offload.
> - `expert-offload-optimization-ideas.md` (+ `-v2.md`) — triaged optimizations.
>
> **Tagging (same convention as the other notes):** `(FACT)` = read from code with a `file:line`; `(ANALYSIS)` = derived/interpretation, verify before relying. Lines from this snapshot; symbol names are the stable anchors. Findings here were produced by reading the code; the highest-stakes claims (EPLB↔offload `log2phy` aliasing, the duplicated offload hook, the absence of KV-cache prefetch) were **re-verified by direct read** and are marked `(FACT, verified)`.

---

## 0. ⚠️ Terminology — four (really five) distinct things people conflate

This is the disambiguation the rest of the note depends on. **"Offload" and "prefetch" are each overloaded.** Two orthogonal axes: *what data moves* (MoE expert **weights** vs attention **KV blocks** vs already-resident **weights into L2**) and *when* (**reactive** = on a miss/eviction, vs **predictive** = before it's needed). Three memory tiers are in play: **CPU RAM ↔ HBM ↔ on-chip L2 cache**.

| # | Concept | Moves WHAT | Tier / direction | Trigger | Status in this repo | Code |
|---|---|---|---|---|---|---|
| 1 | **Expert offloading** | MoE **expert weights** | CPU ⇄ HBM | reactive: decode pages on router miss; prefill bulk-loads | **Implemented** | `expert_offload/`, hook `fused_moe.py:186-191` |
| 2 | **KV cache offloading** | **KV blocks** (per-token attention state) | CPU ⇄ HBM | reactive: evict when HBM full; onload on prefix-cache hit | **Implemented (experimental/deprecated-tested)** | `kv_offload/`, `distributed/kv_transfer/.../cpu_offload/` |
| 3 | **Proactive expert prefetching** | MoE **expert weights** | CPU → HBM | **predictive**: stage experts *before* the router demands them | **NOT implemented (the target)** | — |
| 4 | **KV cache prefetch** | **KV blocks** | CPU → HBM | predictive | **NOT implemented** | — |
| 5 | **Weight prefetch** (adjacent, exists) | already-**resident weights** | HBM → **L2 cache** | reactive, intra-layer, data-dependency-gated | **Implemented** | `ops/weight_prefetch.py` |

**The three sentences to never blur:**
- **Expert offloading (1) ≠ KV cache offloading (2).** Different payloads (expert FFN weights vs attention K/V), different managers (`ExpertOffloadManager` vs upstream `OffloadingConnector`/`CPUOffloadingManager`), different policies (LRC hotness vs prefix-LRU), different fault model (routing "miss" vs prefix "miss"). They only meet at the **shared HBM/PCIe/pinned-RAM budget** (§5; KV-cache note §7). `(FACT)`
- **Proactive expert prefetching (3) is what you want to build, and it does NOT exist yet.** The repo's "prefetch" of *experts* today is only the **reactive** demand-paging of offload (1) plus the prefill pre-staging — neither predicts a future layer's experts. The expert-offload V2 note's "honest gap" is exactly this. `(FACT)`
- **KV cache prefetch (4) is NOT implemented either** — and (5) **weight prefetch** is a *different thing entirely* (it warms L2 with weights already in HBM, 18 MB cap; it never moves bytes across CPU↔HBM). Every `prefetch` string in `attention/mla_v1.py` / `sfa_v1.py` is `maybe_prefetch_mla_or_sla_weight_in_current_stream` = **weight** L2-prefetch of `fused_qkv_a_proj`/`o_proj`, *not* KV-block prefetch. `(FACT, verified — grep of attention/*.py)`

> **So if someone asks "is KV-cache prefetch already done?": No.** The nearest things are (a) `kv_offload` *reactively* loading a KV block H2D on a prefix-cache hit (`kv_offload/cpu_npu.py`), and (b) `weight_prefetch.py` warming L2 with resident weights. Neither is a predictive KV-block prefetcher. Building one would be a *separate* effort from proactive **expert** prefetching, though they'd share the async-stream machinery in §4-tier-2.

---

## 1. Navigation map (the co-design subsystems)

| Surface | Where | Why it matters to offload/prefetch |
|---|---|---|
| **EPLB** (the #1 collision) | `eplb/` (`eplb_updator.py`, `core/eplb_device_transfer_loader.py`, `core/eplb_worker.py`, `adaptor/vllm_adaptor.py`) | A *second* runtime expert-placement system that **aliases the same `log2phy` + same weight storage**; no guard. |
| Offload forward hook (duplicated!) | `ops/fused_moe/fused_moe.py:186-191` **and** `quantization/methods/w8a8_dynamic.py:255-260` | The integration point is **copy-pasted per quant method**; a prefetch change must be mirrored across all. |
| Model layer loop / router timing | `models/deepseek_v4.py:800-819, 932-933` | Determines exact-vs-predictive feasibility (it's predictive-only). |
| ACL-graph capture model | `compilation/acl_graph.py`; `expert_offload_manager.py:581-601`; `worker/v2/aclgraph_utils.py:144` | Host-side prefetch decisions must be capturable (host-func nodes, fixed addresses, event-waits). |
| Async transfer template | `kv_offload/cpu_npu.py:67-72, 206-232` | The reusable two-stream / event-pool / in-flight-deque pattern to copy for overlap. |
| Offload H2D primitive | `expert_offload_manager.py:76, 485-492, 524, 716` | The actual CPU→HBM copy (currently synchronous) to generalize. |
| Predictor substrate | `expert_offload/lrc_policy.py` (`LRCLayerState`) | Per-layer EMA/freq/router stats — the only signal available for prediction. |
| Spec decode / MTP | `spec_decode/`; `worker/model_runner_v1.py:434, 549, 1122, 3337-3345` | Makes "decode" steps multi-token → flips offload into the prefill pool; drafter MoE may be unmanaged. |
| Scheduler tokens/step | `core/scheduler_dynamic_batch.py:171-200, 505` | Sets `topk_ids.size(0)`, which selects cache-vs-pool path. |
| Expert parallelism (EP>1) | `ops/fused_moe/{token_dispatcher,moe_comm_method}.py`; `fused_moe.py:473-479` | Offload **bypasses** EP sharding; multi-device prefetch needs ownership + all-to-all awareness. |
| Per-step context plumbing | `ascend_forward_context.py:130,145-153,149-150,321-344` | Where prefetch state would live; precedent = `prefetch_mlp_*` flags. |
| Shared HBM/PCIe/pinned-RAM | KV-cache note §7 | Offload + KV-offload + prefetch all draw the same bus & host RAM. |

---

## 2. Tier 1 — Will silently corrupt state if ignored: **EPLB**

`eplb/` = **Expert-Parallelism Load Balancing**: periodically measures per-expert token load, computes a new physical placement (rearrange + replicate hot experts across EP ranks), moves expert weights between ranks, and rewrites the per-layer expert map / `log2phy`. It is a *second, independent* runtime expert-placement system — and it is the single most dangerous thing to co-design with.

**The collision is exact and unguarded `(FACT, verified)`:**
- EPLB and offload **alias the same `self.log2phy` tensor**. `fused_moe.py:468` sets `self.log2phy` from `init_eplb_config(...)`; then `fused_moe.py:476-479` **overwrites the same attribute** with `init_log2phy_for_offload(...)` when `enable_expert_offload`. Last-writer-wins; both then mutate that one tensor in place at runtime. `(FACT, verified — read fused_moe.py:462-486)`
- **No mutual-exclusion guard exists.** `ascend_config.py:48-52` builds `EplbConfig` and `ExpertOffloadConfig` independently; neither validator references the other; `grep eplb` in `expert_offload/` = 0 hits and `grep offload` in `eplb/` = 0 hits. They are mutually exclusive **by intent only**, not by enforcement. `(FACT, verified)`
- They also **share the same `w13_weight`/`w2_weight` device storage** — offload writes byte slabs via `untyped_storage()[slot*N:(slot+1)*N].copy_()` (`expert_offload_manager.py:681-686`); EPLB writes via `expert_tensor.copy_(buffer_tensor)` (`adaptor/vllm_adaptor.py:142-147`). Two writers, no arbitration. `(FACT, agent-sourced + consistent with verified aliasing)`

**Different physics, no shared ordering:**
- EPLB moves weights **D2D rank-to-rank over HCCL**: `dist.batch_isend_irecv(comm_op_list)` (`eplb_device_transfer_loader.py:84`) — *not* CPU↔HBM. Offload moves weights **H2D**. `(FACT)`
- EPLB has **no dedicated NPU stream** — transfers ride the EP comm group's default/compute stream and block on `req.wait()`. Offload uses its own `load_stream` (`expert_offload_manager.py:76`). There is **no event/barrier** coordinating the two. `(FACT)`
- Slot-id spaces are **incompatible**: EPLB `log2phy[logical] = local_slot + rank*valid_count` (EP-global, `eplb_utils.py:115`); offload `log2phy[logical] = on-device cache slot ∈ [0, num_device_experts)` (rank-local, `expert_offload/utils.py:31-35`). `(FACT)`
- The killer for any prefetcher: EPLB's `do_update_log2phy_map` does a **whole-tensor `copy_`** of `log2phy` (`vllm_adaptor.py:151`) at each of `num_moe_layers` commit steps — so it **clobbers every slot a prefetcher just wrote**, not a delta. `(FACT)`

**Cadence / wiring:** driven by a single `cur_iterations` counter (`eplb_updator.py:67-96`); a separate **spawned planner process** computes the plan (`eplb_worker.py:39-92`) and can silently revert illegal per-layer placements (`check_expert_placement`); commit happens from `forward_before`/`forward_end` in the model runner (`model_runner_v1.py:1918-1920, 2231-2233`). Enabled by `eplb_config.dynamic_eplb` (`ascend_config.py:556,591`). `(FACT, agent-sourced)`

**What a prefetcher MUST do:** either (a) **hard-assert EPLB-off** and own the `log2phy`/slot authority as an extension of `ExpertOffloadManager` (recommended first step — and add the missing `assert not (dynamic_eplb and expert_offload)` that should already exist); or (b) commit to a **unified placement table** that arbitrates EP-global ownership + rank-local cache slot + CPU residency behind one writer, respecting EPLB's per-layer commit window as a barrier — a much larger project. Never be an independent third writer to `log2phy`/weight storage. `(ANALYSIS)`

---

## 3. Tier 2 — Determines whether prefetch is feasible & where it hooks

### 3a. The execution model forces **predictive**, not exact, prefetch
The decoder runs a plain sequential Python loop — `for layer in islice(self.layers, ...): hidden_states, residual = layer(...)` (`deepseek_v4.py:932-933`). Inside each layer, the MoE router runs at `deepseek_v4.py:336` and exact `topk_ids` emerge at `fused_moe.py:166-181`. **Crucially, layer L+1's router input *is* layer L's not-yet-produced output** — so when L is computing, L+1's expert ids do not exist. There is no precomputed routing table or shared router (the `tid2eid` table at `deepseek_v4.py:283-291` is input-id-based and only for `layer_idx < num_hash_layers`). `(FACT, agent-sourced; loop verified by grep)`

**Consequence:** cross-layer prefetch can only use **historical per-layer statistics** — the `LRCLayerState` (`freq`, `ema`, `router_score`, `recent_queue`) in `lrc_policy.py` — to predict L+1's *likely-hot* experts during L's compute. Exact ids are only knowable intra-layer at `fused_moe.py:181`, which is already where paging fires, so there's no intra-layer slack to hide. This is *why* the expert-offload V2 note says naive same-layer reuse is a no-op and routes prefetch to a drift/cross-layer predictor. `(ANALYSIS, grounded)`

### 3b. ACL-graph capture constrains the prefetch decision
Under graph capture, data-dependent host work (which experts to page) cannot be ordinary Python (it would run once at capture). The offload manager expresses it as a **host-function node**: `if _EXTRA_CTX.capturing: torch_npu.npu._launch_host_func(stream, self._update_weights, args)` (`expert_offload_manager.py:594-601`), re-executed every replay. Constraints a capturable prefetcher must obey `(FACT)`:
- The compute stream must be registered once via `torch_npu.npu._subscribe_report(stream)` before any host func (`:581-585`); a prefetcher on a *new* stream must subscribe it too.
- All touched buffers must be **fixed-address** — staged into pre-allocated pinned CPU mirrors (`topk_ids_h`, `log2phy_h`, `:141-153`) and fixed NPU slots; ACL-graph debug asserts identical `data_ptr()`s across replays (`acl_graph.py:140-141`).
- On the replay path use **event waits (capturable)**, never `synchronize()`/`query()` (host polling, **not** capturable). Router-score is already dropped under capture (`:572`), so a predictive policy can't depend on it in graph mode.

### 3c. Reuse the right primitive — NOT `weight_prefetch.py`
`ops/weight_prefetch.py` is **L2-cache warming of HBM-resident weights** (`torch_npu.npu_prefetch`, hard cap `MAX_PREFETCH_WEIGHT_SIZE = 18*1024*1024`, `:15`); it **cannot move CPU→HBM** and has no per-expert granularity (the MoE path prefetches the whole already-resident `w13_weight`, `experts_selector.py:74`). Wrong tier. `(FACT, agent-sourced)` The correct building blocks already in-repo:
- **H2D primitive:** offload manager's `load_stream` + `untyped_storage().copy_()` (`expert_offload_manager.py:76, 485-492`), currently **synchronous** (`:524, :716`) — generalize to async/event-gated.
- **Async scheduler template:** `kv_offload/cpu_npu.py` is the gold standard — two dedicated streams (`d2h_stream`/`h2d_stream`, `:67-68`), per-direction in-flight `deque`s of `Transfer` records (`:71-72`), an event pool (`:74-75, 134-140`), ordering via `stream.wait_event(last.end_event)` / `wait_stream(current_stream())` (`:206-211`), record bracketing events (`:213-218`), and non-blocking completion via `end_event.query()` (`:232-252`). **Overlap recipe:** issue predicted H2D on a separate prefetch stream during layer L, record an `end_event`, and at L+1's consume point replace the host-blocking `synchronize()` (`expert_offload_manager.py:716`) with `compute_stream.wait_event(prefetch_end_event)`; fold prefetch hits into the existing `already_there / need_to_load` split (`:630-632`). `(FACT, agent-sourced)`

### 3d. Natural hook points
1. **Primary** — inside `update_weights` for layer L, after L's paging completes (`expert_offload_manager.py:534-603`): enqueue a *predicted* H2D for L+1/L+2 from `LRCLayerState[L+1]`.
2. **Consume** — top of L+1's `update_weights`: swap `load_stream.synchronize()` for an event wait; mispredictions fall back to the existing synchronous miss path.
3. **Earliest overlap** — `fused_moe.py:190`, right after L's `update_weights` and before L's `fused_experts` (`:274`, the longest compute), to hide the prefetch behind L's matmul. `(ANALYSIS, grounded)`

---

## 4. Tier 3 — Breaks the "decode = 1 token = cache path" assumption

Path selection is `num_tokens = topk_ids.size(0); if num_tokens > offload_threshold → prefill pool else decode cache`, with `offload_threshold = num_device_experts // top_k` (`expert_offload_manager.py:38`; selectors at `fused_moe.py:189-192` and `w8a8_dynamic.py:260`). At the reference `ndev == topk`, threshold = 1, so **anything with ≥2 tokens leaves the cache**.

### 4a. Spec decode / MTP (`spec_decode/`)
A "decode" step processes `1 + K` tokens per request (`K = num_speculative_tokens`): `decode_token_per_req = 1 + spec_token_num` (`model_runner_v1.py:549`), verify step `num_sampled_tokens = num_draft_tokens + 1` (`:1122`). With batch B, the MoE sees `B·(1+K)` rows → routed to the **prefill bulk-load pool**, not the demand cache, whenever K≥1 or B≥2. `(FACT, agent-sourced)`
- **Drafter MoE may be unmanaged (silent-bug risk):** `_register_offload_layers()` walks only the *target* model and runs *before* the drafter loads (`model_runner_v1.py:3337` vs `:3340-3345`). A DeepSeek-MTP drafter that contains an `AscendFusedMoE` would have `enable_expert_offload=True` but no CPU buffers / no manager registration → `moe_layers.index(layer)` raises, handled by early `return 0`, so it runs on whatever weights are resident. `(ANALYSIS, agent-sourced — verify if MTP is in scope)`

### 4b. Scheduler tokens-per-step (`core/scheduler_dynamic_batch.py`)
`total_num_scheduled_tokens` (`:505`) becomes `topk_ids.size(0)`. Steps with >1 token (⇒ no cache): **batched decode** (B>1), **spec/MTP** (1+K each), **chunked-prefill tails / mixed decode+prefill steps** (`d_lst + p_lst`, `:177-179`), **dynamic-batch budget refinement** (`:107-119`). Only strict B=1, no-spec, no-co-scheduled-prefill uses the cache. `(FACT, agent-sourced)`

**Constraint on a prefetcher:** "decode step" is not a reliable signal. Branch on the actual `topk_ids.size(0)` vs `offload_threshold`, ideally reading the *scheduler output one step ahead* to know whether the next forward is cache- or pool-served and which requests (hence experts) are in-batch. And per project memory, raise `num_device_experts` above `top_k` first or the cache path is unreachable at realistic batch sizes. `(ANALYSIS)`

### 4c. Expert parallelism EP>1 (`token_dispatcher.py`, `moe_comm_method.py`)
With EP off, each rank owns a contiguous slice `local_num_experts = global // ep_size` (`fused_moe.py:474`). **Offload bypasses EP sharding entirely:** that assignment is *skipped* when `enable_expert_offload` (`fused_moe.py:473`), and offload instead maps *all* global experts to `ndev` rank-local slots (`expert_offload/utils.py:25-35`). The whole `expert_offload/` package has **zero** `ep_size`/`ep_rank`/`mc2_group` references — `num_device_experts` is treated as a *global*, single-rank budget. The reference `tp1/dp1/EP1` is the only coherent config today. `(FACT, agent-sourced; consistent with verified fused_moe.py:473-479)`

**Constraint:** a multi-device prefetcher must become ownership-aware — know each rank owns `[ep_rank·local, (ep_rank+1)·local)` (`token_dispatcher.py:369-370`), that under MC2 a routed expert may live on a *remote* rank (fetched via all-to-all, not local CPU), and reconcile the EP `expert_map` (which rank) vs the offload `log2phy` (which slot) into a unified "global expert → (owner_rank, slot|CPU)" table that **does not exist yet**. `(ANALYSIS, agent-sourced)`

---

## 5. Tier 4 — Plumbing you'll inevitably touch

- **Duplicated offload hook across quant methods `(FACT, verified)`.** The `update_weights` call + path selector is copy-pasted: unquantized `fused_moe.py:186-191` **and** W8A8 `quantization/methods/w8a8_dynamic.py:255-260` (and likely other `quantization/methods/*` `apply()` bodies). Any prefetch change must be mirrored across all of them — there is no single chokepoint at the quant-method layer. Consider refactoring the hook into one shared helper as a *prerequisite* to prefetch work, so you patch one place.
- **Shared HBM / PCIe / pinned-RAM budget** (KV-cache note §7). Expert offload, KV-cache offload, and a prefetcher all draw the **same DMA bus** and **pinned host RAM**; prefetch *adds* H2D traffic competing with KV onload and demand paging. Budget the three jointly; "offload everything" is not free. `(ANALYSIS)`
- **`ascend_forward_context.py`** carries per-step state (`num_tokens` `:130`, `capturing` `:106`, `in_profile_run` `:102`, `is_draft_model*` `:152-153`, existing `prefetch_mlp_*` flags `:149-150`, allowlist `:321-344`). It's the precedent and the place to add expert-prefetch state. Note no field auto-advances the per-layer decode index — derive it via `moe_layers.index(layer)` (`expert_offload_manager.py:566`) or `layer.layer_id`. `(FACT, agent-sourced)`

---

## 6. Invariants & assumptions (ground truth for agents)

1. **`self.log2phy` is one shared, in-place-mutated tensor** that EPLB and offload both claim (`fused_moe.py:468` then `:476-479`). EPLB rewrites it whole-vector (`vllm_adaptor.py:151`). Any new writer must be the sole authority or hard-gate the others. `(FACT, verified)`
2. **No guard enforces EPLB ⊥ offload mutual exclusion.** Add `assert not (dynamic_eplb and expert_offload)`. `(FACT, verified)`
3. **The offload hook is duplicated per quant method** (`fused_moe.py:186`, `w8a8_dynamic.py:255`, …). Patch all or refactor first. `(FACT, verified)`
4. **Cross-layer expert prefetch is predictive-only** — exact future ids don't exist before the future layer's router runs (`deepseek_v4.py:932`). `(FACT)`
5. **Capturable host work = `_launch_host_func` on a `_subscribe_report`ed stream, fixed-address buffers, event-waits not `synchronize`/`query`** (`expert_offload_manager.py:581-601`). `(FACT)`
6. **`weight_prefetch.py` is L2 prefetch of resident weights (18 MB cap), not CPU→HBM staging.** Reuse the offload `load_stream` H2D + `kv_offload/cpu_npu.py` async template instead. `(FACT)`
7. **There is NO KV-cache prefetch and NO proactive expert prefetch today.** `kv_offload` onload is reactive (prefix-hit); offload paging is reactive (router miss). `(FACT, verified)`
8. **Path is selected by `topk_ids.size(0)` vs `ndev//topk`** — spec/MTP/batched/chunked steps go to the prefill pool, not the cache. `(FACT)`
9. **Offload bypasses EP sharding** (`fused_moe.py:473`); multi-device needs an ownership-aware unified table that doesn't exist. `(FACT)`
10. **Offload, KV-offload, and prefetch share HBM + PCIe + pinned host RAM** — co-budget them (KV-cache note §7). `(ANALYSIS)`

---

## 7. Recommended sequencing for a prefetcher (grounded)

1. **First, make the cache real and the integration clean:** raise `num_device_experts > top_k` (project memo `[N1]`); refactor the duplicated offload hook (§5) into one helper; add the EPLB-off assert (§2).
2. **Stand up the async H2D plumbing:** generalize the offload `load_stream` copy (currently synchronous) using the `kv_offload/cpu_npu.py` event/double-buffer pattern (§3c) — measured on the eager path first.
3. **Add the predictor on the `LRCLayerState` substrate** (§3a) — drift/cross-layer signal, since same-layer hotness can't predict misses (V2 note `[A①]`).
4. **Hook cross-layer** at `update_weights`-L → consume-at-L+1 (§3d); keep the synchronous miss path as the misprediction fallback.
5. **Defer/scope-out** EP>1 (§4c) and MTP-drafter coverage (§4a) explicitly until the single-rank decode-path version works.
6. **Validate under ACL-graph capture** (§3b), not just `--enforce_eager`, before trusting throughput numbers.

---

## 8. Cross-references
- Expert offload mechanics: `code-summary-vllm-ascend-expert-offload.md` (+ `-v2.md`).
- KV cache + the shared-HBM story: `code-summary-vllm-ascend-kv-cache.md` (§7 is the joint VRAM/PCIe/pinned-RAM budget).
- Optimization hypotheses (incl. the prefetch E1 gap): `expert-offload-optimization-ideas.md` (+ `-v2.md`).
- **Unverified-here internals:** upstream `OffloadingConnector`/`CPUOffloadingManager`, `memory_profiling`, and the EPLB planner-process internals live in `vllm`/spawned processes; agent-sourced claims marked as such — re-verify against the pinned versions before relying on exact behavior.
