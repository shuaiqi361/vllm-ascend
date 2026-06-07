# vLLM-Ascend KV Cache — Code Study & Reference (`moe_offload_v2.0`)

> **Companion to the expert-offload notes.** Read this alongside `code-summary-vllm-ascend-expert-offload.md` (+ its V2). That pair explains *demand-paging MoE experts*; this file explains *the KV cache* — what method is implemented, how it is wired into the forward pipeline, how it decides what to keep vs. evict when VRAM runs out, and — the reason this note exists — **how KV cache and expert offload jointly share the one NPU HBM budget**. The expert-offload feature's entire point is to *free HBM so the KV cache can grow* (`expert-offload §1`); this note is the other half of that sentence.

**Source tree:** `/Users/keyi/Documents/code/vllm-ascend` @ branch `moe_offload_v2.0`.
**Reference workload (shared with the expert-offload notes):** `vllm serve DeepSeek-V2-Lite`, `--enforce_eager`, `tp=1 dp=1 --enable-expert-parallel`, `expert_offload_config={expert_offload:true, num_device_experts:6}`. DeepSeek-V2-Lite uses **MLA** attention, so the KV-cache shapes below are the *MLA* (compressed-latent) shapes, not the dense `[2, …]` shapes.

**Tagging convention (same as the expert-offload V2 note):** `(FACT)` = read directly from code with a `file:line`; `(ANALYSIS)` = derived/opinion, verify before relying on it. Line numbers are from this snapshot and may drift; symbol names are the stable anchors.

> ⚠️ **"Offload" is overloaded in this repo — three unrelated things share the word.** (1) **Expert offload** = MoE expert *weights* paged HBM↔CPU (`ExpertOffloadConfig`, the other notes). (2) **KV-cache CPU offload** = KV *blocks* spilled HBM↔CPU (`vllm_ascend/kv_offload/`, §6 here). (3) **UVA weight offload** = upstream vLLM's `cpu_offload_gb`, which is **asserted off** on the Ascend path (`model_runner_v1.py:4083` `assert ...cpu_offload_gb == 0`). They are separate features with separate configs; do not conflate. `(FACT)`

---

## 0. Navigation map (for quick jumps)

| Concern | Where |
|---|---|
| **The VRAM split formula** (weights+activation+nontorch → KV budget) | `worker/worker.py` `determine_available_memory` (`:327-397`) |
| Weights-memory measurement (incl. shrunken offload tensors) | `worker/model_runner_v1.py` `load_model` `DeviceMemoryProfiler` (`:3329-3361`) |
| KV budget → block count | `worker/model_runner_v1.py` `_reshape_kv_cache_tensors` `num_blocks = bytes // page_size` (`:3788, 3850, 3869`) |
| KV tensor allocation | `_allocate_kv_cache_tensors` (`:3569`), `initialize_kv_cache_tensors` (`:3486`) |
| Per-layer KV spec (block_size, heads, dtype) | `get_kv_cache_spec` (`:4202-4311`); MLA dims `_get_attention_kv_cache_dims` (`:3548-3558`) |
| Dense paged-attn KV layout `[2,nblk,bs,H,D]` | `attention/attention_v1.py` `get_kv_cache_shape` (`:100-107`) |
| MLA compressed-latent layout `[nblk,bs,H,D]` | `attention/mla_v1.py` `get_kv_cache_shape` (`:94-102`) |
| Block table + slot mapping (the page table) | `worker/block_table.py` (`compute_slot_mapping` `:138-161`) |
| NPU paged-attention kernel call | `attention/attention_v1.py` `_npu_paged_attention` (`:977-987`) |
| Eviction when blocks run out (preempt+recompute) | `core/scheduler_dynamic_batch.py:238-258`, `scheduler_profiling_chunk.py:238-261`, `recompute_scheduler.py:320-376` |
| Prefix caching over compressed MLA | `core/single_type_kv_cache_manager.py` `CompressAttentionManager` (`:18-204`) |
| `block_size` coercion to 128 | `utils.py` `refresh_block_size` (`:1271-1306`) |
| KV → CPU offload tier (HBM↔CPU blocks) | `kv_offload/npu.py`, `kv_offload/cpu_npu.py` |
| Disaggregation / remote KV stores | `distributed/kv_transfer/` (`__init__.py:21-65` connector registry) |
| INT8 ("C8") KV quant | `attention/attention_v1.py` `AscendC8AttentionBackendImpl` (`:1149`); `quantization/methods/kv_c8.py` |
| Config: `gpu_memory_utilization`, `kv_cache_memory_bytes` | `worker/worker.py:273, 340-353, 390` |

---

## 1. What the KV cache is, and which method is implemented

**In one line: PagedAttention** — the KV cache is split into fixed-size **blocks (pages)** of `block_size` tokens, stored non-contiguously in HBM, and addressed per-request through a **block table** (a page table). This is the method from the vLLM paper:

> **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.** Core idea: treat the KV cache like OS virtual memory — fixed-size pages, a page table per sequence, near-zero fragmentation, and cheap sharing/copy-on-write of pages across sequences. vLLM-Ascend is the V1-engine implementation of exactly this, with NPU kernels.

The paging is genuine and visible in the code:
- **Block table** documented as "Block addresses per sequence (Seq id -> list of physical block). (batch_size, max_blocks_per_seq)" (`attention_v1.py:189-191`). `(FACT)`
- **Slot mapping** documented with the canonical PagedAttention example: "if `slot_mapping` is [35, 2, 17] and the block size is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0, and 1st slot in block 1" (`attention_v1.py:193-198`). `(FACT)`
- The decode kernel takes the page table directly: `torch_npu._npu_paged_attention(..., block_table=attn_metadata.block_tables, context_lens=attn_metadata.seq_lens, ...)` (`attention_v1.py:977-987`). `(FACT)`

**On top of plain PagedAttention, this repo implements several KV-footprint-reduction methods** (each its own backend, each a documented research technique). Which one runs depends on the model:

| Backend | Model class | KV-cache idea | Paper |
|---|---|---|---|
| `AscendAttentionBackend` (+ `C8` INT8 variant) | GQA/MHA (Qwen, Llama, …) | Standard paged KV; optional static per-channel **INT8** KV quant | PagedAttention (SOSP'23) |
| `AscendMLABackend` | DeepSeek-V2/V3 (**the ref model**) | Cache a **compressed latent** (`kv_lora_rank`+`qk_rope_head_dim`) instead of full K/V | **MLA**, DeepSeek-V2 (Liu et al., 2024) |
| `AscendSFABackend` | DeepSeek sparse | MLA + a **lightning indexer** that selects top-k KV chunks; optional INT8 indexer cache | DeepSeek-V3.2 sparse attn |
| `AscendDSABackend` | `deepseek_v4` | **DeepSeek Sparse Attention** — `index_topk` selects which KV to attend; separate indexer cache | DeepSeek Sparse Attention |
| `kvcomp_attn` (opt-in) | any long-context | **Hamming-hash block selection** — pick top-k KV blocks by binary-code distance | KV-comp / hash-attention family |

`(FACT)` for the backend names/shapes (`attention_v1.py:100-107`, `mla_v1.py:94-102`, `sfa_v1.py:86-94`, `dsa_v1.py:146-151`, `kvcomp_attn/attention_utils.py:72-94`); `(ANALYSIS)` for the paper attributions.

**KV-cache tensor layouts (per attention layer):**
- **Dense** (`AscendAttentionBackend`): `kv_cache.shape = [2, num_blocks, block_size, num_kv_heads, head_size]` — leading `2` holds key and value; `key_cache, value_cache = kv_cache[0], kv_cache[1]` (`attention_v1.py:100-107, 1128`). `(FACT)`
- **MLA** (`AscendMLABackend`, the ref model): `[num_blocks, block_size, num_kv_heads, head_size]` — **no leading-2**. Two tensors: `kv_cache[0]` = compressed latent `kv_c` (dim `kv_lora_rank`, 512 for DeepSeek), `kv_cache[1]` = decoupled RoPE key `k_pe` (dim `qk_rope_head_dim`, 64). The full value V is **never materialized** in the cache — it is reconstructed from the latent. (`mla_v1.py:94-102, 1294-1304`; dims from `_get_attention_kv_cache_dims` returning `kv_lora_rank, qk_rope_head_dim` `:3557-3558`.) `(FACT)` This is the MLA memory win: ~576 cached dims vs the many-hundreds a dense MHA cache would need.

---

## 2. Where the KV cache hooks into the pipeline

Per attention layer, every forward step:

1. **Compute Q,K,V** for the new tokens (attention projection).
2. **Write new KV into pages** — `reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)` scatters each new token's K/V into its `(block_id, offset)` slot (`attention_v1.py:1024-1056`). The `slot_mapping` came from `block_table.compute_slot_mapping` (a Triton-style kernel turning `(positions, block_table) → per-token slots`, `block_table.py:138-161`). `(FACT)`
3. **Attention over the pages** — prefill uses a fused/flash path; decode uses `_npu_paged_attention(..., block_table, context_lens, ...)`, which gathers KV by walking the block table (`attention_v1.py:977-987`). `(FACT)`

The **block table** lives on the worker (`worker/block_table.py`): an int32 `[max_num_reqs, max_num_blocks_per_req]` table plus `num_blocks_per_row`; rows are appended/cleared/swapped as requests gain/lose blocks (`:89-136`). It also supports **hybrid logical/physical blocks**: when a backend's kernel block size (e.g. DSA's 8/32) is smaller than the physical 128-page, one physical block maps to several logical blocks (`_convert_physical_to_logical_blocks`, `:53-84`). `(FACT)`

**Ownership of allocation/eviction is upstream vLLM.** vLLM-Ascend rides the **V1 engine**: the `KVCacheManager` / `BlockPool` / `SingleTypeKVCacheManager` that decide *which* physical block a logical block maps to, hash prefix blocks, and LRU-evict cached blocks all live in upstream `vllm`. The Ascend repo customizes (a) the NPU attention kernels (§1), (b) one KV-manager subclass for compressed MLA prefix caching (§5), and (c) three optional schedulers (§4). `(ANALYSIS, from import structure)`

---

## 3. The VRAM split — how much HBM becomes KV cache (THE core mechanism)

This is the heart of the joint story with expert offload. KV cache is **whatever HBM is left over** after weights and activations, capped by `gpu_memory_utilization`. The decision happens once, at startup, in `worker.py:determine_available_memory` (`:327-397`).

**The exact formula** (`worker.py:357-397`, verified verbatim):

```
requested_memory       = total_memory * gpu_memory_utilization          # :273  (the cap)
non_kv_cache_memory     = weights_memory                                  # :372-374
                        + activation_peak (pre-graph torch peak increase)
                        + non_torch_increase (ACL/HCCL/driver allocations)
available_kv_cache_bytes = requested_memory − non_kv_cache_memory         # :390
```

Then the engine turns bytes into blocks: `num_blocks = available_kv_cache_bytes // (Σ per-layer page_size_bytes)`, re-derived/verified in `_reshape_kv_cache_tensors` (`num_blocks = sum_page_size_bytes // page_size_bytes`, `:3788/3850/3869`, asserted against `kv_cache_config.num_blocks`). `(FACT)`

**The three inputs, and where each comes from:**
- `weights_memory = int(self.model_runner.model_memory_usage)` (`:359`) — set during `load_model` as `m.consumed_memory` of a `DeviceMemoryProfiler` wrapping the *entire* model load (`model_runner_v1.py:3329, 3360`). **This is the number expert offload shrinks** (§5). `(FACT)`
- `activation_peak` — measured by running `profile_run()` (a dummy max-shape forward) inside the `memory_profiling` context and reading `torch.npu.memory_stats(...)["allocated_bytes.all.peak"]` **before** graph capture (so the ACL-graph pool isn't double-counted as activation) (`:361-371`). `(FACT)`
- `non_torch_increase` — driver/ACL/HCCL allocations the torch allocator doesn't see, computed by the upstream `memory_profiling` context. `(ANALYSIS)`

**Manual override:** if `--kv-cache-memory` (`cache_config.kv_cache_memory_bytes`) is set, the whole formula is skipped — it still runs `profile_run()` to compile the model but returns the user value verbatim and **ignores `gpu_memory_utilization`** (`:340-353`). `(FACT)` Use this only when you want exact, reproducible KV sizing.

**What decides "what to cache vs. not" at the byte level:** nothing model-aware. The KV budget is a flat byte pool; the engine fills it with as many uniform 128-token pages as fit. "What to cache" is decided *at runtime* by the scheduler/block-manager (§4), not at sizing time.

---

## 4. When HBM runs out at runtime: eviction = preempt + recompute (not swap)

The sizing in §3 fixes a finite `num_gpu_blocks`. During serving, when a step needs a new block and the pool is empty, `kv_cache_manager.allocate_slots(...)` returns `None`, and the scheduler must free blocks. The **default behavior is preempt-and-recompute**, not swap-to-CPU:

- Pick the lowest-priority running request (`max(priority, arrival_time)` under PRIORITY policy, else `self.running.pop()` FCFS), then `kv_cache_manager.free(victim)`, `encoder_cache_manager.free(...)`, set `status = PREEMPTED`, **`num_computed_tokens = 0`**, and `waiting.prepend_request(victim)` (`scheduler_dynamic_batch.py:238-258`; `scheduler_profiling_chunk.py:238-261`). `(FACT)`
- `num_computed_tokens = 0` ⇒ the victim's KV is **thrown away and recomputed from scratch** when it is rescheduled. This is classic vLLM "recompute-style" preemption — cheaper than it sounds because prefill is compute-bound and parallel. `(ANALYSIS)`

vLLM-Ascend uses **upstream vLLM's V1 `Scheduler` by default**; three optional subclasses are swapped in via `platform.py:500-527` based on `ascend_config`:

| Scheduler | Trigger | What it adds (eviction logic is identical) |
|---|---|---|
| `SchedulerDynamicBatch` | `SLO_limits_for_dynamic_batch != -1` | Dynamically resizes the chunked-prefill token budget from a profiled table (`scheduler_dynamic_batch.py:35-122`) |
| `ProfilingChunkScheduler` | `profiling_chunk_config.enabled` (pp>1) | Fits a quadratic to measured prefill latency → predicts optimal chunk size (`scheduler_profiling_chunk.py:47-197`) |
| `RecomputeScheduler`/`Async…` | `recompute_scheduler_enable` (PD-disagg) | PD-disaggregation-aware preempt: in a decode node, drops the victim back to the prefill node via `RecomputeReqInfo` (`recompute_scheduler.py:336-346`) |

None of these change the block layout or invent a new eviction *algorithm* — they tune chunking and disaggregation. LRU eviction of **cached prefix blocks** (not running requests) is upstream `BlockPool`. `(FACT)` for triggers/files; `(ANALYSIS)` for the "identical eviction" claim.

**Two ways to avoid recompute instead of paying for it:** (a) keep reusable prefixes alive via prefix caching (§5); (b) spill blocks to a CPU tier so they can be reloaded instead of recomputed (§6). Both are opt-in.

---

## 5. Prefix caching & the compressed-MLA manager

Prefix caching (block reuse across requests sharing a prompt prefix) is **supported** via upstream vLLM's block-hash machinery, with one Ascend-specific extension:

- `CompressAttentionManager` (`core/single_type_kv_cache_manager.py:18-204`) subclasses upstream `FullAttentionManager` to make prefix caching work over **compressed MLA** caches: it divides token counts by `compress_ratio` before allocating/caching blocks (`:37-47, 132-157`), `block_pool.touch(...)` pins cache hits against eviction (`:95-97`), and it re-implements `find_longest_cache_hit` walking the block-hash chain (`:160-204`). Selected when `MLAAttentionSpec.compress_ratio > 1` (`:207-212`). `(FACT)`
- Enabling prefix caching (or chunked prefill) **forces `block_size = 128`** (`utils.py:1297-1301`). `(FACT)`
- Prefix caching is **force-disabled** for `TRITON_MLA`/`FLASHINFER` kernels (`models/layer/attention/layer.py:89-94`). `(FACT)`

**Why `block_size` is effectively non-tunable (always 128):** `refresh_block_size` (`utils.py:1271-1306`) defaults unset→128, hard-pins `deepseek_v4`→128, forces 128 when prefix-cache/chunked-prefill is on, and clamps >128→128 under xlite. Every dense backend also advertises `get_supported_kernel_block_sizes() == [128]` (`attention_v1.py:139`, `mla_v1.py:114`, `sfa_v1.py:106`). 128 is the Ascend cube/attention **tiling granularity** — the paging, sparse-chunking, and NZ-format paths are all written against 128-token pages. (DSA additionally allows 8/32 *kernel* blocks that pack into 128-wide physical pages via the hybrid block table, §2.) `(FACT)` for the coercion; `(ANALYSIS)` for the hardware rationale (no comment states it).

---

## 6. KV → CPU offload: the second tier (direct analogue of expert offload)

`vllm_ascend/kv_offload/` is to the KV cache what `expert_offload/` is to MoE weights: **a CPU tier that holds blocks evicted from HBM and reloads them on a hit**, trading PCIe bandwidth + pinned host RAM for effective capacity. It plugs into upstream vLLM's `OffloadingConnector` framework.

- **`npu.py` — `NPUOffloadingSpec(OffloadingSpec)`** (glue/registration): reads `num_cpu_blocks` from `kv_connector_extra_config` (`:20-23`); scheduler side returns an upstream `CPUOffloadingManager` that owns the CPU block pool + **LRU policy** (`:38-42`); worker side builds `CpuNpuOffloadingHandler` and yields the two directions `GPU→CPU` (offload/D2H) and `CPU→GPU` (onload/H2D) (`:45-63`). `(FACT)`
- **`cpu_npu.py` — `CpuNpuOffloadingHandler`** (the transfer engine): allocates a matching **pinned** CPU tensor per layer's KV (`:79-108`, `pin_memory` when available); uses **two dedicated NPU streams** `d2h_stream`/`h2d_stream` with in-flight `Transfer` deques and an event pool (`:67-75`); each transfer is **one batched DMA** `torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, direction)` (`:217`, `direction` 0=H2D / 1=D2H). Correctness ordering: D2H waits on the compute stream (`stream.wait_stream(current_stream())`, `:206-207`) and serializes within a direction via events (`:208-211`). `(FACT)`
- The native op `swap_blocks_batch` (`csrc/torch_binding.cpp:121, 145-160`) maps direction→`ACL_MEMCPY_*` and, on CANN 8.5+, uses **`aclrtMemcpyBatchAsync`** for a single batched async copy (`:162-195`). *(This is the same batched-memcpy capability the expert-offload note flags as "detected but not yet used by the Python path" — here it **is** used.)* `(FACT)`

**How it's enabled:** not a dedicated flag — pass upstream `KVTransferConfig`: `kv_connector="OffloadingConnector"`, `kv_role="kv_both"`, `kv_connector_extra_config={num_cpu_blocks:…, spec_name:"NPUOffloadingSpec", spec_module_path:"vllm_ascend.kv_offload.npu"}` (docs `kv_cache_cpu_offload.md:23-32`). `(FACT)`

**What it actually buys (important nuance):** the upstream `OffloadingConnector` flow is **prefix-cache-oriented** — it stores *completed/inactive* blocks and reloads them on a *prefix hit*. It does **not** page the active window of a single in-flight request in and out token-by-token. So it expands capacity by (a) preserving reusable prefixes the HBM pool would otherwise evict and (b) freeing HBM blocks sooner — converting would-be **recomputes** (§4) into **CPU reloads**. (`docs/.../kv_cache_cpu_offload.md:5,76-77`.) `(FACT/ANALYSIS)`

**Adjacent, different features in `distributed/kv_transfer/`** (connector registry `__init__.py:21-65`): P/D-disaggregation over **Mooncake** (`kv_p2p/`), and remote KV **stores** (`kv_pool/`: Mooncake / Memcache / Yuanrong backends, UCM, LMCache, plus a second shared-memory `CPUOffloadingConnector` sized by `cpu_swap_space_gb` default 800 GB). These move KV between *nodes/stores*, not strictly HBM↔local-CPU. Note the e2e CPU-offload test is currently `@pytest.mark.skip(reason="cpu offload connector is deprecated.")` (`tests/e2e/singlecard/test_cpu_offloading.py:131`) — that skip targets the `kv_pool/cpu_offload` connector, not the `kv_offload/` spec, but treat both CPU-offload paths as **experimental** on this branch. `(FACT)`

**KV quantization (orthogonal HBM saver):** INT8 only — "C8" (`AscendC8AttentionBackendImpl`, `attention_v1.py:1149`; method `quantization/methods/kv_c8.py`), enabled when `quant_config.kv_cache_type != ""`. For MLA, only the **K** cache is quantized to preserve accuracy (`modelslim_config.py:612-621`). **No fp8 KV path exists** (fp8/mxfp8 are weight/activation only). `(FACT)`

---

## 7. ★ How KV cache and expert offload jointly share the VRAM cap

This is the question the note exists to answer. The two features are **coupled through exactly one quantity — `weights_memory` in the §3 formula — and through two shared physical resources (the PCIe bus and pinned host RAM). There is no explicit coordination code between them.**

### 7.1 The coupling is automatic, via profiling (not hand-wired)

Trace the startup ordering (`(FACT)`, file:line):

```
worker.init_device           snapshot total/free HBM; requested_memory = total*util   (worker.py:272-273)
worker.load_model
  └ DeviceMemoryProfiler {                                                              (model_runner_v1.py:3329)
        get_model(...)        # MoE device tensors ALREADY allocated small:
                              #   create_weights sees num_device_experts, not 64        (fused_moe.py:391-400, 518)
        _register_offload_layers()   # CPU mirror (off-HBM) + prefill pool (on-HBM)     (:3336-3337)
    }
  model_memory_usage = m.consumed_memory   # SMALLER, because experts live on CPU       (:3360)
worker.determine_available_memory
        weights_memory = model_memory_usage                                             (worker.py:359)
        available_kv = requested_memory − (weights + activation + nontorch)             (worker.py:390)
→ engine: num_gpu_blocks = available_kv // page_size   →  MORE KV blocks
```

The device MoE tensor is **born small** — `fused_moe.py:391-400` presets the expert map *before* `super().__init__()` so `create_weights` allocates only `num_device_experts` slots (the inline comment: *"so that create_weights allocates the right size (no peak memory)"*). The full expert set is mirrored to **pinned CPU**, never to HBM (`expert_offload_manager.py:104-115`). So `model_memory_usage` is genuinely smaller, `non_kv_cache_memory` is smaller, and `available_kv` is larger — **with no term in `determine_available_memory` ever mentioning experts or offload.** The freed HBM flows into KV purely because profiling measures a smaller weight footprint. `(FACT)` for the chain; the "automatic, uncoordinated" framing is `(ANALYSIS)` but well-supported (grep for `offload` in `worker.py`/`determine_available_memory` finds nothing).

**Concrete magnitude (DeepSeek-V2-Lite, bf16):** full routed experts ≈ 64×26×16.5 MB ≈ **26.8 GiB**; at `num_device_experts=6` the device holds ≈ 2.5 GiB (expert-offload note §1). So offload frees **~24 GiB of HBM**, essentially all of which `determine_available_memory` will hand to the KV cache (minus the prefill-pool cost below). At MLA's ~576 cached dims/token that is on the order of **10⁵–10⁶ extra cached tokens** of aggregate context. `(ANALYSIS, order-of-magnitude)`

### 7.2 The catch: the prefill pool is counted, and it adds HBM back

`create_prefill_pool` runs **inside** the `DeviceMemoryProfiler` (`_register_offload_layers` at `:3337`), so its cost is captured in `weights_memory` automatically. But it allocates `num_device_layers` (default **2**) device tensors **each holding all 64 experts** (`expert_offload_manager.py:341-411`). So the *net* HBM the KV cache gains is:

```
net_freed ≈ (64 − num_device_experts) × per_expert × num_moe_layers      ← saved
          − num_device_layers × 64 × per_expert                          ← prefill pool reclaims
```

At `num_device_experts=6, num_device_layers=2` the pool costs ≈ 2×64×16.5 MB ≈ **2.1 GiB** back. The expert-offload V2 note (`[N6]`) argues `num_device_layers=2` currently buys nothing (no reuse, host-blocked) — so for KV-cache headroom, **`num_device_layers=1` strictly frees ~1 GiB more for KV** until the double-buffer optimization lands. `(ANALYSIS)`

### 7.3 The two knobs spend the same pool — `num_device_experts` ⊥ KV cache

The freed HBM is fungible. Every slot you *don't* give to resident experts becomes KV cache, and vice-versa:

- **Raise `num_device_experts`** → fewer expert page-faults per decode step (less H2D traffic, higher throughput) **but less KV cache** (shorter max context / fewer concurrent seqs).
- **Lower `num_device_experts`** → more KV cache **but more expert paging** (the expert-offload note's `~18 tok/s` paging-bound ceiling at `ndev=topk=6`).

So the joint optimum is a **bandwidth-vs-capacity** trade on one HBM budget. There is no separate KV-cache-size knob to balance against `num_device_experts`; you balance `num_device_experts` against *whatever KV cache is left*, governed by `gpu_memory_utilization`. `(ANALYSIS)`

### 7.4 The deeper coupling: they contend for PCIe and pinned host RAM

Beyond the HBM split, expert offload and KV-CPU-offload (§6) **compete for the same two scarce resources**:
1. **The host↔device DMA bus.** Expert paging streams ~GB/token H2D on `load_stream`; KV offload streams blocks D2H/H2D on its own two streams. Run both and they share PCIe bandwidth — KV onload latency and expert page-fault latency directly trade off. `(ANALYSIS)`
2. **Pinned (non-pageable) host RAM.** Expert offload pins the full expert set (~27 GiB bf16); KV offload pins `num_cpu_blocks` worth of CPU KV. Both must physically fit in pinned memory or allocation fails. Budget them together. `(ANALYSIS)`

This is why "offload everything" is not free: the HBM you save is real, but you pay in DMA bandwidth and pinned host RAM, and the two offload features draw from the same well.

---

## 8. Configuration & tuning — KV cache knobs

| Knob | Source | Effect on KV cache | Default |
|---|---|---|---|
| `gpu_memory_utilization` | `worker.py:273` | The cap: `requested_memory = total*util`. **The master dial** — raise it to give expert-offload's freed HBM to KV. | upstream (≈0.9) |
| `kv_cache_memory_bytes` (`--kv-cache-memory`) | `worker.py:340-353` | Hard override of KV bytes; **skips profiling, ignores `gpu_memory_utilization`**. Reproducible sizing. | unset |
| `block_size` | `utils.py:1271-1306` | Page size in tokens. **Effectively pinned to 128** on Ascend (coerced). Not a real tuning dial. | 128 |
| `enable_prefix_caching` | `single_type_kv_cache_manager.py`; `utils.py:1298` | Block reuse across shared prefixes; gates the CPU-offload tier; forces block_size=128. | upstream |
| `enable_chunked_prefill` / `max_num_batched_tokens` | `ascend_config.py:60,112-120` | Bigger chunks = fewer prefill passes; forces block_size=128. Interacts with expert prefill-pool reload cost. | upstream |
| `max_num_seqs` | `attention/utils.py:39` | More concurrent seqs → more KV demand **and** (per expert-offload note §5) pushes expert paging into the heavy prefill pool. They fight in the same direction — raise `num_device_experts` if you raise this. | upstream |
| `max_model_len` | `attention/utils.py:25` | Max blocks/seq; bounds per-request KV. | upstream |
| `kv_cache_dtype` + C8 quant (`quant_config.kv_cache_type`) | `attention_v1.py:393`; `kv_c8.py` | INT8 KV halves KV bytes → more effective context. MLA: K-only. No fp8. | float |
| `enable_kv_nz` | `ascend_config.py:176-183` | MLA KV cache in FRACTAL_NZ layout (P/D decode node only). | False |
| `enable_sparse_c8` / `enable_hamming_sparse` | `ascend_config.py:190-225` | Sparse-attention KV reduction (DSA INT8 indexer / Hamming-hash block selection). | False |
| KV CPU offload (`kv_connector` + `num_cpu_blocks`) | `kv_offload/npu.py:20-23` | Second tier: spill prefix blocks to pinned CPU; turns recomputes into reloads. **Experimental.** | off |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | `envs.py:112-114` | `aclrtMemcpyBatchAsync` path for KV offload copies. | auto |
| `expert_offload` / `num_device_experts` / `num_device_layers` | `ascend_config.py:619-621` | **Indirect KV knobs:** lower resident experts ⇒ more freed HBM ⇒ more KV (§7). | False/32/2 |

---

## 9. Optimal KV-cache + expert-offload combined settings `(ANALYSIS)`

A practical recipe for the reference class of workloads (DeepSeek MLA + expert offload on one NPU). Treat as starting hypotheses to validate against the `[EXPERT-OFFLOAD-CACHE]` hit-rate logs and the "Available KV cache memory: … GiB" log line (`worker.py:393`).

1. **Set `gpu_memory_utilization` high (≈0.9–0.95).** Offload's whole purpose is to free HBM *for KV*; a low utilization throws that away. This is the single biggest lever.
2. **Decide the regime first — context-bound vs throughput-bound:**
   - **Context/concurrency-bound** (long prompts, many seqs): keep `num_device_experts` near the floor (but **strictly > top_k**, per expert-offload note `[N1]` — e.g. `2×top_k`) to maximize KV headroom, and accept more expert paging. Consider KV CPU offload (§6) to push context further.
   - **Throughput-bound** (short context, latency-sensitive decode): raise `num_device_experts` to the hit-rate knee (read `[EXPERT-OFFLOAD-CACHE] hit_rate`) to cut expert page-faults, spending HBM that would otherwise be KV.
3. **Set `num_device_layers = 1`** until the prefill double-buffer optimization lands — it frees ~1 GiB for KV at zero current cost (§7.2 + expert-offload note `[N6]`).
4. **Use W8A8 / INT8 wherever accuracy allows** — it roughly halves *both* expert bytes (more slots per HBM) *and* KV bytes (C8, MLA-K), compounding the win. Best single lever after `gpu_memory_utilization`.
5. **Watch the shared resources (§7.4).** If you enable KV CPU offload *and* expert offload together, budget pinned host RAM for both and expect PCIe contention; don't assume the freed HBM is "free."
6. **For reproducible benchmarking, pin KV with `--kv-cache-memory`** so expert-offload experiments don't silently change the KV block count (profiling is sensitive to `num_device_experts`).
7. **MLA is already a massive KV saver** — combined with expert offload on a small NPU, the binding constraint is usually expert-paging bandwidth, not KV capacity. Measure paging cost (expert-offload note's instrumentation) before spending HBM on more KV.

---

## 10. Glossary

- **PagedAttention:** vLLM's KV-cache method — fixed-size token pages, per-sequence page table, no fragmentation (SOSP'23).
- **Block / page:** a fixed `block_size`-token slab of KV; on Ascend `block_size = 128` (coerced).
- **Block table:** per-request logical→physical page map (`[max_reqs, max_blocks_per_req]`).
- **Slot mapping:** per-token `(block_id, offset)` destination for the new KV.
- **MLA (Multi-head Latent Attention):** caches a compressed latent (`kv_lora_rank`+`qk_rope_head_dim`) instead of full K/V; V reconstructed on the fly. The ref-model KV method.
- **SFA / DSA / kvcomp:** sparse-attention KV reducers (lightning indexer / DeepSeek Sparse Attention / Hamming-hash block selection) — reduce *attended* (and sometimes stored) KV.
- **C8:** static per-channel INT8 KV quantization (Ascend's only KV quant; no fp8).
- **`page_size_bytes`:** bytes of one block of one layer; `num_blocks = available_kv_bytes // Σ page_size_bytes`.
- **Preempt+recompute:** default out-of-blocks policy — evict a running request's KV, recompute later (vs. swap).
- **KV CPU offload:** optional second tier (`kv_offload/`) — spill prefix blocks HBM↔pinned CPU; turns recomputes into reloads.
- **`gpu_memory_utilization`:** fraction of total HBM vLLM may use; sets the cap that KV cache lives under.
- **Expert offload (cross-ref):** the MoE-weight paging feature; frees HBM that — via §3 profiling — becomes KV cache.

---

## 11. Invariants & assumptions (ground truth for agents)

1. **KV size is decided once, by subtraction:** `available_kv = total*util − (weights + activation_peak + non_torch)` (`worker.py:390`). Any change to weight footprint (e.g. `num_device_experts`) re-prices KV automatically; nothing coordinates the two explicitly. `(FACT)`
2. **The expert↔KV coupling is one number (`model_memory_usage`) + two shared resources (PCIe, pinned host RAM).** There is no cross-feature scheduling. `(ANALYSIS)`
3. **`block_size` is 128** on every supported path (coerced in `refresh_block_size`); do not assume it is tunable. `(FACT)`
4. **MLA caches a latent, not full K/V** — the cache has no leading-`2` dim, and V is never stored. Code that assumes `[2, …]` KV will break on DeepSeek. `(FACT)`
5. **Default out-of-HBM policy is preempt+recompute**, not swap. CPU spill is opt-in (§6) and currently experimental/deprecated-tested. `(FACT)`
6. **The prefill pool is counted in `weights_memory`** (it runs inside `DeviceMemoryProfiler`), so it directly reduces KV headroom — `num_device_layers` is an *indirect KV knob*. `(FACT)`
7. **KV quant on Ascend is INT8/C8 only; MLA quantizes K only; no fp8 KV.** `(FACT)`
8. **`--kv-cache-memory` bypasses profiling and `gpu_memory_utilization`** — when set, none of invariant 1's terms matter. `(FACT)`

---

## 12. Cross-references & next steps

- **Expert-offload mechanics** (the other half of the shared-HBM story): `code-summary-vllm-ascend-expert-offload.md` + `…-v2.md`. Key joint findings: the `[N1]` `ndev>topk` floor, the `[N6]` `num_device_layers` waste, and the `~18 tok/s` paging ceiling — all of which determine how much HBM you *can* redirect to KV.
- **Optimization ideas** for offload: `expert-offload-optimization-ideas.md` + v2.
- **Highest-signal first measurement for the joint system:** log the "Available KV cache memory: … GiB" line (`worker.py:393`) across `num_device_experts ∈ {6, 12, 32}` to see exactly how many KV bytes each freed-expert configuration yields, then cross it with `[EXPERT-OFFLOAD-CACHE] hit_rate` to find the throughput-vs-context knee. Those two numbers fully characterize the trade in §7.3.
- **Unverified upstream internals:** the bodies of `memory_profiling`, `MemorySnapshot`, `DeviceMemoryProfiler` (`vllm.utils.mem_utils`) and `KVCacheManager`/`BlockPool` live in upstream `vllm` (not in this checkout); their semantics here are inferred from call sites and field names. Re-verify against the pinned vLLM version before relying on exact accounting.
