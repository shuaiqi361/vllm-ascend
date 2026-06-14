# `moe_offload_v3.0` vs `moe_offload_v2.0` — Expert-Offload Diff Report

**Source remote:** `upstream = https://github.com/LookAround0301/vllm-ascend.git`
**Branches compared:** `upstream/moe_offload_v3.0` (tip `58e1127c`, 2026‑06‑08) vs `upstream/moe_offload_v2.0` (tip `220cb6f2`, 2026‑06‑05)
**Merge-base (common ancestor):** `8e5a61af` (2026‑05‑20)
**Method:** pure `git diff`/`git show`/`git ls-tree` over both branch tips + the merge-base. Every claim below is backed by a command output; no file was edited. A throwaway worktree at `/tmp/vllm-v3` was used to read v3.0 files.

> **Date written:** 2026‑06‑08. Line numbers are snapshot-relative; commit SHAs and the topology are stable anchors.

---

## 0. TL;DR — the one-paragraph answer

**v3.0 is *not* a refactor, *not* an optimization, and *not* a feature-extension of v2.0's offload pipeline.** The two branches **forked from the same commit** (`8e5a61af`) and each layered different work on top of an **identical base offload commit**. v3.0 = `(shared base offload) + (one new unit-test file)`. v2.0 = `(same shared base offload) + (LRC expert-cache policy) + (two W8A8/bf16 NZ-format fixes)`. So relative to v2.0, v3.0 **adds 756 lines of CPU unit tests** and **drops the LRC cache policy and the NZ fixes**. In capability terms v3.0 is a *narrower, test-hardened re-baselining* of the offload feature, behind v2.0 on runtime functionality. The core `expert_offload_manager.py` runtime is **byte-identical** to v2.0's base — there is no new or optimized pipeline code.

**On the dsv4 question (your follow-up):** the `moe offload support dsv4` commit in v3.0 (`74c309c8`) is **byte-for-byte identical** to the one in v2.0 (`b6679387`). All actual DeepSeek‑V4 *model* code (model def, DSA/indexer attention, MTP, RoPE, tool-call patch) lives in the **shared merge-base** and is **identical in both branches**. **There is nothing dsv4-specific in v3.0 that v2.0 lacks** — nothing to integrate. See §6.

---

## 1. Branch topology (why a flat "v3 = v2 + delta" mental model is wrong)

```
* 58e1127c 2026-06-08  LookAround0301   Add unit tests for expert offload feature.   ← v3.0 tip
* 74c309c8 2026-06-07  LookAround0301   moe offload support dsv4                       (v3.0 base offload)
|                                        ⇡ byte-identical tree to b6679387 ⇣
| * 220cb6f2 2026-06-05 zhangsicheng5    fix weight nz bf16 bug                         ← v2.0 tip
| *   18461882 2026-06-03 linsheng       Merge PR #92
| |\
| * | 46599fa7 2026-05-27 zhangsicheng5  fix w8a8 weight nz format bug
| | * d0d678c3 2026-05-21 linsheng1      Add LRC expert offload cache policy
| |/
| * b6679387 2026-05-20  LookAround0301  moe offload support dsv4                       (v2.0 base offload)
|/
* 8e5a61af 2026-05-20  ZhuQi-seu  [Feature]Replace Triton conv1d... (#8842)            ← MERGE-BASE
                                   (already contains the full DeepSeek-V4 model stack)
```

Key facts proven from git:
- `git diff b6679387 74c309c8` → **empty** (the two "moe offload support dsv4" commits produce identical trees). The base offload feature is the same in both branches.
- v3.0 branched **directly off the merge-base**, not off v2.0's tip. It therefore never inherited v2.0's downstream LRC and NZ-fix commits.
- The DeepSeek‑V4 model files already exist in the merge-base `8e5a61af` (see §6) and are unchanged by either branch.

---

## 2. Complete tip-to-tip diff inventory (the only 11 files that differ)

`git diff --name-status upstream/moe_offload_v2.0 upstream/moe_offload_v3.0`:

| Status | File | Lines | Cause / classification |
|---|---|---|---|
| **A** | `tests/ut/test_expert_offload.py` | +756 | **v3.0-only NEW** — CPU unit-test suite (§4) |
| **D** | `vllm_ascend/expert_offload/lrc_policy.py` | −151 | v2.0-only LRC policy (absent in v3.0) |
| **D** | `tests/ut/expert_offload/test_lrc_policy.py` | −98 | v2.0-only LRC unit test |
| **D** | `tools/analyze_expert_cache_log.py` | −335 | v2.0-only LRC log-analysis tool |
| **D** | `tools/test_analyze_expert_cache_log.py` | −33 | v2.0-only LRC tool test |
| **D** | `docs/moe_offload_lrc_test_flow.md` | −151 | v2.0-only LRC test-flow doc |
| **M** | `vllm_ascend/expert_offload/expert_offload_manager.py` | 226 | v2.0 = base **+ LRC hooks + 2 NZ fixes**; v3.0 = base |
| **M** | `vllm_ascend/ascend_config.py` | 32 | v2.0 adds 9 LRC `cache_*` config knobs (§3) |
| **M** | `vllm_ascend/expert_offload/__init__.py` | 7 | v2.0 lazy `__getattr__` import; v3.0 eager import |
| **M** | `vllm_ascend/ops/fused_moe/fused_moe.py` | 2 | `update_weights(...)` arg count (§3) |
| **M** | `vllm_ascend/quantization/methods/w8a8_dynamic.py` | 2 | `update_weights(...)` arg count (§3) |

**Every one of the 11 differing files is either (a) LRC-related — present in v2.0, absent in v3.0 — or (b) the one new v3.0 test file.** No model files, no attention files, no kernel/op files differ. `git diff` over `deepseek_v4.py`, `deepseek_v4_mtp.py`, `dsa_v1.py`, `rope_dsv4.py`, `models/__init__.py`, `envs.py`, `utils.py` all return **0 lines** (§6).

---

## 3. What v2.0 has that v3.0 LACKS

### 3a. The LRC expert-cache policy (commit `d0d678c3`) — a whole feature
v3.0 has **no eviction-policy module at all**. v2.0's `lrc_policy.py` (`LRCExpertCachePolicy`) implements hotness-ranked victim selection
`hotness = recent_weight·freq + ema_weight·ema + router_weight·router_score − age_weight·age`
used by the decode pager to decide which resident expert to evict on a miss. Removing it means v3.0's decode pager falls back to the base manager's default eviction (no hotness ranking, no router-score signal, no EMA).

The LRC commit also added, in v2.0 only:
- **9 config knobs** in `ascend_config.py` (`ExpertOffloadConfig`):
  `cache_policy_enabled` (default `False`), `cache_recent_window` (32), `cache_ema_beta` (0.9), `cache_recent_weight` (1.0), `cache_ema_weight` (0.5), `cache_router_weight` (0.3), `cache_age_weight` (0.01), `cache_stats_log_interval` (1000), `cache_debug_log_updates` (`False`) — with full type/range validation. **None of these exist in v3.0.**
- A **4-arg `update_weights(layer, topk_ids, log2phy, topk_weights)`** signature (the `topk_weights` is fed to the policy as the router-score signal). v3.0 uses the **base 3-arg `update_weights(layer, topk_ids, log2phy)`** — the call sites in `fused_moe.py:190` and `w8a8_dynamic.py:259` differ by exactly this one argument.
- Tooling: `tools/analyze_expert_cache_log.py` + test, and `docs/moe_offload_lrc_test_flow.md`.

> ⚠️ Practical note carried over from the V2 study notes: the LRC policy is **provably inert when `num_device_experts ≤ top_k`** (it makes no discriminating eviction choice). So for configs at that corner, v3.0 dropping LRC changes *nothing observable*. LRC only earns its keep at `num_device_experts > top_k`. See `notes/code-summary-vllm-ascend-expert-offload-v2.md` §9 invariant 9.

### 3b. The two W8A8 / bf16 NZ-format fixes (commits `46599fa7`, `220cb6f2`) — likely correctness-relevant
These are **v2.0-only** and **not in v3.0** (`git merge-base --is-ancestor 46599fa7 74c309c8` → NO). They rewrote the weight-copy path:

- **`46599fa7` "fix w8a8 weight nz format bug"** added:
  - `pin_memory=True` on **all** CPU weight/scale/offset buffers (faster, async-safe H2D; the base leaves them pageable).
  - A new **`process_weights_after_loading()`** that, for W8A8 (int8), pre-casts the CPU mirror to **`FRACTAL_NZ`** and stores it NZ-matched, then switches the prefill-pool and decode copies to **flat `untyped_storage()` slice copies** — so an H2D copy is a raw DMA with **no implicit ND→NZ format conversion**.
- **`220cb6f2` "fix weight nz bf16 bug"** generalized the slice arithmetic from element-count (`*_element_num`) to **byte size** (`w13/w2_expert_size_bytes = nelement × element_size`), fixing the bf16 case and the decode/prefill storage-slice offsets.

**v3.0's base manager has none of this** — confirmed: `grep -c untyped_storage` on the v3.0 manager = **0**, and it has **no `process_weights_after_loading` method**. v3.0 instead uses plain `dev_tensor.data[slot].copy_(buffers[...][...].to(dev))` tensor copies everywhere (manager lines ~579–596 for decode, ~362–386 / ~417–441 for the prefill pool). Those are **format-safe** (torch handles ND↔NZ) but **slower** (per-copy conversion, non-pinned source).

> 🔴 **Risk for quantized inference:** the `process_weights_after_loading` + NZ-matched mirror is precisely what makes W8A8 expert offload fast *and* avoids silent NZ-tile mismatch. v3.0 predates that fix. If you run **W8A8-quantized** dsv4 with v3.0's offload you get the older `.to(dev)` path; v2.0's path is the more battle-tested one for quantized weights. This matters directly for the dsv4-flash goal (§7).

---

## 4. What v3.0 has that v2.0 LACKS — the unit-test suite (`tests/ut/test_expert_offload.py`, 756 lines)

This is v3.0's **single net-new artifact** (`comm -13` over the two file trees returns exactly this one path). It is a CPU-only suite (mocks `torch_npu.npu.Stream` and `get_ascend_config`, so it runs without NPU hardware). Six `TestBase` classes:

| Class | ~Tests | Covers |
|---|---|---|
| `TestExpertOffloadConfig` | 15 | `ExpertOffloadConfig` defaults (`expert_offload=False`, `num_device_experts=32`, `num_device_layers=2`), unknown-key `ValueError`, attribute access, and type/range validation for every field incl. `expert_map_path` JSON checks |
| `TestInitExpertOffloadConfig` | 6 | `init_expert_offload_config()` enable/disable logic (disabled when `expert_offload=False`, `ndev==0`, or `ndev ≥ num_experts`) + `expert_map` shape/dtype/`-1` tail |
| `TestInitLog2phyForOffload` | 5 | log2phy mapping correctness — **via a CPU re-implementation `_ref_log2phy`, not the real `init_log2phy_for_offload`** (the real fn hard-codes `device='npu'`). Tests the *algorithm*, not the shipped function. |
| `TestExpertOffloadManagerWeightLoading` | 10 | `create_weights` CPU-buffer allocation/shapes, `offload_threshold = ndev//topk`, `load_w13/load_w2` transpose + w1/w3 halves, and the **pending→drain** path (load-before-`create_weights`) |
| `TestExpertOffloadManagerScaleOffset` | 5 | W8A8 scale/offset `_add_pending_scale` + `maybe_create_scale_buffers` drain, incl. "wait for both w1+w3 shards before draining" |
| `TestExpertOffloadManagerLifecycle` | 6 | singleton: `has_/get_/maybe_init_expert_offload_manager`, idempotency, assert-on-uninitialized |

**What the tests cover:** the **load-time / setup machinery** — config parsing & validation, CPU-mirror allocation, weight transpose, the pending/drain race handling, scale/offset assembly, and singleton lifecycle.

**What the tests do *not* cover (coverage gaps to be aware of):**
- The **decode demand-paging hot path** (`update_weights`/`_update_weights`, miss detection, eviction, `log2phy` rewrite, `load_stream.synchronize()`).
- The **prefill bulk-load pool** (`create_prefill_pool`/`_prefill_load_layer`, the 3-field `num_local_experts` patch/restore).
- Any **eviction policy** behavior (consistent with v3.0 having no LRC).
- The **real** `init_log2phy_for_offload` (only a CPU stand-in is exercised).
- Anything requiring **actual NPU** H2D copies / NZ casting.

So these are **structural/unit** correctness tests for the static setup, not behavioral tests of the runtime offload pipeline.

---

## 5. Classification verdict (refactor / optimization / new feature?)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **Refactor of the offload pipeline?** | **No** | `expert_offload_manager.py` runtime in v3.0 is **byte-identical** to v2.0's base offload commit; no restructuring of the pipeline code. (`git diff b6679387 74c309c8` = empty.) |
| **Optimization of the offload pipeline?** | **No — opposite** | v3.0 contains *no* perf changes and actually **lacks** v2.0's pinned-memory + NZ-matched storage-slice copy optimization (the NZ fixes). |
| **New offload features?** | **No — net removal** | v3.0 adds no runtime capability; it **removes** the LRC cache policy + its config knobs + tooling. |
| **Test hardening / re-baselining?** | **Yes** | The only additive change is a 756-line unit-test suite for the setup machinery; v3.0 reads as a clean, test-covered re-cut of the *base* feature, deliberately excluding v2.0's downstream patches. |

**Bottom line:** Treat v3.0 not as "v2.0 + improvements" but as **a parallel, leaner line off the same root**: same core offload engine, plus unit tests, minus LRC, minus the NZ fixes. For *running* the offload feature (especially quantized), **v2.0 is the more complete and more correct base.**

---

## 6. ✅ dsv4 verification (your priority question)

**Question:** does the v3.0 `moe offload support dsv4` commit carry anything DeepSeek‑V4‑specific that the v2.0 branch does not have, so we can integrate it into v2.0 for dsv4-flash inference?

**Answer: No. There is nothing dsv4-specific in v3.0 that v2.0 lacks.** Three independent proofs:

1. **The two "dsv4" commits are identical.** `git diff b6679387 74c309c8` (v2.0's vs v3.0's `moe offload support dsv4` commit) → **empty, all files**. v3.0's dsv4 commit is a re-commit (rebase/cherry-pick) of the exact same tree.

2. **All DeepSeek‑V4 *model* code is in the shared merge-base and identical in both branches.** `"dsv4"` in the commit name refers to the *target model the offload was validated against*, **not** new model code added by the commit. The actual DSV4 stack predates both offload lines — it exists in merge-base `8e5a61af`:
   - `vllm_ascend/models/deepseek_v4.py` (`DeepseekV4ForCausalLM`, `DeepseekV4MoE`, `DeepseekV4Attention`)
   - `vllm_ascend/models/deepseek_v4_mtp.py` (`DeepSeekV4MTP` — multi-token prediction)
   - `vllm_ascend/attention/dsa_v1.py` (DeepSeek Sparse Attention: `DeepseekV4IndexerCache`, SWA-layer, compress-ratio)
   - `vllm_ascend/ops/rope_dsv4.py` (`ComplexExpRotaryEmbedding`, `get_cos_and_sin_dsa`)
   - `vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py`
   - dsv4 helpers in `vllm_ascend/utils.py` (`extract_dsv4_layer_index`, `get_dsv4_spec_layer_idx_from_weight_name`, `get_dsv4_compress_ratio`, `model_type == "deepseek_v4"` handling) and the `VLLM_ASCEND_APPLY_DSV4_PATCH` env in `envs.py`.

   **`git diff upstream/moe_offload_v2.0 upstream/moe_offload_v3.0` over every one of these files = 0 lines.** They are byte-identical in both branches.

3. **The full file-tree delta confirms it.** The *only* file in v3.0 that does not exist in v2.0 is `tests/ut/test_expert_offload.py`. No dsv4 model/attention/op file is unique to v3.0. (`comm` over `git ls-tree` of both branches.)

**"flash":** the only `flash` references in the dsv4 path are generic backend naming (`return "ASCEND_DSA" ... else "FLASH_ATTN"` in `dsa_v1.py:139`, "Flash Comm V1") — present and identical in both branches. There is no v3.0-exclusive "dsv4-flash" code.

### Net for the dsv4-flash goal
- **v2.0 already contains everything dsv4-related that v3.0 has** (the entire DSV4 model + attention + MTP + RoPE stack, *plus* the identical offload engine), and additionally has the **W8A8/bf16 NZ-format fixes** that v3.0 lacks.
- So **there is nothing to port from v3.0 → v2.0 for dsv4 support.** If anything, the integration arrow points the other way (v2.0's NZ fixes would benefit v3.0).
- The one genuinely useful v3.0 artifact you *could* cherry-pick into v2.0 is the **unit-test file** `tests/ut/test_expert_offload.py` — but note it tests the **3-arg base `update_weights`**; on v2.0 (4-arg LRC signature) the call-site assertions won't apply unmodified, and it tests the base manager, not the LRC path.

---

## 7. Recommendations

1. **Stay on v2.0 (your current branch) as the dsv4-flash base.** It has the complete DSV4 model stack *and* the more correct W8A8/bf16 offload copy path. v3.0 offers no dsv4 advantage.
2. **Do not "integrate v3.0's dsv4 part into v2.0"** — there is no such part; the dsv4 commits are identical and the model code is shared. This avoids a wasted merge.
3. If you want v3.0's **test coverage** on v2.0: cherry-pick `tests/ut/test_expert_offload.py`, then adapt the `update_weights` call expectations to v2.0's 4-arg signature and optionally extend the suite to cover the LRC path and the decode/prefill hot paths (current gaps, §4).
4. For **dsv4-flash specifically**, the offload feature is orthogonal to model correctness — verify the DSV4 attention (DSA indexer / SWA, `compressor_ratio ≤ 1` assertion in `dsa_v1.py:1034`) and MTP paths independently; offload only changes *where expert weights live*, not the attention math.

---

## 8. Reproduce this analysis

```bash
git fetch upstream moe_offload_v3.0
B2=upstream/moe_offload_v2.0; B3=upstream/moe_offload_v3.0
git merge-base $B2 $B3                              # 8e5a61af
git log --oneline $B2..$B3                          # v3.0-only: 74c309c8, 58e1127c
git log --oneline $B3..$B2                          # v2.0-only: incl. d0d678c3 (LRC), nz fixes
git diff b6679387 74c309c8                          # EMPTY → dsv4 base commits identical
git diff --name-status $B2 $B3                       # the 11 differing files (all LRC or the test)
git diff $B2 $B3 -- vllm_ascend/models/deepseek_v4.py   # EMPTY → dsv4 model identical
git merge-base --is-ancestor 46599fa7 74c309c8 || echo "NZ fix NOT in v3.0"
```

**Related notes:** `code-summary-vllm-ascend-expert-offload-v2.md` (deep dive on the shared offload engine), `expert-offload-optimization-ideas-v2.md`, `expert-offload-codesign-surfaces.md`.
