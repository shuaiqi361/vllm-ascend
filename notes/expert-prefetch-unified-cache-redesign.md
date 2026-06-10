# Expert prefetch — unified-cache redesign

Supersedes the original "separate staging cache + D2D promote" prefetch design
(commit 4811ece2). Motivated by review feedback: *there should be no D2D copy —
as long as an expert is on device, just use it; the same expert must not live in
two caches; and prefetch must cover all layers, not just the next, following
MoE-Infinity.*

## What was wrong before

The first prefetch implementation kept a **second** on-device cache per layer: a
`num_prefetch_experts`-slot staging tensor (`_pf_w13`/`_pf_w2`), separate from the
layer's real weight pool (`layer.w13_weight`, the LRC cache the kernel reads).
Prefetched experts landed in staging, and on a real need were **promoted** into
an LRC slot with a device-to-device (D2D) copy.

Two problems:
1. **A pointless copy.** The kernel reads experts from one contiguous per-layer
   tensor indexed by `log2phy`. An expert sitting in the *staging* tensor isn't
   addressable by the kernel, hence the D2D. But if the expert had been put in
   the real pool to begin with, no copy is needed at all.
2. **Two caches / duplication.** The same expert could exist in both the staging
   tensor and the LRC pool, with two residency maps to reconcile.

## The redesign — one cache (MoE-Infinity style)

There is now exactly **one** on-device expert cache per MoE layer: the layer's
real weight pool (`num_device_experts` slots). Prefetch loads experts **directly
into that pool**. No staging tensor, no D2D, no second residency map.

- **Residency = a shared host log2phy mirror.** `manager._log2phy_host[layer]`
  (numpy, `expert_id -> slot`, `-1` absent) is the single source of truth, shared
  by demand paging and the prefetcher. A prefetched expert is recorded there the
  moment its copy is issued, so the demand path sees it as already-resident — a
  cache **hit, copied nowhere**. An expert lives in exactly one place. The device
  `layer.log2phy` becomes push-only (refreshed from the mirror at paging time);
  the mirror tracks every device mutation so it stays consistent across the
  ACL-graph capture/eager boundary.
- **Demand and prefetch share one code path.** `manager._copy_expert_into_slot()`
  does the CPU→pool H2D (weights + scale/offset + derived fp32) into a given slot
  on the caller's current stream — `load_stream` for demand, `prefetch_stream`
  for prefetch.
- **All layers, once per step.** At a decode step's first MoE layer the
  prefetcher predicts every *remaining* layer's experts and issues async H2D into
  their pools, nearest first. A per-step cursor stages each layer exactly once
  (recent-union predictions for a layer don't change until that layer is itself
  observed), keeping the sweep O(num_layers). `prefetch_horizon = 0` (the new
  default) means "all remaining layers"; a positive value caps the lookahead.
- **Eviction reuses the LRC policy.** Prefetch picks victims with the same
  `LRCExpertCachePolicy.choose_victim` the demand path uses (hence
  `prefetch_enabled` now requires `cache_policy_enabled`). The dedicated
  `PrefetchCachePolicy` is deleted.

## Correctness: stream ordering

Prefetch copies run on `prefetch_stream` and are tracked by per-`(layer, expert)`
completion events.

- **Hit on an in-flight prefetch:** when a layer is paged, for each needed expert
  that was prefetched the demand path orders `load_stream` after the prefetch
  event (`wait_for_landed`); the trailing `load_stream.synchronize()` then
  guarantees the slot is fully written before the kernel reads it. No host stall
  when the copy already landed.
- **Don't evict an in-flight prefetch:** the set of experts with in-flight
  prefetches (`pending_event_experts`) is passed to `choose_victim` as `loading`,
  and `note_evicted` waits on a victim's event before a demand load overwrites
  its slot — so a copy in flight is never corrupted.
- Prefetch only writes *future* layers' pools, never the layer currently being
  read, so there's no concurrent read/write of the same tensor within a step.

## Files

- `expert_prefetcher.py` — rewritten: no staging tensors / no `try_promote`;
  `run()` (all-layers cursor), `wait_for_landed`, `pending_event_experts`,
  `note_evicted`.
- `expert_offload_manager.py` — host log2phy mirror + `_copy_expert_into_slot` /
  `slot_owner_for_layer` / `pool_slots` / `log2phy_host`; `_update_weights`
  seeds from / writes back to the mirror, waits on prefetch hits, passes the
  loading set; the non-prefetch path is byte-for-byte unchanged.
- `prefetch_cache_policy.py` — **deleted** (one cache → one policy).
- `ascend_config.py` — `prefetch_horizon` default `0` (= all layers, validated
  `>= 0`); `num_prefetch_experts` reinterpreted as a per-layer speculation budget
  on the shared pool (no extra HBM); `prefetch_enabled` requires
  `cache_policy_enabled`.

## Validation status

- **Tested (CPU, pure-Python):** predictor, LRC policy, EAM predictor, and a
  rewritten integration sim that models the unified flow — all-layers cursor,
  pre-staging eliminating synchronous demand loads, the one-expert-one-slot
  invariant under thrash, and no-duplicate-on-resident-prediction. 25 tests pass
  via `pytest --noconftest tests/ut/expert_offload/`.
- **Review-only (not HW-validated):** the NPU glue (`expert_prefetcher.py`, the
  `_update_weights` hot path) imports `torch_npu` and can't run off-device. It
  passed an adversarial static review (host-mirror↔device consistency incl.
  capture, stream ordering on hits/evictions, event lifecycle, byte-match of the
  factored copy). Needs on-device correctness + perf validation before relying on
  it in production.
