"""Proactive expert prefetcher — NPU execution engine for the unified cache.

This is the torch/NPU half of the proactive-prefetch feature (the CPU-only
"brain" lives in :mod:`prefetch_predictor`). It is an *optional extension* of
:class:`ExpertOffloadManager`: when ``prefetch_enabled`` is off, none of this is
constructed and the offload path is byte-for-byte unchanged.

Design — one cache, no copies on a hit (MoE-Infinity style):

* There is exactly **one** on-device expert cache per MoE layer: the layer's
  real weight pool (``layer.w13_weight`` / ``w2_weight``, ``num_device_experts``
  slots). Prefetched experts are loaded **directly into that pool** — there is
  no separate staging buffer and therefore no device-to-device promote. Once an
  expert is on device it is simply *used where it sits*.
* Residency is the manager's per-layer host ``log2phy`` mirror (expert -> slot),
  the single source of truth shared by demand paging and the prefetcher. So a
  prefetched expert is visible to the demand path as already-resident: the
  demand path finds a cache hit and moves on with **no copy**. An expert can
  therefore live in only one place — the unified cache — never two.
* During the decode step's first MoE layer we predict the experts that **every
  remaining layer** of the step will need (MoE-Infinity prefetches the whole
  request ahead, not just the next layer) and issue **async** CPU->pool H2D
  copies on a dedicated ``prefetch_stream``, nearest layers first so the most
  imminent layers win the bandwidth. Each layer is predicted+issued once per
  step (recent-union predictions for a layer are constant until that layer is
  itself observed), keeping the sweep O(num_layers) per step.
* When a layer is paged and a needed expert was prefetched, the manager orders
  the consuming stream after the prefetch's completion event (``wait_for_landed``)
  so the kernel never reads a half-written slot. If the prefetch has not been
  issued (cold / mispredict) the demand path loads it synchronously as before —
  correctness never depends on a prediction being right.

Scope for this version: single-rank, eager mode. Under ACL-graph capture the
manager skips ``run``/event waits, so capture behaves exactly as before this
feature; the host mirror still tracks every device mutation so residency stays
consistent across the capture/eager boundary.
"""

import torch_npu
from vllm.logger import logger

from vllm_ascend.expert_offload.eam_predictor import EAMPredictor
from vllm_ascend.expert_offload.prefetch_predictor import RecentUnionPredictor


class ExpertPrefetcher:
    """Predicts future-layer experts and stages them into the unified cache."""

    def __init__(
        self,
        manager,
        num_moe_layers: int,
        num_total_experts: int,
        capacity: int,
        horizon: int,
        history_window: int,
        predictor_kind: str = "recent_union",
        eamc_capacity: int = 256,
        debug: bool = False,
    ) -> None:
        self.manager = manager
        self.num_moe_layers = num_moe_layers
        self.num_total_experts = num_total_experts
        # Max experts the prefetcher may proactively stage into one layer's pool
        # per step (a speculation budget on the shared cache, not a separate
        # allocation). <= num_device_experts in practice.
        self.capacity = capacity
        # How many layers ahead to prefetch. <= 0 means "all remaining layers
        # this step" (the MoE-Infinity whole-request lookahead).
        self.horizon = horizon
        self.debug = debug

        self.predictor_kind = predictor_kind
        # The EAM predictor needs per-request routing; recent-union does not.
        self.requires_request_context = predictor_kind == "eam"
        if predictor_kind == "eam":
            self.predictor = EAMPredictor(
                num_moe_layers, num_total_experts, eamc_capacity=eamc_capacity)
        else:
            self.predictor = RecentUnionPredictor(num_moe_layers, history_window)

        self.prefetch_stream = torch_npu.npu.Stream()
        # (layer_idx, expert_id) -> completion Event for the in-flight/landed H2D
        self._events: dict[tuple[int, int], "torch_npu.npu.Event"] = {}
        self._event_pool: list["torch_npu.npu.Event"] = []

        # Highest future layer already prefetched this decode step (reset at the
        # step's first MoE layer) so each layer is staged exactly once per step.
        self._cursor = 0
        # Last layer index passed to run(); a non-increasing index marks a new
        # decode step. Seeded above num_moe_layers so the very first call resets.
        self._last_run_layer = num_moe_layers
        self._initialized = False

        # Lightweight counters for [PREFETCH] stats.
        self.n_issued = 0   # CPU->pool prefetch copies issued
        self.n_hits = 0     # needed experts found already-landed in the cache
        self.n_late = 0     # needed experts staged but not yet landed (stalled)

    # ------------------------------------------------------------------ #
    #  Allocation (no staging tensors — just validate prerequisites)      #
    # ------------------------------------------------------------------ #

    def allocate(self) -> None:
        if self._initialized:
            return
        mgr = self.manager
        if not mgr.moe_layers:
            return
        if mgr.cache_policy is None:
            logger.warning(
                "[PREFETCH] disabled: the unified-cache prefetcher needs the "
                "LRC cache policy (cache_policy_enabled) for slot eviction.")
            return
        self._initialized = True
        logger.warning(
            "[PREFETCH] unified-cache prefetch active: layers=%d budget=%d/layer "
            "horizon=%s predictor=%s",
            self.num_moe_layers, self.capacity,
            "all" if self.horizon <= 0 else self.horizon, self.predictor_kind)

    # ------------------------------------------------------------------ #
    #  Prediction substrate                                               #
    # ------------------------------------------------------------------ #

    def observe_routing(self, layer_idx: int, needed: set[int],
                        per_req_experts: dict | None) -> None:
        """Feed this layer's actual routing to the predictor.

        ``per_req_experts`` maps request id -> routed-expert set (EAM mode). For
        the recent-union predictor it is ignored. If EAM mode is on but the
        per-request attribution is unavailable, the whole batch is attributed to
        one synthetic request so prediction still functions (degraded).
        """
        if self.predictor_kind == "eam":
            if per_req_experts:
                for rid, experts in per_req_experts.items():
                    self.predictor.observe_request(rid, layer_idx, experts)
            else:
                self.predictor.observe_request("_batch", layer_idx, needed)
        else:
            self.predictor.observe(layer_idx, needed)

    def sync_requests(self, active_req_ids) -> None:
        """Retire requests no longer in the batch (EAM mode only)."""
        if self.predictor_kind == "eam":
            ids = list(active_req_ids) if active_req_ids else ["_batch"]
            self.predictor.sync_active(ids)

    # ------------------------------------------------------------------ #
    #  Issue: stage predicted experts for every remaining layer this step #
    # ------------------------------------------------------------------ #

    def run(self, current_layer_idx: int) -> None:
        """Predict + async-stage experts for layers ahead of ``current_layer_idx``.

        Called on the host (eager path) after the current layer's demand paging.
        Issues non-blocking CPU->pool H2D copies on ``prefetch_stream``; nothing
        here blocks compute. Each future layer is staged at most once per step
        (cursor-guarded), so the whole sweep is O(num_layers) per step.
        """
        if not self._initialized:
            return
        last = self.num_moe_layers - 1
        # The decode step walks MoE layers in increasing order; a non-increasing
        # index means a new step began, so reset the per-step cursor. (Detecting
        # the boundary by wrap rather than assuming the step's first layer is
        # index 0 keeps this correct if a rank doesn't host layer 0.)
        if current_layer_idx <= self._last_run_layer:
            self._cursor = current_layer_idx + 1
        self._last_run_layer = current_layer_idx
        hi = last if self.horizon <= 0 else min(current_layer_idx + self.horizon, last)
        start = max(self._cursor, current_layer_idx + 1)
        for target in range(start, hi + 1):
            self._prefetch_layer(target)
        self._cursor = max(self._cursor, hi + 1)

        if self.debug:
            logger.warning(
                "[PREFETCH] after l=%d cursor=%d issued_total=%d hits_total=%d "
                "late_total=%d", current_layer_idx, self._cursor,
                self.n_issued, self.n_hits, self.n_late)

    def _prefetch_layer(self, target: int) -> None:
        """Stage ``target``'s predicted, not-yet-resident experts into its pool."""
        mgr = self.manager
        policy = mgr.cache_policy
        predicted = self.predictor.predict(target)
        if not predicted:
            return
        mirror = mgr.log2phy_host(target)
        slot_owner = mgr.slot_owner_for_layer(target)   # {slot: expert_id}
        resident = set(slot_owner.values())
        npool = mgr.pool_slots(target)
        free = [s for s in range(npool) if s not in slot_owner]
        loading = self._loading_set(target)
        protected = set(predicted)
        layer = mgr.moe_layers[target]
        staged = 0

        with torch_npu.npu.stream(self.prefetch_stream):
            for eid in predicted:
                if staged >= self.capacity:
                    break
                if eid in resident:
                    continue  # already in the one unified cache — never duplicate
                if free:
                    slot = free.pop()
                    victim = None
                else:
                    victim = policy.choose_victim(
                        target, slot_owner, protected=protected, loading=loading)
                    if victim is None:
                        break  # pool full of wanted / in-flight experts
                    slot = int(mirror[victim])

                # Async CPU->pool H2D straight into the real slot. No staging
                # tensor, no later D2D: the kernel will read the expert here.
                mgr._copy_expert_into_slot(layer, target, eid, slot,
                                           non_blocking=True)
                event = self._acquire_event()
                event.record(self.prefetch_stream)

                if victim is not None:
                    mirror[victim] = -1
                    slot_owner.pop(slot, None)
                    resident.discard(victim)
                    self._release_event(self._events.pop((target, victim), None))
                mirror[eid] = slot
                slot_owner[slot] = eid
                resident.add(eid)
                loading.add(eid)
                self._events[(target, eid)] = event
                staged += 1
                self.n_issued += 1

    # ------------------------------------------------------------------ #
    #  Consume: the demand path queries these when paging a layer         #
    # ------------------------------------------------------------------ #

    def wait_for_landed(self, layer_idx: int, experts, stream) -> None:
        """Order ``stream`` after the prefetch H2D of each given needed expert.

        Called for the layer's cache *hits*. A hit that came from prefetch may
        not have landed yet; making ``stream`` wait on its completion event (the
        manager then synchronises that stream before the kernel) guarantees the
        slot is fully written before it is read — without a host stall when the
        copy already finished. Experts loaded by ordinary demand paging have no
        event and are skipped.
        """
        for eid in experts:
            event = self._events.pop((layer_idx, eid), None)
            if event is None:
                continue
            if event.query():
                self.n_hits += 1
            else:
                stream.wait_event(event)
                self.n_late += 1
            self._release_event(event)

    def pending_event_experts(self, layer_idx: int) -> set[int]:
        """Experts with an *in-flight* prefetch into ``layer_idx`` (don't evict).

        Recycles any events that have already landed so the table stays bounded.
        """
        pending: set[int] = set()
        landed: list[tuple[int, int]] = []
        for key, event in self._events.items():
            if key[0] != layer_idx:
                continue
            if event.query():
                landed.append(key)
            else:
                pending.add(key[1])
        for key in landed:
            self._release_event(self._events.pop(key, None))
        return pending

    def note_evicted(self, layer_idx: int, expert_id: int, stream) -> None:
        """A demand load is about to overwrite ``expert_id``'s slot.

        If a prefetch into that slot is still in flight, order the overwriting
        ``stream`` after it so we never corrupt a half-written copy. Then drop
        the (now moot) event.
        """
        event = self._events.pop((layer_idx, expert_id), None)
        if event is None:
            return
        if not event.query():
            stream.wait_event(event)
        self._release_event(event)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _loading_set(self, layer_idx: int) -> set[int]:
        """Experts whose prefetch copy hasn't landed — don't evict to make room."""
        loading: set[int] = set()
        for (l_idx, eid), event in self._events.items():
            if l_idx == layer_idx and not event.query():
                loading.add(eid)
        return loading

    def _acquire_event(self) -> "torch_npu.npu.Event":
        if self._event_pool:
            return self._event_pool.pop()
        return torch_npu.npu.Event()

    def _release_event(self, event) -> None:
        if event is not None:
            self._event_pool.append(event)
