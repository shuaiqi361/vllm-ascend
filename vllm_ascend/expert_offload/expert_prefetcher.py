"""Proactive expert prefetcher — the NPU execution engine for the staging cache.

This is the torch/NPU half of the proactive-prefetch feature (the CPU-only
"brain" lives in :mod:`prefetch_predictor` and :mod:`prefetch_cache_policy`).
It is an *optional extension* of :class:`ExpertOffloadManager`: when
``prefetch_enabled`` is off, none of this is constructed and the offload path
is byte-for-byte unchanged.

Design (kept intentionally independent of the LRC demand-paging cache):

* A **separate** per-layer staging tensor of ``num_prefetch_experts`` slots
  holds prefetched experts. It is distinct from ``layer.w13_weight`` (the LRC
  cache) and governed solely by :class:`PrefetchCachePolicy`.
* During layer ``L``'s decode paging, we predict the experts that the next
  ``horizon`` layers will need and issue **async** CPU->staging H2D copies on a
  dedicated ``prefetch_stream``, overlapping them with ongoing compute.
* The two caches meet at exactly one point: before staging an expert we check
  whether it is already resident in the LRC cache (``lrc_resident``) and skip
  it if so.
* When a layer actually needs an expert that we staged, the manager *promotes*
  it into an LRC slot with a fast on-device **D2D** copy (the slow PCIe H2D was
  already paid, ahead of time). On a miss/late-arrival the manager falls back
  to the normal synchronous CPU->LRC H2D, so correctness never depends on a
  prediction being right.

Scope for this (Plan-A) version: single-rank, eager mode. Under ACL-graph
capture the manager bypasses the prefetcher entirely, so capture behaves
exactly as it did before this feature.
"""

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.expert_offload.eam_predictor import EAMPredictor
from vllm_ascend.expert_offload.prefetch_cache_policy import PrefetchCachePolicy
from vllm_ascend.expert_offload.prefetch_predictor import RecentUnionPredictor
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


class ExpertPrefetcher:
    """Predicts and stages future-layer experts into a separate NPU cache."""

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
        self.capacity = capacity
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
        self.policy = PrefetchCachePolicy(num_moe_layers, num_total_experts, capacity)

        self.prefetch_stream = torch_npu.npu.Stream()
        # (layer_idx, expert_id) -> completion Event for the in-flight/landed H2D
        self._events: dict[tuple[int, int], torch_npu.npu.Event] = {}
        self._event_pool: list[torch_npu.npu.Event] = []

        # Per-layer staging tensors (allocated lazily once device weights exist)
        self._pf_w13: list[torch.Tensor] = []
        self._pf_w2: list[torch.Tensor] = []
        self._pf_scale: dict[str, list[torch.Tensor]] = {}
        self._pf_offset: dict[str, list[torch.Tensor]] = {}
        self._pf_w13_scale_fp32: list[torch.Tensor] = []
        self._initialized = False

        # Lightweight counters for [PREFETCH] stats.
        self.n_issued = 0
        self.n_hits = 0
        self.n_late = 0

    # ------------------------------------------------------------------ #
    #  Allocation (called after device weights + scale buffers exist)     #
    # ------------------------------------------------------------------ #

    def allocate(self) -> None:
        if self._initialized:
            return
        mgr = self.manager
        if not mgr.moe_layers:
            return
        layer0 = mgr.moe_layers[0]
        dev = layer0.w13_weight.device
        dt = layer0.w13_weight.dtype
        cap = self.capacity
        is_int8 = dt == torch.int8

        for _ in range(self.num_moe_layers):
            t13 = torch.empty((cap,) + tuple(layer0.w13_weight.shape[1:]),
                              dtype=dt, device=dev)
            t2 = torch.empty((cap,) + tuple(layer0.w2_weight.shape[1:]),
                             dtype=dt, device=dev)
            if is_int8:
                # W8A8 kernels require NZ; match the LRC tensor layout so the
                # storage-slice copies are byte-identical.
                t13 = torch_npu.npu_format_cast(t13, ACL_FORMAT_FRACTAL_NZ)
                t2 = torch_npu.npu_format_cast(t2, ACL_FORMAT_FRACTAL_NZ)
            self._pf_w13.append(t13)
            self._pf_w2.append(t2)

        # Per-attr scale/offset staging, mirroring the manager's CPU buffers.
        for attr in mgr.scale_cpu_buffers:
            self._pf_scale[attr] = []
        for attr in mgr.offset_cpu_buffers:
            self._pf_offset[attr] = []
        for layer in mgr.moe_layers:
            for attr, staging in self._pf_scale.items():
                dev_tensor = getattr(layer, attr, None)
                staging.append(
                    None if dev_tensor is None else
                    torch.empty((cap,) + tuple(dev_tensor.shape[1:]),
                                dtype=dev_tensor.dtype, device=dev))
            for attr, staging in self._pf_offset.items():
                dev_tensor = getattr(layer, attr, None)
                staging.append(
                    None if dev_tensor is None else
                    torch.empty((cap,) + tuple(dev_tensor.shape[1:]),
                                dtype=dev_tensor.dtype, device=dev))
            if hasattr(layer, "w13_weight_scale_fp32"):
                self._pf_w13_scale_fp32.append(
                    torch.empty((cap,) + tuple(layer.w13_weight_scale_fp32.shape[1:]),
                                dtype=torch.float32, device=dev))

        self._initialized = True
        logger.warning(
            "[PREFETCH] allocated staging cache: %d layers x %d slots, "
            "w13[0].shape=%s w2[0].shape=%s horizon=%d",
            self.num_moe_layers, cap,
            tuple(self._pf_w13[0].shape), tuple(self._pf_w2[0].shape),
            self.horizon)

    # ------------------------------------------------------------------ #
    #  Prediction substrate                                               #
    # ------------------------------------------------------------------ #

    def observe_routing(self, layer_idx: int, needed: set[int],
                        per_req_experts: dict | None) -> None:
        """Feed this layer's routing to the predictor.

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
    #  Issue: stage predicted experts for the next `horizon` layers       #
    # ------------------------------------------------------------------ #

    def run(self, current_layer_idx: int) -> None:
        """Predict + async-stage experts for layers ahead of ``current_layer_idx``.

        Must be called on the host (eager path) after the current layer's LRC
        paging. Issues non-blocking H2D copies on ``prefetch_stream``; nothing
        here blocks compute.
        """
        if not self._initialized:
            return
        mgr = self.manager
        sz13 = mgr.w13_expert_size_bytes
        sz2 = mgr.w2_expert_size_bytes
        last = self.num_moe_layers - 1

        for target in range(current_layer_idx + 1,
                            min(current_layer_idx + self.horizon, last) + 1):
            predicted = self.predictor.predict(target)
            if not predicted:
                continue
            lrc_resident = mgr.lrc_resident_for_layer(target)
            loading = self._loading_set(target)
            plan = self.policy.plan_loads(target, predicted, lrc_resident, loading)
            if not plan:
                continue

            with torch_npu.npu.stream(self.prefetch_stream):
                for load in plan:
                    eid, slot = load.expert_id, load.slot
                    self._pf_w13[target].untyped_storage()[
                        slot * sz13:(slot + 1) * sz13].copy_(
                        mgr.w13_weights_cpu[target][eid].untyped_storage(),
                        non_blocking=True)
                    self._pf_w2[target].untyped_storage()[
                        slot * sz2:(slot + 1) * sz2].copy_(
                        mgr.w2_weights_cpu[target][eid].untyped_storage(),
                        non_blocking=True)
                    self._stage_scales(target, eid, slot)

                    event = self._acquire_event()
                    event.record(self.prefetch_stream)
                    # Drop any stale event for an expert this load evicted.
                    if load.evicted_expert is not None:
                        self._release_event(
                            self._events.pop((target, load.evicted_expert), None))
                    self._events[(target, eid)] = event
                    self.n_issued += 1

        if self.debug:
            logger.warning(
                "[PREFETCH] after l=%d issued_total=%d hits_total=%d late_total=%d",
                current_layer_idx, self.n_issued, self.n_hits, self.n_late)

    def _stage_scales(self, target: int, eid: int, slot: int) -> None:
        """Copy W8A8 scale/offset (and derived fp32 scale) into the staging slot."""
        mgr = self.manager
        for attr, buffers in mgr.scale_cpu_buffers.items():
            staging = self._pf_scale.get(attr)
            if (staging is None or target >= len(staging) or staging[target] is None
                    or target >= len(buffers) or eid >= len(buffers[target])):
                continue
            staging[target][slot].copy_(buffers[target][eid], non_blocking=True)
        for attr, buffers in mgr.offset_cpu_buffers.items():
            staging = self._pf_offset.get(attr)
            if (staging is None or target >= len(staging) or staging[target] is None
                    or target >= len(buffers) or eid >= len(buffers[target])):
                continue
            staging[target][slot].copy_(buffers[target][eid], non_blocking=True)
        if self._pf_w13_scale_fp32 and target < len(self._pf_w13_scale_fp32):
            scale_staging = self._pf_scale.get("w13_weight_scale")
            if (scale_staging is not None and target < len(scale_staging)
                    and scale_staging[target] is not None):
                self._pf_w13_scale_fp32[target][slot].copy_(
                    scale_staging[target][slot].to(torch.float32))

    # ------------------------------------------------------------------ #
    #  Consume: promote a staged expert into an LRC slot (fast D2D)        #
    # ------------------------------------------------------------------ #

    def try_promote(self, layer, layer_idx: int, eid: int, lrc_slot: int) -> bool:
        """Promote staged expert ``eid`` into LRC ``lrc_slot`` via on-device copy.

        MUST be called from within the manager's ``load_stream`` context so the
        D2D copies are enqueued on that stream. Returns False (and does nothing)
        when ``eid`` is not staged or its H2D has not landed yet — the caller
        then performs the normal CPU->LRC H2D.
        """
        if not self._initialized:
            return False
        if not self.policy.is_resident(layer_idx, eid):
            return False
        event = self._events.get((layer_idx, eid))
        if event is None or not event.query():
            # Predicted but not yet landed — don't stall; fall back to H2D.
            if event is not None:
                self.n_late += 1
            return False
        pf_slot = self.policy.slot_of(layer_idx, eid)
        mgr = self.manager
        sz13 = mgr.w13_expert_size_bytes
        sz2 = mgr.w2_expert_size_bytes

        # Order the D2D after the staging H2D (no host stall: query() was True).
        mgr.load_stream.wait_event(event)
        layer.w13_weight.data.untyped_storage()[
            lrc_slot * sz13:(lrc_slot + 1) * sz13].copy_(
            self._pf_w13[layer_idx].untyped_storage()[
                pf_slot * sz13:(pf_slot + 1) * sz13])
        layer.w2_weight.data.untyped_storage()[
            lrc_slot * sz2:(lrc_slot + 1) * sz2].copy_(
            self._pf_w2[layer_idx].untyped_storage()[
                pf_slot * sz2:(pf_slot + 1) * sz2])
        for attr, staging in self._pf_scale.items():
            dev_tensor = getattr(layer, attr, None)
            if dev_tensor is None or layer_idx >= len(staging) or staging[layer_idx] is None:
                continue
            dev_tensor.data[lrc_slot].copy_(staging[layer_idx][pf_slot])
        for attr, staging in self._pf_offset.items():
            dev_tensor = getattr(layer, attr, None)
            if dev_tensor is None or layer_idx >= len(staging) or staging[layer_idx] is None:
                continue
            dev_tensor.data[lrc_slot].copy_(staging[layer_idx][pf_slot])
        if hasattr(layer, "w13_weight_scale_fp32"):
            layer.w13_weight_scale_fp32[lrc_slot].copy_(
                layer.w13_weight_scale.data[lrc_slot].to(torch.float32))

        self._release_event(self._events.pop((layer_idx, eid), None))
        self.policy.on_consumed(layer_idx, eid)
        self.n_hits += 1
        return True

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _loading_set(self, layer_idx: int) -> set[int]:
        """Experts whose staging copy hasn't landed — don't evict to make room."""
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
