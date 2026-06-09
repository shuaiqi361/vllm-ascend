"""Eviction policy for the proactive expert-prefetch staging cache.

This governs ONLY the separate prefetch cache (a staging buffer of
``num_prefetch_experts`` slots per MoE layer). It is deliberately independent
of :class:`LRCExpertCachePolicy`, which governs the demand-paging (LRC) cache.
The two policies only meet through a residency query: the prefetcher never
stages an expert that is already resident in the LRC cache.

The policy is intentionally CPU-only and torch-free (mirrors ``lrc_policy.py``)
so it can be unit-tested without NPU hardware. The :class:`ExpertPrefetcher`
engine owns the actual CPU->NPU copies and completion events; this class only
decides which experts to stage and which staged expert to evict.

Eviction follows MoE-Infinity's *production* rule (paper §4.7 as implemented in
the Archer engine): LFU over a per-expert visit count, never evicting an expert
in the current *protected* (just-predicted) set, and never evicting an expert
whose prefetch copy is still in flight (the *loading* set).
"""

from dataclasses import dataclass, field


@dataclass
class PrefetchLayerState:
    """Per-layer occupancy + hotness bookkeeping for the prefetch cache."""

    resident: dict[int, int] = field(default_factory=dict)   # expert_id -> slot
    free_slots: list[int] = field(default_factory=list)      # unused slot ids
    visit_count: dict[int, int] = field(default_factory=dict)  # expert_id -> LFU count
    last_used: dict[int, int] = field(default_factory=dict)  # expert_id -> step (tiebreak)
    step: int = 0


@dataclass
class PrefetchLoad:
    """A single staging decision the engine should execute."""

    expert_id: int          # expert to copy CPU -> prefetch slab
    slot: int               # target prefetch-cache slot
    evicted_expert: int | None  # expert that vacated the slot (None if it was free)


class PrefetchCachePolicy:
    """LFU-with-protected-set eviction for the prefetch staging cache.

    Args:
        num_layers: number of MoE layers (each gets an independent cache).
        num_experts: total routed expert count (for validation only).
        capacity: prefetch slots per layer (== num_prefetch_experts).
    """

    def __init__(self, num_layers: int, num_experts: int, capacity: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.num_experts = num_experts
        self.capacity = capacity
        self.layer_states = [
            PrefetchLayerState(free_slots=list(range(capacity)))
            for _ in range(num_layers)
        ]

    # -- queries ------------------------------------------------------- #

    def is_resident(self, layer_idx: int, expert_id: int) -> bool:
        return expert_id in self.layer_states[layer_idx].resident

    def slot_of(self, layer_idx: int, expert_id: int) -> int | None:
        return self.layer_states[layer_idx].resident.get(expert_id)

    def resident_experts(self, layer_idx: int) -> set[int]:
        return set(self.layer_states[layer_idx].resident)

    # -- planning ------------------------------------------------------ #

    def plan_loads(
        self,
        layer_idx: int,
        predicted: list[int],
        lrc_resident: set[int],
        loading: set[int] | None = None,
    ) -> list[PrefetchLoad]:
        """Decide which predicted experts to stage into the prefetch cache.

        Args:
            layer_idx: target MoE layer.
            predicted: predicted-needed experts, **priority order** (nearest /
                highest weight first); only these are staged.
            lrc_resident: experts currently resident in the LRC cache for this
                layer. Experts already in the LRC cache are never staged — this
                is the sole coupling between the two caches.
            loading: experts whose prefetch copy from a prior round is still in
                flight; never evicted to make room.

        Returns:
            Ordered list of :class:`PrefetchLoad`. Applying them (via
            :meth:`commit_load`) mutates this layer's residency. Predicted
            experts that are already resident (LRC or prefetch) are skipped;
            their hotness is still refreshed so repeated predictions resist
            eviction.
        """
        state = self.layer_states[layer_idx]
        state.step += 1
        loading = loading or set()
        protected = set(predicted)

        # Refresh hotness for every predicted expert (prediction == intent to
        # access) so frequently-predicted experts accumulate LFU weight.
        for eid in predicted:
            state.visit_count[eid] = state.visit_count.get(eid, 0) + 1
            state.last_used[eid] = state.step

        plan: list[PrefetchLoad] = []
        # Reserve slots optimistically as we plan, so a single round never
        # double-assigns a slot.
        for eid in predicted:
            if eid in lrc_resident:
                continue  # already in the LRC cache — do not stage
            if eid in state.resident:
                continue  # already staged — keep it (hotness already bumped)
            if state.free_slots:
                slot = state.free_slots.pop()
                victim = None
            else:
                victim = self._choose_victim(layer_idx, protected, loading)
                if victim is None:
                    # No evictable slot (everything protected / in flight).
                    continue
                slot = state.resident.pop(victim)
            state.resident[eid] = slot  # optimistic; engine confirms via commit
            plan.append(PrefetchLoad(expert_id=eid, slot=slot, evicted_expert=victim))
        return plan

    def _choose_victim(
        self, layer_idx: int, protected: set[int], loading: set[int]
    ) -> int | None:
        """Return the coldest staged expert eligible for eviction (LFU)."""
        state = self.layer_states[layer_idx]
        candidates = [
            eid for eid in state.resident
            if eid not in protected and eid not in loading
        ]
        if not candidates:
            # Relax the loading constraint before giving up (mirrors LRC).
            candidates = [eid for eid in state.resident if eid not in protected]
        if not candidates:
            return None
        return min(candidates, key=lambda eid: self._victim_key(layer_idx, eid))

    def _victim_key(self, layer_idx: int, expert_id: int) -> tuple[int, int, int]:
        state = self.layer_states[layer_idx]
        # Lower visit_count first (LFU); then least-recently-predicted; then id.
        return (
            state.visit_count.get(expert_id, 0),
            state.last_used.get(expert_id, -1),
            expert_id,
        )

    # -- mutations the engine reports back ----------------------------- #

    def commit_load(self, layer_idx: int, load: PrefetchLoad) -> None:
        """No-op hook: plan_loads already updated residency optimistically.

        Kept so the engine can signal a *failed* issue and roll back via
        :meth:`rollback_load` without the policy assuming success.
        """

    def rollback_load(self, layer_idx: int, load: PrefetchLoad) -> None:
        """Undo an optimistic load the engine could not issue."""
        state = self.layer_states[layer_idx]
        if state.resident.get(load.expert_id) == load.slot:
            del state.resident[load.expert_id]
        # Restore the evicted occupant if there was one, else free the slot.
        if load.evicted_expert is not None:
            state.resident[load.evicted_expert] = load.slot
        elif load.slot not in state.free_slots:
            state.free_slots.append(load.slot)

    def on_consumed(self, layer_idx: int, expert_id: int) -> None:
        """A staged expert was promoted into the LRC cache; free its slot.

        Hotness (visit_count / last_used) is intentionally retained so a
        re-predicted expert keeps its history across promotion.
        """
        state = self.layer_states[layer_idx]
        slot = state.resident.pop(expert_id, None)
        if slot is None:
            return
        state.visit_count[expert_id] = state.visit_count.get(expert_id, 0) + 1
        state.last_used[expert_id] = state.step
        if slot not in state.free_slots:
            state.free_slots.append(slot)
