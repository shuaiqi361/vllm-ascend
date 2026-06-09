"""Pure-Python simulation of the manager's prefetch orchestration.

This exercises the real :class:`RecentUnionPredictor` + :class:`PrefetchCachePolicy`
cooperating exactly as ``ExpertOffloadManager._update_weights`` drives them
(predict future layer -> stage non-LRC-resident -> promote on hit), minus the
NPU copies. It is the logic-level integration test for Plan A: a bug in the
residency coupling, the promote/consume bookkeeping, or the predict->stage
ordering would surface here.

Workload: 2 MoE layers, decode (1 token/step), top-2 routing. Layer 1's routed
set alternates {0,1} / {2,3} every step while the LRC cache holds only 2 slots,
so the LRC cache *thrashes* — every step misses both experts. A prefetch cache
of 3 slots + the recent-union predictor should, after a short warmup, stage
exactly the experts layer 1 is about to need, turning every cold miss into a
fast prefetch hit.
"""

from collections import OrderedDict

from vllm_ascend.expert_offload.prefetch_cache_policy import PrefetchCachePolicy
from vllm_ascend.expert_offload.prefetch_predictor import RecentUnionPredictor


class _LRC:
    """Minimal LRU model of one layer's demand-paging cache."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.slots = OrderedDict()  # eid -> True, MRU at end

    def __contains__(self, eid):
        return eid in self.slots

    def resident(self):
        return set(self.slots)

    def touch(self, eid):
        self.slots.move_to_end(eid)

    def insert(self, eid):
        self.slots[eid] = True
        self.slots.move_to_end(eid)
        while len(self.slots) > self.capacity:
            self.slots.popitem(last=False)


def _run_sim(steps):
    NUM_LAYERS = 2
    NUM_EXPERTS = 8
    HORIZON = 1
    predictor = RecentUnionPredictor(NUM_LAYERS, window=4)
    policy = PrefetchCachePolicy(NUM_LAYERS, NUM_EXPERTS, capacity=3)
    lrc = [_LRC(capacity=2) for _ in range(NUM_LAYERS)]

    def routing(step, layer):
        if layer == 0:
            return {6, 7}                      # stable: stays LRC-resident
        return {0, 1} if step % 2 == 0 else {2, 3}  # thrashes the 2-slot LRC

    stats = {"cold": [0, 0], "promote": [0, 0], "lrc_hit": [0, 0]}

    for t in range(steps):
        for layer in range(NUM_LAYERS):
            needed = routing(t, layer)
            # --- page current layer (consume staged experts on a miss) ---
            for eid in sorted(needed):
                if eid in lrc[layer]:
                    lrc[layer].touch(eid)
                    stats["lrc_hit"][layer] += 1
                elif policy.is_resident(layer, eid):
                    stats["promote"][layer] += 1   # prefetch HIT -> fast D2D
                    policy.on_consumed(layer, eid)
                    lrc[layer].insert(eid)
                else:
                    stats["cold"][layer] += 1       # cold miss -> slow H2D
                    lrc[layer].insert(eid)
            # --- observe + stage for future layers (manager.run) ---
            predictor.observe(layer, needed)
            for tgt in range(layer + 1, min(layer + HORIZON, NUM_LAYERS - 1) + 1):
                predicted = predictor.predict(tgt)
                policy.plan_loads(tgt, predicted, lrc[tgt].resident())
    return stats


def test_prefetch_eliminates_cold_misses_after_warmup():
    stats = _run_sim(steps=12)
    # Layer 1 thrashes the LRC cache, so without prefetch every step would be
    # two cold misses. With prefetch, cold misses must be replaced by hits.
    assert stats["promote"][1] > 0
    # Steady state should be dominated by prefetch hits, not cold misses.
    assert stats["promote"][1] > stats["cold"][1]


def test_warmup_cold_then_steady_state_hits():
    # Track per-step to confirm cold misses vanish after the predictor warms up.
    cold_by_step = []
    NUM_LAYERS, NUM_EXPERTS = 2, 8
    predictor = RecentUnionPredictor(NUM_LAYERS, window=4)
    policy = PrefetchCachePolicy(NUM_LAYERS, NUM_EXPERTS, capacity=3)
    lrc = [_LRC(2) for _ in range(NUM_LAYERS)]

    def routing(step, layer):
        return {0, 1} if step % 2 == 0 else {2, 3}

    for t in range(8):
        step_cold = 0
        for layer in range(NUM_LAYERS):
            needed = routing(t, layer)
            for eid in sorted(needed):
                if eid in lrc[layer]:
                    lrc[layer].touch(eid)
                elif policy.is_resident(layer, eid):
                    policy.on_consumed(layer, eid)
                    lrc[layer].insert(eid)
                else:
                    if layer == 1:
                        step_cold += 1
                    lrc[layer].insert(eid)
            predictor.observe(layer, needed)
            for tgt in range(layer + 1, NUM_LAYERS):
                policy.plan_loads(tgt, predictor.predict(tgt), lrc[tgt].resident())
        cold_by_step.append(step_cold)

    # Warmup has cold misses; once the predictor has both routed sets in its
    # window and prefetch has staged them, steady-state cold misses hit zero.
    assert sum(cold_by_step[:2]) > 0
    assert cold_by_step[-1] == 0
    assert cold_by_step[-2] == 0


def test_stable_layer_never_needs_prefetch():
    # Layer 0's routing is stable and fits the LRC cache, so it should never
    # cold-miss after the first step and never need a prefetch promote.
    stats = _run_sim(steps=12)
    assert stats["promote"][0] == 0
    assert stats["cold"][0] == 2  # only the very first step's two experts
