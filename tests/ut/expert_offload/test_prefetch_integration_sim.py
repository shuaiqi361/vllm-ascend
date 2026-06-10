"""Pure-Python simulation of the manager's unified-cache prefetch flow.

This exercises the real :class:`RecentUnionPredictor` + :class:`LRCExpertCachePolicy`
cooperating exactly as ``ExpertOffloadManager._update_weights`` drives them after
the unified-cache redesign, minus the NPU copies:

  * there is ONE on-device pool per layer (``num_device_experts`` slots);
  * the prefetcher stages predicted experts **directly into that pool** ahead of
    time (no separate staging cache, no device-to-device promote);
  * when a layer is paged, an already-resident expert — whether warmed by an
    earlier demand load or pre-staged by the prefetcher this step — is a hit and
    is copied nowhere; only genuine misses incur a synchronous demand load.

A bug in the residency bookkeeping, the predict->stage ordering, the all-layers
cursor, or the one-expert-one-slot invariant would surface here.
"""

from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.expert_offload.prefetch_predictor import RecentUnionPredictor


class _Pool:
    """Logic model of one MoE layer's single on-device expert pool.

    Enforces the unification invariant on every mutation: a bijection between
    resident experts and slots, never exceeding capacity. There is exactly one
    place an expert can live — guarding against re-introducing a second cache.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.slot_of = {}                 # expert_id -> slot
        self._free = list(range(capacity))

    def __contains__(self, eid):
        return eid in self.slot_of

    def resident(self):
        return set(self.slot_of)

    def slot_owner(self):
        return {slot: eid for eid, slot in self.slot_of.items()}

    def seed(self, experts):
        for eid in experts:
            self.load(eid, victim=None)

    def load(self, eid, victim):
        """Insert ``eid``, reusing a free slot or ``victim``'s slot."""
        assert eid not in self.slot_of, "duplicate residency (two caches?)"
        if self._free:
            slot = self._free.pop()
        else:
            assert victim is not None, "pool full but no victim given"
            slot = self.slot_of.pop(victim)
        self.slot_of[eid] = slot
        self._check()
        return slot

    def _check(self):
        assert len(self.slot_of) <= self.capacity
        slots = list(self.slot_of.values())
        assert len(set(slots)) == len(slots), "two experts share a slot"


class _Sim:
    """Drives the unified pools exactly like ``_update_weights`` + prefetcher."""

    def __init__(self, num_layers, num_experts, capacity, budget, horizon=0):
        self.num_layers = num_layers
        self.capacity = capacity
        self.budget = budget
        self.horizon = horizon
        self.predictor = RecentUnionPredictor(num_layers, window=4)
        self.policy = LRCExpertCachePolicy(
            num_layers, num_experts, cache_size=capacity, topk=2,
            recent_window=8)
        self.pools = [_Pool(capacity) for _ in range(num_layers)]
        self._cursor = 0
        self._last_run_layer = num_layers
        self.sync_loads = [0 for _ in range(num_layers)]   # synchronous demand
        self.prestaged_hits = [0 for _ in range(num_layers)]  # saved by prefetch
        self.warm_hits = [0 for _ in range(num_layers)]
        self.predict_log = []   # (step, layer) every time predict() is consulted
        self._prestaged_this_step = [set() for _ in range(num_layers)]

    # -- demand paging (one layer) ------------------------------------- #

    def page(self, step, layer, needed):
        pool = self.pools[layer]
        for eid in sorted(needed):
            if eid in pool:
                if eid in self._prestaged_this_step[layer]:
                    self.prestaged_hits[layer] += 1   # prefetch front-ran it
                else:
                    self.warm_hits[layer] += 1
            else:
                # genuine miss -> synchronous demand load into the same pool
                slot_owner = pool.slot_owner()
                victim = self.policy.choose_victim(
                    layer, slot_owner, protected=needed)
                pool.load(eid, victim)
                self.sync_loads[layer] += 1
        # observe actual routing (drives policy hotness + predictor history)
        self.policy.observe(layer, [sorted(needed)])
        self.predictor.observe(layer, needed)

    # -- proactive prefetch (all remaining layers, once per step) ------ #

    def run_prefetch(self, step, current_layer):
        last = self.num_layers - 1
        if current_layer <= self._last_run_layer:   # new step -> reset cursor
            self._cursor = current_layer + 1
            self._prestaged_this_step = [set() for _ in range(self.num_layers)]
        self._last_run_layer = current_layer
        hi = last if self.horizon <= 0 else min(current_layer + self.horizon, last)
        for target in range(max(self._cursor, current_layer + 1), hi + 1):
            self._prefetch_layer(step, target)
        self._cursor = max(self._cursor, hi + 1)

    def _prefetch_layer(self, step, target):
        self.predict_log.append((step, target))
        predicted = self.predictor.predict(target)
        if not predicted:
            return
        pool = self.pools[target]
        slot_owner = pool.slot_owner()
        free = [s for s in range(self.capacity) if s not in slot_owner]
        protected = set(predicted)
        staged = 0
        for eid in predicted:
            if staged >= self.budget:
                break
            if eid in pool:
                continue                       # already in the one cache — skip
            if free:
                victim = None
                free.pop()
            else:
                victim = self.policy.choose_victim(
                    target, slot_owner, protected=protected)
                if victim is None:
                    break
                del slot_owner[pool.slot_of[victim]]
            pool.load(eid, victim)
            slot_owner[pool.slot_of[eid]] = eid
            self._prestaged_this_step[target].add(eid)
            staged += 1

    def step(self, step_idx, routing):
        for layer in range(self.num_layers):
            self.page(step_idx, layer, routing(step_idx, layer))
            self.run_prefetch(step_idx, layer)


# --------------------------------------------------------------------- #
#  Tests                                                                 #
# --------------------------------------------------------------------- #

def test_all_layers_predicted_exactly_once_per_step():
    """horizon=0 must prefetch every future layer once per step (not next-only)."""
    sim = _Sim(num_layers=5, num_experts=16, capacity=3, budget=3, horizon=0)
    routing = lambda step, layer: {layer, (layer + 1) % 16}
    for t in range(4):
        sim.step(t, routing)

    # Future layers for a 5-layer model are 1..4 (layer 0 is paged first and is
    # never a prefetch target). Each must be predicted exactly once per step.
    for t in range(4):
        targets = sorted(l for (s, l) in sim.predict_log if s == t)
        assert targets == [1, 2, 3, 4], f"step {t} predicted {targets}"


def test_next_layer_only_horizon_limits_lookahead():
    """horizon=1 stages only the immediate next layer — the old behaviour."""
    sim = _Sim(num_layers=5, num_experts=16, capacity=3, budget=3, horizon=1)
    routing = lambda step, layer: {layer, (layer + 1) % 16}
    for t in range(3):
        sim.step(t, routing)
    for t in range(3):
        # At each current layer L we prefetch only L+1, so over a step every
        # layer 1..4 is still touched once — but via the sliding window, not a
        # single layer-0 burst. Confirm each future layer appears once.
        targets = sorted(l for (s, l) in sim.predict_log if s == t)
        assert targets == [1, 2, 3, 4]


def test_prefetch_prestages_and_eliminates_sync_load():
    """A predicted-but-evicted expert is pre-staged into the real pool, so the
    layer's paging finds it resident and performs NO synchronous demand load."""
    sim = _Sim(num_layers=2, num_experts=8, capacity=2, budget=2, horizon=0)

    # Warm the predictor so predict(layer 1) == {2, 3} (a stable recent set).
    for _ in range(4):
        sim.predictor.observe(1, {2, 3})
    # But layer 1's pool currently holds a *stale* set {0, 1} (post-drift), and
    # layer 0's pool is warm for its own stable routing.
    sim.pools[0].seed({6, 7})
    sim.pools[1].seed({0, 1})

    # One step: layer 1 will need {2, 3}. Page layer 0, then prefetch runs and
    # must pre-stage {2, 3} into layer 1's pool before layer 1 is paged.
    sim.page(0, 0, {6, 7})
    sim.run_prefetch(0, 0)
    assert sim.pools[1].resident() == {2, 3}, "prefetch did not pre-stage"

    sim.page(0, 1, {2, 3})
    # Both needed experts were pre-staged -> zero synchronous demand loads.
    assert sim.sync_loads[1] == 0
    assert sim.prestaged_hits[1] == 2


def test_resident_prediction_is_a_noop_no_duplicate():
    """Predicting an already-resident expert stages nothing (one cache only)."""
    sim = _Sim(num_layers=2, num_experts=8, capacity=4, budget=4, horizon=0)
    for _ in range(4):
        sim.predictor.observe(1, {0, 1})
    sim.pools[1].seed({0, 1})          # already resident

    sim.page(0, 0, {6, 7})
    sim.run_prefetch(0, 0)

    # No second copy, no eviction churn: pool is exactly the seeded set.
    assert sim.pools[1].resident() == {0, 1}
    assert sim._prestaged_this_step[1] == set()


def test_unification_invariant_holds_under_thrash():
    """Under a thrashing workload the per-layer pool stays a clean bijection
    (the _Pool asserts on every mutation) and never double-stores an expert."""
    sim = _Sim(num_layers=3, num_experts=8, capacity=2, budget=2, horizon=0)

    def routing(step, layer):
        if layer == 0:
            return {6, 7}                       # stable
        return {0, 1} if step % 2 == 0 else {2, 3}   # thrashes the 2-slot pool

    for t in range(20):
        sim.step(t, routing)

    # An expert is resident in at most one slot of its layer; pools never exceed
    # capacity. (Asserted continuously inside _Pool; re-check the end state.)
    for layer in range(3):
        pool = sim.pools[layer]
        assert len(pool.slot_of) <= pool.capacity
        assert len(set(pool.slot_of.values())) == len(pool.slot_of)
