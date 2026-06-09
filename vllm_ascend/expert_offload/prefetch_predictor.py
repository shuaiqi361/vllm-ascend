"""Future-layer expert predictors for proactive prefetch.

A predictor answers a single question for the prefetcher: *which routed
experts is MoE layer ``L`` about to need?* It is queried for **future** layers
during the current layer's paging, so the answer is necessarily a prediction
from history, not an exact lookahead (layer ``L+1``'s router input is layer
``L``'s not-yet-produced output).

All predictors here are CPU-only and torch-free so they can be unit-tested
without NPU hardware, and they share one tiny interface so the engine can swap
strategies:

    observe(layer_idx, needed)      # record this step's actual routing
    predict(layer_idx) -> list[int] # priority-ordered predicted experts

Plan A ships :class:`RecentUnionPredictor` (a global, request-agnostic
recent-history model). Plan B will add an activation-matrix / similarity
predictor implementing the same interface.
"""

from collections import deque
from collections.abc import Iterable


class RecentUnionPredictor:
    """Predict a layer's experts as the union of its recent routed sets.

    Rationale: in steady-state decode, per-layer routing is highly stable
    step-to-step, so "what layer L routed to over the last ``window`` steps" is
    a strong predictor of "what layer L needs this step" — and, crucially, that
    union includes experts that have recently drifted *out* of the LRC cache
    (the addressable prefetch misses), not just the currently-hot set.

    Experts are returned in priority order: most-frequently then
    most-recently seen first, so the prefetcher (and the bounded prefetch
    cache) spends its slots on the likeliest experts first.

    Args:
        num_layers: number of MoE layers.
        window: number of recent decode steps to remember per layer.
    """

    def __init__(self, num_layers: int, window: int) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        # Each entry is the tuple of unique experts routed at one decode step.
        self._history: list[deque[tuple[int, ...]]] = [
            deque(maxlen=window) for _ in range(num_layers)
        ]

    def observe(self, layer_idx: int, needed: Iterable[int]) -> None:
        """Record the experts actually routed at ``layer_idx`` this step."""
        experts = tuple(sorted({int(e) for e in needed}))
        self._history[layer_idx].append(experts)

    def predict(self, layer_idx: int) -> list[int]:
        """Return predicted-needed experts for ``layer_idx``, priority-ordered."""
        history = self._history[layer_idx]
        count: dict[int, int] = {}
        last_seen: dict[int, int] = {}
        for step_pos, experts in enumerate(history):
            for eid in experts:
                count[eid] = count.get(eid, 0) + 1
                last_seen[eid] = step_pos  # larger == more recent
        # Higher frequency first, then more recent, then stable by id.
        return sorted(
            count,
            key=lambda eid: (-count[eid], -last_seen[eid], eid),
        )
