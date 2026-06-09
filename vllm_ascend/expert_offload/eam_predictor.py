"""MoE-Infinity-style Expert Activation Matrix (EAM) predictor — Plan B.

This is the faithful port of MoE-Infinity's prediction "brain"
(``expert_tracer.py`` + ``expert_predictor.py``): a per-request Expert
Activation Matrix, a fixed-capacity collection of completed-request EAMs
(EAMC), cosine-similarity matching, and layer-proximity decay. It is a drop-in
predictor for the same :class:`ExpertPrefetcher` engine Plan A uses — only the
*prediction* strategy changes.

It is CPU-only and torch-free (cosine via plain ``math``) so it is unit-testable
without NPU hardware, exactly like the LRC policy and the Plan-A predictor.

Adaptation to autoregressive decode (documented because it departs from the
paper's prefill-oriented framing): the paper builds one EAM progressively, layer
by layer, so layer L+1 is a genuine "future" unknown. In vLLM decode every step
re-runs *all* layers, so a request that has decoded for a few steps already has
its own routing history for every layer. We therefore predict layer L's experts
primarily from the **request's own accumulated EAM** (the strongest per-request
signal), and fall back to a **cosine match against the EAMC of finished
requests** only when this request has not yet routed that layer (cold start, the
first decode step / a brand-new request). Layer-proximity decay is applied in
both cases so nearer layers dominate the bounded prefetch cache.
"""

import math
from collections.abc import Iterable


class ExpertActivationMatrix:
    """Per-request routing history: ``matrix[layer][expert]`` = token count."""

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.matrix = [[0] * num_experts for _ in range(num_layers)]
        self.max_layer_seen = -1  # highest layer this request has routed

    def record(self, layer_idx: int, experts: Iterable[int]) -> None:
        row = self.matrix[layer_idx]
        for eid in experts:
            row[int(eid)] += 1
        if layer_idx > self.max_layer_seen:
            self.max_layer_seen = layer_idx

    def has_routed(self, layer_idx: int) -> bool:
        return any(self.matrix[layer_idx])


def _cosine(a: list[int], b: list[int]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class EAMCollection:
    """Fixed-capacity store of completed-request EAMs (the EAMC)."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._entries: list[list[list[int]]] = []  # each is a full matrix copy
        self._access: list[int] = []               # LFU counter per entry

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, matrix: list[list[int]]) -> None:
        snapshot = [row[:] for row in matrix]
        if len(self._entries) < self.capacity:
            self._entries.append(snapshot)
            self._access.append(0)
            return
        # Evict the least-accessed entry (MoE-Infinity production behaviour).
        victim = min(range(len(self._entries)), key=lambda i: self._access[i])
        self._entries[victim] = snapshot
        self._access[victim] = 0

    def find_most_similar(self, matrix: list[list[int]],
                          upto_layer: int) -> list[list[int]] | None:
        """Most similar stored EAM by mean per-layer cosine over [0, upto_layer]."""
        if not self._entries:
            return None
        best_idx = -1
        best_sim = -1.0
        for idx, cand in enumerate(self._entries):
            total = 0.0
            counted = 0
            for layer in range(upto_layer + 1):
                if not any(matrix[layer]) and not any(cand[layer]):
                    continue
                total += _cosine(matrix[layer], cand[layer])
                counted += 1
            sim = total / counted if counted else 0.0
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx < 0:
            return None
        self._access[best_idx] += 1
        return self._entries[best_idx]


class EAMPredictor:
    """Per-request EAM predictor exposing the engine's predict() interface.

    Engine-facing surface mirrors :class:`RecentUnionPredictor`:
        predict(layer_idx) -> list[int]    # priority-ordered, aggregated
    plus per-request lifecycle the manager drives:
        start_request / observe_request / finish_request / sync_active.
    """

    def __init__(self, num_layers: int, num_experts: int,
                 eamc_capacity: int = 256) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.eamc = EAMCollection(eamc_capacity)
        self._requests: dict[str, ExpertActivationMatrix] = {}
        self._current_layer: dict[str, int] = {}

    # -- per-request lifecycle ----------------------------------------- #

    def start_request(self, request_id: str) -> None:
        if request_id not in self._requests:
            self._requests[request_id] = ExpertActivationMatrix(
                self.num_layers, self.num_experts)
            self._current_layer[request_id] = -1

    def observe_request(self, request_id: str, layer_idx: int,
                        experts: Iterable[int]) -> None:
        if request_id not in self._requests:
            self.start_request(request_id)
        self._requests[request_id].record(layer_idx, experts)
        self._current_layer[request_id] = layer_idx

    def finish_request(self, request_id: str) -> None:
        eam = self._requests.pop(request_id, None)
        self._current_layer.pop(request_id, None)
        if eam is not None and eam.max_layer_seen >= 0:
            self.eamc.add(eam.matrix)

    def sync_active(self, active_request_ids: Iterable[str]) -> None:
        """Retire requests no longer in the batch (fold their EAM into EAMC)."""
        active = set(active_request_ids)
        for rid in [r for r in self._requests if r not in active]:
            self.finish_request(rid)

    # -- prediction ---------------------------------------------------- #

    def _layer_decay(self, target_layer: int, current_layer: int) -> float:
        # MoE-Infinity (1 - (target-current)/(L+1)); nearer layers weigh more.
        return max(0.0, 1.0 - (target_layer - current_layer) / (self.num_layers + 1))

    def predict_weights_for_request(self, request_id: str,
                                    target_layer: int) -> dict[int, float]:
        eam = self._requests.get(request_id)
        if eam is None:
            return {}
        current_layer = self._current_layer.get(request_id, -1)
        decay = self._layer_decay(target_layer, current_layer)
        # Primary signal: this request's own accumulated routing at the layer.
        if eam.has_routed(target_layer):
            source = eam.matrix[target_layer]
        else:
            # Cold start: borrow the most similar finished request's pattern.
            similar = self.eamc.find_most_similar(eam.matrix, max(current_layer, 0))
            if similar is None:
                return {}
            source = similar[target_layer]
        return {e: c * decay for e, c in enumerate(source) if c > 0}

    def predict(self, target_layer: int) -> list[int]:
        """Aggregate predicted experts for ``target_layer`` over active requests."""
        weights: dict[int, float] = {}
        for rid in self._requests:
            for eid, w in self.predict_weights_for_request(rid, target_layer).items():
                weights[eid] = weights.get(eid, 0.0) + w
        return sorted(weights, key=lambda eid: (-weights[eid], eid))
