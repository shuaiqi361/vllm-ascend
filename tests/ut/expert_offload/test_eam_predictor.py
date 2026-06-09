from vllm_ascend.expert_offload.eam_predictor import (
    EAMCollection,
    EAMPredictor,
    ExpertActivationMatrix,
)


def test_matrix_records_and_tracks_max_layer():
    m = ExpertActivationMatrix(num_layers=4, num_experts=8)
    m.record(0, [1, 2])
    m.record(2, [3, 3])
    assert m.matrix[0][1] == 1 and m.matrix[0][2] == 1
    assert m.matrix[2][3] == 2
    assert m.max_layer_seen == 2
    assert m.has_routed(2) and not m.has_routed(1)


def test_eamc_cosine_match_picks_closest():
    eamc = EAMCollection(capacity=4)
    a = [[5, 0, 0, 0], [0, 5, 0, 0]]   # routes expert 0 then 1
    b = [[0, 0, 5, 0], [0, 0, 0, 5]]   # routes expert 2 then 3
    eamc.add(a)
    eamc.add(b)
    query = [[3, 0, 0, 0], [0, 0, 0, 0]]  # looks like 'a' on layer 0
    best = eamc.find_most_similar(query, upto_layer=0)
    assert best == a


def test_predict_uses_own_history_in_steady_state():
    p = EAMPredictor(num_layers=3, num_experts=6)
    p.start_request("r1")
    # r1 has decoded several steps; layer 2 consistently routes {4,5}.
    for _ in range(3):
        p.observe_request("r1", 0, [0, 1])
        p.observe_request("r1", 1, [2, 3])
        p.observe_request("r1", 2, [4, 5])
    # While "at" layer 1, predicting layer 2 should surface its own {4,5}.
    p._current_layer["r1"] = 1
    pred = p.predict(2)
    assert set(pred) == {4, 5}


def test_cold_start_borrows_from_eamc():
    p = EAMPredictor(num_layers=3, num_experts=6)
    # A finished request established the pattern layer0={0,1} -> layer2={4,5}.
    p.start_request("done")
    p.observe_request("done", 0, [0, 1])
    p.observe_request("done", 2, [4, 5])
    p.finish_request("done")
    assert len(p.eamc) == 1

    # New request matches 'done' on layer 0 but has NOT routed layer 2 yet.
    p.start_request("new")
    p.observe_request("new", 0, [0, 1])
    pred = p.predict(2)  # layer 2 unseen by 'new' -> cold-start cosine match
    assert set(pred) == {4, 5}


def test_finish_request_folds_into_eamc_and_clears_active():
    p = EAMPredictor(num_layers=2, num_experts=4)
    p.start_request("r")
    p.observe_request("r", 0, [1])
    p.finish_request("r")
    assert "r" not in p._requests
    assert len(p.eamc) == 1


def test_sync_active_retires_departed_requests():
    p = EAMPredictor(num_layers=2, num_experts=4)
    p.start_request("a")
    p.observe_request("a", 0, [1])
    p.start_request("b")
    p.observe_request("b", 0, [2])
    p.sync_active(["a"])  # b departed
    assert "b" not in p._requests
    assert "a" in p._requests
    assert len(p.eamc) == 1  # b folded in


def test_layer_decay_prefers_nearer_layers_in_weight():
    p = EAMPredictor(num_layers=10, num_experts=4)
    p.start_request("r")
    p.observe_request("r", 0, [1])
    p._current_layer["r"] = 0
    # Same raw count at layer 1 vs layer 5; nearer layer must weigh more.
    w_near = p._layer_decay(1, 0)
    w_far = p._layer_decay(5, 0)
    assert w_near > w_far > 0


def test_predict_aggregates_over_active_requests():
    p = EAMPredictor(num_layers=2, num_experts=6)
    for rid, experts in [("r1", [4, 5]), ("r2", [5])]:
        p.start_request(rid)
        p.observe_request(rid, 0, [0])
        p.observe_request(rid, 1, experts)
        p._current_layer[rid] = 0
    pred = p.predict(1)
    # expert 5 chosen by both requests -> highest aggregate weight -> first.
    assert pred[0] == 5
    assert set(pred) == {4, 5}


def test_eamc_capacity_evicts_least_accessed():
    eamc = EAMCollection(capacity=2)
    a = [[5, 0]]
    b = [[0, 5]]
    eamc.add(a)
    eamc.add(b)
    # Access 'a' so 'b' becomes least-accessed.
    eamc.find_most_similar([[5, 0]], upto_layer=0)
    c = [[3, 3]]
    eamc.add(c)  # should evict b, keep a + c
    assert len(eamc) == 2
    assert a in eamc._entries and c in eamc._entries
    assert b not in eamc._entries
