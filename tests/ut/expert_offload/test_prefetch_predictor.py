from vllm_ascend.expert_offload.prefetch_predictor import RecentUnionPredictor


def test_empty_history_predicts_nothing():
    p = RecentUnionPredictor(num_layers=2, window=4)
    assert p.predict(0) == []


def test_predicts_recent_union():
    p = RecentUnionPredictor(num_layers=1, window=4)
    p.observe(0, [1, 2])
    p.observe(0, [2, 3])
    # union of recent steps, regardless of current residency
    assert set(p.predict(0)) == {1, 2, 3}


def test_priority_orders_by_frequency_then_recency():
    p = RecentUnionPredictor(num_layers=1, window=8)
    p.observe(0, [1, 9])   # 9 seen once, oldest
    p.observe(0, [1, 2])   # 1 seen twice
    p.observe(0, [2, 3])   # 2 seen twice, more recent than 1's last sight
    pred = p.predict(0)
    # 1 and 2 both count==2; 2 was seen more recently -> ahead of 1.
    assert pred[0] == 2
    assert pred[1] == 1
    # singletons (3, 9) come after; 3 more recent than 9.
    assert pred.index(3) < pred.index(9)


def test_window_evicts_old_steps():
    p = RecentUnionPredictor(num_layers=1, window=2)
    p.observe(0, [1])
    p.observe(0, [2])
    p.observe(0, [3])  # pushes out [1]
    assert set(p.predict(0)) == {2, 3}


def test_layers_independent():
    p = RecentUnionPredictor(num_layers=3, window=4)
    p.observe(0, [1, 2])
    p.observe(2, [7])
    assert set(p.predict(0)) == {1, 2}
    assert p.predict(1) == []
    assert set(p.predict(2)) == {7}


def test_observe_accepts_list_and_dedupes():
    p = RecentUnionPredictor(num_layers=1, window=4)
    p.observe(0, [5, 5, 6])  # duplicate expert ids collapse
    assert set(p.predict(0)) == {5, 6}
