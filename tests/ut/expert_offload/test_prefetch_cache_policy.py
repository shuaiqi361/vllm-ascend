from vllm_ascend.expert_offload.prefetch_cache_policy import (
    PrefetchCachePolicy,
)


def _make(capacity=4, num_layers=1, num_experts=16):
    return PrefetchCachePolicy(
        num_layers=num_layers, num_experts=num_experts, capacity=capacity
    )


def test_loads_predicted_non_resident_into_free_slots():
    policy = _make(capacity=4)
    plan = policy.plan_loads(0, predicted=[1, 2, 3], lrc_resident=set())
    assert [p.expert_id for p in plan] == [1, 2, 3]
    # distinct slots, all from the free pool, no evictions
    assert len({p.slot for p in plan}) == 3
    assert all(p.evicted_expert is None for p in plan)
    assert policy.resident_experts(0) == {1, 2, 3}


def test_skips_experts_already_in_lrc_cache():
    """The sole coupling: never stage what the LRC cache already holds."""
    policy = _make(capacity=4)
    plan = policy.plan_loads(0, predicted=[1, 2, 3], lrc_resident={2})
    assert [p.expert_id for p in plan] == [1, 3]
    assert policy.is_resident(0, 1)
    assert not policy.is_resident(0, 2)  # 2 served from LRC, not staged
    assert policy.is_resident(0, 3)


def test_skips_experts_already_staged():
    policy = _make(capacity=4)
    policy.plan_loads(0, predicted=[1, 2], lrc_resident=set())
    plan = policy.plan_loads(0, predicted=[1, 2, 3], lrc_resident=set())
    assert [p.expert_id for p in plan] == [3]  # 1,2 already staged


def test_lfu_eviction_keeps_frequently_predicted():
    policy = _make(capacity=2)
    # 1 predicted three times, 2 once -> 1 is hotter (higher LFU count).
    for _ in range(3):
        policy.plan_loads(0, predicted=[1], lrc_resident=set())
    policy.plan_loads(0, predicted=[2], lrc_resident=set())
    assert policy.resident_experts(0) == {1, 2}
    # Now a cold expert 3 arrives; must evict the colder of {1,2} == 2.
    plan = policy.plan_loads(0, predicted=[3], lrc_resident=set())
    assert len(plan) == 1
    assert plan[0].evicted_expert == 2
    assert policy.resident_experts(0) == {1, 3}


def test_protected_set_is_never_evicted():
    policy = _make(capacity=2)
    policy.plan_loads(0, predicted=[1, 2], lrc_resident=set())
    # Predict 1,2,3 together: 1 and 2 are protected (in this prediction), so
    # there is no evictable slot for 3.
    plan = policy.plan_loads(0, predicted=[1, 2, 3], lrc_resident=set())
    assert plan == []
    assert policy.resident_experts(0) == {1, 2}


def test_loading_set_is_not_evicted_when_alternatives_exist():
    policy = _make(capacity=2)
    policy.plan_loads(0, predicted=[1], lrc_resident=set())
    policy.plan_loads(0, predicted=[2], lrc_resident=set())
    # 1 is in flight; predicting only 3 (so neither 1 nor 2 protected) must
    # evict 2, not the in-flight 1.
    plan = policy.plan_loads(0, predicted=[3], lrc_resident=set(), loading={1})
    assert plan[0].evicted_expert == 2


def test_on_consumed_frees_slot():
    policy = _make(capacity=2)
    policy.plan_loads(0, predicted=[1, 2], lrc_resident=set())
    policy.on_consumed(0, 1)  # 1 promoted into LRC
    assert not policy.is_resident(0, 1)
    # Freed slot is reusable without eviction.
    plan = policy.plan_loads(0, predicted=[3], lrc_resident=set())
    assert plan[0].evicted_expert is None
    assert policy.resident_experts(0) == {2, 3}


def test_rollback_restores_prior_state():
    policy = _make(capacity=1)
    policy.plan_loads(0, predicted=[1], lrc_resident=set())
    plan = policy.plan_loads(0, predicted=[2], lrc_resident=set())  # evicts 1
    assert plan[0].evicted_expert == 1
    policy.rollback_load(0, plan[0])  # engine failed to issue
    assert policy.resident_experts(0) == {1}


def test_layers_are_independent():
    policy = _make(capacity=2, num_layers=2)
    policy.plan_loads(0, predicted=[1, 2], lrc_resident=set())
    policy.plan_loads(1, predicted=[5], lrc_resident=set())
    assert policy.resident_experts(0) == {1, 2}
    assert policy.resident_experts(1) == {5}
