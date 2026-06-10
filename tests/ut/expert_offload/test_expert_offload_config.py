"""Validation tests for the prefetch keys on ExpertOffloadConfig.

These import ExpertOffloadConfig directly (no full VllmConfig), so they run in
CI where vllm is importable.
"""

import pytest

from vllm_ascend.ascend_config import ExpertOffloadConfig


def test_prefetch_defaults_off():
    cfg = ExpertOffloadConfig({})
    assert cfg.prefetch_enabled is False
    assert cfg.num_prefetch_experts == 0
    assert cfg.prefetch_predictor == "recent_union"
    assert cfg.prefetch_horizon == 0  # 0 = prefetch all remaining layers
    assert cfg.prefetch_history_window == 8
    assert cfg.prefetch_eamc_capacity == 256


def test_valid_eam_prefetch_config():
    cfg = ExpertOffloadConfig({
        "expert_offload": True,
        "num_device_experts": 16,
        "cache_policy_enabled": True,
        "prefetch_enabled": True,
        "num_prefetch_experts": 8,
        "prefetch_predictor": "eam",
        "prefetch_horizon": 3,
    })
    assert cfg.prefetch_enabled is True
    assert cfg.prefetch_predictor == "eam"
    assert cfg.num_prefetch_experts == 8


def test_valid_all_layers_prefetch_config():
    # horizon == 0 means "prefetch every remaining layer this step".
    cfg = ExpertOffloadConfig({
        "expert_offload": True,
        "num_device_experts": 16,
        "cache_policy_enabled": True,
        "prefetch_enabled": True,
        "num_prefetch_experts": 8,
        "prefetch_horizon": 0,
    })
    assert cfg.prefetch_horizon == 0


def test_prefetch_enabled_requires_expert_offload():
    with pytest.raises(ValueError):
        ExpertOffloadConfig({
            "expert_offload": False,
            "prefetch_enabled": True,
            "num_prefetch_experts": 8,
        })


def test_prefetch_enabled_requires_positive_cache_size():
    with pytest.raises(ValueError):
        ExpertOffloadConfig({
            "expert_offload": True,
            "cache_policy_enabled": True,
            "prefetch_enabled": True,
            "num_prefetch_experts": 0,
        })


def test_prefetch_enabled_requires_cache_policy():
    # The unified-cache prefetcher reuses the LRC policy for slot eviction.
    with pytest.raises(ValueError):
        ExpertOffloadConfig({
            "expert_offload": True,
            "cache_policy_enabled": False,
            "prefetch_enabled": True,
            "num_prefetch_experts": 8,
        })


def test_invalid_predictor_rejected():
    with pytest.raises(ValueError):
        ExpertOffloadConfig({"prefetch_predictor": "bogus"})


def test_negative_horizon_rejected():
    # 0 now means "all remaining layers"; only negatives are invalid.
    with pytest.raises(ValueError):
        ExpertOffloadConfig({"prefetch_horizon": -1})


def test_negative_eamc_capacity_rejected():
    with pytest.raises(ValueError):
        ExpertOffloadConfig({"prefetch_eamc_capacity": 0})
