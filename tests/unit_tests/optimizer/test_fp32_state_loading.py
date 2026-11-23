# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP32 distributed-optimizer restores reuse moments and preserve the next Adam step."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from tests.unit_tests.test_utilities import Utils

FusedAdam = pytest.importorskip("transformer_engine.pytorch.optimizers").FusedAdam


def _wrapper(inner):
    wrapper = DistributedOptimizer.__new__(DistributedOptimizer)
    wrapper.optimizer = inner
    wrapper.config = OptimizerConfig()
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    return wrapper


@pytest.fixture
def adam_state():
    torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
    param = torch.nn.Parameter(torch.ones(4096, device="cuda"))
    optimizer = FusedAdam([param], lr=0.01)
    param.grad = torch.full_like(param, 0.5)
    optimizer.step()
    return param, copy.deepcopy(optimizer.state_dict())


def test_fp32_restore_reuses_moments_and_preserves_next_step(adam_state):
    param, state = adam_state
    restored_param = torch.nn.Parameter(param.detach().clone())
    reference_param = torch.nn.Parameter(param.detach().clone())
    restored = FusedAdam([restored_param], lr=0.01)
    reference = FusedAdam([reference_param], lr=0.01)
    reference.load_state_dict(copy.deepcopy(state))
    _wrapper(restored)._load_inner_optimizer_state_dict(state)
    for key in ("exp_avg", "exp_avg_sq"):
        assert restored.state[restored_param][key].data_ptr() == state["state"][0][key].data_ptr()
    for optimizer, weight in ((restored, restored_param), (reference, reference_param)):
        weight.grad = torch.full_like(weight, 0.25)
        optimizer.step()
    torch.testing.assert_close(restored_param, reference_param, rtol=0, atol=0)
    assert restored.param_groups[0]["step"] == reference.param_groups[0]["step"]
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            restored.state[restored_param][key],
            reference.state[reference_param][key],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "case",
    [
        "precision_aware",
        "offload",
        "fsdp",
        "master",
        "remainder",
        "bf16_moments",
        "hook",
        "extra_state",
        "empty",
    ],
)
def test_other_state_formats_use_te_loader(monkeypatch, adam_state, case):
    param, state = adam_state
    optimizer = FusedAdam([param], lr=0.01)
    wrapper = _wrapper(optimizer)
    if case == "precision_aware":
        wrapper.config.use_precision_aware_optimizer = True
    elif case == "offload":
        wrapper.config.optimizer_cpu_offload = True
    elif case == "fsdp":
        wrapper.ddp_config.use_megatron_fsdp = True
    elif case == "master":
        optimizer.master_weights = True
    elif case == "remainder":
        optimizer.store_param_remainders = True
    elif case == "bf16_moments":
        state["state"][0]["exp_avg"] = state["state"][0]["exp_avg"].to(torch.bfloat16)
    elif case == "hook":
        optimizer.register_load_state_dict_pre_hook(lambda *args: None)
    elif case == "extra_state":
        state["state"][0]["future_state"] = torch.zeros_like(param)
    elif case == "empty":
        state["state"] = {}
    calls = Mock()

    def load(instance, value):
        calls(instance, value)

    monkeypatch.setattr(FusedAdam, "load_state_dict", load)
    wrapper._load_inner_optimizer_state_dict(state)
    calls.assert_called_once_with(optimizer, state)


def test_custom_instance_loader_is_respected(adam_state):
    param, state = adam_state
    optimizer = FusedAdam([param], lr=0.01)
    optimizer.load_state_dict = Mock()
    _wrapper(optimizer)._load_inner_optimizer_state_dict(state)
    optimizer.load_state_dict.assert_called_once_with(state)
