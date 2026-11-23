# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU staging preserves portable Adam state and bounds its GPU expansion."""

import gc

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class StagingModel(torch.nn.Module):
    """Small parameters split across real distributed-optimizer buffers."""

    def __init__(self, count=8, size=4096) -> None:
        super().__init__()
        self.config = TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=1, bf16=True
        )
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(size, device="cuda", dtype=torch.bfloat16))
                for _ in range(count)
            ]
        )

    def sharded_state_dict(self):
        """Describe replicated model parameters, retaining parameter identity."""
        return {
            name: ShardedTensor.from_rank_offsets(name, param, replica_id=(0, 0, Utils.rank))
            for name, param in self.named_parameters()
        }


def make_optimizer(stage, dtype=torch.bfloat16, remainders=True, master_dtype=torch.float32, **kw):
    """Construct the normal Core DDP and optimizer stack."""
    torch.manual_seed(123)
    module = StagingModel(**kw)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    model = DistributedDataParallel(
        module.config,
        DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
        module,
        pg_collection=groups,
    )
    config = OptimizerConfig(
        lr=0.001,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        stage_precision_aware_optimizer_state_on_cpu=stage,
        store_param_remainders=remainders,
        main_params_dtype=master_dtype,
        exp_avg_dtype=dtype,
        exp_avg_sq_dtype=dtype,
        clip_grad=0.0,
    )
    return model, get_megatron_optimizer(config, [model], pg_collection=groups)


def train_step(model, optimizer, iteration):
    """Exercise reduction, Adam and all-gather with changing deterministic gradients."""
    model.zero_grad_buffer()
    optimizer.zero_grad()
    for index, param in enumerate(model.module.parameters()):
        positions = torch.arange(param.numel(), device=param.device, dtype=torch.float32)
        gradient = (positions.remainder(29) - 14).view_as(param.main_grad)
        param.main_grad.copy_(
            gradient * (index + 1) * (iteration + 1) * 0.0001 + (iteration % 3 - 1) * 0.00001
        )
    model.finish_grad_sync()
    optimizer.step()


def snapshot(model, optimizer):
    """Capture values through TE's normal API, independently of Core's staged exporter."""
    result = {name: param.detach().cpu().clone() for name, param in model.module.named_parameters()}
    for group_index, group in enumerate(optimizer.optimizer.param_groups):
        result[f"step_{group_index}"] = group.get("step")
        for param_index, param in enumerate(group["params"]):
            for name in optimizer.optimizer.state[param]:
                key = f"{group_index}.{param_index}.{name}"
                result[key] = optimizer.optimizer.get_unscaled_state(param, name).cpu().clone()
    return result


def checkpoint_state(model, optimizer, layout, loading=False):
    """Use the same format and metadata regardless of the staging setting."""
    model_state = model.module.sharded_state_dict()
    return {
        "model": model_state,
        "optimizer": optimizer.sharded_state_dict(
            model_state,
            is_loading=loading,
            metadata={"distrib_optim_sharding_type": layout, "checkpoint_version": 3.1},
        ),
    }


@pytest.fixture
def distributed_optimizer_groups():
    """Use the available ranks as one data-parallel group."""
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("stage_save", [False, True])
@pytest.mark.parametrize(
    "layout,dtype,remainders,master_dtype",
    [
        ("dp_reshardable", torch.bfloat16, True, torch.float32),
        ("dp_reshardable", torch.float16, False, torch.float16),
        ("fully_reshardable", torch.bfloat16, False, torch.float32),
    ],
)
def test_checkpoint_cpu_staging_roundtrip(
    distributed_optimizer_groups,
    tmp_path_dist_ckpt,
    stage_save,
    layout,
    dtype,
    remainders,
    master_dtype,
):
    """Existing and staged writers interoperate with both readers and subsequent updates."""
    model, optimizer = make_optimizer(stage_save, dtype, remainders, master_dtype)
    for iteration in range(3):
        train_step(model, optimizer, iteration)
    original = snapshot(model, optimizer)
    state = checkpoint_state(model, optimizer, layout)
    if stage_save:
        assert all(
            value.data.device.type == "cpu"
            for value in nested_values(state["optimizer"]["param_state"])
            if isinstance(value, ShardedTensor)
        )
    with TempNamedDir(tmp_path_dist_ckpt / "cpu_staging") as directory:
        save(state, directory)
        torch.testing.assert_close(snapshot(model, optimizer), original, rtol=0, atol=0)
        for iteration in range(3, 6):
            train_step(model, optimizer, iteration)
        uninterrupted = snapshot(model, optimizer)
        resumed = []
        for stage_load in (False, True):
            restored_model, restored_optimizer = make_optimizer(
                stage_load, dtype, remainders, master_dtype
            )
            template = checkpoint_state(restored_model, restored_optimizer, layout, loading=True)
            strategy = None
            if layout == "fully_reshardable":
                strategy = FullyParallelLoadStrategyWrapper(TorchDistLoadShardedStrategy())
            loaded = load(template, directory, sharded_strategy=strategy)
            restored_model.module.load_state_dict(loaded["model"])
            restored_optimizer.load_state_dict(loaded["optimizer"])
            # FP16 scaling can round on restore even without staging; the staged reader must
            # match the baseline reader exactly, including that existing conversion behavior.
            tolerance = (
                {"rtol": 1e-3, "atol": 1e-8} if dtype == torch.float16 else {"rtol": 0, "atol": 0}
            )
            torch.testing.assert_close(
                snapshot(restored_model, restored_optimizer), original, **tolerance
            )
            for iteration in range(3, 6):
                train_step(restored_model, restored_optimizer, iteration)
            resumed.append(snapshot(restored_model, restored_optimizer))
            torch.testing.assert_close(resumed[-1], uninterrupted, **tolerance)
        torch.testing.assert_close(resumed[0], resumed[1], rtol=0, atol=0)


def test_cpu_staging_async_snapshot(distributed_optimizer_groups, tmp_path_dist_ckpt):
    """Training may continue while an async writer consumes the staged snapshot."""
    async_core = pytest.importorskip("nvidia_resiliency_ext.checkpointing.async_ckpt.core")

    model, optimizer = make_optimizer(True)
    train_step(model, optimizer, 0)
    original = snapshot(model, optimizer)
    with TempNamedDir(tmp_path_dist_ckpt / "async_cpu_staging") as directory:
        queue = async_core.AsyncCallsQueue(persistent=False)
        try:
            request = save(
                checkpoint_state(model, optimizer, "dp_reshardable"),
                directory,
                async_sharded_save=True,
            )
            queue.schedule_async_request(request)
            train_step(model, optimizer, 1)
            queue.maybe_finalize_async_calls(blocking=True)
            loaded = load(checkpoint_state(model, optimizer, "dp_reshardable", True), directory)
            model.module.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            torch.testing.assert_close(snapshot(model, optimizer), original, rtol=0, atol=0)
        finally:
            queue.close()


def test_cpu_staging_bounds_gpu_expansion(distributed_optimizer_groups):
    """Many compressed states must not retain a full FP32 expansion on the GPU."""
    model, optimizer = make_optimizer(False, count=32, size=1024 * 1024)
    train_step(model, optimizer, 0)
    peaks = []
    for stage in (False, True):
        optimizer.config.stage_precision_aware_optimizer_state_on_cpu = stage
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        state = checkpoint_state(model, optimizer, "dp_reshardable")
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
        del state
    assert peaks[1] < peaks[0] // 4, peaks


def test_cpu_staging_fp32_is_noop(distributed_optimizer_groups):
    """An FP32 optimizer with no expanded state retains its original devices and aliases."""
    model, optimizer = make_optimizer(True, dtype=torch.float32, remainders=False)
    train_step(model, optimizer, 0)
    state = optimizer.chained_optimizers[0].get_parameter_state_dp_reshardable()
    tensors = [value for value in nested_values(state) if isinstance(value, torch.Tensor)]
    assert tensors and all(tensor.is_cuda for tensor in tensors)
    live_pointers = {
        tensor.data_ptr()
        for param_state in optimizer.optimizer.state.values()
        for tensor in param_state.values()
    }
    assert all(tensor.data_ptr() in live_pointers for tensor in tensors)


@pytest.mark.parametrize(
    "override",
    [
        {"use_precision_aware_optimizer": False},
        {"use_distributed_optimizer": False},
        {"optimizer_cpu_offload": True},
        {"use_layer_wise_distributed_optimizer": True},
        {"fp8_recipe": "mxfp8"},
    ],
)
def test_cpu_staging_rejects_unsupported_optimizer(override):
    """Unsupported optimizers fail before allocating models or state."""
    config = OptimizerConfig(
        **{
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": True,
            "stage_precision_aware_optimizer_state_on_cpu": True,
            **override,
        }
    )
    with pytest.raises(ValueError, match="CPU checkpoint staging requires"):
        config.validate()


@pytest.mark.parametrize(
    "format,fsdp,torch_fsdp",
    [("torch", False, False), ("torch_dist", True, False), ("torch_dist", False, True)],
)
def test_cpu_staging_rejects_unsupported_checkpoint(format, fsdp, torch_fsdp):
    """Training config validation excludes formats and wrappers outside this option's scope."""
    config = OptimizerConfig(stage_precision_aware_optimizer_state_on_cpu=True)
    with pytest.raises(ValueError, match="CPU checkpoint staging requires"):
        config.validate_run(
            ddp_config=DistributedDataParallelConfig(use_megatron_fsdp=fsdp),
            use_torch_fsdp2=torch_fsdp,
            ckpt_format=format,
        )
