# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Measure distributed Adam checkpoint memory/time and verify exact resumed updates.

Run with torch.distributed.run; compare fresh processes with and without
--stage-on-cpu. Synthetic BF16 parameters exercise real Core DDP, TE Adam and
torch_dist checkpoint IO. This is not a transformer throughput benchmark.
"""

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig


class Model(torch.nn.Module):
    """Replicated parameters with configurable optimizer-state volume."""

    def __init__(self, count, size) -> None:
        super().__init__()
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(size, device="cuda", dtype=torch.bfloat16))
                for _ in range(count)
            ]
        )
        with torch.no_grad():
            for param in self.weights:
                param.uniform_(-0.1, 0.1)

    def sharded_state_dict(self):
        """Preserve model parameter identity for distributed optimizer checkpoint mapping."""
        return {
            name: ShardedTensor.from_rank_offsets(name, param, replica_id=(0, 0, dist.get_rank()))
            for name, param in self.named_parameters()
        }


def build(args):
    """Create the real DDP and precision-aware Adam stack with reproducible weights."""
    torch.manual_seed(123)
    module = Model(args.count, args.size)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    ddp = DistributedDataParallel(
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
        clip_grad=0.0,
        stage_precision_aware_optimizer_state_on_cpu=args.stage_on_cpu,
        store_param_remainders=args.remainders,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    optimizer = get_megatron_optimizer(config, [ddp], pg_collection=groups)
    return ddp, optimizer


def step(ddp, optimizer, iteration):
    """Reduce gradients, update Adam state, and gather the updated parameters."""
    # Exercise real reduce-scatter, Adam update, and parameter all-gather.
    ddp.zero_grad_buffer()
    optimizer.zero_grad()
    for index, param in enumerate(ddp.module.parameters()):
        param.main_grad.fill_((index + 1) * (iteration + 1) * 0.00001)
    ddp.finish_grad_sync()
    optimizer.step()


def fingerprint(ddp, optimizer):
    """Hash model, unscaled optimizer state, and counters outside the timed regions."""
    digest = hashlib.sha256()

    def add(tensor):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())

    for param in ddp.module.parameters():
        add(param)
    for group in optimizer.optimizer.param_groups:
        digest.update(str(group.get("step")).encode())
        for param in group["params"]:
            for key in sorted(optimizer.optimizer.state[param]):
                add(optimizer.optimizer.get_unscaled_state(param, key))
    return digest.hexdigest()


def measure(label, fn):
    """Measure one synchronized phase, GPU allocation peaks, and sampled process RSS."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_base = process.memory_info().rss
    peak_rss = [rss_base]
    done = threading.Event()

    def sample():
        while not done.wait(0.005):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample)
    monitor.start()
    start = time.perf_counter()
    try:
        value = fn()
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        elapsed = time.perf_counter() - start
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        done.set()
        monitor.join()
    result = dict(
        phase=label,
        seconds=elapsed,
        gpu_base_bytes=base,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        gpu_extra_bytes=torch.cuda.max_memory_allocated() - base,
        gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        rss_base_bytes=rss_base,
        rss_peak_bytes=peak_rss[0],
    )
    sys.stdout.write(json.dumps({"measurement": result, "rank": dist.get_rank()}) + "\n")
    sys.stdout.flush()
    return value, result


def main():
    """Save, restore a fresh optimizer, and compare three resumed steps bit for bit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-on-cpu", action="store_true")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--size", type=int, default=1048576)
    parser.add_argument("--remainders", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    checkpoint = Path(args.checkpoint)
    if dist.get_rank() == 0:
        checkpoint.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    ddp, optimizer = build(args)
    for iteration in range(3):
        step(ddp, optimizer, iteration)
    before = fingerprint(ddp, optimizer)
    metadata = {"distrib_optim_sharding_type": "dp_reshardable", "checkpoint_version": 3.1}

    def state_dict(loading=False):
        model_state = ddp.module.sharded_state_dict()
        optim_state = optimizer.sharded_state_dict(
            model_state, is_loading=loading, metadata=metadata
        )
        return {"model": model_state, "optimizer": optim_state}

    measurements = []
    state, result = measure("construct_save", state_dict)
    measurements.append(result)
    _, result = measure("serialize", lambda: save(state, checkpoint))
    measurements.append(result)
    del state
    assert fingerprint(ddp, optimizer) == before, "Saving mutated training state"
    for iteration in range(3, 6):
        step(ddp, optimizer, iteration)
    reference = fingerprint(ddp, optimizer)
    # Drop the original model and buffers before timing a genuinely fresh restore.
    del ddp, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    ddp, optimizer = build(args)
    template, result = measure("construct_load", lambda: state_dict(True))
    measurements.append(result)
    loaded, result = measure("deserialize", lambda: load(template, checkpoint))
    measurements.append(result)

    def restore():
        ddp.module.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])

    _, result = measure("restore", restore)
    measurements.append(result)
    del template, loaded
    restored = fingerprint(ddp, optimizer)
    assert restored == before, ("Restore differs", restored, before)
    for iteration in range(3, 6):
        step(ddp, optimizer, iteration)
    resumed = fingerprint(ddp, optimizer)
    assert resumed == reference, ("Resumed updates differ", resumed, reference)
    output = dict(
        arguments=vars(args),
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        saved_sha256=before,
        restored_sha256=restored,
        uninterrupted_sha256=reference,
        resumed_sha256=resumed,
        measurements=measurements,
    )
    Path(f"{args.output}.rank{dist.get_rank()}.json").write_text(json.dumps(output, indent=2))
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(checkpoint)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
