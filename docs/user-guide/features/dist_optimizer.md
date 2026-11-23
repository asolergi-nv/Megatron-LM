{/*
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}

# Distributed Optimizer

The distributed optimizer saves memory by sharding optimizer state across data parallel ranks instead of replicating it on every rank, as described in the [ZeRO paper](https://arxiv.org/abs/1910.02054).

Theoretical memory savings depend on the data types of the model parameters (`param_dtype`) and of the main gradients accumulated across data-parallel replicas (`grad_dtype`). Optimizer steps always use `fp32` main parameters. In the current implementation, the theoretical number of bytes per parameter is as follows (where *d* is the data parallel size):

|        | Non-distributed optim | Distributed optim |
| ------ | ------ | ------ |
| `fp16` parameters, `fp16` gradients | 20 | 4 + 16/d |
| `bf16` parameters, `fp32` gradients    | 18 | 6 + 12/d |
| `fp32` parameters, `fp32` gradients       | 16 | 8 + 8/d  |

This distributed optimizer uses contiguous buffers for parameters and main gradients. Model gradients copy into the main gradients as soon as they finish computing.

Models may retain selected parameters in `fp32` alongside `fp16` or `bf16` weights,
as mHC does. Optimizer checkpoint lookups follow the main-parameter group order
(native `fp32` shards, then low-precision master shards), independently of the
order of the parameter and gradient buffers.
DP-reshardable checkpoints include padding inside the unpadded bucket extent;
padding added only to divide the buffer across DP ranks is excluded.

Checkpoints written before the mixed-dtype index fix in PR #7336 may already
associate optimizer state with the wrong parameter in a mixed-dtype group.
The corrected mapping does not repair affected checkpoint contents; such
checkpoints need independent validation before resuming training.

The following figures show the sharding scheme and the main steps of the parameter update.

## CPU staging for precision-aware checkpoints

Compressed Adam state can fit during training but exceed GPU memory when expanded for
checkpointing. For Transformer Engine's precision-aware distributed Adam, enable
`OptimizerConfig.stage_precision_aware_optimizer_state_on_cpu` or the training flag
`--stage-precision-aware-optimizer-state-on-cpu` to move each expanded checkpoint tensor
to CPU immediately. Metadata-only state dictionaries skip expansion entirely. This
applies during saving and construction/restoration of loading state, including
already-initialized optimizers.

The option defaults to false. It preserves checkpoint keys, values, and dtypes, so
staged and unstaged writers/readers interoperate without a format-version change.
Training state and the Adam update remain on GPU; no Transformer Engine methods are
replaced. FP32-only state is left unchanged because it does not need expansion.

Use `torch_dist` with the distributed precision-aware optimizer. Optimizer CPU offload,
FSDP, and precision modes where Core rather than TE manages master weights are excluded.
Existing restrictions of each checkpoint layout still apply. In particular, this option
does not make the deprecated `fully_sharded_model_space` layout usable with the current
ShardedTensor API, or add remainder support to `fully_reshardable`.

This is a memory-capacity option: host memory must accommodate the unscaled local state
(and temporary copies during restore), and synchronous device-to-host transfers can
increase checkpoint latency. Fully reshardable checkpoints and fully parallel loading
can still allocate GPU communication buffers; staging does not eliminate those buffers.
For applications that previously wrapped TE state access for CPU staging, set the Core
optimizer option and remove that wrapper after adopting a Core version with this option.

To measure the trade-off on a local filesystem, run the following in separate fresh
processes, adding `--stage-on-cpu` for the second run and using different output paths:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tools/benchmark_optimizer_checkpoint.py \
  --count 64 --size 16777216 --remainders \
  --output /tmp/optimizer-baseline --checkpoint /tmp/optimizer-checkpoint-baseline
```

This example uses 1,073,741,824 synthetic BF16 parameters, FP32 gradients, BF16 moments,
and INT16 master-weight remainders. It exercises real Core DDP, TE Adam and synchronous
DP-reshardable checkpoint IO. Each rank writes JSON with per-phase times, peak CUDA
allocated/reserved bytes, and process RSS sampled every 5 ms. CUDA cache cleanup and
correctness checks are outside timed phases. The filesystem cache is not flushed.
Exact hashes cover model weights, unscaled optimizer state, step counters, and three
updates after restoring a fresh optimizer. This measures checkpoint behavior, not
end-to-end transformer throughput. Checkpoint files are removed after a successful run.

## Data Flow

![Diagram of gradient and parameter data flow through reduce-scatter, optimizer step, and all-gather across data parallel ranks](../../images/distrib_optimizer/data_flow.png)

## Sharding Scheme

![Diagram of how optimizer state shards across data parallel ranks](../../images/distrib_optimizer/sharding_scheme.png)

## Key Steps

**Note:** The following steps match the illustrations above. They assume `bf16` model weights, `bf16` model gradients from the backward pass, and `fp32` main gradients for optimizer steps. Optimizer steps use `fp32` main weights.

- Backward pass finishes (gradient buffer holds 16 `fp32` gradient elements).
- Call reduce-scatter on each DP rank.
- Each DP rank now has four elements within the gradient buffer that are fully reduced (remaining 12 elements are garbage).
  - DP rank 0 has gradient values for elements [0:4].
  - DP rank 1 has gradient values for elements [4:8].
  - DP rank 2 has gradient values for elements [8:12].
  - DP rank 3 has gradient values for elements [12:16].
- Optimizer.step().
- Each DP rank copies its four `fp32` main parameter elements into the corresponding `bf16` parameter buffer (each element is cast from `fp32` to `bf16`).
- Call all-gather on each DP rank.
- The parameter buffer now contains all 16 updated `bf16` model parameter elements. Parameters in PyTorch modules already point to the correct views in this buffer, so forward passes can start after the all-gather completes.
- At this point, you can zero the gradient buffer for the next iteration.
