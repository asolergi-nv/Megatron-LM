# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The dataset section is owned by PretrainConfigContainer: construction, lifecycle and upgrade."""

import copy
import functools
from argparse import ArgumentParser, Namespace
from dataclasses import MISSING, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.training import arguments
from megatron.training.argument_utils import (
    dataloader_config_from_args,
    gpt_dataset_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.config import (
    CheckpointConfig,
    DataloaderConfig,
    DatasetRuntimeContext,
    GPTDatasetConfig,
    GPTFIMDatasetConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
    PretrainConfigContainer,
)
from megatron.training.config.dataset_config import DATASET_CLI_EXCLUDE, LEGACY_ARG_MAP
from megatron.training.config.yaml_utils import safe_yaml_representers

DATA_ARGV = [
    "--seq-length", "64",
    "--data-path", "0.3", "prefix-a", "0.7", "prefix-b",
    "--split", "9,1,0",
    "--num-workers", "0",
    "--dataloader-type", "cyclic",
    "--no-data-sharding",
    "--reset-position-ids",
    "--reset-attention-mask",
    "--eod-mask-loss",
    "--dataloader-inter-document-masking",
    "--no-create-attention-mask-in-dataloader",
    "--data-cache-path", "/cache",
    "--dataloader-fast-cache-load",
    "--dataloader-defer-npy-index-mmap",
    "--no-mmap-bin-files",
    "--per-dataset-sequences-path", "/sequences.json",
    "--phase-transition-iterations", "20,10",
    "--sft-mock-dataset-config-json", "{}",
    "--varlen-mock-dataset-config-json", "{\"mode\": \"file\"}",
    "--varlen-sbhd-validation",
    "--allow-ambiguous-pad-tokens",
    "--mid-level-dataset-surplus", "0.1",
    "--num-dataset-builder-threads", "3",
    "--object-storage-cache-path", "/object-cache",
    "--fim-rate", "0.9",
    "--fim-spm-rate", "0.4",
    "--fim-split-sample", "<split>",
    "--fim-fragment-rate", "0.2",
    "--fim-no-prefix", "nofim",
    "--fim-prefix-token", "<p>",
    "--fim-middle-token", "<m>",
    "--fim-suffix-token", "<s>",
    "--fim-pad-token", "<x>",
    "--fim-eod-token", "<e>",
]  # fmt: skip


def _parse(argv):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser.parse_args(list(argv))


def _container_from_argv(monkeypatch, argv, **kwargs):
    from megatron.training import training

    monkeypatch.setattr(
        training, "get_megatron_optimizer_config", lambda args, **_: (OptimizerConfig(), None)
    )
    monkeypatch.setattr(
        training, "get_megatron_ddp_config", lambda args, **_: DistributedDataParallelConfig()
    )
    args = _parse(argv)
    return args, pretrain_cfg_container_from_args(args, **kwargs)


def _raw(cfg):
    return {f.name: copy.deepcopy(getattr(cfg, f.name)) for f in fields(cfg)}


def _same_declaration(left, right):
    # The serialized form is the normalized declaration, so a restored section compares equal to
    # the original after normalization (for example dataloader_type None -> "single").
    return left._normalized_fields() == right._normalized_fields()


def test_container_from_args_declares_every_data_argument(monkeypatch):
    args, cfg = _container_from_argv(monkeypatch, DATA_ARGV)
    dataset = cfg.dataset
    assert type(dataset) is GPTDatasetConfig
    renames = {name: spec["dest"] for name, spec in LEGACY_ARG_MAP.items() if "dest" in spec}
    for item in fields(GPTDatasetConfig):
        if not item.init or item.name in DATASET_CLI_EXCLUDE or item.name == "dataset_type":
            continue
        assert getattr(dataset, item.name) == getattr(
            args, renames.get(item.name, item.name)
        ), item.name
    assert dataset.seq_length == 64 and dataset.sequence_length is None
    assert dataset.data_path == ["0.3", "prefix-a", "0.7", "prefix-b"] and dataset.split == "9,1,0"
    assert (dataset.num_workers, dataset.dataloader_type, dataset.data_sharding) == (
        0,
        "cyclic",
        False,
    )
    assert (
        dataset.path_to_cache == "/cache"
        and dataset.fast_cache_load
        and dataset.defer_npy_index_mmap
    )
    assert dataset.create_attention_mask is False and dataset.inter_document_masking is True
    assert (
        dataset.mmap_bin_files is False and dataset.per_dataset_sequences_path == "/sequences.json"
    )
    assert dataset.phase_transition_iterations == "20,10"  # a raw namespace; validate_args sorts it
    assert dataset.fim_rate == 0.9 and dataset.fim_eod_token == "<e>"
    assert dataset.dataset_type == "gpt" and not dataset.mock_data
    # Runtime inputs are not declared from the CLI.
    assert dataset.tokenizer is None and dataset.random_seed is None
    assert dataset.blend is None and dataset.blend_per_split is None
    assert dataset.data_parallel_size is None and dataset.full_validation is None
    assert dataset.pin_memory and dataset.persistent_workers and dataset.drop_last
    loader = dataloader_config_from_args(args)
    assert loader == DataloaderConfig(dataloader_type="cyclic", num_workers=0, data_sharding=False)


@pytest.mark.parametrize(
    ("argv", "dataset_type"),
    [
        ([], "gpt"),
        (["--sft"], "sft"),
        (["--use-varlen-dataset"], "varlen"),
        (["--fim-data"], "fim"),
    ],
)
def test_alias_flags_select_the_dataset_family(argv, dataset_type):
    dataset = gpt_dataset_config_from_args(_parse(argv))
    assert dataset.dataset_type == dataset_type
    assert dataset == GPTDatasetConfig(dataset_type=dataset_type)


def test_cli_defaults_build_the_default_declaration():
    assert gpt_dataset_config_from_args(_parse([])) == GPTDatasetConfig()
    assert gpt_dataset_config_from_args(Namespace()) == GPTDatasetConfig()
    assert dataloader_config_from_args(Namespace()) == DataloaderConfig()


def test_container_keeps_a_caller_supplied_dataset_section(monkeypatch):
    preset = GPTFIMDatasetConfig(seq_length=8)
    _, cfg = _container_from_argv(monkeypatch, ["--seq-length", "64"], dataset_cfg=preset)
    assert cfg.dataset is preset
    loader_only = DataloaderConfig(num_workers=4)
    _, cfg = _container_from_argv(monkeypatch, [], dataset_cfg=loader_only)
    assert cfg.dataset is loader_only
    cfg.validate()
    cfg.finalize()
    assert cfg.dataset.dataloader_type == "single"


def test_run_config_fixture_carries_a_dataset_section(run_config):
    assert isinstance(run_config.dataset, GPTDatasetConfig)
    assert run_config.dataset.mock_data and run_config.dataset.seq_length == 16


def test_container_validation_is_read_only_and_finalization_normalizes_once(
    monkeypatch, run_config
):
    run_config.dataset = GPTDatasetConfig(
        sequence_length=32,
        data_path="prefix-a prefix-b",
        num_workers=0,
        inter_document_masking=True,
    )
    before_dict = copy.deepcopy(run_config.to_dict())
    before_raw = _raw(run_config.dataset)
    for _ in range(2):
        run_config.validate()
        assert run_config.to_dict() == before_dict
        assert _raw(run_config.dataset) == before_raw
        assert not run_config._finalized
    finalize = Mock(wraps=run_config.dataset.finalize)
    monkeypatch.setattr(run_config.dataset, "finalize", finalize)
    run_config.finalize()
    finalize.assert_called_once_with()
    dataset = run_config.dataset
    assert dataset.seq_length == 32 and dataset.sequence_length is None
    assert dataset.data_path == ["prefix-a", "prefix-b"]
    assert dataset.dataloader_type == "single"
    # Field-dependent defaults are resolved for the runtime object, not written into the section.
    assert dataset.persistent_workers is True
    assert dataset.split is None and dataset.create_attention_mask is True
    resolved = dataset._resolved_fields()
    assert resolved["persistent_workers"] is False
    assert resolved["split"] == "969, 30, 1" and resolved["create_attention_mask"] is False
    assert run_config.to_dict() == before_dict  # serialization does not depend on the lifecycle
    run_config.validate()
    run_config.finalize()
    finalize.assert_called_once_with()
    assert run_config.to_dict() == before_dict


def test_invalid_dataset_declaration_blocks_finalization(run_config):
    run_config.dataset = GPTDatasetConfig(seq_length=8, sequence_length=16, mock_data=True)
    before = _raw(run_config.dataset)
    with pytest.raises(ValueError, match="disagree"):
        run_config.finalize()
    assert not run_config._finalized
    assert _raw(run_config.dataset) == before
    # The invalid section is not serialized into a different, valid-looking one.
    with pytest.raises(ValueError, match="disagree"):
        run_config.to_dict()
    run_config.dataset.sequence_length = None
    run_config.finalize()
    assert run_config._finalized and run_config.dataset.seq_length == 8
    assert run_config.to_dict()["dataset"]["seq_length"] == 8


def test_serialization_round_trip_keeps_the_dataset_section(run_config, tmp_path):
    run_config.dataset = GPTDatasetConfig(
        seq_length=32, data_path=["prefix-a"], split="9,1,0", dataset_type="fim", fim_rate=0.9
    )
    serialized = run_config.to_dict()
    assert serialized["dataset"]["_target_"].endswith("GPTDatasetConfig")
    assert "tokenizer" not in serialized["dataset"] and "mock" not in serialized["dataset"]
    restored = PretrainConfigContainer.from_dict(serialized)
    assert _same_declaration(restored.dataset, run_config.dataset)
    assert restored.dataset != run_config.dataset  # the raw spelling was normalized on output
    assert restored.to_dict() == serialized
    path = tmp_path / "run_config.yaml"
    run_config.to_yaml(str(path))
    from_file = PretrainConfigContainer.from_yaml(str(path))
    assert _same_declaration(from_file.dataset, run_config.dataset)
    assert from_file.to_dict() == serialized


def test_pre_dataset_run_config_is_upgraded_from_legacy_args(run_config, tmp_path):
    serialized = run_config.to_dict()
    legacy = copy.deepcopy(serialized)
    legacy.pop("dataset")
    with pytest.raises(ValueError, match="dataset"):
        PretrainConfigContainer.from_dict(legacy)
    args = _parse(["--seq-length", "24", "--mock-data", "--sft"])
    upgraded = PretrainConfigContainer.from_dict(legacy, legacy_args=args)
    assert _same_declaration(
        upgraded.dataset, GPTDatasetConfig(seq_length=24, mock_data=True, dataset_type="sft")
    )
    assert upgraded.train == run_config.train
    # A file that already has the section ignores legacy_args.
    assert _same_declaration(
        PretrainConfigContainer.from_dict(serialized, legacy_args=args).dataset, run_config.dataset
    )
    path = tmp_path / "run_config.yaml"
    with safe_yaml_representers():
        path.write_text(yaml.safe_dump(legacy, default_flow_style=False))
    with pytest.raises(ValueError, match="dataset"):
        PretrainConfigContainer.from_yaml(str(path))
    from_file = PretrainConfigContainer.from_yaml(str(path), legacy_args=args)
    assert _same_declaration(from_file.dataset, upgraded.dataset)
    assert "dataset" not in legacy  # the caller's dictionary is not modified


def test_inference_container_is_unaffected():
    cfg = InferenceConfigContainer(
        model=None, checkpoint=CheckpointConfig(), inference=InferenceSetupConfig()
    )
    assert not any(f.name == "dataset" for f in fields(cfg))
    serialized = cfg.to_dict()
    assert InferenceConfigContainer.from_dict(serialized).to_dict() == serialized


def test_dataset_runtime_context_reads_the_other_sections(run_config):
    run_config.rng.seed = 77
    run_config.validation.full_validation = True
    run_config.validation.multiple_validation_sets = False
    context = run_config.dataset_runtime_context("tok", data_parallel_size=4)
    assert context.tokenizer == "tok" and context.random_seed == 77
    assert context.data_parallel_size == 4
    assert context.context_parallel_size is None and context.sequence_parallel_size is None
    assert context.hybrid_context_parallel is None
    assert context.full_validation is True and context.multiple_validation_sets is False
    run_config.model = SimpleNamespace(
        transformer=SimpleNamespace(
            context_parallel_size=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            hybrid_context_parallel=True,
        )
    )
    context = run_config.dataset_runtime_context("tok", data_parallel_size=4)
    assert (context.context_parallel_size, context.sequence_parallel_size) == (2, 2)
    assert context.hybrid_context_parallel is True
    overridden = run_config.dataset_runtime_context(
        "tok", data_parallel_size=4, random_seed=5, full_validation=False, sequence_parallel_size=0
    )
    assert overridden.random_seed == 5 and overridden.full_validation is False
    assert overridden.sequence_parallel_size == 0


def test_container_from_args_matches_legacy_runtime_inputs(monkeypatch):
    _, cfg = _container_from_argv(monkeypatch, ["--seed", "99", "--full-validation"])
    context = cfg.dataset_runtime_context("tok", data_parallel_size=1)
    assert context.random_seed == 99 and context.full_validation is True


# The data builders hand the container to providers that declare it (and only to those).


def _recording_providers():
    calls = []

    def plain(train_valid_test_num_samples):
        calls.append(("plain", train_valid_test_num_samples, {}))
        return (1, 2, 3)

    def declared(train_valid_test_num_samples, vp_stage=None, *, dataset_config=None, context=None):
        calls.append(
            (
                "declared",
                train_valid_test_num_samples,
                dict(vp_stage=vp_stage, dataset_config=dataset_config, context=context),
            )
        )
        return (1, 2, 3)

    def kwargs_only(train_valid_test_num_samples, **kwargs):
        calls.append(("kwargs", train_valid_test_num_samples, kwargs))
        return (1, 2, 3)

    return calls, plain, declared, kwargs_only


def test_declared_provider_keywords():
    from megatron.training.training import _declared_provider_keywords as declared_keywords

    _, plain, declared, kwargs_only = _recording_providers()
    names = ("dataset_config", "context")
    assert declared_keywords(plain, names) == ()
    assert declared_keywords(kwargs_only, names) == ()
    assert declared_keywords(declared, names) == names
    assert declared_keywords(functools.partial(declared, vp_stage=1), names) == names
    assert declared_keywords(functools.partial(declared, dataset_config="bound"), names) == (
        "context",
    )

    def positional_or_keyword(train_valid_test_num_samples, dataset_config=None):
        return None

    def context_only(train_valid_test_num_samples, *, context=None):
        return None

    def positional_only(train_valid_test_num_samples, dataset_config=None, /):
        return None

    assert declared_keywords(positional_or_keyword, names) == ("dataset_config",)
    assert declared_keywords(context_only, names) == ("context",)
    assert declared_keywords(positional_only, names) == ()
    assert declared_keywords(42, names) == ()


@pytest.fixture
def runtime_globals(monkeypatch, run_config):
    from megatron.training import global_vars, training

    tokenizer = object()
    args = SimpleNamespace(
        data_parallel_size=4,
        context_parallel_size=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        hybrid_context_parallel=False,
        full_validation=True,
        multiple_validation_sets=False,
    )
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", tokenizer)
    monkeypatch.setattr(training, "get_args", lambda: args)
    return tokenizer, args


def test_builders_pass_the_container_only_to_providers_that_declare_it(
    monkeypatch, run_config, runtime_globals
):
    from megatron.training import global_vars
    from megatron.training.training import build_train_valid_test_datasets

    tokenizer, _ = runtime_globals
    calls, plain, declared, kwargs_only = _recording_providers()
    counts = (10, 20, 30)
    for provider in (plain, kwargs_only, declared):
        assert build_train_valid_test_datasets(provider, counts, cfg_container=run_config) == (
            1,
            2,
            3,
        )
    assert calls[0] == ("plain", counts, {})
    assert calls[1] == ("kwargs", counts, {})
    _, received, kwargs = calls[2]
    assert received == counts and kwargs["vp_stage"] is None
    assert kwargs["dataset_config"] is run_config.dataset
    context = kwargs["context"]
    assert isinstance(context, DatasetRuntimeContext)
    assert context.tokenizer is tokenizer and context.random_seed == run_config.rng.seed
    assert (context.data_parallel_size, context.context_parallel_size) == (4, 2)
    assert context.sequence_parallel_size == 2 and context.hybrid_context_parallel is False
    assert (context.full_validation, context.multiple_validation_sets) == (True, False)
    # A partial keeps what it bound and still receives the rest.
    bound = functools.partial(declared, vp_stage=3, dataset_config="bound")
    build_train_valid_test_datasets(bound, counts, cfg_container=run_config)
    _, _, kwargs = calls[3]
    assert kwargs["vp_stage"] == 3 and kwargs["dataset_config"] == "bound"
    assert isinstance(kwargs["context"], DatasetRuntimeContext)
    # Without an explicit container the run config is used; without either, nothing is injected.
    build_train_valid_test_datasets(declared, counts)
    assert calls[4][2]["dataset_config"] is run_config.dataset
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    build_train_valid_test_datasets(declared, counts)
    assert calls[5][2] == dict(vp_stage=None, dataset_config=None, context=None)


def test_build_dataset_runtime_context_reads_the_legacy_parallel_inputs(
    run_config, runtime_globals
):
    from megatron.training.training import build_dataset_runtime_context

    tokenizer, args = runtime_globals
    run_config.rng.seed = 321
    context = build_dataset_runtime_context(run_config)
    assert context.tokenizer is tokenizer and context.random_seed == 321
    assert (context.data_parallel_size, context.sequence_parallel_size) == (4, 2)
    args.sequence_parallel = False
    assert build_dataset_runtime_context(run_config).sequence_parallel_size == 0
    assert build_dataset_runtime_context().random_seed == 321
    assert build_dataset_runtime_context(run_config, tokenizer="other").tokenizer == "other"
