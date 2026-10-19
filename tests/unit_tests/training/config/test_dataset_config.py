# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Declarative GPT dataset / dataloader configuration: lifecycle, conversion and serialization."""

import copy
import json
from argparse import ArgumentParser, Namespace
from dataclasses import MISSING, fields

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import arguments
from megatron.training.config import (
    DataloaderConfig,
    DatasetRuntimeContext,
    GPTDatasetConfig,
    GPTFIMDatasetConfig,
    PretrainConfigContainer,
)
from megatron.training.config import dataset_config as dataset_config_module
from megatron.training.config.dataset_config import (
    DATASET_CLI_EXCLUDE,
    LEGACY_ARG_MAP,
    LEGACY_SPLIT_DEFAULT,
    dataset_type_from_args,
    project_dataset_type_into_args,
)
from megatron.training.config.instantiate_utils import instantiate
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig as RuntimeGPTFIMDatasetConfig

# Keywords the hybrid-flex factory never forwarded (its runtime policy in PR-1).
FLEX_OMITTED = frozenset(
    {
        "multiple_validation_sets",
        "full_validation",
        "allow_ambiguous_pad_tokens",
        "fast_cache_load",
        "defer_npy_index_mmap",
        "sequences_per_dataset",
        "context_parallel_size",
        "data_parallel_size",
        "sequence_parallel_size",
        "hybrid_context_parallel",
        "sft_mock_dataset_config_json",
        "varlen_mock_dataset_config_json",
        "varlen_sbhd_validation",
    }
)


def _null_tokenizer(vocab_size=2048):
    return build_tokenizer(
        Namespace(
            vocab_size=vocab_size,
            tokenizer_type="NullTokenizer",
            padded_vocab_size=None,
            rank=0,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            pad_vocab_size=True,
        )
    )


@pytest.fixture(scope="module")
def tokenizer():
    return _null_tokenizer()


def _raw(cfg):
    return {f.name: copy.deepcopy(getattr(cfg, f.name)) for f in fields(cfg)}


def _core_fields(core):
    return {f.name: getattr(core, f.name) for f in fields(core) if f.name != "tokenizer"}


def _context(tokenizer, **overrides):
    overrides.setdefault("random_seed", 1234)
    return DatasetRuntimeContext(tokenizer=tokenizer, **overrides)


def _legacy_args(argv=()):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser.parse_args(list(argv))


def test_cli_defaults_match_declaration_defaults():
    args = _legacy_args()
    for item in fields(GPTDatasetConfig):
        if not item.init or item.name in DATASET_CLI_EXCLUDE or item.name == "dataset_type":
            continue
        dest = LEGACY_ARG_MAP.get(item.name, {}).get("dest", item.name)
        default = item.default if item.default is not MISSING else item.default_factory()
        assert hasattr(args, dest), item.name
        assert getattr(args, dest) == default, item.name


def _spelled_declaration(**overrides):
    kwargs = dict(
        sequence_length=32,
        data_path="prefix-a prefix-b",
        phase_transition_iterations="20,10",
        num_workers=0,
        inter_document_masking=True,
    )
    kwargs.update(overrides)
    return GPTDatasetConfig(**kwargs)


def test_validate_is_read_only_and_repeatable():
    cfg = _spelled_declaration()
    before = _raw(cfg)
    serialized = cfg.to_cfg_dict()
    for _ in range(2):
        cfg.validate()
        assert _raw(cfg) == before
        assert cfg.to_cfg_dict() == serialized
    normalized = cfg._normalized_fields()
    assert normalized["seq_length"] == 32 and normalized["sequence_length"] is None
    assert normalized["data_path"] == ["prefix-a", "prefix-b"]
    assert normalized["phase_transition_iterations"] == [10, 20]
    assert normalized["dataloader_type"] == "single"
    # Defaults that depend on another field live on the resolved view only.
    assert normalized["persistent_workers"] is True
    assert normalized["split"] is None and normalized["create_attention_mask"] is True
    resolved = cfg._resolved_fields()
    assert resolved["persistent_workers"] is False
    assert resolved["split"] == LEGACY_SPLIT_DEFAULT
    assert resolved["create_attention_mask"] is False
    assert cfg.seq_length is None and cfg.dataloader_type is None and cfg.split is None


def test_finalize_applies_normalizations_once_and_is_idempotent(monkeypatch, tokenizer):
    warnings = []
    monkeypatch.setattr(dataset_config_module, "_warn", warnings.append)
    cfg = _spelled_declaration()
    serialized = cfg.to_cfg_dict()
    core_before = _core_fields(cfg.to_core_config(_context(tokenizer)))
    assert warnings == []
    cfg.finalize()
    after = _raw(cfg)
    assert after["seq_length"] == 32 and after["sequence_length"] is None
    assert after["data_path"] == ["prefix-a", "prefix-b"]
    assert after["phase_transition_iterations"] == [10, 20]
    assert after["dataloader_type"] == "single"
    # The field-dependent defaults are not written into the declaration; the legacy split default
    # is announced once, as validate_args announces it.
    assert after["persistent_workers"] is True
    assert after["split"] is None and after["create_attention_mask"] is True
    assert len(warnings) == 1 and LEGACY_SPLIT_DEFAULT in warnings[0]
    assert cfg.to_cfg_dict() == serialized
    assert _core_fields(cfg.to_core_config(_context(tokenizer))) == core_before
    cfg.finalize()
    assert _raw(cfg) == after


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(seq_length=8, sequence_length=16), "disagree"),
        (dict(phase_transition_iterations="10,abc"), "comma-separated"),
    ],
)
def test_unnormalizable_input_is_rejected_without_mutation(tokenizer, overrides, message):
    cfg = _spelled_declaration(**overrides)
    before = _raw(cfg)
    actions = (
        cfg.validate,
        cfg.finalize,
        cfg.to_cfg_dict,
        lambda: cfg.to_core_config(_context(tokenizer)),
    )
    for action in actions:
        with pytest.raises(ValueError, match=message):
            action()
        assert _raw(cfg) == before


@pytest.mark.parametrize(
    "overrides",
    [
        dict(seq_length=16, sequence_length=32),
        # The alias carries the only length: a fallback to raw fields would drop it.
        dict(sequence_length=32, phase_transition_iterations="10,abc"),
    ],
)
def test_serialization_never_rewrites_an_invalid_declaration(overrides):
    cfg = GPTDatasetConfig(mock_data=True, **overrides)
    with pytest.raises(ValueError):
        cfg.validate()
    # Serializing raises the same error instead of producing a different, valid declaration.
    with pytest.raises(ValueError):
        cfg.to_cfg_dict()
    with pytest.raises(ValueError):
        PretrainConfigContainer._convert_value_to_dict(cfg)
    assert "sequence_length=32" in repr(cfg)  # the raw fields stay inspectable


def test_sft_mock_default_distribution_matches_validate_args(tokenizer):
    cfg = GPTDatasetConfig(dataset_type="sft", mock_data=True, seq_length=32)
    expected = json.dumps(
        {
            "mode": "distribution",
            "type": "lognormal",
            "min_seq_len": 16,
            "max_seq_len": 32,
            "mean_seq_len": 24,
            "lognormal_sigma": 1.1,
        }
    )
    # Computed for the runtime object; never part of the declaration or its serialized form.
    assert cfg._resolved_fields()["sft_mock_dataset_config_json"] == expected
    assert cfg.to_core_config(_context(tokenizer)).sft_mock_dataset_config_json == expected
    assert cfg._normalized_fields()["sft_mock_dataset_config_json"] is None
    assert cfg.sft_mock_dataset_config_json is None
    assert cfg.to_cfg_dict()["sft_mock_dataset_config_json"] is None
    explicit = GPTDatasetConfig(
        dataset_type="sft", mock_data=True, seq_length=32, sft_mock_dataset_config_json="{}"
    )
    assert explicit._resolved_fields()["sft_mock_dataset_config_json"] == "{}"
    assert explicit.to_cfg_dict()["sft_mock_dataset_config_json"] == "{}"
    # Not an SFT mock run: nothing is synthesized.
    assert (
        GPTDatasetConfig(mock_data=True, seq_length=32)._resolved_fields()[
            "sft_mock_dataset_config_json"
        ]
        is None
    )


def test_round_trip_keeps_field_dependent_defaults_editable(tokenizer):
    # Serializing never freezes a value inferred from another field, so the same edit applied to
    # the original and to the reloaded declaration has the same effect on the runtime object.
    sft = GPTDatasetConfig(dataset_type="sft", mock_data=True, seq_length=16)
    restored = instantiate(sft.to_cfg_dict())
    assert restored.sft_mock_dataset_config_json is None
    sft.seq_length = restored.seq_length = 32
    for cfg in (sft, restored):
        core = cfg.to_core_config(_context(tokenizer))
        assert json.loads(core.sft_mock_dataset_config_json)["max_seq_len"] == 32

    masked = GPTDatasetConfig(mock_data=True, seq_length=16, inter_document_masking=True)
    restored = instantiate(masked.to_cfg_dict())
    assert restored.create_attention_mask is True
    masked.inter_document_masking = restored.inter_document_masking = False
    for cfg in (masked, restored):
        assert cfg.to_core_config(_context(tokenizer)).create_attention_mask is True

    loader = instantiate(
        GPTDatasetConfig(mock_data=True, seq_length=16, num_workers=0).to_cfg_dict()
    )
    assert loader.persistent_workers is True
    assert loader._resolved_fields()["persistent_workers"] is False
    loader.num_workers = 2
    assert loader._resolved_fields()["persistent_workers"] is True

    blend = instantiate(GPTDatasetConfig(seq_length=16, data_path=["a"]).to_cfg_dict())
    assert blend.split is None
    assert blend.to_core_config(_context(tokenizer)).split == LEGACY_SPLIT_DEFAULT


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(data_path=["a"], data_args_path="f"), "data_path and data_args_path"),
        (dict(train_data_path=["a"], per_split_data_args_path="f"), "per_split_data_args_path"),
        (dict(mock_data=True, data_path=["a"]), "single data source"),
        (dict(data_path=["a"], valid_data_path=["b"]), "single data source"),
        (dict(mock_data=True, test_data_path=["b"]), "single data source"),
        # File-backed sources count as sources (validate_args does not count them; no entry point
        # could build these combinations).
        (dict(mock_data=True, data_args_path="f", split="9,1,0"), "single data source"),
        (dict(mock_data=True, per_split_data_args_path="f"), "single data source"),
        (dict(data_args_path="f", per_split_data_args_path="g"), "single data source"),
        (dict(data_path=["a"], per_split_data_args_path="g"), "single data source"),
        (dict(num_dataset_builder_threads=0), "positive"),
        (
            dict(dataset_type="varlen", varlen_sbhd_validation=True, mock_data=True),
            "requires a real dataset",
        ),
        (dict(dataset_type="fim", mock_data=True), "Mock dataset is not supported"),
        (dict(dataset_type="fim", fim_rate=0.0), "fim-rate"),
        (dict(dataset_type="fim", fim_spm_rate=0.0), "fim-spm-rate"),
        (dict(dataset_type="fim", fim_pad_token=None), "FIM extra tokens"),
        (dict(dataset_type="other"), "dataset_type"),
        (dict(dataloader_type="weird"), "dataloader_type"),
        (dict(blend=(["a"], None), data_path=["a"]), "either blend"),
        (
            dict(blend_per_split=[(["a"], None), None, None], train_data_path=["a"]),
            "blend_per_split",
        ),
        (
            dict(sequences_per_dataset={"a": 1}, per_dataset_sequences_path="p"),
            "sequences_per_dataset",
        ),
    ],
)
def test_validation_rejects_the_same_inputs_as_validate_args(overrides, message):
    cfg = GPTDatasetConfig(seq_length=16, **overrides)
    before = _raw(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()
    assert _raw(cfg) == before


@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(seq_length=None),
        dict(mock_data=True, data_path=[]),  # an empty --data-path does not count as a source
        dict(dataset_type="fim", fim_pad_token=None, fim_extra_tokens={"pad": "<p>"}),
        dict(dataset_type="varlen", varlen_sbhd_validation=True),
        dict(dataset_type="gpt", fim_rate=0.0, fim_spm_rate=0.0),
        dict(dataset_type="sft", mock_data=True),
        dict(fast_cache_load=True),  # Megatron Core checks the cache path at build time
        dict(defer_npy_index_mmap=True),
        dict(data_path=["a"], split=None),
        dict(train_data_path=["a"], valid_data_path=[], test_data_path=None),
        dict(data_args_path="f", split="9,1,0"),
        dict(per_split_data_args_path="f"),
        dict(num_workers=0),
    ],
)
def test_validation_accepts_neighbouring_inputs(overrides):
    kwargs = dict(seq_length=16)
    kwargs.update(overrides)
    GPTDatasetConfig(**kwargs).validate()


def test_cache_prerequisites_are_enforced_by_megatron_core_not_the_declaration(tokenizer):
    cfg = GPTDatasetConfig(seq_length=16, mock_data=True, fast_cache_load=True)
    cfg.validate()
    with pytest.raises(AssertionError, match="data-cache-path"):
        cfg.to_core_config(_context(tokenizer))
    core = cfg.to_core_config(_context(tokenizer), exclude={"fast_cache_load"})
    assert core.fast_cache_load is False


def test_to_core_config_builds_a_fresh_runtime_object(tokenizer):
    cfg = GPTDatasetConfig(mock_data=True, seq_length=16)
    before = _raw(cfg)
    first = cfg.to_core_config(_context(tokenizer))
    second = cfg.to_core_config(_context(tokenizer))
    assert type(first) is MCoreGPTDatasetConfig
    assert first is not second
    assert _core_fields(first) == _core_fields(second)
    assert first.mock and first.split_matrix is not None and first.token_dtype_code == 8
    assert first.tokenizer is tokenizer
    assert first.random_seed == 1234 and first.sequence_length == 16
    assert _raw(cfg) == before
    assert cfg.split_matrix is None and cfg.mock is False and cfg.token_dtype_code is None
    assert cfg.tokenizer is None


def test_runtime_context_wins_over_declared_values_and_fills_defaults(tokenizer):
    cfg = GPTDatasetConfig(mock_data=True, seq_length=16)
    core = cfg.to_core_config(_context(tokenizer))
    assert (core.data_parallel_size, core.sequence_parallel_size) == (1, 0)
    assert core.hybrid_context_parallel is False and core.context_parallel_size is None
    assert core.full_validation is None and core.multiple_validation_sets is None
    overridden = cfg.to_core_config(
        DatasetRuntimeContext(
            tokenizer=tokenizer,
            random_seed=7,
            data_parallel_size=4,
            context_parallel_size=2,
            sequence_parallel_size=2,
            hybrid_context_parallel=True,
            full_validation=True,
            multiple_validation_sets=True,
        )
    )
    assert overridden.random_seed == 7
    assert (overridden.data_parallel_size, overridden.context_parallel_size) == (4, 2)
    assert overridden.sequence_parallel_size == 2 and overridden.hybrid_context_parallel is True
    assert overridden.full_validation is True and overridden.multiple_validation_sets is True
    declared = GPTDatasetConfig(mock_data=True, seq_length=16, random_seed=5, data_parallel_size=3)
    core = declared.to_core_config(DatasetRuntimeContext(tokenizer=tokenizer))
    assert (core.random_seed, core.data_parallel_size) == (5, 3)
    with pytest.raises(ValueError, match="random_seed"):
        cfg.to_core_config(DatasetRuntimeContext(tokenizer=tokenizer))
    with pytest.raises(ValueError, match="seq_length"):
        GPTDatasetConfig(mock_data=True).to_core_config(_context(tokenizer))
    with pytest.raises(ValueError, match="tokenizer"):
        cfg.to_core_config(DatasetRuntimeContext(tokenizer=None, random_seed=1))


def test_to_core_config_reads_declared_sources(tmp_path, tokenizer):
    blend = cfg = GPTDatasetConfig(seq_length=16, data_path=["0.3", "a", "0.7", "b"])
    core = cfg.to_core_config(_context(tokenizer))
    assert core.blend == (["a", "b"], [0.3, 0.7]) and core.blend_per_split is None
    assert core.split == LEGACY_SPLIT_DEFAULT and not core.mock
    assert cfg.split is None

    data_args = tmp_path / "data_args.txt"
    data_args.write_text("0.3 a 0.7 b\n")
    from_file = GPTDatasetConfig(seq_length=16, data_args_path=str(data_args), split="9,1,0")
    core = from_file.to_core_config(_context(tokenizer))
    assert core.blend == (["a", "b"], [0.3, 0.7]) and core.split == "9,1,0"

    per_split = GPTDatasetConfig(
        seq_length=16, train_data_path=["0.5", "a", "0.5", "b"], valid_data_path=["c"]
    )
    core = per_split.to_core_config(_context(tokenizer))
    assert core.blend is None
    assert core.blend_per_split == [(["a", "b"], [0.5, 0.5]), (["c"], None), None]

    per_split_json = tmp_path / "per_split.json"
    per_split_json.write_text(json.dumps({"train": "a b", "valid": ["c"], "test": "d"}))
    core = GPTDatasetConfig(
        seq_length=16, per_split_data_args_path=str(per_split_json)
    ).to_core_config(_context(tokenizer))
    assert core.blend_per_split == [(["a", "b"], None), (["c"], None), (["d"], None)]

    sequences = tmp_path / "sequences.json"
    sequences.write_text(json.dumps({"a": 10, "b": 20}))
    with_sequences = GPTDatasetConfig(
        seq_length=16, data_path=["a", "b"], per_dataset_sequences_path=str(sequences)
    )
    assert with_sequences.to_core_config(_context(tokenizer)).sequences_per_dataset == {
        "a": 10,
        "b": 20,
    }
    excluded = with_sequences.to_core_config(_context(tokenizer), exclude={"sequences_per_dataset"})
    assert excluded.sequences_per_dataset is None
    assert with_sequences.sequences_per_dataset is None


def test_exclude_leaves_megatron_core_defaults(tokenizer):
    declared = dict(
        dataset_type="sft",
        mock_data=True,
        seq_length=16,
        allow_ambiguous_pad_tokens=True,
        fast_cache_load=True,
        defer_npy_index_mmap=True,
        path_to_cache="/cache/that/is/never/read",
        data_parallel_size=8,
        sequence_parallel_size=2,
        hybrid_context_parallel=True,
        varlen_mock_dataset_config_json="{}",
    )
    cfg = GPTDatasetConfig(**declared)
    context = _context(
        tokenizer, context_parallel_size=2, full_validation=True, multiple_validation_sets=True
    )
    full = cfg.to_core_config(context)
    assert full.allow_ambiguous_pad_tokens and full.fast_cache_load and full.defer_npy_index_mmap
    assert (full.data_parallel_size, full.sequence_parallel_size) == (8, 2)
    assert full.hybrid_context_parallel and full.context_parallel_size == 2
    assert full.full_validation and full.multiple_validation_sets
    assert full.sft_mock_dataset_config_json is not None
    assert full.varlen_mock_dataset_config_json == "{}"
    filtered = cfg.to_core_config(context, exclude=FLEX_OMITTED)
    defaults = MCoreGPTDatasetConfig(
        random_seed=1234,
        sequence_length=16,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    for name in FLEX_OMITTED:
        assert getattr(filtered, name) == getattr(defaults, name), name
    assert filtered.path_to_cache == "/cache/that/is/never/read"
    assert filtered.sft_mock_dataset_config_json is None
    assert _raw(cfg)["sft_mock_dataset_config_json"] is None
    # A misspelled omission policy is an error, not a silently different runtime object.
    with pytest.raises(ValueError, match="exclude names"):
        cfg.to_core_config(context, exclude={"sft_mock_dataset_config_jsn"})
    with pytest.raises(ValueError, match="exclude names"):
        cfg.to_core_config(context, exclude={"seq_length"})  # the declaration's spelling
    # SBHD validation cannot coexist with hybrid context parallelism on the runtime object.
    sbhd = GPTDatasetConfig(
        **{**declared, "hybrid_context_parallel": None, "varlen_sbhd_validation": True}
    )
    assert sbhd.to_core_config(context).varlen_sbhd_validation is True
    assert sbhd.to_core_config(context, exclude=FLEX_OMITTED).varlen_sbhd_validation is False


def test_fim_declaration_builds_the_runtime_fim_config(tokenizer):
    cfg = GPTFIMDatasetConfig(seq_length=16, data_path=["a"], split="1,1,1", fim_rate=0.9)
    assert cfg.dataset_type == "fim"
    core = cfg.to_core_config(_context(tokenizer))
    assert type(core) is RuntimeGPTFIMDatasetConfig
    assert core.fim_rate == 0.9 and core.fim_spm_rate == 0.5
    assert core.fim_extra_tokens == {
        "prefix": "<fim_prefix>",
        "middle": "<fim_middle>",
        "suffix": "<fim_suffix>",
        "pad": "<fim_pad>",
        "eod": "<|endoftext|>",
    }
    assert core.fim_split_sample is None and core.fim_fragment_rate is None
    explicit = GPTDatasetConfig(
        dataset_type="fim",
        seq_length=16,
        data_path=["a"],
        split="1,1,1",
        fim_extra_tokens={"prefix": "P", "middle": "M", "suffix": "S", "pad": "X", "eod": "E"},
    )
    core = explicit.to_core_config(_context(tokenizer))
    assert core.fim_extra_tokens["prefix"] == "P"
    assert core.fim_extra_tokens is not explicit.fim_extra_tokens
    plain = GPTDatasetConfig(seq_length=16, data_path=["a"], split="1,1,1", fim_rate=0.9)
    assert type(plain.to_core_config(_context(tokenizer))) is MCoreGPTDatasetConfig


def test_serialization_round_trip(tokenizer):
    cfg = GPTDatasetConfig(
        seq_length=16,
        blend=(["a", "b"], [0.25, 0.75]),
        split="9,1,0",
        phase_transition_iterations="30,20",
        num_workers=0,
        dataset_type="gpt",
        random_seed=11,
    )
    serialized = PretrainConfigContainer._convert_value_to_dict(cfg)
    assert serialized["_target_"] == "megatron.training.config.dataset_config.GPTDatasetConfig"
    for absent in ("tokenizer", "sequence_length", "split_matrix", "mock", "token_dtype_code"):
        assert absent not in serialized
    for unset_context_input in ("data_parallel_size", "full_validation"):
        assert unset_context_input not in serialized
    assert "sequences_per_dataset" not in serialized
    assert serialized["random_seed"] == 11
    assert serialized["blend"] == [["a", "b"], [0.25, 0.75]]
    assert serialized["phase_transition_iterations"] == [20, 30]
    assert serialized["dataloader_type"] == "single"  # the spelling of "unset"
    assert serialized["num_workers"] == 0 and serialized["persistent_workers"] is True
    restored = instantiate(copy.deepcopy(serialized))
    assert type(restored) is GPTDatasetConfig
    assert restored.to_cfg_dict() == serialized
    assert _core_fields(restored.to_core_config(_context(tokenizer))) == _core_fields(
        cfg.to_core_config(_context(tokenizer))
    )
    assert restored.to_core_config(_context(tokenizer)).blend == (["a", "b"], [0.25, 0.75])
    cfg.finalize()
    assert cfg.to_cfg_dict() == serialized

    preset = PretrainConfigContainer._convert_value_to_dict(GPTFIMDatasetConfig(seq_length=8))
    assert preset["_target_"].endswith("GPTFIMDatasetConfig") and preset["dataset_type"] == "fim"
    assert type(instantiate(preset)) is GPTFIMDatasetConfig


def test_inter_document_masking_disables_the_dataloader_attention_mask(tokenizer):
    cfg = GPTDatasetConfig(seq_length=16, mock_data=True, inter_document_masking=True)
    assert cfg.create_attention_mask is True
    assert cfg.to_core_config(_context(tokenizer)).create_attention_mask is False
    # Computed on the runtime object only: neither serialized nor written by finalize().
    assert cfg.to_cfg_dict()["create_attention_mask"] is True
    cfg.finalize()
    assert cfg.create_attention_mask is True
    assert cfg.to_core_config(_context(tokenizer)).create_attention_mask is False


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({}, "gpt"),
        ({"sft": True}, "sft"),
        ({"use_varlen_dataset": True}, "varlen"),
        ({"fim_data": True}, "fim"),
        ({"dataset_type": "varlen"}, "varlen"),
        ({"dataset_type": "gpt"}, "gpt"),
        ({"dataset_type": "sft", "sft": True}, "sft"),
        ({"dataset_type": "fim", "fim_data": True}, "fim"),
        ({"sft": True, "fim_data": True}, "sft"),
        ({"use_varlen_dataset": True, "fim_data": True}, "varlen"),
        ({"dataset_type": "sft", "sft": True, "fim_data": True}, "sft"),
        ({"sft": True, "use_varlen_dataset": True}, ValueError("mutually exclusive")),
        ({"dataset_type": "gpt", "fim_data": True}, ValueError("contradicts")),
        ({"dataset_type": "fim", "sft": True}, ValueError("contradicts")),
        ({"dataset_type": "sft", "fim_data": True}, ValueError("contradicts")),
        ({"dataset_type": "varlen", "sft": True}, ValueError("contradicts")),
        ({"dataset_type": "bogus"}, ValueError("must be one of")),
    ],
)
def test_dataset_type_resolution(monkeypatch, flags, expected):
    warnings = []
    monkeypatch.setattr(dataset_config_module, "_warn", warnings.append)
    args = Namespace(dataset_type=None, sft=False, use_varlen_dataset=False, fim_data=False)
    vars(args).update(flags)
    if isinstance(expected, ValueError):
        with pytest.raises(ValueError, match=str(expected)):
            dataset_type_from_args(args)
        with pytest.raises(ValueError, match=str(expected)):
            project_dataset_type_into_args(args)
        return
    assert dataset_type_from_args(args) == expected
    aliases = sum(bool(getattr(args, dest)) for dest in ("sft", "use_varlen_dataset", "fim_data"))
    assert len(warnings) == (1 if aliases > 1 else 0)
    assert dataset_type_from_args(args, warn=False) == expected
    assert len(warnings) == (1 if aliases > 1 else 0)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"sft": True, "fim_data": True}, "sft"),
        ({"use_varlen_dataset": True, "fim_data": True}, "varlen"),
        ({"dataset_type": "varlen"}, "varlen"),
        ({"dataset_type": "sft"}, "sft"),
        ({"dataset_type": "fim"}, "fim"),
        ({"dataset_type": "gpt"}, "gpt"),
        ({}, "gpt"),
        ({"fim_data": True}, "fim"),
    ],
)
def test_projection_is_stable_and_preserves_legacy_flags(monkeypatch, flags, expected):
    warnings = []
    monkeypatch.setattr(dataset_config_module, "_warn", warnings.append)
    args = Namespace(dataset_type=None, sft=False, use_varlen_dataset=False, fim_data=False)
    vars(args).update(flags)
    passed = {dest for dest in ("sft", "use_varlen_dataset", "fim_data") if getattr(args, dest)}
    assert project_dataset_type_into_args(args) == expected
    assert args.dataset_type == expected
    for dest, family in (("sft", "sft"), ("use_varlen_dataset", "varlen"), ("fim_data", "fim")):
        assert getattr(args, dest) is (dest in passed or family == expected)
    projected = dict(vars(args))
    # The factory path resolves again and must agree without warning twice.
    assert dataset_type_from_args(args, warn=False) == expected
    assert project_dataset_type_into_args(args) == expected
    assert vars(args) == projected
    assert len(warnings) == (2 if len(passed) > 1 else 0)  # once per projection call only


def test_runtime_context_from_args():
    args = Namespace(
        seed=3,
        data_parallel_size=2,
        context_parallel_size=1,
        tensor_model_parallel_size=4,
        sequence_parallel=True,
        hybrid_context_parallel=False,
        full_validation=True,
        multiple_validation_sets=False,
    )
    context = DatasetRuntimeContext.from_args(args, tokenizer="tok")
    assert context.tokenizer == "tok" and context.random_seed == 3
    assert (context.data_parallel_size, context.context_parallel_size) == (2, 1)
    assert context.sequence_parallel_size == 4 and context.hybrid_context_parallel is False
    assert context.full_validation is True and context.multiple_validation_sets is False
    sparse = DatasetRuntimeContext.from_args(Namespace(tensor_model_parallel_size=4), "tok")
    assert sparse.random_seed is None and sparse.sequence_parallel_size == 0
    with pytest.raises(AttributeError):
        context.random_seed = 4  # frozen


@pytest.mark.parametrize(
    ("overrides", "packed"),
    [
        (dict(dataset_type="sft"), True),
        (dict(dataset_type="varlen"), True),
        (dict(dataset_type="varlen", varlen_sbhd_validation=True), False),
        (dict(dataset_type="gpt"), False),
        (dict(dataset_type="fim"), False),
    ],
)
def test_is_packed_sequence(overrides, packed):
    assert GPTDatasetConfig(**overrides).is_packed_sequence is packed


def test_dataloader_config_lifecycle():
    cfg = DataloaderConfig(num_workers=0)
    cfg.validate()
    assert cfg.dataloader_type is None and cfg.persistent_workers is True
    assert cfg._resolved_loader_fields()["persistent_workers"] is False
    cfg.finalize()
    assert cfg.dataloader_type == "single" and cfg.persistent_workers is True
    cfg.finalize()
    assert cfg.dataloader_type == "single"
    with pytest.raises(ValueError, match="dataloader_type"):
        DataloaderConfig(dataloader_type="weird").validate()
    assert DataloaderConfig(dataloader_type="external").dataloader_type == "external"
