# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Each GPT entry point builds the runtime dataset config and dataset class it built before.

The reference is the frozen factory and provider logic of commit 438ef0025
(``legacy_dataset_factories.py``), run on the same validated arguments as the new path
(``cfg.dataset`` -> ``to_core_config`` -> each entry point's policy).
"""

import ast
import inspect
import json
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.datasets.utils import Split, compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import global_vars, training
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import GPTDatasetConfig
from megatron.training.datasets import sft_dataset, varlen_dataset
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training import legacy_dataset_factories as legacy

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32 "
    "--tokenizer-type NullTokenizer --vocab-size 2048"
).split()

PARALLEL_OVERRIDES = dict(
    context_parallel_size=2,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    data_parallel_size=4,
    hybrid_context_parallel=True,
    full_validation=True,
    multiple_validation_sets=True,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset_providers")
    data_args = root / "data_args.txt"
    data_args.write_text("0.3 prefix-a 0.7 prefix-b\n")
    per_split = root / "per_split.json"
    per_split.write_text(
        json.dumps({"train": "prefix-a prefix-b", "valid": ["prefix-c"], "test": "prefix-d"})
    )
    sequences = root / "sequences.json"
    sequences.write_text(json.dumps({"prefix-a": 10, "prefix-b": 20}))
    return {
        "mock": (["--mock-data"], {}),
        "mock_parallel": (["--mock-data"], PARALLEL_OVERRIDES),
        "blend": (["--data-path", "0.3", "prefix-a", "0.7", "prefix-b", "--split", "9,1,0"], {}),
        "blend_default_split": (["--data-path", "prefix-a"], {}),
        "per_split": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c"],
            {},
        ),
        "data_args_file": (["--data-args-path", str(data_args), "--split", "9,1,0"], {}),
        "per_split_file": (["--per-split-data-args-path", str(per_split)], {}),
        "sft_mock": (["--sft", "--mock-data"], {}),
        "sft": (["--sft", "--data-path", "prefix-a"], {}),
        "varlen_mock": (
            ["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss", "--max-seqlen-per-dp-cp-rank", "32"],
            {},
        ),
        "varlen_sbhd": (["--use-varlen-dataset", "--data-path", "prefix-a", "--varlen-sbhd-validation"], {}),
        "fim": (
            ["--fim-data", "--data-path", "prefix-a", "--fim-rate", "0.9", "--fim-spm-rate", "0.4",
             "--fim-split-sample", "<split>", "--fim-fragment-rate", "0.2", "--fim-no-prefix", "nofim",
             "--fim-prefix-token", "<p>", "--fim-eod-token", "<e>"],
            {},
        ),
        "cache": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c",
             "--data-cache-path", "/cache", "--dataloader-fast-cache-load", "--dataloader-defer-npy-index-mmap"],
            {},
        ),
        "knobs": (
            ["--data-path", "prefix-a", "--phase-transition-iterations", "20,10",
             "--dataloader-inter-document-masking", "--reset-position-ids", "--reset-attention-mask",
             "--eod-mask-loss", "--no-create-attention-mask-in-dataloader", "--num-dataset-builder-threads", "3",
             "--mid-level-dataset-surplus", "0.1", "--object-storage-cache-path", "/object-cache",
             "--allow-ambiguous-pad-tokens", "--no-mmap-bin-files", "--per-dataset-sequences-path", str(sequences),
             "--sft-mock-dataset-config-json", "{}", "--varlen-mock-dataset-config-json", "{}",
             "--varlen-sbhd-validation", "--num-workers", "0", "--dataloader-type", "cyclic"],
            {},
        ),
        "sft_fim": (["--sft", "--fim-data", "--data-path", "prefix-a"], {}),
        "varlen_fim": (
            ["--use-varlen-dataset", "--fim-data", "--data-path", "prefix-a", "--calculate-per-token-loss",
             "--max-seqlen-per-dp-cp-rank", "32"],
            {},
        ),
        "implicit_mock": ([], {}),
        # Accepted by validate_args; rejected by the declaration (see the dedicated test).
        "mock_data_args_file": (["--mock-data", "--data-args-path", str(data_args), "--split", "9,1,0"], {}),
        "mock_per_split_file": (["--mock-data", "--per-split-data-args-path", str(per_split)], {}),
    }  # fmt: skip


CASE_NAMES = [
    "mock", "mock_parallel", "blend", "blend_default_split", "per_split", "data_args_file",
    "per_split_file", "sft_mock", "sft", "varlen_mock", "varlen_sbhd", "fim", "cache", "knobs",
    "sft_fim", "varlen_fim", "implicit_mock",
]  # fmt: skip
MIXED_FIM_CASES = {"sft_fim", "varlen_fim"}


def _entry_point(name):
    if name == "gpt":
        import pretrain_gpt as module
    elif name == "hybrid":
        import pretrain_hybrid as module
    elif name == "hybrid_flex":
        from megatron.elastification import pretrain_hybrid_flex as module
    else:
        from tools import prepare_cache as module
    return module


ENTRY_POINTS = {
    "gpt": (legacy.legacy_gpt_dataset_config, legacy.legacy_gpt_dataset_class),
    "hybrid": (legacy.legacy_hybrid_dataset_config, legacy.legacy_hybrid_dataset_class),
    "hybrid_flex": (
        legacy.legacy_hybrid_flex_dataset_config,
        lambda args: (legacy.legacy_hybrid_flex_dataset_class(args), None),
    ),
    "prepare_cache": (legacy.legacy_prepare_cache_dataset_config, None),
}


def _prepare(monkeypatch, extra, overrides):
    with patch("sys.argv", ["test_dataset_providers.py", *MODEL_ARGV, *extra]):
        args = validate_args(parse_args())
    for name, value in overrides.items():
        setattr(args, name, value)
    cfg = pretrain_cfg_container_from_args(args)
    tokenizer = build_tokenizer(args)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", tokenizer)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    monkeypatch.setattr(training, "get_args", lambda: args)
    return args, cfg, training.build_dataset_runtime_context(cfg)


def _tokenizer_signature(tokenizer):
    signature = {"type": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size}
    for name in ("eod", "pad", "bos", "eos", "mask", "cls", "sep"):
        try:
            signature[name] = getattr(tokenizer, name)
        except Exception:  # noqa: BLE001 - a tokenizer may not define the special token
            signature[name] = "unavailable"
    return signature


def _signature(config, names):
    signature = {name: getattr(config, name) for name in names}
    signature["tokenizer"] = _tokenizer_signature(signature["tokenizer"])
    return signature


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("entry_point", list(ENTRY_POINTS))
def test_entry_points_build_the_same_runtime_config(monkeypatch, corpus, entry_point, case):
    extra, overrides = corpus[case]
    args, cfg, context = _prepare(monkeypatch, extra, overrides)
    legacy_factory, legacy_classes = ENTRY_POINTS[entry_point]
    expected = legacy_factory(args)
    module = _entry_point(entry_point)
    if entry_point == "prepare_cache":
        dataset_class, actual, packed = None, module.core_gpt_dataset_config_from_args(args), None
    else:
        dataset_class, actual, packed = module._resolve_dataset_build(cfg.dataset, context)
    if entry_point == "gpt" and case in MIXED_FIM_CASES:
        # Documented exception: the dataset class and every shared field are unchanged; the unused
        # FIM runtime wrapper around an SFT / varlen dataset is gone.
        assert type(expected) is GPTFIMDatasetConfig and type(actual) is MCoreGPTDatasetConfig
        names = [f.name for f in fields(MCoreGPTDatasetConfig)]
    else:
        assert type(actual) is type(expected)
        names = [f.name for f in fields(expected)]
    assert _signature(actual, names) == _signature(expected, names)
    if legacy_classes is not None:
        expected_class, expected_packed = legacy_classes(args)
        if case == "implicit_mock":
            # Documented difference: the legacy providers keyed on --mock-data and paired a mock
            # runtime config with the real dataset class; the new tables key on the resolved mock.
            assert expected_class is GPTDataset and dataset_class is MockGPTDataset
        else:
            assert dataset_class is expected_class
        if expected_packed is not None:
            assert packed is expected_packed
        wrapper = module.core_gpt_dataset_config_from_args(args)
        assert type(wrapper) is type(actual)
        assert _signature(wrapper, names) == _signature(actual, names)
    # The declaration itself carried no runtime state.
    assert cfg.dataset.tokenizer is None and cfg.dataset.split_matrix is None
    assert cfg.dataset.mock is False and cfg.dataset.token_dtype_code is None


@pytest.mark.parametrize("entry_point", ["gpt", "hybrid", "hybrid_flex"])
def test_explicit_provider_path_builds_no_second_tokenizer(monkeypatch, corpus, entry_point):
    args, cfg, context = _prepare(monkeypatch, *corpus["mock"])
    module = _entry_point(entry_point)
    if hasattr(module, "build_tokenizer"):
        monkeypatch.setattr(module, "build_tokenizer", Mock(side_effect=AssertionError("rebuilt")))
    _, config, _ = module._resolve_dataset_build(cfg.dataset, context)
    assert config.tokenizer is global_vars.get_tokenizer()
    assert training.build_dataset_runtime_context().tokenizer is global_vars.get_tokenizer()


def test_sft_and_varlen_datasets_read_no_fim_field():
    """The mixed-FIM exception rests on this: the packed datasets never read a fim_* attribute."""
    for module in (sft_dataset, varlen_dataset):
        tree = ast.parse(inspect.getsource(module))
        attribute_reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr.lower().startswith("fim")
        }
        assert not attribute_reads, (module.__name__, attribute_reads)


@pytest.mark.parametrize("case", ["mock_data_args_file", "mock_per_split_file"])
@pytest.mark.parametrize("entry_point", ["gpt", "hybrid"])
def test_mock_data_next_to_a_file_backed_source_is_rejected(monkeypatch, corpus, entry_point, case):
    """validate_args accepts --mock-data next to a file-backed source. The legacy providers then
    selected MockGPTDataset (keyed on --mock-data) around a runtime config built from the file
    (mock=False), which MockGPTDataset refuses, so the combination never trained. The declaration
    rejects it at container validation instead of resolving the file or ignoring it."""
    args, cfg, _ = _prepare(monkeypatch, *corpus[case])
    legacy_factory, legacy_classes = ENTRY_POINTS[entry_point]
    legacy_config = legacy_factory(args)
    assert legacy_config.mock is False
    assert legacy_config.blend is not None or legacy_config.blend_per_split is not None
    assert legacy_classes(args)[0] is MockGPTDataset
    with pytest.raises(ValueError, match="single data source"):
        cfg.dataset.validate()
    with pytest.raises(ValueError, match="single data source"):
        cfg.validate()


def test_fim_declaration_without_a_source_is_rejected(monkeypatch, corpus):
    _, cfg, context = _prepare(monkeypatch, *corpus["mock"])
    with pytest.raises(ValueError, match="declare a data source"):
        _entry_point("gpt")._resolve_dataset_build(
            GPTDatasetConfig(dataset_type="fim", seq_length=32), context
        )


@pytest.mark.parametrize(
    ("entry_point", "dataset_type", "warns"),
    [
        ("hybrid", "fim", True),
        ("hybrid", "varlen", False),
        ("hybrid_flex", "fim", True),
        ("hybrid_flex", "varlen", True),
        ("hybrid_flex", "sft", False),
    ],
)
def test_entry_point_policies_warn_when_they_substitute_the_family(
    monkeypatch, corpus, entry_point, dataset_type, warns
):
    _, cfg, context = _prepare(monkeypatch, *corpus["mock"])
    module = _entry_point(entry_point)
    warnings = []
    monkeypatch.setattr(module, "warn_rank_0", warnings.append)
    requested = GPTDatasetConfig(dataset_type=dataset_type, seq_length=32, data_path=["prefix-a"])
    effective = module._effective_dataset_config(requested)
    assert requested.dataset_type == dataset_type
    assert effective.dataset_type == ("gpt" if warns else dataset_type)
    assert len(warnings) == int(warns)
    if warns:
        assert dataset_type in warnings[0] and "as before" in warnings[0]


class TestProvidersBuildMockDatasets:
    """The providers build mock datasets through the real builder, as the legacy code did."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1)
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    # pretrain_hybrid and pretrain_hybrid_flex have no mock SFT dataset: --sft --mock-data selects
    # the real SFTDataset there and fails to build, before and after this change.
    @pytest.mark.parametrize(
        ("entry_point", "case"),
        [
            ("gpt", "mock"),
            ("gpt", "sft_mock"),
            ("gpt", "varlen_mock"),
            ("hybrid", "mock"),
            ("hybrid", "varlen_mock"),
            ("hybrid_flex", "mock"),
            ("hybrid_flex", "varlen_mock"),
        ],
    )
    def test_provider_matches_the_legacy_build(self, monkeypatch, corpus, entry_point, case):
        args, cfg, context = _prepare(monkeypatch, *corpus[case])
        module = _entry_point(entry_point)
        legacy_factory, legacy_classes = ENTRY_POINTS[entry_point]
        legacy_class = legacy_classes(args)[0]
        expected = BlendedMegatronDatasetBuilder(
            legacy_class, (8, 8, 8), lambda: True, legacy_factory(args)
        ).build()
        actual = module.train_valid_test_datasets_provider(
            (8, 8, 8), dataset_config=cfg.dataset, context=context
        )
        fallback = module.train_valid_test_datasets_provider((8, 8, 8))
        for built in (actual, fallback):
            _assert_same_datasets(expected, built)

    @pytest.mark.parametrize("case", ["sft_mock", "varlen_mock"])
    def test_packed_mock_datasets_ignore_the_fim_wrapper(self, monkeypatch, corpus, case):
        """Mixed-FIM exception: an SFT / varlen dataset builds the same samples whether its runtime
        config is the FIM wrapper pretrain_gpt used to build for --sft --fim-data or the plain
        Megatron Core config the provider builds now (mock datasets; validate_args rejects
        --fim-data with --mock-data, so the configs are constructed directly here)."""
        args, cfg, context = _prepare(monkeypatch, *corpus[case])
        plain = cfg.dataset.to_core_config(context)
        shared = {f.name: getattr(plain, f.name) for f in fields(MCoreGPTDatasetConfig) if f.init}
        wrapped = GPTFIMDatasetConfig(
            **shared,
            fim_rate=0.9,
            fim_spm_rate=0.4,
            fim_extra_tokens={"prefix": "<p>", "middle": "<m>", "suffix": "<s>", "pad": "<x>", "eod": "<e>"},
        )  # fmt: skip
        dataset_class = legacy.legacy_gpt_dataset_class(args)[0]
        with_wrapper = BlendedMegatronDatasetBuilder(
            dataset_class, (8, 8, 8), lambda: True, wrapped
        ).build()
        without = BlendedMegatronDatasetBuilder(
            dataset_class, (8, 8, 8), lambda: True, plain
        ).build()
        _assert_same_datasets(with_wrapper, without)

    def test_legacy_mock_data_next_to_a_file_backed_source_did_not_build(self, monkeypatch, corpus):
        """The legacy outcome behind test_mock_data_next_to_a_file_backed_source_is_rejected: the
        class the legacy provider selected refuses the runtime config the legacy factory built.
        Constructed directly (the builder constructs on rank 0 before a barrier, so an exception
        there would leave the other rank waiting)."""
        args, _, _ = _prepare(monkeypatch, *corpus["mock_data_args_file"])
        legacy_config = legacy.legacy_gpt_dataset_config(args)
        legacy_class = legacy.legacy_gpt_dataset_class(args)[0]
        assert legacy_class is MockGPTDataset and legacy_config.mock is False
        with pytest.raises(AssertionError):
            legacy_class(None, None, None, 8, Split.train, legacy_config)


def _assert_same_datasets(expected, built):
    for expected_split, split in zip(expected, built):
        assert type(split) is type(expected_split)
        assert len(split) == len(expected_split)
        sample, expected_sample = split[0], expected_split[0]
        assert sample.keys() == expected_sample.keys()
        for key in expected_sample:
            assert (sample[key] == expected_sample[key]).all(), key
