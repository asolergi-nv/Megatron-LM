# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The data argument group is generated from GPTDatasetConfig and keeps the legacy CLI contract.

``legacy_data_arguments.json`` is a snapshot of the data-related argparse actions of the parser
before the group was generated (Megatron-LM commit 438ef0025): option strings, dest, default,
nargs, type, choices, action class, required and const.
"""

import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from megatron.training import arguments
from megatron.training.argument_utils import (
    ArgumentGroupFactory,
    DatasetArgumentGroupFactory,
    gpt_dataset_config_from_args,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import GPTDatasetConfig
from megatron.training.config.dataset_config import (
    DATASET_CLI_EXCLUDE,
    LEGACY_ALIAS_FLAGS,
    LEGACY_ALIAS_HELP,
    LEGACY_ARG_MAP,
)

LEGACY = json.loads((Path(__file__).parent / "legacy_data_arguments.json").read_text())

# Additional spellings the generated group adds to existing dests (store_false flags get both).
ADDED_SPELLINGS = {
    "data_sharding": ["--disable-data-sharding"],
    "mmap_bin_files": ["--disable-mmap-bin-files"],
}

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32"
).split()


def _full_parser():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser


def _actions_by_dest(parser, dests=None):
    """Map dest -> action; a dest in ``dests`` (default: the data dests) must be registered once."""
    dests = set(LEGACY) | {"dataset_type"} if dests is None else set(dests)
    actions = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.dest in dests:
            assert action.dest not in actions, f"{action.dest} is registered twice"
            actions[action.dest] = action
        else:
            actions.setdefault(action.dest, action)
    return actions


def _describe(action):
    return dict(
        option_strings=list(action.option_strings),
        default=action.default,
        nargs=action.nargs,
        type=None if action.type is None else getattr(action.type, "__name__", repr(action.type)),
        choices=None if action.choices is None else list(action.choices),
        action=type(action).__name__,
        required=action.required,
        const=action.const,
    )


def test_every_legacy_data_argument_is_unchanged():
    actions = _actions_by_dest(_full_parser())
    for dest, legacy in LEGACY.items():
        assert dest in actions, dest
        current = _describe(actions[dest])
        extra = [s for s in current["option_strings"] if s not in legacy["option_strings"]]
        assert extra == ADDED_SPELLINGS.get(dest, []), dest
        assert [
            s for s in legacy["option_strings"] if s not in current["option_strings"]
        ] == [], dest
        current["option_strings"] = legacy["option_strings"]
        assert current == legacy, dest


def test_dataset_type_is_the_only_new_data_argument():
    legacy_dests = set(LEGACY)
    data_only = ArgumentParser()
    arguments._add_data_args(data_only)
    generated = set(_actions_by_dest(data_only))
    assert generated - legacy_dests == {"dataset_type"}
    # Every field of the declaration that is not excluded has a flag, and nothing else does.
    expected = {
        LEGACY_ARG_MAP.get(f.name, {}).get("dest", f.name)
        for f in GPTDatasetConfig.__dataclass_fields__.values()
        if f.init and f.name not in DATASET_CLI_EXCLUDE
    }
    expected |= {dest for dest, _family in LEGACY_ALIAS_FLAGS.values()}
    expected |= {
        "encoder_seq_length",
        "decoder_seq_length",
        "sample_rate",
        "mask_prob",
        "short_seq_prob",
    }
    assert generated == expected
    action = _actions_by_dest(data_only)["dataset_type"]
    assert _describe(action) == dict(
        option_strings=["--dataset-type"],
        default=None,
        nargs=None,
        type="str",
        choices=["gpt", "fim", "varlen", "sft"],
        action="_StoreAction",
        required=False,
        const=None,
    )
    for flag, (dest, _family) in LEGACY_ALIAS_FLAGS.items():
        alias = _actions_by_dest(data_only)[dest]
        assert alias.option_strings == [flag] and alias.help == LEGACY_ALIAS_HELP[flag]


def test_full_parser_has_no_duplicate_option_strings():
    seen = {}
    for action in _full_parser()._actions:
        for option in action.option_strings:
            assert option not in seen, option
            seen[option] = action.dest


def test_generated_group_builds_without_type_inference_warnings(capsys):
    parser = ArgumentParser()
    arguments._add_data_args(parser)
    assert "WARNING" not in capsys.readouterr().out
    args = parser.parse_args(
        ["--data-path", "0.5", "a", "0.5", "b", "--phase-transition-iterations", "20,10"]
    )
    assert args.data_path == ["0.5", "a", "0.5", "b"]
    assert args.phase_transition_iterations == "20,10"
    assert parser.parse_args(["--data-path"]).data_path == []


@dataclass
class _Base:
    """Base with field docstrings."""

    shared: int = 1
    """base shared"""

    base_only: int = 2
    """base only"""


@dataclass
class _Mixin:
    """Mixin with field docstrings."""

    shared: int = 3
    """mixin shared"""

    mixin_only: int = 4
    """mixin only"""


@dataclass
class _Child(_Base, _Mixin):
    """Child redeclaring a documented field."""

    shared: int = 5
    """child shared"""

    child_only: int = 6
    """child only"""


def test_field_docstrings_follow_the_method_resolution_order():
    docstrings = ArgumentGroupFactory(_Child).field_docstrings
    assert docstrings == {
        "shared": "child shared",
        "base_only": "base only",
        "mixin_only": "mixin only",
        "child_only": "child only",
    }

    @dataclass
    class _Grandchild(_Child):
        """No own docstrings."""

        extra: int = 7

    assert ArgumentGroupFactory(_Grandchild).field_docstrings["shared"] == "child shared"
    assert ArgumentGroupFactory(_Grandchild).field_docstrings["mixin_only"] == "mixin only"


@dataclass
class _Root:
    """Root of a diamond."""

    diamond: int = 0
    """root diamond"""


@dataclass
class _Left(_Root):
    """Left side: no docstring of its own for the shared field."""

    left_only: int = 1
    """left only"""


@dataclass
class _Right(_Root):
    """Right side: redeclares the shared field."""

    diamond: int = 2
    """right diamond"""


@dataclass
class _Diamond(_Left, _Right):
    """Linearized as _Diamond, _Left, _Right, _Root."""

    tip: int = 3
    """tip"""


def test_field_docstrings_follow_the_c3_linearization_through_a_diamond():
    assert _Diamond.__mro__[1:4] == (_Left, _Right, _Root)
    docstrings = ArgumentGroupFactory(_Diamond).field_docstrings
    # A depth-first walk of the bases would reach _Root through _Left before _Right.
    assert docstrings["diamond"] == "right diamond"
    assert docstrings["left_only"] == "left only" and docstrings["tip"] == "tip"


def test_dataset_factory_applies_legacy_overrides_last():
    @dataclass
    class _Config:
        """Factory input."""

        path_to_cache: str | None = None
        """cache path"""

        values: str | list[int] | None = field(
            default=None, metadata={"argparse_meta": {"nargs": "+", "type": int}}
        )
        """union described by metadata"""

        enabled: bool = True
        """default-on flag"""

    parser = ArgumentParser()
    DatasetArgumentGroupFactory(
        _Config,
        legacy={"path_to_cache": {"arg_names": ["--cache"], "dest": "cache", "type": None}},
        aliases={"--legacy-on": ("legacy_on", "gpt")},
    ).build_group(parser, "test")
    actions = _actions_by_dest(parser, dests={"cache", "values", "enabled", "legacy_on"})
    assert actions["cache"].option_strings == ["--cache"] and actions["cache"].type is None
    assert actions["values"].nargs == "+" and actions["values"].type is int
    assert actions["enabled"].option_strings == ["--no-enabled", "--disable-enabled"]
    assert actions["legacy_on"].option_strings == ["--legacy-on"]
    assert "Alias of --dataset-type gpt" in actions["legacy_on"].help
    args = parser.parse_args(["--cache", "/c", "--values", "3", "1", "--legacy-on"])
    assert (args.cache, args.values, args.legacy_on, args.enabled) == ("/c", [3, 1], True, True)


def _validated(extra):
    argv = ["test_data_arguments.py", *MODEL_ARGV, *extra]
    with patch("sys.argv", argv):
        return validate_args(parse_args())


def _comparable(args):
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


@pytest.mark.parametrize(
    ("legacy_argv", "selector_argv"),
    [
        ([], ["--dataset-type", "gpt"]),
        (["--mock-data"], ["--dataset-type", "gpt", "--mock-data"]),
        (["--sft", "--mock-data"], ["--dataset-type", "sft", "--mock-data"]),
        (
            ["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss"],
            ["--dataset-type", "varlen", "--mock-data", "--calculate-per-token-loss"],
        ),
        (["--fim-data", "--data-path", "a"], ["--dataset-type", "fim", "--data-path", "a"]),
    ],
)
def test_dataset_type_and_alias_flags_validate_to_the_same_arguments(legacy_argv, selector_argv):
    legacy = _validated(legacy_argv)
    selector = _validated(selector_argv)
    assert _comparable(legacy) == _comparable(selector)
    assert gpt_dataset_config_from_args(legacy) == gpt_dataset_config_from_args(selector)
    assert legacy.dataset_type == selector.dataset_type
    assert legacy.dataset_type == (selector_argv[1] if selector_argv else "gpt")


def test_validate_args_keeps_the_legacy_data_normalizations():
    args = _validated(["--sft", "--mock-data", "--phase-transition-iterations", "20,10"])
    assert args.sft is True and args.dataset_type == "sft"
    assert args.phase_transition_iterations == [10, 20]
    assert json.loads(args.sft_mock_dataset_config_json)["max_seq_len"] == 32
    assert args.dataloader_type == "single"
    declared = gpt_dataset_config_from_args(args)
    declared.validate()
    assert declared.phase_transition_iterations == [10, 20]
    # A declaration built from validated arguments carries the defaults validate_args materialized
    # (here the SFT mock distribution) explicitly, and serializes them as declared values.
    assert declared.sft_mock_dataset_config_json == args.sft_mock_dataset_config_json
    assert (
        declared.to_cfg_dict()["sft_mock_dataset_config_json"] == args.sft_mock_dataset_config_json
    )
    assert declared.split is None and declared.dataloader_type == "single"
    varlen = _validated(["--dataset-type", "varlen", "--mock-data", "--calculate-per-token-loss"])
    assert varlen.use_varlen_dataset is True and varlen.sequence_packing_scheduler == "dp_balanced"
    assert varlen.variable_seq_lengths is True


def test_legacy_mixed_fim_flags_still_validate():
    for extra in (
        ["--sft", "--fim-data"],
        ["--use-varlen-dataset", "--fim-data", "--calculate-per-token-loss"],
    ):
        args = _validated([*extra, "--data-path", "a"])
        assert args.fim_data is True
        assert args.dataset_type == ("sft" if "--sft" in extra else "varlen")
        gpt_dataset_config_from_args(args).validate()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--dataset-type", "gpt", "--fim-data"], "contradicts"),
        (["--dataset-type", "fim", "--sft"], "contradicts"),
        (["--sft", "--use-varlen-dataset"], "mutually exclusive"),
        (["--dataset-type", "fim", "--mock-data"], "Mock dataset is not supported"),
    ],
)
def test_validate_args_rejects_contradictory_selectors(extra, message):
    with pytest.raises((ValueError, AssertionError), match=message):
        _validated(extra)


def test_validate_args_rejects_an_unknown_dataset_type():
    with pytest.raises(SystemExit):
        _validated(["--dataset-type", "bogus"])


@pytest.mark.parametrize(("rate", "warns"), [("1.0", False), ("0.5", True)])
def test_sample_rate_is_deprecated(monkeypatch, rate, warns):
    warnings = []
    monkeypatch.setattr(arguments, "warn_rank_0", warnings.append)
    args = _validated(["--sample-rate", rate])
    assert args.sample_rate == float(rate)
    assert any("--sample-rate is deprecated" in w for w in warnings) is warns


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--mock-data"],
        ["--data-path", "0.3", "a", "0.7", "b", "--split", "9,1,0"],
        ["--data-path", "a"],
        ["--train-data-path", "a", "--valid-data-path", "b", "--test-data-path", "c"],
        ["--sft", "--mock-data", "--sft-mock-dataset-config-json", "{}"],
        ["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss"],
        ["--use-varlen-dataset", "--data-path", "a", "--varlen-sbhd-validation"],
        ["--fim-data", "--data-path", "a", "--fim-rate", "0.9"],
        ["--data-path", "a", "--data-cache-path", "/c", "--dataloader-fast-cache-load"],
        ["--data-path", "a", "--dataloader-inter-document-masking", "--reset-position-ids"],
        ["--data-path", "a", "--phase-transition-iterations", "5,1", "--dataloader-type", "cyclic"],
        ["--data-path", "a", "--no-data-sharding", "--no-mmap-bin-files", "--num-workers", "0"],
        ["--mock-data", "--no-create-attention-mask-in-dataloader", "--eod-mask-loss"],
    ],
)
def test_declaration_validation_accepts_everything_validate_args_accepts(extra):
    args = _validated(extra)
    declared = gpt_dataset_config_from_args(args)
    declared.validate()
    assert declared.create_attention_mask == args.create_attention_mask_in_dataloader
    assert declared.dataloader_type == args.dataloader_type
