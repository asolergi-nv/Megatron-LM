# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Declarative dataset and dataloader configuration owned by ``PretrainConfigContainer``.

Objects here are *declarations*: what the user asked for, with no tokenizer, no file reads and no
derived state, so they serialize and compare cleanly. ``GPTDatasetConfig.to_core_config()`` turns a
declaration plus a ``DatasetRuntimeContext`` into the Megatron Core runtime object that
``BlendedMegatronDatasetBuilder`` consumes; the declaration itself is never mutated by that step.
``GPTFIMDatasetConfig`` declares the fim family and carries the FIM-only fields; the declaration
class follows the family (``declaration_class_for_dataset_type``).

Two read-only views of a declaration, neither of which mutates it:

* the *normalized* view applies the spelling normalizations only (``sequence_length`` folded into
  ``seq_length``, a string ``data_path`` split into a list, ``phase_transition_iterations`` as a
  sorted list, an unset ``dataloader_type`` as ``"single"``); it is what ``finalize()`` writes into
  the declaration and what ``to_cfg_dict()`` serializes, so a serialized declaration stays editable:
  nothing inferred from another field is frozen into it;
* the *resolved* view adds the defaults that depend on other fields (the legacy ``split`` when only
  ``data_path`` is given, ``create_attention_mask`` under inter-document masking, the SFT mock
  distribution from ``seq_length``, ``persistent_workers`` with no workers); ``validate()`` checks
  it and ``to_core_config()`` builds from it, and it is never written back.

A declaration built from the CLI copies the validated ``args``, where ``validate_args`` has already
materialized those defaults; such a declaration carries them explicitly.

Lifecycle, following ``ConfigContainerBase``: ``validate()`` is read-only and may run repeatedly;
``finalize()`` applies the spelling normalizations once, from ``PretrainConfigContainer._finalize()``.
Conversion and serialization behave identically before and after ``finalize()``.
"""

import json
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Collection, Literal

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig

DATASET_TYPES = ("gpt", "fim", "varlen", "sft")
DATALOADER_TYPES = ("single", "cyclic", "external")

LEGACY_SPLIT_DEFAULT = "969, 30, 1"


def _warn(message: str) -> None:
    from megatron.training.utils import warn_rank_0

    warn_rank_0(message)


@dataclass(kw_only=True)
class DataloaderConfig:
    """Settings for building torch DataLoaders on top of a dataset."""

    dataloader_type: Literal["single", "cyclic", "external"] | None = None
    """Single pass vs multiple pass data loader. 'external' passes the dataset through unchanged.
    Unset selects 'single'."""

    num_workers: int = 2
    """Dataloader number of workers."""

    data_sharding: bool = True
    """Shard the dataset per data-parallel rank before shuffling (cyclic sampler only)."""

    pin_memory: bool = True
    """Pin host memory in the torch DataLoader. Declared only for now: the training loop keeps its
    built-in value until the loader consumers read this config."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs. Forced to False when num_workers == 0. Declared
    only for now: the training loop keeps its built-in value until the loader consumers read it."""

    drop_last: bool = True
    """Drop the last incomplete micro-batch. Declared only for now: the training loop keeps its
    built-in value until the loader consumers read this config."""

    def _normalized_loader_fields(self) -> dict[str, Any]:
        """Loader settings after the spelling normalization (unset type -> "single"); never
        mutates ``self``."""
        normalized = {
            "dataloader_type": self.dataloader_type,
            "num_workers": self.num_workers,
            "data_sharding": self.data_sharding,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": self.drop_last,
        }
        if normalized["dataloader_type"] is None:
            normalized["dataloader_type"] = "single"
        return normalized

    def _resolved_loader_fields(self) -> dict[str, Any]:
        """Loader settings as a consumer applies them: ``persistent_workers`` needs workers. Never
        written back into the declaration, so editing ``num_workers`` later keeps its effect."""
        resolved = self._normalized_loader_fields()
        if resolved["num_workers"] == 0 and resolved["persistent_workers"]:
            resolved["persistent_workers"] = False
        return resolved

    def validate(self) -> None:
        """Check the loader settings without mutation."""
        resolved = self._resolved_loader_fields()
        if resolved["dataloader_type"] not in DATALOADER_TYPES:
            raise ValueError(
                f"dataloader_type must be one of {DATALOADER_TYPES}, got {self.dataloader_type!r}."
            )

    def finalize(self) -> None:
        """Apply the loader spelling normalization to the declaration (idempotent)."""
        for name, value in self._normalized_loader_fields().items():
            setattr(self, name, value)


@dataclass(frozen=True)
class DatasetRuntimeContext:
    """Runtime inputs needed to turn a declarative dataset config into a Megatron Core config.

    Training builds it with ``PretrainConfigContainer.dataset_runtime_context()``; standalone tools
    build it with ``from_args()``. A value set here wins over the same field declared on the config;
    ``None`` keeps the declared value (Megatron-Bridge recipes declare ``random_seed`` directly).
    """

    tokenizer: Any
    random_seed: int | None = None
    data_parallel_size: int | None = None
    context_parallel_size: int | None = None
    sequence_parallel_size: int | None = None
    hybrid_context_parallel: bool | None = None
    full_validation: bool | None = None
    multiple_validation_sets: bool | None = None

    @classmethod
    def from_args(cls, args: Namespace, tokenizer: Any) -> "DatasetRuntimeContext":
        """Context for tools that run from legacy arguments without a config container."""
        return cls(
            tokenizer=tokenizer,
            random_seed=getattr(args, "seed", None),
            data_parallel_size=getattr(args, "data_parallel_size", None),
            context_parallel_size=getattr(args, "context_parallel_size", None),
            sequence_parallel_size=getattr(args, "tensor_model_parallel_size", 1)
            * int(bool(getattr(args, "sequence_parallel", False))),
            hybrid_context_parallel=getattr(args, "hybrid_context_parallel", None),
            full_validation=getattr(args, "full_validation", None),
            multiple_validation_sets=getattr(args, "multiple_validation_sets", None),
        )


# Legacy CLI spellings that differ from the field names or from what ``ArgumentGroupFactory`` would
# infer. Applied last by ``DatasetArgumentGroupFactory`` (flag generation) and read by
# ``gpt_dataset_config_from_args`` (args -> config) through the ``dest`` entries.
LEGACY_ARG_MAP: dict[str, dict[str, Any]] = {
    "path_to_cache": dict(arg_names=["--data-cache-path"], dest="data_cache_path", type=None),
    "fast_cache_load": dict(
        arg_names=["--dataloader-fast-cache-load"], dest="dataloader_fast_cache_load"
    ),
    "defer_npy_index_mmap": dict(
        arg_names=["--dataloader-defer-npy-index-mmap"], dest="dataloader_defer_npy_index_mmap"
    ),
    "create_attention_mask": dict(
        arg_names=["--no-create-attention-mask-in-dataloader"],
        dest="create_attention_mask_in_dataloader",
    ),
    "inter_document_masking": dict(
        arg_names=["--dataloader-inter-document-masking"], dest="dataloader_inter_document_masking"
    ),
    # These legacy flags register no argparse ``type`` (values stay the parsed strings).
    "data_path": dict(nargs="*", type=None),
    "train_data_path": dict(nargs="*", type=None),
    "valid_data_path": dict(nargs="*", type=None),
    "test_data_path": dict(nargs="*", type=None),
    "per_dataset_sequences_path": dict(type=None),
    # The parser leaves --dataset-type unset so that an explicit value is distinguishable from the
    # dataclass default "gpt"; dataset_type_from_args maps None to "gpt" when no alias flag is set.
    "dataset_type": dict(default=None),
}

# Legacy family flags kept as CLI aliases of dataset_type: flag -> (dest, dataset_type value).
# The order is today's provider precedence (sft, then varlen, then fim).
LEGACY_ALIAS_FLAGS: dict[str, tuple[str, str]] = {
    "--sft": ("sft", "sft"),
    "--use-varlen-dataset": ("use_varlen_dataset", "varlen"),
    "--fim-data": ("fim_data", "fim"),
}

LEGACY_ALIAS_HELP: dict[str, str] = {
    "--sft": "Megatron SFT training. Alias of --dataset-type sft.",
    "--use-varlen-dataset": (
        "Train with VarlenDataset, a variable-length packed (THD) dataset that consumes "
        "instruction-tuning data from a HuggingFace Hub repo id, a local parquet file, or a local "
        "jsonl file. Schema (alpaca / sharegpt / openai-messages) is auto-detected from the dataset "
        "columns. Mutually exclusive with --sft. Auto-picks a sequence packing scheduler when none "
        "is given: dp_balanced. Combine with --mock-data for a synthetic lognormal sequence-length "
        "distribution; see --varlen-mock-dataset-config-json. Alias of --dataset-type varlen."
    ),
    "--fim-data": "Whether to use the FIM dataset. Alias of --dataset-type fim.",
}

# Not CLI flags: runtime-context inputs, Megatron Core spellings, structured values, and loader
# knobs the legacy CLI never exposed (YAML / Python only).
DATASET_CLI_EXCLUDE = [
    "tokenizer",
    "random_seed",
    "data_parallel_size",
    "context_parallel_size",
    "sequence_parallel_size",
    "hybrid_context_parallel",
    "full_validation",
    "multiple_validation_sets",
    "sequence_length",
    "blend",
    "blend_per_split",
    "sequences_per_dataset",
    "fim_extra_tokens",
    "drop_last_partial_validation_sequence",
    "add_extra_token_to_sequence",
    "pin_memory",
    "persistent_workers",
    "drop_last",
]

_CONTEXT_FIELDS = (
    "random_seed",
    "data_parallel_size",
    "context_parallel_size",
    "sequence_parallel_size",
    "hybrid_context_parallel",
    "full_validation",
    "multiple_validation_sets",
)
_MCORE_RUNTIME_DEFAULTS = {
    "data_parallel_size": 1,
    "sequence_parallel_size": 0,
    "hybrid_context_parallel": False,
}
_NOT_SERIALIZED = frozenset({"tokenizer", "sequence_length"})


def dataset_type_from_args(args: Namespace, *, warn: bool = True) -> str:
    """Resolve the dataset family from ``--dataset-type`` and the legacy alias flags.

    The rule is stable under repetition, so ``validate_args`` (which writes the result back with
    ``project_dataset_type_into_args``) and the config factories agree: an explicit ``dataset_type``
    is consistent with alias flags when it equals their legacy resolution (``sft`` over ``varlen``
    over ``fim``, as today's providers choose); any other disagreement is an error, a combination
    that could not be written before. ``--sft`` with ``--use-varlen-dataset`` is rejected as
    ``validate_args`` does. Several alias flags warn once (``warn=True``) where today the losers
    are silently ignored.
    """
    explicit = getattr(args, "dataset_type", None)
    aliases = [
        family for _flag, (dest, family) in LEGACY_ALIAS_FLAGS.items() if getattr(args, dest, False)
    ]
    if "sft" in aliases and "varlen" in aliases:
        raise ValueError(
            "--use-varlen-dataset and --sft are mutually exclusive; both select the "
            "packed-sequence dataset family. Pick one."
        )
    if explicit is not None and explicit not in DATASET_TYPES:
        raise ValueError(f"--dataset-type must be one of {DATASET_TYPES}, got {explicit!r}.")
    legacy = aliases[0] if aliases else None
    if explicit is not None and legacy is not None and legacy != explicit:
        raise ValueError(
            f"--dataset-type {explicit} contradicts the legacy flag(s) selecting {aliases}."
        )
    if warn and len(aliases) > 1:
        _warn(f"{aliases} were all requested; '{aliases[0]}' wins, as in today's providers.")
    if explicit is not None:
        return explicit
    return legacy or "gpt"


def project_dataset_type_into_args(args: Namespace) -> str:
    """Resolve the family once and write it back into ``args`` so unmigrated consumers follow it.

    Called early in ``validate_args``, before the raw-flag checks and normalizations that read
    ``args.sft`` / ``args.use_varlen_dataset`` / ``args.fim_data``: ``--dataset-type varlen`` then
    auto-selects the packing scheduler, collates and batches exactly like ``--use-varlen-dataset``.
    Flags the user passed are never cleared, so legacy namespaces are unchanged field by field, and
    calling this again (or ``dataset_type_from_args``) returns the same family.
    """
    resolved = dataset_type_from_args(args)
    args.dataset_type = resolved
    for _flag, (dest, family) in LEGACY_ALIAS_FLAGS.items():
        if family == resolved:
            setattr(args, dest, True)
    return resolved


def _as_blend(blend: Any) -> tuple[list[str], list[float] | None] | None:
    """YAML gives lists; Megatron Core expects a (prefixes, weights) tuple."""
    if blend is None:
        return None
    prefixes, weights = blend
    return (list(prefixes), None if weights is None else list(weights))


@dataclass(kw_only=True)
class GPTDatasetConfig(MCoreGPTDatasetConfig, DataloaderConfig):
    """Declarative GPT dataset configuration (``PretrainConfigContainer.dataset``).

    Two classes carry this name, for two roles. The Megatron Core class
    (``megatron.core.datasets.gpt_dataset.GPTDatasetConfig``) is the runtime object the dataset
    builder consumes: it needs a tokenizer instance to construct and derives ``split_matrix``,
    ``mock`` and ``token_dtype_code`` eagerly. This class is the declaration: what the user asked
    for, built at parse time, from YAML or in Python, without a tokenizer, serializable, and
    identical before and after ``finalize()``. It inherits the Megatron Core field names so that
    CLI, YAML, Python and Megatron-Bridge recipes declare the same thing, and it is never handed to
    dataset code: ``to_core_config()`` is the only place the two meet; it builds a fresh runtime
    object from the declaration and a ``DatasetRuntimeContext``. Modules that see both classes
    import the Megatron Core one as ``MCoreGPTDatasetConfig``. The fim family is declared with the
    ``GPTFIMDatasetConfig`` subclass, which carries the FIM-only fields (``as_family()`` switches
    between the two).
    """

    # Megatron Core fields redeclared so Python / YAML construction gets the CLI defaults, and so
    # the runtime-context inputs are optional at declaration time.
    random_seed: int | None = None
    """Seed for dataset shuffling. Supplied by cfg.rng through the runtime context; may be declared
    directly by callers that have no container."""

    sequence_length: int | None = None
    """Megatron Core spelling of seq_length, accepted as an input alias and folded into seq_length."""

    reset_position_ids: bool = False
    """Reset position ids after end-of-document token."""

    reset_attention_mask: bool = False
    """Reset self attention mask after end-of-document token."""

    eod_mask_loss: bool = False
    """Mask loss for the end of document tokens."""

    allow_ambiguous_pad_tokens: bool = False
    """Whether to prevent pad tokens already present in the dataset from being masked out when the
    pad token incorrectly shares the same id with other special tokens in the tokenizer. Note that
    this argument has no effect when the tokenizer correctly provides a unique id for the pad.
    Masking out such ambiguous pad tokens results in training instability. Such a scenario is best
    resolved by fixing the tokenizer; leaving this option as False provides a workaround. When left
    to the default of False, any token ids that collide with the pad token id - as provided by the
    tokenizer - will not be masked out of the loss calculation: it cannot be determined whether they
    are truly pad. If instead this argument is set, the training flow will treat all tokens that
    share the same id as the pad token as true pad tokens, potentially causing severe training
    instability."""

    data_parallel_size: int | None = None
    """Runtime-context input: data-parallel world size."""

    sequence_parallel_size: int | None = None
    """Runtime-context input: tensor-parallel size when sequence parallelism is on, else 0."""

    hybrid_context_parallel: bool | None = None
    """Runtime-context input, owned by the model config."""

    # Declared by the user. CLI spellings follow LEGACY_ARG_MAP; everything else is the field name.
    seq_length: int | None = None
    """Maximum sequence length to process."""

    data_path: str | list[str] | None = field(
        default=None, metadata={"argparse_meta": {"nargs": "*"}}
    )
    """The weight and prefix list for a set of train, validation, and test datasets which split
    according to --split. The accepted formats are: (1) a single prefix, (2) a list of weight prefix
    pairs e.g. weight1 prefix1 weight2 prefix2, (3) a list of prefixes e.g. prefix1 prefix2. For (3),
    weights are inferred from the lengths of the contributing datasets. A single string is split on
    whitespace."""

    train_data_path: list[str] | None = field(
        default=None, metadata={"argparse_meta": {"nargs": "*"}}
    )
    """The weight and prefix list for an independent train dataset. Follows the same pattern rules
    as --data-path."""

    valid_data_path: list[str] | None = field(
        default=None, metadata={"argparse_meta": {"nargs": "*"}}
    )
    """The weight and prefix list for an independent validation dataset. Follows the same pattern
    rules as --data-path."""

    test_data_path: list[str] | None = field(
        default=None, metadata={"argparse_meta": {"nargs": "*"}}
    )
    """The weight and prefix list for an independent test dataset. Follows the same pattern rules
    as --data-path."""

    data_args_path: str | None = None
    """Path to data-args. Instead of feeding `--data-path` with weighted dataset, we pass in a file
    path from which we read that argument. This is useful when the list of data is too big."""

    per_split_data_args_path: str | None = None
    """Path to per-split-data-args. Instead of feeding `--(train|valid|test)-data-path` with weighted
    dataset, we pass in a file path from which we read those arguments. This is useful when the list
    of data is too big. Format is a json file with `train`, `valid`, `test` keys."""

    per_dataset_sequences_path: str | None = None
    """Path to a json file with the sequences per dataset. Check the
    tools/build_sequences_per_dataset.py script to build this file."""

    phase_transition_iterations: str | list[int] | None = field(
        default=None, metadata={"argparse_meta": {"type": str}}
    )
    """Comma-separated list of iterations where phase transitions occur, for example '1000,2000'.
    Normalized to a sorted list of integers."""

    mock_data: bool = False
    """Skip data loading and validation and opt for artificial generation of mock data when an
    implementation is available."""

    dataset_type: Literal["gpt", "fim", "varlen", "sft"] = "gpt"
    """Dataset family the GPT provider builds: gpt (GPTDataset), fim (GPTFIMDataset, declared with
    GPTFIMDatasetConfig), varlen (VarlenDataset, variable-length packed THD) or sft (SFTDataset,
    packed conversations). --sft, --use-varlen-dataset and --fim-data are kept as aliases."""

    def __post_init__(self) -> None:
        """Megatron Core's post-init needs a tokenizer; it runs on the runtime object instead."""

    @property
    def is_packed_sequence(self) -> bool:
        """Packed (THD) samples: built on every pipeline stage, and get_batch expects cu_seqlens."""
        return self.dataset_type == "sft" or (
            self.dataset_type == "varlen" and not self.varlen_sbhd_validation
        )

    def _check_family_matches_class(self, dataset_type: str) -> None:
        """The fim family is declared with ``GPTFIMDatasetConfig`` and nothing else is.

        The FIM-only fields live on that subclass, so the class and the selector must agree. Checked
        by ``validate()`` and again by ``to_core_config()``, which tools call without a container.
        """
        declares_fim = isinstance(self, GPTFIMDatasetConfig)
        if dataset_type == "fim" and not declares_fim:
            raise ValueError(
                "dataset_type='fim' must be declared with GPTFIMDatasetConfig (--fim-data / "
                "--dataset-type fim build one); the FIM fields live on that class."
            )
        if declares_fim and dataset_type != "fim":
            raise ValueError(
                f"GPTFIMDatasetConfig declares the fim family; got dataset_type={dataset_type!r}. "
                "Use as_family() or declare a GPTDatasetConfig for another family."
            )

    def as_family(self, dataset_type: str) -> "GPTDatasetConfig":
        """A new declaration of another family with the same shared fields; ``self`` is unchanged.

        Entry points that build a different family than the one requested (hybrid builds gpt for
        a fim request) use this instead of ``dataclasses.replace(..., dataset_type=...)``: the
        declaration class follows the family (``GPTFIMDatasetConfig`` for fim), so the FIM-only
        fields are dropped when leaving the fim family and take their defaults when entering it.
        """
        target = declaration_class_for_dataset_type(dataset_type)
        shared = {f.name for f in fields(self) if f.init}
        kwargs: dict[str, Any] = {}
        for f in fields(target):
            if not f.init or f.name not in shared or f.name == "dataset_type":
                continue
            value = getattr(self, f.name)
            kwargs[f.name] = value if f.name == "tokenizer" else deepcopy(value)
        return target(dataset_type=dataset_type, **kwargs)

    def _normalized_fields(self) -> dict[str, Any]:
        """Declared inputs after the spelling normalizations; never mutates ``self``.

        These are representation changes only (the ones ``validate_args`` applies to the legacy
        namespace): nothing is inferred from another field, so the normalized declaration means
        exactly what the user declared. Raises ``ValueError`` on inputs that cannot be normalized.
        """
        n = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        n.update(self._normalized_loader_fields())
        if n["sequence_length"] is not None:
            if n["seq_length"] is not None and n["seq_length"] != n["sequence_length"]:
                raise ValueError(
                    f"seq_length={n['seq_length']} and sequence_length={n['sequence_length']} "
                    "disagree; declare one of them."
                )
            n["seq_length"], n["sequence_length"] = n["sequence_length"], None
        if isinstance(n["data_path"], str):
            n["data_path"] = n["data_path"].split()
        if isinstance(n["phase_transition_iterations"], str) and n["phase_transition_iterations"]:
            try:
                n["phase_transition_iterations"] = sorted(
                    int(x.strip()) for x in n["phase_transition_iterations"].split(",")
                )
            except ValueError as error:
                raise ValueError(
                    "phase_transition_iterations must be a comma-separated list of integers, got "
                    f"{self.phase_transition_iterations!r}."
                ) from error
        return n

    def _resolved_fields(self) -> dict[str, Any]:
        """The normalized view plus the defaults that depend on other fields; never mutates
        ``self`` and is never written back into it.

        The defaults are the ones ``validate_args`` materializes in the legacy namespace (the
        legacy ``split`` when only ``data_path`` is given, no attention mask under inter-document
        masking, the lognormal SFT mock distribution from ``seq_length``) plus the loader rule for
        ``persistent_workers``. Kept out of the declaration so that editing the controlling field
        (``seq_length``, ``inter_document_masking``, ``num_workers``) after a round trip has the
        same effect as on a fresh declaration.
        """
        n = self._normalized_fields()
        n.update(self._resolved_loader_fields())
        if n["data_path"] is not None and n["split"] is None:
            n["split"] = LEGACY_SPLIT_DEFAULT
        if n["inter_document_masking"] and n["create_attention_mask"]:
            # The dataset omits attention_mask when inter-document masking is enabled.
            n["create_attention_mask"] = False
        if (
            n["dataset_type"] == "sft"
            and n["mock_data"]
            and n["sft_mock_dataset_config_json"] is None
            and n["seq_length"] is not None
        ):
            n["sft_mock_dataset_config_json"] = json.dumps(
                {
                    "mode": "distribution",
                    "type": "lognormal",
                    "min_seq_len": n["seq_length"] // 2,
                    "max_seq_len": n["seq_length"],
                    "mean_seq_len": n["seq_length"] // 4 * 3,
                    "lognormal_sigma": 1.1,
                }
            )
        return n

    def validate(self) -> None:
        """Section-local checks on the resolved view; no tokenizer, no file access, no mutation.

        The checks are the ones ``validate_args`` applies to the data arguments today, plus the
        spelling rules that only Python / YAML declarations can break, plus one gap closed here:
        ``mock_data`` next to a file-backed source (``data_args_path`` /
        ``per_split_data_args_path``), which ``validate_args`` accepts and no entry point could
        build. Cross-section checks (parallelism, schedulers) stay in ``validate_args``; the FIM
        checks live on ``GPTFIMDatasetConfig.validate()``.
        """
        n = self._resolved_fields()
        if n["dataloader_type"] not in DATALOADER_TYPES:
            raise ValueError(
                f"dataloader_type must be one of {DATALOADER_TYPES}, got {self.dataloader_type!r}."
            )
        if n["dataset_type"] not in DATASET_TYPES:
            raise ValueError(
                f"dataset_type must be one of {DATASET_TYPES}, got {self.dataset_type!r}."
            )
        self._check_family_matches_class(n["dataset_type"])
        per_split_paths = (n["train_data_path"], n["valid_data_path"], n["test_data_path"])
        uses_per_split_paths = any(p is not None for p in per_split_paths)
        # One way of declaring each data source (a serialized config cannot carry two).
        if n["data_path"] is not None and n["data_args_path"] is not None:
            raise ValueError("Use only one of data_path and data_args_path.")
        if uses_per_split_paths and n["per_split_data_args_path"] is not None:
            raise ValueError(
                "Use only one of per_split_data_args_path and the per-split data paths."
            )
        if n["blend"] is not None and (
            n["data_path"] is not None or n["data_args_path"] is not None
        ):
            raise ValueError("Declare either blend or data_path / data_args_path, not both.")
        if n["blend_per_split"] is not None and (
            uses_per_split_paths or n["per_split_data_args_path"] is not None
        ):
            raise ValueError(
                "Declare either blend_per_split or the per-split data paths, not both."
            )
        if n["sequences_per_dataset"] is not None and n["per_dataset_sequences_path"] is not None:
            raise ValueError(
                "Declare either sequences_per_dataset or per_dataset_sequences_path, not both."
            )
        # The validate_args count (mock data, a --data-path style source, a per-split source), with
        # the file-backed spellings of the last two counted as the sources they are.
        uses_blend = (
            bool(n["data_path"]) or n["data_args_path"] is not None or n["blend"] is not None
        )
        uses_per_split = (
            any(per_split_paths)
            or n["per_split_data_args_path"] is not None
            or n["blend_per_split"] is not None
        )
        if int(n["mock_data"]) + int(uses_blend) + int(uses_per_split) > 1:
            raise ValueError("A single data source must be provided in training mode, else None")
        if n["num_dataset_builder_threads"] <= 0:
            raise ValueError("num_dataset_builder_threads must be positive.")
        if n["dataset_type"] == "varlen" and n["varlen_sbhd_validation"] and n["mock_data"]:
            raise ValueError(
                "--varlen-sbhd-validation is not supported with --mock-data; "
                "SBHD validation requires a real dataset."
            )

    def finalize(self) -> None:
        """Write the normalized view into the declaration (idempotent).

        Called once per run by ``PretrainConfigContainer._finalize()``; conversion and
        serialization do not depend on it having run. The field-dependent defaults stay out of the
        declaration (``_resolved_fields``); the legacy ``split`` default is announced here, once,
        as ``validate_args`` announces it for the CLI.
        """
        normalized = self._normalized_fields()
        if normalized["data_path"] is not None and normalized["split"] is None:
            _warn(
                "Please specify --split when using --data-path. Using legacy default value "
                f'of "{LEGACY_SPLIT_DEFAULT}"'
            )
        for name, value in normalized.items():
            setattr(self, name, value)

    def _core_kwargs(
        self, context: DatasetRuntimeContext, *, exclude: Collection[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Constructor keywords of the Megatron Core config, and the resolved view they came from.

        Shared by ``to_core_config()`` and the ``GPTFIMDatasetConfig`` override, so that each
        runtime object is constructed exactly once from one set of keywords.
        """
        from megatron.training.utils import get_blend_and_blend_per_split

        if context.tokenizer is None:
            raise ValueError("A built tokenizer is required to build the dataset config.")
        n = self._resolved_fields()
        self._check_family_matches_class(n["dataset_type"])
        if n["seq_length"] is None:
            raise ValueError("seq_length is required to build a GPT dataset config.")
        exclude = set(exclude)
        kwargs: dict[str, Any] = {
            f.name: deepcopy(n[f.name]) for f in fields(MCoreGPTDatasetConfig) if f.init
        }
        unknown = exclude - set(kwargs)
        if unknown:
            raise ValueError(
                f"exclude names must be Megatron Core GPTDatasetConfig constructor keywords, got "
                f"{sorted(unknown)}."
            )
        for name in _CONTEXT_FIELDS:
            value = getattr(context, name)
            if value is None:
                value = n[name]
            if value is None:
                value = _MCORE_RUNTIME_DEFAULTS.get(name)
            kwargs[name] = value
        if kwargs["random_seed"] is None:
            raise ValueError(
                "random_seed must be supplied by the runtime context (cfg.rng.seed) or declared."
            )
        kwargs["tokenizer"] = context.tokenizer
        kwargs["sequence_length"] = n["seq_length"]
        if kwargs["blend"] is None and kwargs["blend_per_split"] is None and not n["mock_data"]:
            # The helper reads the six path inputs from a namespace; feed it the declared ones.
            sources = Namespace(
                data_path=n["data_path"],
                data_args_path=n["data_args_path"],
                per_split_data_args_path=n["per_split_data_args_path"],
                train_data_path=n["train_data_path"],
                valid_data_path=n["valid_data_path"],
                test_data_path=n["test_data_path"],
            )
            kwargs["blend"], kwargs["blend_per_split"] = get_blend_and_blend_per_split(sources)
        kwargs["blend"] = _as_blend(kwargs["blend"])
        if kwargs["blend_per_split"] is not None:
            kwargs["blend_per_split"] = [_as_blend(b) for b in kwargs["blend_per_split"]]
        if (
            kwargs["sequences_per_dataset"] is None
            and n["per_dataset_sequences_path"] is not None
            and "sequences_per_dataset" not in exclude
        ):
            with open(n["per_dataset_sequences_path"], "r") as f:
                kwargs["sequences_per_dataset"] = json.load(f)
        for name in exclude:
            kwargs.pop(name, None)
        return kwargs, n

    def to_core_config(
        self, context: DatasetRuntimeContext, *, exclude: Collection[str] = ()
    ) -> MCoreGPTDatasetConfig:
        """Build the Megatron Core runtime config that ``BlendedMegatronDatasetBuilder`` consumes.

        Non-mutating: ``self`` is never modified and every call returns a new object, so changed
        declarations or runtime inputs are always reflected (it does read the files the declaration
        names: data-args and per-dataset-sequences JSON, through the same ``get_blend_and_blend_per_split``
        helper the legacy factories call). Returns a plain Megatron Core ``GPTDatasetConfig``;
        ``GPTFIMDatasetConfig`` overrides it to return the runtime
        ``megatron.training.datasets.fim_dataset.GPTFIMDatasetConfig``.

        Args:
            context: tokenizer and runtime inputs; a value set on the context wins over the
                declaration.
            exclude: constructor keywords an entry point does not forward. They are left out of the
                call entirely, so Megatron Core's own defaults apply, exactly as for a factory that
                never passed them. Names that are not Megatron Core constructor keywords are
                rejected, so a misspelled policy cannot silently change what is built.
        """
        kwargs, _ = self._core_kwargs(context, exclude=exclude)
        return MCoreGPTDatasetConfig(**kwargs)

    def to_cfg_dict(self) -> dict[str, Any]:
        """Serialize the normalized declared inputs: no tokenizer, no derived state, no
        field-dependent defaults, no unset runtime inputs. Identical before and after
        ``finalize()``. A declaration that cannot be normalized (conflicting lengths, malformed
        phase transitions) raises the ``ValueError`` that ``validate()`` raises, rather than
        serializing a different, valid-looking declaration; ``repr()`` shows the raw fields."""
        from megatron.training.config.container import ConfigContainerBase

        n = self._normalized_fields()
        out: dict[str, Any] = {"_target_": f"{type(self).__module__}.{type(self).__qualname__}"}
        for f in fields(self):
            if f.name.startswith("_") or not f.init or f.name in _NOT_SERIALIZED:
                continue
            value = n[f.name]
            if f.name in _CONTEXT_FIELDS and value is None:
                continue
            if f.name == "sequences_per_dataset" and value is None:
                continue  # a declared mapping is kept; the usual spelling is the path
            out[f.name] = ConfigContainerBase._convert_value_to_dict(value)
        return out


@dataclass(kw_only=True)
class GPTFIMDatasetConfig(GPTDatasetConfig):
    """Declarative GPT dataset configuration with fill-in-the-middle (``dataset_type="fim"``).

    The FIM-only fields live here, mirroring the runtime split (the runtime
    ``megatron.training.datasets.fim_dataset.GPTFIMDatasetConfig`` extends the Megatron Core config)
    and Megatron-Bridge's ``GPTFIMDatasetConfig``. ``--fim-data`` / ``--dataset-type fim`` build this
    class, a plain ``GPTDatasetConfig`` cannot declare the fim family, and this class declares no
    other; the data argument group is generated from it, so one CLI group covers both classes.
    """

    dataset_type: Literal["gpt", "fim", "varlen", "sft"] = "fim"

    fim_rate: float = 0.5
    """Probability to convert a training sample into a FIM format."""

    fim_spm_rate: float = 0.5
    """Probability that the a FIM sample uses the SPM format over the PSM format."""

    fim_split_sample: str | None = None
    """String around which to split the sample for FIM."""

    fim_fragment_rate: float | None = None
    """Rate of FIM on each fragment when --fim-split-sample is not None."""

    fim_no_prefix: str | None = None
    """Do not apply FIM to fragments that start with this prefix"""

    fim_prefix_token: str = "<fim_prefix>"
    """FIM prefix token"""

    fim_middle_token: str = "<fim_middle>"
    """FIM middle token"""

    fim_suffix_token: str = "<fim_suffix>"
    """FIM suffix token"""

    fim_pad_token: str = "<fim_pad>"
    """FIM PAD token"""

    fim_eod_token: str = "<|endoftext|>"
    """FIM EOD token"""

    fim_extra_tokens: dict[str, str] | None = None
    """Explicit FIM token mapping with keys prefix, middle, suffix, pad and eod. When None it is
    built from the five fim_*_token fields."""

    def validate(self) -> None:
        """The shared checks, then the FIM checks ``validate_args`` applies to ``--fim-data``."""
        super().validate()
        n = self._resolved_fields()
        if n["mock_data"]:
            raise ValueError("Mock dataset is not supported with FIM dataset.")
        if not n["fim_rate"]:
            raise ValueError("--fim-rate should be specified.")
        if not n["fim_spm_rate"]:
            raise ValueError("--fim-spm-rate should be specified.")
        tokens = (
            n["fim_prefix_token"],
            n["fim_middle_token"],
            n["fim_suffix_token"],
            n["fim_pad_token"],
            n["fim_eod_token"],
        )
        if n["fim_extra_tokens"] is None and any(token is None for token in tokens):
            raise ValueError("FIM extra tokens should be specified.")

    def to_core_config(
        self, context: DatasetRuntimeContext, *, exclude: Collection[str] = ()
    ) -> MCoreGPTDatasetConfig:
        """Build the runtime ``megatron.training.datasets.fim_dataset.GPTFIMDatasetConfig``.

        The shared keywords come from the base class, the FIM extras are added, and the runtime
        object is constructed once. ``fim_extra_tokens`` defaults to the five ``fim_*_token`` fields.
        """
        from megatron.training.datasets.fim_dataset import (
            GPTFIMDatasetConfig as RuntimeGPTFIMDatasetConfig,
        )

        kwargs, n = self._core_kwargs(context, exclude=exclude)
        fim_extra_tokens = deepcopy(n["fim_extra_tokens"]) or {
            "prefix": n["fim_prefix_token"],
            "middle": n["fim_middle_token"],
            "suffix": n["fim_suffix_token"],
            "pad": n["fim_pad_token"],
            "eod": n["fim_eod_token"],
        }
        return RuntimeGPTFIMDatasetConfig(
            **kwargs,
            fim_rate=n["fim_rate"],
            fim_spm_rate=n["fim_spm_rate"],
            fim_extra_tokens=fim_extra_tokens,
            fim_split_sample=n["fim_split_sample"],
            fim_fragment_rate=n["fim_fragment_rate"],
            fim_no_prefix=n["fim_no_prefix"],
        )


def declaration_class_for_dataset_type(dataset_type: str) -> type[GPTDatasetConfig]:
    """The declaration class of a dataset family: ``GPTFIMDatasetConfig`` for fim, else the base.

    The CLI factory and ``as_family()`` select the class with it, so the class and the selector
    always agree.
    """
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}.")
    return GPTFIMDatasetConfig if dataset_type == "fim" else GPTDatasetConfig
