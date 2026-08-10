"""
Configuration system for the preprocessing pipeline.

All configurable parameters -- image processing, tokenization, dataset source,
shard layout, pipeline behavior -- are defined as frozen dataclasses and loaded
from a single YAML file.  Nothing is hardcoded: behavior is changed by editing
config, not code.

The config hierarchy is:
    PipelineConfig
        DatasetConfig      -- what to ingest and how to map columns
        ImageConfig         -- resize, crop, normalization, color space
        TokenizerConfig     -- model name, sequence length, padding
        ShardConfig         -- output directory, shard size, compression

Frozen dataclasses prevent accidental mutation after loading, which matters
when configs are shared across workers in a multiprocessing pipeline.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config validation error
# ---------------------------------------------------------------------------
class ConfigError(ValueError):
    """Raised when a configuration file is invalid, missing, or malformed."""


# ---------------------------------------------------------------------------
# Utility: deep merge two dicts
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *base*, returning a new dict.

    Nested dicts are merged recursively.  Non-dict values in overrides
    replace the corresponding value in base.  Keys only in base are kept.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Utility: convert YAML lists to tuples for immutable config fields
# ---------------------------------------------------------------------------
_TUPLE_FIELDS = frozenset({
    "target_size",
    "normalization_mean",
    "normalization_std",
})


def _listify_to_tuples(data: dict[str, Any]) -> dict[str, Any]:
    """Convert lists to tuples for fields that the dataclasses expect as tuples.

    YAML produces lists, but our frozen dataclasses use tuples for
    immutable sequences like normalization stats and image dimensions.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _listify_to_tuples(value)
        elif isinstance(value, list) and key in _TUPLE_FIELDS:
            result[key] = tuple(value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the dataset source and column mapping.

    Attributes
    ----------
    hf_dataset_id : str
        HuggingFace dataset identifier.
    hf_config_name : str
        HuggingFace dataset configuration/subset name.
    split : str
        Dataset split to load.
    column_mapping : dict[str, str]
        Maps canonical field names to dataset-specific column names.
        Keys: "image", "question", "answer", "id", "reasoning".
        Values: the actual column names in the dataset.
    max_samples : int | None
        If set, only ingest this many samples (useful for debugging).
    streaming : bool
        If True, use HuggingFace streaming mode.
    """

    hf_dataset_id: str = "trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning"
    hf_config_name: str = "preview"
    split: str = "train"
    column_mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "image": "image",
        "question": "question",
        "answer": "model_answer",
        "id": "id",
        "reasoning": "model_reasoning",
    })
    max_samples: int | None = None
    streaming: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Image configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ImageConfig:
    """Configuration for image preprocessing.

    Attributes
    ----------
    target_size : tuple[int, int]
        Target (height, width) after resize/crop.
    resize_strategy : str
        One of "resize_and_pad", "center_crop", "resize".
    color_space : str
        Target color space ("RGB", "BGR", "L").
    normalization_mean : tuple[float, ...]
        Per-channel mean for normalization (ImageNet default).
    normalization_std : tuple[float, ...]
        Per-channel std for normalization (ImageNet default).
    interpolation : str
        Resize interpolation method ("bicubic", "bilinear", "lanczos", "nearest").
    pad_value : int
        Pixel value used for padding (0-255).
    """

    target_size: tuple[int, int] = (384, 384)
    resize_strategy: str = "resize_and_pad"
    color_space: str = "RGB"
    normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    interpolation: str = "bicubic"
    pad_value: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Tokenizer configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for text tokenization.

    Attributes
    ----------
    model_name_or_path : str
        Pretrained tokenizer identifier (HuggingFace hub or local path).
    max_length : int
        Maximum sequence length after tokenization.
    padding : str
        Padding strategy: "max_length", "longest", or "do_not_pad".
    truncation : bool
        Whether to truncate sequences exceeding max_length.
    trust_remote_code : bool
        Whether to allow custom tokenizer code (required by some models
        like jais-13b-chat).
    add_special_tokens : bool
        Whether the tokenizer should add BOS/EOS tokens.
    """

    model_name_or_path: str = "inceptionai/jais-13b-chat"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    trust_remote_code: bool = True
    add_special_tokens: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizerConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Shard configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ShardConfig:
    """Configuration for binary shard output.

    Attributes
    ----------
    output_dir : str
        Directory where shards are written.
    shard_size_mb : int
        Target shard size in megabytes.
    max_samples_per_shard : int | None
        Maximum number of samples per shard file.  When reached the
        current shard is finalized and a new one is opened.  None means
        no sample-count limit (only size-based rotation applies).
    compression : str | None
        Compression algorithm (None for uncompressed, "lz4" for LZ4).
    alignment_bytes : int
        Byte alignment for memory-mapped access in C++.
    """

    output_dir: str = "./output/shards"
    shard_size_mb: int = 256
    max_samples_per_shard: int | None = None
    compression: str | None = None
    alignment_bytes: int = 64

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration.

    Composes all sub-configs into a single immutable object that fully
    describes a preprocessing run.

    Attributes
    ----------
    dataset : DatasetConfig
        Dataset source and column mapping.
    image : ImageConfig
        Image preprocessing parameters.
    tokenizer : TokenizerConfig
        Text tokenization parameters.
    shard : ShardConfig
        Binary shard output parameters.
    shuffling : str
        Sample shuffling strategy: ``"none"`` (default, no shuffling),
        ``"local"`` (in-shard shuffling), or ``"global"``
        (cross-shard shuffling of the entire dataset).
    num_workers : int
        Number of parallel preprocessing workers (0 = main thread only).
    log_level : str
        Logging verbosity.
    seed : int
        Global random seed for reproducibility.
    """

    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    image: ImageConfig = dataclasses.field(default_factory=ImageConfig)
    tokenizer: TokenizerConfig = dataclasses.field(default_factory=TokenizerConfig)
    shard: ShardConfig = dataclasses.field(default_factory=ShardConfig)
    shuffling: str = "none"
    num_workers: int = 4
    log_level: str = "INFO"
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a PipelineConfig from a nested dictionary.

        Sub-dicts keyed "dataset", "image", "tokenizer", and "shard" are
        parsed into their respective typed config dataclasses.
        """
        dataset = DatasetConfig.from_dict(data.get("dataset", {}))
        image = ImageConfig.from_dict(data.get("image", {}))
        tokenizer = TokenizerConfig.from_dict(data.get("tokenizer", {}))
        shard = ShardConfig.from_dict(data.get("shard", {}))

        top_level_keys = {"shuffling", "num_workers", "log_level", "seed"}
        top = {k: v for k, v in data.items() if k in top_level_keys}

        return cls(
            dataset=dataset,
            image=image,
            tokenizer=tokenizer,
            shard=shard,
            **top,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dictionary."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    PipelineConfig
        A fully-typed, immutable configuration object.

    Raises
    ------
    ConfigError
        If the file cannot be read, is not valid YAML, or does not
        contain a top-level mapping.
    """
    path = Path(path)

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Expected a YAML mapping at top level in {path}, got {type(raw).__name__}"
        )

    # Convert YAML lists to tuples where the dataclasses expect them
    raw = _listify_to_tuples(raw)

    try:
        config = PipelineConfig.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Failed to construct PipelineConfig from {path}: {exc}") from exc

    # Validate shuffling parameter
    _VALID_SHUFFLING = {"none", "local", "global"}
    if config.shuffling not in _VALID_SHUFFLING:
        raise ConfigError(
            f"Invalid shuffling value '{config.shuffling}' in {path}. "
            f"Must be one of: {sorted(_VALID_SHUFFLING)}"
        )

    logger.info(
        "Configuration loaded from %s: dataset=%s, split=%s",
        path, config.dataset.hf_dataset_id, config.dataset.split,
    )
    return config
