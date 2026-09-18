"""
Configuration system for the preprocessing pipeline.

All configurable parameters are defined as frozen dataclasses and loaded
from a single YAML file (pipeline.yaml in this directory). Nothing is
hardcoded: behavior is changed by editing config, not code.

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


class ConfigError(ValueError):
    """Raised when a configuration file is invalid, missing, or malformed."""


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *base*, returning a new dict.

    Nested dicts are merged recursively. Non-dict values in overrides
    replace the corresponding value in base. Keys only in base are kept.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


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
        Per-channel mean for normalization.
    normalization_std : tuple[float, ...]
        Per-channel std for normalization.
    interpolation : str
        Resize interpolation method ("bicubic", "bilinear", "lanczos", "nearest").
    pad_value : int
        Pixel value used for padding (0-255).
    max_image_dim : int
        Maximum image dimension for aspect-ratio-preserving resize (v2 format).
    storage_dtype : str
        Storage data type: "float32" (v1) or "uint8" (v2, deferred normalization).
    dynamic_padding : bool
        If True, store per-sample dimensions and pad at batch collation time.
    """

    target_size: tuple[int, int] = (384, 384)
    resize_strategy: str = "resize_and_pad"
    color_space: str = "RGB"
    normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    interpolation: str = "bicubic"
    pad_value: int = 0
    max_image_dim: int = 384
    storage_dtype: str = "float32"
    dynamic_padding: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


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
        Whether to allow custom tokenizer code (required by some models).
    add_special_tokens : bool
        Whether the tokenizer should add BOS/EOS tokens.
    dynamic_text_padding : bool
        If True, tokenize without padding and defer padding to batch
        collation time. Actual token lengths are stored in metadata.
        If False (default), pad every sequence to max_length at
        preprocessing time (static padding).
    """

    model_name_or_path: str = "inceptionai/jais-13b-chat"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    trust_remote_code: bool = True
    add_special_tokens: bool = True
    dynamic_text_padding: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizerConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


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
        Maximum samples per shard file. None means size-based rotation only.
    compression : str | None
        Compression algorithm (None or "lz4").
    alignment_bytes : int
        Byte alignment for memory-mapped access (must be power of 2).
    format_version : int
        Binary shard format version: 1 (legacy float32) or 2 (uint8 + per-sample dims).
    manifest_path : str | None
        Path for the shard manifest JSON output. None disables manifest generation.
    """

    output_dir: str = "./output/shards"
    shard_size_mb: int = 256
    max_samples_per_shard: int | None = None
    compression: str | None = None
    alignment_bytes: int = 64
    format_version: int = 2
    manifest_path: str | None = "./output/shards/manifest.json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardConfig:
        """Construct from a dictionary, ignoring unknown keys."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


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
        Sample shuffling strategy: "none", "local", or "global".
    num_workers : int
        Number of parallel preprocessing workers (0 = main thread only).
    log_level : str
        Logging verbosity.
    seed : int
        Global random seed for reproducibility.
    progress_interval : int
        How often (in samples) to log progress during processing.
    streaming_chunk_size : int
        Default chunk size for the streaming pipeline when
        max_samples_per_shard is not set.
    fail_fast : bool
        If True, abort on first sample processing error.
        If False, skip failed samples and continue.
    """

    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    image: ImageConfig = dataclasses.field(default_factory=ImageConfig)
    tokenizer: TokenizerConfig = dataclasses.field(default_factory=TokenizerConfig)
    shard: ShardConfig = dataclasses.field(default_factory=ShardConfig)
    shuffling: str = "none"
    num_workers: int = 4
    log_level: str = "INFO"
    seed: int = 42
    progress_interval: int = 50
    streaming_chunk_size: int = 500
    fail_fast: bool = False

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

        top_level_keys = {
            "shuffling", "num_workers", "log_level", "seed",
            "progress_interval", "streaming_chunk_size", "fail_fast",
        }
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


_VALID_SHUFFLING = {"none", "local", "global"}
_VALID_RESIZE_STRATEGIES = {"resize_and_pad", "center_crop", "resize"}
_VALID_INTERPOLATIONS = {"bicubic", "bilinear", "lanczos", "nearest"}
_VALID_COLOR_SPACES = {"RGB", "BGR", "L"}
_VALID_STORAGE_DTYPES = {"float32", "uint8"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_COMPRESSIONS = {None, "lz4"}
_VALID_FORMAT_VERSIONS = {1, 2}


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate_config(config: PipelineConfig, path: Path) -> None:
    """Validate all config fields, raising ConfigError on invalid values."""
    if config.shuffling not in _VALID_SHUFFLING:
        raise ConfigError(
            f"Invalid shuffling '{config.shuffling}' in {path}. "
            f"Must be one of: {sorted(_VALID_SHUFFLING)}"
        )

    if config.log_level.upper() not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"Invalid log_level '{config.log_level}' in {path}. "
            f"Must be one of: {sorted(_VALID_LOG_LEVELS)}"
        )

    if config.num_workers < 0:
        raise ConfigError(f"num_workers must be >= 0, got {config.num_workers}")

    if config.progress_interval < 1:
        raise ConfigError(f"progress_interval must be >= 1, got {config.progress_interval}")

    if config.streaming_chunk_size < 1:
        raise ConfigError(
            f"streaming_chunk_size must be >= 1, got {config.streaming_chunk_size}"
        )

    img = config.image
    if img.resize_strategy.lower() not in _VALID_RESIZE_STRATEGIES:
        raise ConfigError(
            f"Invalid resize_strategy '{img.resize_strategy}'. "
            f"Must be one of: {sorted(_VALID_RESIZE_STRATEGIES)}"
        )

    if img.interpolation.lower() not in _VALID_INTERPOLATIONS:
        raise ConfigError(
            f"Invalid interpolation '{img.interpolation}'. "
            f"Must be one of: {sorted(_VALID_INTERPOLATIONS)}"
        )

    if img.color_space.upper() not in _VALID_COLOR_SPACES:
        raise ConfigError(
            f"Invalid color_space '{img.color_space}'. "
            f"Must be one of: {sorted(_VALID_COLOR_SPACES)}"
        )

    if img.storage_dtype not in _VALID_STORAGE_DTYPES:
        raise ConfigError(
            f"Invalid storage_dtype '{img.storage_dtype}'. "
            f"Must be one of: {sorted(_VALID_STORAGE_DTYPES)}"
        )

    if len(img.target_size) != 2 or any(d < 1 for d in img.target_size):
        raise ConfigError(
            f"target_size must be a pair of positive integers, got {img.target_size}"
        )

    if img.max_image_dim < 1:
        raise ConfigError(f"max_image_dim must be >= 1, got {img.max_image_dim}")

    if img.pad_value < 0 or img.pad_value > 255:
        raise ConfigError(f"pad_value must be in [0, 255], got {img.pad_value}")

    shard = config.shard
    if shard.format_version not in _VALID_FORMAT_VERSIONS:
        raise ConfigError(
            f"Invalid format_version {shard.format_version}. "
            f"Must be one of: {sorted(_VALID_FORMAT_VERSIONS)}"
        )

    if shard.compression not in _VALID_COMPRESSIONS:
        raise ConfigError(
            f"Invalid compression '{shard.compression}'. "
            f"Must be one of: {sorted(_VALID_COMPRESSIONS, key=str)}"
        )

    if not _is_power_of_two(shard.alignment_bytes):
        raise ConfigError(
            f"alignment_bytes must be a power of 2, got {shard.alignment_bytes}"
        )

    if shard.shard_size_mb < 1:
        raise ConfigError(f"shard_size_mb must be >= 1, got {shard.shard_size_mb}")

    tok = config.tokenizer
    if tok.max_length < 1:
        raise ConfigError(f"tokenizer.max_length must be >= 1, got {tok.max_length}")


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
        If the file cannot be read, is not valid YAML, or contains
        invalid parameter values.
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

    raw = _listify_to_tuples(raw)

    try:
        config = PipelineConfig.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Failed to construct PipelineConfig from {path}: {exc}") from exc

    _validate_config(config, path)

    logger.info(
        "Configuration loaded from %s: dataset=%s, split=%s",
        path, config.dataset.hf_dataset_id, config.dataset.split,
    )
    return config
