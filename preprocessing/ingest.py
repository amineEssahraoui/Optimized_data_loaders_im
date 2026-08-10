"""
Dataset ingestion module for the VQA preprocessing pipeline.

Responsibilities:
- Download/load the HuggingFace dataset specified in config.
- Map dataset-specific column names to canonical schema fields using
  the column_mapping from config.
- Convert each row into a validated VQASample instance.
- Handle edge cases: missing images, empty text, non-standard image
  formats (PIL vs. raw bytes).

The ingestion stage is intentionally separate from normalization and
tokenization so that the canonical schema acts as a clean interface
boundary.  Downstream stages never need to know the source format.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Iterator

from datasets import load_dataset
from PIL import Image

from preprocessing.config import PipelineConfig
from preprocessing.schema import SchemaValidationError, VQASample

logger = logging.getLogger(__name__)


def _extract_image_bytes(image_value: Any) -> tuple[bytes, int, int]:
    """Convert a dataset image value to raw bytes and dimensions.

    HuggingFace datasets can store images as PIL.Image objects (when
    using the Image feature type) or as raw bytes.  This function
    handles both cases and returns a consistent (bytes, width, height)
    tuple.

    Parameters
    ----------
    image_value : Any
        The raw value from the dataset's image column.  Expected to be
        either a PIL.Image.Image or bytes.

    Returns
    -------
    tuple[bytes, int, int]
        (encoded_bytes, width, height) where bytes is PNG-encoded.

    Raises
    ------
    ValueError
        If the image value is neither a PIL Image nor bytes, or if
        the bytes cannot be decoded.
    """
    if isinstance(image_value, Image.Image):
        # PIL Image from HuggingFace Image feature -- encode to PNG
        # so downstream stages get consistent encoded bytes.
        width, height = image_value.size
        buffer = io.BytesIO()
        image_value.save(buffer, format="PNG")
        return buffer.getvalue(), width, height

    if isinstance(image_value, bytes):
        # Raw bytes -- decode to get dimensions, keep original bytes
        try:
            img = Image.open(io.BytesIO(image_value))
            width, height = img.size
            return image_value, width, height
        except Exception as exc:
            raise ValueError(f"Could not decode image bytes: {exc}") from exc

    raise ValueError(
        f"Unsupported image type: {type(image_value).__name__}. "
        f"Expected PIL.Image.Image or bytes."
    )


def ingest_row(
    row: dict[str, Any],
    column_mapping: dict[str, str],
    dataset_name: str,
    row_index: int,
) -> VQASample:
    """Convert a single dataset row into a canonical VQASample.

    Parameters
    ----------
    row : dict[str, Any]
        A single row from the HuggingFace dataset.
    column_mapping : dict[str, str]
        Maps canonical field names to dataset column names.
    dataset_name : str
        Identifier for the source dataset (from config).
    row_index : int
        Row index, used as fallback sample_id if no id column exists.

    Returns
    -------
    VQASample
        A validated canonical sample.

    Raises
    ------
    SchemaValidationError
        If the constructed sample fails validation.
    ValueError
        If required columns are missing or image extraction fails.
    """
    # Extract required fields using the column mapping
    image_col = column_mapping.get("image", "image")
    question_col = column_mapping.get("question", "question")
    answer_col = column_mapping.get("answer", "model_answer")
    id_col = column_mapping.get("id", "id")
    reasoning_col = column_mapping.get("reasoning", "model_reasoning")

    # Sample ID: use the dataset's id column if available, otherwise
    # generate one from the row index.
    raw_id = row.get(id_col)
    sample_id = str(raw_id) if raw_id is not None else f"row_{row_index}"

    # Image: extract bytes and dimensions
    if image_col not in row or row[image_col] is None:
        raise ValueError(
            f"Row {row_index} (id={sample_id}): missing image column '{image_col}'"
        )
    image_bytes, width, height = _extract_image_bytes(row[image_col])

    # Text fields: question and answer are required, must be non-empty
    question = row.get(question_col, "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            f"Row {row_index} (id={sample_id}): missing or empty question "
            f"in column '{question_col}'"
        )

    answer = row.get(answer_col, "")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"Row {row_index} (id={sample_id}): missing or empty answer "
            f"in column '{answer_col}'"
        )

    # Metadata: optional auxiliary fields (e.g. model_reasoning)
    metadata: dict[str, Any] = {}
    reasoning = row.get(reasoning_col)
    if reasoning is not None and isinstance(reasoning, str) and reasoning.strip():
        metadata["model_reasoning"] = reasoning

    sample = VQASample(
        sample_id=sample_id,
        image_bytes=image_bytes,
        image_width=width,
        image_height=height,
        question=question,
        answer=answer,
        metadata=metadata,
        dataset_name=dataset_name,
    )

    # Validate at construction time to catch issues immediately
    sample.validate()
    return sample


def ingest_dataset(config: PipelineConfig) -> list[VQASample]:
    """Ingest the full dataset specified in config into canonical samples.

    Downloads (or streams) the HuggingFace dataset, iterates over all
    rows, and converts each to a VQASample.  Rows that fail conversion
    are logged and skipped rather than crashing the entire pipeline.

    Parameters
    ----------
    config : PipelineConfig
        The pipeline configuration with dataset source and column mapping.

    Returns
    -------
    list[VQASample]
        Successfully ingested samples.  The list may be shorter than
        the dataset if some rows failed validation.
    """
    ds_cfg = config.dataset

    logger.info(
        "Loading dataset: %s (config=%s, split=%s)",
        ds_cfg.hf_dataset_id,
        ds_cfg.hf_config_name,
        ds_cfg.split,
    )

    # Load the dataset from HuggingFace
    dataset = load_dataset(
        ds_cfg.hf_dataset_id,
        name=ds_cfg.hf_config_name,
        split=ds_cfg.split,
        streaming=ds_cfg.streaming,
    )

    samples: list[VQASample] = []
    skipped = 0
    total = 0

    for row_index, row in enumerate(dataset):
        # Respect max_samples limit for debugging
        if ds_cfg.max_samples is not None and total >= ds_cfg.max_samples:
            logger.info(
                "Reached max_samples limit (%d), stopping ingestion.",
                ds_cfg.max_samples,
            )
            break

        total += 1

        try:
            sample = ingest_row(
                row=row,
                column_mapping=ds_cfg.column_mapping,
                dataset_name=ds_cfg.hf_dataset_id,
                row_index=row_index,
            )
            samples.append(sample)
        except (ValueError, SchemaValidationError) as exc:
            logger.warning("Skipping row %d: %s", row_index, exc)
            skipped += 1

    logger.info(
        "Ingestion complete: %d samples ingested, %d skipped, %d total rows processed.",
        len(samples),
        skipped,
        total,
    )
    return samples


def ingest_dataset_iter(config: PipelineConfig) -> Iterator[VQASample]:
    """Iterator variant of ingest_dataset for memory-efficient processing.

    Yields one VQASample at a time instead of collecting all into a list.
    Useful when the dataset is large and the caller wants to process
    samples in a streaming fashion (e.g. writing shards incrementally).

    Parameters
    ----------
    config : PipelineConfig
        The pipeline configuration.

    Yields
    ------
    VQASample
        Successfully ingested samples, one at a time.
    """
    ds_cfg = config.dataset

    dataset = load_dataset(
        ds_cfg.hf_dataset_id,
        name=ds_cfg.hf_config_name,
        split=ds_cfg.split,
        streaming=ds_cfg.streaming,
    )

    total = 0
    for row_index, row in enumerate(dataset):
        if ds_cfg.max_samples is not None and total >= ds_cfg.max_samples:
            break

        total += 1
        try:
            sample = ingest_row(
                row=row,
                column_mapping=ds_cfg.column_mapping,
                dataset_name=ds_cfg.hf_dataset_id,
                row_index=row_index,
            )
            yield sample
        except (ValueError, SchemaValidationError) as exc:
            logger.warning("Skipping row %d: %s", row_index, exc)
