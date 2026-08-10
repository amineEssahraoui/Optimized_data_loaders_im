"""
Pipeline orchestrator for the VQA preprocessing pipeline.

Connects all stages into a single end-to-end workflow:
    1. Ingest: download/parse the dataset into VQASample objects.
    2. Normalize: process each sample's image through the image pipeline.
    3. Tokenize: encode question and answer text into token arrays.
    4. Write: serialize everything into binary shard files.

This module is the top-level entry point for the Python preprocessing
stage.  It reads all behavior from the PipelineConfig and does not
hardcode any processing parameters.

Usage from the command line::

    python -m preprocessing.pipeline --config configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.config import PipelineConfig, load_config
from preprocessing.image_processor import normalize_image
from preprocessing.ingest import ingest_dataset
from preprocessing.schema import VQASample
from preprocessing.shard_writer import ShardWriter
from preprocessing.tokenizer import TextTokenizer

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    """Configure root logger with the specified verbosity level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _preprocess_sample(
    sample: VQASample,
    config: PipelineConfig,
    tokenizer: TextTokenizer,
) -> dict[str, Any] | None:
    """Preprocess a single sample (normalize + tokenize).

    Returns a dict ready for ``ShardWriter.add_sample``, or None on error.
    """
    try:
        image_tensor = normalize_image(sample.image_bytes, config.image)
        q_tokens = tokenizer.tokenize(sample.question)
        a_tokens = tokenizer.tokenize(sample.answer)

        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata["original_width"] = sample.image_width
        metadata["original_height"] = sample.image_height

        return {
            "image_tensor": image_tensor,
            "question_ids": q_tokens.input_ids,
            "question_mask": q_tokens.attention_mask,
            "answer_ids": a_tokens.input_ids,
            "answer_mask": a_tokens.attention_mask,
            "metadata": metadata,
        }
    except Exception as exc:
        logger.warning(
            "Failed to process sample (id=%s): %s", sample.sample_id, exc,
        )
        return None


def _run_local_shuffle_pipeline(
    samples: list[VQASample],
    config: PipelineConfig,
    tokenizer: TextTokenizer,
    output_dir: Path,
    rng: np.random.RandomState,
) -> tuple[int, int, int, int]:
    """Execute the pipeline with local (in-shard) shuffling.

    Preprocesses all samples first, groups them into shard-sized chunks
    using ``max_samples_per_shard``, shuffles each chunk independently,
    then writes them to shard files.

    Parameters
    ----------
    samples : list[VQASample]
        Ingested samples in original order.
    config : PipelineConfig
        Full pipeline config.
    tokenizer : TextTokenizer
        Initialized tokenizer.
    output_dir : Path
        Directory where shard files are written.
    rng : np.random.RandomState
        Seeded random state for reproducible shuffling.

    Returns
    -------
    tuple[int, int, int, int]
        (last_shard_index, total_bytes, processed_count, error_count)
    """
    # Phase 1: Preprocess all samples
    preprocessed: list[dict[str, Any]] = []
    error_count = 0

    for i, sample in enumerate(samples):
        result = _preprocess_sample(sample, config, tokenizer)
        if result is not None:
            preprocessed.append(result)
        else:
            error_count += 1

        if (i + 1) % 50 == 0:
            logger.info("Preprocessed %d / %d samples...", i + 1, len(samples))

    processed_count = len(preprocessed)
    logger.info(
        "Preprocessed %d samples (%d errors). Grouping into shards...",
        processed_count, error_count,
    )

    # Phase 2: Group into shard-sized chunks
    max_per_shard = config.shard.max_samples_per_shard
    if max_per_shard is None or max_per_shard <= 0:
        # No sample-count limit: put all samples in one group
        # (size-based rotation not used in local shuffle mode since
        # we cannot predict byte sizes without writing)
        shard_groups: list[list[dict[str, Any]]] = [preprocessed]
    else:
        shard_groups = [
            preprocessed[start:start + max_per_shard]
            for start in range(0, len(preprocessed), max_per_shard)
        ]

    # Phase 3: Shuffle each group and write
    shard_index = 0
    total_bytes = 0

    for group in shard_groups:
        # Shuffle within this shard
        indices = rng.permutation(len(group)).tolist()
        shuffled_group = [group[i] for i in indices]

        writer = ShardWriter(config.shard, config.image, config.tokenizer)
        shard_path = output_dir / f"shard_{shard_index:04d}.bin"
        writer.open(shard_path)

        for sample_data in shuffled_group:
            writer.add_sample(**sample_data)

        total_bytes += writer.current_size_bytes
        writer.close()
        logger.info(
            "Wrote shard %d with %d locally-shuffled samples.",
            shard_index, len(shuffled_group),
        )
        shard_index += 1

    # Return last shard index (0-based), not count.
    # If no shards were written (all samples errored), return 0.
    last_index = max(shard_index - 1, 0)
    return last_index, total_bytes, processed_count, error_count


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Complete pipeline configuration.

    Returns
    -------
    dict[str, Any]
        Summary statistics: total samples, shards written, processing
        time, bytes written.
    """
    _setup_logging(config.log_level)

    start_time = time.time()

    # Stage 1: Ingest the dataset
    logger.info("=== Stage 1: Ingestion ===")
    samples = ingest_dataset(config)
    logger.info("Ingested %d samples.", len(samples))

    if not samples:
        logger.warning("No samples ingested. Pipeline has nothing to process.")
        return {"total_samples": 0, "shards_written": 0, "elapsed_seconds": 0.0}

    # Stage 1.5: Apply global shuffling (if configured)
    rng = np.random.RandomState(config.seed)

    if config.shuffling == "global":
        logger.info("=== Shuffling: global (cross-shard) ===")
        indices = rng.permutation(len(samples)).tolist()
        samples = [samples[i] for i in indices]
        logger.info("Shuffled %d samples globally.", len(samples))
    elif config.shuffling == "local":
        logger.info("=== Shuffling: local (in-shard) -- will shuffle per shard ===")
    else:
        logger.info("=== Shuffling: none ===")

    # Stage 2: Initialize the tokenizer (do this once, not per sample)
    logger.info("=== Stage 2: Tokenizer initialization ===")
    tokenizer = TextTokenizer(config.tokenizer)

    # Stage 3 + 4: Process each sample and write to shards
    logger.info("=== Stage 3+4: Normalize, tokenize, and write shards ===")

    output_dir = Path(config.shard.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    total_bytes = 0
    processed_count = 0
    error_count = 0

    if config.shuffling == "local":
        # Local shuffling: buffer preprocessed samples per shard, shuffle
        # the buffer before writing.  This requires a two-pass approach:
        # first preprocess all samples, then group and shuffle per shard.
        shard_index, total_bytes, processed_count, error_count = (
            _run_local_shuffle_pipeline(
                samples, config, tokenizer, output_dir, rng,
            )
        )
    else:
        # None or global: write samples in current order (global already
        # shuffled the list above).
        writer = ShardWriter(config.shard, config.image, config.tokenizer)
        shard_path = output_dir / f"shard_{shard_index:04d}.bin"
        writer.open(shard_path)

        for i, sample in enumerate(samples):
            try:
                # Normalize the image
                image_tensor = normalize_image(sample.image_bytes, config.image)

                # Tokenize question and answer
                q_tokens = tokenizer.tokenize(sample.question)
                a_tokens = tokenizer.tokenize(sample.answer)

                # Prepare metadata (include sample_id for traceability)
                metadata = dict(sample.metadata)
                metadata["sample_id"] = sample.sample_id
                metadata["original_width"] = sample.image_width
                metadata["original_height"] = sample.image_height

                # Write to the current shard
                writer.add_sample(
                    image_tensor=image_tensor,
                    question_ids=q_tokens.input_ids,
                    question_mask=q_tokens.attention_mask,
                    answer_ids=a_tokens.input_ids,
                    answer_mask=a_tokens.attention_mask,
                    metadata=metadata,
                )

                processed_count += 1

                # Check if we need to rotate to a new shard
                if writer.should_rotate():
                    total_bytes += writer.current_size_bytes
                    writer.close()
                    shard_index += 1
                    shard_path = output_dir / f"shard_{shard_index:04d}.bin"
                    writer = ShardWriter(config.shard, config.image, config.tokenizer)
                    writer.open(shard_path)
                    logger.info("Rotated to shard %d.", shard_index)

            except Exception as exc:
                logger.warning(
                    "Failed to process sample %d (id=%s): %s",
                    i, sample.sample_id, exc,
                )
                error_count += 1

            # Progress logging every 50 samples
            if (i + 1) % 50 == 0:
                logger.info("Processed %d / %d samples...", i + 1, len(samples))

        # Close the final shard
        total_bytes += writer.current_size_bytes
        writer.close()

    elapsed = time.time() - start_time

    summary = {
        "total_samples": len(samples),
        "processed_samples": processed_count,
        "error_samples": error_count,
        "shards_written": shard_index + 1,
        "total_bytes": total_bytes,
        "elapsed_seconds": round(elapsed, 2),
        "samples_per_second": round(processed_count / elapsed, 2) if elapsed > 0 else 0,
    }

    logger.info("=== Pipeline complete ===")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)

    return summary


def main() -> None:
    """CLI entry point: parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="VLM VQA Preprocessing Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_pipeline(config)

    # Exit with error code if any samples failed
    if summary.get("error_samples", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
