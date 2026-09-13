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

Phase 4 additions:
    - Streaming shard-by-shard preprocessing (memory-bounded).
    - Multiprocessing via ``concurrent.futures.ProcessPoolExecutor``.
    - Per-worker tokenizer initialization (HuggingFace tokenizers are
      not fork-safe, so each worker process creates its own instance).

Usage from the command line::

    python -m preprocessing.pipeline --config configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.config import PipelineConfig, load_config
from preprocessing.image_processor import normalize_image, resize_preserve_aspect
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


# ---------------------------------------------------------------------------
# Per-worker state for multiprocessing
# ---------------------------------------------------------------------------
# Module-level globals used by worker processes.  Initialised once per
# worker via ``_init_worker_tokenizer``.  This avoids pickling the
# tokenizer (which would fail) and ensures each subprocess has its own
# HuggingFace tokenizer instance.
_worker_config: PipelineConfig | None = None
_worker_tokenizer: TextTokenizer | None = None


def _init_worker_tokenizer(config: PipelineConfig) -> None:
    """Initializer called once per ``ProcessPoolExecutor`` worker.

    Creates a dedicated ``TextTokenizer`` instance in the worker
    process.  HuggingFace tokenizers are **not** fork-safe, so they
    must be constructed inside the child process, not in the parent.

    The ``PipelineConfig`` dataclass is frozen and lightweight, so it
    can be pickled safely across the process boundary.
    """
    global _worker_config, _worker_tokenizer  # noqa: PLW0603
    _worker_config = config
    _worker_tokenizer = TextTokenizer(config.tokenizer)


def _process_sample_in_worker(
    sample: VQASample,
) -> dict[str, Any] | None:
    """Process a single sample inside a worker process.

    Uses the module-level ``_worker_config`` and ``_worker_tokenizer``
    that were initialised by ``_init_worker_tokenizer``.  Returns a
    dict ready for ``ShardWriter.add_sample``, or ``None`` on error.
    """
    config = _worker_config
    tokenizer = _worker_tokenizer
    if config is None or tokenizer is None:
        raise RuntimeError(
            "_process_sample_in_worker called before _init_worker_tokenizer"
        )
    return _preprocess_sample(sample, config, tokenizer)


# ---------------------------------------------------------------------------
# Single-sample preprocessing (used by both sequential and parallel paths)
# ---------------------------------------------------------------------------

def _preprocess_sample(
    sample: VQASample,
    config: PipelineConfig,
    tokenizer: TextTokenizer,
) -> dict[str, Any] | None:
    """Preprocess a single sample (normalize + tokenize).

    Returns a dict ready for ``ShardWriter.add_sample``, or None on error.
    Uses the v2 path (uint8 + per-sample dims) when dynamic_padding is enabled.
    """
    try:
        use_v2 = (
            config.shard.format_version >= 2
            and config.image.dynamic_padding
            and config.image.storage_dtype == "uint8"
        )

        if use_v2:
            image_arr, orig_h, orig_w, actual_h, actual_w = (
                resize_preserve_aspect(sample.image_bytes, config.image)
            )
        else:
            image_arr = normalize_image(sample.image_bytes, config.image)
            orig_h = sample.image_height
            orig_w = sample.image_width
            actual_h = actual_w = None

        q_tokens = tokenizer.tokenize(sample.question)
        a_tokens = tokenizer.tokenize(sample.answer)

        metadata = dict(sample.metadata)
        metadata["sample_id"] = sample.sample_id
        metadata["original_width"] = orig_w
        metadata["original_height"] = orig_h

        result: dict[str, Any] = {
            "image_tensor": image_arr,
            "question_ids": q_tokens.input_ids,
            "question_mask": q_tokens.attention_mask,
            "answer_ids": a_tokens.input_ids,
            "answer_mask": a_tokens.attention_mask,
            "metadata": metadata,
        }

        # V2: pass per-sample dimensions to the shard writer
        if use_v2 and actual_h is not None:
            result["orig_height"] = orig_h
            result["orig_width"] = orig_w

        return result
    except Exception as exc:
        logger.warning(
            "Failed to process sample (id=%s): %s", sample.sample_id, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Sequential (legacy) pipeline paths — unchanged from Phase 1-3
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Streaming multiprocessing pipeline (Phase 4)
# ---------------------------------------------------------------------------

def _process_chunk_with_executor(
    chunk: list[VQASample],
    config: PipelineConfig,
    executor: ProcessPoolExecutor | None,
    tokenizer: TextTokenizer | None,
) -> tuple[list[dict[str, Any]], int]:
    """Process a chunk of samples using a pre-created executor or tokenizer.

    When ``executor`` is provided, work is dispatched to the pool (whose
    workers already have their tokenizers initialised).  When ``executor``
    is None, ``tokenizer`` must be provided for sequential processing.

    Parameters
    ----------
    chunk : list[VQASample]
        Samples to process in this chunk.
    config : PipelineConfig
        Frozen pipeline config.
    executor : ProcessPoolExecutor | None
        A pre-created, long-lived pool.  Workers must already have
        been initialised via ``_init_worker_tokenizer``.
    tokenizer : TextTokenizer | None
        Pre-created tokenizer for sequential fallback.

    Returns
    -------
    tuple[list[dict], int]
        (processed_results, error_count).
    """
    processed: list[dict[str, Any]] = []
    error_count = 0

    if executor is None:
        # Sequential path
        assert tokenizer is not None
        for sample in chunk:
            result = _preprocess_sample(sample, config, tokenizer)
            if result is not None:
                processed.append(result)
            else:
                error_count += 1
        return processed, error_count

    # Parallel path — use the pre-created executor
    effective_workers = executor._max_workers  # type: ignore[attr-defined]
    try:
        results = list(executor.map(
            _process_sample_in_worker,
            chunk,
            chunksize=max(1, len(chunk) // (effective_workers * 4)),
        ))
    except Exception as exc:
        logger.error(
            "Multiprocessing pool error: %s. Falling back to sequential.",
            exc,
        )
        # Fallback: process sequentially in the main process
        fallback_tok = tokenizer or TextTokenizer(config.tokenizer)
        for sample in chunk:
            result = _preprocess_sample(sample, config, fallback_tok)
            if result is not None:
                processed.append(result)
            else:
                error_count += 1
        return processed, error_count

    for result in results:
        if result is not None:
            processed.append(result)
        else:
            error_count += 1

    return processed, error_count


# Keep the old name as an alias for backward compatibility (used by tests
# and the benchmark script).  Creates a one-shot pool internally.
def _process_chunk_parallel(
    chunk: list[VQASample],
    config: PipelineConfig,
    num_workers: int,
) -> tuple[list[dict[str, Any]], int]:
    """Convenience wrapper: process a chunk, creating a pool if needed.

    Prefer ``_process_chunk_with_executor`` for hot-path code that
    processes many chunks — it avoids repeated pool/tokenizer init.
    """
    effective_workers = min(num_workers, len(chunk), os.cpu_count() or 1)

    if effective_workers <= 1:
        tokenizer = TextTokenizer(config.tokenizer)
        return _process_chunk_with_executor(chunk, config, None, tokenizer)

    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_init_worker_tokenizer,
        initargs=(config,),
    ) as executor:
        return _process_chunk_with_executor(chunk, config, executor, None)


def _run_streaming_pipeline(
    samples: list[VQASample],
    config: PipelineConfig,
    output_dir: Path,
    rng: np.random.RandomState,
) -> tuple[int, int, int, int]:
    """Streaming multiprocessing pipeline (Phase 4).

    Processes the dataset shard-by-shard to keep memory usage bounded:

    1. Partition the sample list into chunks of ``max_samples_per_shard``
       (or a default of 500 if not configured).
    2. Create a **single** ``ProcessPoolExecutor`` whose workers
       initialise their tokenizers once and are reused across all
       chunks (avoiding repeated startup overhead).
    3. For each chunk, dispatch processing to the pool.
    4. Optionally shuffle the chunk (if ``config.shuffling == "local"``).
    5. Write the processed chunk to a shard file.
    6. Flush the chunk from memory before processing the next one.

    Parameters
    ----------
    samples : list[VQASample]
        Ingested samples (already globally-shuffled if configured).
    config : PipelineConfig
        Full pipeline config.
    output_dir : Path
        Directory where shard files are written.
    rng : np.random.RandomState
        Seeded random state for reproducible shuffling.

    Returns
    -------
    tuple[int, int, int, int]
        (last_shard_index, total_bytes, processed_count, error_count)
    """
    max_per_shard = config.shard.max_samples_per_shard
    if max_per_shard is None or max_per_shard <= 0:
        max_per_shard = 500  # sensible default for streaming

    num_workers = config.num_workers
    do_local_shuffle = config.shuffling == "local"

    shard_index = 0
    total_bytes = 0
    total_processed = 0
    total_errors = 0

    # Partition into chunks
    num_chunks = (len(samples) + max_per_shard - 1) // max_per_shard

    # Decide whether to use a process pool or sequential processing.
    effective_workers = min(num_workers, os.cpu_count() or 1)
    use_pool = effective_workers >= 2

    # ── Create the pool / tokenizer ONCE for all chunks ───────────
    executor: ProcessPoolExecutor | None = None
    tokenizer: TextTokenizer | None = None

    if use_pool:
        executor = ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_worker_tokenizer,
            initargs=(config,),
        )
        logger.info(
            "Created process pool with %d workers (tokenizers initialising).",
            effective_workers,
        )
    else:
        tokenizer = TextTokenizer(config.tokenizer)

    try:
        for chunk_idx in range(num_chunks):
            start = chunk_idx * max_per_shard
            end = min(start + max_per_shard, len(samples))
            chunk = samples[start:end]

            logger.info(
                "Processing chunk %d/%d (%d samples) with %d workers...",
                chunk_idx + 1, num_chunks, len(chunk), num_workers,
            )

            # Process the chunk (pool is reused across chunks)
            processed, error_count = _process_chunk_with_executor(
                chunk, config, executor, tokenizer,
            )
            total_errors += error_count

            if not processed:
                logger.warning(
                    "Chunk %d produced no valid samples. Skipping shard.",
                    chunk_idx + 1,
                )
                continue

            # Optional local shuffle within this shard's samples
            if do_local_shuffle:
                perm = rng.permutation(len(processed)).tolist()
                processed = [processed[i] for i in perm]

            # Write the processed chunk to a shard
            writer = ShardWriter(config.shard, config.image, config.tokenizer)
            shard_path = output_dir / f"shard_{shard_index:04d}.bin"
            writer.open(shard_path)

            for sample_data in processed:
                writer.add_sample(**sample_data)

            total_bytes += writer.current_size_bytes
            samples_in_shard = writer.sample_count
            writer.close()

            total_processed += samples_in_shard
            logger.info(
                "Wrote shard %d: %d samples, %.2f MB",
                shard_index, samples_in_shard,
                writer.current_size_bytes / (1024 * 1024),
            )
            shard_index += 1

            # ── Memory flush ──────────────────────────────────────
            # Release the processed chunk and trigger garbage
            # collection so we don't accumulate tensors across chunks.
            del processed, chunk
            gc.collect()

    finally:
        # ── Ensure the pool is shut down cleanly ──────────────────
        if executor is not None:
            executor.shutdown(wait=True)
            logger.info("Process pool shut down.")

    last_index = max(shard_index - 1, 0)
    return last_index, total_bytes, total_processed, total_errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    # Stage 2 + 3 + 4: Process and write shards
    output_dir = Path(config.shard.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    total_bytes = 0
    processed_count = 0
    error_count = 0

    # ── Dispatch to the appropriate pipeline strategy ─────────────
    if config.num_workers >= 1:
        # Phase 4: Streaming multiprocessing pipeline
        logger.info(
            "=== Stage 2+3+4: Streaming multiprocessing (%d workers) ===",
            config.num_workers,
        )
        shard_index, total_bytes, processed_count, error_count = (
            _run_streaming_pipeline(samples, config, output_dir, rng)
        )
    elif config.shuffling == "local":
        # Legacy: local shuffling (sequential)
        logger.info("=== Stage 2+3+4: Sequential local-shuffle pipeline ===")
        tokenizer = TextTokenizer(config.tokenizer)
        shard_index, total_bytes, processed_count, error_count = (
            _run_local_shuffle_pipeline(
                samples, config, tokenizer, output_dir, rng,
            )
        )
    else:
        # Legacy: sequential pipeline (no shuffling or global-shuffled)
        logger.info("=== Stage 2+3+4: Sequential pipeline ===")
        tokenizer = TextTokenizer(config.tokenizer)

        writer = ShardWriter(config.shard, config.image, config.tokenizer)
        shard_path = output_dir / f"shard_{shard_index:04d}.bin"
        writer.open(shard_path)

        for i, sample in enumerate(samples):
            try:
                # Prepare metadata (include sample_id for traceability)
                metadata = dict(sample.metadata)
                metadata["sample_id"] = sample.sample_id
                metadata["original_width"] = sample.image_width
                metadata["original_height"] = sample.image_height

                # Determine preprocessing path
                use_v2 = (
                    config.shard.format_version >= 2
                    and config.image.dynamic_padding
                    and config.image.storage_dtype == "uint8"
                )

                if use_v2:
                    image_arr, orig_h, orig_w, actual_h, actual_w = (
                        resize_preserve_aspect(sample.image_bytes, config.image)
                    )
                    metadata["original_width"] = orig_w
                    metadata["original_height"] = orig_h
                else:
                    image_arr = normalize_image(sample.image_bytes, config.image)

                # Tokenize question and answer
                q_tokens = tokenizer.tokenize(sample.question)
                a_tokens = tokenizer.tokenize(sample.answer)

                # Write to the current shard
                write_kwargs: dict[str, Any] = {
                    "image_tensor": image_arr,
                    "question_ids": q_tokens.input_ids,
                    "question_mask": q_tokens.attention_mask,
                    "answer_ids": a_tokens.input_ids,
                    "answer_mask": a_tokens.attention_mask,
                    "metadata": metadata,
                }
                if use_v2:
                    write_kwargs["orig_height"] = orig_h
                    write_kwargs["orig_width"] = orig_w

                writer.add_sample(**write_kwargs)

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
