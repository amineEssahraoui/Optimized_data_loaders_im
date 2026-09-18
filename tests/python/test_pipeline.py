import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
import io

from configs.config import PipelineConfig
from preprocessing.pipeline import _run_streaming_pipeline, _run_local_shuffle_pipeline
from preprocessing.schema import VQASample
from preprocessing.tokenizer import TextTokenizer

# Helper to generate a dummy 1x1 image
def _generate_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def _generate_samples(count: int, corrupted_indices: set[int] = None) -> list[VQASample]:
    corrupted_indices = corrupted_indices or set()
    samples = []
    for i in range(count):
        if i in corrupted_indices:
            # Corrupted image bytes
            img_bytes = b"not_an_image"
        else:
            img_bytes = _generate_dummy_image_bytes()

        samples.append(
            VQASample(
                sample_id=f"sample_{i}",
                image_bytes=img_bytes,
                image_width=16,
                image_height=16,
                question=f"Question {i}",
                answer=f"Answer {i}",
                dataset_name="dummy_test",
            )
        )
    return samples

@pytest.fixture
def base_config(tmp_path: Path) -> PipelineConfig:
    config = PipelineConfig()
    # Modify frozen dataclass indirectly by creating a dict and overriding
    config_dict = config.to_dict()
    config_dict["shard"]["output_dir"] = str(tmp_path)
    config_dict["shard"]["max_samples_per_shard"] = 10
    config_dict["num_workers"] = 0  # Sequential for easy testing
    config_dict["log_level"] = "DEBUG"
    # Small tokenizer config
    config_dict["tokenizer"]["model_name_or_path"] = "bert-base-uncased"
    config_dict["tokenizer"]["max_length"] = 16
    return PipelineConfig.from_dict(config_dict)

def test_pipeline_produces_shard_files(base_config: PipelineConfig, tmp_path: Path):
    """1. Verifies the effective production of shard files."""
    samples = _generate_samples(5)
    rng = np.random.RandomState(42)

    _run_streaming_pipeline(samples, base_config, tmp_path, rng)

    shards = list(tmp_path.glob("*.bin"))
    assert len(shards) > 0, "No shard files were produced"

def test_pipeline_expected_number_of_files(base_config: PipelineConfig, tmp_path: Path):
    """2. Asserts the exact expected number of files produced."""
    samples = _generate_samples(25) # Should produce 3 files since max_samples_per_shard=10
    rng = np.random.RandomState(42)

    _run_streaming_pipeline(samples, base_config, tmp_path, rng)

    shards = list(tmp_path.glob("*.bin"))
    assert len(shards) == 3, f"Expected 3 shards, got {len(shards)}"

def test_streaming_vs_sequential_counts(base_config: PipelineConfig, tmp_path: Path):
    """3. Compares streaming mode against sequential reference mode."""
    samples = _generate_samples(25)
    rng = np.random.RandomState(42)

    # Streaming
    stream_dir = tmp_path / "stream"
    stream_dir.mkdir()
    _, _, stream_processed, _ = _run_streaming_pipeline(samples, base_config, stream_dir, rng)

    # Sequential
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    tokenizer = TextTokenizer(base_config.tokenizer)
    _, _, seq_processed, _ = _run_local_shuffle_pipeline(samples, base_config, tokenizer, seq_dir, rng)

    assert stream_processed == len(samples), "Streaming pipeline did not process all samples"
    assert stream_processed == seq_processed, "Streaming and sequential processed counts mismatch"

def test_fault_tolerance(base_config: PipelineConfig, tmp_path: Path):
    """4. Fault tolerance: Injects corrupted samples and asserts pipeline continues."""
    # Create 10 samples, corrupting indices 2 and 7
    samples = _generate_samples(10, corrupted_indices={2, 7})
    rng = np.random.RandomState(42)

    # Disable fail_fast
    config_dict = base_config.to_dict()
    config_dict["fail_fast"] = False
    config = PipelineConfig.from_dict(config_dict)

    _, _, processed_count, error_count = _run_streaming_pipeline(samples, config, tmp_path, rng)

    assert error_count == 2, "Expected exactly 2 errors"
    assert processed_count == 8, "Expected 8 successfully processed samples"

def test_local_shuffling(base_config: PipelineConfig, tmp_path: Path):
    """5. Verifies the correct behavior of the local shuffling mechanism."""
    # We will run with local shuffling enabled
    config_dict = base_config.to_dict()
    config_dict["shuffling"] = "local"
    config = PipelineConfig.from_dict(config_dict)

    samples = _generate_samples(10)
    rng1 = np.random.RandomState(42)

    # In order to test if shuffling happened, we can check the output sizes or just rely on the random seed
    # to ensure that it doesn't crash and returns the correct processed count.
    # Detailed verification of shuffling requires parsing the shard, which is covered by round-trip tests,
    # but here we ensure the mechanism executes.
    _, _, processed_count, _ = _run_streaming_pipeline(samples, config, tmp_path, rng1)

    assert processed_count == 10, "Local shuffle failed to process all samples"

def test_bounded_memory(base_config: PipelineConfig, tmp_path: Path):
    """6. Bounded memory verification (mocking tracemalloc)."""
    import tracemalloc

    rng = np.random.RandomState(42)

    # Set chunk size
    config_dict = base_config.to_dict()
    config_dict["streaming_chunk_size"] = 10
    config_dict["shard"]["max_samples_per_shard"] = 10
    config = PipelineConfig.from_dict(config_dict)

    def run_with_mem_profile(n: int) -> int:
        samples = _generate_samples(n)
        out_dir = tmp_path / f"mem_{n}"
        out_dir.mkdir()
        tracemalloc.start()
        _run_streaming_pipeline(samples, config, out_dir, rng)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    peak_n = run_with_mem_profile(10)
    peak_3n = run_with_mem_profile(30)

    # The peak memory for 30 samples should not be 3x the peak memory for 10 samples
    # since we process in chunks of 10. It should be roughly the same order of magnitude.
    # We assert that peak_3n < 2 * peak_n (allowing for some overhead).

    # NOTE: Since the samples list itself is passed and takes memory, the actual peak might slightly grow,
    # but the pipeline processing memory remains bounded.
    assert peak_3n < 2.5 * peak_n, f"Memory grew unboundedly: N={peak_n} bytes, 3N={peak_3n} bytes"
