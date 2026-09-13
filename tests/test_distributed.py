#!/usr/bin/env python3
"""
Tests for Phase 4: Distributed Shard Manifest & Streaming Pipeline.

Covers:
    - ShardManifest round-trip serialization (save → load)
    - Deterministic shard assignment (assign_shards)
    - Full coverage and no-overlap guarantees across ranks
    - Edge cases (world_size > shards, single rank, etc.)
    - Streaming pipeline correctness (shards written, data integrity)
    - Memory-bounded streaming (peak memory does not scale with N)

Run::

    python -m pytest tests/test_distributed.py -v
"""

from __future__ import annotations

import gc
import io
import json
import shutil
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.config import (
    ImageConfig,
    PipelineConfig,
    ShardConfig,
    TokenizerConfig,
)
from preprocessing.schema import VQASample
from preprocessing.shard_manifest import (
    ShardInfo,
    ShardManifest,
    assign_shards,
    build_manifest,
)
from preprocessing.shard_writer import ShardWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_shard_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for shard files."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    return shard_dir


@pytest.fixture
def image_config() -> ImageConfig:
    """Minimal image config for testing."""
    return ImageConfig(
        target_size=(32, 32),
        max_image_dim=32,
        color_space="RGB",
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
        storage_dtype="uint8",
        dynamic_padding=True,
        interpolation="bilinear",
    )


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    """Minimal tokenizer config for testing."""
    return TokenizerConfig(
        model_name_or_path="bert-base-uncased",
        max_length=16,
        padding="max_length",
        truncation=True,
        trust_remote_code=False,
    )


@pytest.fixture
def shard_config(tmp_shard_dir: Path) -> ShardConfig:
    """Shard config for testing."""
    return ShardConfig(
        output_dir=str(tmp_shard_dir),
        shard_size_mb=999,
        max_samples_per_shard=10,
        compression=None,
        alignment_bytes=64,
        format_version=2,
    )


def _make_synthetic_image(w: int = 32, h: int = 32) -> bytes:
    """Create a tiny JPEG image for testing."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return buf.getvalue()


def _make_samples(n: int, img_w: int = 32, img_h: int = 32) -> list[VQASample]:
    """Generate N synthetic VQASample instances."""
    samples = []
    for i in range(n):
        samples.append(VQASample(
            sample_id=f"test_{i:04d}",
            image_bytes=_make_synthetic_image(img_w, img_h),
            image_width=img_w,
            image_height=img_h,
            question=f"What is sample {i}?",
            answer=f"This is the answer for sample {i}.",
            metadata={"index": i},
            dataset_name="test_dataset",
        ))
    return samples


def _write_shards(
    samples: list[VQASample],
    shard_config: ShardConfig,
    image_config: ImageConfig,
    tokenizer_config: TokenizerConfig,
    output_dir: Path,
) -> int:
    """Write synthetic samples to shard files using v1 float32 format.

    Returns the number of shards written.
    """
    writer = ShardWriter(shard_config, image_config, tokenizer_config)
    shard_idx = 0
    shard_path = output_dir / f"shard_{shard_idx:04d}.bin"
    writer.open(shard_path)

    for i, _sample in enumerate(samples):
        c, h, w = 3, 32, 32
        img = np.random.randint(0, 256, (c, h, w), dtype=np.uint8)
        q_ids = np.full(tokenizer_config.max_length, i * 10, dtype=np.int32)
        q_mask = np.ones(tokenizer_config.max_length, dtype=np.int32)
        a_ids = np.full(tokenizer_config.max_length, i * 100, dtype=np.int32)
        a_mask = np.ones(tokenizer_config.max_length, dtype=np.int32)
        meta = {"sample_id": _sample.sample_id, "index": i}

        writer.add_sample(img, q_ids, q_mask, a_ids, a_mask, meta,
                          orig_height=h, orig_width=w)

        if writer.should_rotate():
            writer.close()
            shard_idx += 1
            shard_path = output_dir / f"shard_{shard_idx:04d}.bin"
            writer = ShardWriter(shard_config, image_config, tokenizer_config)
            writer.open(shard_path)

    writer.close()
    return shard_idx + 1


# ===========================================================================
# Tests: ShardManifest serialization
# ===========================================================================

class TestShardManifestSerialization:
    """Test manifest save/load round-trip."""

    def test_manifest_round_trip(
        self, tmp_shard_dir: Path, image_config: ImageConfig,
        tokenizer_config: TokenizerConfig, shard_config: ShardConfig,
    ) -> None:
        """Build a manifest, save it, reload it, and verify equality."""
        # Write some shards
        samples = _make_samples(25)
        _write_shards(samples, shard_config, image_config, tokenizer_config,
                      tmp_shard_dir)

        # Build and save manifest
        manifest = build_manifest(tmp_shard_dir)
        manifest_path = tmp_shard_dir / "manifest.json"
        manifest.save(manifest_path)

        # Reload and compare
        loaded = ShardManifest.load(manifest_path)

        assert loaded.version == manifest.version
        assert loaded.total_samples == manifest.total_samples
        assert loaded.total_shards == manifest.total_shards
        assert loaded.total_bytes == manifest.total_bytes
        assert len(loaded.shards) == len(manifest.shards)

        for orig, reloaded in zip(manifest.shards, loaded.shards):
            assert reloaded.path == orig.path
            assert reloaded.sample_count == orig.sample_count
            assert reloaded.size_bytes == orig.size_bytes
            assert reloaded.checksum_crc32 == orig.checksum_crc32

    def test_manifest_total_samples(
        self, tmp_shard_dir: Path, image_config: ImageConfig,
        tokenizer_config: TokenizerConfig, shard_config: ShardConfig,
    ) -> None:
        """Verify total_samples equals sum of per-shard counts."""
        samples = _make_samples(25)
        _write_shards(samples, shard_config, image_config, tokenizer_config,
                      tmp_shard_dir)

        manifest = build_manifest(tmp_shard_dir)
        expected_total = sum(s.sample_count for s in manifest.shards)
        assert manifest.total_samples == expected_total

    def test_manifest_json_structure(
        self, tmp_shard_dir: Path, image_config: ImageConfig,
        tokenizer_config: TokenizerConfig, shard_config: ShardConfig,
    ) -> None:
        """Verify the raw JSON has the expected schema."""
        samples = _make_samples(15)
        _write_shards(samples, shard_config, image_config, tokenizer_config,
                      tmp_shard_dir)

        manifest = build_manifest(tmp_shard_dir)
        manifest_path = tmp_shard_dir / "manifest.json"
        manifest.save(manifest_path)

        with open(manifest_path) as fh:
            raw = json.load(fh)

        assert "version" in raw
        assert "created_at" in raw
        assert "total_samples" in raw
        assert "total_shards" in raw
        assert "total_bytes" in raw
        assert "shards" in raw
        assert isinstance(raw["shards"], list)

        for shard in raw["shards"]:
            assert "path" in shard
            assert "sample_count" in shard
            assert "size_bytes" in shard
            assert "checksum_crc32" in shard

    def test_manifest_empty_dir_raises(self, tmp_path: Path) -> None:
        """build_manifest should raise ValueError for empty directory."""
        empty_dir = tmp_path / "empty_shards"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No shard files"):
            build_manifest(empty_dir)

    def test_manifest_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        """build_manifest should raise FileNotFoundError for missing dir."""
        with pytest.raises(FileNotFoundError):
            build_manifest(tmp_path / "nonexistent")


# ===========================================================================
# Tests: Distributed shard assignment
# ===========================================================================

class TestAssignShards:
    """Test deterministic shard assignment for DDP."""

    def _make_manifest(self, num_shards: int, samples_per_shard: int = 100) -> ShardManifest:
        """Create a synthetic manifest for testing assignment."""
        shards = [
            ShardInfo(
                path=f"shard_{i:04d}.bin",
                sample_count=samples_per_shard,
                size_bytes=samples_per_shard * 1024,
                checksum_crc32=f"{i:08x}",
            )
            for i in range(num_shards)
        ]
        return ShardManifest(
            version=1,
            created_at="2026-09-12T17:30:00+00:00",
            total_samples=num_shards * samples_per_shard,
            total_shards=num_shards,
            total_bytes=sum(s.size_bytes for s in shards),
            shards=shards,
        )

    def test_deterministic(self) -> None:
        """Same inputs must always produce the same output."""
        manifest = self._make_manifest(8, 100)

        for rank in range(4):
            result_a = assign_shards(manifest, rank, 4)
            result_b = assign_shards(manifest, rank, 4)
            assert len(result_a) == len(result_b)
            for a, b in zip(result_a, result_b):
                assert a.path == b.path
                assert a.sample_count == b.sample_count

    def test_full_coverage(self) -> None:
        """Union of all ranks' shards must equal the full shard set."""
        manifest = self._make_manifest(10, 50)
        world_size = 4

        all_paths: set[str] = set()
        for rank in range(world_size):
            assigned = assign_shards(manifest, rank, world_size)
            for s in assigned:
                all_paths.add(s.path)

        expected_paths = {s.path for s in manifest.shards}
        assert all_paths == expected_paths

    def test_no_overlap(self) -> None:
        """No shard should be assigned to two different ranks."""
        manifest = self._make_manifest(12, 50)
        world_size = 4

        rank_shards: list[set[str]] = []
        for rank in range(world_size):
            assigned = assign_shards(manifest, rank, world_size)
            rank_shards.append({s.path for s in assigned})

        for r1 in range(world_size):
            for r2 in range(r1 + 1, world_size):
                overlap = rank_shards[r1] & rank_shards[r2]
                assert not overlap, (
                    f"Overlap between rank {r1} and {r2}: {overlap}"
                )

    def test_single_rank(self) -> None:
        """world_size=1: the single rank gets all shards."""
        manifest = self._make_manifest(5, 100)
        assigned = assign_shards(manifest, 0, 1)
        assert len(assigned) == 5
        assert [s.path for s in assigned] == [s.path for s in manifest.shards]

    def test_world_size_exceeds_shards(self) -> None:
        """When world_size > num_shards, some ranks get nothing."""
        manifest = self._make_manifest(3, 100)
        world_size = 8

        non_empty_count = 0
        empty_count = 0
        all_paths: set[str] = set()

        for rank in range(world_size):
            assigned = assign_shards(manifest, rank, world_size)
            if assigned:
                non_empty_count += 1
                for s in assigned:
                    all_paths.add(s.path)
            else:
                empty_count += 1

        # Exactly 3 ranks should get shards, 5 should be empty
        assert non_empty_count == 3
        assert empty_count == 5
        assert all_paths == {s.path for s in manifest.shards}

    def test_equal_distribution(self) -> None:
        """Shards should be roughly equally distributed."""
        manifest = self._make_manifest(16, 50)
        world_size = 4

        counts = []
        for rank in range(world_size):
            assigned = assign_shards(manifest, rank, world_size)
            counts.append(len(assigned))

        # With 16 shards and 4 workers, each should get exactly 4
        assert all(c == 4 for c in counts)

    def test_uneven_distribution(self) -> None:
        """With non-divisible counts, distribution differs by at most 1."""
        manifest = self._make_manifest(7, 50)
        world_size = 3

        counts = []
        for rank in range(world_size):
            assigned = assign_shards(manifest, rank, world_size)
            counts.append(len(assigned))

        # 7 shards / 3 workers: some get 3, some get 2
        assert max(counts) - min(counts) <= 1
        assert sum(counts) == 7

    def test_invalid_rank_raises(self) -> None:
        """rank >= world_size should raise ValueError."""
        manifest = self._make_manifest(4, 100)
        with pytest.raises(ValueError, match="rank"):
            assign_shards(manifest, 4, 4)

    def test_invalid_world_size_raises(self) -> None:
        """world_size < 1 should raise ValueError."""
        manifest = self._make_manifest(4, 100)
        with pytest.raises(ValueError, match="world_size"):
            assign_shards(manifest, 0, 0)

    def test_negative_rank_raises(self) -> None:
        """Negative rank should raise ValueError."""
        manifest = self._make_manifest(4, 100)
        with pytest.raises(ValueError, match="rank"):
            assign_shards(manifest, -1, 4)


# ===========================================================================
# Tests: Streaming pipeline
# ===========================================================================

class TestStreamingPipeline:
    """Test the Phase 4 streaming multiprocessing pipeline."""

    def _make_pipeline_config(
        self, output_dir: Path, num_workers: int = 1,
    ) -> PipelineConfig:
        """Create a pipeline config for testing."""
        return PipelineConfig(
            image=ImageConfig(
                target_size=(32, 32),
                max_image_dim=32,
                storage_dtype="uint8",
                dynamic_padding=True,
                color_space="RGB",
                interpolation="bilinear",
            ),
            tokenizer=TokenizerConfig(
                model_name_or_path="bert-base-uncased",
                max_length=16,
                padding="max_length",
                truncation=True,
                trust_remote_code=False,
            ),
            shard=ShardConfig(
                output_dir=str(output_dir),
                shard_size_mb=999,
                max_samples_per_shard=10,
                compression=None,
                alignment_bytes=64,
                format_version=2,
            ),
            num_workers=num_workers,
            shuffling="none",
            log_level="WARNING",
            seed=42,
        )

    def test_streaming_produces_shards(self, tmp_path: Path) -> None:
        """Streaming pipeline should create shard files."""
        from preprocessing.pipeline import _run_streaming_pipeline

        output_dir = tmp_path / "stream_shards"
        output_dir.mkdir()
        config = self._make_pipeline_config(output_dir, num_workers=1)
        samples = _make_samples(25)
        rng = np.random.RandomState(42)

        _last_idx, total_bytes, processed, errors = _run_streaming_pipeline(
            samples, config, output_dir, rng,
        )

        shard_files = list(output_dir.glob("shard_*.bin"))
        assert len(shard_files) > 0, "No shard files produced"
        assert processed == 25
        assert errors == 0
        assert total_bytes > 0

    def test_streaming_shard_count(self, tmp_path: Path) -> None:
        """Number of shards should match ceil(N / max_samples_per_shard)."""
        from preprocessing.pipeline import _run_streaming_pipeline

        output_dir = tmp_path / "shard_count"
        output_dir.mkdir()
        config = self._make_pipeline_config(output_dir, num_workers=1)
        samples = _make_samples(25)
        rng = np.random.RandomState(42)

        _run_streaming_pipeline(samples, config, output_dir, rng)

        shard_files = list(output_dir.glob("shard_*.bin"))
        # 25 samples, max 10 per shard → 3 shards
        expected = (25 + 10 - 1) // 10
        assert len(shard_files) == expected

    def test_streaming_vs_sequential_sample_count(self, tmp_path: Path) -> None:
        """Streaming and sequential should produce same total samples."""
        from preprocessing.pipeline import _run_streaming_pipeline
        from preprocessing.shard_manifest import build_manifest

        num_samples = 20

        # Sequential
        seq_dir = tmp_path / "sequential"
        seq_dir.mkdir()
        seq_config = self._make_pipeline_config(seq_dir, num_workers=0)
        samples = _make_samples(num_samples)

        # Use the streaming pipeline even with 0 workers (it falls back
        # to sequential processing within _process_chunk_parallel).
        rng1 = np.random.RandomState(42)
        _, _, seq_processed, seq_errors = _run_streaming_pipeline(
            samples, seq_config, seq_dir, rng1,
        )

        # Streaming with 2 workers
        par_dir = tmp_path / "parallel"
        par_dir.mkdir()
        par_config = self._make_pipeline_config(par_dir, num_workers=2)
        rng2 = np.random.RandomState(42)
        _, _, par_processed, par_errors = _run_streaming_pipeline(
            samples, par_config, par_dir, rng2,
        )

        assert seq_processed == par_processed
        assert seq_errors == par_errors

    def test_streaming_handles_errors_gracefully(self, tmp_path: Path) -> None:
        """Pipeline should skip samples with invalid images."""
        from preprocessing.pipeline import _run_streaming_pipeline

        output_dir = tmp_path / "error_handling"
        output_dir.mkdir()
        config = self._make_pipeline_config(output_dir, num_workers=1)

        # Mix valid and invalid samples
        samples = _make_samples(5)
        # Add a sample with corrupted image bytes
        samples.append(VQASample(
            sample_id="bad_image",
            image_bytes=b"not_a_real_image",
            image_width=32,
            image_height=32,
            question="What is this?",
            answer="This should fail.",
            metadata={},
            dataset_name="test",
        ))
        samples.extend(_make_samples(3, img_w=32, img_h=32))

        rng = np.random.RandomState(42)
        _, _, processed, errors = _run_streaming_pipeline(
            samples, config, output_dir, rng,
        )

        # 8 valid samples should succeed, 1 should error
        assert processed == 8
        assert errors == 1

    def test_streaming_with_local_shuffle(self, tmp_path: Path) -> None:
        """Local shuffling should still produce all samples."""
        from preprocessing.pipeline import _run_streaming_pipeline
        import dataclasses as dc

        output_dir = tmp_path / "local_shuffle"
        output_dir.mkdir()
        config = self._make_pipeline_config(output_dir, num_workers=1)
        config = dc.replace(config, shuffling="local")
        samples = _make_samples(20)
        rng = np.random.RandomState(42)

        _, _, processed, errors = _run_streaming_pipeline(
            samples, config, output_dir, rng,
        )

        assert processed == 20
        assert errors == 0

    def test_streaming_memory_bounded(self, tmp_path: Path) -> None:
        """Peak memory should not scale linearly with dataset size.

        Runs the streaming pipeline twice: once with N samples, once
        with 2N samples.  The peak memory should not double.
        """
        from preprocessing.pipeline import _run_streaming_pipeline

        def _measure_peak_memory(n_samples: int) -> float:
            d = tmp_path / f"mem_test_{n_samples}"
            d.mkdir()
            config = self._make_pipeline_config(d, num_workers=1)
            samps = _make_samples(n_samples)
            rng = np.random.RandomState(42)

            gc.collect()
            tracemalloc.start()

            _run_streaming_pipeline(samps, config, d, rng)

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak / (1024 * 1024)  # MB

        peak_small = _measure_peak_memory(20)
        peak_large = _measure_peak_memory(60)

        # Memory should not triple even though samples tripled.
        # Allow generous headroom (2x) to account for tokenizer cache, etc.
        assert peak_large < peak_small * 3.0, (
            f"Memory appears to scale linearly: "
            f"20 samples → {peak_small:.1f} MB, "
            f"60 samples → {peak_large:.1f} MB"
        )


# ===========================================================================
# Tests: Manifest integration with streaming pipeline
# ===========================================================================

class TestManifestIntegration:
    """Test that build_manifest works on shards produced by streaming pipeline."""

    def test_manifest_from_streaming_output(self, tmp_path: Path) -> None:
        """build_manifest should work on shards written by streaming pipeline."""
        from preprocessing.pipeline import _run_streaming_pipeline

        output_dir = tmp_path / "manifest_integration"
        output_dir.mkdir()

        config = PipelineConfig(
            image=ImageConfig(
                target_size=(32, 32),
                max_image_dim=32,
                storage_dtype="uint8",
                dynamic_padding=True,
                color_space="RGB",
                interpolation="bilinear",
            ),
            tokenizer=TokenizerConfig(
                model_name_or_path="bert-base-uncased",
                max_length=16,
                padding="max_length",
                truncation=True,
                trust_remote_code=False,
            ),
            shard=ShardConfig(
                output_dir=str(output_dir),
                shard_size_mb=999,
                max_samples_per_shard=10,
                compression=None,
                alignment_bytes=64,
                format_version=2,
            ),
            num_workers=1,
            shuffling="none",
            log_level="WARNING",
        )

        samples = _make_samples(30)
        rng = np.random.RandomState(42)
        _run_streaming_pipeline(samples, config, output_dir, rng)

        # Build manifest from the output
        manifest = build_manifest(output_dir)

        assert manifest.total_samples == 30
        assert manifest.total_shards == 3  # ceil(30/10)
        assert all(s.checksum_crc32 != "" for s in manifest.shards)

        # Assign shards across 2 ranks
        for rank in range(2):
            assigned = assign_shards(manifest, rank, 2)
            assert len(assigned) > 0

        # Verify full coverage
        all_paths = set()
        for rank in range(2):
            for s in assign_shards(manifest, rank, 2):
                all_paths.add(s.path)
        assert all_paths == {s.path for s in manifest.shards}
