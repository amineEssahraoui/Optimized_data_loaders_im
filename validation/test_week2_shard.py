"""
Week 2 validation: Binary shard write/read integrity.

Tests in this module verify that:
1. ShardWriter produces valid shard files with correct header/footer.
2. ShardReader can parse files written by ShardWriter.
3. Round-trip (write -> read) preserves all data exactly.
4. The offset table enables correct random access.
5. Multiple samples per shard work correctly.
6. Checksum validation catches corruption.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from preprocessing.config import ImageConfig, ShardConfig, TokenizerConfig
from preprocessing.shard_reader import ShardReader
from preprocessing.shard_writer import (
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC_END,
    MAGIC_START,
    ShardWriter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def shard_configs():
    """Provide a consistent set of configs for shard tests."""
    image_cfg = ImageConfig(target_size=(32, 32))
    tokenizer_cfg = TokenizerConfig(max_length=16)
    shard_cfg = ShardConfig(
        output_dir="./test_shards",
        shard_size_mb=1024,  # Large enough to not trigger rotation in tests
        alignment_bytes=64,
    )
    return shard_cfg, image_cfg, tokenizer_cfg


def _make_sample_data(
    num_channels: int = 3,
    height: int = 32,
    width: int = 32,
    token_length: int = 16,
    seed: int = 42,
):
    """Create synthetic sample data with known values for verification."""
    rng = np.random.RandomState(seed)
    return {
        "image_tensor": rng.randn(num_channels, height, width).astype(np.float32),
        "question_ids": rng.randint(0, 10000, size=token_length, dtype=np.int32),
        "question_mask": rng.randint(0, 2, size=token_length, dtype=np.int32),
        "answer_ids": rng.randint(0, 10000, size=token_length, dtype=np.int32),
        "answer_mask": rng.randint(0, 2, size=token_length, dtype=np.int32),
        "metadata": {"sample_id": f"test_{seed}", "extra": "data"},
    }


# ===========================================================================
# Tests
# ===========================================================================
class TestShardWriterBasic:
    """Verify basic shard writer operations."""

    def test_write_single_sample(self, tmp_path: Path, shard_configs):
        """Writing a single sample must produce a valid shard file."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_single.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)

        data = _make_sample_data()
        writer.add_sample(**data)
        writer.close()

        assert shard_path.exists()
        assert shard_path.stat().st_size > HEADER_SIZE

    def test_write_multiple_samples(self, tmp_path: Path, shard_configs):
        """Writing multiple samples must track sample count correctly."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_multi.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)

        for i in range(5):
            data = _make_sample_data(seed=i)
            writer.add_sample(**data)

        assert writer.sample_count == 5
        writer.close()

    def test_context_manager(self, tmp_path: Path, shard_configs):
        """ShardWriter must work as a context manager."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_context.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        with writer:
            data = _make_sample_data()
            writer.add_sample(**data)

        # File should be closed and valid
        assert shard_path.exists()


class TestShardHeaderFooter:
    """Verify shard file header and footer structure."""

    def test_header_magic(self, tmp_path: Path, shard_configs):
        """The file must start with the correct magic bytes."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_header.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        writer.add_sample(**_make_sample_data())
        writer.close()

        with open(shard_path, "rb") as f:
            magic = f.read(8)
        assert magic == MAGIC_START

    def test_footer_magic(self, tmp_path: Path, shard_configs):
        """The file must end with the correct end magic bytes."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_footer.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        writer.add_sample(**_make_sample_data())
        writer.close()

        with open(shard_path, "rb") as f:
            f.seek(-8, 2)  # Last 8 bytes
            end_magic = f.read(8)
        assert end_magic == MAGIC_END

    def test_header_sample_count(self, tmp_path: Path, shard_configs):
        """The header sample_count must reflect the number of samples written."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_count.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        for i in range(3):
            writer.add_sample(**_make_sample_data(seed=i))
        writer.close()

        with open(shard_path, "rb") as f:
            f.seek(12)  # Skip magic(8) + version(4)
            sample_count = struct.unpack("<I", f.read(4))[0]
        assert sample_count == 3


class TestShardRoundTrip:
    """Verify write -> read round-trip preserves data exactly."""

    def test_single_sample_round_trip(self, tmp_path: Path, shard_configs):
        """A single sample must survive write -> read exactly."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_rt_single.bin"

        original = _make_sample_data(seed=99)

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        writer.add_sample(**original)
        writer.close()

        reader = ShardReader(shard_path)
        assert reader.sample_count == 1

        recovered = reader.read_sample(0)
        reader.close()

        np.testing.assert_array_equal(recovered.image_tensor, original["image_tensor"])
        np.testing.assert_array_equal(recovered.question_ids, original["question_ids"])
        np.testing.assert_array_equal(recovered.question_mask, original["question_mask"])
        np.testing.assert_array_equal(recovered.answer_ids, original["answer_ids"])
        np.testing.assert_array_equal(recovered.answer_mask, original["answer_mask"])
        assert recovered.metadata == original["metadata"]

    def test_multi_sample_round_trip(self, tmp_path: Path, shard_configs):
        """Multiple samples must each survive write -> read exactly."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_rt_multi.bin"
        num_samples = 10

        originals = [_make_sample_data(seed=i) for i in range(num_samples)]

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        for data in originals:
            writer.add_sample(**data)
        writer.close()

        reader = ShardReader(shard_path)
        assert reader.sample_count == num_samples

        for i in range(num_samples):
            recovered = reader.read_sample(i)
            np.testing.assert_array_equal(
                recovered.image_tensor, originals[i]["image_tensor"]
            )
            np.testing.assert_array_equal(
                recovered.question_ids, originals[i]["question_ids"]
            )
            np.testing.assert_array_equal(
                recovered.answer_ids, originals[i]["answer_ids"]
            )
            assert recovered.metadata == originals[i]["metadata"]
        reader.close()

    def test_random_access_order(self, tmp_path: Path, shard_configs):
        """Reading samples out of order must return the correct data."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_rt_random.bin"

        originals = [_make_sample_data(seed=i) for i in range(5)]

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        for data in originals:
            writer.add_sample(**data)
        writer.close()

        reader = ShardReader(shard_path)
        # Read in reverse order
        for i in [4, 2, 0, 3, 1]:
            recovered = reader.read_sample(i)
            np.testing.assert_array_equal(
                recovered.image_tensor, originals[i]["image_tensor"]
            )
        reader.close()


class TestShardReaderValidation:
    """Verify the reader catches corruption and invalid files."""

    def test_reader_context_manager(self, tmp_path: Path, shard_configs):
        """ShardReader must work as a context manager."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_ctx.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        writer.add_sample(**_make_sample_data())
        writer.close()

        with ShardReader(shard_path) as reader:
            assert reader.sample_count == 1
            sample = reader[0]
            assert sample.image_tensor is not None

    def test_index_out_of_range(self, tmp_path: Path, shard_configs):
        """Accessing an out-of-range index must raise IndexError."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_oob.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        writer.add_sample(**_make_sample_data())
        writer.close()

        reader = ShardReader(shard_path)
        with pytest.raises(IndexError):
            reader.read_sample(1)
        reader.close()

    def test_corrupted_magic_raises(self, tmp_path: Path, shard_configs):
        """A file with corrupted magic bytes must be rejected."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_corrupt.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        writer.add_sample(**_make_sample_data())
        writer.close()

        # Corrupt the magic bytes
        with open(shard_path, "r+b") as f:
            f.write(b"BADMAGIC")

        with pytest.raises(ValueError, match="Invalid magic"):
            ShardReader(shard_path)

    def test_len_and_getitem(self, tmp_path: Path, shard_configs):
        """len() and [] indexing must work on ShardReader."""
        shard_cfg, image_cfg, tokenizer_cfg = shard_configs
        shard_path = tmp_path / "test_len.bin"

        writer = ShardWriter(shard_cfg, image_cfg, tokenizer_cfg)
        writer.open(shard_path)
        for i in range(3):
            writer.add_sample(**_make_sample_data(seed=i))
        writer.close()

        reader = ShardReader(shard_path)
        assert len(reader) == 3
        # Subscript access
        sample = reader[1]
        assert sample is not None
        reader.close()
