"""
Python-side binary shard reader for validation purposes.

This module reads the shard files produced by shard_writer.py and
reconstructs the original data (image tensors, token arrays, metadata).
Its primary purpose is round-trip validation: write data with the writer,
read it back with this reader, and verify exact equality.

The C++ loader (loader/) is the production reader.  This Python reader
exists only for testing and debugging, and mirrors the C++ reader's
parsing logic exactly so that any discrepancy indicates a bug.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.shard_writer import (
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC_END,
    MAGIC_START,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ShardHeader:
    """Parsed header from a shard file."""
    magic: bytes
    version: int
    sample_count: int
    offset_table_pos: int
    image_channels: int
    image_height: int
    image_width: int
    token_length: int


@dataclasses.dataclass
class ShardSample:
    """A single sample reconstructed from a shard file."""
    image_tensor: np.ndarray      # float32, shape (C, H, W)
    question_ids: np.ndarray      # int32, shape (T,)
    question_mask: np.ndarray     # int32, shape (T,)
    answer_ids: np.ndarray        # int32, shape (T,)
    answer_mask: np.ndarray       # int32, shape (T,)
    metadata: dict[str, Any]


class ShardReader:
    """Reads binary shard files produced by ShardWriter.

    Usage::

        reader = ShardReader("output/shard_0000.bin")
        print(reader.header.sample_count)
        sample = reader.read_sample(0)
        print(sample.image_tensor.shape)
        reader.close()

    Supports both sequential and random access via the offset table.
    """

    def __init__(self, path: str | Path) -> None:
        """Open a shard file and parse its header and offset table.

        Parameters
        ----------
        path : str | Path
            Path to the shard file.

        Raises
        ------
        ValueError
            If the file has an invalid magic, version, or checksum.
        """
        self._path = Path(path)
        self._file = open(self._path, "rb")

        self._header = self._read_header()
        self._offsets = self._read_offset_table()
        self._verify_footer()

        logger.info(
            "Opened shard: %s (%d samples, %dx%dx%d images, %d tokens)",
            self._path.name,
            self._header.sample_count,
            self._header.image_channels,
            self._header.image_height,
            self._header.image_width,
            self._header.token_length,
        )

    def _read_header(self) -> ShardHeader:
        """Parse the 64-byte fixed header."""
        self._file.seek(0)
        raw = self._file.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise ValueError(
                f"Shard file too small for header: {len(raw)} < {HEADER_SIZE}"
            )

        # Unpack matches the write order in ShardWriter._write_header
        (
            magic,
            version,
            sample_count,
            offset_table_pos,
            channels,
            height,
            width,
            token_length,
            _reserved_rest,
        ) = struct.unpack("<8sIIQIIII24s", raw)

        if magic != MAGIC_START:
            raise ValueError(
                f"Invalid magic: expected {MAGIC_START!r}, got {magic!r}"
            )

        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported version: expected {FORMAT_VERSION}, got {version}"
            )

        return ShardHeader(
            magic=magic,
            version=version,
            sample_count=sample_count,
            offset_table_pos=offset_table_pos,
            image_channels=channels,
            image_height=height,
            image_width=width,
            token_length=token_length,
        )

    def _read_offset_table(self) -> list[tuple[int, int]]:
        """Parse the offset table at the position specified in the header."""
        self._file.seek(self._header.offset_table_pos)
        offsets = []
        for _ in range(self._header.sample_count):
            raw = self._file.read(16)
            if len(raw) < 16:
                raise ValueError("Truncated offset table")
            offset, length = struct.unpack("<QQ", raw)
            offsets.append((offset, length))
        return offsets

    def _verify_footer(self) -> None:
        """Verify the CRC32 checksum and end magic in the footer."""
        # Footer position: after the offset table
        footer_pos = (
            self._header.offset_table_pos
            + self._header.sample_count * 16
        )
        self._file.seek(footer_pos)
        footer_data = self._file.read(12)  # 4 bytes checksum + 8 bytes magic
        if len(footer_data) < 12:
            raise ValueError("Truncated footer")

        stored_checksum = struct.unpack("<I", footer_data[:4])[0]
        end_magic = footer_data[4:]

        if end_magic != MAGIC_END:
            raise ValueError(
                f"Invalid end magic: expected {MAGIC_END!r}, got {end_magic!r}"
            )

        # Recompute CRC32 over everything before the footer
        self._file.seek(0)
        content = self._file.read(footer_pos)
        computed_checksum = zlib.crc32(content) & 0xFFFFFFFF

        if stored_checksum != computed_checksum:
            raise ValueError(
                f"CRC32 mismatch: stored={stored_checksum:#010x}, "
                f"computed={computed_checksum:#010x}"
            )

    @property
    def header(self) -> ShardHeader:
        """Return the parsed shard header."""
        return self._header

    @property
    def sample_count(self) -> int:
        """Return the number of samples in this shard."""
        return self._header.sample_count

    def read_sample(self, index: int) -> ShardSample:
        """Read a single sample by index using the offset table.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        ShardSample
            The reconstructed sample with all fields.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        if index < 0 or index >= self._header.sample_count:
            raise IndexError(
                f"Sample index {index} out of range [0, {self._header.sample_count})"
            )

        offset, _length = self._offsets[index]
        self._file.seek(offset)

        h = self._header

        # Read image tensor
        img_size = h.image_channels * h.image_height * h.image_width * 4
        img_data = self._file.read(img_size)
        image_tensor = np.frombuffer(img_data, dtype=np.float32).reshape(
            h.image_channels, h.image_height, h.image_width
        ).copy()

        # Read question token IDs
        token_bytes = h.token_length * 4
        q_ids = np.frombuffer(self._file.read(token_bytes), dtype=np.int32).copy()

        # Read question attention mask
        q_mask = np.frombuffer(self._file.read(token_bytes), dtype=np.int32).copy()

        # Read answer token IDs
        a_ids = np.frombuffer(self._file.read(token_bytes), dtype=np.int32).copy()

        # Read answer attention mask
        a_mask = np.frombuffer(self._file.read(token_bytes), dtype=np.int32).copy()

        # Read metadata
        meta_len = struct.unpack("<I", self._file.read(4))[0]
        meta_json = self._file.read(meta_len).decode("utf-8")
        metadata = json.loads(meta_json)

        return ShardSample(
            image_tensor=image_tensor,
            question_ids=q_ids,
            question_mask=q_mask,
            answer_ids=a_ids,
            answer_mask=a_mask,
            metadata=metadata,
        )

    def read_all(self) -> list[ShardSample]:
        """Read all samples from the shard sequentially."""
        return [self.read_sample(i) for i in range(self._header.sample_count)]

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __len__(self) -> int:
        return self._header.sample_count

    def __getitem__(self, index: int) -> ShardSample:
        return self.read_sample(index)
