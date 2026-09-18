"""
Python-side binary shard reader for validation purposes.

Reads shard files produced by shard_writer.py and reconstructs the
original data (image tensors, token arrays, metadata). Its primary
purpose is round-trip validation: write with the writer, read back
with this reader, and verify exact equality.

The C++ loader (loader/) is the production reader. This Python reader
exists for testing and debugging, and mirrors the C++ parsing logic.
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
    FLAG_LZ4_COMPRESSED,
    FLAG_PER_SAMPLE_DIMS,
    FLAG_UINT8_STORAGE,
    FORMAT_VERSION,
    FORMAT_VERSION_V2,
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
    flags: int = 0


@dataclasses.dataclass
class ShardSample:
    """A single sample reconstructed from a shard file."""
    image_tensor: np.ndarray
    question_ids: np.ndarray
    question_mask: np.ndarray
    answer_ids: np.ndarray
    answer_mask: np.ndarray
    metadata: dict[str, Any]
    orig_height: int | None = None
    orig_width: int | None = None
    actual_height: int | None = None
    actual_width: int | None = None


class ShardReader:
    """Reads binary shard files produced by ShardWriter.

    Usage::

        reader = ShardReader("output/shard_0000.bin")
        sample = reader.read_sample(0)
        reader.close()

    Supports both sequential and random access via the offset table.
    """

    def __init__(self, path: str | Path) -> None:
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
        """Parse the 64-byte fixed header (v1 or v2)."""
        self._file.seek(0)
        raw = self._file.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise ValueError(
                f"Shard file too small for header: {len(raw)} < {HEADER_SIZE}"
            )

        magic = raw[:8]
        if magic != MAGIC_START:
            raise ValueError(
                f"Invalid magic: expected {MAGIC_START!r}, got {magic!r}"
            )

        version = struct.unpack_from("<I", raw, 8)[0]

        if version == FORMAT_VERSION:
            (
                magic, version, sample_count, offset_table_pos,
                channels, height, width, token_length, _reserved,
            ) = struct.unpack("<8sIIQIIII24s", raw)

            return ShardHeader(
                magic=magic, version=version,
                sample_count=sample_count,
                offset_table_pos=offset_table_pos,
                image_channels=channels,
                image_height=height, image_width=width,
                token_length=token_length, flags=0,
            )

        elif version == FORMAT_VERSION_V2:
            (
                magic, version, sample_count, offset_table_pos,
                channels, height, width, token_length, flags, _reserved,
            ) = struct.unpack("<8sIIQIIIII20s", raw)

            return ShardHeader(
                magic=magic, version=version,
                sample_count=sample_count,
                offset_table_pos=offset_table_pos,
                image_channels=channels,
                image_height=height, image_width=width,
                token_length=token_length, flags=flags,
            )

        else:
            raise ValueError(
                f"Unsupported version: expected {FORMAT_VERSION} or "
                f"{FORMAT_VERSION_V2}, got {version}"
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
        footer_pos = (
            self._header.offset_table_pos
            + self._header.sample_count * 16
        )
        self._file.seek(footer_pos)
        footer_data = self._file.read(12)
        if len(footer_data) < 12:
            raise ValueError("Truncated footer")

        stored_checksum = struct.unpack("<I", footer_data[:4])[0]
        end_magic = footer_data[4:]

        if end_magic != MAGIC_END:
            raise ValueError(
                f"Invalid end magic: expected {MAGIC_END!r}, got {end_magic!r}"
            )

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
        """Read a single sample by index using the offset table."""
        if index < 0 or index >= self._header.sample_count:
            raise IndexError(
                f"Sample index {index} out of range [0, {self._header.sample_count})"
            )

        offset, length = self._offsets[index]
        self._file.seek(offset)
        raw_data = self._file.read(length)

        h = self._header
        is_v2 = h.version >= FORMAT_VERSION_V2
        has_per_sample_dims = is_v2 and (h.flags & FLAG_PER_SAMPLE_DIMS)
        is_uint8 = is_v2 and (h.flags & FLAG_UINT8_STORAGE)
        is_lz4 = is_v2 and (h.flags & FLAG_LZ4_COMPRESSED)

        if is_lz4:
            import lz4.block
            uncompressed_size = struct.unpack_from("<I", raw_data)[0]
            buffer = lz4.block.decompress(
                raw_data[4:], uncompressed_size=uncompressed_size
            )
        else:
            buffer = raw_data

        buf_offset = 0

        orig_height = orig_width = actual_height = actual_width = None
        if has_per_sample_dims:
            orig_height, orig_width, actual_height, actual_width = (
                struct.unpack_from("<HHHH", buffer, buf_offset)
            )
            buf_offset += 8
            img_h, img_w = actual_height, actual_width
        else:
            img_h, img_w = h.image_height, h.image_width

        if is_uint8:
            img_size = h.image_channels * img_h * img_w
            image_tensor = np.frombuffer(
                buffer, dtype=np.uint8, count=img_size, offset=buf_offset,
            ).reshape(h.image_channels, img_h, img_w).copy().astype(np.float32) / 255.0
            buf_offset += img_size
        else:
            img_size = h.image_channels * img_h * img_w * 4
            image_tensor = np.frombuffer(
                buffer, dtype=np.float32, count=img_size // 4, offset=buf_offset,
            ).reshape(h.image_channels, img_h, img_w).copy()
            buf_offset += img_size

        token_count = h.token_length
        token_bytes = token_count * 4

        q_ids = np.frombuffer(
            buffer, dtype=np.int32, count=token_count, offset=buf_offset,
        ).copy()
        buf_offset += token_bytes

        q_mask = np.frombuffer(
            buffer, dtype=np.int32, count=token_count, offset=buf_offset,
        ).copy()
        buf_offset += token_bytes

        a_ids = np.frombuffer(
            buffer, dtype=np.int32, count=token_count, offset=buf_offset,
        ).copy()
        buf_offset += token_bytes

        a_mask = np.frombuffer(
            buffer, dtype=np.int32, count=token_count, offset=buf_offset,
        ).copy()
        buf_offset += token_bytes

        meta_len = struct.unpack_from("<I", buffer, buf_offset)[0]
        buf_offset += 4

        meta_json = buffer[buf_offset:buf_offset + meta_len].decode("utf-8")
        metadata = json.loads(meta_json)

        return ShardSample(
            image_tensor=image_tensor,
            question_ids=q_ids,
            question_mask=q_mask,
            answer_ids=a_ids,
            answer_mask=a_mask,
            metadata=metadata,
            orig_height=orig_height,
            orig_width=orig_width,
            actual_height=actual_height,
            actual_width=actual_width,
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