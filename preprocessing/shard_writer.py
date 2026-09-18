"""
Binary shard writer for the VQA preprocessing pipeline.

Writes preprocessed samples (normalized images + tokenized text) into a
custom binary format designed for efficient, Python-free reading by the
C++ loader.

Shard File Layout
-----------------
    [HEADER]  64 bytes
        magic (8B), version (4B), sample_count (4B), offset_table_pos (8B),
        image_channels (4B), image_height (4B), image_width (4B),
        token_length (4B), flags/reserved (24B)

    [SAMPLE RECORDS]  sequential, aligned
        Per-sample dimension prefix (v2): 4 x uint16 = 8 bytes
        image_tensor: C*H*W bytes (uint8 or float32)
        question_ids, question_mask, answer_ids, answer_mask: 4 x T x int32
        metadata_length (4B) + metadata_json (variable)
        padding to alignment boundary

    [OFFSET TABLE]  16 bytes per sample (uint64 offset, uint64 length)

    [FOOTER]  12 bytes (CRC32 checksum + "SHARDEND" magic)
"""

from __future__ import annotations

import json
import logging
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from configs.config import ImageConfig, ShardConfig, TokenizerConfig

logger = logging.getLogger(__name__)

MAGIC_START = b"VLMSHARD"
MAGIC_END = b"SHARDEND"
FORMAT_VERSION = 1
FORMAT_VERSION_V2 = 2
HEADER_SIZE = 64

# V2 header flags (stored in bytes 40-43)
FLAG_UINT8_STORAGE = 0x01
FLAG_PER_SAMPLE_DIMS = 0x02
FLAG_LZ4_COMPRESSED = 0x04
FLAG_ZSTD_COMPRESSED = 0x08


class ShardWriter:
    """Writes preprocessed VQA samples into binary shard files.

    Usage::

        writer = ShardWriter(shard_config, image_config, tokenizer_config)
        writer.open("output/shard_0000.bin")
        for sample in processed_samples:
            writer.add_sample(...)
        writer.close()
    """

    def __init__(
        self,
        shard_config: ShardConfig,
        image_config: ImageConfig,
        tokenizer_config: TokenizerConfig,
    ) -> None:
        self._shard_config = shard_config
        self._image_config = image_config
        self._tokenizer_config = tokenizer_config
        self._alignment = shard_config.alignment_bytes
        self._format_version = shard_config.format_version
        self._use_lz4 = (
            shard_config.compression == "lz4"
            and shard_config.format_version >= 2
        )

        self._file = None
        self._file_path: str | None = None
        self._sample_offsets: list[tuple[int, int]] = []
        self._bytes_written: int = 0

    def open(self, path: str | Path) -> None:
        """Open a new shard file and write the header."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._file = open(path, "wb+")
        self._file_path = str(path)
        self._sample_offsets = []
        self._bytes_written = 0

        self._write_header()
        logger.info("Opened shard file: %s", path)

    def _write_header(self) -> None:
        """Write the 64-byte fixed header.

        sample_count and offset_table_pos are written as 0 initially
        and updated when close() is called.

        For v2, the reserved block is split into a 4-byte flags field
        followed by 20 bytes of remaining reserved space.
        """
        color_space = self._image_config.color_space.upper()
        num_channels = 1 if color_space == "L" else 3

        if self._format_version >= 2:
            max_dim = self._image_config.max_image_dim
            flags = 0
            if self._image_config.storage_dtype == "uint8":
                flags |= FLAG_UINT8_STORAGE
            if self._image_config.dynamic_padding:
                flags |= FLAG_PER_SAMPLE_DIMS
            if self._use_lz4:
                flags |= FLAG_LZ4_COMPRESSED

            header = struct.pack(
                "<8sIIQIIIII20s",
                MAGIC_START,
                FORMAT_VERSION_V2,
                0,
                0,
                num_channels,
                max_dim,
                max_dim,
                self._tokenizer_config.max_length,
                flags,
                b"\x00" * 20,
            )
        else:
            target_h, target_w = self._image_config.target_size
            header = struct.pack(
                "<8sIIQIIII24s",
                MAGIC_START,
                FORMAT_VERSION,
                0,
                0,
                num_channels,
                target_h,
                target_w,
                self._tokenizer_config.max_length,
                b"\x00" * 24,
            )

        assert len(header) == HEADER_SIZE, (
            f"Header size mismatch: expected {HEADER_SIZE}, got {len(header)}"
        )

        self._file.write(header)
        self._bytes_written = HEADER_SIZE

    def _align(self) -> None:
        """Write padding bytes to reach the next aligned boundary."""
        remainder = self._bytes_written % self._alignment
        if remainder != 0:
            pad_size = self._alignment - remainder
            self._file.write(b"\x00" * pad_size)
            self._bytes_written += pad_size

    def add_sample(
        self,
        image_tensor: np.ndarray,
        question_ids: np.ndarray,
        question_mask: np.ndarray,
        answer_ids: np.ndarray,
        answer_mask: np.ndarray,
        metadata: dict[str, Any] | None = None,
        *,
        orig_height: int | None = None,
        orig_width: int | None = None,
    ) -> None:
        """Write a single preprocessed sample to the shard.

        Parameters
        ----------
        image_tensor : np.ndarray
            Image array. v1: float32 (C,H,W). v2: uint8 (C,actual_H,actual_W).
        question_ids, question_mask, answer_ids, answer_mask : np.ndarray
            Int32 arrays of shape (seq_len,).
        metadata : dict | None
            Optional metadata dictionary, serialized as JSON.
        orig_height, orig_width : int | None
            Original image dimensions before resize (v2 only).
        """
        if self._file is None:
            raise RuntimeError("ShardWriter is not open. Call open() first.")

        self._align()
        sample_start = self._bytes_written

        raw = self._serialize_sample_fields(
            image_tensor, question_ids, question_mask,
            answer_ids, answer_mask, metadata,
            orig_height=orig_height, orig_width=orig_width,
        )

        if self._use_lz4:
            import lz4.block

            compressed = lz4.block.compress(raw, store_size=False)
            self._file.write(struct.pack("<I", len(raw)))
            self._file.write(compressed)
            self._bytes_written += 4 + len(compressed)
        else:
            self._file.write(raw)
            self._bytes_written += len(raw)

        sample_length = self._bytes_written - sample_start
        self._sample_offsets.append((sample_start, sample_length))

    def _serialize_sample_fields(
        self,
        image_tensor: np.ndarray,
        question_ids: np.ndarray,
        question_mask: np.ndarray,
        answer_ids: np.ndarray,
        answer_mask: np.ndarray,
        metadata: dict[str, Any] | None,
        *,
        orig_height: int | None = None,
        orig_width: int | None = None,
    ) -> bytes:
        """Serialize all sample fields into a contiguous byte buffer."""
        parts: list[bytes] = []

        if self._format_version >= 2:
            actual_h, actual_w = image_tensor.shape[1], image_tensor.shape[2]
            oh = orig_height if orig_height is not None else actual_h
            ow = orig_width if orig_width is not None else actual_w
            parts.append(struct.pack(
                "<HHHH",
                min(oh, 65535), min(ow, 65535), actual_h, actual_w,
            ))
            parts.append(
                np.ascontiguousarray(image_tensor, dtype=np.uint8).tobytes()
            )
        else:
            parts.append(
                np.ascontiguousarray(image_tensor, dtype=np.float32).tobytes()
            )

        parts.append(
            np.ascontiguousarray(question_ids, dtype=np.int32).tobytes()
        )
        parts.append(
            np.ascontiguousarray(question_mask, dtype=np.int32).tobytes()
        )
        parts.append(
            np.ascontiguousarray(answer_ids, dtype=np.int32).tobytes()
        )
        parts.append(
            np.ascontiguousarray(answer_mask, dtype=np.int32).tobytes()
        )

        meta_json = json.dumps(
            metadata or {}, ensure_ascii=False,
        ).encode("utf-8")
        parts.append(struct.pack("<I", len(meta_json)))
        parts.append(meta_json)

        return b"".join(parts)

    def close(self) -> None:
        """Finalize the shard: write offset table, update header, then footer."""
        if self._file is None:
            return

        self._align()
        offset_table_pos = self._bytes_written

        for offset, length in self._sample_offsets:
            self._file.write(struct.pack("<QQ", offset, length))
            self._bytes_written += 16

        sample_count = len(self._sample_offsets)
        self._file.seek(12)
        self._file.write(struct.pack("<I", sample_count))
        self._file.write(struct.pack("<Q", offset_table_pos))

        self._file.flush()
        self._file.seek(0)
        content = self._file.read()
        checksum = zlib.crc32(content) & 0xFFFFFFFF

        self._file.seek(0, 2)
        self._file.write(struct.pack("<I", checksum))
        self._file.write(MAGIC_END)

        self._file.flush()
        self._file.close()

        logger.info(
            "Closed shard file: %s (%d samples, %d bytes)",
            self._file_path,
            sample_count,
            self._bytes_written,
        )

        self._file = None
        self._file_path = None

    @property
    def current_size_bytes(self) -> int:
        """Return the number of bytes written so far."""
        return self._bytes_written

    @property
    def current_size_mb(self) -> float:
        """Return the number of megabytes written so far."""
        return self._bytes_written / (1024 * 1024)

    @property
    def sample_count(self) -> int:
        """Return the number of samples written so far."""
        return len(self._sample_offsets)

    def should_rotate(self) -> bool:
        """Check if the current shard has reached any configured limit.

        Returns True if size or sample-count limit is reached.
        """
        if self.current_size_mb >= self._shard_config.shard_size_mb:
            return True
        max_samples = self._shard_config.max_samples_per_shard
        if max_samples is not None and self.sample_count >= max_samples:
            return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
