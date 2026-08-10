"""
Binary shard writer for the VQA preprocessing pipeline.

Writes preprocessed samples (normalized images + tokenized text) into a
custom binary format designed for efficient, Python-free reading by the
C++ loader.

Shard File Layout
-----------------
The binary format is designed to be memory-mappable and random-accessible:

    [HEADER]
        magic:          8 bytes   "VLMSHARD" (ASCII)
        version:        4 bytes   uint32, currently 1
        sample_count:   4 bytes   uint32, number of samples in this shard
        offset_table_pos: 8 bytes uint64, byte offset of the offset table
        image_channels: 4 bytes   uint32, number of image channels (e.g. 3)
        image_height:   4 bytes   uint32, target image height
        image_width:    4 bytes   uint32, target image width
        token_length:   4 bytes   uint32, padded token sequence length
        reserved:       24 bytes  zeros, reserved for future fields
        (total header: 64 bytes, aligned to alignment_bytes)

    [SAMPLE RECORDS]
        For each sample, written sequentially:
            image_tensor:       C*H*W * 4 bytes   float32 in CHW layout
            question_ids:       T * 4 bytes        int32 token IDs
            question_mask:      T * 4 bytes        int32 attention mask
            answer_ids:         T * 4 bytes        int32 token IDs
            answer_mask:        T * 4 bytes        int32 attention mask
            metadata_length:    4 bytes            uint32, length of JSON
            metadata_json:      variable           UTF-8 JSON string
            padding:            0-63 bytes         zeros to next alignment

    [OFFSET TABLE]
        For each sample:
            offset:   8 bytes   uint64, byte offset from file start
            length:   8 bytes   uint64, byte length of the sample record

    [FOOTER]
        checksum:     4 bytes   uint32, CRC32 of everything before footer
        magic_end:    8 bytes   "SHARDEND" (ASCII)

Every sample record starts at an aligned boundary.  The offset table
enables O(1) random access to any sample without scanning the file.
"""

from __future__ import annotations

import json
import logging
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.config import ImageConfig, ShardConfig, TokenizerConfig

logger = logging.getLogger(__name__)

# Constants for the binary format
MAGIC_START = b"VLMSHARD"
MAGIC_END = b"SHARDEND"
FORMAT_VERSION = 1
HEADER_SIZE = 64  # Fixed header size in bytes


class ShardWriter:
    """Writes preprocessed VQA samples into binary shard files.

    Usage::

        writer = ShardWriter(shard_config, image_config, tokenizer_config)
        writer.open("output/shard_0000.bin")
        for sample in processed_samples:
            writer.add_sample(
                image_tensor=sample.image,
                question_ids=sample.q_ids,
                question_mask=sample.q_mask,
                answer_ids=sample.a_ids,
                answer_mask=sample.a_mask,
                metadata=sample.metadata,
            )
        writer.close()

    The writer tracks offsets internally and writes the offset table and
    footer when close() is called.
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

        # Per-shard state, initialized in open()
        self._file = None
        self._file_path: str | None = None
        self._sample_offsets: list[tuple[int, int]] = []  # (offset, length)
        self._bytes_written: int = 0

    def open(self, path: str | Path) -> None:
        """Open a new shard file and write the header.

        Parameters
        ----------
        path : str | Path
            Path to the shard file to create.
        """
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

        The offset_table_pos field is written as 0 initially and
        updated when close() is called, once the actual position
        is known.
        """
        target_h, target_w = self._image_config.target_size
        # Determine number of channels from color space
        color_space = self._image_config.color_space.upper()
        num_channels = 1 if color_space == "L" else 3

        header = struct.pack(
            "<8sIIQIIII24s",
            MAGIC_START,               # 8 bytes: magic
            FORMAT_VERSION,            # 4 bytes: version
            0,                         # 4 bytes: sample_count (filled at close)
            0,                         # 8 bytes: offset_table_pos (filled at close)
            num_channels,              # 4 bytes: image_channels
            target_h,                  # 4 bytes: image_height
            target_w,                  # 4 bytes: image_width
            self._tokenizer_config.max_length,  # 4 bytes: token_length
            b"\x00" * 24,              # 24 bytes: remaining reserved
        )

        # The struct above packs to exactly 64 bytes
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
    ) -> None:
        """Write a single preprocessed sample to the shard.

        Parameters
        ----------
        image_tensor : np.ndarray
            Float32 array of shape (C, H, W).
        question_ids : np.ndarray
            Int32 array of shape (seq_len,).
        question_mask : np.ndarray
            Int32 array of shape (seq_len,).
        answer_ids : np.ndarray
            Int32 array of shape (seq_len,).
        answer_mask : np.ndarray
            Int32 array of shape (seq_len,).
        metadata : dict | None
            Optional metadata dictionary, serialized as JSON.
        """
        if self._file is None:
            raise RuntimeError("ShardWriter is not open. Call open() first.")

        # Ensure alignment before writing the sample
        self._align()

        sample_start = self._bytes_written

        # Write image tensor (float32, CHW layout, contiguous)
        img_data = np.ascontiguousarray(image_tensor, dtype=np.float32).tobytes()
        self._file.write(img_data)
        self._bytes_written += len(img_data)

        # Write question token IDs and attention mask
        q_ids_data = np.ascontiguousarray(question_ids, dtype=np.int32).tobytes()
        self._file.write(q_ids_data)
        self._bytes_written += len(q_ids_data)

        q_mask_data = np.ascontiguousarray(question_mask, dtype=np.int32).tobytes()
        self._file.write(q_mask_data)
        self._bytes_written += len(q_mask_data)

        # Write answer token IDs and attention mask
        a_ids_data = np.ascontiguousarray(answer_ids, dtype=np.int32).tobytes()
        self._file.write(a_ids_data)
        self._bytes_written += len(a_ids_data)

        a_mask_data = np.ascontiguousarray(answer_mask, dtype=np.int32).tobytes()
        self._file.write(a_mask_data)
        self._bytes_written += len(a_mask_data)

        # Write metadata as JSON
        meta_json = json.dumps(metadata or {}, ensure_ascii=False).encode("utf-8")
        # Write the length prefix (4 bytes, uint32)
        self._file.write(struct.pack("<I", len(meta_json)))
        self._bytes_written += 4
        # Write the JSON bytes
        self._file.write(meta_json)
        self._bytes_written += len(meta_json)

        sample_length = self._bytes_written - sample_start
        self._sample_offsets.append((sample_start, sample_length))

    def close(self) -> None:
        """Finalize the shard: write offset table, update header, then footer."""
        if self._file is None:
            return

        # 1. Align before offset table
        self._align()
        offset_table_pos = self._bytes_written

        # 2. Write the offset table: array of (uint64 offset, uint64 length) pairs
        for offset, length in self._sample_offsets:
            self._file.write(struct.pack("<QQ", offset, length))
            self._bytes_written += 16

        # 3. Update header fields FIRST (9bel CRC32)
        sample_count = len(self._sample_offsets)
        self._file.seek(12)  # Skip magic (8) + version (4)
        self._file.write(struct.pack("<I", sample_count))
        self._file.write(struct.pack("<Q", offset_table_pos))

        # 4. Compute CRC32 of everything written so far
        self._file.flush()
        self._file.seek(0)
        content = self._file.read()
        checksum = zlib.crc32(content) & 0xFFFFFFFF

        # 5. Write footer
        self._file.seek(0, 2)  # Seek to end
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

        Returns True if the shard exceeds the size limit (shard_size_mb)
        or the sample-count limit (max_samples_per_shard), whichever is
        reached first.
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
