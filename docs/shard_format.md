# Binary Shard Format Specification

Version: 1.0

## Overview

The binary shard format is designed for efficient, Python-free reading by the C++ runtime loader. It supports memory-mapped random access to individual samples through an offset table, and integrity checking via CRC32 checksum.

## File Layout

```
+--------------------------------------------------+
| HEADER (64 bytes, fixed)                         |
+--------------------------------------------------+
| SAMPLE RECORD 0 (variable, aligned)              |
| SAMPLE RECORD 1 (variable, aligned)              |
| ...                                              |
| SAMPLE RECORD N-1 (variable, aligned)            |
+--------------------------------------------------+
| OFFSET TABLE (N * 16 bytes)                      |
+--------------------------------------------------+
| FOOTER (12 bytes)                                |
+--------------------------------------------------+
```

## Header (64 bytes)

| Offset | Size | Type   | Field             | Description |
|--------|------|--------|-------------------|-------------|
| 0      | 8    | bytes  | magic             | `"VLMSHARD"` (ASCII) |
| 8      | 4    | uint32 | version           | Format version (currently `1`) |
| 12     | 4    | uint32 | sample_count      | Number of samples in this shard |
| 16     | 8    | uint64 | offset_table_pos  | Byte offset of the offset table |
| 24     | 4    | uint32 | image_channels    | Number of image channels (e.g. `3` for RGB) |
| 28     | 4    | uint32 | image_height      | Target image height in pixels |
| 32     | 4    | uint32 | image_width       | Target image width in pixels |
| 36     | 4    | uint32 | token_length      | Padded token sequence length |
| 40     | 24   | bytes  | reserved          | Reserved for future fields (zeros) |

All integers are little-endian.

## Sample Record

Each sample record starts at an aligned boundary (default: 64 bytes). The record contains:

| Order | Size | Type    | Field           | Description |
|-------|------|---------|-----------------|-------------|
| 1     | C*H*W*4 | float32[] | image_tensor  | Normalized image in CHW layout |
| 2     | T*4  | int32[] | question_ids     | Question token IDs |
| 3     | T*4  | int32[] | question_mask    | Question attention mask (0/1) |
| 4     | T*4  | int32[] | answer_ids       | Answer token IDs |
| 5     | T*4  | int32[] | answer_mask      | Answer attention mask (0/1) |
| 6     | 4    | uint32  | metadata_length  | Length of the metadata JSON in bytes |
| 7     | var  | bytes   | metadata_json    | UTF-8 JSON string |
| 8     | 0-63 | bytes   | padding          | Zeros to next alignment boundary |

Where:
- `C` = `image_channels`, `H` = `image_height`, `W` = `image_width`
- `T` = `token_length`

## Offset Table

Located at `offset_table_pos` (from the header). Contains one entry per sample:

| Size | Type   | Field  | Description |
|------|--------|--------|-------------|
| 8    | uint64 | offset | Byte offset of the sample record from file start |
| 8    | uint64 | length | Byte length of the sample record |

Total size: `sample_count * 16` bytes.

## Footer (12 bytes)

| Offset | Size | Type   | Field     | Description |
|--------|------|--------|-----------|-------------|
| 0      | 4    | uint32 | checksum  | CRC32 of all bytes before the footer |
| 4      | 8    | bytes  | magic_end | `"SHARDEND"` (ASCII) |

## Design Rationale

1. **Alignment**: Sample records start at aligned boundaries (configurable, default 64 bytes) to enable efficient memory-mapped access. 64-byte alignment matches typical CPU cache lines.

2. **Offset table at the end**: Written after all samples, so the writer does not need to know total sample count upfront. The header's `offset_table_pos` field enables jumping directly to the table.

3. **CRC32 checksum**: Covers all data before the footer. Detects file corruption during transfer or storage. Computed using the standard CRC-32/ISO 3309 polynomial (same as Python's `zlib.crc32`).

4. **No compression by default**: The format supports uncompressed data for fastest C++ reads. Compression (e.g. LZ4) can be added as a future extension via the reserved header fields.

5. **Fixed tensor dimensions**: All samples in a shard share the same image dimensions and token length (stored in the header). This simplifies the C++ reader and enables batch construction without per-sample shape checks.
