# Binary Shard Format Specification

Version: 2.0 (backward-compatible with v1)

## Overview

The binary shard format is designed for efficient, Python-free reading by the C++ runtime loader. It supports memory-mapped random access to individual samples through an offset table, and integrity checking via CRC32 checksum.

**Format v2** extends v1 with:
- **uint8 image storage** (4× I/O reduction vs float32)
- **Per-sample dimension headers** (variable-size images, no static padding)
- **Flags field** for feature toggles (compression, storage type)

The C++ reader supports both v1 and v2 seamlessly.

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

The header layout is identical for v1 and v2 — the only difference is how the reserved bytes at offset 40 are interpreted.

| Offset | Size | Type   | Field             | v1                              | v2                                  |
|--------|------|--------|-------------------|---------------------------------|-------------------------------------|
| 0      | 8    | bytes  | magic             | `"VLMSHARD"` (ASCII)            | Same                                |
| 8      | 4    | uint32 | version           | `1`                             | `2`                                 |
| 12     | 4    | uint32 | sample_count      | Number of samples               | Same                                |
| 16     | 8    | uint64 | offset_table_pos  | Byte offset of offset table     | Same                                |
| 24     | 4    | uint32 | image_channels    | Number of channels (e.g. `3`)   | Same                                |
| 28     | 4    | uint32 | image_height      | Fixed target height             | `max_image_dim` (resize cap)        |
| 32     | 4    | uint32 | image_width       | Fixed target width              | `max_image_dim` (resize cap)        |
| 36     | 4    | uint32 | token_length      | Padded token sequence length    | Same                                |
| 40     | 4    | uint32 | flags             | Reserved (zeros)                | Feature flags (see below)           |
| 44     | 20   | bytes  | reserved          | Reserved (zeros)                | Reserved (zeros)                    |

All integers are little-endian.

### Flags Field (v2, offset 40)

| Bit | Mask   | Name                | Description                                    |
|-----|--------|---------------------|------------------------------------------------|
| 0   | `0x01` | `FLAG_UINT8_STORAGE`   | Image data is uint8 (not float32)              |
| 1   | `0x02` | `FLAG_PER_SAMPLE_DIMS` | Each sample has an 8-byte dimension prefix     |
| 2   | `0x04` | `FLAG_LZ4_COMPRESSED`  | Sample data is LZ4 frame-compressed            |
| 3   | `0x08` | `FLAG_ZSTD_COMPRESSED` | Sample data is zstd-compressed                 |
| 4-31|        | Reserved            | Must be zero                                   |

## Sample Record

### v1 Sample Record

Each sample starts at an aligned boundary (default: 64 bytes):

| Order | Size        | Type      | Field           | Description                            |
|-------|-------------|-----------|-----------------|----------------------------------------|
| 1     | C×H×W×4     | float32[] | image_tensor    | Normalized image in CHW layout         |
| 2     | T×4         | int32[]   | question_ids    | Question token IDs                     |
| 3     | T×4         | int32[]   | question_mask   | Question attention mask (0/1)          |
| 4     | T×4         | int32[]   | answer_ids      | Answer token IDs                       |
| 5     | T×4         | int32[]   | answer_mask     | Answer attention mask (0/1)            |
| 6     | 4           | uint32    | metadata_length | Length of metadata JSON in bytes       |
| 7     | var         | bytes     | metadata_json   | UTF-8 JSON string                      |
| 8     | 0-63        | bytes     | padding         | Zeros to next alignment boundary       |

### v2 Sample Record

v2 adds an 8-byte dimension prefix and uses uint8 storage:

| Order | Size            | Type      | Field           | Description                          |
|-------|-----------------|-----------|-----------------|--------------------------------------|
| 1     | 8               | 4×uint16  | dimensions      | `orig_h, orig_w, actual_h, actual_w` |
| 2     | C×actual_H×actual_W | uint8[] | image_data     | Raw pixels in CHW layout (no norm)   |
| 3     | T×4             | int32[]   | question_ids    | Question token IDs                   |
| 4     | T×4             | int32[]   | question_mask   | Question attention mask (0/1)        |
| 5     | T×4             | int32[]   | answer_ids      | Answer token IDs                     |
| 6     | T×4             | int32[]   | answer_mask     | Answer attention mask (0/1)          |
| 7     | 4               | uint32    | metadata_length | Length of metadata JSON in bytes     |
| 8     | var             | bytes     | metadata_json   | UTF-8 JSON string                    |
| 9     | 0-63            | bytes     | padding         | Zeros to next alignment boundary     |

Key differences from v1:
- **Dimension prefix** (8 bytes): `orig_h` and `orig_w` are the original image dimensions before resize; `actual_h` and `actual_w` are the dimensions after aspect-ratio-preserving resize. All uint16.
- **uint8 storage**: Image pixels are stored as raw uint8 values (0-255), NOT normalized float32. Normalization `(pixel/255 - μ) / σ` is applied by the C++ loader at batch collation time.
- **Variable image size**: Each sample's image is `C × actual_h × actual_w` bytes (not fixed). The offset table handles this.

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

4. **No compression by default**: The format supports uncompressed data for fastest C++ reads. LZ4 or zstd compression can be enabled via the flags field (v2).

5. **v1: Fixed tensor dimensions**: All samples in a shard share the same image dimensions and token length (stored in the header). This simplifies the C++ reader.

6. **v2: Variable image dimensions**: Each sample stores its own dimensions. The header's `image_height`/`image_width` fields store the maximum possible dimension (the resize cap). Dynamic padding to per-batch `max(H) × max(W)` is performed at batch collation time.

7. **Deferred normalization (v2)**: Storing uint8 instead of float32 achieves a 4× I/O reduction. The ImageNet normalization `(pixel/255 - mean) / std` is applied by the C++ loader at batch assembly time, where SIMD vectorisation can be exploited.

8. **Backward compatibility**: The v1 and v2 headers share the same 64-byte size and byte layout for the first 40 bytes. The v2 reader accepts both versions. The v1 reader rejects v2 files via the version check.
