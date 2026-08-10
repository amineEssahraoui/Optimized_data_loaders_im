# Week 2 Report: Full Python Preprocessing Pipeline

## What Was Built

### Image Normalization (`preprocessing/image_processor.py`)
- Decode from raw bytes using Pillow.
- Three resize strategies, all config-driven:
  - `resize_and_pad`: preserves aspect ratio, pads shorter side.
  - `center_crop`: preserves aspect ratio, crops longer side.
  - `resize`: exact resize (may distort).
- Color space conversion: RGB, BGR, grayscale.
- Per-channel mean/std normalization with configurable stats.
- Output: float32 numpy array in CHW layout (channel-first).
- `denormalize_image()` for visual verification.

### Text Tokenization (`preprocessing/tokenizer.py`)
- Wraps `AutoTokenizer` from HuggingFace `transformers`.
- Loads the model specified in config (`inceptionai/jais-13b-chat`).
- Produces int32 numpy arrays for `input_ids` and `attention_mask`.
- Handles missing pad tokens by falling back to EOS token.
- Single-text and batch tokenization APIs.
- Round-trip decode for verification.

### Binary Shard Writer (`preprocessing/shard_writer.py`)
- Custom binary format documented in `docs/shard_format.md`.
- 64-byte header with magic, version, dimensions, offset table position.
- Per-sample records: image tensor (float32 CHW) + 4 token arrays (int32) + JSON metadata.
- Configurable alignment (default 64 bytes) for memory-mapped access.
- Offset table for O(1) random access.
- CRC32 checksum in footer for integrity verification.
- Automatic shard rotation when size limit reached.

### Python Shard Reader (`preprocessing/shard_reader.py`)
- Mirrors the C++ reader's parsing logic for validation.
- Parses header, offset table, and footer with checksum verification.
- Random access via offset table.
- Context manager support.

### Pipeline Orchestrator (`preprocessing/pipeline.py`)
- Connects ingestion -> normalization -> tokenization -> shard writing.
- CLI entry point: `python -m preprocessing.pipeline --config configs/pipeline.yaml`.
- Progress logging and error statistics.

## Cumulative Validation Suite

| Test Module | Tests | Status |
|-------------|-------|--------|
| `test_week1_schema.py` | 13 | Pass |
| `test_week1_ingestion.py` | 10 | Pass |
| `test_week2_image.py` | 10 | Pass |
| `test_week2_tokenizer.py` | 10 | Pass |
| `test_week2_shard.py` | 11 | Pass |
| **Total** | **54** | **Pass** |

## Blockers
None.

## What Remains
- Week 3: C++ loader, pybind11 bindings, distributed sharding.
