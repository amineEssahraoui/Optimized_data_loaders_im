# VLM Multimodal Preprocessing Pipeline

A high-performance, config-driven preprocessing pipeline for Vision-Language Model (VLM)
post-training on Visual Question Answering (VQA) datasets. Converts raw HuggingFace
datasets into optimized binary shards that a zero-dependency C++ runtime loader can
consume directly.

## Architecture

The system is split into two stages with a clear binary boundary:

| Stage | Language | Role |
|-------|----------|------|
| **Preprocessing** | Python | Ingest raw datasets, normalize images, tokenize text, serialize to binary shards |
| **Runtime Loader** | C++ | Read shards via memory-mapping, decode content, build batches, handle distributed sharding |
| **Bridge** | pybind11 | Development/testing only -- not a production dependency |

All pipeline behavior is controlled exclusively through `configs/pipeline.yaml`. No
processing parameters are hardcoded. The configuration reader (`configs/config.py`)
resides in the same directory as the YAML file.

## Repository Layout

```
configs/               Configuration system (pipeline.yaml + config.py reader)
preprocessing/         Python ingestion, normalization, tokenization, shard I/O
  ingest.py            Dataset ingestion from HuggingFace
  image_processor.py   Image resize, crop, pad, normalize (v1 float32 / v2 uint8)
  tokenizer.py         Text tokenization with static or dynamic padding
  shard_writer.py      Binary shard writer (v1 + v2 formats)
  shard_reader.py      Python shard reader (validation/debugging only)
  shard_manifest.py    JSON manifest and distributed shard assignment
  pipeline.py          End-to-end orchestrator with multiprocessing support
  schema.py            Canonical VQASample dataclass
loader/                C++ shard reader, batching, async prefetch, SIMD normalization
  include/             Public headers (shard_reader.h, batch.h, async_loader.h, ...)
  src/                 Implementation (shard_reader.cpp, async_loader.cpp, ...)
  third_party/lz4/     Vendored LZ4 decompression library
bindings/              pybind11 bridge between C++ loader and Python (dev/test)
```

## Quick Start

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
pip install -e .

# Optional: LZ4 compression support
pip install -e ".[lz4]"
```

### Running the Pipeline

All behavior is controlled by `configs/pipeline.yaml`. Edit the file to configure
the dataset source, image processing, tokenization, shard output, and pipeline
parameters.

```bash
# Run with default config
python -m preprocessing.pipeline

# Run with a custom config
python -m preprocessing.pipeline --config path/to/your/config.yaml
```

### Building the C++ Loader

```bash
cd loader
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

This produces a static library (`vlm_loader.lib` / `libvlm_loader.a`) and optionally
the pybind11 Python module (`vlm_loader_py`) if pybind11 is installed.

## Configuration Reference

All parameters live in `configs/pipeline.yaml`. The configuration is organized into
five sections:

### dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hf_dataset_id` | string | (see YAML) | HuggingFace dataset identifier |
| `hf_config_name` | string | `"preview"` | Dataset configuration/subset name |
| `split` | string | `"train"` | Dataset split to load |
| `column_mapping` | dict | (see YAML) | Maps canonical fields to dataset columns |
| `max_samples` | int/null | `null` | Limit samples (null = all) |
| `streaming` | bool | `false` | Use HuggingFace streaming mode |

### image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | [int, int] | `[384, 384]` | Target (H, W) after resize/crop |
| `resize_strategy` | string | `"resize_and_pad"` | `resize_and_pad`, `center_crop`, or `resize` |
| `color_space` | string | `"RGB"` | `RGB`, `BGR`, or `L` |
| `normalization_mean` | [float] | `[0.485, 0.456, 0.406]` | Per-channel mean |
| `normalization_std` | [float] | `[0.229, 0.224, 0.225]` | Per-channel std |
| `interpolation` | string | `"bicubic"` | `bicubic`, `bilinear`, `lanczos`, `nearest` |
| `pad_value` | int | `0` | Pixel value for padding (0-255) |
| `max_image_dim` | int | `384` | Max dimension for aspect-ratio resize (v2) |
| `storage_dtype` | string | `"uint8"` | `float32` (v1) or `uint8` (v2) |
| `dynamic_padding` | bool | `true` | Dynamic image padding (v2) |

### tokenizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | string | `"inceptionai/jais-13b-chat"` | HuggingFace tokenizer |
| `max_length` | int | `512` | Maximum sequence length |
| `padding` | string | `"max_length"` | HuggingFace padding strategy |
| `truncation` | bool | `true` | Truncate sequences exceeding max_length |
| `trust_remote_code` | bool | `true` | Allow custom tokenizer code |
| `add_special_tokens` | bool | `true` | Add BOS/EOS tokens |
| `dynamic_text_padding` | bool | `false` | Dynamic text padding (see below) |

**Text Padding Modes:**
- **Static** (`dynamic_text_padding: false`): Every sequence is padded to `max_length` at
  preprocessing time. Simpler, fixed-size token arrays in shards.
- **Dynamic** (`dynamic_text_padding: true`): Sequences are tokenized without padding,
  then padded to `max_length` for shard format compatibility. Actual token lengths are
  stored in sample metadata (`actual_question_length`, `actual_answer_length`), enabling
  the loader to reconstruct tight batches at collation time.

### shard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | `"./output/shards"` | Shard output directory |
| `shard_size_mb` | int | `256` | Target shard size in MB |
| `max_samples_per_shard` | int/null | `null` | Max samples per shard |
| `compression` | string/null | `null` | `null` (none) or `"lz4"` |
| `alignment_bytes` | int | `64` | Byte alignment (power of 2) |
| `format_version` | int | `2` | Shard format: 1 (legacy) or 2 (recommended) |
| `manifest_path` | string/null | (see YAML) | Manifest JSON output path |

### Pipeline-Level

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shuffling` | string | `"none"` | `none`, `local`, or `global` |
| `num_workers` | int | `4` | Parallel preprocessing workers (0 = sequential) |
| `log_level` | string | `"INFO"` | Logging verbosity |
| `seed` | int | `42` | Global random seed |
| `progress_interval` | int | `50` | Log progress every N samples |
| `streaming_chunk_size` | int | `500` | Default chunk size for streaming pipeline |
| `fail_fast` | bool | `false` | Abort on first error vs skip and continue |

## Binary Shard Format

The binary shard format is designed for memory-mapped, random-access reading by the
C++ loader without any Python dependency.

### Format v2 (Recommended)

- **Header**: 64 bytes (magic, version, counts, dimensions, flags)
- **Samples**: Sequential, aligned records with per-sample dimension prefix
- **Offset Table**: O(1) random access to any sample
- **Footer**: CRC32 integrity checksum

Format v2 stores images as raw uint8 CHW arrays with per-sample dimensions, deferring
normalization to the C++ loader where SIMD (AVX2 + FMA) acceleration is available.
This reduces shard I/O by 4x compared to float32 storage.

### Compression

Optional LZ4 compression can be enabled per-shard (`shard.compression: "lz4"`).
Each sample record is individually compressed, allowing random access without
decompressing the entire file.

## Distributed Training

The `shard_manifest.py` module provides:

- `build_manifest(shard_dir)`: Scan a shard directory and build a JSON manifest
- `assign_shards(manifest, rank, world_size)`: Deterministic round-robin shard
  assignment for PyTorch DDP

## License

Apache-2.0
