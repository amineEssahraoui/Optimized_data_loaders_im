# Configuration Reference

All configurable parameters for the VLM preprocessing pipeline are defined in `configs/pipeline.yaml`. This document provides a field-by-field reference.

## Top-Level Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_workers` | int | 4 | Number of parallel preprocessing workers (0 = single-threaded) |
| `log_level` | string | "INFO" | Logging verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `seed` | int | 42 | Global random seed for reproducibility |

## dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hf_dataset_id` | string | "trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning" | HuggingFace dataset identifier |
| `hf_config_name` | string | "preview" | HuggingFace dataset config/subset name |
| `split` | string | "train" | Dataset split to load |
| `column_mapping` | dict | See below | Maps canonical field names to dataset columns |
| `max_samples` | int or null | null | Limit samples to process (null = all) |
| `streaming` | bool | false | Use HuggingFace streaming mode |

### column_mapping defaults

| Canonical Key | Dataset Column |
|--------------|---------------|
| `image` | "image" |
| `question` | "question" |
| `answer` | "model_answer" |
| `id` | "id" |
| `reasoning` | "model_reasoning" |

## image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_size` | [int, int] | [384, 384] | Target (height, width) after resize/crop |
| `resize_strategy` | string | "resize_and_pad" | One of: "resize_and_pad", "center_crop", "resize" |
| `color_space` | string | "RGB" | Target color space: "RGB", "BGR", or "L" |
| `normalization_mean` | [float, ...] | [0.485, 0.456, 0.406] | Per-channel mean for normalization |
| `normalization_std` | [float, ...] | [0.229, 0.224, 0.225] | Per-channel std for normalization |
| `interpolation` | string | "bicubic" | Resize method: "bicubic", "bilinear", "lanczos", "nearest" |
| `pad_value` | int | 0 | Pixel value for padding (0-255) |

### Resize Strategies

- **resize_and_pad**: Resize longest side to target, pad shorter side (preserves aspect ratio)
- **center_crop**: Resize shortest side to target, center crop longer side (preserves aspect ratio)
- **resize**: Resize to exact target (may distort aspect ratio)

## tokenizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | string | "inceptionai/jais-13b-chat" | HuggingFace tokenizer model |
| `max_length` | int | 512 | Maximum sequence length |
| `padding` | string | "max_length" | Padding strategy: "max_length", "longest", "do_not_pad" |
| `truncation` | bool | true | Truncate sequences exceeding max_length |
| `trust_remote_code` | bool | true | Allow custom tokenizer code from model repo |
| `add_special_tokens` | bool | true | Add BOS/EOS special tokens |

## shard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | "./output/shards" | Directory for shard output files |
| `shard_size_mb` | int | 256 | Target shard size in megabytes |
| `max_samples_per_shard` | int or null | null | Maximum samples per shard (null = no limit) |
| `compression` | string or null | null | Compression: null (none) or "lz4" |
| `alignment_bytes` | int | 64 | Byte alignment for memory-mapped access (power of 2) |
