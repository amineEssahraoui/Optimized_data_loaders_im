# VQA Schema Specification

## Overview

The canonical VQA (Visual Question Answering) schema defines the data structure that every VQA-type dataset must be normalized into before entering the preprocessing pipeline. This schema is the interface boundary between ingestion (dataset-specific) and processing (dataset-agnostic).

## Schema Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sample_id` | string | Yes | Unique identifier for this sample within the dataset |
| `image_bytes` | bytes | Yes | Raw image data in its original encoding (PNG, JPEG) |
| `image_width` | integer (>= 1) | Yes | Original image width in pixels |
| `image_height` | integer (>= 1) | Yes | Original image height in pixels |
| `question` | string (non-empty) | Yes | The question text associated with the image |
| `answer` | string (non-empty) | Yes | The answer text associated with the question |
| `metadata` | dict | No | Free-form dictionary for auxiliary data |
| `dataset_name` | string | Yes | Identifier of the source dataset |

## Validation Rules

All samples are validated at construction time:

1. `sample_id` must be a non-empty string.
2. `image_bytes` must be non-empty bytes.
3. `image_width` and `image_height` must be >= 1.
4. `question` must be a non-empty, non-whitespace-only string.
5. `answer` must be a non-empty, non-whitespace-only string.

Additionally, `validate_image_decodable()` can be called to verify that `image_bytes` can be decoded by Pillow.

## JSON Schema

A formal JSON Schema definition is available in `preprocessing/schema.py` as `VQA_JSON_SCHEMA`. It follows JSON Schema draft 2020-12.

## Column Mapping

The pipeline uses a configurable column mapping to translate dataset-specific column names to canonical schema fields. For the validation dataset (`trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning`):

| Canonical Field | Dataset Column |
|----------------|---------------|
| `image` | `image` |
| `question` | `question` |
| `answer` | `model_answer` |
| `id` | `id` |
| `reasoning` (metadata) | `model_reasoning` |

This mapping is defined in `configs/pipeline.yaml` and can be changed for different datasets without code modifications.

## Processed Schema (Post-Normalization)

After preprocessing, each sample contains:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `image_tensor` | float32 | (C, H, W) | Normalized image in CHW layout |
| `question_ids` | int32 | (T,) | Tokenized question token IDs |
| `question_mask` | int32 | (T,) | Question attention mask (0/1) |
| `answer_ids` | int32 | (T,) | Tokenized answer token IDs |
| `answer_mask` | int32 | (T,) | Answer attention mask (0/1) |
| `metadata` | JSON | variable | Includes sample_id, original dimensions |

Where C, H, W are determined by `image.target_size` and `image.color_space` in config, and T is `tokenizer.max_length`.
