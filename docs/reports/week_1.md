# Week 1 Report: Canonical Schema, Config System, and Ingestion

## What Was Built

### Repository Skeleton
- Created the full directory structure: `preprocessing/`, `loader/`, `bindings/`, `configs/`, `validation/`, `docs/reports/`.

### Canonical VQA Schema (`preprocessing/schema.py`)
- Defined `VQASample` dataclass with all required fields: `sample_id`, `image_bytes`, `image_width`, `image_height`, `question`, `answer`, `metadata`, `dataset_name`.
- Built-in validation via `validate()` and `validate_image_decodable()` methods.
- JSON Schema constant (`VQA_JSON_SCHEMA`) for external validation and documentation.

### Config System (`preprocessing/config.py`, `configs/pipeline.yaml`)
- Hierarchical frozen dataclass config: `PipelineConfig` -> `DatasetConfig`, `ImageConfig`, `TokenizerConfig`, `ShardConfig`.
- YAML loading with deep merge support and list-to-tuple conversion.
- All parameters externalized: image processing, tokenizer model, shard layout, dataset source, column mapping, worker count, logging level, random seed.

### Dataset Ingestion (`preprocessing/ingest.py`)
- Downloads and parses the HuggingFace validation dataset (`trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning`, preview config, ~300 rows).
- Maps dataset columns to canonical schema via `column_mapping` from config.
- Handles PIL Image -> bytes conversion, optional `model_reasoning` -> metadata.
- Provides both list-based (`ingest_dataset`) and iterator-based (`ingest_dataset_iter`) APIs.
- Graceful error handling: invalid rows are logged and skipped.

## Cumulative Validation Suite

| Test Module | Tests | Status |
|-------------|-------|--------|
| `test_week1_schema.py` | 13 tests (validation, JSON schema, config) | Pass |
| `test_week1_ingestion.py` | 10 tests (ingestion, column mapping, limits) | Pass |

## Blockers
None.

## What Remains
- Week 2: Image normalization, tokenization, binary shard writer.
