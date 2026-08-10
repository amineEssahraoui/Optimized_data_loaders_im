# VLM Multimodal Preprocessing Pipeline

Image-side preprocessing pipeline for Vision-Language Model (VLM) post-training,
targeting the VQA (Visual Question Answering) task.

## Architecture

The system is split into two stages with a clear binary boundary:

| Stage | Language | Role |
|-------|----------|------|
| **Preprocessing** | Python | Ingest raw datasets, normalize to a canonical schema, tokenize text, serialize to binary shards |
| **Runtime Loader** | C++ | Read shards, decode content, build batches, handle distributed sharding across workers/nodes |
| **Bridge** | pybind11 | Development/testing only -- not a production dependency |

## Repository Layout

```
preprocessing/   Python ingestion, normalization, tokenization, shard writer
loader/          C++ shard reader, batching, sharding/distribution
bindings/        pybind11 bridge (dev/test only)
configs/         All YAML configuration files
validation/      Single cumulative validation suite
docs/            All documentation and weekly reports
```

## Sprint Plan (4 Weeks)

1. **Week 1** -- Canonical schema, config system, dataset ingestion
2. **Week 2** -- Image normalization, text tokenization, binary shard writer
3. **Week 3** -- C++ loader, pybind11 bindings, simulated distribution
4. **Week 4** -- End-to-end run, throughput measurement, final validation

## Validation Dataset

[trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning](https://huggingface.co/datasets/trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning)
(preview config, ~300 rows, columns: image, question, model_reasoning, model_answer, id)

## Tokenizer

[inceptionai/jais-13b-chat](https://huggingface.co/inceptionai/jais-13b-chat)
(loaded via config, swappable without code changes)

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the cumulative validation suite
pytest validation/ -v

# Process the dataset (Week 2+)
python -m preprocessing.ingest --config configs/pipeline.yaml
```

## Configuration

All configurable parameters live in `configs/pipeline.yaml`. See `configs/README.md` for
a field-by-field reference.
