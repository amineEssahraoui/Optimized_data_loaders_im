# Week 4 Report — Dynamic Sharding, Distributed C++ Loader & E2E Integration

## Overview

This week adds three major capabilities to the VLM preprocessing pipeline:

1. **Dynamic Sharding** — sample-count-based shard rotation
2. **Intrinsically Distributed C++ Loader** — data partitioning built into the loader
3. **End-to-End Integration Tests** — real-data stress tests (not mocked unit tests)

---

## 1. Dynamic Sharding (`max_samples_per_shard`)

### Problem

The existing pipeline only supported **size-based** shard rotation (`shard_size_mb`). For workloads where sample count matters more than file size (e.g., balanced training batches, reproducible data splits), there was no way to cap the number of samples per shard.

### Solution

| File | Change |
|------|--------|
| `configs/pipeline.yaml` | Added `max_samples_per_shard: null` under `shard:` |
| `preprocessing/config.py` | Added `max_samples_per_shard: int \| None` to `ShardConfig` |
| `preprocessing/shard_writer.py` | Updated `should_rotate()` to check both size and sample-count limits |
| `preprocessing/pipeline.py` | Fixed bytes-accounting bug in shard rotation (bytes were captured *after* `close()` cleared state) |

### How It Works

`ShardWriter.should_rotate()` now returns `True` if **either** limit is reached:

```python
def should_rotate(self) -> bool:
    if self.current_size_mb >= self._shard_config.shard_size_mb:
        return True
    max_samples = self._shard_config.max_samples_per_shard
    if max_samples is not None and self.sample_count >= max_samples:
        return True
    return False
```

Setting `max_samples_per_shard: null` (the default) preserves backward compatibility — only size-based rotation applies.

---

## 2. Intrinsically Distributed C++ Loader

### Architecture

The distributed loading logic is implemented entirely in C++ within the `loader/` directory, following the project's domain separation principle:

```
loader/
├── include/
│   ├── batch.h              # Sample + Batch structs (pre-existing)
│   ├── shard_reader.h        # ShardFileHeader + ShardReader class
│   └── distributed.h         # Partitioning utilities
├── src/
│   ├── shard_reader.cpp      # Binary format parser + reader
│   └── distributed.cpp       # Contiguous & interleaved partitioning
└── CMakeLists.txt            # Build configuration (pre-existing)
```

### ShardReader (`shard_reader.h` / `shard_reader.cpp`)

The C++ `ShardReader` mirrors the Python `shard_reader.py` and reads the same binary format:

- **Constructor**: Opens file, validates header magic/version, parses offset table, verifies CRC32 checksum
- **`read_sample(index)`**: O(1) random access via offset table
- **`read_batch(indices)`**: Reads multiple samples into a contiguous `Batch` struct (NCHW layout)
- **CRC32**: Uses the ISO 3309 polynomial (0xEDB88320), matching Python's `zlib.crc32` exactly

### Distributed Partitioning (`distributed.h` / `distributed.cpp`)

The partitioning is **intrinsic to the loader** — it's not an external orchestrator. Each worker computes its own disjoint index set locally with zero inter-worker communication.

Two strategies are supported:

| Strategy | Assignment Rule | Best For |
|----------|----------------|----------|
| `contiguous` | Worker *r* gets indices `[r*⌈N/W⌉, (r+1)*⌈N/W⌉)` | Sequential I/O locality |
| `interleaved` | Worker *r* gets indices `r, r+W, r+2W, …` | Load balance with variable sample sizes |

Both guarantee: `∪ partitions = {0, …, N−1}` and `∩ = ∅`.

**API:**
```cpp
std::vector<uint32_t> get_worker_indices(
    uint32_t total_samples, uint32_t worker_id,
    uint32_t num_workers, const std::string& strategy);

bool verify_partition(
    uint32_t total_samples, uint32_t num_workers,
    const std::string& strategy);
```

### Python Bindings

The pre-existing `bindings/bind_loader.cpp` exposes both classes to Python via pybind11. Build with:

```bash
cd loader
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
cmake --build .
```

---

## 3. End-to-End Integration Tests

All E2E tests live in `tests/` and are standalone scripts (not pytest mocks):

| Script | Purpose | Network? |
|--------|---------|----------|
| `e2e_01_ingest_hf.py` | Load 10 real samples from HuggingFace, print metadata, save image | Yes |
| `e2e_02_preprocess_check.py` | Normalize images + tokenize text, print tensor stats | Yes |
| `e2e_03_pipeline_sharding.py` | Full pipeline with `max_samples_per_shard=8`, verify shard counts | Yes |
| `e2e_04_distributed_streaming.py` | Test distributed partitioning (C++ or Python fallback) | No |
| `e2e_05_roundtrip_integrity.py` | Bitwise round-trip verification with hardcoded data | No |

### Running

```bash
# Offline tests (no network, no HF download):
python tests/e2e_05_roundtrip_integrity.py
python tests/e2e_04_distributed_streaming.py

# Online tests (download HF dataset + tokenizer on first run):
python tests/e2e_01_ingest_hf.py
python tests/e2e_02_preprocess_check.py
python tests/e2e_03_pipeline_sharding.py
```

### Key Design Decisions

- **No mocks**: Every test exercises real I/O — real binary files, real shard parsing, real CRC32 verification.
- **C++ fallback**: Tests 04 and 05 try to import `vlm_loader_py` (C++ bindings). If unavailable, they fall back to the Python `ShardReader` with identical partition logic, printing a clear warning.
- **Deterministic synthetic data**: Tests 04 and 05 use hardcoded values (image fill = sample index, token IDs = `i*10+position`) so any data corruption, byte-swap, or off-by-one error is immediately detectable.

---

## Bug Fix: Pipeline Bytes Accounting

The existing `pipeline.py` had a bytes-accounting bug in shard rotation:

```python
# BUG: close() resets internal state, so current_size_bytes is 0 after this
writer.close()
total_bytes += writer.current_size_bytes  # always 0!
```

Fixed to capture bytes **before** closing:

```python
total_bytes += writer.current_size_bytes  # capture while still valid
writer.close()
```
