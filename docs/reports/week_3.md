# Week 3 Report: C++ Loader and pybind11 Bindings

## What Was Built

### C++ Shard Reader (`loader/src/shard_reader.cpp`, `loader/include/shard_reader.h`)
- Memory-mapped file reader using platform-specific APIs:
  - Windows: `CreateFileMapping` / `MapViewOfFile`.
  - POSIX: `mmap`.
- Parses the same binary format written by Python, with identical validation:
  - Magic byte verification.
  - Version check.
  - Offset table parsing.
  - CRC32 checksum verification (table-driven, same polynomial as Python `zlib.crc32`).
- Random access to individual samples via offset table.
- Batch reading: collects multiple samples into contiguous NCHW arrays.
- Thread-safe for concurrent reads (read-only mmap).
- Move-only semantics (non-copyable).

### Batch Structure (`loader/include/batch.h`)
- `Sample` struct: per-sample vectors for image tensor, token arrays, metadata JSON.
- `Batch` struct: contiguous storage in NCHW layout for GPU-friendly transfer.
- `Batch::from_samples()` factory for assembling batches from individual samples.

### Distributed Sharding (`loader/src/distributed.cpp`, `loader/include/distributed.h`)
- Two partitioning strategies:
  - **Contiguous**: each worker gets a sequential block of indices.
  - **Interleaved**: round-robin distribution across workers.
- Both guarantee: no overlap, no gaps, full coverage.
- `verify_partition()` utility for testing.
- Handles edge cases: 1 worker, more workers than samples.

### pybind11 Bindings (`bindings/bind_loader.cpp`)
- Exposes `ShardReader`, `Sample`, `Batch`, `ShardFileHeader` to Python.
- Numpy integration: returns image tensors and token arrays as numpy arrays.
- Distribution utilities: `get_worker_indices()`, `verify_partition()`.
- Development/testing only, not a production dependency.

### CMake Build System
- `loader/CMakeLists.txt`: builds `vlm_loader` static library (C++17).
- `bindings/CMakeLists.txt`: builds `vlm_loader_py` pybind11 module.
- Cross-platform: MSVC and GCC/Clang support.
- Optional bindings: gracefully skips if pybind11 not found.

## Cumulative Validation Suite

| Test Module | Tests | Status |
|-------------|-------|--------|
| `test_week1_schema.py` | 13 | Pass |
| `test_week1_ingestion.py` | 10 | Pass |
| `test_week2_image.py` | 10 | Pass |
| `test_week2_tokenizer.py` | 10 | Pass |
| `test_week2_shard.py` | 11 | Pass |
| `test_week3_cpp_roundtrip.py` | 9 | Skipped (requires C++ build) |
| `test_week3_distributed.py` | 14 Python-only + 10 C++ | Python: Pass, C++: Skipped |
| **Total** | **~87** | **Pass (Python), C++ tests pending build** |

Note: C++ tests require building the pybind11 module. Build instructions:
```bash
pip install pybind11
cd loader && mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
cmake --build .
```

## Blockers
- C++ tests depend on pybind11 module being built and importable.

## What Remains
- Week 4: End-to-end run, throughput measurement, final documentation.
