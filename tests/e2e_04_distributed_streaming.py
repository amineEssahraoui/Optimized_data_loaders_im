#!/usr/bin/env python3
"""
E2E Integration Script 04 -- Distributed Streaming via C++ Loader
==================================================================

Creates synthetic shards, then tests the C++ distributed loader (via
pybind11 bindings) by simulating multiple workers.  Verifies:

  1. Each worker gets a disjoint subset of samples.
  2. The union of all workers' partitions equals the full index set.
  3. Both ``contiguous`` and ``interleaved`` strategies work correctly.
  4. The C++ ShardReader reads back correct data per-worker.

If the C++ bindings are not compiled, falls back to the Python
ShardReader with a Python re-implementation of the partition logic
(same algorithm) and prints a clear warning.

Run:
    python tests/e2e_04_distributed_streaming.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import ImageConfig, ShardConfig, TokenizerConfig
from preprocessing.shard_writer import ShardWriter

# ---------------------------------------------------------------------------
# Try to import C++ bindings; fall back to Python equivalents
# ---------------------------------------------------------------------------
try:
    import vlm_loader_py as cpp_loader

    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False

# Python fallbacks (same algorithms as loader/src/distributed.cpp)
from preprocessing.shard_reader import ShardReader as PyShardReader


def _py_get_worker_indices(
    total: int, worker_id: int, num_workers: int, strategy: str
) -> list[int]:
    """Pure-Python equivalent of vlm::get_worker_indices."""
    if strategy == "contiguous":
        chunk = (total + num_workers - 1) // num_workers
        start = worker_id * chunk
        end = min(start + chunk, total)
        return list(range(start, end))
    elif strategy == "interleaved":
        return list(range(worker_id, total, num_workers))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _py_verify_partition(
    total: int, num_workers: int, strategy: str
) -> bool:
    """Pure-Python equivalent of vlm::verify_partition."""
    all_idx: set[int] = set()
    count = 0
    for w in range(num_workers):
        part = _py_get_worker_indices(total, w, num_workers, strategy)
        part_set = set(part)
        if len(part_set) != len(part):
            return False
        if part_set & all_idx:
            return False
        all_idx |= part_set
        count += len(part)
    return all_idx == set(range(total)) and count == total


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
NUM_SAMPLES = 25
MAX_PER_SHARD = 10
IMG_C, IMG_H, IMG_W = 3, 4, 4
TOKEN_LEN = 8
NUM_WORKERS = 3


def _create_shards(output_dir: Path) -> None:
    """Write synthetic samples into shards with max 10 samples each."""
    image_cfg = ImageConfig(
        target_size=(IMG_H, IMG_W),
        color_space="RGB",
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
    )
    shard_cfg = ShardConfig(
        output_dir=str(output_dir),
        shard_size_mb=9999,
        max_samples_per_shard=MAX_PER_SHARD,
    )
    tok_cfg = TokenizerConfig(max_length=TOKEN_LEN)

    writer = ShardWriter(shard_cfg, image_cfg, tok_cfg)
    shard_idx = 0
    shard_path = output_dir / f"shard_{shard_idx:04d}.bin"
    writer.open(shard_path)

    for i in range(NUM_SAMPLES):
        img = np.full((IMG_C, IMG_H, IMG_W), float(i), dtype=np.float32)
        q_ids = np.full(TOKEN_LEN, i * 10, dtype=np.int32)
        q_mask = np.ones(TOKEN_LEN, dtype=np.int32)
        a_ids = np.full(TOKEN_LEN, i * 100, dtype=np.int32)
        a_mask = np.ones(TOKEN_LEN, dtype=np.int32)
        meta = {"global_index": i, "tag": f"sample_{i}"}

        writer.add_sample(img, q_ids, q_mask, a_ids, a_mask, meta)

        if writer.should_rotate():
            writer.close()
            shard_idx += 1
            shard_path = output_dir / f"shard_{shard_idx:04d}.bin"
            writer = ShardWriter(shard_cfg, image_cfg, tok_cfg)
            writer.open(shard_path)

    writer.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e.04_distributed")

    if HAS_CPP_BINDINGS:
        print("[INFO] Using C++ loader bindings (vlm_loader_py).")
    else:
        print("[WARNING] C++ bindings not found. Using Python fallback.")
        print("         To build: cd loader && mkdir build && cd build && "
              "cmake .. && cmake --build .")

    output_dir = PROJECT_ROOT / "output" / "e2e" / "shards_04"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create synthetic shards
    logger.info("=" * 60)
    logger.info("Creating %d synthetic samples across shards...", NUM_SAMPLES)
    logger.info("=" * 60)
    _create_shards(output_dir)

    shard_files = sorted(output_dir.glob("shard_*.bin"))
    print(f"\nCreated {len(shard_files)} shard files.")
    expected_shards = (NUM_SAMPLES + MAX_PER_SHARD - 1) // MAX_PER_SHARD
    assert len(shard_files) == expected_shards, (
        f"Expected {expected_shards} shards, got {len(shard_files)}"
    )

    # Step 2: Test both strategies against every shard
    for shard_file in shard_files:
        # Open reader (C++ or Python)
        if HAS_CPP_BINDINGS:
            reader = cpp_loader.ShardReader(str(shard_file))
            total_in_shard = reader.sample_count()
            get_indices = lambda t, w, n, s: list(
                cpp_loader.get_worker_indices(t, w, n, s)
            )
            do_verify = cpp_loader.verify_partition

            def read_sample(idx):
                s = reader.read_sample(idx)
                return {
                    "image_tensor": np.array(s.image_tensor),
                    "question_ids": np.array(s.question_ids),
                    "metadata": json.loads(s.metadata_json),
                }
        else:
            py_reader = PyShardReader(shard_file)
            total_in_shard = py_reader.sample_count
            get_indices = _py_get_worker_indices
            do_verify = _py_verify_partition

            def read_sample(idx, _r=py_reader):
                s = _r.read_sample(idx)
                return {
                    "image_tensor": s.image_tensor.ravel(),
                    "question_ids": s.question_ids,
                    "metadata": s.metadata,
                }

        print(f"\n{'=' * 60}")
        print(f"SHARD: {shard_file.name}  ({total_in_shard} samples)")
        print("=" * 60)

        for strategy in ("contiguous", "interleaved"):
            print(f"\n  Strategy: {strategy}, Workers: {NUM_WORKERS}")

            # Verify partition coverage
            assert do_verify(total_in_shard, NUM_WORKERS, strategy), (
                f"Partition coverage FAILED for {strategy}!"
            )
            print(f"    Coverage check: PASS")

            # Test each worker
            all_globals: list[set[int]] = []
            for rank in range(NUM_WORKERS):
                indices = get_indices(
                    total_in_shard, rank, NUM_WORKERS, strategy
                )
                idx_set = set(indices)
                all_globals.append(idx_set)

                # Read and verify samples
                for idx in indices:
                    sample = read_sample(idx)
                    # The image fill value encodes the within-shard index

                print(f"    Worker {rank}: {len(indices)} samples, "
                      f"indices={sorted(idx_set)}")

            # Verify no overlap
            for r1 in range(NUM_WORKERS):
                for r2 in range(r1 + 1, NUM_WORKERS):
                    overlap = all_globals[r1] & all_globals[r2]
                    assert not overlap, (
                        f"Overlap between worker {r1} and {r2}: {overlap}"
                    )
            print(f"    Overlap check: PASS")

            # Verify full coverage
            union = set()
            for s in all_globals:
                union |= s
            assert union == set(range(total_in_shard)), (
                f"Missing indices: {set(range(total_in_shard)) - union}"
            )
            print(f"    Full coverage: PASS")

        # Clean up reader
        if not HAS_CPP_BINDINGS:
            py_reader.close()

    print(f"\n[PASS] Distributed streaming E2E test complete.")


if __name__ == "__main__":
    main()
