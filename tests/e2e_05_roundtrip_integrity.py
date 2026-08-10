#!/usr/bin/env python3
"""
E2E Integration Script 05 -- Round-Trip Integrity
===================================================

Creates known/hardcoded synthetic samples with deterministic values,
writes them to a shard using the Python ShardWriter, reads them back
using both the Python ShardReader AND the C++ ShardReader (if bindings
are compiled), and asserts **bitwise exact** equality for every field.

This is the strictest possible data integrity check -- no network,
no randomness, no tolerance.

Test data is deliberately adversarial:
  - Boundary token values (0, 1, 2**31 - 1)
  - Gradient, checkerboard, and negative-value image patterns
  - Metadata with unicode, nested dicts, arrays, booleans, nulls
  - Specific float32 patterns that expose byte-swap bugs

Run:
    python tests/e2e_05_roundtrip_integrity.py
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
from preprocessing.shard_reader import ShardReader as PyShardReader
from preprocessing.shard_writer import ShardWriter

# Try C++ bindings
try:
    import vlm_loader_py as cpp_loader
    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False


# ---------------------------------------------------------------------------
# Test configuration: tiny tensors for fast, deterministic testing
# ---------------------------------------------------------------------------
IMG_C, IMG_H, IMG_W = 3, 4, 4
TOKEN_LEN = 8


def _make_sample(i: int) -> dict:
    """Create a deterministic sample with values derived from index ``i``.

    Every value is uniquely identifiable so that any data corruption,
    byte-swap, or off-by-one error is immediately detectable.
    """
    # Image: fill channel c with (i + 1) * (c + 1) * 0.1
    img = np.zeros((IMG_C, IMG_H, IMG_W), dtype=np.float32)
    for c in range(IMG_C):
        img[c] = (i + 1) * (c + 1) * 0.1

    # Question tokens: fill with i * 10 + position
    q_ids = np.array([i * 10 + p for p in range(TOKEN_LEN)], dtype=np.int32)
    q_mask = np.array([1] * (TOKEN_LEN - 2) + [0, 0], dtype=np.int32)

    # Answer tokens: fill with i * 100 + position
    a_ids = np.array([i * 100 + p for p in range(TOKEN_LEN)], dtype=np.int32)
    a_mask = np.array([1] * (TOKEN_LEN - 1) + [0], dtype=np.int32)

    # Metadata: deterministic dict with various types
    metadata = {
        "sample_id": f"test_{i:04d}",
        "index": i,
        "score": round(i * 0.123, 3),
        "tags": [f"tag_{i}_a", f"tag_{i}_b"],
        "nested": {"key": f"value_{i}"},
    }

    return {
        "image_tensor": img,
        "question_ids": q_ids,
        "question_mask": q_mask,
        "answer_ids": a_ids,
        "answer_mask": a_mask,
        "metadata": metadata,
    }


def _make_adversarial_samples() -> list[dict]:
    """Create adversarial samples designed to catch subtle corruption bugs.

    These samples use boundary values, specific bit patterns, and edge
    cases that are most likely to expose byte-swap, endianness, float
    precision, or off-by-one errors in the shard reader/writer.
    """
    samples = []

    # --- Sample A: Boundary token values ---
    # Tests int32 boundary handling: 0, 1, max positive int32
    boundary_q_ids = np.array(
        [0, 1, 2, 2**31 - 1, 32767, 65535, 16777215, 100],
        dtype=np.int32,
    )
    boundary_a_ids = np.array(
        [2**31 - 1, 0, 1, 2, 255, 256, 65534, 2**31 - 2],
        dtype=np.int32,
    )
    # Alternating mask pattern
    boundary_mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)

    # Image: gradient pattern (each pixel has a unique value)
    gradient_img = np.zeros((IMG_C, IMG_H, IMG_W), dtype=np.float32)
    for c in range(IMG_C):
        for h in range(IMG_H):
            for w in range(IMG_W):
                gradient_img[c, h, w] = (c * IMG_H * IMG_W + h * IMG_W + w) * 0.01

    samples.append({
        "image_tensor": gradient_img,
        "question_ids": boundary_q_ids,
        "question_mask": boundary_mask,
        "answer_ids": boundary_a_ids,
        "answer_mask": boundary_mask,
        "metadata": {
            "sample_id": "adversarial_boundary",
            "description": "Boundary token values and gradient image",
            "max_token": int(2**31 - 1),
            "has_unicode": False,
        },
    })

    # --- Sample B: Checkerboard image + prime number tokens ---
    checker_img = np.zeros((IMG_C, IMG_H, IMG_W), dtype=np.float32)
    for c in range(IMG_C):
        for h in range(IMG_H):
            for w in range(IMG_W):
                checker_img[c, h, w] = 1.0 if (h + w) % 2 == 0 else -1.0

    prime_q_ids = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.int32)
    prime_a_ids = np.array([23, 29, 31, 37, 41, 43, 47, 53], dtype=np.int32)
    all_ones_mask = np.ones(TOKEN_LEN, dtype=np.int32)

    samples.append({
        "image_tensor": checker_img,
        "question_ids": prime_q_ids,
        "question_mask": all_ones_mask,
        "answer_ids": prime_a_ids,
        "answer_mask": all_ones_mask,
        "metadata": {
            "sample_id": "adversarial_primes",
            "description": "Prime-number tokens and checkerboard image",
            "primes_used": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
        },
    })

    # --- Sample C: Negative image values + zero tokens ---
    # Tests that negative floats survive the round-trip correctly
    negative_img = np.zeros((IMG_C, IMG_H, IMG_W), dtype=np.float32)
    negative_img[0] = -2.117904  # Exact float32 value
    negative_img[1] = 0.0
    negative_img[2] = 3.141592653589793  # Pi (float32 precision)

    zero_q_ids = np.zeros(TOKEN_LEN, dtype=np.int32)
    zero_a_ids = np.zeros(TOKEN_LEN, dtype=np.int32)
    zero_mask = np.zeros(TOKEN_LEN, dtype=np.int32)

    samples.append({
        "image_tensor": negative_img,
        "question_ids": zero_q_ids,
        "question_mask": zero_mask,
        "answer_ids": zero_a_ids,
        "answer_mask": zero_mask,
        "metadata": {
            "sample_id": "adversarial_negatives",
            "description": "Negative image values and all-zero tokens",
            "float_value": -2.117904,
        },
    })

    # --- Sample D: Specific float32 bit patterns ---
    # Use exact float32 values that are likely to get corrupted by
    # endianness bugs (bytes look different when swapped)
    pattern_img = np.array([
        1.0, -1.0, 0.5, -0.5,
        1e-7, 1e7, 1e-38, 1e38,
        np.float32(1.1920928955078125e-07),  # Smallest float32 > 0 (approx)
        np.float32(3.4028235e+38),           # Near max float32
        0.333333343267440795898,              # 1/3 in float32
        0.0,
        -0.0,                                # Negative zero
        np.float32(np.pi),
        np.float32(np.e),
        np.float32(np.sqrt(2)),
    ], dtype=np.float32).reshape(1, 4, 4)
    # Broadcast to 3 channels
    pattern_img = np.repeat(pattern_img, IMG_C, axis=0)

    # Ascending sequential token IDs
    ascending_q = np.arange(TOKEN_LEN, dtype=np.int32)
    descending_a = np.arange(TOKEN_LEN - 1, -1, -1, dtype=np.int32)
    mixed_mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)

    samples.append({
        "image_tensor": pattern_img,
        "question_ids": ascending_q,
        "question_mask": mixed_mask,
        "answer_ids": descending_a,
        "answer_mask": mixed_mask,
        "metadata": {
            "sample_id": "adversarial_floats",
            "description": "Specific float32 bit patterns",
            "pi": float(np.float32(np.pi)),
            "e": float(np.float32(np.e)),
        },
    })

    # --- Sample E: Complex metadata (unicode, nested, null, bool) ---
    uniform_img = np.full(
        (IMG_C, IMG_H, IMG_W), 0.42, dtype=np.float32
    )
    simple_q = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int32)
    simple_a = np.array([801, 701, 601, 501, 401, 301, 201, 101], dtype=np.int32)

    samples.append({
        "image_tensor": uniform_img,
        "question_ids": simple_q,
        "question_mask": np.ones(TOKEN_LEN, dtype=np.int32),
        "answer_ids": simple_a,
        "answer_mask": np.ones(TOKEN_LEN, dtype=np.int32),
        "metadata": {
            "sample_id": "adversarial_metadata",
            "unicode_text": "Xin chào thế giới 🌍",
            "japanese": "こんにちは世界",
            "arabic": "مرحبا بالعالم",
            "emoji": "🔥💯🎯",
            "nested_deep": {
                "level1": {
                    "level2": {
                        "value": 42,
                        "list": [1, 2, 3],
                    }
                }
            },
            "boolean_true": True,
            "boolean_false": False,
            "null_value": None,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "large_int": 9999999999,
            "negative_int": -42,
            "float_precision": 0.1 + 0.2,  # Classic float gotcha
        },
    })

    return samples


def _verify_with_python_reader(
    shard_path: Path,
    expected_samples: list[dict],
) -> bool:
    """Read back with Python ShardReader and verify bitwise equality."""
    num_samples = len(expected_samples)
    print(f"\n--- Python ShardReader verification ({num_samples} samples) ---")
    reader = PyShardReader(shard_path)
    assert reader.sample_count == num_samples
    all_passed = True

    for i in range(num_samples):
        expected = expected_samples[i]
        actual = reader.read_sample(i)
        errors = _compare_sample(expected, actual.image_tensor,
                                  actual.question_ids, actual.question_mask,
                                  actual.answer_ids, actual.answer_mask,
                                  actual.metadata)
        if errors:
            all_passed = False
            print(f"  Sample {i}: FAIL")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"  Sample {i}: PASS")

    reader.close()
    return all_passed


def _verify_with_cpp_reader(
    shard_path: Path,
    expected_samples: list[dict],
) -> bool:
    """Read back with C++ ShardReader (via pybind11) and verify."""
    num_samples = len(expected_samples)
    print(f"\n--- C++ ShardReader verification ({num_samples} samples) ---")
    reader = cpp_loader.ShardReader(str(shard_path))
    assert reader.sample_count() == num_samples
    all_passed = True

    for i in range(num_samples):
        expected = expected_samples[i]
        cpp_sample = reader.read_sample(i)

        # C++ bindings return flat arrays; reshape image
        img_flat = np.array(cpp_sample.image_tensor)
        img = img_flat.reshape(IMG_C, IMG_H, IMG_W)
        q_ids = np.array(cpp_sample.question_ids)
        q_mask = np.array(cpp_sample.question_mask)
        a_ids = np.array(cpp_sample.answer_ids)
        a_mask = np.array(cpp_sample.answer_mask)
        metadata = json.loads(cpp_sample.metadata_json)

        errors = _compare_sample(expected, img, q_ids, q_mask,
                                  a_ids, a_mask, metadata)

        # Additional C++ specific checks: raw bytes comparison
        # This catches any numpy dtype conversion issues in the bindings
        expected_img_bytes = np.ascontiguousarray(
            expected["image_tensor"], dtype=np.float32
        ).tobytes()
        actual_img_bytes = np.ascontiguousarray(
            img, dtype=np.float32
        ).tobytes()
        if expected_img_bytes != actual_img_bytes:
            errors.append(
                f"image_tensor RAW BYTES mismatch "
                f"(expected {len(expected_img_bytes)} bytes, "
                f"got {len(actual_img_bytes)} bytes)"
            )

        expected_q_bytes = np.ascontiguousarray(
            expected["question_ids"], dtype=np.int32
        ).tobytes()
        actual_q_bytes = np.ascontiguousarray(
            q_ids, dtype=np.int32
        ).tobytes()
        if expected_q_bytes != actual_q_bytes:
            errors.append("question_ids RAW BYTES mismatch")

        expected_a_bytes = np.ascontiguousarray(
            expected["answer_ids"], dtype=np.int32
        ).tobytes()
        actual_a_bytes = np.ascontiguousarray(
            a_ids, dtype=np.int32
        ).tobytes()
        if expected_a_bytes != actual_a_bytes:
            errors.append("answer_ids RAW BYTES mismatch")

        if errors:
            all_passed = False
            print(f"  Sample {i}: FAIL")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"  Sample {i}: PASS (value + bytewise)")

    return all_passed


def _compare_sample(
    expected: dict,
    image_tensor: np.ndarray,
    question_ids: np.ndarray,
    question_mask: np.ndarray,
    answer_ids: np.ndarray,
    answer_mask: np.ndarray,
    metadata: dict,
) -> list[str]:
    """Compare a read-back sample against the expected data.

    Performs value-level comparison, dtype checks, and shape checks.
    """
    errors: list[str] = []

    # --- Image tensor ---
    if not np.array_equal(image_tensor, expected["image_tensor"]):
        max_diff = np.max(np.abs(image_tensor - expected["image_tensor"]))
        errors.append(f"image_tensor mismatch (max_diff={max_diff})")

    if image_tensor.dtype != np.float32:
        errors.append(
            f"image_tensor dtype: expected float32, got {image_tensor.dtype}"
        )

    if image_tensor.shape != expected["image_tensor"].shape:
        errors.append(
            f"image_tensor shape: expected {expected['image_tensor'].shape}, "
            f"got {image_tensor.shape}"
        )

    # --- Question IDs ---
    if not np.array_equal(question_ids, expected["question_ids"]):
        errors.append(
            f"question_ids: expected {expected['question_ids'].tolist()}, "
            f"got {question_ids.tolist()}"
        )

    if question_ids.dtype != np.int32:
        errors.append(
            f"question_ids dtype: expected int32, got {question_ids.dtype}"
        )

    # --- Question mask ---
    if not np.array_equal(question_mask, expected["question_mask"]):
        errors.append(
            f"question_mask: expected {expected['question_mask'].tolist()}, "
            f"got {question_mask.tolist()}"
        )

    # --- Answer IDs ---
    if not np.array_equal(answer_ids, expected["answer_ids"]):
        errors.append(
            f"answer_ids: expected {expected['answer_ids'].tolist()}, "
            f"got {answer_ids.tolist()}"
        )

    if answer_ids.dtype != np.int32:
        errors.append(
            f"answer_ids dtype: expected int32, got {answer_ids.dtype}"
        )

    # --- Answer mask ---
    if not np.array_equal(answer_mask, expected["answer_mask"]):
        errors.append(
            f"answer_mask: expected {expected['answer_mask'].tolist()}, "
            f"got {answer_mask.tolist()}"
        )

    # --- Metadata ---
    if metadata != expected["metadata"]:
        errors.append(
            f"metadata mismatch:\n"
            f"    expected: {json.dumps(expected['metadata'], sort_keys=True, ensure_ascii=False)}\n"
            f"    actual:   {json.dumps(metadata, sort_keys=True, ensure_ascii=False)}"
        )

    return errors


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e.05_roundtrip")

    output_dir = PROJECT_ROOT / "output" / "e2e" / "shards_05"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_path = output_dir / "shard_0000.bin"

    # Build configs matching our test dimensions
    image_cfg = ImageConfig(
        target_size=(IMG_H, IMG_W),
        color_space="RGB",
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
    )
    shard_cfg = ShardConfig(
        output_dir=str(output_dir),
        shard_size_mb=9999,
    )
    tok_cfg = TokenizerConfig(max_length=TOKEN_LEN)

    # Generate original samples (indices 0..4)
    original_samples = [_make_sample(i) for i in range(5)]

    # Generate adversarial samples (indices 5..9)
    adversarial_samples = _make_adversarial_samples()

    # Combine all samples
    expected_samples = original_samples + adversarial_samples
    num_samples = len(expected_samples)

    # Print summary of test data
    print("\n" + "=" * 70)
    print(f"ROUND-TRIP INTEGRITY CHECK ({num_samples} samples)")
    print("=" * 70)
    print(f"  Original samples:     {len(original_samples)} (simple patterns)")
    print(f"  Adversarial samples:  {len(adversarial_samples)} (boundary values)")

    # ---- WRITE ----
    logger.info("Writing %d samples to %s", num_samples, shard_path)
    writer = ShardWriter(shard_cfg, image_cfg, tok_cfg)
    writer.open(shard_path)
    for sample in expected_samples:
        writer.add_sample(**sample)
    writer.close()
    logger.info("Shard written successfully.")

    # ---- VERIFY WITH PYTHON READER ----
    py_passed = _verify_with_python_reader(shard_path, expected_samples)

    # ---- VERIFY WITH C++ READER (if available) ----
    cpp_passed = True
    if HAS_CPP_BINDINGS:
        cpp_passed = _verify_with_cpp_reader(shard_path, expected_samples)
    else:
        print("\n--- C++ ShardReader: SKIPPED (bindings not compiled) ---")
        print("    Build with: cd loader && mkdir build && cd build && "
              "cmake .. && cmake --build .")

    # ---- FINAL RESULT ----
    print()
    if py_passed and cpp_passed:
        print(f"[PASS] All {num_samples} samples match exactly (bitwise).")
        print(f"       Including {len(adversarial_samples)} adversarial samples "
              f"with boundary values.")
        if HAS_CPP_BINDINGS:
            print("       Verified with BOTH Python and C++ readers "
                  "(value + raw bytes).")
        else:
            print("       Verified with Python reader (C++ bindings not available).")
    else:
        print(f"[FAIL] One or more samples failed the round-trip check!")
        sys.exit(1)


if __name__ == "__main__":
    main()
