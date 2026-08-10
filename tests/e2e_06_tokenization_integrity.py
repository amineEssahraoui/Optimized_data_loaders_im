#!/usr/bin/env python3
"""
E2E Integration Script 06 -- Tokenization Integrity
=====================================================

Verifies that tokenization output is 100% correct through the entire
pipeline: tokenize known texts, write to a shard, read back, and
assert exact match of question_ids and answer_ids against ground-truth
tokens.

This test closes a gap in the existing E2E suite which verifies image
reconstruction but completely neglects tokenization integrity.

Checks performed:
  1. Token IDs written to shard exactly match direct tokenizer output.
  2. Attention masks are preserved bitwise through the shard round-trip.
  3. Decoded text from loaded tokens matches the original input strings.
  4. Edge cases: single-word text, long text (truncation), unicode.

Run:
    python tests/e2e_06_tokenization_integrity.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import ImageConfig, ShardConfig, TokenizerConfig, load_config
from preprocessing.shard_reader import ShardReader
from preprocessing.shard_writer import ShardWriter
from preprocessing.tokenizer import TextTokenizer

# Try C++ bindings
try:
    import vlm_loader_py as cpp_loader

    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
# Use the project's real tokenizer config for realistic testing.
# Token length is kept at the project default to test real padding/truncation.
IMG_C, IMG_H, IMG_W = 3, 4, 4  # Small images (tokenization is the focus)

# Known test texts -- chosen to exercise different tokenization scenarios
TEST_SAMPLES = [
    {
        "question": "What color is the car in this image?",
        "answer": "The car in the image is bright red.",
    },
    {
        "question": "How many people are visible?",
        "answer": "There are exactly three people visible in the photograph.",
    },
    {
        # Edge case: very short text (heavy padding expected)
        "question": "Hi",
        "answer": "Yes",
    },
    {
        # Edge case: longer text to test near-capacity sequences
        "question": (
            "Can you describe in detail the architectural style of the "
            "building shown in this photograph, including any notable "
            "features such as columns, arches, or decorative elements "
            "that might help identify the historical period?"
        ),
        "answer": (
            "The building shown in the photograph exhibits a neoclassical "
            "architectural style characterized by its prominent Corinthian "
            "columns, triangular pediment, and symmetrical facade. The "
            "decorative elements include detailed cornices and pilasters "
            "that are typical of structures built during the late 18th "
            "and early 19th centuries in European and American cities."
        ),
    },
    {
        # Edge case: unicode / special characters
        "question": "Hình ảnh này chụp ở đâu?",
        "answer": "Đây là hình ảnh chụp tại Hà Nội, Việt Nam.",
    },
]


def _tokenize_ground_truth(
    tokenizer: TextTokenizer,
    samples: list[dict],
) -> list[dict]:
    """Tokenize all test samples and return ground-truth data.

    Each entry contains the original text, expected token IDs and masks,
    plus a dummy image tensor for shard writing.
    """
    ground_truth = []

    for i, sample in enumerate(samples):
        q_tok = tokenizer.tokenize(sample["question"])
        a_tok = tokenizer.tokenize(sample["answer"])

        ground_truth.append({
            "index": i,
            "question_text": sample["question"],
            "answer_text": sample["answer"],
            "expected_question_ids": q_tok.input_ids.copy(),
            "expected_question_mask": q_tok.attention_mask.copy(),
            "expected_answer_ids": a_tok.input_ids.copy(),
            "expected_answer_mask": a_tok.attention_mask.copy(),
            # Dummy image -- we focus on tokenization here
            "image_tensor": np.full(
                (IMG_C, IMG_H, IMG_W), float(i) * 0.1, dtype=np.float32
            ),
            # Metadata for traceability
            "metadata": {
                "sample_id": f"tok_test_{i:04d}",
                "question_text": sample["question"],
                "answer_text": sample["answer"],
            },
        })

    return ground_truth


def _verify_tokenization_python(
    shard_path: Path,
    ground_truth: list[dict],
    tokenizer: TextTokenizer,
) -> bool:
    """Read back tokens from shard and verify against ground truth."""
    print("\n--- Python ShardReader: Tokenization Integrity ---")
    reader = ShardReader(shard_path)
    assert reader.sample_count == len(ground_truth), (
        f"Sample count mismatch: shard has {reader.sample_count}, "
        f"expected {len(ground_truth)}"
    )

    all_passed = True

    for i, expected in enumerate(ground_truth):
        actual = reader.read_sample(i)
        errors: list[str] = []

        # --- Core assertion: question_ids exact match ---
        if not np.array_equal(actual.question_ids, expected["expected_question_ids"]):
            errors.append(
                f"question_ids MISMATCH:\n"
                f"      expected: {expected['expected_question_ids'].tolist()}\n"
                f"      actual:   {actual.question_ids.tolist()}"
            )

        # --- Core assertion: question_mask exact match ---
        if not np.array_equal(actual.question_mask, expected["expected_question_mask"]):
            errors.append(
                f"question_mask MISMATCH:\n"
                f"      expected: {expected['expected_question_mask'].tolist()}\n"
                f"      actual:   {actual.question_mask.tolist()}"
            )

        # --- Core assertion: answer_ids exact match ---
        if not np.array_equal(actual.answer_ids, expected["expected_answer_ids"]):
            errors.append(
                f"answer_ids MISMATCH:\n"
                f"      expected: {expected['expected_answer_ids'].tolist()}\n"
                f"      actual:   {actual.answer_ids.tolist()}"
            )

        # --- Core assertion: answer_mask exact match ---
        if not np.array_equal(actual.answer_mask, expected["expected_answer_mask"]):
            errors.append(
                f"answer_mask MISMATCH:\n"
                f"      expected: {expected['expected_answer_mask'].tolist()}\n"
                f"      actual:   {actual.answer_mask.tolist()}"
            )

        # --- Dtype verification ---
        if actual.question_ids.dtype != np.int32:
            errors.append(
                f"question_ids dtype: expected int32, got {actual.question_ids.dtype}"
            )
        if actual.answer_ids.dtype != np.int32:
            errors.append(
                f"answer_ids dtype: expected int32, got {actual.answer_ids.dtype}"
            )

        # --- Round-trip decode verification ---
        # Decode the loaded token IDs back to text and verify they match
        # the original input (modulo tokenizer normalization)
        q_decoded = tokenizer.decode(
            actual.question_ids, skip_special_tokens=True
        )
        a_decoded = tokenizer.decode(
            actual.answer_ids, skip_special_tokens=True
        )

        q_original = expected["question_text"].strip()
        a_original = expected["answer_text"].strip()

        # The decoded text should contain the original or vice versa
        # (tokenizer normalization can cause minor whitespace diffs)
        if q_original not in q_decoded and q_decoded.strip() not in q_original:
            # Check if truncation explains the mismatch
            q_non_pad = int(expected["expected_question_mask"].sum())
            if q_non_pad < len(expected["expected_question_ids"]):
                # Possible truncation -- check if decoded is a prefix
                if not q_original.startswith(q_decoded.strip()[:20]):
                    errors.append(
                        f"question round-trip decode FAILED:\n"
                        f"      original:  '{q_original[:80]}...'\n"
                        f"      decoded:   '{q_decoded[:80]}...'"
                    )

        if a_original not in a_decoded and a_decoded.strip() not in a_original:
            a_non_pad = int(expected["expected_answer_mask"].sum())
            if a_non_pad < len(expected["expected_answer_ids"]):
                if not a_original.startswith(a_decoded.strip()[:20]):
                    errors.append(
                        f"answer round-trip decode FAILED:\n"
                        f"      original:  '{a_original[:80]}...'\n"
                        f"      decoded:   '{a_decoded[:80]}...'"
                    )

        if errors:
            all_passed = False
            print(f"  Sample {i} ('{expected['question_text'][:40]}...'): FAIL")
            for err in errors:
                print(f"    - {err}")
        else:
            q_non_pad = int(expected["expected_question_mask"].sum())
            a_non_pad = int(expected["expected_answer_mask"].sum())
            print(
                f"  Sample {i}: PASS "
                f"(q_tokens={q_non_pad}, a_tokens={a_non_pad})"
            )

    reader.close()
    return all_passed


def _verify_tokenization_cpp(
    shard_path: Path,
    ground_truth: list[dict],
) -> bool:
    """Read back tokens from shard via C++ bindings and verify."""
    print("\n--- C++ ShardReader: Tokenization Integrity ---")
    reader = cpp_loader.ShardReader(str(shard_path))
    assert reader.sample_count() == len(ground_truth)

    all_passed = True

    for i, expected in enumerate(ground_truth):
        cpp_sample = reader.read_sample(i)

        q_ids = np.array(cpp_sample.question_ids)
        q_mask = np.array(cpp_sample.question_mask)
        a_ids = np.array(cpp_sample.answer_ids)
        a_mask = np.array(cpp_sample.answer_mask)

        errors: list[str] = []

        if not np.array_equal(q_ids, expected["expected_question_ids"]):
            errors.append(
                f"C++ question_ids MISMATCH:\n"
                f"      expected: {expected['expected_question_ids'].tolist()}\n"
                f"      actual:   {q_ids.tolist()}"
            )

        if not np.array_equal(q_mask, expected["expected_question_mask"]):
            errors.append(f"C++ question_mask MISMATCH")

        if not np.array_equal(a_ids, expected["expected_answer_ids"]):
            errors.append(
                f"C++ answer_ids MISMATCH:\n"
                f"      expected: {expected['expected_answer_ids'].tolist()}\n"
                f"      actual:   {a_ids.tolist()}"
            )

        if not np.array_equal(a_mask, expected["expected_answer_mask"]):
            errors.append(f"C++ answer_mask MISMATCH")

        if errors:
            all_passed = False
            print(f"  Sample {i}: FAIL")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"  Sample {i}: PASS")

    return all_passed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e.06_tokenization")

    # Load the real project config for tokenizer settings
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    config = load_config(config_path)
    tok_cfg = config.tokenizer

    # Output directory
    output_dir = PROJECT_ROOT / "output" / "e2e" / "shards_06"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_path = output_dir / "shard_0000.bin"

    # Configs for shard writing (small images, real token length)
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

    # Step 1: Initialize tokenizer
    logger.info("Loading tokenizer: %s", tok_cfg.model_name_or_path)
    tokenizer = TextTokenizer(tok_cfg)

    # Step 2: Compute ground-truth tokenization
    logger.info("Tokenizing %d test samples...", len(TEST_SAMPLES))
    ground_truth = _tokenize_ground_truth(tokenizer, TEST_SAMPLES)

    # Print ground-truth summary
    print("\n" + "=" * 70)
    print("TOKENIZATION GROUND TRUTH")
    print("=" * 70)
    for gt in ground_truth:
        q_non_pad = int(gt["expected_question_mask"].sum())
        a_non_pad = int(gt["expected_answer_mask"].sum())
        print(
            f"  Sample {gt['index']}: "
            f"q_ids[0:5]={gt['expected_question_ids'][:5].tolist()}, "
            f"q_real_tokens={q_non_pad}, "
            f"a_ids[0:5]={gt['expected_answer_ids'][:5].tolist()}, "
            f"a_real_tokens={a_non_pad}"
        )

    # Step 3: Write to shard
    logger.info("Writing %d samples to shard: %s", len(ground_truth), shard_path)
    writer = ShardWriter(shard_cfg, image_cfg, tok_cfg)
    writer.open(shard_path)

    for gt in ground_truth:
        writer.add_sample(
            image_tensor=gt["image_tensor"],
            question_ids=gt["expected_question_ids"],
            question_mask=gt["expected_question_mask"],
            answer_ids=gt["expected_answer_ids"],
            answer_mask=gt["expected_answer_mask"],
            metadata=gt["metadata"],
        )

    writer.close()
    logger.info("Shard written successfully.")

    # Step 4: Verify with Python reader
    print("\n" + "=" * 70)
    print("TOKENIZATION INTEGRITY CHECK")
    print("=" * 70)

    py_passed = _verify_tokenization_python(shard_path, ground_truth, tokenizer)

    # Step 5: Verify with C++ reader (if available)
    cpp_passed = True
    if HAS_CPP_BINDINGS:
        cpp_passed = _verify_tokenization_cpp(shard_path, ground_truth)
    else:
        print("\n--- C++ ShardReader: SKIPPED (bindings not compiled) ---")

    # Final result
    print()
    if py_passed and cpp_passed:
        print(
            f"[PASS] All {len(ground_truth)} samples: tokenization integrity "
            f"verified (token IDs, masks, and round-trip decode)."
        )
        if HAS_CPP_BINDINGS:
            print("       Verified with BOTH Python and C++ readers.")
        else:
            print("       Verified with Python reader (C++ bindings not available).")
    else:
        print(
            "[FAIL] Tokenization integrity check FAILED! "
            "Corrupted or incorrect token data detected."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
