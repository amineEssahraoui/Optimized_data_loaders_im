#!/usr/bin/env python3
"""
E2E Integration Script 02 -- Preprocessing Check
==================================================

Ingests 5 real samples, runs the full image normalization and text
tokenization, then prints tensor shapes, value ranges, and token
statistics.  Also saves a denormalized image to verify visual
round-trip correctness.

Run:
    python tests/e2e_02_preprocess_check.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import (
    DatasetConfig,
    PipelineConfig,
    load_config,
)
from preprocessing.image_processor import denormalize_image, normalize_image
from preprocessing.ingest import ingest_dataset
from preprocessing.tokenizer import TextTokenizer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e.02_preprocess")

    # Load config, override to 5 samples
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    config = load_config(config_path)

    ds_overrides = {
        f.name: getattr(config.dataset, f.name)
        for f in config.dataset.__dataclass_fields__.values()
    }
    ds_overrides["max_samples"] = 5
    config = PipelineConfig(
        dataset=DatasetConfig(**ds_overrides),
        image=config.image,
        tokenizer=config.tokenizer,
        shard=config.shard,
        num_workers=config.num_workers,
        log_level=config.log_level,
        seed=config.seed,
    )

    # Stage 1: Ingest
    logger.info("=" * 60)
    logger.info("Ingesting 5 samples...")
    logger.info("=" * 60)
    samples = ingest_dataset(config)
    assert len(samples) > 0, "No samples ingested!"

    # Stage 2: Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = TextTokenizer(config.tokenizer)

    # Stage 3: Process each sample
    print("\n" + "=" * 70)
    print("PREPROCESSING RESULTS")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "output" / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i} (id={sample.sample_id}) ---")

        # Image normalization
        img_tensor = normalize_image(sample.image_bytes, config.image)
        print(f"  Image tensor shape:  {img_tensor.shape}")
        print(f"  Image dtype:         {img_tensor.dtype}")
        print(f"  Image value range:   [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        print(f"  Image mean/ch:       "
              f"{[f'{img_tensor[c].mean():.4f}' for c in range(img_tensor.shape[0])]}")

        # Tokenize question
        q_tok = tokenizer.tokenize(sample.question)
        print(f"  Question IDs shape:  {q_tok.input_ids.shape}")
        print(f"  Question non-pad:    {q_tok.attention_mask.sum()}")
        print(f"  Question ID range:   [{q_tok.input_ids.min()}, {q_tok.input_ids.max()}]")

        # Tokenize answer
        a_tok = tokenizer.tokenize(sample.answer)
        print(f"  Answer IDs shape:    {a_tok.input_ids.shape}")
        print(f"  Answer non-pad:      {a_tok.attention_mask.sum()}")
        print(f"  Answer ID range:     [{a_tok.input_ids.min()}, {a_tok.input_ids.max()}]")

        # Save denormalized image for first sample
        if i == 0:
            denorm = denormalize_image(img_tensor, config.image)
            img_out = Image.fromarray(denorm)
            img_path = out_dir / "02_denormalized.png"
            img_out.save(img_path)
            print(f"  Saved denormalized image: {img_path}")

    print(f"\n[PASS] Preprocessing check complete for {len(samples)} samples.")


if __name__ == "__main__":
    main()
