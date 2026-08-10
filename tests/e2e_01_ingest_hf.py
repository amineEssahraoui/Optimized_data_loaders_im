#!/usr/bin/env python3
"""
E2E Integration Script 01 -- Ingest from HuggingFace
=====================================================

Loads a real HuggingFace dataset (preview split, 10 samples), converts
each row to the canonical VQASample schema, and prints/saves results
for visual verification.

Run:
    python tests/e2e_01_ingest_hf.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import load_config, PipelineConfig, DatasetConfig
from preprocessing.ingest import ingest_dataset
from PIL import Image


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e.01_ingest")

    # Load the default config, override to only fetch 10 samples
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    config = load_config(config_path)

    # Override max_samples to 10 for this test
    ds_overrides = {
        f.name: getattr(config.dataset, f.name)
        for f in config.dataset.__dataclass_fields__.values()
    }
    ds_overrides["max_samples"] = 10
    config = PipelineConfig(
        dataset=DatasetConfig(**ds_overrides),
        image=config.image,
        tokenizer=config.tokenizer,
        shard=config.shard,
        num_workers=config.num_workers,
        log_level=config.log_level,
        seed=config.seed,
    )

    # Ingest
    logger.info("=" * 60)
    logger.info("STAGE: Ingesting %d samples from HuggingFace", 10)
    logger.info("=" * 60)

    samples = ingest_dataset(config)

    logger.info("Ingested %d samples successfully.", len(samples))

    # Print details
    print("\n" + "=" * 70)
    print(f"{'IDX':>4}  {'SAMPLE_ID':<20}  {'IMG_W':>5} x {'IMG_H':<5}  "
          f"{'IMG_BYTES':>10}  Q_LEN  A_LEN")
    print("-" * 70)
    for i, s in enumerate(samples):
        print(
            f"{i:>4}  {s.sample_id:<20}  {s.image_width:>5} x {s.image_height:<5}  "
            f"{len(s.image_bytes):>10}  {len(s.question):>5}  {len(s.answer):>5}"
        )

    # Print first sample's text
    if samples:
        s = samples[0]
        print("\n--- Sample 0 detail ---")
        print(f"  Question: {s.question[:120]}{'...' if len(s.question) > 120 else ''}")
        print(f"  Answer:   {s.answer[:120]}{'...' if len(s.answer) > 120 else ''}")
        if s.metadata:
            for k, v in s.metadata.items():
                val_str = str(v)[:80]
                print(f"  Metadata[{k}]: {val_str}{'...' if len(str(v)) > 80 else ''}")

        # Save first image to disk
        out_dir = PROJECT_ROOT / "output" / "e2e"
        out_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(io.BytesIO(s.image_bytes))
        img_path = out_dir / "01_sample_image.png"
        img.save(img_path)
        print(f"\n  Saved sample image to: {img_path}")

    print("\n[PASS] Ingestion E2E test complete.")


if __name__ == "__main__":
    main()
