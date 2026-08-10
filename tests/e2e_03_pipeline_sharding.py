#!/usr/bin/env python3
"""
E2E Integration Script 03 -- Pipeline Sharding with max_samples_per_shard
==========================================================================

Runs the full preprocessing pipeline on 20 real samples with
``max_samples_per_shard=8``, verifying that multiple shard files
are created.  Reads back each shard header to confirm sample counts.

Run:
    python tests/e2e_03_pipeline_sharding.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.config import (
    DatasetConfig,
    PipelineConfig,
    ShardConfig,
    load_config,
)
from preprocessing.pipeline import run_pipeline
from preprocessing.shard_reader import ShardReader


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("e2e.03_sharding")

    # Load config and override for this test
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    config = load_config(config_path)

    # Output to a dedicated test directory
    output_dir = PROJECT_ROOT / "output" / "e2e" / "shards_03"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override: 20 samples, max 8 per shard
    ds_overrides = {
        f.name: getattr(config.dataset, f.name)
        for f in config.dataset.__dataclass_fields__.values()
    }
    ds_overrides["max_samples"] = 20

    shard_overrides = {
        f.name: getattr(config.shard, f.name)
        for f in config.shard.__dataclass_fields__.values()
    }
    shard_overrides["output_dir"] = str(output_dir)
    shard_overrides["max_samples_per_shard"] = 8

    config = PipelineConfig(
        dataset=DatasetConfig(**ds_overrides),
        image=config.image,
        tokenizer=config.tokenizer,
        shard=ShardConfig(**shard_overrides),
        num_workers=config.num_workers,
        log_level=config.log_level,
        seed=config.seed,
    )

    # Run the full pipeline
    logger.info("=" * 60)
    logger.info("Running pipeline: 20 samples, max_samples_per_shard=8")
    logger.info("=" * 60)

    summary = run_pipeline(config)

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Read back each shard and verify
    shard_files = sorted(output_dir.glob("shard_*.bin"))
    print(f"\nShard files found: {len(shard_files)}")
    total_read_back = 0

    print(f"\n{'SHARD':<30}  {'SAMPLES':>8}  {'C':>3}  {'H':>4}  {'W':>4}  {'T':>4}")
    print("-" * 60)
    for sf in shard_files:
        reader = ShardReader(sf)
        h = reader.header
        print(
            f"{sf.name:<30}  {h.sample_count:>8}  "
            f"{h.image_channels:>3}  {h.image_height:>4}  "
            f"{h.image_width:>4}  {h.token_length:>4}"
        )
        total_read_back += h.sample_count
        reader.close()

    # Assertions
    processed = summary["processed_samples"]
    expected_shards = (processed + 7) // 8  # ceil(processed / 8)

    print(f"\nTotal samples read back: {total_read_back}")
    print(f"Expected shards (ceil({processed}/8)): {expected_shards}")
    print(f"Actual shards: {len(shard_files)}")

    assert total_read_back == processed, (
        f"Sample count mismatch: wrote {processed}, read back {total_read_back}"
    )
    assert len(shard_files) == expected_shards, (
        f"Shard count mismatch: expected {expected_shards}, got {len(shard_files)}"
    )

    # Verify no shard (except possibly the last) exceeds 8 samples
    for sf in shard_files[:-1]:
        reader = ShardReader(sf)
        assert reader.sample_count == 8, (
            f"{sf.name} has {reader.sample_count} samples, expected 8"
        )
        reader.close()

    print("\n[PASS] Pipeline sharding E2E test complete.")


if __name__ == "__main__":
    main()
