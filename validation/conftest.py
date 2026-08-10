"""
Shared pytest fixtures for the cumulative validation suite.

Fixtures defined here are available to all test modules in the
validation/ directory without explicit imports.

Design notes:
- The config fixture loads the real pipeline.yaml so tests validate
  against the actual configuration, not a synthetic one.
- The dataset fixture is session-scoped to avoid downloading the
  dataset multiple times across test modules.
- Temporary directories for shard output are function-scoped to
  ensure test isolation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from preprocessing.config import PipelineConfig, load_config

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
# Repository root is two levels up from validation/conftest.py
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "pipeline.yaml"


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    """Load the real pipeline configuration from configs/pipeline.yaml.

    Session-scoped because the config is immutable (frozen dataclass)
    and does not change between tests.
    """
    return load_config(CONFIG_PATH)


@pytest.fixture
def config_path() -> Path:
    """Return the path to the pipeline configuration file."""
    return CONFIG_PATH


# ---------------------------------------------------------------------------
# Dataset fixtures (session-scoped to cache the download)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ingested_samples(pipeline_config: PipelineConfig):
    """Ingest the validation dataset and cache the result for the session.

    This fixture is session-scoped so the dataset is only downloaded
    and ingested once, even if multiple test modules use it.

    Uses max_samples=10 by default for faster test runs.  Set the
    environment variable VLM_TEST_FULL_DATASET=1 to ingest all rows.
    """
    from preprocessing.config import DatasetConfig
    from preprocessing.ingest import ingest_dataset

    # For CI / fast runs, limit to 10 samples unless full mode is requested
    full_mode = os.environ.get("VLM_TEST_FULL_DATASET", "0") == "1"

    if not full_mode:
        # Create a modified config with max_samples=10 for speed.
        # We cannot mutate frozen dataclasses, so we reconstruct.
        ds_dict = {
            "hf_dataset_id": pipeline_config.dataset.hf_dataset_id,
            "hf_config_name": pipeline_config.dataset.hf_config_name,
            "split": pipeline_config.dataset.split,
            "column_mapping": pipeline_config.dataset.column_mapping,
            "max_samples": 10,
            "streaming": pipeline_config.dataset.streaming,
        }
        test_config = PipelineConfig(
            dataset=DatasetConfig.from_dict(ds_dict),
            image=pipeline_config.image,
            tokenizer=pipeline_config.tokenizer,
            shard=pipeline_config.shard,
            num_workers=pipeline_config.num_workers,
            log_level=pipeline_config.log_level,
            seed=pipeline_config.seed,
        )
    else:
        test_config = pipeline_config

    return ingest_dataset(test_config)


@pytest.fixture
def shard_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for shard output, cleaned up after test."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    return shard_dir
