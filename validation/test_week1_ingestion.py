"""
Week 1 validation: Dataset ingestion.

Tests in this module verify that:
1. The ingestion module can download and parse the validation dataset.
2. Every ingested sample is a valid VQASample.
3. Images are decodable.
4. Column mapping correctly maps dataset columns to schema fields.
5. The max_samples limit works correctly.
"""

from __future__ import annotations

import pytest

from preprocessing.config import DatasetConfig, PipelineConfig
from preprocessing.ingest import ingest_dataset, ingest_row
from preprocessing.schema import VQASample


class TestIngestion:
    """Verify dataset ingestion against the real validation dataset."""

    def test_ingested_samples_not_empty(self, ingested_samples: list[VQASample]):
        """Ingestion must produce at least one sample."""
        assert len(ingested_samples) > 0, "No samples were ingested"

    def test_all_samples_are_vqa_sample(self, ingested_samples: list[VQASample]):
        """Every ingested object must be a VQASample instance."""
        for sample in ingested_samples:
            assert isinstance(sample, VQASample)

    def test_all_samples_pass_validation(self, ingested_samples: list[VQASample]):
        """Every ingested sample must pass schema validation."""
        for sample in ingested_samples:
            sample.validate()  # Should not raise

    def test_all_images_decodable(self, ingested_samples: list[VQASample]):
        """Every ingested sample's image must be decodable by Pillow."""
        for sample in ingested_samples:
            sample.validate_image_decodable()  # Should not raise

    def test_sample_ids_are_unique(self, ingested_samples: list[VQASample]):
        """All sample IDs within a batch must be unique."""
        ids = [s.sample_id for s in ingested_samples]
        assert len(ids) == len(set(ids)), "Duplicate sample IDs found"

    def test_questions_are_nonempty(self, ingested_samples: list[VQASample]):
        """Every sample must have a non-empty question."""
        for sample in ingested_samples:
            assert len(sample.question.strip()) > 0

    def test_answers_are_nonempty(self, ingested_samples: list[VQASample]):
        """Every sample must have a non-empty answer."""
        for sample in ingested_samples:
            assert len(sample.answer.strip()) > 0

    def test_image_dimensions_positive(self, ingested_samples: list[VQASample]):
        """Every sample must have positive image dimensions."""
        for sample in ingested_samples:
            assert sample.image_width > 0
            assert sample.image_height > 0

    def test_dataset_name_set(self, ingested_samples: list[VQASample]):
        """Every sample must have the dataset name from config."""
        for sample in ingested_samples:
            assert sample.dataset_name == (
                "trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning"
            )

    def test_max_samples_limit(self, pipeline_config: PipelineConfig):
        """Ingestion with max_samples=3 must produce exactly 3 samples."""
        # Build a config with max_samples=3
        ds_dict = {
            "hf_dataset_id": pipeline_config.dataset.hf_dataset_id,
            "hf_config_name": pipeline_config.dataset.hf_config_name,
            "split": pipeline_config.dataset.split,
            "column_mapping": pipeline_config.dataset.column_mapping,
            "max_samples": 3,
            "streaming": pipeline_config.dataset.streaming,
        }
        limited_config = PipelineConfig(
            dataset=DatasetConfig.from_dict(ds_dict),
            image=pipeline_config.image,
            tokenizer=pipeline_config.tokenizer,
            shard=pipeline_config.shard,
            num_workers=pipeline_config.num_workers,
            log_level=pipeline_config.log_level,
            seed=pipeline_config.seed,
        )
        samples = ingest_dataset(limited_config)
        assert len(samples) == 3
