"""
Week 1 validation: Schema correctness and config loading.

Tests in this module verify that:
1. The VQASample dataclass correctly validates well-formed samples.
2. Invalid samples are rejected with appropriate errors.
3. The JSON Schema matches the dataclass structure.
4. The config system loads pipeline.yaml and produces correct types.
5. Config error handling works for missing/malformed files.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from preprocessing.config import (
    ConfigError,
    DatasetConfig,
    ImageConfig,
    PipelineConfig,
    ShardConfig,
    TokenizerConfig,
    load_config,
)
from preprocessing.schema import (
    VQA_JSON_SCHEMA,
    SchemaValidationError,
    VQASample,
)



# Helper: create a minimal valid PNG image as bytes

def _make_test_image(width: int = 64, height: int = 48) -> bytes:
    """Create a small RGB PNG image and return its bytes."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()



# Schema tests

class TestVQASampleValidation:
    """Verify VQASample.validate() catches all invariant violations."""

    def test_valid_sample_passes(self):
        """A well-formed sample must pass validation without error."""
        sample = VQASample(
            sample_id="test_001",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="What color is the object?",
            answer="Red",
            dataset_name="unit_test",
        )
        # Should not raise
        sample.validate()

    def test_empty_sample_id_rejected(self):
        """An empty sample_id must be rejected."""
        sample = VQASample(
            sample_id="",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="sample_id"):
            sample.validate()

    def test_empty_image_bytes_rejected(self):
        """Empty image bytes must be rejected."""
        sample = VQASample(
            sample_id="test_002",
            image_bytes=b"",
            image_width=64,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="image_bytes"):
            sample.validate()

    def test_zero_width_rejected(self):
        """Zero image width must be rejected."""
        sample = VQASample(
            sample_id="test_003",
            image_bytes=_make_test_image(),
            image_width=0,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="image_width"):
            sample.validate()

    def test_zero_height_rejected(self):
        """Zero image height must be rejected."""
        sample = VQASample(
            sample_id="test_004",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=0,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="image_height"):
            sample.validate()

    def test_empty_question_rejected(self):
        """An empty question string must be rejected."""
        sample = VQASample(
            sample_id="test_005",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="question"):
            sample.validate()

    def test_whitespace_only_question_rejected(self):
        """A question containing only whitespace must be rejected."""
        sample = VQASample(
            sample_id="test_006",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="   ",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="question"):
            sample.validate()

    def test_empty_answer_rejected(self):
        """An empty answer string must be rejected."""
        sample = VQASample(
            sample_id="test_007",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="Question?",
            answer="",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="answer"):
            sample.validate()

    def test_metadata_defaults_to_empty_dict(self):
        """Metadata should default to an empty dict."""
        sample = VQASample(
            sample_id="test_008",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        assert sample.metadata == {}

    def test_image_decodable_check(self):
        """validate_image_decodable() should pass for valid PNG bytes."""
        sample = VQASample(
            sample_id="test_009",
            image_bytes=_make_test_image(),
            image_width=64,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        # Should not raise
        sample.validate_image_decodable()

    def test_image_decodable_rejects_garbage(self):
        """validate_image_decodable() should reject non-image bytes."""
        sample = VQASample(
            sample_id="test_010",
            image_bytes=b"not an image at all",
            image_width=64,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        with pytest.raises(SchemaValidationError, match="could not be decoded"):
            sample.validate_image_decodable()

    def test_to_dict_excludes_raw_bytes(self):
        """to_dict() should include byte length, not the raw bytes themselves."""
        img_bytes = _make_test_image()
        sample = VQASample(
            sample_id="test_011",
            image_bytes=img_bytes,
            image_width=64,
            image_height=48,
            question="Question?",
            answer="Answer",
            dataset_name="unit_test",
        )
        d = sample.to_dict()
        assert "image_bytes" not in d
        assert d["image_bytes_length"] == len(img_bytes)



# JSON Schema tests

class TestJSONSchema:
    """Verify the JSON Schema constant is structurally correct."""

    def test_required_fields_present(self):
        """All required fields from the schema must be listed."""
        required = set(VQA_JSON_SCHEMA["required"])
        expected = {
            "sample_id", "image_bytes", "image_width", "image_height",
            "question", "answer", "dataset_name",
        }
        assert required == expected

    def test_properties_match_dataclass(self):
        """JSON Schema properties should cover all dataclass fields."""
        import dataclasses
        dc_fields = {f.name for f in dataclasses.fields(VQASample)}
        schema_props = set(VQA_JSON_SCHEMA["properties"].keys())
        # Schema properties should be a superset of (or equal to) dataclass fields
        assert dc_fields <= schema_props



# Config loading tests

class TestConfigLoading:
    """Verify the config system loads YAML correctly."""

    def test_load_real_config(self, pipeline_config: PipelineConfig):
        """Loading configs/pipeline.yaml must produce a valid PipelineConfig."""
        assert isinstance(pipeline_config, PipelineConfig)
        assert isinstance(pipeline_config.dataset, DatasetConfig)
        assert isinstance(pipeline_config.image, ImageConfig)
        assert isinstance(pipeline_config.tokenizer, TokenizerConfig)
        assert isinstance(pipeline_config.shard, ShardConfig)

    def test_dataset_id_matches(self, pipeline_config: PipelineConfig):
        """The dataset ID in config must match the expected validation dataset."""
        assert pipeline_config.dataset.hf_dataset_id == (
            "trannhiem/TranNhiem-Vietnamese-ImageText-Reasoning"
        )

    def test_tokenizer_model_matches(self, pipeline_config: PipelineConfig):
        """The tokenizer model in config must be jais-13b-chat."""
        assert pipeline_config.tokenizer.model_name_or_path == (
            "inceptionai/jais-13b-chat"
        )

    def test_image_config_types(self, pipeline_config: PipelineConfig):
        """Image config fields must have correct types."""
        img = pipeline_config.image
        assert isinstance(img.target_size, tuple)
        assert len(img.target_size) == 2
        assert isinstance(img.normalization_mean, tuple)
        assert isinstance(img.normalization_std, tuple)
        assert img.resize_strategy in ("resize_and_pad", "center_crop", "resize")

    def test_column_mapping_has_required_keys(self, pipeline_config: PipelineConfig):
        """Column mapping must include image, question, answer, and id."""
        mapping = pipeline_config.dataset.column_mapping
        for key in ("image", "question", "answer", "id"):
            assert key in mapping, f"Missing key '{key}' in column_mapping"

    def test_missing_file_raises_config_error(self):
        """Loading a non-existent file must raise ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_malformed_yaml_raises_config_error(self, tmp_path: Path):
        """Malformed YAML must raise ConfigError."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{{{invalid yaml content")
        with pytest.raises(ConfigError):
            load_config(bad_file)

    def test_non_mapping_yaml_raises_config_error(self, tmp_path: Path):
        """A YAML file that is not a mapping must raise ConfigError."""
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="Expected a YAML mapping"):
            load_config(bad_file)

    def test_config_round_trip(self, pipeline_config: PipelineConfig):
        """Serializing to dict and back should preserve all values."""
        d = pipeline_config.to_dict()
        reconstructed = PipelineConfig.from_dict(d)
        assert reconstructed.to_dict() == d

    def test_config_is_frozen(self, pipeline_config: PipelineConfig):
        """Config objects must be immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            pipeline_config.num_workers = 99  # type: ignore[misc]
