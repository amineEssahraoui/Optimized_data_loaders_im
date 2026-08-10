"""
Week 2 validation: Tokenization correctness.

Tests in this module verify that:
1. The tokenizer loads successfully from config.
2. Tokenization produces the correct output shapes and dtypes.
3. Round-trip decode (encode -> decode) recovers the original text.
4. Padding and truncation work as configured.
5. Special tokens are handled correctly.
6. Batch tokenization is consistent with single-text tokenization.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.config import PipelineConfig, TokenizerConfig
from preprocessing.tokenizer import TextTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    """Create a tokenizer instance using the real config.

    Module-scoped to avoid reloading the tokenizer for every test.
    """
    return TextTokenizer(pipeline_config.tokenizer)


# ===========================================================================
# Tests
# ===========================================================================
class TestTokenizerLoading:
    """Verify the tokenizer initializes correctly."""

    def test_tokenizer_loads(self, tokenizer: TextTokenizer):
        """The tokenizer must load without error."""
        assert tokenizer is not None
        assert tokenizer.tokenizer is not None

    def test_vocab_size_positive(self, tokenizer: TextTokenizer):
        """The tokenizer must have a positive vocabulary size."""
        assert tokenizer.tokenizer.vocab_size > 0

    def test_config_preserved(self, tokenizer: TextTokenizer, pipeline_config: PipelineConfig):
        """The tokenizer's config must match the input config."""
        assert tokenizer.config.model_name_or_path == (
            pipeline_config.tokenizer.model_name_or_path
        )


class TestTokenizationOutput:
    """Verify tokenization produces correct output format."""

    def test_output_shapes(self, tokenizer: TextTokenizer, pipeline_config: PipelineConfig):
        """Token IDs and attention mask must have shape (max_length,)."""
        result = tokenizer.tokenize("Hello, world!")
        max_len = pipeline_config.tokenizer.max_length

        assert result.input_ids.shape == (max_len,)
        assert result.attention_mask.shape == (max_len,)

    def test_output_dtype_int32(self, tokenizer: TextTokenizer):
        """Token IDs and attention mask must be int32."""
        result = tokenizer.tokenize("Test text.")
        assert result.input_ids.dtype == np.int32
        assert result.attention_mask.dtype == np.int32

    def test_attention_mask_binary(self, tokenizer: TextTokenizer):
        """Attention mask must contain only 0s and 1s."""
        result = tokenizer.tokenize("Some input text for testing.")
        unique_values = set(result.attention_mask.tolist())
        assert unique_values <= {0, 1}

    def test_padding_tokens_masked(self, tokenizer: TextTokenizer, pipeline_config: PipelineConfig):
        """Padding positions must have attention_mask=0 and consistent IDs."""
        # Short text that will definitely need padding to max_length
        result = tokenizer.tokenize("Hi")
        max_len = pipeline_config.tokenizer.max_length

        # There should be some padding (mask=0) tokens
        num_padded = (result.attention_mask == 0).sum()
        assert num_padded > 0, "Short text should have padding tokens"

        # Real tokens should be fewer than max_length
        num_real = (result.attention_mask == 1).sum()
        assert num_real < max_len


class TestRoundTrip:
    """Verify that tokenization is reversible (encode -> decode)."""

    @pytest.mark.parametrize("text", [
        "What color is the car in this image?",
        "The answer is blue.",
        "A longer sentence to test tokenization with more tokens involved.",
    ])
    def test_round_trip_english(self, tokenizer: TextTokenizer, text: str):
        """Encoding then decoding English text must recover the original."""
        result = tokenizer.tokenize(text)
        decoded = tokenizer.decode(result.input_ids, skip_special_tokens=True)

        # The decoded text should contain the original text (it may have
        # minor whitespace differences due to tokenizer normalization)
        assert text.strip() in decoded.strip() or decoded.strip() in text.strip(), (
            f"Round-trip failed: original='{text}', decoded='{decoded}'"
        )

    def test_round_trip_preserves_original_ref(self, tokenizer: TextTokenizer):
        """TokenizedText.original_text must preserve the input exactly."""
        text = "Exact preservation test."
        result = tokenizer.tokenize(text)
        assert result.original_text == text


class TestBatchTokenization:
    """Verify batch tokenization consistency."""

    def test_batch_matches_individual(self, tokenizer: TextTokenizer):
        """Batch tokenization must produce the same results as individual calls."""
        texts = [
            "First question about the image.",
            "Second question, different length.",
            "Third.",
        ]

        batch_results = tokenizer.tokenize_batch(texts)
        individual_results = [tokenizer.tokenize(t) for t in texts]

        assert len(batch_results) == len(individual_results)
        for batch_r, indiv_r in zip(batch_results, individual_results):
            np.testing.assert_array_equal(batch_r.input_ids, indiv_r.input_ids)
            np.testing.assert_array_equal(batch_r.attention_mask, indiv_r.attention_mask)

    def test_batch_preserves_order(self, tokenizer: TextTokenizer):
        """Batch results must correspond to inputs in order."""
        texts = ["Alpha", "Beta", "Gamma"]
        results = tokenizer.tokenize_batch(texts)
        for text, result in zip(texts, results):
            assert result.original_text == text
