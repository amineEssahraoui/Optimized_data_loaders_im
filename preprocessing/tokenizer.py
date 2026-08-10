"""
Text tokenization module for the VQA preprocessing pipeline.

Wraps HuggingFace AutoTokenizer to provide a config-driven interface.
The tokenizer model is loaded from the name/path specified in config,
so swapping to a different tokenizer requires only a config change.

Design decisions:
- The tokenizer is loaded lazily (on first use) and cached, because
  loading large tokenizer models is expensive (~seconds).
- Results are returned as numpy int32 arrays, which is the format
  written into binary shards and reconstructed by the C++ loader.
- The module provides both single-text and batch tokenization for
  flexibility.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from preprocessing.config import TokenizerConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TokenizedText:
    """Result of tokenizing a single text string.

    Attributes
    ----------
    input_ids : np.ndarray
        Token IDs as int32 array of shape (seq_len,).
    attention_mask : np.ndarray
        Attention mask as int32 array of shape (seq_len,).
        1 for real tokens, 0 for padding.
    original_text : str
        The original text before tokenization (for round-trip checks).
    """
    input_ids: np.ndarray
    attention_mask: np.ndarray
    original_text: str


class TextTokenizer:
    """Config-driven text tokenizer backed by HuggingFace AutoTokenizer.

    Usage::

        tokenizer = TextTokenizer(config.tokenizer)
        result = tokenizer.tokenize("What is in this image?")
        print(result.input_ids.shape)  # (max_length,)

    The tokenizer instance is created once and reused for all calls.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize the tokenizer from config.

        Parameters
        ----------
        config : TokenizerConfig
            Tokenizer configuration specifying the model, max length,
            padding strategy, etc.
        """
        self._config = config

        logger.info(
            "Loading tokenizer: %s (max_length=%d, padding=%s)",
            config.model_name_or_path,
            config.max_length,
            config.padding,
        )

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Some tokenizers (like jais) may not have a pad token set.
        # Fall back to EOS token to avoid errors during padding.
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info(
                    "Tokenizer has no pad_token; using eos_token ('%s') as pad_token.",
                    self._tokenizer.eos_token,
                )
            else:
                logger.warning(
                    "Tokenizer has neither pad_token nor eos_token. "
                    "Padding may fail."
                )

        logger.info(
            "Tokenizer loaded: vocab_size=%d, pad_token='%s'",
            self._tokenizer.vocab_size,
            self._tokenizer.pad_token,
        )

    @property
    def config(self) -> TokenizerConfig:
        """Return the tokenizer configuration."""
        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the underlying HuggingFace tokenizer instance."""
        return self._tokenizer

    def tokenize(self, text: str) -> TokenizedText:
        """Tokenize a single text string.

        Parameters
        ----------
        text : str
            The input text to tokenize.

        Returns
        -------
        TokenizedText
            Contains input_ids and attention_mask as int32 numpy arrays,
            plus the original text for round-trip verification.
        """
        encoded = self._tokenizer(
            text,
            max_length=self._config.max_length,
            padding=self._config.padding,
            truncation=self._config.truncation,
            add_special_tokens=self._config.add_special_tokens,
            return_tensors=None,  # Return plain lists, not PyTorch tensors
        )

        input_ids = np.array(encoded["input_ids"], dtype=np.int32)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)

        return TokenizedText(
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_text=text,
        )

    def tokenize_batch(self, texts: list[str]) -> list[TokenizedText]:
        """Tokenize multiple texts.

        Parameters
        ----------
        texts : list[str]
            List of input texts to tokenize.

        Returns
        -------
        list[TokenizedText]
            One TokenizedText per input string.
        """
        encoded = self._tokenizer(
            texts,
            max_length=self._config.max_length,
            padding=self._config.padding,
            truncation=self._config.truncation,
            add_special_tokens=self._config.add_special_tokens,
            return_tensors=None,
        )

        results = []
        for i, text in enumerate(texts):
            results.append(TokenizedText(
                input_ids=np.array(encoded["input_ids"][i], dtype=np.int32),
                attention_mask=np.array(encoded["attention_mask"][i], dtype=np.int32),
                original_text=text,
            ))
        return results

    def decode(self, input_ids: np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        input_ids : np.ndarray
            Token IDs as a 1-D integer array.
        skip_special_tokens : bool
            Whether to remove special tokens (BOS, EOS, PAD) from output.

        Returns
        -------
        str
            The decoded text string.
        """
        # Convert numpy array to Python list for the HuggingFace API
        ids_list = input_ids.tolist()
        return self._tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)
