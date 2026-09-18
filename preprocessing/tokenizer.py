"""
Text tokenization module for the VQA preprocessing pipeline.

Wraps HuggingFace AutoTokenizer to provide a config-driven interface.
The tokenizer model is loaded from the name/path specified in config,
so swapping to a different tokenizer requires only a config change.

Supports both static and dynamic text padding:
- Static (dynamic_text_padding=false): every sequence is padded to
  max_length at tokenization time.
- Dynamic (dynamic_text_padding=true): sequences are tokenized without
  padding. The actual token count is recorded, and the arrays are
  manually padded to max_length before shard writing (to maintain binary
  format compatibility). At batch collation time, the attention mask
  and stored actual lengths allow the loader to reconstruct tight batches.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from configs.config import TokenizerConfig

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
        The original text before tokenization.
    actual_length : int
        Number of real (non-padding) tokens. Equals seq_len when
        static padding is used; may be less when dynamic padding
        pads to max_length after tokenization.
    """
    input_ids: np.ndarray
    attention_mask: np.ndarray
    original_text: str
    actual_length: int = 0


class TextTokenizer:
    """Config-driven text tokenizer backed by HuggingFace AutoTokenizer.

    Usage::

        tokenizer = TextTokenizer(config.tokenizer)
        result = tokenizer.tokenize("What is in this image?")
        print(result.input_ids.shape)       # (max_length,)
        print(result.actual_length)         # actual token count
    """

    def __init__(self, config: TokenizerConfig) -> None:
        self._config = config
        self._dynamic = config.dynamic_text_padding

        logger.info(
            "Loading tokenizer: %s (max_length=%d, padding=%s, dynamic=%s)",
            config.model_name_or_path,
            config.max_length,
            config.padding,
            self._dynamic,
        )

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Some tokenizers may not have a pad token set; fall back to EOS
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

        self._pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else 0
        )

        logger.info(
            "Tokenizer loaded: vocab_size=%d, pad_token='%s', dynamic_padding=%s",
            self._tokenizer.vocab_size,
            self._tokenizer.pad_token,
            self._dynamic,
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

        When dynamic_text_padding is enabled, the text is first tokenized
        without padding, then manually padded to max_length. The
        actual_length field records how many tokens are real.

        When static padding is used, HuggingFace handles the padding
        directly to max_length.

        In both cases, the returned arrays have shape (max_length,).
        """
        max_len = self._config.max_length

        if self._dynamic:
            # Dynamic: tokenize without padding, then manually pad
            encoded = self._tokenizer(
                text,
                max_length=max_len,
                padding=False,
                truncation=self._config.truncation,
                add_special_tokens=self._config.add_special_tokens,
                return_tensors=None,
            )

            raw_ids = encoded["input_ids"]
            raw_mask = encoded["attention_mask"]
            actual_length = len(raw_ids)

            # Pad to max_length for binary shard format compatibility
            input_ids = np.full(max_len, self._pad_id, dtype=np.int32)
            attention_mask = np.zeros(max_len, dtype=np.int32)

            fill_len = min(actual_length, max_len)
            input_ids[:fill_len] = raw_ids[:fill_len]
            attention_mask[:fill_len] = raw_mask[:fill_len]
        else:
            # Static: HuggingFace pads directly to max_length
            encoded = self._tokenizer(
                text,
                max_length=max_len,
                padding=self._config.padding,
                truncation=self._config.truncation,
                add_special_tokens=self._config.add_special_tokens,
                return_tensors=None,
            )

            input_ids = np.array(encoded["input_ids"], dtype=np.int32)
            attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)
            actual_length = int(attention_mask.sum())

        return TokenizedText(
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_text=text,
            actual_length=actual_length,
        )

    def tokenize_batch(self, texts: list[str]) -> list[TokenizedText]:
        """Tokenize multiple texts."""
        return [self.tokenize(text) for text in texts]

    def decode(self, input_ids: np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ids_list = input_ids.tolist()
        return self._tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)
