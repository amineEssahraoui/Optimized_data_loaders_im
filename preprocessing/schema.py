"""
Canonical VQA schema for the preprocessing pipeline.

Every dataset ingested by this pipeline must produce instances of VQASample.
This guarantees a uniform interface between ingestion and downstream processing
(normalization, tokenization, shard writing), regardless of the source format.

Design decisions:
- image_bytes stores the raw encoded image (PNG/JPEG bytes), not decoded pixels.
  This defers the decode-time choice to the image normalization stage.
- metadata is a free-form dict for optional/auxiliary fields (e.g. model_reasoning).
- Validation is built into the class so that every sample can be checked at
  construction time, catching data issues as early as possible.
"""

from __future__ import annotations

import dataclasses
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when a VQASample fails validation checks."""


@dataclasses.dataclass
class VQASample:
    """A single Visual Question Answering sample in canonical form.

    Attributes
    ----------
    sample_id : str
        Unique identifier for this sample.
    image_bytes : bytes
        Raw image data in its original encoding (PNG, JPEG, etc.).
    image_width : int
        Width of the original image in pixels.
    image_height : int
        Height of the original image in pixels.
    question : str
        The question text.
    answer : str
        The answer text.
    metadata : dict[str, Any]
        Free-form dictionary for auxiliary data (e.g. model_reasoning).
    dataset_name : str
        Identifier for the source dataset.
    """

    sample_id: str
    image_bytes: bytes
    image_width: int
    image_height: int
    question: str
    answer: str
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    dataset_name: str = ""

    def validate(self) -> None:
        """Check that this sample satisfies all schema invariants.

        Raises SchemaValidationError if any invariant is violated.
        """
        if not self.sample_id:
            raise SchemaValidationError("sample_id must be a non-empty string")

        if not isinstance(self.image_bytes, bytes) or len(self.image_bytes) == 0:
            raise SchemaValidationError(
                f"image_bytes must be non-empty bytes, got {type(self.image_bytes).__name__} "
                f"with length {len(self.image_bytes) if isinstance(self.image_bytes, bytes) else 'N/A'}"
            )

        if self.image_width < 1:
            raise SchemaValidationError(
                f"image_width must be >= 1, got {self.image_width}"
            )

        if self.image_height < 1:
            raise SchemaValidationError(
                f"image_height must be >= 1, got {self.image_height}"
            )

        if not isinstance(self.question, str) or len(self.question.strip()) == 0:
            raise SchemaValidationError("question must be a non-empty string")

        if not isinstance(self.answer, str) or len(self.answer.strip()) == 0:
            raise SchemaValidationError("answer must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (without image_bytes for logging)."""
        return {
            "sample_id": self.sample_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_bytes_length": len(self.image_bytes),
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata,
            "dataset_name": self.dataset_name,
        }
