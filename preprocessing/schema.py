"""
Canonical VQA schema for the preprocessing pipeline.

Every dataset ingested by this pipeline must produce instances of VQASample.
This guarantees a uniform interface between ingestion and downstream processing
(normalization, tokenization, shard writing), regardless of the source dataset's
native format.

Design decisions:
- image_bytes stores the raw encoded image (PNG/JPEG bytes), not decoded pixels.
  This defers the decode-time choice to the image normalization stage, which may
  need the original encoding for quality or format detection.
- metadata is a free-form dict for optional/auxiliary fields (e.g. model_reasoning).
  This avoids schema changes when adding non-essential data.
- Validation is built into the class so that every sample can be checked at
  construction time, catching data issues as early as possible.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# JSON Schema representation (for documentation and external validation)
# ---------------------------------------------------------------------------
VQA_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "VQASample",
    "description": (
        "Canonical schema for a single Visual Question Answering sample. "
        "All VQA-type datasets must be normalized into this format before "
        "entering the preprocessing pipeline."
    ),
    "type": "object",
    "properties": {
        "sample_id": {
            "type": "string",
            "description": "Unique identifier for this sample within the dataset.",
        },
        "image_bytes": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Raw image bytes (PNG or JPEG encoded).",
        },
        "image_width": {
            "type": "integer",
            "minimum": 1,
            "description": "Original image width in pixels.",
        },
        "image_height": {
            "type": "integer",
            "minimum": 1,
            "description": "Original image height in pixels.",
        },
        "question": {
            "type": "string",
            "minLength": 1,
            "description": "The question text associated with the image.",
        },
        "answer": {
            "type": "string",
            "minLength": 1,
            "description": "The answer text associated with the question.",
        },
        "metadata": {
            "type": "object",
            "description": "Free-form metadata (e.g. model_reasoning, source URL).",
        },
        "dataset_name": {
            "type": "string",
            "description": "Identifier of the source dataset.",
        },
    },
    "required": [
        "sample_id",
        "image_bytes",
        "image_width",
        "image_height",
        "question",
        "answer",
        "dataset_name",
    ],
}


# ---------------------------------------------------------------------------
# Schema validation errors
# ---------------------------------------------------------------------------
class SchemaValidationError(ValueError):
    """Raised when a VQASample fails validation checks."""


# ---------------------------------------------------------------------------
# Canonical VQA sample dataclass
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class VQASample:
    """A single Visual Question Answering sample in canonical form.

    Attributes
    ----------
    sample_id : str
        Unique identifier for this sample (from the dataset's id column,
        or a generated index-based id if the dataset has no id column).
    image_bytes : bytes
        Raw image data in its original encoding (PNG, JPEG, etc.).
        Not decoded -- decoding is deferred to the normalization stage.
    image_width : int
        Width of the original image in pixels.
    image_height : int
        Height of the original image in pixels.
    question : str
        The question text.
    answer : str
        The answer text.
    metadata : dict[str, Any]
        Free-form dictionary for auxiliary data that is not part of the
        core schema.  For example, the TranNhiem dataset's
        ``model_reasoning`` field is stored here.
    dataset_name : str
        Identifier for the source dataset, set from config.
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

        Raises
        ------
        SchemaValidationError
            If any invariant is violated.  The message describes which
            check failed so the caller can log or skip the sample.
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

    def validate_image_decodable(self) -> None:
        """Verify that image_bytes can be decoded by Pillow.

        This is a heavier check than validate() and is intended for use
        during ingestion quality checks, not on every sample access.

        Raises
        ------
        SchemaValidationError
            If the image bytes cannot be decoded.
        """
        try:
            img = Image.open(io.BytesIO(self.image_bytes))
            img.verify()
        except Exception as exc:
            raise SchemaValidationError(
                f"image_bytes for sample {self.sample_id} could not be decoded: {exc}"
            ) from exc

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
