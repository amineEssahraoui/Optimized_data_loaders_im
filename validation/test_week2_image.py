"""
Week 2 validation: Image normalization correctness.

Tests in this module verify that:
1. normalize_image produces the correct output shape and dtype.
2. All three resize strategies (resize_and_pad, center_crop, resize) work.
3. Per-channel mean/std normalization is applied correctly.
4. Denormalization recovers pixel values within acceptable tolerance.
5. Color space conversion works (RGB, grayscale).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from preprocessing.config import ImageConfig
from preprocessing.image_processor import denormalize_image, normalize_image


# ---------------------------------------------------------------------------
# Helper: create test images with known properties
# ---------------------------------------------------------------------------
def _make_test_image_bytes(
    width: int = 200, height: int = 150, color: tuple = (128, 64, 32)
) -> bytes:
    """Create a solid-color RGB PNG image and return its bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rectangular_image_bytes(width: int, height: int) -> bytes:
    """Create a gradient image with distinct width != height for testing."""
    img = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# Tests
# ===========================================================================
class TestNormalizeImageShape:
    """Verify output shape and dtype for different configurations."""

    def test_default_config_produces_correct_shape(self):
        """Default config (384x384 RGB) must produce (3, 384, 384) float32."""
        config = ImageConfig()
        img_bytes = _make_test_image_bytes()
        result = normalize_image(img_bytes, config)

        assert result.dtype == np.float32
        assert result.shape == (3, 384, 384)

    def test_custom_target_size(self):
        """A custom target size must be reflected in the output shape."""
        config = ImageConfig(target_size=(224, 224))
        img_bytes = _make_test_image_bytes()
        result = normalize_image(img_bytes, config)

        assert result.shape == (3, 224, 224)

    def test_grayscale_produces_single_channel(self):
        """Grayscale color space must produce (1, H, W) output."""
        config = ImageConfig(
            color_space="L",
            normalization_mean=(0.5,),
            normalization_std=(0.5,),
        )
        img_bytes = _make_test_image_bytes()
        result = normalize_image(img_bytes, config)

        assert result.shape == (1, 384, 384)


class TestResizeStrategies:
    """Verify each resize strategy produces correct output dimensions."""

    @pytest.mark.parametrize("strategy", ["resize_and_pad", "center_crop", "resize"])
    def test_all_strategies_produce_target_size(self, strategy: str):
        """Every strategy must produce an image matching target_size."""
        config = ImageConfig(
            target_size=(256, 256),
            resize_strategy=strategy,
        )
        # Use a non-square image to test aspect ratio handling
        img_bytes = _make_rectangular_image_bytes(400, 200)
        result = normalize_image(img_bytes, config)

        assert result.shape == (3, 256, 256)

    def test_resize_and_pad_preserves_aspect_ratio(self):
        """resize_and_pad must not distort the image (aspect ratio preserved)."""
        config = ImageConfig(
            target_size=(256, 256),
            resize_strategy="resize_and_pad",
            pad_value=0,
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
        )
        # Wide image: 400x100. After scaling to fit 256x256, it becomes
        # 256x64 (aspect ratio 4:1). The top/bottom should be padded.
        img_bytes = _make_test_image_bytes(width=400, height=100, color=(255, 255, 255))
        result = normalize_image(img_bytes, config)

        # With no normalization shift, padded areas should be 0.0
        # Check that the top and bottom rows are approximately 0 (padding)
        assert result.shape == (3, 256, 256)
        # Top rows should be near 0 (padding)
        assert np.allclose(result[:, 0, :], 0.0, atol=0.01)

    def test_unsupported_strategy_raises(self):
        """An unsupported resize strategy must raise ValueError."""
        config = ImageConfig(resize_strategy="random_crop")
        img_bytes = _make_test_image_bytes()
        with pytest.raises(ValueError, match="Unsupported resize strategy"):
            normalize_image(img_bytes, config)


class TestNormalization:
    """Verify per-channel mean/std normalization is applied correctly."""

    def test_identity_normalization(self):
        """With mean=0, std=1, output should equal input / 255."""
        config = ImageConfig(
            target_size=(64, 64),
            resize_strategy="resize",
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
        )
        # Create a solid red image (255, 0, 0)
        img_bytes = _make_test_image_bytes(64, 64, (255, 0, 0))
        result = normalize_image(img_bytes, config)

        # Red channel should be ~1.0, green and blue ~0.0
        assert result[0].mean() == pytest.approx(1.0, abs=0.01)
        assert result[1].mean() == pytest.approx(0.0, abs=0.01)
        assert result[2].mean() == pytest.approx(0.0, abs=0.01)

    def test_denormalize_round_trip(self):
        """Denormalize(normalize(image)) should approximately recover the original."""
        config = ImageConfig(
            target_size=(64, 64),
            resize_strategy="resize",
        )
        original_color = (200, 100, 50)
        img_bytes = _make_test_image_bytes(64, 64, original_color)
        normalized = normalize_image(img_bytes, config)
        recovered = denormalize_image(normalized, config)

        # Recovered image should be close to the original color
        # (some rounding error is acceptable due to uint8 conversion)
        assert recovered.dtype == np.uint8
        assert recovered.shape == (64, 64, 3)
        mean_pixel = recovered.mean(axis=(0, 1))
        assert mean_pixel[0] == pytest.approx(original_color[0], abs=2)
        assert mean_pixel[1] == pytest.approx(original_color[1], abs=2)
        assert mean_pixel[2] == pytest.approx(original_color[2], abs=2)


class TestImageErrors:
    """Verify error handling for invalid inputs."""

    def test_invalid_bytes_raises(self):
        """Non-image bytes must raise ValueError."""
        config = ImageConfig()
        with pytest.raises(ValueError, match="Failed to decode"):
            normalize_image(b"not an image", config)

    def test_channel_mismatch_raises(self):
        """Normalization mean/std channel count must match the image."""
        config = ImageConfig(
            color_space="L",
            normalization_mean=(0.5, 0.5, 0.5),  # 3 values for 1-channel image
            normalization_std=(0.5, 0.5, 0.5),
        )
        img_bytes = _make_test_image_bytes()
        with pytest.raises(ValueError, match="channels"):
            normalize_image(img_bytes, config)
