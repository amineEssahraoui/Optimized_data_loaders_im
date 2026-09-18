"""
Image normalization module for the VQA preprocessing pipeline.

Responsibilities:
- Decode raw image bytes (PNG/JPEG) into pixel arrays.
- Resize/pad/crop to the target dimensions specified in config.
- Convert to the target color space (RGB, BGR, or grayscale).
- Apply per-channel mean/std normalization to produce float32 tensors.

The output is a numpy array of shape (C, H, W) in float32, suitable
for direct consumption by vision encoders. All behavior is driven by
ImageConfig.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from configs.config import ImageConfig

logger = logging.getLogger(__name__)

# Map config interpolation names to Pillow resampling constants
_INTERPOLATION_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _get_interpolation(name: str) -> Image.Resampling:
    """Resolve a config interpolation name to a Pillow constant."""
    name_lower = name.lower()
    if name_lower not in _INTERPOLATION_MAP:
        raise ValueError(
            f"Unsupported interpolation method '{name}'. "
            f"Supported: {list(_INTERPOLATION_MAP.keys())}"
        )
    return _INTERPOLATION_MAP[name_lower]


def _resize_and_pad(
    img: Image.Image,
    target_h: int,
    target_w: int,
    resample: Image.Resampling,
    pad_value: int,
) -> Image.Image:
    """Resize so the longest side matches the target, then pad the shorter side.

    Preserves the aspect ratio and centers the image on a padded canvas.
    """
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = img.resize((new_w, new_h), resample=resample)
    canvas = Image.new(img.mode, (target_w, target_h), color=pad_value)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def _center_crop(
    img: Image.Image,
    target_h: int,
    target_w: int,
    resample: Image.Resampling,
) -> Image.Image:
    """Resize so the shortest side matches the target, then center crop."""
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = img.resize((new_w, new_h), resample=resample)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _resize_exact(
    img: Image.Image,
    target_h: int,
    target_w: int,
    resample: Image.Resampling,
) -> Image.Image:
    """Resize directly to the exact target dimensions (may distort aspect ratio)."""
    return img.resize((target_w, target_h), resample=resample)


def normalize_image(image_bytes: bytes, config: ImageConfig) -> np.ndarray:
    """Apply the full image normalization pipeline to raw image bytes.

    Steps: decode, color convert, resize/pad/crop, float32 normalize, CHW transpose.

    Parameters
    ----------
    image_bytes : bytes
        Raw encoded image (PNG, JPEG, etc.).
    config : ImageConfig
        Image preprocessing configuration.

    Returns
    -------
    np.ndarray
        Float32 array of shape (C, H, W).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc

    target_cs = config.color_space.upper()
    if target_cs == "RGB":
        img = img.convert("RGB")
    elif target_cs == "BGR":
        img = img.convert("RGB")
    elif target_cs == "L":
        img = img.convert("L")
    else:
        raise ValueError(f"Unsupported color space: {config.color_space}")

    target_h, target_w = config.target_size
    resample = _get_interpolation(config.interpolation)

    strategy = config.resize_strategy.lower()
    if strategy == "resize_and_pad":
        img = _resize_and_pad(img, target_h, target_w, resample, config.pad_value)
    elif strategy == "center_crop":
        img = _center_crop(img, target_h, target_w, resample)
    elif strategy == "resize":
        img = _resize_exact(img, target_h, target_w, resample)
    else:
        raise ValueError(
            f"Unsupported resize strategy: {config.resize_strategy}. "
            f"Supported: resize_and_pad, center_crop, resize"
        )

    arr = np.array(img, dtype=np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    else:
        if target_cs == "BGR":
            arr = arr[:, :, ::-1]
        arr = arr.transpose(2, 0, 1)

    mean = np.array(config.normalization_mean, dtype=np.float32)
    std = np.array(config.normalization_std, dtype=np.float32)

    num_channels = arr.shape[0]
    if len(mean) != num_channels:
        raise ValueError(
            f"normalization_mean has {len(mean)} values but image has "
            f"{num_channels} channels"
        )
    if len(std) != num_channels:
        raise ValueError(
            f"normalization_std has {len(std)} values but image has "
            f"{num_channels} channels"
        )

    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    arr = (arr - mean) / std

    return arr


def resize_preserve_aspect(
    image_bytes: bytes,
    config: ImageConfig,
) -> tuple[np.ndarray, int, int, int, int]:
    """Resize an image preserving aspect ratio, returning raw uint8 CHW.

    The longest side is scaled to config.max_image_dim. No padding or
    normalization is applied (deferred to C++ loader for Format v2).

    Returns
    -------
    tuple[np.ndarray, int, int, int, int]
        (image_uint8_chw, orig_h, orig_w, actual_h, actual_w)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc

    target_cs = config.color_space.upper()
    if target_cs == "RGB":
        img = img.convert("RGB")
    elif target_cs == "BGR":
        img = img.convert("RGB")
    elif target_cs == "L":
        img = img.convert("L")
    else:
        raise ValueError(f"Unsupported color space: {config.color_space}")

    orig_w, orig_h = img.size

    max_dim = config.max_image_dim
    scale = min(max_dim / orig_w, max_dim / orig_h)
    if scale < 1.0:
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        resample = _get_interpolation(config.interpolation)
        img = img.resize((new_w, new_h), resample=resample)
    else:
        new_w, new_h = orig_w, orig_h

    arr = np.array(img, dtype=np.uint8)

    if target_cs == "BGR" and arr.ndim == 3:
        arr = arr[:, :, ::-1]

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    else:
        arr = arr.transpose(2, 0, 1)

    return np.ascontiguousarray(arr), orig_h, orig_w, new_h, new_w
