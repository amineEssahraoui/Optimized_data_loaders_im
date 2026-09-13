"""
Image normalization module for the VQA preprocessing pipeline.

Responsibilities:
- Decode raw image bytes (PNG/JPEG) into pixel arrays.
- Resize/pad/crop to the target dimensions specified in config.
- Convert to the target color space (RGB, BGR, or grayscale).
- Apply per-channel mean/std normalization to produce float32 tensors.

The output is a numpy array of shape (C, H, W) in float32, suitable
for direct consumption by vision encoders.  The channel-first layout
(CHW) is standard for PyTorch-based models and is also what the C++
loader will reconstruct.

All behavior is driven by ImageConfig -- changing the resize strategy,
target size, or normalization stats requires only a config edit.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from preprocessing.config import ImageConfig

logger = logging.getLogger(__name__)

# Map config interpolation names to Pillow resampling constants.
# This dict is the single source of truth for supported methods.
_INTERPOLATION_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _get_interpolation(name: str) -> Image.Resampling:
    """Resolve a config interpolation name to a Pillow constant.

    Raises ValueError if the name is not in the supported set.
    """
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

    This preserves the aspect ratio and fills the remaining space with
    pad_value.  The image is centered within the target canvas.
    """
    orig_w, orig_h = img.size

    # Scale factor is determined by the longest side
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = img.resize((new_w, new_h), resample=resample)

    # Create a canvas of the target size, filled with pad_value
    canvas = Image.new(img.mode, (target_w, target_h), color=pad_value)

    # Center the resized image on the canvas
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
    """Resize so the shortest side matches the target, then center crop.

    This preserves the aspect ratio by cropping excess pixels from
    the longer side, centered.
    """
    orig_w, orig_h = img.size

    # Scale factor is determined by the shortest side
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = img.resize((new_w, new_h), resample=resample)

    # Center crop to target dimensions
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

    Steps:
    1. Decode from bytes using Pillow.
    2. Convert to the target color space (RGB, BGR, or L).
    3. Resize/pad/crop to the target dimensions.
    4. Convert to float32 in [0, 1] range.
    5. Apply per-channel mean/std normalization.
    6. Transpose to CHW layout.

    Parameters
    ----------
    image_bytes : bytes
        Raw encoded image (PNG, JPEG, etc.).
    config : ImageConfig
        Image preprocessing configuration.

    Returns
    -------
    np.ndarray
        Float32 array of shape (C, H, W) where C is 3 for RGB/BGR
        or 1 for grayscale.

    Raises
    ------
    ValueError
        If the image cannot be decoded or the config specifies an
        unsupported color space or resize strategy.
    """
    # Step 1: Decode image bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc

    # Step 2: Color space conversion
    target_cs = config.color_space.upper()
    if target_cs == "RGB":
        img = img.convert("RGB")
    elif target_cs == "BGR":
        # Pillow does not have a native BGR mode, so convert to RGB
        # and flip channels after converting to numpy.
        img = img.convert("RGB")
    elif target_cs == "L":
        img = img.convert("L")
    else:
        raise ValueError(f"Unsupported color space: {config.color_space}")

    # Step 3: Resize/pad/crop
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

    # Step 4: Convert to float32 in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Handle grayscale: add channel dimension if needed
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (1, H, W)
    else:
        # Step 5a: Handle BGR channel flip (before normalization)
        if target_cs == "BGR":
            arr = arr[:, :, ::-1]  # RGB -> BGR

        # Transpose from HWC to CHW
        arr = arr.transpose(2, 0, 1)  # (C, H, W)

    # Step 5: Per-channel mean/std normalization
    mean = np.array(config.normalization_mean, dtype=np.float32)
    std = np.array(config.normalization_std, dtype=np.float32)

    # Validate that mean/std dimensions match the number of channels
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

    # Reshape for broadcasting: (C, 1, 1)
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)

    arr = (arr - mean) / std

    return arr


def resize_preserve_aspect(
    image_bytes: bytes,
    config: ImageConfig,
) -> tuple[np.ndarray, int, int, int, int]:
    """Resize an image preserving aspect ratio, returning raw uint8 CHW.

    The longest side is scaled to ``config.max_image_dim``.  No padding
    or normalization is applied — those are deferred to the C++ loader
    at batch collation time (Format v2 pipeline).

    Parameters
    ----------
    image_bytes : bytes
        Raw encoded image (PNG, JPEG, etc.).
    config : ImageConfig
        Image preprocessing configuration.  Uses ``max_image_dim``,
        ``color_space``, and ``interpolation``.

    Returns
    -------
    tuple[np.ndarray, int, int, int, int]
        ``(image_uint8_chw, orig_h, orig_w, actual_h, actual_w)``
        where ``image_uint8_chw`` is a contiguous ``uint8`` array of
        shape ``(C, actual_h, actual_w)``.

    Raises
    ------
    ValueError
        If the image cannot be decoded or the color space is unsupported.
    """
    # Step 1: Decode
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {exc}") from exc

    # Step 2: Color space conversion
    target_cs = config.color_space.upper()
    if target_cs == "RGB":
        img = img.convert("RGB")
    elif target_cs == "BGR":
        img = img.convert("RGB")  # flip channels later in numpy
    elif target_cs == "L":
        img = img.convert("L")
    else:
        raise ValueError(f"Unsupported color space: {config.color_space}")

    orig_w, orig_h = img.size

    # Step 3: Resize — scale longest side to max_image_dim
    max_dim = config.max_image_dim
    scale = min(max_dim / orig_w, max_dim / orig_h)
    if scale < 1.0:
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        resample = _get_interpolation(config.interpolation)
        img = img.resize((new_w, new_h), resample=resample)
    else:
        new_w, new_h = orig_w, orig_h

    # Step 4: Convert to uint8 numpy array
    arr = np.array(img, dtype=np.uint8)

    # Handle BGR channel flip
    if target_cs == "BGR" and arr.ndim == 3:
        arr = arr[:, :, ::-1]

    # Transpose HWC → CHW (or add channel dim for grayscale)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)  # (C, H, W)

    return np.ascontiguousarray(arr), orig_h, orig_w, new_h, new_w


def prepare_image_uint8(image_bytes: bytes, config: ImageConfig) -> np.ndarray:
    """Prepare an image as a uint8 CHW array for Format v2 shard storage.

    This is the v2 equivalent of :func:`normalize_image`.  Normalization
    is *not* applied here — it is deferred to the C++ loader at batch
    collation time, where SIMD vectorisation can be exploited.

    Parameters
    ----------
    image_bytes : bytes
        Raw encoded image (PNG, JPEG, etc.).
    config : ImageConfig
        Image preprocessing configuration.

    Returns
    -------
    np.ndarray
        Contiguous ``uint8`` array of shape ``(C, H, W)``.
    """
    arr, _, _, _, _ = resize_preserve_aspect(image_bytes, config)
    return arr


def denormalize_image(tensor: np.ndarray, config: ImageConfig) -> np.ndarray:
    """Reverse the normalization to recover displayable pixel values.

    Useful for visual inspection during validation: apply this to a
    normalized tensor and save the result as a PNG to verify correctness.

    Parameters
    ----------
    tensor : np.ndarray
        Float32 array of shape (C, H, W), as produced by normalize_image.
    config : ImageConfig
        The same config used for normalization.

    Returns
    -------
    np.ndarray
        Uint8 array of shape (H, W, C) in [0, 255], suitable for saving
        as an image file.
    """
    mean = np.array(config.normalization_mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(config.normalization_std, dtype=np.float32).reshape(-1, 1, 1)

    # Reverse normalization
    arr = tensor * std + mean

    # Clip to [0, 1] and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)

    # CHW -> HWC
    if arr.shape[0] == 1:
        arr = arr[0]  # (H, W) for grayscale
    else:
        arr = arr.transpose(1, 2, 0)  # (H, W, C)

    return (arr * 255.0).astype(np.uint8)
