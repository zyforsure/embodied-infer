"""Shared camera image normalization for backends and simulators.

Backends receive HWC uint8 images from the canonical request, but each
wire protocol wants a different layout/dtype (CHW uint8 for OpenPI, HWC
float [0,1] for the Embodied.cpp proto, JPEG bytes for the TCP gateway).
These helpers are the single place that knows how to convert.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_hwc(value: Any) -> np.ndarray:
    """Normalize a camera image to HWC layout with exactly 3 channels."""

    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"camera image must be rank 3, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"camera image must be HWC or CHW RGB, got {image.shape}")
    return np.ascontiguousarray(image[..., :3])


def as_hwc_uint8(value: Any) -> np.ndarray:
    """HWC uint8; float images are assumed to span [0, 1] and scaled."""

    image = as_hwc(value)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def as_chw_uint8(value: Any) -> np.ndarray:
    """CHW uint8 for OpenPI-style observation dicts."""

    return np.ascontiguousarray(np.moveaxis(as_hwc_uint8(value), -1, 0))


def as_hwc_float01(value: Any) -> np.ndarray:
    """HWC float32 in [0, 1]; uint8-like ranges are rescaled."""

    image = as_hwc(value).astype(np.float32, copy=False)
    if image.size and float(np.max(image)) > 1.5:
        image = image / 255.0
    return np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32)


def encode_jpeg(value: Any, *, quality: int = 95) -> bytes:
    """JPEG bytes via Pillow, falling back to OpenCV when unavailable."""

    array = as_hwc_uint8(value)
    try:
        from PIL import Image
        import io

        output = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(output, format="JPEG", quality=quality)
        return output.getvalue()
    except ImportError:
        import cv2

        ok, encoded = cv2.imencode(".jpg", array[..., ::-1])
        if not ok:
            raise RuntimeError("failed to encode image as JPEG")
        return encoded.tobytes()


__all__ = [
    "as_chw_uint8",
    "as_hwc",
    "as_hwc_float01",
    "as_hwc_uint8",
    "encode_jpeg",
]
