"""Contract tests for the shared camera image conversions."""

import numpy as np
import pytest

from embodied_infer_deploy.images import (
    as_chw_uint8,
    as_hwc,
    as_hwc_float01,
    as_hwc_uint8,
)


def test_hwc_passthrough_keeps_layout_and_drops_alpha():
    image = np.zeros((4, 5, 4), dtype=np.uint8)
    out = as_hwc(image)
    assert out.shape == (4, 5, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_chw_input_is_transposed_to_hwc():
    image = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    out = as_hwc(image)
    assert out.shape == (4, 5, 3)
    np.testing.assert_array_equal(out[0, 0], image[:, 0, 0])


def test_float01_scales_to_uint8():
    image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    out = as_hwc_uint8(image)
    assert out.dtype == np.uint8
    assert np.all(out == 127)


def test_chw_uint8_matches_remote_layout():
    image = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    out = as_chw_uint8(image)
    assert out.shape == (3, 2, 4)
    np.testing.assert_array_equal(out[0], image[..., 0])


def test_float01_uint8_rescales():
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = as_hwc_float01(image)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)
    small = np.full((2, 2, 3), 0.25, dtype=np.float32)
    np.testing.assert_array_equal(as_hwc_float01(small), small)


def test_invalid_rank_and_channels_rejected():
    with pytest.raises(ValueError, match="rank 3"):
        as_hwc(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB"):
        as_hwc(np.zeros((5, 4, 5), dtype=np.uint8))
