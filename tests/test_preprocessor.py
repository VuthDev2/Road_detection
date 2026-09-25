"""
tests/test_preprocessor.py
===========================
Unit tests for src.core.preprocessor.Preprocessor.
Tests run on small synthetic images — no YOLO model needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.core.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor(contrast_factor=2.5, sharpness_factor=2.0)


@pytest.fixture
def sample_pil_image():
    """128×128 gray PIL Image in RGB mode."""
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_bgr_frame():
    """128×128 gray numpy BGR frame (as from cv2)."""
    return np.full((128, 128, 3), 128, dtype=np.uint8)


class TestPreprocessor:

    def test_enhance_image_returns_pil_image(self, preprocessor, sample_pil_image):
        result = preprocessor.enhance_image(sample_pil_image)
        assert isinstance(result, Image.Image)

    def test_enhance_image_preserves_size(self, preprocessor, sample_pil_image):
        result = preprocessor.enhance_image(sample_pil_image)
        assert result.size == sample_pil_image.size

    def test_enhance_frame_returns_numpy_array(self, preprocessor, sample_bgr_frame):
        result = preprocessor.enhance_frame(sample_bgr_frame)
        assert isinstance(result, np.ndarray)

    def test_enhance_frame_preserves_shape(self, preprocessor, sample_bgr_frame):
        result = preprocessor.enhance_frame(sample_bgr_frame)
        assert result.shape == sample_bgr_frame.shape

    def test_enhance_frame_preserves_dtype(self, preprocessor, sample_bgr_frame):
        result = preprocessor.enhance_frame(sample_bgr_frame)
        assert result.dtype == np.uint8
