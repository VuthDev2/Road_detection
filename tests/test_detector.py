"""
tests/test_detector.py
======================
Unit tests for src.core.detector.
These tests use mocking so they run without a GPU or model weights.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import FALLBACK_MODEL_PATH, InferenceConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Return a MagicMock that mimics a YOLO model."""
    model = MagicMock()
    model.names = {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack",
        3: "Pothole",
    }
    # predict / track return a list with one Results-like object
    fake_result = MagicMock()
    fake_result.boxes.cls = None  # no detections by default
    fake_result.boxes.id = None
    model.predict.return_value = [fake_result]
    model.track.return_value = [fake_result]
    return model


@pytest.fixture
def detector(mock_model):
    """Return a Detector with the YOLO model swapped for a mock."""
    from src.core.detector import Detector

    with patch.object(Detector, "_load", return_value=mock_model):
        det = Detector(Path("models/pothole_model.pt"))
    return det


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetector:

    def test_predict_calls_model_predict(self, detector, mock_model):
        import numpy as np
        dummy_image = MagicMock()  # PIL Image placeholder
        cfg = InferenceConfig()

        detector.predict(dummy_image, cfg)

        mock_model.predict.assert_called_once()
        call_kwargs = mock_model.predict.call_args.kwargs
        assert call_kwargs["conf"] == cfg.confidence_threshold
        assert call_kwargs["iou"] == cfg.iou_threshold
        assert call_kwargs["imgsz"] == cfg.image_size

    def test_track_calls_model_track(self, detector, mock_model):
        import numpy as np
        dummy_frame = MagicMock()  # numpy frame placeholder
        cfg = InferenceConfig()

        detector.track(dummy_frame, cfg)

        mock_model.track.assert_called_once()
        call_kwargs = mock_model.track.call_args.kwargs
        assert call_kwargs["conf"] == cfg.confidence_threshold
        assert call_kwargs["persist"] is True
        assert call_kwargs["tracker"] == cfg.tracker

    def test_class_names_returns_dict(self, detector, mock_model):
        assert isinstance(detector.class_names, dict)
        assert detector.class_names[3] == "Pothole"

    def test_general_model_uses_road_damage_fallback(self, mock_model):
        from src.core.detector import Detector

        general_model = MagicMock()
        general_model.names = {0: "toilet"}

        with patch.object(
            Detector, "_load", side_effect=[general_model, mock_model]
        ) as load_model:
            detector = Detector(Path("yolo26.pt"))

        assert detector.requested_model_path == Path("yolo26.pt")
        assert detector.model_path == FALLBACK_MODEL_PATH
        assert "Expected" in detector.fallback_reason
        assert detector.class_names == mock_model.names
        load_model.assert_any_call(str(FALLBACK_MODEL_PATH))

    def test_fallback_failure_is_reported(self):
        from src.core.detector import Detector

        general_model = MagicMock()
        general_model.names = {0: "toilet"}

        with patch.object(
            Detector,
            "_load",
            side_effect=[general_model, FileNotFoundError("missing weights")],
        ):
            with pytest.raises(RuntimeError, match="fallback model"):
                Detector(Path("yolo26.pt"))
