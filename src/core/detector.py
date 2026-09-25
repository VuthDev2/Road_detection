"""
src/core/detector.py
====================
YOLO inference wrapper — completely decoupled from the Streamlit UI.
Supports both single-frame prediction (images) and per-frame tracking (video).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from ultralytics import YOLO

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    import numpy as np
    from PIL.Image import Image as PILImage

from config.settings import InferenceConfig, DEFAULT_SETTINGS


class Detector:
    """
    Thin wrapper around an Ultralytics YOLO model.

    Parameters
    ----------
    model_path : str | Path
        Path to a .pt weights file.

    Usage
    -----
    >>> det = Detector("models/pothole_model.pt")
    >>> results = det.predict(image, cfg)
    >>> results = det.track(frame, cfg)
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model: YOLO = self._load(str(self.model_path))

    @staticmethod
    @st.cache_resource
    def _load(path: str) -> YOLO:
        """Load and cache the YOLO model across Streamlit reruns."""
        return YOLO(path)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict(
        self,
        source: "PILImage | np.ndarray",
        cfg: InferenceConfig = DEFAULT_SETTINGS,
    ) -> list["Results"]:
        """
        Run object detection on a static image.

        Parameters
        ----------
        source : PIL.Image | np.ndarray
            Input image.
        cfg : InferenceConfig
            Inference hyperparameters.

        Returns
        -------
        list[Results]
            Ultralytics Results list (typically length 1 for a single image).
        """
        return self._model.predict(
            source=source,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            imgsz=cfg.image_size,
            verbose=False,
        )

    def track(
        self,
        frame: "np.ndarray",
        cfg: InferenceConfig = DEFAULT_SETTINGS,
    ) -> list["Results"]:
        """
        Run object detection + ByteTrack on a single video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from cv2.VideoCapture.
        cfg : InferenceConfig
            Inference hyperparameters.

        Returns
        -------
        list[Results]
            Ultralytics Results list with .boxes.id populated by ByteTrack.
        """
        return self._model.track(
            source=frame,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            imgsz=cfg.image_size,
            tracker=cfg.tracker,
            persist=True,
            verbose=False,
        )

    @property
    def class_names(self) -> dict[int, str]:
        """Return the model's class index → name mapping."""
        return self._model.names
