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

from config.settings import (
    DAMAGE_CLASSES,
    DEFAULT_SETTINGS,
    FALLBACK_MODEL_PATH,
    InferenceConfig,
)


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
        self.requested_model_path = Path(model_path)
        self.model_path = self.requested_model_path
        self.fallback_reason: str | None = None

        try:
            model = self._load(str(self.model_path))
            self._validate_road_damage_model(model)
        except Exception as primary_error:
            fallback_path = FALLBACK_MODEL_PATH
            if self.model_path.resolve() == fallback_path.resolve():
                raise RuntimeError(
                    f"Road-damage model `{self.model_path}` could not be loaded: "
                    f"{primary_error}"
                ) from primary_error

            try:
                model = self._load(str(fallback_path))
                self._validate_road_damage_model(model)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Selected model `{self.model_path}` is unavailable or "
                    f"incompatible ({primary_error}); fallback model "
                    f"`{fallback_path}` also failed ({fallback_error})."
                ) from fallback_error

            self.model_path = fallback_path
            self.fallback_reason = str(primary_error)

        self._model: YOLO = model

    @staticmethod
    def _validate_road_damage_model(model: YOLO) -> None:
        """Reject general-purpose models that can label unrelated objects."""
        names = model.names
        actual_classes = set(names.values()) if isinstance(names, dict) else set()
        expected_classes = set(DAMAGE_CLASSES)
        if actual_classes != expected_classes:
            raise ValueError(
                "Model classes do not match the road-damage classes. "
                f"Expected {sorted(expected_classes)}; got {sorted(actual_classes)}."
            )

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
