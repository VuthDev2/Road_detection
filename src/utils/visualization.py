"""
src/utils/visualization.py
==========================
Frame annotation and color-space conversion helpers.
Centralizes the repeated BGR → RGB conversion and result plotting pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


def annotated_to_rgb(result: "Results") -> np.ndarray:
    """
    Plot YOLO annotations on a frame and return it as an RGB numpy array
    suitable for use with ``st.image()``.

    Parameters
    ----------
    result : Results
        Single-frame Ultralytics Results object.

    Returns
    -------
    np.ndarray
        Annotated frame in RGB channel order (H × W × 3, uint8).
    """
    annotated_bgr: np.ndarray = result.plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
