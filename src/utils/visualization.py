"""
src/utils/visualization.py
==========================
Frame annotation and color-space conversion helpers.
Centralizes the repeated BGR → RGB conversion and result plotting pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

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


def extract_detected_crops(
    image: Image.Image,
    result: "Results",
    max_crops: int = 12,
    padding: int = 4,
) -> list[dict[str, Any]]:
    """
    Extract cropped image patches for all detected defect bounding boxes.

    Parameters
    ----------
    image : Image.Image
        The original input PIL Image.
    result : Results
        Ultralytics Results object containing detected boxes.
    max_crops : int, optional
        Maximum number of crops to extract (default is 12).
    padding : int, optional
        Pixel padding around the bounding box (default is 4).

    Returns
    -------
    list[dict[str, Any]]
        List of defect dicts containing:
        - "crop": PIL.Image of the cropped defect
        - "class_name": str
        - "confidence": float (0.0 to 1.0)
        - "box": tuple of (x1, y1, x2, y2)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    crops: list[dict[str, Any]] = []
    width, height = image.size
    class_names = result.names

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.int().cpu().numpy()

    for i in range(min(len(xyxy), max_crops)):
        box = xyxy[i]
        conf = float(confs[i])
        cls_id = int(clses[i])
        class_name = class_names.get(cls_id, f"Class {cls_id}")

        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(width, int(box[2]) + padding)
        y2 = min(height, int(box[3]) + padding)

        if x2 > x1 and y2 > y1:
            crop_img = image.crop((x1, y1, x2, y2))
            crops.append(
                {
                    "crop": crop_img,
                    "class_name": class_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                }
            )

    return crops
