"""
src/core/preprocessor.py
========================
Image and video frame enhancement utilities.
Works on both PIL Images (for static image uploads) and numpy BGR frames
(from cv2.VideoCapture), so the same logic is shared across both pipelines.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from config.settings import CONTRAST_FACTOR, SHARPNESS_FACTOR


class Preprocessor:
    """
    Applies contrast and sharpness enhancement to images or video frames.

    These enhancements are designed to improve detection accuracy on
    muddy or low-contrast road surfaces.

    Parameters
    ----------
    contrast_factor : float
        Multiplier for PIL ImageEnhance.Contrast. Default from settings.
    sharpness_factor : float
        Multiplier for PIL ImageEnhance.Sharpness. Default from settings.
    """

    def __init__(
        self,
        contrast_factor: float = CONTRAST_FACTOR,
        sharpness_factor: float = SHARPNESS_FACTOR,
    ) -> None:
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance contrast and sharpness of a PIL Image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image.

        Returns
        -------
        PIL.Image.Image
            Enhanced image.
        """
        image = ImageEnhance.Contrast(image).enhance(self.contrast_factor)
        image = ImageEnhance.Sharpness(image).enhance(self.sharpness_factor)
        return image

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast and sharpness of a BGR numpy frame (from cv2).

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame from cv2.VideoCapture.

        Returns
        -------
        np.ndarray
            Enhanced BGR frame.
        """
        # Convert BGR → RGB for PIL, enhance, convert back
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image = self.enhance_image(pil_image)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
