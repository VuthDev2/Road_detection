"""
app/pages/image_analysis.py
============================
Handles the static image upload → inference → display pipeline.
Calls src/ modules for all heavy lifting; only Streamlit rendering lives here.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from config.settings import InferenceConfig, DAMAGE_CLASSES
from src.core.detector import Detector
from src.core.preprocessor import Preprocessor
from src.utils.visualization import annotated_to_rgb
from app.components.telemetry import (
    TelemetryWidgets,
    update_telemetry,
)


def render_image_page(
    uploaded_file,
    detector: Detector,
    cfg: InferenceConfig,
    media_placeholder,
    telemetry: TelemetryWidgets,
) -> None:
    """
    Run the full image analysis pipeline and update the UI.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object (image).
    detector : Detector
        Pre-loaded inference wrapper.
    cfg : InferenceConfig
        Active inference hyperparameters.
    media_placeholder : DeltaGenerator
        Streamlit placeholder for displaying the annotated image.
    telemetry : TelemetryWidgets
        Telemetry widget references to update with detection counts.
    """
    preprocessor = Preprocessor()

    with st.spinner("🔍 Running inference…"):
        image = Image.open(uploaded_file)

        if cfg.enhance_contrast:
            image = preprocessor.enhance_image(image)

        results = detector.predict(image, cfg)

    # ── Count detections by class ──────────────────────────────────────
    class_counts: dict[str, int] = {name: 0 for name in DAMAGE_CLASSES}
    total_defects = 0

    if results[0].boxes.cls is not None:
        class_names_map = results[0].names
        for cls_idx in results[0].boxes.cls.int().cpu().tolist():
            total_defects += 1
            name = class_names_map[cls_idx]
            if name in class_counts:
                class_counts[name] += 1

    # ── Update UI ──────────────────────────────────────────────────────
    update_telemetry(telemetry, class_counts, "Total Defects", total_defects)

    frame_rgb = annotated_to_rgb(results[0])
    media_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    st.success(f"✅ Image analysis complete — **{total_defects}** defect(s) detected.")
