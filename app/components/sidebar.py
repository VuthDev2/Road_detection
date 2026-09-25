"""
app/components/sidebar.py
=========================
Renders the Streamlit sidebar and returns a fully-populated InferenceConfig.
All UI state is isolated here — the pages never touch st.sidebar directly.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_SETTINGS,
    InferenceConfig,
)


def render_sidebar() -> tuple[Path, InferenceConfig]:
    """
    Render the sidebar controls and return the selected model path
    and a configured InferenceConfig dataclass.

    Returns
    -------
    tuple[Path, InferenceConfig]
        (model_path, cfg) — pass these into the Detector and page functions.
    """
    st.sidebar.header("⚙️ Inference Settings")
    st.sidebar.markdown("---")

    # ── Model selection ────────────────────────────────────────────────
    st.sidebar.subheader("🤖 Model")
    model_label = st.sidebar.selectbox(
        "Architecture",
        options=list(AVAILABLE_MODELS.keys()),
        help="Switch between the available trained model weights.",
    )
    model_path = AVAILABLE_MODELS[model_label]

    st.sidebar.markdown("---")

    # ── Detection hyperparameters ──────────────────────────────────────
    st.sidebar.subheader("🎯 Detection")
    d = DEFAULT_SETTINGS  # shorthand for bounds

    conf = st.sidebar.slider(
        "Confidence Threshold",
        min_value=d.CONF_MIN,
        max_value=d.CONF_MAX,
        value=d.confidence_threshold,
        step=d.CONF_STEP,
        help="Minimum score for a detection to be shown.",
    )

    iou = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=d.IOU_MIN,
        max_value=d.IOU_MAX,
        value=d.iou_threshold,
        step=d.IOU_STEP,
        help="Non-maximum suppression overlap threshold.",
    )

    img_size = st.sidebar.selectbox(
        "Image Size (imgsz)",
        options=list(d.IMAGE_SIZE_OPTIONS),
        index=0,
        help="Input resolution fed to the model. Larger = slower but more accurate.",
    )

    st.sidebar.markdown("---")

    # ── Preprocessing ──────────────────────────────────────────────────
    st.sidebar.subheader("🔧 Preprocessing")
    enhance = st.sidebar.checkbox(
        "Contrast Enhancement",
        value=d.enhance_contrast,
        help="Boosts contrast & sharpness — useful for muddy or low-light roads.",
    )

    cfg = InferenceConfig(
        confidence_threshold=conf,
        iou_threshold=iou,
        image_size=img_size,
        enhance_contrast=enhance,
    )

    return model_path, cfg
