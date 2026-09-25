"""
app/main.py
===========
Streamlit application entry point for Road Damage Detection.

Responsibilities (thin orchestrator only):
  1. Page config & global styling
  2. Render sidebar → get model path + inference config
  3. Load the model (cached by Detector)
  4. Accept file upload
  5. Route to image_analysis or video_analysis page
"""

import streamlit as st

from config.settings import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, ALL_EXTENSIONS
from src.core.detector import Detector
from app.components.sidebar import render_sidebar
from app.components.telemetry import render_telemetry_panel
from app.pages.image_analysis import render_image_page
from app.pages.video_analysis import render_video_page


def main() -> None:
    # ── Page config ────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Road Damage Detection",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Header ─────────────────────────────────────────────────────────
    st.title("🛣️ Real-Time Road Damage Detection & Tracking")
    st.markdown(
        "Automated defect logging powered by **YOLOv8 / YOLO26** and **ByteTrack**. "
        "Upload dashcam footage or a road image to get started."
    )
    st.divider()

    # ── Sidebar → settings ─────────────────────────────────────────────
    model_path, cfg = render_sidebar()

    # ── Load model (cached) ────────────────────────────────────────────
    try:
        detector = Detector(model_path)
        st.sidebar.success(f"✓ Model loaded: `{model_path.name}`")
    except Exception as exc:
        st.sidebar.error(f"Error loading model: {exc}")
        st.stop()

    # ── File upload ────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📁 Upload Dashcam Footage or Road Image",
        type=ALL_EXTENSIONS,
        help="Supported formats: MP4, MOV, AVI (video) · PNG, JPG, JPEG (image)",
    )

    if uploaded_file is None:
        st.info("👆 Upload a file above to begin analysis.")
        return

    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    # ── Layout: media panel (left) + telemetry (right) ─────────────────
    col_media, col_telemetry = st.columns([3, 1])
    media_placeholder = col_media.empty()

    with col_telemetry:
        telemetry = render_telemetry_panel()

    # ── Start button ───────────────────────────────────────────────────
    if not st.button("▶️ Start Processing", type="primary", use_container_width=False):
        return

    # ── Route to the correct page ──────────────────────────────────────
    if file_ext in IMAGE_EXTENSIONS:
        render_image_page(
            uploaded_file=uploaded_file,
            detector=detector,
            cfg=cfg,
            media_placeholder=media_placeholder,
            telemetry=telemetry,
        )
    elif file_ext in VIDEO_EXTENSIONS:
        render_video_page(
            uploaded_file=uploaded_file,
            file_extension=file_ext,
            detector=detector,
            cfg=cfg,
            media_placeholder=media_placeholder,
            telemetry=telemetry,
        )
    else:
        st.error(f"Unsupported file type: `.{file_ext}`")


if __name__ == "__main__":
    main()
