"""
app/pages/video_analysis.py
============================
Handles the video upload → frame-by-frame tracking → export pipeline.
All business logic (tracking state, CSV export) lives in src/ modules.
"""

from __future__ import annotations

import tempfile
import time

import cv2
import streamlit as st

from config.settings import InferenceConfig
from src.core.detector import Detector
from src.core.preprocessor import Preprocessor
from src.core.tracker import TrackerParser, TrackingState
from src.utils.export import build_csv_buffer
from src.utils.visualization import annotated_to_rgb
from app.components.telemetry import (
    TelemetryWidgets,
    update_telemetry,
)


def render_video_page(
    uploaded_file,
    file_extension: str,
    detector: Detector,
    cfg: InferenceConfig,
    media_placeholder,
    telemetry: TelemetryWidgets,
) -> None:
    """
    Run the full video analysis pipeline: read frames, track defects,
    update the live UI, and offer a CSV download when done.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object (video).
    file_extension : str
        File extension without the dot (e.g. "mp4").
    detector : Detector
        Pre-loaded inference wrapper.
    cfg : InferenceConfig
        Active inference hyperparameters.
    media_placeholder : DeltaGenerator
        Streamlit placeholder for the live annotated frame feed.
    telemetry : TelemetryWidgets
        Telemetry widget references updated each frame.
    """
    preprocessor = Preprocessor()
    parser = TrackerParser()
    state = TrackingState()

    # ── Write upload to a temp file so cv2 can read it ─────────────────
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{file_extension}"
    ) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open video file. Please try a different file.")
        return

    progress_bar = st.progress(0, text="Processing video…")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0
    last_ui_update = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Preprocessing ──────────────────────────────────────────────
        if cfg.enhance_contrast:
            frame = preprocessor.enhance_frame(frame)

        # ── Inference + tracking ───────────────────────────────────────
        results = detector.track(frame, cfg)
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ── Update tracking state ──────────────────────────────────────
        parser.update(state, results[0], timestamp_sec)

        # ── Throttle UI Updates (Max 4 FPS to prevent WebSocket freeze) ──
        current_time = time.time()
        if current_time - last_ui_update > 0.25 or frame_idx == total_frames - 1:
            last_ui_update = current_time
            
            # ── Update telemetry widgets ───────────────────────────────────
            counts = TrackerParser.class_counts(state)
            total_label = "Unique Tracked Defects"
            total_value = f"{len(state.unique_ids)} (Raw: {state.raw_detections})"
            update_telemetry(telemetry, counts, total_label, total_value)

            # ── Show annotated frame (resized for smooth Cloud playback) ───
            frame_rgb = annotated_to_rgb(results[0])
            
            # Resize for UI display to drastically reduce network payload
            h, w = frame_rgb.shape[:2]
            max_width = 640
            if w > max_width:
                scale = max_width / w
                new_w, new_h = int(w * scale), int(h * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                
            media_placeholder.image(
                frame_rgb, channels="RGB", use_container_width=True
            )
            
            # Yield to Tornado event loop to flush WebSocket messages
            time.sleep(0.05)

            # ── Progress bar ───────────────────────────────────────────────
            progress_bar.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx}/{total_frames}",
            )
        
        frame_idx += 1

    cap.release()
    progress_bar.empty()

    st.success(
        f"✅ Video analysis complete — "
        f"**{len(state.unique_ids)}** unique defect(s) tracked."
    )

    # ── CSV export ─────────────────────────────────────────────────────
    if state.defect_log:
        csv_buffer = build_csv_buffer(state)
        st.download_button(
            label="⬇️ Export Defect Log (CSV)",
            data=csv_buffer.getvalue(),
            file_name="road_damage_log.csv",
            mime="text/csv",
        )
