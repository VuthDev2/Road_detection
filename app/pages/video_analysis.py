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


import os
import subprocess

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

    # ── Setup Video Writer ─────────────────────────────────────────────
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    # Setup Video Writer with multi-codec fallback (avc1 works on macOS AVFoundation, mp4v on Linux)
    temp_out_path = video_path.replace(f".{file_extension}", "_out.mp4")
    final_out_path = video_path.replace(f".{file_extension}", "_final.mp4")
    
    writer = None
    for codec in ["avc1", "H264", "mp4v", "MJPG"]:
        try:
            w = cv2.VideoWriter(temp_out_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if w.isOpened():
                writer = w
                break
        except Exception:
            continue

    if writer is None or not writer.isOpened():
        st.error("❌ Failed to initialize video encoder on this system.")
        return

    progress_bar = st.progress(0, text="Processing video in background (this ensures smooth playback)...")
    frame_idx = 0
    last_ui_update = time.time()

    # Clear the media placeholder with an informative message
    media_placeholder.info("⚙️ Processing video frame-by-frame... The final smooth video will appear here when complete.")

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
        
        # ── Write annotated frame to video file ────────────────────────
        # plot() returns a BGR numpy array natively, which cv2.VideoWriter requires
        annotated_bgr = results[0].plot()
        writer.write(annotated_bgr)

        # ── Throttle UI Updates (Max 5 FPS) ────────────────────────────
        current_time = time.time()
        if current_time - last_ui_update > 0.15 or frame_idx == total_frames - 1:
            last_ui_update = current_time
            
            # ── Live annotated frame preview ──────────────────────────────
            live_frame_rgb = annotated_to_rgb(results[0])
            media_placeholder.image(
                live_frame_rgb,
                caption=f"⚡ Live Processing Preview — Frame {frame_idx + 1}/{total_frames}",
                use_container_width=True,
            )

            # ── Update telemetry widgets ───────────────────────────────────
            counts = TrackerParser.class_counts(state)
            total_label = "Unique Tracked Defects"
            total_value = f"{len(state.unique_ids)} (Raw: {state.raw_detections})"
            update_telemetry(telemetry, counts, total_label, total_value)

            # ── Progress bar ───────────────────────────────────────────────
            progress_bar.progress(
                min((frame_idx + 1) / total_frames, 1.0),
                text=f"Frame {frame_idx + 1}/{total_frames} processed...",
            )
        
        frame_idx += 1

    cap.release()
    writer.release()
    
    # ── Convert to H264 for HTML5 Web Playback ─────────────────────────
    progress_bar.progress(1.0, text="Finalizing video encoding for web playback...")
    
    import shutil
    import sys

    ffmpeg_bin = (
        shutil.which("ffmpeg")
        or shutil.which("ffmpeg", path=f"{sys.prefix}/bin")
        or (os.path.join(sys.prefix, "bin", "ffmpeg") if os.path.exists(os.path.join(sys.prefix, "bin", "ffmpeg")) else None)
    )

    display_path = temp_out_path
    if ffmpeg_bin:
        try:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i", temp_out_path,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    final_out_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            display_path = final_out_path
        except Exception:
            display_path = temp_out_path

    progress_bar.empty()
    media_placeholder.empty()

    # ── Play the completely smooth final video! ────────────────────────
    if os.path.exists(display_path):
        with open(display_path, 'rb') as f:
            video_bytes = f.read()
        media_placeholder.video(video_bytes)
    elif os.path.exists(temp_out_path):
        with open(temp_out_path, 'rb') as f:
            video_bytes = f.read()
        media_placeholder.video(video_bytes)
    else:
        st.error("Processed video output file could not be generated.")

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
