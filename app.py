import streamlit as st
import cv2
import csv
import io
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage & Pothole Tracker", layout="wide")
st.title("🛣️ Real-Time Road Damage Detection & Tracking")
st.markdown("Automated defect logging powered by custom YOLOv8 and ByteTrack.")

# Sidebar settings
st.sidebar.header("Inference Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.12, 0.05)
model_path = st.sidebar.text_input("Model Weights Path", "pothole_model.pt")

@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_path)
    st.sidebar.success("✓ Model loaded")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

uploaded_video = st.file_uploader("Upload Dashcam Footage", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video.read())
        video_path = tfile.name

    col1, col2 = st.columns([3, 1])
    video_placeholder = col1.empty()
    
    with col2:
        st.subheader("Live Telemetry")
        class_metrics = {
            "Longitudinal Crack": st.empty(),
            "Transverse Crack": st.empty(),
            "Alligator Crack": st.empty(),
            "Pothole": st.empty(),
        }
        total_defects_metric = st.empty()

    cap = cv2.VideoCapture(video_path)
    unique_ids = set()
    class_ids = {class_name: set() for class_name in class_metrics}
    defect_log = {}

    start_btn = st.button("Start Processing")

    if start_btn:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO + ByteTrack
            results = model.track(
                source=frame,
                conf=conf_threshold,
                imgsz=640,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )

            # Extract tracking details
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()
                class_names = results[0].names
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                for tid, cls_idx in zip(track_ids, classes):
                    unique_ids.add(tid)
                    class_name = class_names[cls_idx]
                    if class_name in class_ids:
                        class_ids[class_name].add(tid)
                    if tid not in defect_log:
                        defect_log[tid] = {
                            "Track ID": tid,
                            "Timestamp (seconds)": round(timestamp, 3),
                            "Class": class_name,
                        }

            # Render live counters
            for class_name, metric in class_metrics.items():
                metric.metric(class_name, len(class_ids[class_name]))
            total_defects_metric.metric("Total Unique Defects", len(unique_ids))

            # Display annotated frame
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success("Analysis complete.")

        csv_buffer = io.StringIO()
        fieldnames = ["Track ID", "Timestamp (seconds)", "Class"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(defect_log.values())
        st.download_button(
            "Export Log",
            data=csv_buffer.getvalue(),
            file_name="road_damage_log.csv",
            mime="text/csv",
        )