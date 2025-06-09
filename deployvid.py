import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import numpy as np
import io

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ganti dengan yolov11 atau custom model kamu

model = load_model()

st.title("🎥 YOLOv11 Video Object Detection")
st.markdown("Upload a video (.mp4) and detect objects frame by frame.")

uploaded_video = st.file_uploader("Upload MP4 video", type=["mp4"])

if uploaded_video is not None:
    # Simpan video sementara
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())

    st.video(tfile.name, format='video/mp4')

    st.info("Processing video with YOLOv11... please wait.")

    # Output path
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Gunakan codec yang lebih kompatibel
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Proses frame demi frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame.astype(np.uint8))

    cap.release()
    out.release()

    # Baca file sebagai bytes untuk diputar
    with open(out_path, 'rb') as f:
        video_bytes = io.BytesIO(f.read())

    st.success("✅ Detection complete. See result below:")
    st.video(video_bytes)
    try:
        os.remove(tfile.name)
        os.remove(out_path)
    except Exception as e:
        st.warning(f"Gagal menghapus file sementara: {e}")

