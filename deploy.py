import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Load the YOLOv11 model (or your custom model path)
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Change to "yolov11n.pt" or custom path if needed

model = load_model()

st.title("🚀 YOLOv11 Object Detection")
st.markdown("Upload an image and detect objects using YOLOv11")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            results = model(temp.name)

    # Get the result image
    result_image = results[0].plot()  # draws bounding boxes on the image
    st.image(result_image, caption="Detected Image", use_column_width=True)

    # Show results table
    st.subheader("Detection Results")
    boxes = results[0].boxes
    if boxes is not None:
        st.dataframe(boxes.data.cpu().numpy())
    else:
        st.write("No objects detected.")
