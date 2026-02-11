import streamlit as st
from ultralytics import YOLO
import os

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = r"C:\Users\Manoj\Documents\DL_SPACE CLEAN"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "yolov11_space_objects",   # change to yolov11_space_objects2 if needed
    "weights",
    "best.pt"
)

# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Space Object Detection", layout="centered")

st.title("🚀 Space Object Detection using YOLOv11")
st.write("Upload an astronomical image to detect:")
st.write("• Comet  • Galaxy  • Globular Cluster  • Nebula")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# INFERENCE
# ==============================

if uploaded_file is not None:

    # Save uploaded file with correct extension
    file_extension = uploaded_file.name.split(".")[-1]
    temp_path = os.path.join(BASE_DIR, f"temp_upload.{file_extension}")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Running detection...")

    # Run YOLO prediction
    results = model.predict(
        source=temp_path,
        conf=0.25
    )

    # Plot result
    result_image = results[0].plot()

    st.image(result_image, caption="Detection Result", use_container_width=True)


    # Remove temporary file
    os.remove(temp_path)
