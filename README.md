# 🚀 Space Object Detection using YOLOv8–YOLOv11

## 📌 Project Overview
This project implements a deep learning–based space object detection system to identify:

- 🌠 Comets  
- 🌌 Galaxies  
- ✨ Globular Clusters  
- 🌫 Nebulae  

The project compares YOLOv8 and YOLOv11 architectures under controlled training settings and evaluates performance improvements across key object detection metrics.

---

## 🧠 Objectives
- Train YOLOv8 and YOLOv11 on an astronomical object detection dataset
- Perform fair architectural comparison
- Evaluate using Precision, Recall, mAP@0.5, and mAP@0.5:0.95
- Deploy the best model using Streamlit for real-time inference

---

## 📂 Dataset
- Source: Kaggle Astronomical Object Detection Dataset
- Structured into:
  - Train
  - Validation
  - Test splits
- Total Classes: 4

---

## ⚙️ Model Training
- Framework: Ultralytics YOLO
- GPU: NVIDIA RTX 3050
- Epochs: 50
- Image Size: 640
- Batch Size: 16

Both YOLOv8 and YOLOv11 were trained under identical settings for fair comparison.

---

## 📊 Performance Comparison

| Metric        | YOLOv8 | YOLOv11 |
|--------------|--------|---------|
| Precision    | ~0.79  | ~0.83   |
| Recall       | ~0.67  | ~0.71   |
| mAP@0.5      | ~0.74  | ~0.78   |
| mAP@0.5:0.95 | ~0.43  | ~0.46   |

📈 YOLOv11 achieved consistent improvement across all evaluation metrics.

---

## 🖥 Deployment (Streamlit App)

The trained YOLOv11 model is deployed locally using Streamlit.

### Run the App

```bash
conda activate yolo_space
streamlit run app.py
