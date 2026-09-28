# ⛑️ Helmet Detection using YOLO11

A real-time helmet detection web app built with **YOLO11n** and **Streamlit**.  
Detects whether a person is wearing a helmet or not from images and videos.

---

## 🚀 Demo

Upload an image or video → get instant bounding box detections with confidence scores.

---

## 🧠 Model

- **Architecture:** YOLO11n (nano) — lightweight, fast, accurate
- **Dataset:** [Kaggle Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) — 764 images
- **Classes:** `helmet` · `no_helmet`
- **Training:** 50 epochs · 640px · Google Colab (Tesla T4)
- **Framework:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)

---

## ✨ Features

- 🖼️ **Image Detection** — Upload JPG/PNG and get annotated results instantly
- 🎬 **Video Detection** — Upload MP4/AVI/MOV, process frame-by-frame, download result
- 📊 **Detection Summary** — Total detections, with/without helmet count, per-detection confidence
- ⚠️ **Violation Alert** — Red alert if any person is detected without a helmet
- ⬇️ **Download** — Save the annotated image or video

---

## 🛠️ Installation

```bash
git clone https://github.com/ankitapeddakota926/helmet-detection.git
cd helmet-detection
pip install -r requirements.txt
▶️ Run
streamlit run app.py
Opens at http://localhost:8501

📁 Project Structure
helmet-detection/
├── app.py              # Streamlit web app
├── best.pt             # Trained YOLO11n weights
├── requirements.txt    # Python dependencies
└── Untitled1.ipynb     # Google Colab training notebook
📦 Requirements
Python 3.10+
streamlit
ultralytics
opencv-python
torch / torchvision
Pillow · numpy
📜 License
This project uses the Ultralytics YOLO framework.
Dataset: CC0 1.0 Public Domain
