Here is a professional and attractive README.md for your Helmet Detection using YOLO11 project. You can copy and paste it directly into the `README.md` file in your GitHub repository.

⛑️ Helmet Detection Using YOLO11

# ⛑️ Helmet Detection Using YOLO11

A real-time helmet detection web application built using **YOLO11n and Streamlit** to identify whether a person is wearing a helmet or not. The application detects helmets and no-helmet cases from uploaded images and videos and displays the results with bounding boxes and confidence scores.

## 🚀 Live Demo

🔗 **Try the Application:** [Helmet Detection Web App](https://helmet-detection-wrd3rbxe37qcphyt28jrca.streamlit.app/)

🔗 **GitHub Repository:** [Helmet Detection using YOLO11](https://github.com/ankitapeddakota926/helmet-detection)

Upload an image or video to detect helmet usage and view the detection results.

---

## 📌 Project Overview

Wearing helmets is essential for protecting people from head injuries, especially in construction sites and other workplaces.

This project uses a pre-trained and fine-tuned **YOLO11n object detection model** to identify people wearing helmets and those not wearing helmets. It provides a simple web interface where users can upload images or videos and view the detection results.

### 🎯 Objectives

* Detect helmet and no-helmet cases using deep learning.

* Identify objects with bounding boxes and confidence scores.

* Provide an easy-to-use web application for helmet detection.

* Display detection summaries and highlight possible safety violations.

---

## 🧠 Model Details

| Parameter            | Description                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------- |
| Model Architecture   | YOLO11n (Nano)                                                                                |
| Dataset              | [Kaggle Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) |
| Number of Images     | 764 images                                                                                    |
| Classes              | `helmet`, `no_helmet`                                                                         |
| Training Epochs      | 50                                                                                            |
| Image Size           | 640 × 640                                                                                     |
| Training Environment | Google Colab                                                                                  |
| GPU                  | NVIDIA Tesla T4                                                                               |
| Framework            | Ultralytics YOLO11                                                                            |

---

## ✨ Features

### 🖼️ Image Detection

* Upload images in JPG, JPEG, or PNG format.

* Detect helmet and no-helmet cases.

* Display annotated images with bounding boxes and confidence scores.

### 🎬 Video Detection

* Upload videos in supported formats such as MP4, AVI, and MOV.

* Process video frames to identify helmet usage.

* Download the processed video with detection results.

### 📊 Detection Summary

* Display the total number of detections.

* Show helmet and no-helmet counts.

* Display confidence scores for detected objects.

### ⚠️ Safety Violation Alert

* Highlight detections of people without helmets.

* Display a warning when a no-helmet case is detected.

### ⬇️ Download Results

* Download annotated images.

* Save processed videos with detection results.

---

## 🛠️ Technologies Used

* **Python** – Main programming language

* **YOLO11n** – Object detection model

* **Streamlit** – Web application interface

* **Ultralytics** – YOLO model framework

* **OpenCV** – Image and video processing

* **PyTorch** – Deep learning framework

* **Pillow** – Image processing

* **NumPy** – Numerical operations

* **Google Colab** – Model training environment

---

## 📂 Project Structure

```
helmet-detection/
│
├── app.py                # Streamlit web application
├── best.pt               # Trained YOLO11 model weights
├── requirements.txt      # Project dependencies
├── Untitled1.ipynb       # Model training notebook
└── README.md             # Project documentation
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```
git clone https://github.com/ankitapeddakota926/helmet-detection.git
```

### 2. Navigate to the Project Folder

```
cd helmet-detection
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run the Application

```
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 📦 Requirements

* Python 3.10 or later

* Streamlit

* Ultralytics

* OpenCV

* PyTorch

* Torchvision

* Pillow

* NumPy

Install all dependencies using the `requirements.txt` file.

---

## 🔄 Project Workflow

```
Input Image / Video
        ↓
Image or Video Preprocessing
        ↓
YOLO11n Detection Model
        ↓
Helmet / No-Helmet Detection
        ↓
Bounding Boxes and Confidence Scores
        ↓
Detection Summary and Safety Alert
        ↓
Display and Download Results
```

---

## 🌍 Applications

* Construction site safety monitoring

* Industrial workplace safety

* Road safety and helmet compliance monitoring

* Personal protective equipment (PPE) detection

* Safety awareness and monitoring systems

---

## 📈 Future Enhancements

* Real-time detection using CCTV cameras.

* Integration with automated safety monitoring systems.

* Improved detection accuracy in different lighting conditions.

* Multi-person tracking and violation logging.

* Automated notifications for repeated safety violations.

---

## 📜 Dataset and License

**Dataset:** [Kaggle Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)

The dataset is described as being available under the **CC0 1.0 Public Domain Dedication**. Please refer to the original dataset page for licensing details.

**Model Framework:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

Refer to the Ultralytics repository for its licensing terms.

---

## 👩‍💻 Author

**Ankita Peddakota**

B.Tech – Computer Science and Engineering

🔗 [GitHub Profile](https://github.com/ankitapeddakota926)

---

⭐ If you find this project useful, consider giving the repository a star!

