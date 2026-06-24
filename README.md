# 🧠 ADHD Diagnostics Pro

**ADHD Diagnostics Pro** is an advanced, deep-learning-based web application designed to detect Attention-Deficit/Hyperactivity Disorder (ADHD) from neural signal captures (EEG / fMRI images). It features a highly premium, modern, and intuitive user interface built with Streamlit.

To ensure transparency and trust in its predictions, the application incorporates a **Neural Explainability Engine (XAI Suite)** utilizing simultaneous **Grad-CAM** and **LIME** analysis to visualize the internal decision pathways of the neural network.

---

## 🌟 Key Features

- **Advanced Diagnostics AI**: Utilizes the state-of-the-art **EfficientNetV2** architecture, optimized for medical spectral analysis and pattern recognition.
- **Hyper-Premium UI Dashboard**: A sleek, dark-themed responsive user interface featuring glassmorphism elements, dynamic color themes based on inference outcomes, and smooth micro-animations.
- **Simultaneous Explainable AI (XAI)**:
  - **Grad-CAM**: Generates a spatial activation heatmap highlighting the discriminative regions that drove the model's prediction.
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Provides super-pixel attribution for granular feature importance, highlighting pixel clusters that positively or negatively influence the model's decision.
- **Robust Session Management**: Prevents UI flickering and disappearing components by robustly tracking the application state in Streamlit's session state. 

---

## 🛠️ Technology Stack

- **Python**: Core programming language.
- **Streamlit**: Web application framework used for the frontend dashboard.
- **TensorFlow / Keras**: Used for loading the EfficientNetV2 neural network model and executing inference.
- **LIME & OpenCV (cv2)**: Utilized for generating visual attributions and spatial heatmaps.
- **Matplotlib & PIL**: Employed for image processing and chart generation.

---

## ⚙️ Setup and Installation

### Prerequisites
Make sure you have Python installed (preferably version 3.8 to 3.11).

1. Clone or download the project directory.
2. Ensure you have the required dependencies installed. You can install them via pip:
   ```bash
   pip install streamlit tensorflow numpy Pillow matplotlib opencv-python lime
   ```
3. **Model Placement**: Ensure that the pre-trained model file named `new_joint_best_model.keras` is placed in the root directory alongside `app.py`.

---

## 🚀 Usage Guide

1. **Launch the application**:
   Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
   *Alternatively, if you are on Windows, you can double-click the `run_app.bat` file.*

2. **Upload a Scan**:
   - Use the drag-and-drop uploader to input a valid EEG or fMRI neural signal capture (Formats: `.jpg`, `.jpeg`, `.png`).
   
3. **Analyze**:
   - Click the **"🚀 Analyze Scan"** button. The AI will preprocess the image and pass it through the EfficientNetV2 network.
   - The UI will dynamically display the **Diagnostic Inference** (ADHD or Control) alongside a Confidence Score.

4. **Run XAI Suite**:
   - After the classification, click on **"⚡ Run Simultaneous XAI Analysis"**.
   - The app will concurrently calculate and display both the **Grad-CAM** heatmap and the **LIME** attribution maps, explaining *why* the model made its decision.

---

## ⚠️ Clinical Disclaimer

This system is designed for **research and educational purposes only**. It is not a substitute for professional clinical diagnosis. Decisions regarding health should always involve a qualified medical practitioner.

