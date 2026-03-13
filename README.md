# AI Sign Language Alphabet Translator

A real-time American Sign Language (ASL) alphabet translator built with **Python**, **MediaPipe**, and **TensorFlow**. This project uses computer vision to detect hand landmarks and a Deep Neural Network (DNN) to classify them into letters.

## 🚀 Features

- **Real-time Detection:** High-speed hand landmark tracking using Google MediaPipe.
- **Modern GUI:** Built with `CustomTkinter` for a sleek, user-friendly interface.
- **Landmark-Based AI:** Instead of raw pixels, the model trains on numeric (x, y, z) coordinates, making it lightweight and fast.
- **Supports 29 Classes:** Alphabets (A-Z), Space, Delete, and Nothing.

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **AI Framework:** TensorFlow / Keras
- **Computer Vision:** MediaPipe, OpenCV
- **GUI:** CustomTkinter
- **Training Ground:** Google Colab (GPU)

## 📁 Project Architecture

```text
sign-language-translator/
├── dataset/                 # ASL Alphabet images (A-Z, Space, etc.)
├── svenv/                   # Virtual Environment
├── app.py                   # Main GUI Application (Real-time detection)
├── utils.py                 # Utility functions & MediaPipe logic
├── collect_landmarks.py     # Script to convert images to numeric data
├── setup_data.py            # Folder initialization script
├── sign_language_model.keras # Trained AI Model
├── X_data.npy               # Processed hand landmarks
└── y_data.npy               # Labels for training
```

## ⚙️ Installation & Setup

1. **Clone the repository and enter the folder:**

   ```bash
   cd Sign_language_project
   ```

2. **Create and activate the virtual environment:**

   ```bash
   python -m venv svenv
   svenv\Scripts\activate  # Windows
   source svenv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install mediapipe opencv-python tensorflow==2.16.1 numpy<2.0.0 customtkinter pillow
   ```

## 📑 Workflow

### 1. Data Preparation

Organize your images in `dataset/asl_alphabet_train/`. Each letter should have its own folder (A, B, C...).
Run the extraction script to convert images into landmarks:

```bash
python collect_landmarks.py
```

This generates `X_data.npy` and `y_data.npy`.

### 2. Model Training

1. Upload `X_data.npy` and `y_data.npy` to **Google Colab**.
2. Build and train the `Sequential` Dense Neural Network.
3. Download the resulting `sign_language_model.keras` file and place it in the project root.

### 3. Real-time Translation

Run the main application:

```bash
python app.py
```

Click **"Start Translator"** to begin detecting signs via your webcam.

## ⚠️ Important Notes

- **Mirror Effect:** The camera feed is horizontally flipped to act as a mirror for easier signing.
- **Detection Confidence:** The model only displays predictions with a confidence score higher than **70%** to avoid flickering.
- **Environment:** Ensure you use the specific library versions listed to avoid compatibility issues with Python 3.12.

## 🤝 Credits

- **MediaPipe:** For the hand landmark solution.
- **Kaggle:** For the ASL Alphabet dataset.

---
