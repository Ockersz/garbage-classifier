# 🗑️ Garbage Classification Streamlit App

This is a simple **Streamlit-based image classifier** for garbage detection (plastic / organic / metal).  
It loads a trained MobileNetV3-Small checkpoint and predicts the class of an uploaded or camera-captured image.

---

## 🚀 Features
- Hardcoded model path (`./weights/garbage_cnn_model.pth` by default)
- Upload an image OR capture from your camera
- Preprocessing matches training (resize, center crop, normalization)
- Prediction shown as top class + probability bar chart + JSON output
- Runs on **CUDA GPU** if available, else CPU

---

## 📂 Project Structure
```
.
├── streamlit_garbage_app_fixed.py   # Streamlit app
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── weights/
    └── garbage_cnn_model.pth        # Your trained model checkpoint
```

---

## 🛠 Installation

1. Clone this repo or copy the files to your project directory.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # on Linux/Mac
   .\.venv\Scripts\activate     # on Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

⚠️ For CUDA: install `torch` and `torchvision` with the correct CUDA wheels from [PyTorch](https://pytorch.org/get-started/locally/).

---

## ▶️ Running the App
```bash
streamlit run streamlit_garbage_app_fixed.py
```

The app will open in your browser at:
- Local: [http://localhost:8501](http://localhost:8501)
- Network: your LAN IP at port 8501

---

## 📊 Usage
- **Upload Image**: Drag & drop or browse for a `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` file.
- **Camera Capture**: Use your webcam directly from the browser.
- The app shows:
  - Input image
  - Predicted class
  - Class probability bar chart
  - JSON with raw probabilities

---

## 🧾 Notes
- Ensure your `weights/garbage_cnn_model.pth` checkpoint matches the training setup.
- If your class order differs, update the `FALLBACK_CLASSES` list in `streamlit_garbage_app_fixed.py`.
- For production, you can Dockerize or deploy on services like Streamlit Cloud or Hugging Face Spaces.

---

## 📜 License
MIT License — use freely for educational or commercial purposes.
