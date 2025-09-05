#!/usr/bin/env python3
# streamlit_garbage_app_fixed.py
# --------------------------------
# Minimal Streamlit UI for single-image inference with a hardcoded model path & classes.
# - No sidebar; meant for deployment.
# - Upload an image OR capture from camera.
# - Applies same preprocessing and shows prediction + probabilities.
#
# Run:
#   streamlit run streamlit_garbage_app_fixed.py
#
# Place your checkpoint here relative to this file (adjust as needed):
#   ./weights/garbage_cnn_model.pth   (plain state_dict)  OR
#   ./weights/garbage_cnn_best.pt     (dict with {"model","classes","img_size"})
#
from __future__ import annotations
import io
from pathlib import Path
from typing import List, Dict

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn

# TorchVision imports (with a small compatibility shim)
try:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
except Exception:
    # Older torchvision fallback
    try:
        from torchvision.models.mobilenetv3 import mobilenet_v3_small  # type: ignore
        MobileNet_V3_Small_Weights = None  # type: ignore
    except Exception as e:
        st.error(f"torchvision is too old/new to import MobileNetV3-Small: {e}")
        st.stop()

from torchvision import transforms

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
# Hardcode the relative path to your model checkpoint (adjust if needed)
MODEL_REL_PATH = Path("weights/garbage_cnn_model.pth")  # or "weights/garbage_cnn_best.pt"
CKPT_PATH = (BASE_DIR / MODEL_REL_PATH).resolve()

# Prefer reading classes/img_size from checkpoint if available (recommended)
PREFER_CLASSES_FROM_CKPT = True

# If the checkpoint lacks classes, fall back to this list (keep training order!).
FALLBACK_CLASSES: List[str] = ["metal", "organic", "plastic"]

# If your checkpoint lacks img_size metadata, this value is used.
DEFAULT_IMG_SIZE = 224

# Device preference (auto-falls back to CPU if CUDA not present)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------- Model & Preprocess ----------------
def load_checkpoint(ckpt_path: Path, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        classes = obj.get("classes")
        img_size = int(obj.get("img_size", DEFAULT_IMG_SIZE))
        state_dict = obj["model"]
        return state_dict, classes, img_size
    # plain state dict (no metadata)
    return obj, None, DEFAULT_IMG_SIZE


def build_model(num_classes: int) -> nn.Module:
    # Initialize base backbone (weights optional; we replace classifier head anyway)
    try:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)  # type: ignore[arg-type]
    except Exception:
        model = mobilenet_v3_small(weights=None)  # type: ignore[call-arg]
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model


def build_preprocess(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@st.cache_resource(show_spinner=False)
def load_model_cached(ckpt_path: str):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        return None

    state_dict, classes, img_size = load_checkpoint(ckpt, DEVICE)

    # Resolve classes and preprocessing size
    if not (PREFER_CLASSES_FROM_CKPT and classes):
        classes = FALLBACK_CLASSES
    preprocess = build_preprocess(img_size if PREFER_CLASSES_FROM_CKPT else DEFAULT_IMG_SIZE)

    model = build_model(num_classes=len(classes)).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Not fatal; typically just classifier head names or pretrain buffers
        pass
    model.eval()

    return {"model": model, "classes": classes, "preprocess": preprocess, "img_size": img_size, "device": DEVICE}


def predict_image(model: nn.Module, preprocess, device: torch.device, img: Image.Image, classes: List[str]) -> Dict[str, float]:
    with torch.no_grad():
        x = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().tolist()
    results = dict(sorted({c: float(p) for c, p in zip(classes, prob)}.items(), key=lambda kv: kv[1], reverse=True))
    return results


def plot_probs(probs: Dict[str, float]):
    import matplotlib.pyplot as plt
    labels = list(probs.keys())
    values = [probs[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig, clear_figure=True)


# ---------------- UI ----------------
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Classification")

# Load model once (hardcoded path)
loaded = load_model_cached(str(CKPT_PATH))
if not loaded:
    st.error(f"Checkpoint not found at {CKPT_PATH}. Place your model at this path or edit MODEL_REL_PATH in the file.")
    st.stop()

# Small status line
st.caption(
    f"Using checkpoint: `{MODEL_REL_PATH.as_posix()}` • "
    f"Classes: {', '.join(loaded['classes'])} • "
    f"Device: {loaded['device']} • "
    f"img_size: {loaded['img_size']}"
)

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

def run_predict_and_show(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)
    with st.spinner("Predicting..."):
        results = predict_image(loaded["model"], loaded["preprocess"], loaded["device"], img, loaded["classes"])
    top_cls = next(iter(results.keys()))
    st.success(f"Prediction: **{top_cls}**")
    st.write("Probabilities:")
    plot_probs(results)
    st.json(results)

with tab_upload:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        run_predict_and_show(up.read())

with tab_camera:
    shot = st.camera_input("Capture from camera")
    if shot is not None:
        run_predict_and_show(shot.getvalue())
