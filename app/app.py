"""Flask web application for fruit and vegetable quality classification.

Accepts an uploaded image, runs EfficientNet-B0 inference, generates a
Grad-CAM heatmap overlay, and returns the top-3 predictions alongside
the heatmap for display in the browser.

Usage:
    python -m app.app

Requires the trained model weights to exist at:
    outputs/efficientnet_b0/best_model.pth

Run training first if weights are missing:
    python -m src.train --architecture efficientnet_b0 --epochs 10
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, render_template, request
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data_utils import EVAL_TRANSFORMS
from src.model_utils import build_model, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE = "efficientnet_b0"
WEIGHTS_PATH = PROJECT_ROOT / "outputs" / ARCHITECTURE / "best_model.pth"
METRICS_PATH = PROJECT_ROOT / "outputs" / ARCHITECTURE / "metrics.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# --------------------------------------------------------------------------
# Model loading — done once at startup, shared across requests
# --------------------------------------------------------------------------

def _load_model_and_cam():
    """Loads the trained model and sets up Grad-CAM at startup.
    Returns (model, cam, class_names, device).
    """
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            f"Run: python -m src.train --architecture {ARCHITECTURE} --epochs 10"
        )
    with open(METRICS_PATH) as f:
        class_names = json.load(f)["class_names"]

    device = get_device()
    model = build_model(ARCHITECTURE, num_classes=len(class_names)).to(device)
    model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    print(f"Loaded {ARCHITECTURE} ({len(class_names)} classes) on {device}")
    return model, cam, class_names, device


try:
    MODEL, CAM, CLASS_NAMES, DEVICE = _load_model_and_cam()
    MODEL_READY = True
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    MODEL_READY = False
    CLASS_NAMES = []


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------

def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Applies eval transforms and returns a batch tensor on the right device."""
    return EVAL_TRANSFORMS(pil_image).unsqueeze(0).to(DEVICE)


def _run_inference(pil_image: Image.Image) -> tuple[list[dict], str]:
    """Runs inference and Grad-CAM on a PIL image.

    Returns:
        predictions: list of dicts with keys 'class_name', 'confidence', 'rank'
        heatmap_b64: base64-encoded PNG of the Grad-CAM overlay
    """
    resized = pil_image.resize((224, 224)).convert("RGB")
    tensor = _pil_to_tensor(resized)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3_indices = torch.argsort(probs, descending=True)[:3].tolist()
    predictions = [
        {
            "rank": i + 1,
            "class_name": CLASS_NAMES[idx].replace("__", " "),
            "raw_class": CLASS_NAMES[idx],
            "confidence": round(float(probs[idx]) * 100, 1),
        }
        for i, idx in enumerate(top3_indices)
    ]

    # Grad-CAM for the top predicted class
    top_class_index = top3_indices[0]
    targets = [ClassifierOutputTarget(top_class_index)]
    rgb_float = np.array(resized).astype(np.float32) / 255.0
    grayscale_cam = CAM(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Encode heatmap as base64 PNG for inline display
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return predictions, heatmap_b64


def _original_image_b64(pil_image: Image.Image) -> str:
    """Returns a base64-encoded PNG of the uploaded image at 224x224."""
    resized = pil_image.resize((224, 224)).convert("RGB")
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL_READY)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        pil_image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image. Upload a JPG or PNG."}), 400

    try:
        predictions, heatmap_b64 = _run_inference(pil_image)
        original_b64 = _original_image_b64(pil_image)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    return jsonify({
        "predictions": predictions,
        "heatmap": heatmap_b64,
        "original": original_b64,
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": MODEL_READY,
        "architecture": ARCHITECTURE,
        "num_classes": len(CLASS_NAMES),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)