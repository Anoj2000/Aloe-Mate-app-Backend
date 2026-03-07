from __future__ import annotations

import os
import io
import json
import logging
import threading
from typing import List, Tuple

from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH         = os.path.join(os.path.dirname(__file__), "best_aloe_vera_model_4class_fixed.pth")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "class_mapping.json")
IMG_SIZE           = (224, 224)
DEFAULT_CLASSES    = ["IMMATURE", "INTERMEDIATE", "MATURE", "NON_ALOE"]

logger      = logging.getLogger(__name__)
_model      = None
_model_lock = threading.Lock()
_device_str = None


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    global _device_str
    if _device_str is None:
        import torch
        _device_str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device_str}")
    return _device_str


# ── Model loader ──────────────────────────────────────────────────────────────

def _get_model():
    """Thread-safe lazy loader. Rebuilds EfficientNetB0 and loads saved weights."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return None

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as tv_models

            device = _get_device()
            logger.info(f"Loading EfficientNetB0 from {MODEL_PATH}")

            # Rebuild exact architecture used during training
            base        = tv_models.efficientnet_b0(weights=None)
            in_features = base.classifier[1].in_features   # 1280

            # Must match training head exactly — no AdaptiveAvgPool2d or Flatten
            base.classifier = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4),
            )

            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            base.load_state_dict(state_dict)
            base.to(device)
            base.eval()

            _model = base
            logger.info("Model loaded and ready.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return None

    return _model


# ── Class mapping ─────────────────────────────────────────────────────────────

def _get_classes() -> List[str]:
    """Load class names from JSON or fall back to defaults."""
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, "r") as f:
                mapping = json.load(f)

            # Supports both {"0": "immature", ...} and {"immature": 0, ...}
            if all(isinstance(k, str) and k.isdigit() for k in mapping.keys()):
                classes = [
                    str(v).upper()
                    for _, v in sorted(mapping.items(), key=lambda x: int(x[0]))
                ]
            else:
                classes = [
                    str(k).upper()
                    for k, _ in sorted(mapping.items(), key=lambda x: x[1])
                ]

            logger.info(f"Classes loaded: {classes}")
            return classes

        except Exception as e:
            logger.error(f"Error reading class mapping: {e}")

    logger.info(f"Using default classes: {DEFAULT_CLASSES}")
    return DEFAULT_CLASSES


# ── Transform ─────────────────────────────────────────────────────────────────

def _build_transform(img_w: int, img_h: int):
    """
    Build inference transform that matches the training val_transform exactly:
        Resize((224,224)) -> ToTensor() -> Normalize(ImageNet mean/std)

    Smart resize to avoid unnecessary upscaling:
        224x224 input  -> skip resize (already correct, frontend sent exact size)
        > 512px input  -> two-step: Resize(256) -> Resize(224)  [anti-aliasing]
        other          -> single Resize(224)
    """
    import torchvision.transforms as T

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if img_w == 224 and img_h == 224:
        # Frontend already sent 224x224 — just normalise, no spatial change
        return T.Compose([T.ToTensor(), norm])

    if max(img_w, img_h) > 512:
        # High-res photo: two-step downscale reduces aliasing
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            norm,
        ])

    # Medium-res: single resize step
    return T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        norm,
    ])


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_cnn(image_bytes: bytes) -> Tuple[str, float]:
    """
    Predict plant maturity from raw image bytes.
    Returns (class_name, confidence)  e.g. ("MATURE", 0.87)
    """
    import torch

    model = _get_model()
    if model is None:
        return "UNKNOWN", 0.0

    try:
        device       = _get_device()
        img          = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = img.size
        logger.info(f"Input image size: {img_w}x{img_h}")

        transform    = _build_transform(img_w, img_h)
        tensor       = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        probs_np   = probs.cpu().numpy()
        pred_idx   = int(probs_np.argmax())
        confidence = float(probs_np[pred_idx])

        classes = _get_classes()
        label   = classes[pred_idx] if pred_idx < len(classes) else f"UNKNOWN_{pred_idx}"

        raw = {classes[i]: round(float(p), 4) for i, p in enumerate(probs_np)}
        logger.info(f"Prediction: {label} ({confidence:.4f}) | scores: {raw}")

        return label, confidence

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return "ERROR", 0.0


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = _get_model()
    if m:
        print("Model loaded successfully.")
        print(f"Classes: {_get_classes()}")
    else:
        print("Model failed to load.")