from __future__ import annotations

import os
import io
import gc
import json
import logging
import threading
from typing import List, Tuple

from PIL import Image, ImageOps

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
        _device_str = "cpu"  # Force CPU on Render free tier — no GPU available
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

            # Must match training head exactly
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

            # Free state_dict from memory after loading
            del state_dict
            gc.collect()

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

def _build_transform():
    """
    Inference transform matching training val_transform exactly:
        Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet mean/std)
    """
    import torchvision.transforms as T

    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        device  = _get_device()
        raw_img = Image.open(io.BytesIO(image_bytes))

        # Apply EXIF rotation before converting to RGB
        img = ImageOps.exif_transpose(raw_img).convert("RGB")

        # Free raw image from memory
        del raw_img
        gc.collect()

        img_w, img_h = img.size
        logger.info(f"Input image size: {img_w}x{img_h}")

        transform = _build_transform()
        tensor    = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # Free PIL image after tensor is built
        del img
        gc.collect()

        # ── Prediction ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.nn.functional.softmax(logits[0], dim=0)

        probs_np = probs.cpu().numpy()

        # Free tensors from memory
        del tensor, logits, probs
        gc.collect()

        classes = _get_classes()

        # ── Two-stage classification ──────────────────────────────────────────
        NON_ALOE_THRESHOLD = 0.50

        try:
            non_aloe_idx = classes.index("NON_ALOE")
        except ValueError:
            non_aloe_idx = -1

        if non_aloe_idx != -1:
            non_aloe_prob = float(probs_np[non_aloe_idx])
            aloe_indices  = [i for i in range(len(classes)) if i != non_aloe_idx]
        else:
            non_aloe_prob = 0.0
            aloe_indices  = list(range(len(classes)))

        aloe_probs = probs_np[aloe_indices]

        # Stage 1: Is it Aloe Vera?
        if non_aloe_prob >= NON_ALOE_THRESHOLD:
            label      = "NON_ALOE"
            confidence = non_aloe_prob
        else:
            # Stage 2: Best maturity class
            best_local = int(probs_np[aloe_indices].argmax())
            label      = classes[aloe_indices[best_local]]
            confidence = float(aloe_probs[best_local])

        raw = {classes[i]: round(float(p), 4) for i, p in enumerate(probs_np)}
        logger.info(
            f"Prediction: {label} ({confidence:.4f}) | "
            f"non_aloe_prob: {non_aloe_prob:.4f} | scores: {raw}"
        )

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