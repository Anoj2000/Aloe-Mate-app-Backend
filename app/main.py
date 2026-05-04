import os
import gc

# ── Memory optimization for Render free tier ──────────────────────────────────
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ExifTags
from datetime import date
from dateutil.relativedelta import relativedelta
import io

from .schemas   import PredictResponse, ROI
from .geo       import geo_area_px2, geo_class_from_area, geo_confidence
from .cnn.model import predict_cnn

app = FastAPI(title="Aloe Hybrid Maturity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── EXIF helper ───────────────────────────────────────────────────────────────

def get_exif_aware_dims(pil_image: Image.Image) -> tuple[int, int]:
    """
    Return (width, height) as the image visually appears on screen —
    i.e. after applying EXIF orientation rotation.

    PIL's image.size always returns the raw pixel-buffer dimensions.
    Gallery photos from phones are almost always stored landscape in the
    buffer (e.g. 4032×3024) but tagged with EXIF orientation=6 (rotate 90°
    CW to display as portrait). image.size returns (4032, 3024) — WRONG.
    This function returns (3024, 4032) — the correct visual dimensions,
    matching what the frontend's normalisePhotoDims() also produces.

    EXIF orientation values that require a w/h swap (90° or 270° rotation):
        5 = Mirror horizontal + rotate 90° CW
        6 = Rotate 90° CW          ← most common on iPhones in portrait mode
        7 = Mirror horizontal + rotate 270° CW
        8 = Rotate 270° CW
    All other values (1-4) are no-ops or flips — dimensions stay the same.
    """
    raw_w, raw_h = pil_image.size

    try:
        exif = pil_image._getexif()
        if exif is None:
            return raw_w, raw_h

        orientation_tag = next(
            (tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"),
            None,
        )
        if orientation_tag is None:
            return raw_w, raw_h

        orientation = exif.get(orientation_tag)
        if orientation in (5, 6, 7, 8):
            return raw_h, raw_w
        return raw_w, raw_h

    except Exception:
        return raw_w, raw_h


# ── Health endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Aloe Hybrid Maturity API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Predict endpoint ───────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(
    cnn_image: UploadFile = File(...),
    geo_image: UploadFile = File(...),
    roi_x:     float      = Form(...),
    roi_y:     float      = Form(...),
    roi_r:     float      = Form(...),
):
    roi_obj = ROI(x=roi_x, y=roi_y, r=roi_r)

    # ── Step 1: CNN branch (always runs first) ────────────────────────────────
    cnn_bytes         = await cnn_image.read()
    cnn_cls, cnn_conf = predict_cnn(cnn_bytes)

    # Free CNN image bytes from memory immediately
    del cnn_bytes
    gc.collect()

    # ── Step 2: Validation gate ───────────────────────────────────────────────
    if cnn_cls == "NON_ALOE":
        return {
            "is_aloe_vera":     False,
            "cnn_model": {
                "predicted_class": cnn_cls,
                "confidence":      float(cnn_conf),
            },
            "geo_algorithm":    None,
            "classes_match":    False,
            "harvest_required": False,
            "harvest_message":  None,
        }

    # ── Step 3: GEO branch (only runs when CNN confirms aloe vera) ────────────
    geo_bytes = await geo_image.read()

    geo_pil_raw = Image.open(io.BytesIO(geo_bytes))

    # Free geo bytes after opening image
    del geo_bytes
    gc.collect()

    w, h = get_exif_aware_dims(geo_pil_raw)

    geo_pil = geo_pil_raw.convert("RGB")  # noqa: F841

    area     = geo_area_px2(roi_obj.r, w, h)
    geo_cls  = geo_class_from_area(area)
    geo_conf = geo_confidence(area)

    # Free image from memory after geo calculation
    del geo_pil_raw, geo_pil
    gc.collect()

    print("====== GEO ALGORITHM DEBUG ======")
    print(f"PIL Raw size (w, h) : {w, h}")
    print(f"EXIF Aware (w, h)   : {(w, h)}")
    print(f"Frontend roi_r      : {roi_obj.r}")
    print(f"Calculated Area (px): {area}")
    print(f"Predicted Class     : {geo_cls}")
    print("=================================")

    # ── Step 4: Compare CNN and Geo outputs ───────────────────────────────────
    classes_match    = (cnn_cls == geo_cls)
    harvest_required = classes_match and cnn_cls == "MATURE"
    harvest_message: str | None = (
        "Ready to harvest! Best time: Morning or Evening"
        if harvest_required else None
    )

    return {
        "is_aloe_vera":     True,
        "cnn_model": {
            "predicted_class": cnn_cls,
            "confidence":      float(cnn_conf),
        },
        "geo_algorithm": {
            "detected_area":   float(area),
            "predicted_class": geo_cls,
            "confidence":      float(geo_conf),
        },
        "classes_match":    classes_match,
        "harvest_required": harvest_required,
        "harvest_message":  harvest_message,
    }