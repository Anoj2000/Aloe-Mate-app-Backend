from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ExifTags
import io

from .schemas   import PredictResponse, ROI
from .geo       import geo_area_px2, geo_class_from_area, geo_confidence
from .cnn.model import predict_cnn
from .fusion    import fuse_decision_tree, Out

app = FastAPI(title="Aloe Hybrid Maturity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        exif = pil_image._getexif()          # None for PNG/WEBP or missing EXIF
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
            return raw_h, raw_w              # swap for 90°/270° rotations
        return raw_w, raw_h

    except Exception:
        return raw_w, raw_h                  # safe fallback — never crash


@app.get("/")
def root():
    return {"message": "Aloe Hybrid Maturity API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    cnn_image: UploadFile = File(...),
    geo_image: UploadFile = File(...),
    roi_x:     float      = Form(...),
    roi_y:     float      = Form(...),
    roi_r:     float      = Form(...),
    plant_age: int        = Form(3),
):
    roi_obj = ROI(x=roi_x, y=roi_y, r=roi_r)

    # ── GEO branch ────────────────────────────────────────────────────────────
    geo_bytes = await geo_image.read()

    # Open BEFORE convert("RGB") so EXIF data is still intact for dim extraction.
    geo_pil_raw = Image.open(io.BytesIO(geo_bytes))

    # ✅ FIXED: read visual dimensions from EXIF-aware helper BEFORE converting.
    # The old code did:
    #   geo_pil = Image.open(...).convert("RGB")  ← strips EXIF
    #   w, h    = geo_pil.size                    ← raw buffer dims, wrong for
    #                                                portrait gallery photos
    # For a portrait iPhone photo the buffer is 4032×3024 (landscape) with
    # EXIF orientation=6. The old code passed w=4032, h=3024 to geo_area_px2,
    # giving min_dim=3024 when it should be 3024 — this case happens to be the
    # same, BUT for orientation=8 (rotate 270°) and some Android cameras the
    # swap goes the other way and min_dim resolves to the wrong axis entirely,
    # making the computed area jump by up to (long/short)² ≈ 1.78×, which
    # easily crosses the T1/T2 thresholds and produces the wrong maturity class.
    w, h = get_exif_aware_dims(geo_pil_raw)

    # Now safe to convert for any downstream pixel processing if needed.
    geo_pil = geo_pil_raw.convert("RGB")

    area     = geo_area_px2(roi_obj.r, w, h)
    geo_cls  = geo_class_from_area(area)
    geo_conf = geo_confidence(area)

    # ── CNN branch 
    cnn_bytes         = await cnn_image.read()
    cnn_cls, cnn_conf = predict_cnn(cnn_bytes)

    # ── Fusion 
    result = fuse_decision_tree(
        cnn=Out(cnn_cls, float(cnn_conf)),
        geo=Out(geo_cls, float(geo_conf)),
        geo_area_px2=area,
        age_months=int(plant_age),
    )

    return {
        "cnn_model": {
            "predicted_class": cnn_cls,
            "confidence":      float(cnn_conf),
        },
        "geo_algorithm": {
            "detected_area":   float(area),
            "predicted_class": geo_cls,
            "confidence":      float(geo_conf),
        },
        "ensemble_prediction": {
            "predicted_class": result.cls,
            "confidence":      float(result.conf),
            "decision_reason": result.reason,
            "low_confidence":  result.low_confidence,
        },
    }