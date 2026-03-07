from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
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
    geo_pil   = Image.open(io.BytesIO(geo_bytes)).convert("RGB")
    w, h      = geo_pil.size
    area      = geo_area_px2(roi_obj.r, w, h)
    geo_cls   = geo_class_from_area(area)
    geo_conf  = geo_confidence(area)

    # ── CNN branch ────────────────────────────────────────────────────────────
    cnn_bytes         = await cnn_image.read()
    cnn_cls, cnn_conf = predict_cnn(cnn_bytes)

    # ── Fusion ────────────────────────────────────────────────────────────────
    # Returns one of: IMMATURE | INTERMEDIATE | MATURE | NON_ALOE
    # low_confidence = True when CNN conf <= 0.65 OR Geo conf <= 0.65
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