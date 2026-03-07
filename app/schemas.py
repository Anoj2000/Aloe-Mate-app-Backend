from pydantic import BaseModel, Field


class ROI(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    r: float = Field(..., gt=0.0, le=2.0)


class ModelOutput(BaseModel):
    predicted_class: str
    confidence:      float


class GeoOutput(BaseModel):
    detected_area:   float
    predicted_class: str
    confidence:      float


class EnsembleOutput(BaseModel):
    predicted_class: str
    confidence:      float
    decision_reason: str
    low_confidence:  bool = False   # True when CNN or Geo conf <= 0.65


class PredictResponse(BaseModel):
    cnn_model:           ModelOutput
    geo_algorithm:       GeoOutput
    ensemble_prediction: EnsembleOutput