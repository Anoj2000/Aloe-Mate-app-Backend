from __future__ import annotations
from typing import Optional
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


class PredictResponse(BaseModel):
    """
    Separate CNN and Geo outputs with no fused decision algorithm.

    is_aloe_vera:     False when CNN detects NON_ALOE (geo_algorithm will be None)
    geo_algorithm:    None when is_aloe_vera is False (Geo never ran)
    classes_match:    True when CNN class == Geo class (both are aloe)
    harvest_required: True when both agree on MATURE (harvest now)
    harvest_message:  Non-null only when harvest_required is True
    """
    is_aloe_vera:     bool
    cnn_model:        ModelOutput
    geo_algorithm:    Optional[GeoOutput] = None
    classes_match:    bool                = False
    harvest_required: bool                = False
    harvest_message:  Optional[str]       = None