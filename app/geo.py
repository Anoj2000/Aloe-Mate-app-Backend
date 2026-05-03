"""
Geo algorithm — area-based maturity classification.
Thresholds are the original calibrated values — do not change them.

IMPORTANT: Always pass EXIF-corrected dimensions to geo_area_px2().
Use get_exif_aware_dims() in main.py before calling this function.
Never pass raw PIL image.size — it ignores EXIF rotation.
"""

import math

T1 = 1461935
T2 = 4142291


def geo_area_px2(roi_r: float, img_w: int, img_h: int) -> float:
    """
    Compute the plant area in pixels² from the normalised ROI radius.

    roi_r is normalised to min(visual_width, visual_height) by the frontend.
    
    FIX: The original thresholds (T1=1461935, T2=4142291) were calibrated on a 
    specific camera resolution (2296 x 4080) where the short edge was 2296.
    If a user uploads a 900x1600 image (like from WhatsApp) or an 8K image, 
    the raw pixel area will arbitrarily shrink or explode, breaking the thresholds.
    
    To fix this, we ignore the actual image dimensions and project the roi_r 
    onto the calibrated baseline dimension (2296). This makes the Geo Algorithm 
    100% resolution-independent!
    """
    CALIBRATION_MIN_DIM = 2296
    radius_px = roi_r * CALIBRATION_MIN_DIM
    return math.pi * radius_px * radius_px


def geo_class_from_area(area_px2: float) -> str:
    if area_px2 < T1:
        return "IMMATURE"
    if area_px2 < T2:
        return "INTERMEDIATE"
    return "MATURE"


def geo_confidence(area_px2: float) -> float:
    d = min(abs(area_px2 - T1), abs(area_px2 - T2))
    if d > 400000:
        return 0.9
    if d > 200000:
        return 0.75
    return 0.6