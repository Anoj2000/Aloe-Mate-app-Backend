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
    img_w and img_h MUST be EXIF-corrected visual dimensions.
    Use get_exif_aware_dims() in main.py to obtain these values.

    Formula:
        radius_px = roi_r * min(img_w, img_h)
        area      = π * radius_px²
    """
    min_dim   = min(img_w, img_h)
    radius_px = roi_r * min_dim
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