"""
Geo algorithm — area-based maturity classification.
Thresholds and logic are unchanged from the original.
"""

import math

T1 = 1461935
T2 = 4142291


def geo_area_px2(roi_r: float, img_w: int, img_h: int) -> float:
    """
    ROI.r is normalized to min(screenW, screenH).
    radiusPx = roi.r * min(img_w, img_h)
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