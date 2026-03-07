"""
Fusion decision tree — Geo algorithm is the primary signal.
CNN acts as a validator and non-aloe detector only.

Valid output classes (match training exactly):
    IMMATURE | INTERMEDIATE | MATURE | NON_ALOE

Confidence rule:
    If CNN conf <= 0.65 OR Geo conf <= 0.65, result is still returned
    but FusionResult.low_confidence = True so the frontend can warn the user.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .geo import T1, T2


@dataclass
class Out:
    cls:  str
    conf: float


@dataclass
class FusionResult:
    cls:            str
    conf:           float
    reason:         str
    low_confidence: bool = field(default=False)


def age_prior(age_months: int) -> str:
    if age_months < 2:
        return "IMMATURE"
    if age_months <= 8:
        return "INTERMEDIATE"
    return "MATURE"


def fuse_decision_tree(
    cnn: Out,
    geo: Out,
    geo_area_px2: float,
    age_months: int,
) -> FusionResult:
    """
    Fuse CNN and Geo predictions. Always returns one of the 4 training classes.

    Priority order:
      1. NON_ALOE  — CNN only (Geo has no concept of non-aloe)
      2. Geo far from thresholds — most reliable Geo result (primary signal)
      3. CNN >= 0.92 override
      4. Both agree
      5. Age prior tie-break (Geo wins first)
      6. Fallback — Geo always wins
    """

    # Flag low confidence when either signal is weak
    low_conf = (cnn.conf <= 0.65) or (geo.conf <= 0.65)

    # 1. Non-aloe — only CNN can detect this
    if cnn.cls == "NON_ALOE":
        return FusionResult(
            cls="NON_ALOE",
            conf=cnn.conf,
            reason="CNN detected non-aloe plant texture",
            low_confidence=low_conf,
        )

    # 2. Geo far from both thresholds — most reliable
    margin = 250000
    d = min(abs(geo_area_px2 - T1), abs(geo_area_px2 - T2))
    if d >= margin:
        return FusionResult(
            cls=geo.cls,
            conf=max(cnn.conf, geo.conf),
            reason="Geo area far from thresholds — Geo trusted",
            low_confidence=low_conf,
        )

    # 3. CNN extremely confident — override Geo
    if cnn.conf >= 0.92:
        return FusionResult(
            cls=cnn.cls,
            conf=max(cnn.conf, geo.conf),
            reason="CNN confidence >= 0.92 override",
            low_confidence=low_conf,
        )

    # 4. Both agree
    if cnn.cls == geo.cls:
        return FusionResult(
            cls=geo.cls,
            conf=max(cnn.conf, geo.conf),
            reason="Geo and CNN agree",
            low_confidence=low_conf,
        )

    # 5. Disagreement — age prior tie-break
    prior = age_prior(age_months)
    if prior == geo.cls:
        return FusionResult(
            cls=geo.cls,
            conf=max(cnn.conf, geo.conf),
            reason="Tie-break: age matches Geo",
            low_confidence=low_conf,
        )
    if prior == cnn.cls:
        return FusionResult(
            cls=cnn.cls,
            conf=max(cnn.conf, geo.conf),
            reason="Tie-break: age matches CNN",
            low_confidence=low_conf,
        )

    # 6. Fallback — Geo is the primary signal
    return FusionResult(
        cls=geo.cls,
        conf=geo.conf,
        reason="Fallback: Geo is primary signal",
        low_confidence=low_conf,
    )