from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .roi import RoiCache


@dataclass(frozen=True)
class FeatureConfig:
    d0_ratio: float = 0.015
    ov0: float = 0.15
    g0_ratio: float = 0.01
    g1_ratio: float = 0.02
    lower_ratio: float = 0.20

    @classmethod
    def from_score_cfg(cls, score_cfg: Mapping[str, Any]) -> "FeatureConfig":
        norms = score_cfg.get("norms", {}) if isinstance(score_cfg, Mapping) else {}
        band = score_cfg.get("band", {}) if isinstance(score_cfg, Mapping) else {}
        return cls(
            d0_ratio=float(norms.get("d0_ratio", 0.015)),
            ov0=float(norms.get("ov0", 0.15)),
            g0_ratio=float(norms.get("g0_ratio", 0.01)),
            g1_ratio=float(norms.get("g1_ratio", 0.02)),
            lower_ratio=float(band.get("lower_ratio", 0.20)),
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp01(v: float) -> float:
    return _clamp(v, 0.0, 1.0)


def _as_feature_cfg(cfg_norms: Mapping[str, Any] | FeatureConfig) -> FeatureConfig:
    if isinstance(cfg_norms, FeatureConfig):
        return cfg_norms
    return FeatureConfig(
        d0_ratio=float(cfg_norms.get("d0_ratio", 0.015)),
        ov0=float(cfg_norms.get("ov0", 0.15)),
        g0_ratio=float(cfg_norms.get("g0_ratio", 0.01)),
        g1_ratio=float(cfg_norms.get("g1_ratio", 0.02)),
        lower_ratio=float(cfg_norms.get("lower_ratio", 0.20)),
    )


def _integral_sum(integral, x1: int, y1: int, x2: int, y2: int) -> int:
    return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def compute_bbox_factors(
    bbox: Sequence[float],
    roi_cache: RoiCache,
    cfg_norms: Mapping[str, Any] | FeatureConfig,
    image_w: int,
    image_h: int,
) -> dict[str, Any]:
    cfg = _as_feature_cfg(cfg_norms)
    if len(bbox) < 4:
        raise ValueError(f"bbox must have at least 4 values, got {bbox}")
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"Invalid image size: {image_w}x{image_h}")

    x1 = float(bbox[0])
    y1 = float(bbox[1])
    x2 = float(bbox[2])
    y2 = float(bbox[3])
    conf = float(bbox[4]) if len(bbox) >= 5 else 1.0

    x1 = _clamp(x1, 0.0, float(image_w - 1))
    x2 = _clamp(x2, 0.0, float(image_w - 1))
    y1 = _clamp(y1, 0.0, float(image_h - 1))
    y2 = _clamp(y2, 0.0, float(image_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(float(image_w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_h - 1), y1 + 1.0)

    bc_x = 0.5 * (x1 + x2)
    bc_y = y2
    bc_xi = int(round(_clamp(bc_x, 0.0, float(image_w - 1))))
    bc_yi = int(round(_clamp(bc_y, 0.0, float(image_h - 1))))
    sd = float(roi_cache.signed_dist[bc_yi, bc_xi])

    d0 = max(1e-6, float(cfg.d0_ratio) * float(max(image_w, image_h)))
    d_in = d0
    d_out = 2.0 * d0
    if sd <= 0.0:
        f_dist = _clamp01((-sd) / d_in)
    else:
        f_dist = _clamp01((d_out - sd) / d_out)

    bbox_h = max(1.0, y2 - y1)
    lower_ratio = _clamp(float(cfg.lower_ratio), 1e-6, 1.0)
    band_h = max(1.0, bbox_h * lower_ratio)
    band_y1 = y2 - band_h

    ix1 = int(math.floor(x1))
    ix2 = int(math.ceil(x2))
    iy1 = int(math.floor(band_y1))
    iy2 = int(math.ceil(y2))

    ix1 = int(_clamp(float(ix1), 0.0, float(image_w)))
    ix2 = int(_clamp(float(ix2), 0.0, float(image_w)))
    iy1 = int(_clamp(float(iy1), 0.0, float(image_h)))
    iy2 = int(_clamp(float(iy2), 0.0, float(image_h)))

    if ix2 <= ix1:
        ix2 = min(image_w, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(image_h, iy1 + 1)

    band_area = float(max(1, (ix2 - ix1) * (iy2 - iy1)))
    roi_sum = float(_integral_sum(roi_cache.integral, ix1, iy1, ix2, iy2))
    ov = roi_sum / band_area
    f_ov = _clamp01(ov / max(1e-6, float(cfg.ov0)))

    x_rep = int(round(_clamp(0.5 * (x1 + x2), 0.0, float(image_w - 1))))
    roi_by = int(roi_cache.bottom_y[x_rep])
    g0 = float(cfg.g0_ratio) * float(image_h)
    g1 = max(1e-6, float(cfg.g1_ratio) * float(image_h))
    if roi_by < 0:
        gap_up = None
        p_gap = 1.0
    else:
        gap_up = float(roi_by) - float(y2)
        p_gap = _clamp01((gap_up - g0) / g1)

    return {
        "f_dist": float(f_dist),
        "f_ov": float(f_ov),
        "p_gap": float(p_gap),
        "sd": float(sd),
        "ov": float(ov),
        "gap_up": gap_up,
        "bc": (float(bc_x), float(bc_y)),
        "bbox_clamped": [float(x1), float(y1), float(x2), float(y2), float(conf)],
    }
