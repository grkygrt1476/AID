from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    np = None  # type: ignore[assignment]


@dataclass
class RoiCache:
    roi_id: str
    poly: np.ndarray
    mask: np.ndarray
    signed_dist: np.ndarray
    integral: np.ndarray
    bottom_y: np.ndarray


def _require_roi_deps() -> None:
    missing: list[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if missing:
        raise RuntimeError(
            "Missing required Python dependencies for ROI processing: "
            + ", ".join(missing)
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read ROI json '{path}': {exc}") from exc


def _parse_image_size(obj: dict[str, Any]) -> tuple[int, int]:
    img_size = obj.get("image_size", {})
    if isinstance(img_size, dict):
        w = int(img_size.get("width", 0) or obj.get("img_w", 0) or obj.get("width", 0))
        h = int(img_size.get("height", 0) or obj.get("img_h", 0) or obj.get("height", 0))
        return max(0, w), max(0, h)
    if isinstance(img_size, list) and len(img_size) >= 2:
        return max(0, int(img_size[0])), max(0, int(img_size[1]))
    w = int(obj.get("img_w", 0) or obj.get("width", 0))
    h = int(obj.get("img_h", 0) or obj.get("height", 0))
    return max(0, w), max(0, h)


def _to_float_vertices(raw: Any, key: str, roi_path: Path) -> Optional[list[tuple[float, float]]]:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"Invalid '{key}' in ROI json '{roi_path}'")
    points: list[tuple[float, float]] = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValueError(f"Invalid point #{i} in '{key}' of ROI json '{roi_path}'")
        points.append((float(pt[0]), float(pt[1])))
    return points


def _convert_norm_to_px(vertices_norm: list[tuple[float, float]], iw: int, ih: int, roi_path: Path) -> list[tuple[float, float]]:
    if iw <= 0 or ih <= 0:
        raise ValueError(f"Normalized ROI requires image_size in '{roi_path}'")
    return [(x * float(iw), y * float(ih)) for (x, y) in vertices_norm]


def _extract_vertices_px(obj: dict[str, Any], roi_path: Path) -> list[tuple[float, float]]:
    iw, ih = _parse_image_size(obj)

    vertices_px = _to_float_vertices(obj.get("vertices_px"), "vertices_px", roi_path)
    if vertices_px is None:
        vertices_norm = None
        for key in ("vertices_norm", "vertices_normalized", "points_norm", "polygon_norm"):
            cand = _to_float_vertices(obj.get(key), key, roi_path)
            if cand is not None:
                vertices_norm = cand
                break

        if vertices_norm is not None:
            vertices_px = _convert_norm_to_px(vertices_norm, iw=iw, ih=ih, roi_path=roi_path)
        else:
            for key in ("vertices", "points", "polygon"):
                cand = _to_float_vertices(obj.get(key), key, roi_path)
                if cand is None:
                    continue
                is_norm = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for (x, y) in cand)
                vertices_px = _convert_norm_to_px(cand, iw=iw, ih=ih, roi_path=roi_path) if is_norm else cand
                break

    if vertices_px is None:
        raise ValueError(f"ROI vertices not found in '{roi_path}'")

    labeled_on = str(obj.get("labeled_on", "")).strip().lower()
    disp_scale_used = obj.get("disp_scale_used", None)
    if labeled_on == "disp" and disp_scale_used is not None:
        scale = float(disp_scale_used)
        if scale <= 0:
            raise ValueError(f"Invalid disp_scale_used in '{roi_path}': {disp_scale_used}")
        return [(x / scale, y / scale) for (x, y) in vertices_px]

    return vertices_px


def load_roi_polygon(path: str | Path) -> np.ndarray:
    _require_roi_deps()
    roi_path = Path(path)
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI json not found: '{roi_path}'")
    obj = _read_json(roi_path)
    points_px = _extract_vertices_px(obj, roi_path=roi_path)
    poly = np.asarray([[int(round(x)), int(round(y))] for (x, y) in points_px], dtype=np.int32)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError(f"Invalid polygon in ROI json '{roi_path}'")
    return poly


def build_roi_mask(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    _require_roi_deps()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for ROI mask: {width}x{height}")
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError("ROI polygon must be shaped (N,2) with N>=3")

    poly_i = poly.astype(np.int32, copy=True)
    poly_i[:, 0] = np.clip(poly_i[:, 0], 0, max(0, width - 1))
    poly_i[:, 1] = np.clip(poly_i[:, 1], 0, max(0, height - 1))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_i.reshape((-1, 1, 2))], color=1)
    return mask


def build_signed_distance(mask: np.ndarray) -> np.ndarray:
    _require_roi_deps()
    if mask.ndim != 2:
        raise ValueError("ROI mask must be 2D")
    roi = (mask > 0).astype(np.uint8)
    outside = (roi == 0).astype(np.uint8)
    dt_outside = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    dt_inside = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
    signed = dt_outside.astype(np.float32, copy=False)
    signed[roi == 1] = -dt_inside[roi == 1]
    return signed


def build_integral(mask: np.ndarray) -> np.ndarray:
    _require_roi_deps()
    if mask.ndim != 2:
        raise ValueError("ROI mask must be 2D")
    roi = (mask > 0).astype(np.uint8)
    integral = cv2.integral(roi, sdepth=cv2.CV_32S)
    return integral.astype(np.int32, copy=False)


def build_roi_bottom_y(mask: np.ndarray) -> np.ndarray:
    _require_roi_deps()
    if mask.ndim != 2:
        raise ValueError("ROI mask must be 2D")
    h, w = mask.shape
    y_grid = np.arange(h, dtype=np.int32).reshape(h, 1)
    y_if_inside = np.where(mask > 0, y_grid, -1)
    return np.max(y_if_inside, axis=0).astype(np.int32, copy=False)


def build_roi_cache(roi_id: str, poly: np.ndarray, width: int, height: int) -> RoiCache:
    _require_roi_deps()
    mask = build_roi_mask(poly=poly, width=width, height=height)
    return RoiCache(
        roi_id=str(roi_id),
        poly=poly.astype(np.int32, copy=True),
        mask=mask,
        signed_dist=build_signed_distance(mask),
        integral=build_integral(mask),
        bottom_y=build_roi_bottom_y(mask),
    )
