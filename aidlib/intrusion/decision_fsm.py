from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
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

from .features import FeatureConfig, compute_bbox_factors
from .roi import RoiCache, build_roi_cache, load_roi_polygon
from .score import ScoreWeights, compute_score


STATE_OUT = "OUT"
STATE_CANDIDATE = "CANDIDATE"
STATE_IN_CONFIRMED = "IN_CONFIRMED"

LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


def _require_decision_deps() -> None:
    missing: list[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if missing:
        raise RuntimeError(
            "Missing required Python dependencies for the 03_03 intrusion decision pass: "
            + ", ".join(missing)
        )


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    try:
        return int(float(text))
    except ValueError:
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_numpy(value: Any) -> np.ndarray | None:
    _require_decision_deps()
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _clamp_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1i = max(0, min(int(width) - 1, int(round(x1))))
    y1i = max(0, min(int(height) - 1, int(round(y1))))
    x2i = max(0, min(int(width) - 1, int(round(x2))))
    y2i = max(0, min(int(height) - 1, int(round(y2))))
    if x2i <= x1i:
        x2i = min(int(width) - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(int(height) - 1, y1i + 1)
    return x1i, y1i, x2i, y2i


def bbox_roi_overlap_ratio(x1: float, y1: float, x2: float, y2: float, roi_mask: np.ndarray) -> float:
    _require_decision_deps()
    h, w = roi_mask.shape[:2]
    x1i, y1i, x2i, y2i = _clamp_xyxy(x1, y1, x2, y2, w, h)
    patch = roi_mask[y1i:y2i, x1i:x2i]
    if patch.size == 0:
        return 0.0
    inter = float(np.count_nonzero(patch))
    area = float(max(1, (x2i - x1i) * (y2i - y1i)))
    return inter / area


def point_in_roi(point_xy: tuple[float, float], roi_mask: np.ndarray) -> bool:
    _require_decision_deps()
    h, w = roi_mask.shape[:2]
    x = int(round(_clamp(point_xy[0], 0.0, float(w - 1))))
    y = int(round(_clamp(point_xy[1], 0.0, float(h - 1))))
    return bool(roi_mask[y, x] > 0)


def bbox_center_xyxy(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_roi_min_distance_px(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    roi_signed_dist: np.ndarray,
) -> float:
    _require_decision_deps()
    h, w = roi_signed_dist.shape[:2]
    x1i, y1i, x2i, y2i = _clamp_xyxy(x1, y1, x2, y2, w, h)
    patch = roi_signed_dist[y1i:y2i, x1i:x2i]
    if patch.size == 0:
        return float("inf")
    min_signed = float(np.min(patch))
    if min_signed <= 0.0:
        return 0.0
    return float(min_signed)


def nearest_point_on_segment(
    point_xy: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> tuple[float, float]:
    px, py = map(float, point_xy)
    ax, ay = map(float, seg_a)
    bx, by = map(float, seg_b)
    ab_x = bx - ax
    ab_y = by - ay
    denom = (ab_x * ab_x) + (ab_y * ab_y)
    if denom <= 1e-6:
        return (ax, ay)
    t = ((px - ax) * ab_x + (py - ay) * ab_y) / denom
    t = _clamp(t, 0.0, 1.0)
    return (ax + (ab_x * t), ay + (ab_y * t))


def nearest_point_on_roi_poly(point_xy: tuple[float, float], roi_poly: np.ndarray) -> tuple[float, float]:
    _require_decision_deps()
    if roi_poly.ndim != 2 or roi_poly.shape[0] < 3 or roi_poly.shape[1] != 2:
        return point_xy
    best_point = (float(roi_poly[0, 0]), float(roi_poly[0, 1]))
    best_dist_sq = float("inf")
    for idx in range(int(roi_poly.shape[0])):
        ax, ay = roi_poly[idx]
        bx, by = roi_poly[(idx + 1) % int(roi_poly.shape[0])]
        candidate = nearest_point_on_segment(
            point_xy=point_xy,
            seg_a=(float(ax), float(ay)),
            seg_b=(float(bx), float(by)),
        )
        dx = float(candidate[0]) - float(point_xy[0])
        dy = float(candidate[1]) - float(point_xy[1])
        dist_sq = (dx * dx) + (dy * dy)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = candidate
    return best_point


@dataclass(frozen=True)
class SidecarRow:
    frame_num: int
    source_id: int
    track_id: int
    mode: str
    proxy_active: bool
    proxy_age: int
    event: str
    stop_reason: str
    handoff_reason: str
    bbox_xyxy: list[float]
    patch_xyxy: list[float]
    patch_source: str
    pose_anchor_source: str
    tracked_points: int
    flow_dx: float
    flow_dy: float
    flow_mag: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "SidecarRow":
        proxy_left = _to_float(row.get("proxy_left"))
        proxy_top = _to_float(row.get("proxy_top"))
        proxy_width = _to_float(row.get("proxy_width"))
        proxy_height = _to_float(row.get("proxy_height"))
        patch_left = _to_float(row.get("patch_left"))
        patch_top = _to_float(row.get("patch_top"))
        patch_width = _to_float(row.get("patch_width"))
        patch_height = _to_float(row.get("patch_height"))
        return cls(
            frame_num=_to_int(row.get("frame_num")),
            source_id=_to_int(row.get("source_id")),
            track_id=_to_int(row.get("track_id")),
            mode=str(row.get("mode", "")).strip(),
            proxy_active=_to_bool(row.get("proxy_active")),
            proxy_age=_to_int(row.get("proxy_age")),
            event=str(row.get("event", "")).strip(),
            stop_reason=str(row.get("stop_reason", "")).strip(),
            handoff_reason=str(row.get("handoff_reason", "")).strip(),
            bbox_xyxy=[
                proxy_left,
                proxy_top,
                proxy_left + max(0.0, proxy_width),
                proxy_top + max(0.0, proxy_height),
            ],
            patch_xyxy=[
                patch_left,
                patch_top,
                patch_left + max(0.0, patch_width),
                patch_top + max(0.0, patch_height),
            ],
            patch_source=str(row.get("patch_source", "")).strip(),
            pose_anchor_source=str(row.get("pose_anchor_source", "")).strip(),
            tracked_points=_to_int(row.get("tracked_points")),
            flow_dx=_to_float(row.get("flow_dx")),
            flow_dy=_to_float(row.get("flow_dy")),
            flow_mag=_to_float(row.get("flow_mag")),
        )

    @property
    def has_valid_bbox(self) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) > 1.0 and (y2 - y1) > 1.0


def load_sidecar_rows(sidecar_csv: str | Path) -> tuple[dict[int, dict[int, SidecarRow]], dict[str, Any]]:
    sidecar_path = Path(sidecar_csv)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Tracking sidecar not found: '{sidecar_path}'")

    rows_by_frame: dict[int, dict[int, SidecarRow]] = {}
    frame_min: Optional[int] = None
    frame_max: Optional[int] = None
    modes: set[str] = set()
    row_count = 0

    with sidecar_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = SidecarRow.from_csv_row(raw)
            rows_by_frame.setdefault(row.frame_num, {})[row.track_id] = row
            frame_min = row.frame_num if frame_min is None else min(frame_min, row.frame_num)
            frame_max = row.frame_num if frame_max is None else max(frame_max, row.frame_num)
            modes.add(row.mode)
            row_count += 1

    summary = {
        "sidecar_csv": str(sidecar_path),
        "row_count": row_count,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "modes": sorted(modes),
    }
    return rows_by_frame, summary


def load_roi_cache_from_json(roi_json: str | Path, width: int, height: int) -> tuple[RoiCache, dict[str, Any]]:
    _require_decision_deps()
    roi_path = Path(roi_json)
    obj = json.loads(roi_path.read_text(encoding="utf-8"))
    roi_id = str(obj.get("roi_id", roi_path.stem)).strip() or roi_path.stem
    poly = load_roi_polygon(roi_path)
    cache = build_roi_cache(roi_id=roi_id, poly=poly, width=width, height=height)
    return cache, {
        "roi_json": str(roi_path),
        "roi_id": roi_id,
        "roi_type": obj.get("roi_type", ""),
        "image_size": obj.get("image_size", {}),
    }


@dataclass(frozen=True)
class DecisionParams:
    candidate_enter_n: int = 2
    confirm_enter_n: int = 1
    exit_n: int = 5
    grace_frames: int = 30
    candidate_iou_or_overlap_thr: float = 0.05
    confirm_requires_ankle: bool = True
    candidate_score_thr: float = 0.35
    proxy_start_max_age_frames: int = 3
    cand_distance_enter_px: float = 42.0
    cand_distance_sustain_px: float = 60.0
    cand_motion_toward_score_enter: float = 0.20
    cand_motion_toward_score_sustain: float = 0.05
    cand_motion_min_speed_px: float = 2.0
    cand_motion_max_frame_gap: int = 2
    klt_confirm_recent_real_max_frames: int = 12
    klt_confirm_max_loss_frames: int = 6
    klt_continuity_max_proxy_age_frames: int = 10
    klt_continuity_bonus_proxy_age_frames: int = 20
    klt_continuity_bonus_recent_real_max_frames: int = 20
    klt_continuity_bonus_min_tracked_points: int = 10
    klt_confirm_min_tracked_points: int = 6
    klt_confirm_min_flow_mag: float = 0.75
    klt_confirm_boundary_max_distance_px: float = 72.0
    klt_candidate_recent_context_frames: int = 12
    klt_progression_confirm_distance_px: float = 24.0
    klt_progression_confirm_toward_score: float = 0.55
    klt_progression_confirm_loss_gap_frames: int = 1
    klt_progression_recent_window_frames: int = 12
    klt_progression_min_observations: int = 3
    klt_progression_min_distance_improvement_px: float = 4.0
    klt_head_confirm_support_window_frames: int = 3
    klt_head_confirm_min_support_frames: int = 2
    klt_candidate_accum_window_frames: int = 10
    klt_candidate_accum_min_frames: int = 6
    klt_candidate_accum_min_head_support_frames: int = 4
    klt_candidate_accum_min_motion_frames: int = 4
    klt_candidate_accum_min_progression_frames: int = 4
    klt_candidate_accum_min_motion_toward_score: float = 0.60
    klt_candidate_accum_max_roi_distance_px: float = 4.0
    klt_candidate_accum_ready_hold_frames: int = 4


@dataclass(frozen=True)
class PoseProbeSettings:
    model_path: str = ""
    input_size: int = 640
    conf: float = 0.25
    keypoint_conf: float = 0.35
    pad_x_ratio: float = 0.08
    pad_top_ratio: float = 0.05
    pad_bottom_ratio: float = 0.15


@dataclass(frozen=True)
class PoseProbeResult:
    attempted: bool
    status: str
    ankle_in_roi: bool
    ankles: list[dict[str, Any]]

    @classmethod
    def skipped(cls, status: str) -> "PoseProbeResult":
        return cls(attempted=False, status=status, ankle_in_roi=False, ankles=[])


class PoseAnkleProbe:
    def __init__(self, roi_cache: RoiCache, settings: PoseProbeSettings):
        self.roi_cache = roi_cache
        self.settings = settings
        self._model = None
        self._load_status: Optional[str] = None

    @property
    def model_status(self) -> str:
        self._ensure_model()
        return self._load_status or "ready"

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_status is not None:
            return

        model_path = str(self.settings.model_path).strip()
        if not model_path:
            self._load_status = "pose_model_not_configured"
            return

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            self._load_status = f"pose_model_missing:{model_path_obj}"
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - import failure depends on runtime
            self._load_status = f"ultralytics_import_failed:{exc.__class__.__name__}"
            return

        try:
            self._model = YOLO(str(model_path_obj))
        except Exception as exc:  # pragma: no cover - model loading depends on runtime
            self._load_status = f"pose_model_load_failed:{exc.__class__.__name__}"
            return

        self._load_status = "ready"

    def _select_pose_candidate(self, result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        _require_decision_deps()
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None, None

        keypoints_xy = _to_numpy(getattr(keypoints, "xy", None))
        if keypoints_xy is None or keypoints_xy.ndim != 3 or keypoints_xy.shape[0] == 0:
            return None, None

        keypoints_conf = _to_numpy(getattr(keypoints, "conf", None))
        if keypoints_conf is None:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
        elif keypoints_conf.ndim == 1:
            if keypoints_xy.shape[0] == 1 and keypoints_conf.shape[0] == keypoints_xy.shape[1]:
                keypoints_conf = keypoints_conf.reshape(1, -1)
            else:
                keypoints_conf = np.broadcast_to(keypoints_conf.reshape(1, -1), keypoints_xy.shape[:2]).copy()

        best_idx = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            box_conf = _to_numpy(boxes.conf)
            if box_conf is not None and len(box_conf) >= keypoints_xy.shape[0]:
                best_idx = int(np.argmax(box_conf.reshape(-1)[: keypoints_xy.shape[0]]))
        elif keypoints_conf.ndim == 2:
            conf_score = np.nanmean(keypoints_conf, axis=1)
            best_idx = int(np.argmax(conf_score))

        return keypoints_xy[best_idx].astype(np.float32), keypoints_conf[best_idx].astype(np.float32).reshape(-1)

    def _extract_valid_ankle(self, keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, index: int) -> tuple[np.ndarray | None, float]:
        _require_decision_deps()
        if index >= len(keypoints_xy) or index >= len(keypoints_conf):
            return None, 0.0
        conf = float(keypoints_conf[index])
        point = keypoints_xy[index]
        if conf < float(self.settings.keypoint_conf) or not np.all(np.isfinite(point)):
            return None, conf
        return point.astype(np.float32), conf

    def probe(self, frame: np.ndarray, bbox_xyxy: list[float]) -> PoseProbeResult:
        _require_decision_deps()
        self._ensure_model()
        if self._model is None:
            return PoseProbeResult.skipped(self.model_status)

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        crop_x1 = x1 - bbox_w * float(self.settings.pad_x_ratio)
        crop_x2 = x2 + bbox_w * float(self.settings.pad_x_ratio)
        crop_y1 = y1 - bbox_h * float(self.settings.pad_top_ratio)
        crop_y2 = y2 + bbox_h * float(self.settings.pad_bottom_ratio)
        x1i, y1i, x2i, y2i = _clamp_xyxy(crop_x1, crop_y1, crop_x2, crop_y2, frame_w, frame_h)
        crop = frame[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            return PoseProbeResult(attempted=True, status="pose_crop_empty", ankle_in_roi=False, ankles=[])

        try:
            results = self._model.predict(
                source=crop,
                imgsz=int(self.settings.input_size),
                conf=float(self.settings.conf),
                verbose=False,
                stream=False,
            )
        except Exception as exc:  # pragma: no cover - runtime inference failure depends on env
            return PoseProbeResult(
                attempted=True,
                status=f"pose_infer_failed:{exc.__class__.__name__}",
                ankle_in_roi=False,
                ankles=[],
            )

        result = results[0] if results else None
        if result is None:
            return PoseProbeResult(attempted=True, status="pose_missing", ankle_in_roi=False, ankles=[])

        keypoints_xy, keypoints_conf = self._select_pose_candidate(result)
        if keypoints_xy is None or keypoints_conf is None:
            return PoseProbeResult(attempted=True, status="pose_missing", ankle_in_roi=False, ankles=[])

        keypoints_xy[:, 0] += float(x1i)
        keypoints_xy[:, 1] += float(y1i)

        ankles: list[dict[str, Any]] = []
        ankle_inside = False
        for key_name, key_index in (("left_ankle", LEFT_ANKLE_IDX), ("right_ankle", RIGHT_ANKLE_IDX)):
            point, conf = self._extract_valid_ankle(keypoints_xy, keypoints_conf, key_index)
            if point is None:
                continue
            inside = point_in_roi((float(point[0]), float(point[1])), self.roi_cache.mask)
            ankles.append(
                {
                    "name": key_name,
                    "x": round(float(point[0]), 2),
                    "y": round(float(point[1]), 2),
                    "conf": round(float(conf), 4),
                    "inside_roi": bool(inside),
                }
            )
            ankle_inside = ankle_inside or inside

        if ankle_inside:
            status = "ankle_in_roi"
        elif ankles:
            status = "ankle_outside_roi"
        else:
            status = "ankle_missing"

        return PoseProbeResult(
            attempted=True,
            status=status,
            ankle_in_roi=bool(ankle_inside),
            ankles=ankles,
        )


@dataclass(frozen=True)
class CandidateEvidence:
    bbox_xyxy: list[float]
    source_mode: str
    candidate_geom: bool
    real_candidate: bool
    proxy_candidate: bool
    proxy_start_allowed: bool
    bbox_overlap: float
    lower_band_overlap: float
    bottom_center: tuple[float, float]
    bottom_center_in_roi: bool
    signed_distance_px: float
    roi_min_distance_px: float
    motion_toward_score: float
    motion_speed_px: float
    early_candidate_pretrigger: bool
    score: float
    score_factors: dict[str, Any]
    reasons: list[str]


@dataclass(frozen=True)
class KltBoundaryConfirmSeed:
    frame_num: int
    anchor_source: str
    anchor_kind: str
    anchor_xy: tuple[float, float]
    anchor_inside_roi: bool
    boundary_near_or_inside: bool
    recent_real_seed: bool
    boundary_context: bool
    motion_toward_ok: bool
    continuity_seeded: bool
    reliability_ok: bool
    tracked_points: int
    flow_mag: float
    proxy_mode: str
    proxy_age: int
    roi_min_distance_px: float
    patch_roi_min_distance_px: float
    signed_distance_px: float
    motion_toward_score: float
    motion_speed_px: float


@dataclass(frozen=True)
class KltCandidateSeed:
    frame_num: int
    anchor_source: str
    anchor_kind: str
    anchor_xy: tuple[float, float]
    anchor_inside_roi: bool
    boundary_near_or_inside: bool
    recent_real_seed: bool
    inward_progression_ok: bool
    continuity_seeded: bool
    reliability_ok: bool
    tracked_points: int
    flow_mag: float
    proxy_mode: str
    proxy_age: int
    patch_roi_min_distance_px: float
    motion_toward_score: float
    motion_speed_px: float


@dataclass(frozen=True)
class ProjectedBottomGeometry:
    frame_num: int
    anchor_source: str
    dx_over_patch_h: float
    dy_over_patch_h: float
    patch_height: float


def klt_progress_distance_px(seed: KltBoundaryConfirmSeed) -> float:
    if bool(seed.anchor_inside_roi):
        return 0.0
    patch_dist = float(seed.patch_roi_min_distance_px)
    if patch_dist > 0.0:
        return patch_dist
    return float(seed.roi_min_distance_px)


def summarize_recent_klt_progression(
    *,
    frame_num: int,
    history: list[KltBoundaryConfirmSeed],
    params: DecisionParams,
) -> dict[str, Any]:
    window_frames = max(1, int(params.klt_progression_recent_window_frames))
    min_speed = float(params.cand_motion_min_speed_px)
    toward_thr = float(params.klt_progression_confirm_toward_score)
    distance_cap = max(
        float(params.klt_confirm_boundary_max_distance_px),
        float(params.cand_distance_sustain_px),
    )
    recent_history = [
        item
        for item in history
        if (int(frame_num) - int(item.frame_num)) <= window_frames
    ]

    strong_items: list[tuple[KltBoundaryConfirmSeed, float]] = []
    for item in recent_history:
        progress_dist = float(klt_progress_distance_px(item))
        strong_ok = bool(
            item.recent_real_seed
            and item.boundary_context
            and item.boundary_near_or_inside
            and item.reliability_ok
            and item.motion_speed_px >= min_speed
            and item.motion_toward_score >= toward_thr
            and progress_dist <= distance_cap
        )
        if strong_ok:
            strong_items.append((item, progress_dist))

    non_inside_items = [(item, dist) for (item, dist) in strong_items if not bool(item.anchor_inside_roi)]
    strong_frames = [int(item.frame_num) for (item, _) in strong_items]
    non_inside_frames = [int(item.frame_num) for (item, _) in non_inside_items]
    progression_start_distance = float(non_inside_items[0][1]) if non_inside_items else 0.0
    progression_best_distance = min((dist for (_, dist) in non_inside_items), default=0.0)
    progression_distance_delta = max(0.0, progression_start_distance - progression_best_distance)
    progression_ok = bool(
        len(non_inside_items) >= max(1, int(params.klt_progression_min_observations))
        and progression_distance_delta >= float(params.klt_progression_min_distance_improvement_px)
    )
    return {
        "window_frames": int(window_frames),
        "strong_count": int(len(strong_items)),
        "strong_frames": strong_frames,
        "non_inside_count": int(len(non_inside_items)),
        "non_inside_frames": non_inside_frames,
        "start_distance_px": round(float(progression_start_distance), 3),
        "best_distance_px": round(float(progression_best_distance), 3),
        "distance_delta_px": round(float(progression_distance_delta), 3),
        "progression_ok": bool(progression_ok),
    }


def summarize_recent_klt_head_support(
    *,
    frame_num: int,
    history: list[KltBoundaryConfirmSeed],
    params: DecisionParams,
) -> dict[str, Any]:
    window_frames = max(1, int(params.klt_head_confirm_support_window_frames))
    recent_history = [
        item
        for item in history
        if (int(frame_num) - int(item.frame_num)) <= window_frames
    ]
    support_items = [
        item
        for item in recent_history
        if (
            item.recent_real_seed
            and item.boundary_context
            and item.motion_toward_ok
            and item.continuity_seeded
            and item.reliability_ok
        )
    ]
    support_frames = [int(item.frame_num) for item in support_items]
    support_ok = bool(len(support_items) >= max(1, int(params.klt_head_confirm_min_support_frames)))
    active_continuity = bool(
        support_ok
        and any(item.proxy_mode in {"proxy", "frozen_hold", "real"} for item in support_items)
    )
    return {
        "window_frames": int(window_frames),
        "support_count": int(len(support_items)),
        "support_frames": support_frames,
        "support_ok": bool(support_ok),
        "active_continuity": bool(active_continuity),
    }


def summarize_recent_klt_inside_support(
    *,
    frame_num: int,
    history: list[KltBoundaryConfirmSeed],
    params: DecisionParams,
) -> dict[str, Any]:
    window_frames = max(
        1,
        int(params.klt_head_confirm_support_window_frames),
        int(params.klt_progression_confirm_loss_gap_frames) + 3,
    )
    recent_history = [
        item
        for item in history
        if (int(frame_num) - int(item.frame_num)) <= window_frames
    ]
    support_items = [
        item
        for item in recent_history
        if (
            item.recent_real_seed
            and item.boundary_context
            and item.motion_toward_ok
            and item.continuity_seeded
            and item.reliability_ok
            and item.anchor_inside_roi
        )
    ]
    support_frames = [int(item.frame_num) for item in support_items]
    support_ok = bool(len(support_items) >= max(2, int(params.klt_head_confirm_min_support_frames)))
    return {
        "window_frames": int(window_frames),
        "support_count": int(len(support_items)),
        "support_frames": support_frames,
        "support_ok": bool(support_ok),
    }


def summarize_recent_candidate_accumulation(
    *,
    frame_num: int,
    state: "TrackDecisionState",
    params: DecisionParams,
) -> dict[str, Any]:
    window_frames = max(1, int(params.klt_candidate_accum_window_frames))
    candidate_frames = [
        int(item)
        for item in state.candidate_history_frames
        if (int(frame_num) - int(item)) <= window_frames
    ]
    if state.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED} and int(frame_num) not in candidate_frames:
        candidate_frames.append(int(frame_num))
    candidate_frames = sorted(set(candidate_frames))

    recent_history = [
        item
        for item in state.klt_boundary_history
        if (int(frame_num) - int(item.frame_num)) <= window_frames
    ]
    head_support_items = [
        item
        for item in recent_history
        if (
            item.recent_real_seed
            and item.boundary_context
            and item.motion_toward_ok
            and item.continuity_seeded
            and item.reliability_ok
        )
    ]
    motion_items = [
        item
        for item in head_support_items
        if (
            item.motion_speed_px >= float(params.cand_motion_min_speed_px)
            and item.motion_toward_score >= float(params.klt_candidate_accum_min_motion_toward_score)
        )
    ]
    progression_items = [
        item
        for item in recent_history
        if (
            item.recent_real_seed
            and item.boundary_context
            and item.boundary_near_or_inside
            and item.reliability_ok
            and item.motion_speed_px >= float(params.cand_motion_min_speed_px)
            and item.motion_toward_score >= float(params.klt_progression_confirm_toward_score)
        )
    ]
    best_roi_distance_px = min((float(item.roi_min_distance_px) for item in recent_history), default=float("inf"))
    best_patch_distance_px = min((float(item.patch_roi_min_distance_px) for item in recent_history), default=float("inf"))
    motion_mean = (
        float(sum(float(item.motion_toward_score) for item in motion_items)) / float(len(motion_items))
        if motion_items
        else 0.0
    )
    return {
        "window_frames": int(window_frames),
        "candidate_count": int(len(candidate_frames)),
        "candidate_frames": candidate_frames,
        "head_support_count": int(len(head_support_items)),
        "head_support_frames": [int(item.frame_num) for item in head_support_items],
        "motion_count": int(len(motion_items)),
        "motion_frames": [int(item.frame_num) for item in motion_items],
        "motion_toward_mean": round(float(motion_mean), 4),
        "progression_count": int(len(progression_items)),
        "progression_frames": [int(item.frame_num) for item in progression_items],
        "best_roi_distance_px": round(float(best_roi_distance_px), 3) if best_roi_distance_px != float("inf") else float("inf"),
        "best_patch_distance_px": round(float(best_patch_distance_px), 3) if best_patch_distance_px != float("inf") else float("inf"),
    }


def confirm_path_to_basis(confirm_path: str) -> str:
    path = str(confirm_path).strip()
    if path == "ankle_in_roi":
        return "ankle"
    if path == "klt_ankle_proxy_confirm":
        return "ankle(proxy)"
    if path in {
        "lower_body_overlap_confirm",
        "klt_bottom_center_proxy_confirm",
        "klt_current_lowerband_confirm",
        "klt_projected_bottom_center_confirm",
        "display_continuity_confirm",
        "klt_display_continuity_confirm",
    }:
        return "lower-body"
    return "none"


def bbox_has_area_xyxy(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return (x2 - x1) > 1.0 and (y2 - y1) > 1.0


def project_point_between_bboxes(
    point_xy: tuple[float, float],
    src_bbox_xyxy: list[float] | tuple[float, float, float, float],
    dst_bbox_xyxy: list[float] | tuple[float, float, float, float],
) -> tuple[float, float] | None:
    sx1, sy1, sx2, sy2 = map(float, src_bbox_xyxy)
    dx1, dy1, dx2, dy2 = map(float, dst_bbox_xyxy)
    src_w = max(1e-6, sx2 - sx1)
    src_h = max(1e-6, sy2 - sy1)
    dst_w = max(1e-6, dx2 - dx1)
    dst_h = max(1e-6, dy2 - dy1)
    px, py = map(float, point_xy)
    rel_x = _clamp((px - sx1) / src_w, 0.0, 1.0)
    rel_y = _clamp((py - sy1) / src_h, 0.0, 1.0)
    return (dx1 + rel_x * dst_w, dy1 + rel_y * dst_h)


def classify_klt_anchor_source(raw_source: str) -> tuple[str, str]:
    text = str(raw_source or "").strip().lower()
    if not text:
        return "", ""

    for token in ("head_like_keypoint", "nose", "eye_center", "ear_center", "head_mean"):
        if token in text:
            return "head_like_keypoint", token
    for token in ("shoulder_fallback", "shoulder_center", "left_shoulder", "right_shoulder"):
        if token in text:
            return "shoulder_fallback", token
    return "", ""


def klt_proxy_age_limit(
    *,
    frame_num: int,
    row: SidecarRow | None,
    evidence: CandidateEvidence | None,
    state: "TrackDecisionState",
    params: DecisionParams,
    recent_candidate_context_ok: bool,
) -> tuple[int, bool]:
    base_age = max(1, int(params.klt_continuity_max_proxy_age_frames))
    bonus_age = max(base_age, int(params.klt_continuity_bonus_proxy_age_frames))
    if (
        row is None
        or row.mode != "proxy"
        or not row.proxy_active
        or int(row.proxy_age) <= base_age
        or bonus_age <= base_age
    ):
        return base_age, False
    if int(row.proxy_age) > bonus_age:
        return base_age, False

    anchor_source, _ = classify_klt_anchor_source(str(row.pose_anchor_source or row.patch_source or "").strip())
    upper_anchor_valid = bool(anchor_source) and bbox_has_area_xyxy(row.patch_xyxy)
    tracked_points_ok = bool(
        row.tracked_points >= max(
            int(params.klt_confirm_min_tracked_points),
            int(params.klt_continuity_bonus_min_tracked_points),
        )
    )
    recent_real_ok = bool(
        state.last_real_frame is not None
        and (int(frame_num) - int(state.last_real_frame))
        <= max(base_age, int(params.klt_continuity_bonus_recent_real_max_frames))
    )
    boundary_context_ok = bool(
        recent_candidate_context_ok
        or (
            evidence is not None
            and (
                bool(evidence.bottom_center_in_roi)
                or float(evidence.lower_band_overlap) > 0.0
                or float(evidence.bbox_overlap) >= float(params.candidate_iou_or_overlap_thr)
                or float(evidence.roi_min_distance_px) <= float(params.cand_distance_sustain_px)
            )
        )
    )
    motion_compatible_ok = bool(
        evidence is not None
        and (
            (
                float(evidence.motion_speed_px) >= float(params.cand_motion_min_speed_px)
                and float(evidence.motion_toward_score) >= float(params.cand_motion_toward_score_sustain)
            )
            or bool(evidence.bottom_center_in_roi)
            or float(evidence.lower_band_overlap) >= 0.50
            or float(evidence.bbox_overlap) >= float(params.candidate_iou_or_overlap_thr)
        )
    )
    bonus_ok = bool(
        upper_anchor_valid
        and tracked_points_ok
        and recent_real_ok
        and boundary_context_ok
        and motion_compatible_ok
    )
    return (bonus_age if bonus_ok else base_age), bonus_ok


def row_has_loss_hint(row: SidecarRow | None) -> bool:
    if row is None:
        return True
    loss_text = " ".join((row.event, row.stop_reason, row.handoff_reason)).strip().lower()
    if not loss_text:
        return False
    return any(token in loss_text for token in ("lost", "handoff", "stop", "drop", "occl", "miss"))


def build_klt_boundary_confirm_seed(
    *,
    frame_num: int,
    row: SidecarRow | None,
    evidence: CandidateEvidence | None,
    state: "TrackDecisionState",
    roi_cache: RoiCache,
    params: DecisionParams,
) -> KltBoundaryConfirmSeed | None:
    if row is None or evidence is None:
        return None
    if not bbox_has_area_xyxy(row.patch_xyxy):
        return None

    anchor_source_raw = str(row.pose_anchor_source or row.patch_source or "").strip()
    anchor_source, anchor_kind = classify_klt_anchor_source(anchor_source_raw)
    if not anchor_source:
        return None

    anchor_xy = bbox_center_xyxy(row.patch_xyxy)
    anchor_inside_roi = point_in_roi(anchor_xy, roi_cache.mask)
    patch_x1, patch_y1, patch_x2, patch_y2 = row.patch_xyxy
    patch_roi_min_distance_px = bbox_roi_min_distance_px(
        float(patch_x1),
        float(patch_y1),
        float(patch_x2),
        float(patch_y2),
        roi_cache.signed_dist,
    )

    recent_real_seed = bool(
        state.last_real_frame is not None
        and (int(frame_num) - int(state.last_real_frame)) <= max(1, int(params.klt_confirm_recent_real_max_frames))
    )
    recent_candidate_context_ok = bool(
        state.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}
        or (
            state.last_candidate_context_frame is not None
            and (int(frame_num) - int(state.last_candidate_context_frame))
            <= max(1, int(params.klt_candidate_recent_context_frames))
        )
    )
    boundary_distance_thr = max(
        float(params.klt_confirm_boundary_max_distance_px),
        float(params.cand_distance_sustain_px),
    )
    boundary_near_or_inside = bool(
        anchor_inside_roi
        or patch_roi_min_distance_px <= boundary_distance_thr
        or evidence.roi_min_distance_px <= boundary_distance_thr
        or abs(float(evidence.signed_distance_px)) <= boundary_distance_thr
    )
    boundary_context = bool(
        boundary_near_or_inside
        and (evidence.candidate_geom or evidence.early_candidate_pretrigger or recent_candidate_context_ok)
    )
    motion_toward_ok = bool(
        evidence.motion_speed_px >= float(params.cand_motion_min_speed_px)
        and evidence.motion_toward_score >= float(params.cand_motion_toward_score_enter)
    )

    continuity_seeded = bool(
        row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
        and (
            row.proxy_active
            or row.mode in {"proxy", "frozen_hold"}
            or bool(str(row.patch_source or "").strip())
        )
    )
    real_upper_seed_ok = bool(
        row.mode == "real"
        and row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
        and bool(str(row.patch_source or "").strip())
    )
    proxy_age_limit, _ = klt_proxy_age_limit(
        frame_num=int(frame_num),
        row=row,
        evidence=evidence,
        state=state,
        params=params,
        recent_candidate_context_ok=bool(recent_candidate_context_ok),
    )
    proxy_age_ok = bool(
        (not row.proxy_active)
        or row.proxy_age <= int(proxy_age_limit)
    )
    flow_quality_ok = bool(
        row.flow_mag >= float(params.klt_confirm_min_flow_mag)
        or (abs(float(row.flow_dx)) + abs(float(row.flow_dy))) >= float(params.klt_confirm_min_flow_mag)
    )
    reliability_ok = bool(continuity_seeded and proxy_age_ok and (flow_quality_ok or real_upper_seed_ok))
    if not (recent_real_seed and boundary_context and motion_toward_ok and reliability_ok):
        return None

    return KltBoundaryConfirmSeed(
        frame_num=int(frame_num),
        anchor_source=anchor_source,
        anchor_kind=anchor_kind,
        anchor_xy=(float(anchor_xy[0]), float(anchor_xy[1])),
        anchor_inside_roi=bool(anchor_inside_roi),
        boundary_near_or_inside=bool(boundary_near_or_inside),
        recent_real_seed=bool(recent_real_seed),
        boundary_context=bool(boundary_context),
        motion_toward_ok=bool(motion_toward_ok),
        continuity_seeded=bool(continuity_seeded),
        reliability_ok=bool(reliability_ok),
        tracked_points=int(row.tracked_points),
        flow_mag=float(row.flow_mag),
        proxy_mode=str(row.mode),
        proxy_age=int(row.proxy_age),
        roi_min_distance_px=float(evidence.roi_min_distance_px),
        patch_roi_min_distance_px=float(patch_roi_min_distance_px),
        signed_distance_px=float(evidence.signed_distance_px),
        motion_toward_score=float(evidence.motion_toward_score),
        motion_speed_px=float(evidence.motion_speed_px),
    )


def build_klt_candidate_seed(
    *,
    frame_num: int,
    row: SidecarRow | None,
    evidence: CandidateEvidence | None,
    state: "TrackDecisionState",
    roi_cache: RoiCache,
    params: DecisionParams,
) -> KltCandidateSeed | None:
    if row is None or evidence is None:
        return None
    if not bbox_has_area_xyxy(row.patch_xyxy):
        return None

    anchor_source_raw = str(row.pose_anchor_source or row.patch_source or "").strip()
    anchor_source, anchor_kind = classify_klt_anchor_source(anchor_source_raw)
    if not anchor_source:
        return None

    anchor_xy = bbox_center_xyxy(row.patch_xyxy)
    anchor_inside_roi = point_in_roi(anchor_xy, roi_cache.mask)
    patch_x1, patch_y1, patch_x2, patch_y2 = row.patch_xyxy
    patch_roi_min_distance_px = bbox_roi_min_distance_px(
        float(patch_x1),
        float(patch_y1),
        float(patch_x2),
        float(patch_y2),
        roi_cache.signed_dist,
    )
    boundary_distance_thr = max(
        float(params.klt_confirm_boundary_max_distance_px),
        float(params.cand_distance_sustain_px),
    )
    boundary_near_or_inside = bool(anchor_inside_roi or patch_roi_min_distance_px <= boundary_distance_thr)
    recent_real_seed = bool(
        state.last_real_frame is not None
        and (int(frame_num) - int(state.last_real_frame)) <= max(1, int(params.klt_confirm_recent_real_max_frames))
    )
    recent_candidate_context_ok = bool(
        state.last_candidate_context_frame is not None
        and (int(frame_num) - int(state.last_candidate_context_frame))
        <= max(1, int(params.klt_candidate_recent_context_frames))
    )
    motion_toward_ok = bool(
        evidence.motion_speed_px >= float(params.cand_motion_min_speed_px)
        and evidence.motion_toward_score >= float(params.cand_motion_toward_score_enter)
    )
    inward_progression_ok = bool(anchor_inside_roi or motion_toward_ok)
    continuity_seeded = bool(
        row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
        and (
            row.proxy_active
            or row.mode in {"proxy", "frozen_hold"}
            or bool(str(row.patch_source or "").strip())
        )
    )
    real_upper_seed_ok = bool(
        row.mode == "real"
        and row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
        and bool(str(row.patch_source or "").strip())
    )
    proxy_age_limit, _ = klt_proxy_age_limit(
        frame_num=int(frame_num),
        row=row,
        evidence=evidence,
        state=state,
        params=params,
        recent_candidate_context_ok=bool(recent_candidate_context_ok),
    )
    proxy_age_ok = bool(
        (not row.proxy_active)
        or row.proxy_age <= int(proxy_age_limit)
    )
    flow_quality_ok = bool(
        row.flow_mag >= float(params.klt_confirm_min_flow_mag)
        or (abs(float(row.flow_dx)) + abs(float(row.flow_dy))) >= float(params.klt_confirm_min_flow_mag)
    )
    reliability_ok = bool(continuity_seeded and proxy_age_ok and (flow_quality_ok or real_upper_seed_ok))
    if not (recent_real_seed and boundary_near_or_inside and inward_progression_ok and reliability_ok):
        return None

    return KltCandidateSeed(
        frame_num=int(frame_num),
        anchor_source=anchor_source,
        anchor_kind=anchor_kind,
        anchor_xy=(float(anchor_xy[0]), float(anchor_xy[1])),
        anchor_inside_roi=bool(anchor_inside_roi),
        boundary_near_or_inside=bool(boundary_near_or_inside),
        recent_real_seed=bool(recent_real_seed),
        inward_progression_ok=bool(inward_progression_ok),
        continuity_seeded=bool(continuity_seeded),
        reliability_ok=bool(reliability_ok),
        tracked_points=int(row.tracked_points),
        flow_mag=float(row.flow_mag),
        proxy_mode=str(row.mode),
        proxy_age=int(row.proxy_age),
        patch_roi_min_distance_px=float(patch_roi_min_distance_px),
        motion_toward_score=float(evidence.motion_toward_score),
        motion_speed_px=float(evidence.motion_speed_px),
    )


def evaluate_klt_candidate_signal(
    *,
    frame_num: int,
    row: SidecarRow | None,
    state: "TrackDecisionState",
    params: DecisionParams,
) -> dict[str, Any]:
    # display_continuity rows carry no live KLT tracking information
    # (tracked_points=0, flow=0, frozen bbox) and must not sustain the
    # candidate signal — treat them as "no row present".
    if row is not None and row.mode == "display_continuity":
        row = None
    seed = state.klt_candidate_seed
    frames_since_seed = None if seed is None else max(0, int(frame_num) - int(seed.frame_num))
    short_gap_ok = bool(
        seed is not None
        and frames_since_seed is not None
        and int(frames_since_seed) <= max(1, int(params.klt_confirm_max_loss_frames))
    )
    recent_candidate_context_ok = bool(
        state.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}
        or (
            state.last_candidate_context_frame is not None
            and (int(frame_num) - int(state.last_candidate_context_frame))
            <= max(1, int(params.klt_candidate_recent_context_frames))
        )
    )
    sparse_gap_now = bool(
        row is None
        or row_has_loss_hint(row)
        or (
            row is not None
            and row.mode != "real"
            and row.tracked_points < max(1, int(params.klt_confirm_min_tracked_points))
        )
    )
    open_ok = bool(
        seed is not None
        and state.state == STATE_OUT
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_near_or_inside)
        and bool(seed.inward_progression_ok)
        and bool(seed.reliability_ok)
        and (
            row is not None
            or (bool(recent_candidate_context_ok) and bool(short_gap_ok) and bool(sparse_gap_now))
        )
    )
    sustain_ok = bool(
        seed is not None
        and bool(recent_candidate_context_ok)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_near_or_inside)
        and bool(seed.inward_progression_ok)
        and bool(seed.reliability_ok)
        and (
            row is not None
            or (bool(short_gap_ok) and bool(sparse_gap_now))
        )
    )
    signal = bool(open_ok or sustain_ok)
    reason = ""
    if open_ok:
        reason = "klt_candidate_open"
    elif sustain_ok:
        reason = "klt_candidate_sustain"
    return {
        "signal": bool(signal),
        "open_ok": bool(open_ok),
        "sustain_ok": bool(sustain_ok),
        "reason": reason,
        "anchor_source": seed.anchor_source if seed is not None else "",
        "anchor_kind": seed.anchor_kind if seed is not None else "",
        "anchor_inside_roi": bool(seed.anchor_inside_roi) if seed is not None else False,
        "boundary_near_or_inside": bool(seed.boundary_near_or_inside) if seed is not None else False,
        "anchor_xy": [round(float(seed.anchor_xy[0]), 2), round(float(seed.anchor_xy[1]), 2)] if seed is not None else [],
        "seed_frame_num": int(seed.frame_num) if seed is not None else None,
        "frames_since_seed": int(frames_since_seed) if frames_since_seed is not None else None,
        "recent_real_seed": bool(seed.recent_real_seed) if seed is not None else False,
        "recent_candidate_context_ok": bool(recent_candidate_context_ok),
        "inward_progression_ok": bool(seed.inward_progression_ok) if seed is not None else False,
        "continuity_seeded": bool(seed.continuity_seeded) if seed is not None else False,
        "reliability_ok": bool(seed.reliability_ok) if seed is not None else False,
        "tracked_points": int(seed.tracked_points) if seed is not None else 0,
        "flow_mag": round(float(seed.flow_mag), 4) if seed is not None else 0.0,
        "proxy_mode": seed.proxy_mode if seed is not None else "",
        "proxy_age": int(seed.proxy_age) if seed is not None else 0,
        "patch_roi_min_distance_px": round(float(seed.patch_roi_min_distance_px), 3) if seed is not None else 0.0,
        "motion_toward_score": round(float(seed.motion_toward_score), 4) if seed is not None else 0.0,
        "motion_speed_px": round(float(seed.motion_speed_px), 3) if seed is not None else 0.0,
        "sparse_gap_now": bool(sparse_gap_now),
    }


def evaluate_klt_boundary_confirm(
    *,
    frame_num: int,
    row: SidecarRow | None,
    evidence: CandidateEvidence | None,
    state: "TrackDecisionState",
    params: DecisionParams,
    pose_result: PoseProbeResult,
    roi_cache: RoiCache,
) -> dict[str, Any]:
    seed = state.klt_boundary_seed
    frames_since_seed = None if seed is None else max(0, int(frame_num) - int(seed.frame_num))
    loss_gap_frames = None if state.last_seen_frame is None else max(0, int(frame_num) - int(state.last_seen_frame))
    progression_history = summarize_recent_klt_progression(
        frame_num=frame_num,
        history=state.klt_boundary_history,
        params=params,
    )
    head_support_history = summarize_recent_klt_head_support(
        frame_num=frame_num,
        history=state.klt_boundary_history,
        params=params,
    )
    inside_support_history = summarize_recent_klt_inside_support(
        frame_num=frame_num,
        history=state.klt_boundary_history,
        params=params,
    )
    candidate_accum_history = summarize_recent_candidate_accumulation(
        frame_num=frame_num,
        state=state,
        params=params,
    )
    recent_candidate_context_ok = bool(
        state.state == STATE_CANDIDATE
        or (
            state.last_candidate_context_frame is not None
            and (int(frame_num) - int(state.last_candidate_context_frame))
            <= max(1, int(params.klt_candidate_recent_context_frames))
        )
    )
    recent_accumulated_ready = bool(
        state.last_accumulated_head_ready_frame is not None
        and (int(frame_num) - int(state.last_accumulated_head_ready_frame))
        <= max(1, int(params.klt_candidate_accum_ready_hold_frames))
    )
    current_loss_like = bool(
        row is None
        or row_has_loss_hint(row)
        or (
            row is not None
            and row.mode != "real"
            and (
                row.mode == "frozen_hold"
                or row.tracked_points < max(1, int(params.klt_confirm_min_tracked_points))
            )
        )
    )
    current_flow_quality_ok = bool(
        row is not None
        and (
            row.flow_mag >= float(params.klt_confirm_min_flow_mag)
            or (abs(float(row.flow_dx)) + abs(float(row.flow_dy))) >= float(params.klt_confirm_min_flow_mag)
        )
    )
    proxy_age_limit, proxy_age_bonus_active = klt_proxy_age_limit(
        frame_num=int(frame_num),
        row=row,
        evidence=evidence,
        state=state,
        params=params,
        recent_candidate_context_ok=bool(recent_candidate_context_ok),
    )
    current_reliable_proxy = bool(
        row is not None
        and row.mode in {"proxy", "frozen_hold"}
        and row.proxy_active
        and row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
        and current_flow_quality_ok
        and 1 <= int(row.proxy_age) <= int(proxy_age_limit)
    )
    current_real_continuity = bool(
        row is not None
        and row.mode == "real"
        and row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
        and bbox_has_area_xyxy(row.patch_xyxy)
        and bool(classify_klt_anchor_source(str(row.pose_anchor_source or row.patch_source or "").strip())[0])
    )
    current_reliable_continuity = bool(
        current_reliable_proxy
        or current_real_continuity
    )
    ankle_visible_now = bool(isinstance(pose_result.ankles, list) and len(pose_result.ankles) > 0)
    current_anchor_source, current_anchor_token = classify_klt_anchor_source(
        str(row.pose_anchor_source or row.patch_source or "").strip() if row is not None else ""
    )
    current_head_anchor_present = bool(
        row is not None
        and current_anchor_source == "head_like_keypoint"
        and bbox_has_area_xyxy(row.patch_xyxy)
    )
    current_upper_anchor_present = bool(
        row is not None
        and current_anchor_source in {"head_like_keypoint", "shoulder_fallback"}
        and bbox_has_area_xyxy(row.patch_xyxy)
    )
    current_lower_band_overlap = float(evidence.lower_band_overlap) if evidence is not None else 0.0
    current_lowerband_recent_real_ok = bool(
        current_real_continuity
        or (
            state.last_real_frame is not None
            and (int(frame_num) - int(state.last_real_frame))
            <= max(1, int(params.klt_confirm_recent_real_max_frames))
        )
    )
    current_lowerband_boundary_ok = bool(
        recent_candidate_context_ok
        or (
            evidence is not None
            and (
                bool(evidence.bottom_center_in_roi)
                or float(evidence.lower_band_overlap) >= 0.50
                or float(evidence.bbox_overlap) >= float(params.candidate_iou_or_overlap_thr)
                or float(evidence.roi_min_distance_px) <= float(params.cand_distance_sustain_px)
            )
        )
        or (seed is not None and bool(seed.boundary_context))
    )
    current_lowerband_reliability_ok = bool(
        (seed is not None and bool(seed.reliability_ok))
        or current_real_continuity
        or current_reliable_proxy
    )
    current_bbox_xyxy = None
    if row is not None and row.has_valid_bbox:
        current_bbox_xyxy = list(row.bbox_xyxy)
    elif state.last_bbox_xyxy is not None and bbox_has_area_xyxy(state.last_bbox_xyxy):
        current_bbox_xyxy = list(state.last_bbox_xyxy)
    ankle_proxy_seed_recent = bool(
        state.last_real_ankle_frame is not None
        and (int(frame_num) - int(state.last_real_ankle_frame)) <= max(1, int(params.klt_confirm_recent_real_max_frames))
    )
    ankle_proxy_available = False
    ankle_proxy_points: list[dict[str, Any]] = []
    ankle_proxy_in_roi = False
    if (
        current_bbox_xyxy is not None
        and state.last_real_ankle_bbox_xyxy is not None
        and bbox_has_area_xyxy(state.last_real_ankle_bbox_xyxy)
        and ankle_proxy_seed_recent
        and state.last_real_ankles_xy
    ):
        for ankle_name, ankle_xy in sorted(state.last_real_ankles_xy.items()):
            projected_xy = project_point_between_bboxes(
                ankle_xy,
                state.last_real_ankle_bbox_xyxy,
                current_bbox_xyxy,
            )
            if projected_xy is None:
                continue
            inside_roi = point_in_roi(projected_xy, roi_cache.mask)
            ankle_proxy_points.append(
                {
                    "name": ankle_name,
                    "x": round(float(projected_xy[0]), 2),
                    "y": round(float(projected_xy[1]), 2),
                    "inside_roi": bool(inside_roi),
                }
            )
            ankle_proxy_in_roi = ankle_proxy_in_roi or bool(inside_roi)
        ankle_proxy_available = bool(ankle_proxy_points)
    bottom_center_proxy_xy = (
        [round(float(evidence.bottom_center[0]), 2), round(float(evidence.bottom_center[1]), 2)]
        if evidence is not None
        else []
    )
    bottom_center_proxy_in_roi = bool(evidence.bottom_center_in_roi) if evidence is not None else False
    current_candidate_support_frame = bool(
        current_reliable_continuity
        and current_lowerband_recent_real_ok
        and current_lowerband_boundary_ok
        and current_lowerband_reliability_ok
        and current_upper_anchor_present
        and (
            current_lower_band_overlap > 0.0
            or bool(bottom_center_proxy_in_roi)
            or (
                evidence is not None
                and (
                    bool(evidence.bottom_center_in_roi)
                    or float(evidence.bbox_overlap) >= float(params.candidate_iou_or_overlap_thr)
                )
            )
        )
    )
    support_window_frames = max(1, int(params.klt_candidate_recent_context_frames))
    recent_candidate_support_frames = [
        int(item)
        for item in state.klt_candidate_support_frames
        if (int(frame_num) - int(item)) <= support_window_frames
    ]
    current_candidate_support_count = int(
        len(recent_candidate_support_frames)
        + (0 if int(frame_num) in recent_candidate_support_frames else (1 if current_candidate_support_frame else 0))
    )
    current_candidate_context_ok = bool(
        recent_candidate_context_ok
        or current_candidate_support_count >= max(1, int(params.candidate_enter_n))
    )
    projected_geometry = state.projected_bottom_geometry
    projected_bottom_geometry_available = bool(
        projected_geometry is not None
        and current_upper_anchor_present
        and row is not None
        and bbox_has_area_xyxy(row.patch_xyxy)
    )
    projected_bottom_geometry_age_frames = (
        None
        if projected_geometry is None
        else max(0, int(frame_num) - int(projected_geometry.frame_num))
    )
    projected_bottom_geometry_fresh = bool(
        projected_bottom_geometry_age_frames is not None
        and int(projected_bottom_geometry_age_frames) <= max(1, int(params.klt_confirm_recent_real_max_frames))
    )
    projected_bottom_geometry_anchor_ok = bool(
        projected_geometry is not None
        and current_anchor_source
        and str(projected_geometry.anchor_source) == str(current_anchor_source)
    )
    projected_bottom_center_xy: list[float] = []
    projected_bottom_center_in_roi = False
    if (
        projected_geometry is not None
        and projected_bottom_geometry_available
        and projected_bottom_geometry_fresh
        and projected_bottom_geometry_anchor_ok
        and row is not None
    ):
        current_anchor_xy = bbox_center_xyxy(row.patch_xyxy)
        current_patch_h = max(1.0, float(row.patch_xyxy[3]) - float(row.patch_xyxy[1]))
        projected_xy = (
            float(current_anchor_xy[0]) + (float(projected_geometry.dx_over_patch_h) * current_patch_h),
            float(current_anchor_xy[1]) + (float(projected_geometry.dy_over_patch_h) * current_patch_h),
        )
        projected_bottom_center_xy = [round(float(projected_xy[0]), 2), round(float(projected_xy[1]), 2)]
        projected_bottom_center_in_roi = bool(point_in_roi(projected_xy, roi_cache.mask))
    short_loss_ok = bool(
        seed is not None
        and frames_since_seed is not None
        and 1 <= int(frames_since_seed) <= max(1, int(params.klt_confirm_max_loss_frames))
    )
    seed_window_ok = bool(
        seed is not None
        and frames_since_seed is not None
        and int(frames_since_seed) <= max(1, int(params.klt_confirm_recent_real_max_frames))
    )
    progression_distance_ok = bool(
        seed is not None
        and (
            bool(seed.anchor_inside_roi)
            or float(seed.patch_roi_min_distance_px) <= float(params.klt_progression_confirm_distance_px)
            or float(seed.roi_min_distance_px) <= float(params.klt_progression_confirm_distance_px)
        )
    )
    progression_motion_strong = bool(
        seed is not None
        and float(seed.motion_speed_px) >= float(params.cand_motion_min_speed_px)
        and float(seed.motion_toward_score) >= float(params.klt_progression_confirm_toward_score)
    )
    immediate_loss_ok = bool(
        current_loss_like
        and loss_gap_frames is not None
        and int(loss_gap_frames) <= max(1, int(params.klt_progression_confirm_loss_gap_frames))
    )
    non_inside_stricter_guard = bool(seed is not None and not bool(seed.anchor_inside_roi))
    repeated_progression_ok = bool(progression_history["progression_ok"])
    head_support_ok = bool(head_support_history["support_ok"])
    head_active_continuity_ok = bool(head_support_history["active_continuity"])
    inside_support_ok = bool(inside_support_history["support_ok"])
    candidate_accumulated_ok = bool(
        int(candidate_accum_history["candidate_count"]) >= max(1, int(params.klt_candidate_accum_min_frames))
        and int(candidate_accum_history["head_support_count"])
        >= max(1, int(params.klt_candidate_accum_min_head_support_frames))
        and int(candidate_accum_history["motion_count"]) >= max(1, int(params.klt_candidate_accum_min_motion_frames))
        and int(candidate_accum_history["progression_count"])
        >= max(1, int(params.klt_candidate_accum_min_progression_frames))
        and float(candidate_accum_history["motion_toward_mean"])
        >= float(params.klt_candidate_accum_min_motion_toward_score)
    )
    accumulated_entry_depth_ok = bool(
        float(candidate_accum_history["best_roi_distance_px"])
        <= float(params.klt_candidate_accum_max_roi_distance_px)
    )
    head_only_raw_detection_blocked = bool(
        seed is not None
        and bool(seed.reliability_ok)
        and bool(seed.motion_toward_ok)
        and not bool(head_support_ok and head_active_continuity_ok)
    )
    shallow_head_blocked = bool(
        seed is not None
        and not pose_result.ankle_in_roi
        and not bool(seed.anchor_inside_roi)
        and not bool(inside_support_ok)
        and (
            not bool(candidate_accumulated_ok or recent_accumulated_ready)
            or not bool(accumulated_entry_depth_ok)
            or not bool((head_support_ok and head_active_continuity_ok) or recent_accumulated_ready)
        )
    )
    strict_entry_progression_ok = bool(
        seed is not None
        and bool(recent_candidate_context_ok)
        and not pose_result.ankle_in_roi
        and bool(seed.anchor_inside_roi)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.motion_toward_ok)
        and bool(seed.reliability_ok)
        and bool(head_support_ok)
        and bool(head_active_continuity_ok)
        and bool(inside_support_ok)
        and bool(current_reliable_continuity)
        and bool(seed_window_ok)
    )
    strict_loss_after_entry_ok = bool(
        seed is not None
        and bool(recent_candidate_context_ok)
        and not pose_result.ankle_in_roi
        and bool(seed.anchor_inside_roi)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.motion_toward_ok)
        and bool(seed.reliability_ok)
        and bool(head_support_ok)
        and bool(head_active_continuity_ok)
        and bool(short_loss_ok)
        and bool(current_loss_like)
    )
    deep_inside_confirm_ok = bool(strict_entry_progression_ok or strict_loss_after_entry_ok)
    klt_head_lowerband_confirm_ok = bool(
        seed is not None
        and not pose_result.ankle_in_roi
        and not ankle_visible_now
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.motion_toward_ok)
        and bool(seed.continuity_seeded)
        and bool(seed.reliability_ok)
        and bool(current_reliable_continuity)
        and bool(current_head_anchor_present)
        and current_lower_band_overlap >= 0.50
    )
    accumulated_candidate_confirm_ok = bool(
        seed is not None
        and bool(recent_candidate_context_ok)
        and not pose_result.ankle_in_roi
        and not bool(seed.anchor_inside_roi)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.boundary_near_or_inside)
        and bool(seed.motion_toward_ok)
        and bool(seed.reliability_ok)
        and bool((head_support_ok and head_active_continuity_ok) or recent_accumulated_ready)
        and bool(candidate_accumulated_ok or recent_accumulated_ready)
        and bool(accumulated_entry_depth_ok)
        and bool(current_reliable_continuity or immediate_loss_ok)
        and not bool(shallow_head_blocked)
    )
    boundary_progression_then_lost_ok = bool(
        seed is not None
        and bool(recent_candidate_context_ok)
        and not pose_result.ankle_in_roi
        and bool(non_inside_stricter_guard)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.boundary_near_or_inside)
        and bool(seed.motion_toward_ok)
        and bool(seed.reliability_ok)
        and bool(head_support_ok)
        and bool(head_active_continuity_ok)
        and bool(seed_window_ok)
        and bool(repeated_progression_ok)
        and bool(progression_distance_ok)
        and bool(progression_motion_strong)
        and bool(immediate_loss_ok)
        and not bool(candidate_accumulated_ok and accumulated_entry_depth_ok)
        and not bool(shallow_head_blocked)
    )
    klt_ankle_proxy_confirm_ok = bool(
        seed is not None
        and not pose_result.ankle_in_roi
        and not ankle_visible_now
        and bool(current_candidate_context_ok)
        and bool(seed.recent_real_seed)
        and bool(seed.continuity_seeded)
        and bool(seed.reliability_ok)
        and bool(current_reliable_continuity)
        and bool(ankle_proxy_available)
        and bool(ankle_proxy_seed_recent)
        and bool(ankle_proxy_in_roi)
    )
    klt_bottom_center_proxy_confirm_ok = bool(
        seed is not None
        and not pose_result.ankle_in_roi
        and not ankle_visible_now
        and bool(current_candidate_context_ok)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.motion_toward_ok)
        and bool(seed.continuity_seeded)
        and bool(seed.reliability_ok)
        and bool(current_reliable_continuity)
        and not bool(ankle_proxy_available)
        and bool(bottom_center_proxy_in_roi)
    )
    klt_current_lowerband_confirm_ok = bool(
        not pose_result.ankle_in_roi
        and not ankle_visible_now
        and bool(current_candidate_context_ok)
        and bool(current_lowerband_recent_real_ok)
        and bool(current_lowerband_boundary_ok)
        and bool(current_lowerband_reliability_ok)
        and bool(current_reliable_continuity)
        and bool(current_upper_anchor_present)
        and current_lower_band_overlap >= 0.50
    )
    klt_projected_bottom_center_confirm_ok = bool(
        seed is not None
        and not pose_result.ankle_in_roi
        and not ankle_visible_now
        and bool(current_candidate_context_ok)
        and bool(seed.recent_real_seed)
        and bool(seed.boundary_context)
        and bool(seed.motion_toward_ok)
        and bool(seed.continuity_seeded)
        and bool(seed.reliability_ok)
        and bool(current_reliable_continuity)
        and not bool(ankle_proxy_available and ankle_proxy_in_roi)
        and not bool(bottom_center_proxy_in_roi)
        and bool(current_upper_anchor_present)
        and bool(projected_bottom_geometry_available)
        and bool(projected_bottom_geometry_fresh)
        and bool(projected_bottom_geometry_anchor_ok)
        and bool(projected_bottom_center_in_roi)
    )
    # Display continuity confirm: relaxed path for frozen bbox rows emitted
    # after the KLT proxy age limit.  Uses the candidate-level overlap threshold
    # instead of the strict lower-body 0.50 overlap, allowing boundary hard-cases
    # where the upper body crosses the ROI but the lower body has not yet entered.
    klt_display_continuity_confirm_ok = bool(
        row is not None
        and row.mode == "display_continuity"
        and row.proxy_active
        and not pose_result.ankle_in_roi
        and not ankle_visible_now
        and bool(current_candidate_context_ok)
        and evidence is not None
        and float(evidence.bbox_overlap) >= 0.30
        and state.last_real_frame is not None
        and (int(frame_num) - int(state.last_real_frame))
        <= max(1, int(params.klt_confirm_recent_real_max_frames) * 6)
    )
    head_based_confirm_disabled = bool(
        deep_inside_confirm_ok
        or klt_head_lowerband_confirm_ok
        or accumulated_candidate_confirm_ok
        or boundary_progression_then_lost_ok
    )
    fired = bool(klt_ankle_proxy_confirm_ok)
    pattern = ""
    reason = ""
    variant_reason = ""
    if klt_ankle_proxy_confirm_ok:
        reason = "klt_ankle_proxy_confirm"
        variant_reason = "klt_ankle_proxy_confirm"
        pattern = "ankle_proxy_in_roi"
    elif klt_bottom_center_proxy_confirm_ok:
        reason = "lower_body_overlap_confirm"
        variant_reason = "klt_bottom_center_proxy_confirm"
        pattern = "bottom_center_proxy_in_roi"
    elif klt_current_lowerband_confirm_ok:
        reason = "lower_body_overlap_confirm"
        variant_reason = "klt_current_lowerband_confirm"
        pattern = "current_lower_band_overlap"
    elif klt_projected_bottom_center_confirm_ok:
        reason = "lower_body_overlap_confirm"
        variant_reason = "klt_projected_bottom_center_confirm"
        pattern = "projected_bottom_center_in_roi"
    elif klt_display_continuity_confirm_ok:
        reason = "display_continuity_confirm"
        variant_reason = "klt_display_continuity_confirm"
        pattern = "display_continuity_bbox_overlap"
    return {
        "fired": bool(fired),
        "reason": reason,
        "variant_reason": variant_reason,
        "pattern": pattern,
        "anchor_source": seed.anchor_source if seed is not None else "",
        "anchor_kind": seed.anchor_kind if seed is not None else "",
        "anchor_inside_roi": bool(seed.anchor_inside_roi) if seed is not None else False,
        "boundary_near_or_inside": bool(seed.boundary_near_or_inside) if seed is not None else False,
        "anchor_xy": [round(float(seed.anchor_xy[0]), 2), round(float(seed.anchor_xy[1]), 2)] if seed is not None else [],
        "seed_frame_num": int(seed.frame_num) if seed is not None else None,
        "frames_since_seed": int(frames_since_seed) if frames_since_seed is not None else None,
        "loss_gap_frames": int(loss_gap_frames) if loss_gap_frames is not None else None,
        "recent_real_seed": bool(seed.recent_real_seed) if seed is not None else False,
        "recent_candidate_context_ok": bool(recent_candidate_context_ok),
        "boundary_context": bool(seed.boundary_context) if seed is not None else False,
        "motion_toward_ok": bool(seed.motion_toward_ok) if seed is not None else False,
        "continuity_seeded": bool(seed.continuity_seeded) if seed is not None else False,
        "reliability_ok": bool(seed.reliability_ok) if seed is not None else False,
        "tracked_points": int(seed.tracked_points) if seed is not None else 0,
        "flow_mag": round(float(seed.flow_mag), 4) if seed is not None else 0.0,
        "proxy_mode": seed.proxy_mode if seed is not None else "",
        "proxy_age": int(seed.proxy_age) if seed is not None else 0,
        "roi_min_distance_px": round(float(seed.roi_min_distance_px), 3) if seed is not None else 0.0,
        "patch_roi_min_distance_px": round(float(seed.patch_roi_min_distance_px), 3) if seed is not None else 0.0,
        "signed_distance_px": round(float(seed.signed_distance_px), 3) if seed is not None else 0.0,
        "motion_toward_score": round(float(seed.motion_toward_score), 4) if seed is not None else 0.0,
        "motion_speed_px": round(float(seed.motion_speed_px), 3) if seed is not None else 0.0,
        "entry_progression_ok": bool(strict_entry_progression_ok),
        "loss_after_entry": bool(strict_loss_after_entry_ok),
        "deep_inside_confirm_ok": bool(deep_inside_confirm_ok),
        "head_lowerband_confirm": bool(klt_head_lowerband_confirm_ok),
        "accumulated_candidate_confirm": bool(accumulated_candidate_confirm_ok),
        "boundary_progression_then_lost": bool(boundary_progression_then_lost_ok),
        "head_based_confirm_disabled": bool(head_based_confirm_disabled),
        "disabled_head_deep_inside_would_fire": bool(deep_inside_confirm_ok),
        "disabled_head_lowerband_would_fire": bool(klt_head_lowerband_confirm_ok),
        "disabled_head_accumulated_would_fire": bool(accumulated_candidate_confirm_ok),
        "disabled_head_progression_would_fire": bool(boundary_progression_then_lost_ok),
        "ankle_proxy_confirm": bool(klt_ankle_proxy_confirm_ok),
        "bottom_center_proxy_confirm": bool(klt_bottom_center_proxy_confirm_ok),
        "ankle_proxy_available": bool(ankle_proxy_available),
        "ankle_proxy_seed_recent": bool(ankle_proxy_seed_recent),
        "ankle_proxy_in_roi": bool(ankle_proxy_in_roi),
        "ankle_proxy_points": ankle_proxy_points,
        "bottom_center_proxy_xy": bottom_center_proxy_xy,
        "bottom_center_proxy_in_roi": bool(bottom_center_proxy_in_roi),
        "current_lowerband_confirm": bool(klt_current_lowerband_confirm_ok),
        "current_lowerband_recent_real_ok": bool(current_lowerband_recent_real_ok),
        "current_lowerband_boundary_ok": bool(current_lowerband_boundary_ok),
        "current_lowerband_reliability_ok": bool(current_lowerband_reliability_ok),
        "current_candidate_support_frame": bool(current_candidate_support_frame),
        "current_candidate_support_count": int(current_candidate_support_count),
        "current_candidate_support_frames": list(recent_candidate_support_frames),
        "current_candidate_context_ok": bool(current_candidate_context_ok),
        "current_lowerband_blocked_no_continuity": bool(
            not current_reliable_continuity and current_upper_anchor_present and not pose_result.ankle_in_roi
        ),
        "current_lowerband_blocked_low_overlap": bool(
            current_reliable_continuity and current_upper_anchor_present and not (current_lower_band_overlap >= 0.50)
        ),
        "current_lowerband_blocked_stronger_path": bool(
            klt_current_lowerband_confirm_ok
            and (
                bool(pose_result.ankle_in_roi)
                or bool(klt_ankle_proxy_confirm_ok)
                or bool(klt_bottom_center_proxy_confirm_ok)
            )
        ),
        "display_continuity_confirm": bool(klt_display_continuity_confirm_ok),
        "projected_bottom_center_confirm": bool(klt_projected_bottom_center_confirm_ok),
        "projected_bottom_geometry_available": bool(projected_bottom_geometry_available),
        "projected_bottom_geometry_source_frame_num": int(projected_geometry.frame_num) if projected_geometry is not None else None,
        "projected_bottom_geometry_age_frames": int(projected_bottom_geometry_age_frames) if projected_bottom_geometry_age_frames is not None else None,
        "projected_bottom_geometry_anchor_source": str(projected_geometry.anchor_source) if projected_geometry is not None else "",
        "projected_bottom_center_xy": projected_bottom_center_xy,
        "projected_bottom_center_in_roi": bool(projected_bottom_center_in_roi),
        "projected_bottom_blocked_stale_geometry": bool(projected_bottom_geometry_available and not projected_bottom_geometry_fresh),
        "projected_bottom_blocked_no_continuity": bool(current_upper_anchor_present and not current_reliable_continuity),
        "projected_bottom_blocked_anchor_mismatch": bool(projected_bottom_geometry_available and projected_bottom_geometry_fresh and not projected_bottom_geometry_anchor_ok),
        "current_loss_like": bool(current_loss_like),
        "current_proxy_age_limit": int(proxy_age_limit),
        "current_proxy_age_bonus_active": bool(proxy_age_bonus_active),
        "current_reliable_proxy": bool(current_reliable_proxy),
        "current_real_continuity": bool(current_real_continuity),
        "current_reliable_continuity": bool(current_reliable_continuity),
        "current_flow_quality_ok": bool(current_flow_quality_ok),
        "head_lowerband_current_anchor_source": current_anchor_source,
        "head_lowerband_current_anchor_token": current_anchor_token,
        "head_lowerband_current_head_present": bool(current_head_anchor_present),
        "head_lowerband_lower_band_overlap": round(float(current_lower_band_overlap), 4),
        "head_lowerband_overlap_ok": bool(current_lower_band_overlap >= 0.50),
        "head_lowerband_blocked_no_continuity": bool(
            not current_reliable_continuity and current_head_anchor_present and not pose_result.ankle_in_roi
        ),
        "head_lowerband_blocked_low_overlap": bool(
            current_reliable_continuity and current_head_anchor_present and not (current_lower_band_overlap >= 0.50)
        ),
        "seed_window_ok": bool(seed_window_ok),
        "progression_window_frames": int(progression_history["window_frames"]),
        "progression_evidence_count": int(progression_history["strong_count"]),
        "progression_recent_frames": list(progression_history["strong_frames"]),
        "progression_non_inside_count": int(progression_history["non_inside_count"]),
        "progression_non_inside_frames": list(progression_history["non_inside_frames"]),
        "progression_distance_delta_px": float(progression_history["distance_delta_px"]),
        "progression_start_distance_px": float(progression_history["start_distance_px"]),
        "progression_best_distance_px": float(progression_history["best_distance_px"]),
        "non_inside_stricter_guard": bool(non_inside_stricter_guard),
        "repeated_progression_ok": bool(repeated_progression_ok),
        "head_support_window_frames": int(head_support_history["window_frames"]),
        "head_support_count": int(head_support_history["support_count"]),
        "head_support_frames": list(head_support_history["support_frames"]),
        "head_support_ok": bool(head_support_ok),
        "head_active_continuity_ok": bool(head_active_continuity_ok),
        "inside_support_window_frames": int(inside_support_history["window_frames"]),
        "inside_support_count": int(inside_support_history["support_count"]),
        "inside_support_frames": list(inside_support_history["support_frames"]),
        "inside_support_ok": bool(inside_support_ok),
        "candidate_accum_window_frames": int(candidate_accum_history["window_frames"]),
        "candidate_accum_count": int(candidate_accum_history["candidate_count"]),
        "candidate_accum_frames": list(candidate_accum_history["candidate_frames"]),
        "candidate_accum_head_support_count": int(candidate_accum_history["head_support_count"]),
        "candidate_accum_head_support_frames": list(candidate_accum_history["head_support_frames"]),
        "candidate_accum_motion_count": int(candidate_accum_history["motion_count"]),
        "candidate_accum_motion_frames": list(candidate_accum_history["motion_frames"]),
        "candidate_accum_motion_toward_mean": float(candidate_accum_history["motion_toward_mean"]),
        "candidate_accum_progression_count": int(candidate_accum_history["progression_count"]),
        "candidate_accum_progression_frames": list(candidate_accum_history["progression_frames"]),
        "candidate_accum_best_roi_distance_px": float(candidate_accum_history["best_roi_distance_px"]),
        "candidate_accum_best_patch_distance_px": float(candidate_accum_history["best_patch_distance_px"]),
        "candidate_accumulated_ok": bool(candidate_accumulated_ok),
        "candidate_accum_recent_ready": bool(recent_accumulated_ready),
        "candidate_accum_ready_frame_num": int(state.last_accumulated_head_ready_frame)
        if state.last_accumulated_head_ready_frame is not None
        else None,
        "accumulated_entry_depth_ok": bool(accumulated_entry_depth_ok),
        "shallow_head_blocked": bool(shallow_head_blocked),
        "head_only_raw_detection_blocked": bool(head_only_raw_detection_blocked),
        "progression_distance_ok": bool(progression_distance_ok),
        "progression_motion_strong": bool(progression_motion_strong),
        "immediate_loss_ok": bool(immediate_loss_ok),
    }


@dataclass
class TrackDecisionState:
    track_id: int
    source_id: int = 0
    state: str = STATE_OUT
    candidate_streak: int = 0
    confirm_streak: int = 0
    exit_streak: int = 0
    grace_left: int = 0
    last_real_frame: Optional[int] = None
    last_seen_frame: Optional[int] = None
    last_bbox_xyxy: list[float] | None = None
    last_real_ankle_frame: Optional[int] = None
    last_real_ankle_bbox_xyxy: list[float] | None = None
    last_real_ankles_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    projected_bottom_geometry: ProjectedBottomGeometry | None = None
    klt_boundary_seed: KltBoundaryConfirmSeed | None = None
    klt_boundary_history: list[KltBoundaryConfirmSeed] = field(default_factory=list)
    klt_candidate_seed: KltCandidateSeed | None = None
    last_candidate_context_frame: Optional[int] = None
    candidate_history_frames: list[int] = field(default_factory=list)
    klt_candidate_support_frames: list[int] = field(default_factory=list)
    last_accumulated_head_ready_frame: Optional[int] = None
    active_confirm_path: str = "none"
    active_confirm_basis: str = "none"
    event_index: int = 0
    active_event_id: str = ""

    def _begin_new_event(self, frame_num: int) -> None:
        self.event_index += 1
        self.active_event_id = f"track-{self.track_id}-event-{self.event_index}"
        self.last_seen_frame = frame_num

    def _clear_event(self) -> None:
        self.candidate_streak = 0
        self.confirm_streak = 0
        self.exit_streak = 0
        self.grace_left = 0
        self.last_real_ankle_frame = None
        self.last_real_ankle_bbox_xyxy = None
        self.last_real_ankles_xy.clear()
        self.projected_bottom_geometry = None
        self.klt_boundary_seed = None
        self.klt_boundary_history.clear()
        self.klt_candidate_seed = None
        self.last_candidate_context_frame = None
        self.candidate_history_frames.clear()
        self.klt_candidate_support_frames.clear()
        self.last_accumulated_head_ready_frame = None
        self.active_confirm_path = "none"
        self.active_confirm_basis = "none"
        self.active_event_id = ""

    def update(
        self,
        *,
        frame_num: int,
        ts_sec: float,
        row: SidecarRow | None,
        evidence: CandidateEvidence | None,
        pose_result: PoseProbeResult,
        params: DecisionParams,
        roi_id: str,
        roi_cache: RoiCache,
    ) -> dict[str, Any]:
        if row is not None:
            self.source_id = row.source_id
            self.last_seen_frame = frame_num
            if row.mode == "real":
                self.last_real_frame = frame_num
            if row.has_valid_bbox:
                self.last_bbox_xyxy = list(row.bbox_xyxy)
        if isinstance(pose_result.ankles, list):
            current_real_ankles_xy: dict[str, tuple[float, float]] = {}
            for ankle in pose_result.ankles:
                if not isinstance(ankle, dict):
                    continue
                ankle_name = str(ankle.get("name", "")).strip()
                if not ankle_name:
                    continue
                try:
                    ankle_xy = (float(ankle["x"]), float(ankle["y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                current_real_ankles_xy[ankle_name] = ankle_xy
            if current_real_ankles_xy and self.last_bbox_xyxy is not None and bbox_has_area_xyxy(self.last_bbox_xyxy):
                self.last_real_ankle_frame = int(frame_num)
                self.last_real_ankle_bbox_xyxy = list(self.last_bbox_xyxy)
                self.last_real_ankles_xy = dict(current_real_ankles_xy)
        if (
            row is not None
            and evidence is not None
            and row.mode == "real"
            and row.tracked_points >= max(1, int(params.klt_confirm_min_tracked_points))
            and row.has_valid_bbox
            and bbox_has_area_xyxy(row.patch_xyxy)
            and isinstance(pose_result.ankles, list)
            and len(pose_result.ankles) > 0
        ):
            anchor_source, _anchor_kind = classify_klt_anchor_source(str(row.pose_anchor_source or row.patch_source or "").strip())
            if anchor_source in {"head_like_keypoint", "shoulder_fallback"}:
                patch_h = max(1.0, float(row.patch_xyxy[3]) - float(row.patch_xyxy[1]))
                anchor_xy = bbox_center_xyxy(row.patch_xyxy)
                bottom_center_xy = tuple(map(float, evidence.bottom_center))
                if patch_h > 1.0 and bottom_center_xy[1] > anchor_xy[1]:
                    self.projected_bottom_geometry = ProjectedBottomGeometry(
                        frame_num=int(frame_num),
                        anchor_source=str(anchor_source),
                        dx_over_patch_h=float((bottom_center_xy[0] - anchor_xy[0]) / patch_h),
                        dy_over_patch_h=float((bottom_center_xy[1] - anchor_xy[1]) / patch_h),
                        patch_height=float(patch_h),
                    )

        klt_seed = build_klt_boundary_confirm_seed(
            frame_num=frame_num,
            row=row,
            evidence=evidence,
            state=self,
            roi_cache=roi_cache,
            params=params,
        )
        if klt_seed is not None:
            self.klt_boundary_seed = klt_seed
            if not self.klt_boundary_history or int(self.klt_boundary_history[-1].frame_num) != int(klt_seed.frame_num):
                self.klt_boundary_history.append(klt_seed)
            history_window = max(
                1,
                int(params.klt_progression_recent_window_frames),
                int(params.klt_confirm_recent_real_max_frames),
            )
            self.klt_boundary_history = [
                item
                for item in self.klt_boundary_history
                if (int(frame_num) - int(item.frame_num)) <= history_window
            ]
        klt_candidate_seed = build_klt_candidate_seed(
            frame_num=frame_num,
            row=row,
            evidence=evidence,
            state=self,
            roi_cache=roi_cache,
            params=params,
        )
        if klt_candidate_seed is not None:
            self.klt_candidate_seed = klt_candidate_seed

        prev_state = self.state
        continuity_id = self.active_event_id or f"track-{self.track_id}"
        klt_candidate_debug = evaluate_klt_candidate_signal(
            frame_num=frame_num,
            row=row,
            state=self,
            params=params,
        )
        candidate_seen = bool(evidence and (evidence.real_candidate or evidence.proxy_candidate)) or bool(
            klt_candidate_debug["signal"]
        )
        ankle_confirm_seen = bool(pose_result.ankle_in_roi)
        ankle_visible_now = bool(isinstance(pose_result.ankles, list) and len(pose_result.ankles) > 0)
        klt_confirm_debug = evaluate_klt_boundary_confirm(
            frame_num=frame_num,
            row=row,
            evidence=evidence,
            state=self,
            params=params,
            pose_result=pose_result,
            roi_cache=roi_cache,
        )
        if bool(klt_confirm_debug["accumulated_candidate_confirm"]):
            self.last_accumulated_head_ready_frame = int(frame_num)
        klt_candidate_support_frame = bool(klt_confirm_debug["current_candidate_support_frame"])
        klt_confirm_seen = bool(klt_confirm_debug["fired"]) and not ankle_confirm_seen and not ankle_visible_now
        confirm_seen = bool(ankle_confirm_seen or klt_confirm_seen)
        confirm_path = "ankle_in_roi" if ankle_confirm_seen else (str(klt_confirm_debug["reason"]) if klt_confirm_seen else "none")
        confirm_basis = confirm_path_to_basis(confirm_path)
        transitions: list[str] = []
        reasons = list(evidence.reasons if evidence is not None else [])
        if bool(klt_candidate_debug["open_ok"]) or bool(klt_candidate_debug["sustain_ok"]):
            reasons.append(str(klt_candidate_debug["reason"]))
        # KLT continuity sustain: if a proxy/frozen_hold row is present and the FSM
        # is already in CANDIDATE or IN_CONFIRMED, treat the KLT-backed row itself
        # as a candidate sustain signal.  This ensures KLT continuity takes priority
        # over grace/dwell fallback (Goal 2: real → KLT → hold → grace → out).
        klt_continuity_sustain = bool(
            row is not None
            and row.proxy_active
            and row.mode in {"proxy", "frozen_hold", "real_support_only"}
            and self.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}
        )
        candidate_seen = (
            bool(evidence and (evidence.real_candidate or evidence.proxy_candidate))
            or bool(klt_candidate_debug["signal"])
            or bool(klt_candidate_support_frame)
            or bool(klt_continuity_sustain)
        )
        if candidate_seen or self.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
            self.last_candidate_context_frame = frame_num

        if candidate_seen:
            self.candidate_streak += 1
        elif self.state == STATE_OUT:
            self.candidate_streak = 0

        event_type = "out_keep"

        if self.state == STATE_OUT:
            self.confirm_streak = 0
            self.exit_streak = 0
            self.grace_left = 0
            if candidate_seen and self.candidate_streak >= max(1, int(params.candidate_enter_n)):
                self.state = STATE_CANDIDATE
                self._begin_new_event(frame_num)
                self.grace_left = max(0, int(params.grace_frames))
                transitions.append(f"{STATE_OUT}->{STATE_CANDIDATE}")
                event_type = "candidate_start"
            elif candidate_seen:
                event_type = "candidate_observed"

        if self.state == STATE_CANDIDATE:
            if candidate_seen:
                self.exit_streak = 0
                self.grace_left = max(0, int(params.grace_frames))
                if event_type == "out_keep":
                    event_type = "candidate_keep"
            else:
                self.confirm_streak = 0
                if self.grace_left > 0:
                    self.grace_left -= 1
                    event_type = "candidate_grace"
                else:
                    self.exit_streak += 1
                    event_type = "candidate_lost"

            if confirm_seen:
                self.confirm_streak += 1
                reasons.append("ankle_in_roi" if ankle_confirm_seen else str(klt_confirm_debug["reason"]))
            elif candidate_seen:
                self.confirm_streak = 0

            if self.confirm_streak >= max(1, int(params.confirm_enter_n)):
                self.state = STATE_IN_CONFIRMED
                self.exit_streak = 0
                self.grace_left = max(0, int(params.grace_frames))
                self.active_confirm_path = confirm_path
                self.active_confirm_basis = confirm_basis
                transitions.append(f"{STATE_CANDIDATE}->{STATE_IN_CONFIRMED}")
                event_type = "in_confirmed"
            elif self.exit_streak >= max(1, int(params.exit_n)):
                continuity_id = self.active_event_id or continuity_id
                self.state = STATE_OUT
                transitions.append(f"{STATE_CANDIDATE}->{STATE_OUT}")
                event_type = "exit"
                self._clear_event()

        if self.state == STATE_IN_CONFIRMED:
            if ankle_confirm_seen and self.active_confirm_path != "ankle_in_roi":
                self.active_confirm_path = "ankle_in_roi"
                self.active_confirm_basis = "ankle"
            if candidate_seen or confirm_seen:
                self.exit_streak = 0
                self.grace_left = max(0, int(params.grace_frames))
                if event_type == "out_keep":
                    event_type = "in_keep"
            else:
                if self.grace_left > 0:
                    self.grace_left -= 1
                    event_type = "in_grace"
                else:
                    self.exit_streak += 1
                    event_type = "in_lost"

            if self.exit_streak >= max(1, int(params.exit_n)):
                continuity_id = self.active_event_id or continuity_id
                self.state = STATE_OUT
                transitions.append(f"{STATE_IN_CONFIRMED}->{STATE_OUT}")
                event_type = "exit"
                self._clear_event()

        if candidate_seen or self.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
            if not self.candidate_history_frames or int(self.candidate_history_frames[-1]) != int(frame_num):
                self.candidate_history_frames.append(int(frame_num))
        if klt_candidate_support_frame:
            if not self.klt_candidate_support_frames or int(self.klt_candidate_support_frames[-1]) != int(frame_num):
                self.klt_candidate_support_frames.append(int(frame_num))
        history_window = max(
            1,
            int(params.klt_candidate_accum_window_frames),
            int(params.klt_candidate_recent_context_frames),
        )
        self.candidate_history_frames = [
            int(item)
            for item in self.candidate_history_frames
            if (int(frame_num) - int(item)) <= history_window
        ]
        self.klt_candidate_support_frames = [
            int(item)
            for item in self.klt_candidate_support_frames
            if (int(frame_num) - int(item)) <= history_window
        ]

        bbox_xyxy = list(self.last_bbox_xyxy or (evidence.bbox_xyxy if evidence is not None else []))
        active_confirm_path = self.active_confirm_path if self.state == STATE_IN_CONFIRMED else "none"
        active_confirm_basis = self.active_confirm_basis if self.state == STATE_IN_CONFIRMED else "none"
        emitted_confirm_path = active_confirm_path if self.state == STATE_IN_CONFIRMED else confirm_path
        emitted_confirm_basis = active_confirm_basis if self.state == STATE_IN_CONFIRMED else confirm_basis
        return {
            "frame_num": int(frame_num),
            "ts_sec": round(float(ts_sec), 4),
            "source_id": int(self.source_id),
            "track_id": int(self.track_id),
            "continuity_id": continuity_id,
            "state": self.state,
            "state_prev": prev_state,
            "event_type": event_type,
            "transition": " ".join(transitions),
            "roi_id": roi_id,
            "mode": row.mode if row is not None else "none",
            "proxy_age": int(row.proxy_age) if row is not None else None,
            "tracking_event": row.event if row is not None else "",
            "tracking_stop_reason": row.stop_reason if row is not None else "",
            "tracking_handoff_reason": row.handoff_reason if row is not None else "",
            "bbox": [round(float(v), 2) for v in bbox_xyxy] if bbox_xyxy else [],
            "evidence": {
                "real_track_candidate": bool(evidence.real_candidate) if evidence is not None else False,
                "proxy_candidate": bool(evidence.proxy_candidate) if evidence is not None else False,
                "proxy_start_allowed": bool(evidence.proxy_start_allowed) if evidence is not None else False,
                "klt_candidate_signal": bool(klt_candidate_debug["signal"]),
                "klt_continuity_sustain": bool(klt_continuity_sustain),
                "ankle_confirm": bool(ankle_confirm_seen),
                "klt_boundary_confirm": bool(klt_confirm_seen),
            },
            "candidate_metrics": {
                "bbox_overlap": round(float(evidence.bbox_overlap), 4) if evidence is not None else 0.0,
                "lower_band_overlap": round(float(evidence.lower_band_overlap), 4) if evidence is not None else 0.0,
                "bottom_center": [
                    round(float(evidence.bottom_center[0]), 2),
                    round(float(evidence.bottom_center[1]), 2),
                ]
                if evidence is not None
                else [],
                "bottom_center_in_roi": bool(evidence.bottom_center_in_roi) if evidence is not None else False,
                "signed_distance_px": round(float(evidence.signed_distance_px), 3) if evidence is not None else 0.0,
                "roi_min_distance_px": round(float(evidence.roi_min_distance_px), 3) if evidence is not None else 0.0,
                "motion_toward_score": round(float(evidence.motion_toward_score), 4) if evidence is not None else 0.0,
                "motion_speed_px": round(float(evidence.motion_speed_px), 3) if evidence is not None else 0.0,
                "early_candidate_pretrigger": bool(evidence.early_candidate_pretrigger) if evidence is not None else False,
                "klt_candidate_signal": bool(klt_candidate_debug["signal"]),
                "klt_candidate_open": bool(klt_candidate_debug["open_ok"]),
                "klt_candidate_sustain": bool(klt_candidate_debug["sustain_ok"]),
                "klt_candidate_reason": str(klt_candidate_debug["reason"]),
                "klt_candidate_anchor_source": str(klt_candidate_debug["anchor_source"]),
                "klt_candidate_anchor_kind": str(klt_candidate_debug["anchor_kind"]),
                "klt_candidate_anchor_inside_roi": bool(klt_candidate_debug["anchor_inside_roi"]),
                "klt_candidate_boundary_near_or_inside": bool(klt_candidate_debug["boundary_near_or_inside"]),
                "klt_candidate_anchor_xy": list(klt_candidate_debug["anchor_xy"]),
                "klt_candidate_seed_frame_num": klt_candidate_debug["seed_frame_num"],
                "klt_candidate_frames_since_seed": klt_candidate_debug["frames_since_seed"],
                "klt_candidate_recent_real_seed": bool(klt_candidate_debug["recent_real_seed"]),
                "klt_candidate_recent_candidate_context": bool(klt_candidate_debug["recent_candidate_context_ok"]),
                "klt_candidate_inward_progression_ok": bool(klt_candidate_debug["inward_progression_ok"]),
                "klt_candidate_continuity_seeded": bool(klt_candidate_debug["continuity_seeded"]),
                "klt_candidate_reliability_ok": bool(klt_candidate_debug["reliability_ok"]),
                "klt_candidate_sparse_gap_now": bool(klt_candidate_debug["sparse_gap_now"]),
                "klt_candidate_tracked_points": int(klt_candidate_debug["tracked_points"]),
                "klt_candidate_flow_mag": float(klt_candidate_debug["flow_mag"]),
                "klt_candidate_proxy_mode": str(klt_candidate_debug["proxy_mode"]),
                "klt_candidate_proxy_age": int(klt_candidate_debug["proxy_age"]),
                "klt_candidate_patch_roi_min_distance_px": float(klt_candidate_debug["patch_roi_min_distance_px"]),
                "klt_candidate_motion_toward_score": float(klt_candidate_debug["motion_toward_score"]),
                "klt_candidate_motion_speed_px": float(klt_candidate_debug["motion_speed_px"]),
                "score": round(float(evidence.score), 4) if evidence is not None else 0.0,
            },
            "confirm": {
                "required_ankle": bool(params.confirm_requires_ankle and not klt_confirm_seen),
                "ankle_confirm_enabled": bool(params.confirm_requires_ankle),
                "klt_confirm_enabled": True,
                "confirm_path": emitted_confirm_path,
                "confirm_basis": emitted_confirm_basis,
                "active_confirm_path": active_confirm_path,
                "active_confirm_basis": active_confirm_basis,
                "status": pose_result.status,
                "attempted": bool(pose_result.attempted),
                "ankle_visible_now": bool(ankle_visible_now),
                "ankles": pose_result.ankles,
                "klt_boundary_confirm": bool(klt_confirm_seen),
                "klt_reason": str(klt_confirm_debug["reason"]),
                "klt_variant_reason": str(klt_confirm_debug["variant_reason"]),
                "klt_pattern": str(klt_confirm_debug["pattern"]),
                "klt_anchor_source": str(klt_confirm_debug["anchor_source"]),
                "klt_anchor_kind": str(klt_confirm_debug["anchor_kind"]),
                "klt_anchor_inside_roi": bool(klt_confirm_debug["anchor_inside_roi"]),
                "klt_boundary_near_or_inside": bool(klt_confirm_debug["boundary_near_or_inside"]),
                "klt_anchor_xy": list(klt_confirm_debug["anchor_xy"]),
                "klt_seed_frame_num": klt_confirm_debug["seed_frame_num"],
                "klt_frames_since_seed": klt_confirm_debug["frames_since_seed"],
                "klt_loss_gap_frames": klt_confirm_debug["loss_gap_frames"],
                "klt_recent_real_seed": bool(klt_confirm_debug["recent_real_seed"]),
                "klt_recent_candidate_context": bool(klt_confirm_debug["recent_candidate_context_ok"]),
                "klt_boundary_context": bool(klt_confirm_debug["boundary_context"]),
                "klt_motion_toward_ok": bool(klt_confirm_debug["motion_toward_ok"]),
                "klt_continuity_seeded": bool(klt_confirm_debug["continuity_seeded"]),
                "klt_reliability_ok": bool(klt_confirm_debug["reliability_ok"]),
                "klt_entry_progression_ok": bool(klt_confirm_debug["entry_progression_ok"]),
                "klt_loss_after_entry": bool(klt_confirm_debug["loss_after_entry"]),
                "klt_deep_inside_confirm_ok": bool(klt_confirm_debug["deep_inside_confirm_ok"]),
                "klt_head_lowerband_confirm": bool(klt_confirm_debug["head_lowerband_confirm"]),
                "klt_accumulated_candidate_confirm": bool(klt_confirm_debug["accumulated_candidate_confirm"]),
                "klt_boundary_progression_then_lost": bool(klt_confirm_debug["boundary_progression_then_lost"]),
                "klt_head_based_confirm_disabled": bool(klt_confirm_debug["head_based_confirm_disabled"]),
                "klt_disabled_head_deep_inside_would_fire": bool(
                    klt_confirm_debug["disabled_head_deep_inside_would_fire"]
                ),
                "klt_disabled_head_lowerband_would_fire": bool(
                    klt_confirm_debug["disabled_head_lowerband_would_fire"]
                ),
                "klt_disabled_head_accumulated_would_fire": bool(
                    klt_confirm_debug["disabled_head_accumulated_would_fire"]
                ),
                "klt_disabled_head_progression_would_fire": bool(
                    klt_confirm_debug["disabled_head_progression_would_fire"]
                ),
                "klt_ankle_proxy_confirm": bool(klt_confirm_debug["ankle_proxy_confirm"]),
                "klt_bottom_center_proxy_confirm": bool(klt_confirm_debug["bottom_center_proxy_confirm"]),
                "klt_ankle_proxy_available": bool(klt_confirm_debug["ankle_proxy_available"]),
                "klt_ankle_proxy_seed_recent": bool(klt_confirm_debug["ankle_proxy_seed_recent"]),
                "klt_ankle_proxy_in_roi": bool(klt_confirm_debug["ankle_proxy_in_roi"]),
                "klt_ankle_proxy_points": list(klt_confirm_debug["ankle_proxy_points"]),
                "klt_bottom_center_proxy_xy": list(klt_confirm_debug["bottom_center_proxy_xy"]),
                "klt_bottom_center_proxy_in_roi": bool(klt_confirm_debug["bottom_center_proxy_in_roi"]),
                "klt_current_lowerband_confirm": bool(klt_confirm_debug["current_lowerband_confirm"]),
                "klt_current_lowerband_recent_real_ok": bool(
                    klt_confirm_debug["current_lowerband_recent_real_ok"]
                ),
                "klt_current_lowerband_boundary_ok": bool(
                    klt_confirm_debug["current_lowerband_boundary_ok"]
                ),
                "klt_current_lowerband_reliability_ok": bool(
                    klt_confirm_debug["current_lowerband_reliability_ok"]
                ),
                "klt_candidate_support_frame": bool(klt_confirm_debug["current_candidate_support_frame"]),
                "klt_candidate_support_count": int(klt_confirm_debug["current_candidate_support_count"]),
                "klt_candidate_support_frames": list(klt_confirm_debug["current_candidate_support_frames"]),
                "klt_candidate_context_ok": bool(klt_confirm_debug["current_candidate_context_ok"]),
                "klt_current_lowerband_blocked_no_continuity": bool(
                    klt_confirm_debug["current_lowerband_blocked_no_continuity"]
                ),
                "klt_current_lowerband_blocked_low_overlap": bool(
                    klt_confirm_debug["current_lowerband_blocked_low_overlap"]
                ),
                "klt_current_lowerband_blocked_stronger_path": bool(
                    klt_confirm_debug["current_lowerband_blocked_stronger_path"]
                ),
                "klt_display_continuity_confirm": bool(klt_confirm_debug["display_continuity_confirm"]),
                "klt_projected_bottom_center_confirm": bool(klt_confirm_debug["projected_bottom_center_confirm"]),
                "klt_projected_bottom_geometry_available": bool(
                    klt_confirm_debug["projected_bottom_geometry_available"]
                ),
                "klt_projected_bottom_geometry_source_frame_num": klt_confirm_debug[
                    "projected_bottom_geometry_source_frame_num"
                ],
                "klt_projected_bottom_geometry_age_frames": klt_confirm_debug[
                    "projected_bottom_geometry_age_frames"
                ],
                "klt_projected_bottom_geometry_anchor_source": str(
                    klt_confirm_debug["projected_bottom_geometry_anchor_source"]
                ),
                "klt_projected_bottom_center_xy": list(klt_confirm_debug["projected_bottom_center_xy"]),
                "klt_projected_bottom_center_in_roi": bool(klt_confirm_debug["projected_bottom_center_in_roi"]),
                "klt_projected_bottom_blocked_stale_geometry": bool(
                    klt_confirm_debug["projected_bottom_blocked_stale_geometry"]
                ),
                "klt_projected_bottom_blocked_no_continuity": bool(
                    klt_confirm_debug["projected_bottom_blocked_no_continuity"]
                ),
                "klt_projected_bottom_blocked_anchor_mismatch": bool(
                    klt_confirm_debug["projected_bottom_blocked_anchor_mismatch"]
                ),
                "klt_current_loss_like": bool(klt_confirm_debug["current_loss_like"]),
                "klt_current_proxy_age_limit": int(klt_confirm_debug["current_proxy_age_limit"]),
                "klt_current_proxy_age_bonus_active": bool(klt_confirm_debug["current_proxy_age_bonus_active"]),
                "klt_current_reliable_proxy": bool(klt_confirm_debug["current_reliable_proxy"]),
                "klt_current_real_continuity": bool(klt_confirm_debug["current_real_continuity"]),
                "klt_current_reliable_continuity": bool(klt_confirm_debug["current_reliable_continuity"]),
                "klt_current_flow_quality_ok": bool(klt_confirm_debug["current_flow_quality_ok"]),
                "klt_head_lowerband_current_anchor_source": str(
                    klt_confirm_debug["head_lowerband_current_anchor_source"]
                ),
                "klt_head_lowerband_current_anchor_token": str(
                    klt_confirm_debug["head_lowerband_current_anchor_token"]
                ),
                "klt_head_lowerband_current_head_present": bool(
                    klt_confirm_debug["head_lowerband_current_head_present"]
                ),
                "klt_head_lowerband_lower_band_overlap": round(
                    float(klt_confirm_debug["head_lowerband_lower_band_overlap"]), 4
                ),
                "klt_head_lowerband_overlap_ok": bool(klt_confirm_debug["head_lowerband_overlap_ok"]),
                "klt_head_lowerband_blocked_no_continuity": bool(
                    klt_confirm_debug["head_lowerband_blocked_no_continuity"]
                ),
                "klt_head_lowerband_blocked_low_overlap": bool(
                    klt_confirm_debug["head_lowerband_blocked_low_overlap"]
                ),
                "klt_seed_window_ok": bool(klt_confirm_debug["seed_window_ok"]),
                "klt_progression_window_frames": int(klt_confirm_debug["progression_window_frames"]),
                "klt_progression_evidence_count": int(klt_confirm_debug["progression_evidence_count"]),
                "klt_progression_recent_frames": list(klt_confirm_debug["progression_recent_frames"]),
                "klt_progression_non_inside_count": int(klt_confirm_debug["progression_non_inside_count"]),
                "klt_progression_non_inside_frames": list(klt_confirm_debug["progression_non_inside_frames"]),
                "klt_progression_distance_delta_px": float(klt_confirm_debug["progression_distance_delta_px"]),
                "klt_progression_start_distance_px": float(klt_confirm_debug["progression_start_distance_px"]),
                "klt_progression_best_distance_px": float(klt_confirm_debug["progression_best_distance_px"]),
                "klt_non_inside_stricter_guard": bool(klt_confirm_debug["non_inside_stricter_guard"]),
                "klt_repeated_progression_ok": bool(klt_confirm_debug["repeated_progression_ok"]),
                "klt_head_support_window_frames": int(klt_confirm_debug["head_support_window_frames"]),
                "klt_head_support_count": int(klt_confirm_debug["head_support_count"]),
                "klt_head_support_frames": list(klt_confirm_debug["head_support_frames"]),
                "klt_head_support_ok": bool(klt_confirm_debug["head_support_ok"]),
                "klt_head_active_continuity_ok": bool(klt_confirm_debug["head_active_continuity_ok"]),
                "klt_inside_support_window_frames": int(klt_confirm_debug["inside_support_window_frames"]),
                "klt_inside_support_count": int(klt_confirm_debug["inside_support_count"]),
                "klt_inside_support_frames": list(klt_confirm_debug["inside_support_frames"]),
                "klt_inside_support_ok": bool(klt_confirm_debug["inside_support_ok"]),
                "klt_candidate_accum_window_frames": int(klt_confirm_debug["candidate_accum_window_frames"]),
                "klt_candidate_accum_count": int(klt_confirm_debug["candidate_accum_count"]),
                "klt_candidate_accum_frames": list(klt_confirm_debug["candidate_accum_frames"]),
                "klt_candidate_accum_head_support_count": int(klt_confirm_debug["candidate_accum_head_support_count"]),
                "klt_candidate_accum_head_support_frames": list(klt_confirm_debug["candidate_accum_head_support_frames"]),
                "klt_candidate_accum_motion_count": int(klt_confirm_debug["candidate_accum_motion_count"]),
                "klt_candidate_accum_motion_frames": list(klt_confirm_debug["candidate_accum_motion_frames"]),
                "klt_candidate_accum_motion_toward_mean": float(klt_confirm_debug["candidate_accum_motion_toward_mean"]),
                "klt_candidate_accum_progression_count": int(klt_confirm_debug["candidate_accum_progression_count"]),
                "klt_candidate_accum_progression_frames": list(klt_confirm_debug["candidate_accum_progression_frames"]),
                "klt_candidate_accum_best_roi_distance_px": float(klt_confirm_debug["candidate_accum_best_roi_distance_px"]),
                "klt_candidate_accum_best_patch_distance_px": float(klt_confirm_debug["candidate_accum_best_patch_distance_px"]),
                "klt_candidate_accumulated_ok": bool(klt_confirm_debug["candidate_accumulated_ok"]),
                "klt_candidate_accum_recent_ready": bool(klt_confirm_debug["candidate_accum_recent_ready"]),
                "klt_candidate_accum_ready_frame_num": klt_confirm_debug["candidate_accum_ready_frame_num"],
                "klt_accumulated_entry_depth_ok": bool(klt_confirm_debug["accumulated_entry_depth_ok"]),
                "klt_shallow_head_blocked": bool(klt_confirm_debug["shallow_head_blocked"]),
                "klt_head_only_raw_detection_blocked": bool(klt_confirm_debug["head_only_raw_detection_blocked"]),
                "klt_progression_distance_ok": bool(klt_confirm_debug["progression_distance_ok"]),
                "klt_progression_motion_strong": bool(klt_confirm_debug["progression_motion_strong"]),
                "klt_immediate_loss_ok": bool(klt_confirm_debug["immediate_loss_ok"]),
                "klt_tracked_points": int(klt_confirm_debug["tracked_points"]),
                "klt_flow_mag": float(klt_confirm_debug["flow_mag"]),
                "klt_proxy_mode": str(klt_confirm_debug["proxy_mode"]),
                "klt_proxy_age": int(klt_confirm_debug["proxy_age"]),
                "klt_roi_min_distance_px": float(klt_confirm_debug["roi_min_distance_px"]),
                "klt_patch_roi_min_distance_px": float(klt_confirm_debug["patch_roi_min_distance_px"]),
                "klt_signed_distance_px": float(klt_confirm_debug["signed_distance_px"]),
                "klt_motion_toward_score": float(klt_confirm_debug["motion_toward_score"]),
                "klt_motion_speed_px": float(klt_confirm_debug["motion_speed_px"]),
            },
            "reasons": sorted({str(reason) for reason in reasons if str(reason).strip()}),
            "counters": {
                "candidate_streak": int(self.candidate_streak),
                "confirm_streak": int(self.confirm_streak),
                "exit_streak": int(self.exit_streak),
                "grace_left": int(self.grace_left),
            },
        }


def compute_motion_toward_roi_metrics(
    *,
    row: SidecarRow,
    state: TrackDecisionState,
    roi_cache: RoiCache,
    frame_num: int,
    params: DecisionParams,
) -> tuple[float, float]:
    if state.last_bbox_xyxy is None or state.last_seen_frame is None:
        return 0.0, 0.0
    if int(frame_num) - int(state.last_seen_frame) > max(1, int(params.cand_motion_max_frame_gap)):
        return 0.0, 0.0

    curr_center = bbox_center_xyxy(row.bbox_xyxy)
    prev_center = bbox_center_xyxy(state.last_bbox_xyxy)
    vel_x = float(curr_center[0]) - float(prev_center[0])
    vel_y = float(curr_center[1]) - float(prev_center[1])
    speed_px = float((vel_x * vel_x + vel_y * vel_y) ** 0.5)
    if speed_px < float(params.cand_motion_min_speed_px):
        return 0.0, speed_px

    nearest_roi_point = nearest_point_on_roi_poly(curr_center, roi_cache.poly)
    to_roi_x = float(nearest_roi_point[0]) - float(curr_center[0])
    to_roi_y = float(nearest_roi_point[1]) - float(curr_center[1])
    to_roi_norm = float((to_roi_x * to_roi_x + to_roi_y * to_roi_y) ** 0.5)
    if to_roi_norm <= 1e-6:
        return 1.0, speed_px
    toward_score = ((vel_x * to_roi_x) + (vel_y * to_roi_y)) / max(1e-6, speed_px * to_roi_norm)
    return float(_clamp(toward_score, -1.0, 1.0)), speed_px


def build_candidate_evidence(
    *,
    row: SidecarRow | None,
    state: TrackDecisionState,
    params: DecisionParams,
    roi_cache: RoiCache,
    feature_cfg: FeatureConfig,
    score_weights: ScoreWeights,
    image_w: int,
    image_h: int,
) -> CandidateEvidence | None:
    if row is None or not row.has_valid_bbox:
        return None

    x1, y1, x2, y2 = row.bbox_xyxy
    factors = compute_bbox_factors(
        bbox=[x1, y1, x2, y2, 1.0],
        roi_cache=roi_cache,
        cfg_norms=feature_cfg,
        image_w=image_w,
        image_h=image_h,
    )
    bbox_overlap = bbox_roi_overlap_ratio(x1, y1, x2, y2, roi_cache.mask)
    score = compute_score(factors, score_weights)
    bottom_center = factors["bc"]
    bottom_center_in_roi = point_in_roi(bottom_center, roi_cache.mask)
    roi_min_distance_px = bbox_roi_min_distance_px(x1, y1, x2, y2, roi_cache.signed_dist)
    motion_toward_score, motion_speed_px = compute_motion_toward_roi_metrics(
        row=row,
        state=state,
        roi_cache=roi_cache,
        frame_num=int(row.frame_num),
        params=params,
    )
    in_candidate_hysteresis = state.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}
    cand_distance_thr = float(params.cand_distance_sustain_px if in_candidate_hysteresis else params.cand_distance_enter_px)
    cand_toward_thr = float(
        params.cand_motion_toward_score_sustain if in_candidate_hysteresis else params.cand_motion_toward_score_enter
    )
    overlap_candidate = False

    reasons: list[str] = []
    if bottom_center_in_roi:
        reasons.append("bottom_center_in_roi")
        overlap_candidate = True
    if bbox_overlap >= float(params.candidate_iou_or_overlap_thr):
        reasons.append("bbox_overlap")
        overlap_candidate = True
    if float(factors["ov"]) >= float(params.candidate_iou_or_overlap_thr):
        reasons.append("lower_band_overlap")
        overlap_candidate = True
    early_candidate_pretrigger = bool(
        roi_min_distance_px <= cand_distance_thr
        and motion_speed_px >= float(params.cand_motion_min_speed_px)
        and motion_toward_score >= cand_toward_thr
    )
    if early_candidate_pretrigger:
        reasons.append("near_roi_and_moving_toward_roi")
    candidate_geom = bool(overlap_candidate or early_candidate_pretrigger)
    is_real = row.mode == "real"
    is_proxy = row.mode == "proxy" and row.proxy_active
    is_frozen = row.mode == "frozen_hold" and row.proxy_active
    proxy_start_allowed = bool(is_proxy and row.proxy_age <= max(1, int(params.proxy_start_max_age_frames)))
    real_candidate = bool(is_real and candidate_geom)
    proxy_candidate = False
    if candidate_geom and (is_proxy or is_frozen):
        if state.state != STATE_OUT:
            proxy_candidate = True
        elif is_proxy and proxy_start_allowed:
            proxy_candidate = True
        else:
            reasons.append("proxy_start_blocked")

    return CandidateEvidence(
        bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        source_mode=row.mode,
        candidate_geom=bool(candidate_geom),
        real_candidate=bool(real_candidate),
        proxy_candidate=bool(proxy_candidate),
        proxy_start_allowed=bool(proxy_start_allowed),
        bbox_overlap=float(bbox_overlap),
        lower_band_overlap=float(factors["ov"]),
        bottom_center=(float(bottom_center[0]), float(bottom_center[1])),
        bottom_center_in_roi=bool(bottom_center_in_roi),
        signed_distance_px=float(factors["sd"]),
        roi_min_distance_px=float(roi_min_distance_px),
        motion_toward_score=float(motion_toward_score),
        motion_speed_px=float(motion_speed_px),
        early_candidate_pretrigger=bool(early_candidate_pretrigger),
        score=float(score),
        score_factors=dict(factors),
        reasons=reasons,
    )


def should_emit_record(row: SidecarRow | None, record: dict[str, Any]) -> bool:
    transition = str(record.get("transition", "")).strip()
    if transition:
        return True
    if row is not None:
        return True
    state_prev = str(record.get("state_prev", ""))
    state_now = str(record.get("state", ""))
    if state_prev != STATE_OUT or state_now != STATE_OUT:
        return True
    return False


def run_intrusion_decision_pass(
    *,
    video_path: str | Path,
    roi_json: str | Path,
    sidecar_csv: str | Path,
    events_path: str | Path,
    params: DecisionParams,
    feature_cfg: FeatureConfig,
    score_weights: ScoreWeights,
    pose_probe_settings: PoseProbeSettings | None,
) -> dict[str, Any]:
    _require_decision_deps()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: '{video_path}'")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            raise RuntimeError(f"Could not decode frames from video: '{video_path}'")
        height, width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    roi_cache, roi_meta = load_roi_cache_from_json(roi_json=roi_json, width=width, height=height)
    rows_by_frame, sidecar_summary = load_sidecar_rows(sidecar_csv)
    pose_probe = PoseAnkleProbe(roi_cache=roi_cache, settings=pose_probe_settings) if pose_probe_settings is not None else None

    contexts: dict[int, TrackDecisionState] = {}
    emitted_count = 0
    confirmed_count = 0
    tracks_seen: set[int] = set()

    events_path_obj = Path(events_path)
    events_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with events_path_obj.open("w", encoding="utf-8") as f:
        frame_num = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_rows = rows_by_frame.get(frame_num, {})
            active_track_ids = set(frame_rows.keys())
            active_track_ids.update(
                track_id
                for (track_id, ctx) in contexts.items()
                if ctx.state != STATE_OUT or ctx.grace_left > 0 or ctx.exit_streak > 0
            )

            for track_id in sorted(active_track_ids):
                row = frame_rows.get(track_id)
                ctx = contexts.setdefault(track_id, TrackDecisionState(track_id=track_id))
                evidence = build_candidate_evidence(
                    row=row,
                    state=ctx,
                    params=params,
                    roi_cache=roi_cache,
                    feature_cfg=feature_cfg,
                    score_weights=score_weights,
                    image_w=width,
                    image_h=height,
                )

                pose_bbox_xyxy: list[float] | None = None
                if evidence is not None and evidence.candidate_geom:
                    pose_bbox_xyxy = list(evidence.bbox_xyxy)
                elif ctx.state in {STATE_CANDIDATE, STATE_IN_CONFIRMED} and row is not None and row.has_valid_bbox:
                    pose_bbox_xyxy = list(row.bbox_xyxy)

                if pose_probe is None:
                    pose_result = PoseProbeResult.skipped("pose_probe_disabled")
                elif ctx.state == STATE_IN_CONFIRMED:
                    if ctx.active_confirm_basis == "ankle":
                        pose_result = PoseProbeResult.skipped("pose_not_needed_in_confirmed_ankle")
                    elif pose_bbox_xyxy is not None:
                        pose_result = pose_probe.probe(frame=frame, bbox_xyxy=pose_bbox_xyxy)
                    else:
                        pose_result = PoseProbeResult.skipped("pose_upgrade_no_bbox")
                elif pose_bbox_xyxy is None:
                    pose_result = PoseProbeResult.skipped(
                        "pose_not_needed_no_bbox" if ctx.state == STATE_CANDIDATE else "pose_not_needed"
                    )
                else:
                    pose_result = pose_probe.probe(frame=frame, bbox_xyxy=pose_bbox_xyxy)

                record = ctx.update(
                    frame_num=frame_num,
                    ts_sec=float(frame_num) / max(1e-6, fps),
                    row=row,
                    evidence=evidence,
                    pose_result=pose_result,
                    params=params,
                    roi_id=roi_meta["roi_id"],
                    roi_cache=roi_cache,
                )

                if should_emit_record(row=row, record=record):
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    emitted_count += 1
                    tracks_seen.add(track_id)
                    if str(record.get("event_type", "")) == "in_confirmed":
                        confirmed_count += 1

            frame_num += 1

    cap.release()

    open_tracks: list[dict[str, Any]] = []
    for ctx in sorted(contexts.values(), key=lambda item: item.track_id):
        if ctx.state != STATE_OUT:
            open_tracks.append(
                {
                    "track_id": int(ctx.track_id),
                    "state": ctx.state,
                    "active_event_id": ctx.active_event_id,
                }
            )

    return {
        "video_path": str(video_path),
        "frame_count": int(frame_count),
        "fps": float(fps),
        "image_size": {"width": int(width), "height": int(height)},
        "roi": roi_meta,
        "sidecar": sidecar_summary,
        "events_path": str(events_path_obj),
        "records_emitted": int(emitted_count),
        "tracks_seen": sorted(int(track_id) for track_id in tracks_seen),
        "confirmed_events": int(confirmed_count),
        "open_tracks_at_video_end": open_tracks,
        "pose_probe_status": pose_probe.model_status if pose_probe is not None else "pose_probe_disabled",
        "decision_params": {
            "candidate_enter_n": int(params.candidate_enter_n),
            "confirm_enter_n": int(params.confirm_enter_n),
            "exit_n": int(params.exit_n),
            "grace_frames": int(params.grace_frames),
            "candidate_iou_or_overlap_thr": float(params.candidate_iou_or_overlap_thr),
            "cand_distance_enter_px": float(params.cand_distance_enter_px),
            "cand_distance_sustain_px": float(params.cand_distance_sustain_px),
            "cand_motion_toward_score_enter": float(params.cand_motion_toward_score_enter),
            "cand_motion_toward_score_sustain": float(params.cand_motion_toward_score_sustain),
            "cand_motion_min_speed_px": float(params.cand_motion_min_speed_px),
            "cand_motion_max_frame_gap": int(params.cand_motion_max_frame_gap),
            "klt_confirm_recent_real_max_frames": int(params.klt_confirm_recent_real_max_frames),
            "klt_confirm_max_loss_frames": int(params.klt_confirm_max_loss_frames),
            "klt_continuity_max_proxy_age_frames": int(params.klt_continuity_max_proxy_age_frames),
            "klt_continuity_bonus_proxy_age_frames": int(params.klt_continuity_bonus_proxy_age_frames),
            "klt_continuity_bonus_recent_real_max_frames": int(
                params.klt_continuity_bonus_recent_real_max_frames
            ),
            "klt_continuity_bonus_min_tracked_points": int(params.klt_continuity_bonus_min_tracked_points),
            "klt_confirm_min_tracked_points": int(params.klt_confirm_min_tracked_points),
            "klt_confirm_min_flow_mag": float(params.klt_confirm_min_flow_mag),
            "klt_confirm_boundary_max_distance_px": float(params.klt_confirm_boundary_max_distance_px),
            "klt_candidate_recent_context_frames": int(params.klt_candidate_recent_context_frames),
            "klt_progression_confirm_distance_px": float(params.klt_progression_confirm_distance_px),
            "klt_progression_confirm_toward_score": float(params.klt_progression_confirm_toward_score),
            "klt_progression_confirm_loss_gap_frames": int(params.klt_progression_confirm_loss_gap_frames),
            "klt_progression_recent_window_frames": int(params.klt_progression_recent_window_frames),
            "klt_progression_min_observations": int(params.klt_progression_min_observations),
            "klt_progression_min_distance_improvement_px": float(params.klt_progression_min_distance_improvement_px),
            "klt_head_confirm_support_window_frames": int(params.klt_head_confirm_support_window_frames),
            "klt_head_confirm_min_support_frames": int(params.klt_head_confirm_min_support_frames),
            "klt_candidate_accum_window_frames": int(params.klt_candidate_accum_window_frames),
            "klt_candidate_accum_min_frames": int(params.klt_candidate_accum_min_frames),
            "klt_candidate_accum_min_head_support_frames": int(params.klt_candidate_accum_min_head_support_frames),
            "klt_candidate_accum_min_motion_frames": int(params.klt_candidate_accum_min_motion_frames),
            "klt_candidate_accum_min_progression_frames": int(params.klt_candidate_accum_min_progression_frames),
            "klt_candidate_accum_min_motion_toward_score": float(params.klt_candidate_accum_min_motion_toward_score),
            "klt_candidate_accum_max_roi_distance_px": float(params.klt_candidate_accum_max_roi_distance_px),
            "klt_candidate_accum_ready_hold_frames": int(params.klt_candidate_accum_ready_hold_frames),
            "confirm_requires_ankle": bool(params.confirm_requires_ankle),
            "candidate_score_thr": float(params.candidate_score_thr),
            "proxy_start_max_age_frames": int(params.proxy_start_max_age_frames),
        },
        "candidate_definition": "overlap_or_(near_roi_boundary_and_moving_toward_roi)_or_klt_upper_head_boundary_continuity",
        "confirmed_intrusion_definition": "ankle_keypoint_confirmed_or_klt_hard_case_confirm",
    }
