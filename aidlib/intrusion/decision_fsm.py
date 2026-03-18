from __future__ import annotations

import csv
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
    score: float
    score_factors: dict[str, Any]
    reasons: list[str]


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
    ) -> dict[str, Any]:
        if row is not None:
            self.source_id = row.source_id
            self.last_seen_frame = frame_num
            if row.mode == "real":
                self.last_real_frame = frame_num
            if row.has_valid_bbox:
                self.last_bbox_xyxy = list(row.bbox_xyxy)

        prev_state = self.state
        continuity_id = self.active_event_id or f"track-{self.track_id}"
        candidate_seen = bool(evidence and (evidence.real_candidate or evidence.proxy_candidate))
        confirm_seen = bool(pose_result.ankle_in_roi)
        transitions: list[str] = []
        reasons = list(evidence.reasons if evidence is not None else [])

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
                reasons.append("ankle_in_roi")
            elif candidate_seen:
                self.confirm_streak = 0

            if self.confirm_streak >= max(1, int(params.confirm_enter_n)):
                self.state = STATE_IN_CONFIRMED
                self.exit_streak = 0
                self.grace_left = max(0, int(params.grace_frames))
                transitions.append(f"{STATE_CANDIDATE}->{STATE_IN_CONFIRMED}")
                event_type = "in_confirmed"
            elif self.exit_streak >= max(1, int(params.exit_n)):
                continuity_id = self.active_event_id or continuity_id
                self.state = STATE_OUT
                transitions.append(f"{STATE_CANDIDATE}->{STATE_OUT}")
                event_type = "exit"
                self._clear_event()

        if self.state == STATE_IN_CONFIRMED:
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

        bbox_xyxy = list(self.last_bbox_xyxy or (evidence.bbox_xyxy if evidence is not None else []))
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
                "ankle_confirm": bool(confirm_seen),
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
                "score": round(float(evidence.score), 4) if evidence is not None else 0.0,
            },
            "confirm": {
                "required_ankle": bool(params.confirm_requires_ankle),
                "status": pose_result.status,
                "attempted": bool(pose_result.attempted),
                "ankles": pose_result.ankles,
            },
            "reasons": sorted({str(reason) for reason in reasons if str(reason).strip()}),
            "counters": {
                "candidate_streak": int(self.candidate_streak),
                "confirm_streak": int(self.confirm_streak),
                "exit_streak": int(self.exit_streak),
                "grace_left": int(self.grace_left),
            },
        }


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

    reasons: list[str] = []
    if bottom_center_in_roi:
        reasons.append("bottom_center_in_roi")
    if bbox_overlap >= float(params.candidate_iou_or_overlap_thr):
        reasons.append("bbox_overlap")
    if float(factors["ov"]) >= float(params.candidate_iou_or_overlap_thr):
        reasons.append("lower_band_overlap")
    if score >= float(params.candidate_score_thr):
        reasons.append("candidate_score")

    candidate_geom = bool(reasons)
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
                elif ctx.state == STATE_CANDIDATE and row is not None and row.has_valid_bbox:
                    pose_bbox_xyxy = list(row.bbox_xyxy)

                if pose_probe is None:
                    pose_result = PoseProbeResult.skipped("pose_probe_disabled")
                elif ctx.state == STATE_IN_CONFIRMED:
                    pose_result = PoseProbeResult.skipped("pose_not_needed_in_confirmed")
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
            "confirm_requires_ankle": bool(params.confirm_requires_ankle),
            "candidate_score_thr": float(params.candidate_score_thr),
            "proxy_start_max_age_frames": int(params.proxy_start_max_age_frames),
        },
    }
