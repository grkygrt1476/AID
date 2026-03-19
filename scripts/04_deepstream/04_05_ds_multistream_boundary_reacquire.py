#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    np = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_stage0403_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "04_deepstream" / "04_03_ds_multistream_intrusion.py"
    spec = importlib.util.spec_from_file_location("aid_stage0403_intrusion", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Stage 04.03 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage0403 = load_stage0403_module()

from aidlib.intrusion.decision_fsm import (  # noqa: E402
    STATE_CANDIDATE,
    STATE_IN_CONFIRMED,
    STATE_OUT,
    DecisionParams,
    PoseProbeSettings,
    SidecarRow,
    load_sidecar_rows,
    run_intrusion_decision_pass,
)
from aidlib.intrusion.io import write_json  # noqa: E402
from aidlib.run_utils import common_argparser, dump_run_meta, init_run  # noqa: E402


STAGE = "04_deepstream"
STAGE_STEP = "04.05"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_TEMPLATE = stage0403.DEFAULT_TEMPLATE
DEFAULT_PLUGIN_LIB = stage0403.DEFAULT_PLUGIN_LIB
DEFAULT_POSE_MODEL = stage0403.DEFAULT_POSE_MODEL
DEFAULT_HEAVY_POSE11M = PROJECT_ROOT / "yolo11m-pose.pt"
DEFAULT_HEAVY_POSE11S = PROJECT_ROOT / "yolo11s-pose.pt"
DEFAULT_OUT_BASE = "multistream4_boundary_reacquire"
DEFAULT_CROP_MANIFEST_REL = Path("crop_assets") / "crop_assets_manifest.json"
SOURCE_COUNT = stage0403.SOURCE_COUNT
TILER_ROWS = stage0403.TILER_ROWS
TILER_COLUMNS = stage0403.TILER_COLUMNS
MISSING_EVENT_TYPES = {"candidate_grace", "candidate_lost", "in_grace", "in_lost"}
REACQUIRE_DRAW_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 15, 16]
HEAD_KEYPOINT_INDICES = [0, 1, 2, 3, 4]
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
REAL_ADOPT = "REAL_ADOPT"
REAL_SUPPORT_ONLY = "REAL_SUPPORT_ONLY"
REAL_REJECT = "REAL_REJECT"
SIDECAR_DEBUG_FIELDS = [
    "row_source",
    "real_reacquire_class",
    "predicted_anchor_x",
    "predicted_anchor_y",
    "real_anchor_innovation_px",
    "real_height_ratio",
    "real_aspect_ratio_ratio",
    "recent_real_adoptable_count",
    "klt_reliable_chain_length",
]


@dataclass(frozen=True)
class BoundaryReacquireConfig:
    boundary_band_px: float
    memory_frames: int
    pose_imgsz: int
    pose_conf: float
    keypoint_conf: float
    min_box_conf: float
    max_center_dist_norm: float
    min_select_score: float
    real_track_probe_band_scale: float
    headlike_refresh_conf_thr: float
    shoulder_refresh_conf_thr: float
    headlike_refresh_score_thr: float
    shoulder_refresh_score_thr: float
    headlike_support_window_frames: int
    shoulder_support_window_frames: int
    headlike_hold_missing_frames: int
    shoulder_hold_missing_frames: int
    headlike_boundary_band_scale: float
    shoulder_boundary_band_scale: float
    headlike_hold_center_drift_ratio: float
    shoulder_hold_center_drift_ratio: float
    headlike_hold_anchor_drift_ratio: float
    shoulder_hold_anchor_drift_ratio: float
    heavy_enabled: bool
    heavy_trigger_select_score: float
    heavy_trigger_anchor_score: float
    heavy_trigger_box_conf: float
    heavy_boundary_band_scale: float
    kp_hold_protect_min_strength: float
    heavy_override_hold_iou_min: float
    heavy_override_hold_center_score_min: float
    heavy_override_hold_score_gain: float
    heavy_override_hold_anchor_gain: float
    heavy_override_hold_kp_gain: float
    heavy_override_hold_box_conf_min: float


@dataclass(frozen=True)
class CropAssetMeta:
    source_id: int
    clip_label: str
    source_clip_path: Path
    crop_clip_path: Path
    crop_metadata_path: Path
    source_width: int
    source_height: int
    source_fps: float
    source_frame_count: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    roi_polygon_source: tuple[tuple[int, int], ...]
    roi_polygon_crop_local: tuple[tuple[int, int], ...]
    resize_applied: bool
    letterbox_applied: bool
    frame_index_mapping: str


@dataclass(frozen=True)
class CropReacquireSettings:
    model_path: str
    input_size: int
    conf: float
    keypoint_conf: float
    min_box_conf: float
    max_center_dist_norm: float
    min_select_score: float
    label: str = "light"


@dataclass(frozen=True)
class HeavyModelRequest:
    enabled: bool
    mode: str
    model_kind: str
    model_path: Path | None


@dataclass(frozen=True)
class ReacquireDetection:
    bbox_crop_xyxy: list[float]
    bbox_source_xyxy: list[float]
    keypoints_crop_local: list[tuple[float, float, float]]
    keypoints_source: list[tuple[float, float, float]]
    box_conf: float
    kp_score: float
    upper_anchor_xy_source: tuple[float, float] | None
    upper_anchor_source: str
    upper_anchor_kind: str
    upper_anchor_conf: float
    upper_anchor_score: float
    selection_score: float


@dataclass
class BoundaryTrackMemory:
    track_id: int
    last_state: str = STATE_OUT
    last_real_frame: int = -1
    last_real_bbox_xyxy: list[float] = field(default_factory=list)
    last_boundary_distance_px: float = float("inf")
    last_reacquire_frame: int = -1
    last_reacquire_bbox_xyxy: list[float] = field(default_factory=list)
    last_reacquire_keypoints_source: list[tuple[float, float, float]] = field(default_factory=list)
    last_kp_refresh_frame: int = -1
    last_kp_bbox_xyxy: list[float] = field(default_factory=list)
    last_kp_keypoints_source: list[tuple[float, float, float]] = field(default_factory=list)
    last_kp_anchor_xy_source: tuple[float, float] | None = None
    last_kp_anchor_source: str = ""
    last_kp_anchor_kind: str = ""
    last_kp_anchor_conf: float = 0.0
    last_kp_anchor_score: float = 0.0
    last_bbox_size_wh: tuple[float, float] | None = None
    last_anchor_offset_xy: tuple[float, float] | None = None
    kp_missing_streak: int = 0


@dataclass(frozen=True)
class UpperAnchorEvidence:
    point_xy: tuple[float, float]
    source: str
    kind: str
    confidence: float


@dataclass
class KltContinuityTrackState:
    track_id: int
    last_real_frame: int = -1
    last_bbox_xyxy: list[float] = field(default_factory=list)
    last_patch_xyxy: list[float] = field(default_factory=list)
    pose_anchor_source: str = ""
    patch_source: str = "upper_klt_seed"
    features: list[tuple[float, float]] = field(default_factory=list)
    recent_motion_dx: list[float] = field(default_factory=list)
    recent_motion_dy: list[float] = field(default_factory=list)
    rejected_hold_streak: int = 0
    last_valid_anchor_xy: tuple[float, float] | None = None
    last_valid_row_source: str = ""
    klt_reliable_chain_length: int = 0
    recent_real_adoptable_flags: list[int] = field(default_factory=list)
    recent_real_adoptable_frames: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RealKltSeedResult:
    patch_xyxy: list[float] | None
    features: list[tuple[float, float]] | None
    pose_anchor_source: str
    seed_mode: str
    anchor_xy_source: tuple[float, float] | None = None
    anchor_score: float = 0.0


@dataclass(frozen=True)
class RealReacquireDecision:
    classification: str
    stop_reason: str
    predicted_anchor_xy: tuple[float, float] | None
    observed_anchor_xy: tuple[float, float] | None
    anchor_innovation_px: float
    height_ratio: float
    aspect_ratio_ratio: float
    recent_real_adoptable_count: int
    klt_reliable_chain_length: int
    adoptable_now: bool


@dataclass
class RenderSourceContext:
    spec: stage0403.SourceSpec
    overlay: stage0403.OverlaySpec
    artifacts: stage0403.SourceArtifacts
    crop_asset: CropAssetMeta
    cap: Any
    crop_cap: Any
    fps: float
    width: int
    height: int
    frame_num: int
    sidecar_rows_by_frame: dict[int, dict[int, SidecarRow]]
    sidecar_summary: dict[str, Any]
    records_by_frame: dict[int, dict[int, dict[str, Any]]]
    decision_summary: dict[str, Any]
    memories: dict[int, BoundaryTrackMemory] = field(default_factory=dict)
    pose_debug_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    active: bool = True


class CropPoseReacquirer:
    def __init__(self, settings: CropReacquireSettings):
        self.settings = settings
        self._model = None
        self._load_status: str | None = None

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
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._load_status = f"ultralytics_import_failed:{exc.__class__.__name__}"
            return

        try:
            self._model = YOLO(str(model_path_obj))
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._load_status = f"pose_model_load_failed:{exc.__class__.__name__}"
            return

        self._load_status = "ready"

    def detect(
        self,
        *,
        crop_frame: np.ndarray,
        expected_box_crop_xyxy: list[float] | None,
        crop_asset: CropAssetMeta,
    ) -> ReacquireDetection | None:
        self._ensure_model()
        if self._model is None or np is None:
            return None

        try:
            results = self._model.predict(
                source=crop_frame,
                imgsz=int(self.settings.input_size),
                conf=float(self.settings.conf),
                verbose=False,
                stream=False,
            )
        except Exception:  # pragma: no cover - inference depends on runtime env
            return None

        result = results[0] if results else None
        if result is None:
            return None

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return None

        boxes_xyxy = to_numpy(getattr(boxes, "xyxy", None))
        boxes_conf = to_numpy(getattr(boxes, "conf", None))
        boxes_cls = to_numpy(getattr(boxes, "cls", None))
        keypoints = getattr(result, "keypoints", None)
        keypoints_xy = to_numpy(getattr(keypoints, "xy", None)) if keypoints is not None else None
        keypoints_conf = to_numpy(getattr(keypoints, "conf", None)) if keypoints is not None else None

        if boxes_xyxy is None or boxes_conf is None or boxes_xyxy.ndim != 2 or len(boxes_xyxy) == 0:
            return None

        crop_diag = math.hypot(float(crop_asset.crop_width), float(crop_asset.crop_height))
        expected_box = expected_box_crop_xyxy
        expected_center = bbox_center_xy(expected_box) if expected_box is not None else (crop_asset.crop_width * 0.5, crop_asset.crop_height * 0.5)
        best_detection: ReacquireDetection | None = None
        best_score = -1.0

        for idx in range(len(boxes_xyxy)):
            box_conf = float(boxes_conf.reshape(-1)[idx])
            if box_conf < self.settings.min_box_conf:
                continue
            if boxes_cls is not None and len(boxes_cls.reshape(-1)) > idx:
                cls_id = int(round(float(boxes_cls.reshape(-1)[idx])))
                if cls_id != 0:
                    continue

            box_crop = clamp_box_xyxy(boxes_xyxy[idx].reshape(-1).tolist(), crop_asset.crop_width, crop_asset.crop_height)
            iou_score = bbox_iou(box_crop, expected_box) if expected_box is not None else 0.0
            center_score = clamp01(1.0 - (math.hypot(bbox_center_xy(box_crop)[0] - expected_center[0], bbox_center_xy(box_crop)[1] - expected_center[1]) / max(1.0, crop_diag * self.settings.max_center_dist_norm)))
            kp_points_crop = collect_keypoints_for_detection(
                keypoints_xy=keypoints_xy,
                keypoints_conf=keypoints_conf,
                det_idx=idx,
                crop_asset=crop_asset,
            )
            kp_score = mean_keypoint_confidence(kp_points_crop, self.settings.keypoint_conf)
            kp_source = crop_local_keypoints_to_source(kp_points_crop, crop_asset)
            upper_anchor = resolve_upper_anchor_from_points(
                points_source=kp_source,
                bbox_source_xyxy=crop_local_box_to_source(box_crop, crop_asset),
                min_conf=self.settings.keypoint_conf,
            )
            upper_anchor_score = compute_upper_anchor_score(
                anchor=upper_anchor,
                kp_score=kp_score,
                box_conf=box_conf,
            )
            select_score = (
                (0.33 * box_conf)
                + (0.27 * iou_score)
                + (0.16 * center_score)
                + (0.10 * kp_score)
                + (0.14 * upper_anchor_score)
            )
            if select_score < self.settings.min_select_score:
                continue

            box_source = crop_local_box_to_source(box_crop, crop_asset)
            detection = ReacquireDetection(
                bbox_crop_xyxy=box_crop,
                bbox_source_xyxy=box_source,
                keypoints_crop_local=kp_points_crop,
                keypoints_source=kp_source,
                box_conf=box_conf,
                kp_score=kp_score,
                upper_anchor_xy_source=tuple(upper_anchor.point_xy) if upper_anchor is not None else None,
                upper_anchor_source=upper_anchor.source if upper_anchor is not None else "",
                upper_anchor_kind=upper_anchor.kind if upper_anchor is not None else "",
                upper_anchor_conf=float(upper_anchor.confidence) if upper_anchor is not None else 0.0,
                upper_anchor_score=upper_anchor_score,
                selection_score=select_score,
            )
            if select_score > best_score:
                best_detection = detection
                best_score = select_score

        return best_detection


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_root=str(DEFAULT_LOG_ROOT))
    parser.add_argument(
        "--inputs",
        action="append",
        nargs="+",
        default=None,
        help="Exactly 4 input videos. Can be passed once with 4 paths or repeated.",
    )
    parser.add_argument("--ds_config_template", default=str(DEFAULT_TEMPLATE), help="DeepStream app config template path")
    parser.add_argument("--plugin_lib", default=str(DEFAULT_PLUGIN_LIB), help="Path to the Stage 04.03 intrusion export plugin library.")
    parser.add_argument("--pose_model", default=str(DEFAULT_POSE_MODEL), help="Pose model path reused for 04.03 truth semantics and crop-based boundary reacquire.")
    parser.add_argument("--candidate_enter_n", type=int, default=2, help="Consecutive candidate frames required to enter CANDIDATE.")
    parser.add_argument("--confirm_enter_n", type=int, default=1, help="Consecutive ankle-in-ROI frames required to enter IN_CONFIRMED.")
    parser.add_argument("--exit_n", type=int, default=5, help="Sustained no-evidence frames required to return to OUT after grace.")
    parser.add_argument("--grace_frames", type=int, default=-1, help="Evidence-loss grace in frames. Default is derived from configs/intrusion/mvp_v1.yaml and source FPS.")
    parser.add_argument("--candidate_iou_or_overlap_thr", type=float, default=0.05, help="ROI overlap threshold used for weak candidate geometry.")
    parser.add_argument("--out_dir", default="", help="Alias for --out_root; output root directory for this stage.")
    parser.add_argument("--crop_manifest", default="", help="Optional crop asset manifest path from Stage 04.05a. Defaults to the same run_ts/output root.")
    parser.add_argument("--reacquire_boundary_band_px", type=float, default=72.0, help="Boundary-near band in source pixels where crop-based reacquire is allowed.")
    parser.add_argument("--reacquire_memory_frames", type=int, default=90, help="Recent-frame window where boundary-near missing tracks remain eligible for crop-based reacquire.")
    parser.add_argument("--reacquire_pose_imgsz", type=int, default=960, help="Inference size used for crop-based pose reacquire.")
    parser.add_argument("--reacquire_pose_conf", type=float, default=0.20, help="Minimum pose detector confidence for crop-based reacquire.")
    parser.add_argument("--reacquire_keypoint_conf", type=float, default=0.30, help="Minimum keypoint confidence to draw projected crop-based pose points.")
    parser.add_argument(
        "--reacquire_heavy_model",
        choices=["off", "pose11m", "pose11s", "custom_pose"],
        default="off",
        help="Optional conditional heavy-model second pass on the ROI crop. Runs only when the ordinary crop reacquire is missing or weak.",
    )
    parser.add_argument(
        "--reacquire_heavy_model_path",
        default="",
        help="Optional explicit weights path overriding the built-in path for pose11m/pose11s/custom_pose.",
    )
    parser.add_argument(
        "--reacquire_heavy_pose_imgsz",
        type=int,
        default=1280,
        help="Inference size used for the conditional heavy crop pose pass.",
    )
    parser.add_argument(
        "--reacquire_heavy_pose_conf",
        type=float,
        default=0.16,
        help="Minimum detector confidence used for the conditional heavy crop pose pass.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs, crop manifest, and planned commands without running DeepStream, decision, or final render.")
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def to_numpy(value: Any) -> np.ndarray | None:
    if np is None or value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def build_reacquire_config(args: argparse.Namespace) -> BoundaryReacquireConfig:
    memory_frames = max(1, int(args.reacquire_memory_frames))
    return BoundaryReacquireConfig(
        boundary_band_px=max(1.0, float(args.reacquire_boundary_band_px)),
        memory_frames=memory_frames,
        pose_imgsz=max(256, int(args.reacquire_pose_imgsz)),
        pose_conf=max(0.01, float(args.reacquire_pose_conf)),
        keypoint_conf=max(0.0, float(args.reacquire_keypoint_conf)),
        min_box_conf=max(0.01, float(args.reacquire_pose_conf)),
        max_center_dist_norm=0.60,
        min_select_score=0.24,
        real_track_probe_band_scale=1.45,
        headlike_refresh_conf_thr=0.50,
        shoulder_refresh_conf_thr=0.44,
        headlike_refresh_score_thr=0.60,
        shoulder_refresh_score_thr=0.52,
        headlike_support_window_frames=max(memory_frames, 180),
        shoulder_support_window_frames=max(memory_frames, 120),
        headlike_hold_missing_frames=max(36, int(round(memory_frames * 0.60))),
        shoulder_hold_missing_frames=max(18, int(round(memory_frames * 0.30))),
        headlike_boundary_band_scale=1.45,
        shoulder_boundary_band_scale=1.20,
        headlike_hold_center_drift_ratio=2.25,
        shoulder_hold_center_drift_ratio=1.70,
        headlike_hold_anchor_drift_ratio=1.70,
        shoulder_hold_anchor_drift_ratio=1.30,
        heavy_enabled=str(getattr(args, "reacquire_heavy_model", "off")).strip().lower() != "off",
        heavy_trigger_select_score=0.60,
        heavy_trigger_anchor_score=0.54,
        heavy_trigger_box_conf=0.34,
        heavy_boundary_band_scale=1.25,
        kp_hold_protect_min_strength=0.56,
        heavy_override_hold_iou_min=0.22,
        heavy_override_hold_center_score_min=0.48,
        heavy_override_hold_score_gain=0.10,
        heavy_override_hold_anchor_gain=0.10,
        heavy_override_hold_kp_gain=0.12,
        heavy_override_hold_box_conf_min=0.38,
    )


def require_render_deps() -> None:
    stage0403.require_render_deps()


def resolve_heavy_model_request(args: argparse.Namespace) -> HeavyModelRequest:
    mode = str(getattr(args, "reacquire_heavy_model", "off")).strip().lower()
    explicit_path = str(getattr(args, "reacquire_heavy_model_path", "")).strip()
    if mode == "off":
        return HeavyModelRequest(enabled=False, mode="off", model_kind="none", model_path=None)
    if mode == "pose11m":
        path = stage0403.project_path(explicit_path) if explicit_path else DEFAULT_HEAVY_POSE11M
        return HeavyModelRequest(enabled=True, mode=mode, model_kind="pose", model_path=path)
    if mode == "pose11s":
        path = stage0403.project_path(explicit_path) if explicit_path else DEFAULT_HEAVY_POSE11S
        return HeavyModelRequest(enabled=True, mode=mode, model_kind="pose", model_path=path)
    if mode == "custom_pose":
        if not explicit_path:
            raise SystemExit("--reacquire_heavy_model_path is required when --reacquire_heavy_model=custom_pose")
        return HeavyModelRequest(enabled=True, mode=mode, model_kind="pose", model_path=stage0403.project_path(explicit_path))
    raise SystemExit(f"Unsupported heavy model mode: {mode}")


def bbox_center_xy(box_xyxy: list[float] | tuple[float, float, float, float] | None) -> tuple[float, float]:
    if box_xyxy is None:
        return (0.0, 0.0)
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def bbox_iou(box_a: list[float] | tuple[float, float, float, float] | None, box_b: list[float] | tuple[float, float, float, float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_bbox_upper_proxy(box_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (0.5 * (x1 + x2), y1 + ((y2 - y1) * 0.2))


def compute_boundary_distance(box_xyxy: list[float], overlay: stage0403.OverlaySpec) -> float:
    if cv2 is None or np is None or len(overlay.roi_polygon_source) < 3:
        return float("inf")
    upper_proxy = compute_bbox_upper_proxy(box_xyxy)
    contour = np.asarray(overlay.roi_polygon_source, dtype=np.float32)
    return float(cv2.pointPolygonTest(contour, (float(upper_proxy[0]), float(upper_proxy[1])), True))


def clamp_box_xyxy(box_xyxy: list[float], frame_w: int, frame_h: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    x1 = max(0.0, min(float(frame_w), x1))
    y1 = max(0.0, min(float(frame_h), y1))
    x2 = max(0.0, min(float(frame_w), x2))
    y2 = max(0.0, min(float(frame_h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def source_box_to_crop_local(box_xyxy: list[float], crop_asset: CropAssetMeta) -> list[float] | None:
    x1, y1, x2, y2 = map(float, box_xyxy)
    local_box = [
        x1 - float(crop_asset.crop_x),
        y1 - float(crop_asset.crop_y),
        x2 - float(crop_asset.crop_x),
        y2 - float(crop_asset.crop_y),
    ]
    local_box = clamp_box_xyxy(local_box, crop_asset.crop_width, crop_asset.crop_height)
    if local_box[2] <= local_box[0] or local_box[3] <= local_box[1]:
        return None
    return local_box


def crop_local_box_to_source(box_xyxy: list[float], crop_asset: CropAssetMeta) -> list[float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    return [
        x1 + float(crop_asset.crop_x),
        y1 + float(crop_asset.crop_y),
        x2 + float(crop_asset.crop_x),
        y2 + float(crop_asset.crop_y),
    ]


def crop_local_keypoints_to_source(
    keypoints_crop_local: list[tuple[float, float, float]],
    crop_asset: CropAssetMeta,
) -> list[tuple[float, float, float]]:
    return [
        (float(x) + float(crop_asset.crop_x), float(y) + float(crop_asset.crop_y), float(conf))
        for (x, y, conf) in keypoints_crop_local
    ]


def collect_keypoints_for_detection(
    *,
    keypoints_xy: np.ndarray | None,
    keypoints_conf: np.ndarray | None,
    det_idx: int,
    crop_asset: CropAssetMeta,
) -> list[tuple[float, float, float]]:
    if np is None or keypoints_xy is None or keypoints_xy.ndim != 3 or det_idx >= keypoints_xy.shape[0]:
        return []
    det_xy = keypoints_xy[det_idx]
    if keypoints_conf is None:
        det_conf = np.ones(det_xy.shape[0], dtype=np.float32)
    elif keypoints_conf.ndim == 2 and det_idx < keypoints_conf.shape[0]:
        det_conf = keypoints_conf[det_idx].reshape(-1)
    elif keypoints_conf.ndim == 1:
        det_conf = keypoints_conf.reshape(-1)
    else:
        det_conf = np.ones(det_xy.shape[0], dtype=np.float32)

    points: list[tuple[float, float, float]] = []
    for kp_idx, point in enumerate(det_xy):
        conf = float(det_conf[kp_idx]) if kp_idx < len(det_conf) else 1.0
        x = float(point[0])
        y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        x = max(0.0, min(float(crop_asset.crop_width - 1), x))
        y = max(0.0, min(float(crop_asset.crop_height - 1), y))
        points.append((x, y, conf))
    return points


def mean_keypoint_confidence(points: list[tuple[float, float, float]], min_conf: float) -> float:
    good = [float(conf) for (_, _, conf) in points if float(conf) >= min_conf]
    if not good:
        return 0.0
    return sum(good) / max(1, len(good))


def get_valid_detection_point(
    points: list[tuple[float, float, float]],
    idx: int,
    min_conf: float,
) -> tuple[float, float] | None:
    if idx >= len(points):
        return None
    x, y, conf = points[idx]
    if float(conf) < float(min_conf):
        return None
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None
    return (float(x), float(y))


def mean_valid_detection_points(
    points: list[tuple[float, float, float]],
    indices: list[int],
    min_conf: float,
) -> tuple[float, float] | None:
    valid = [get_valid_detection_point(points, idx, min_conf) for idx in indices]
    valid = [point for point in valid if point is not None]
    if not valid:
        return None
    x = sum(point[0] for point in valid) / max(1, len(valid))
    y = sum(point[1] for point in valid) / max(1, len(valid))
    return (float(x), float(y))


def clamp_point_to_box(
    point_xy: tuple[float, float],
    box_xyxy: list[float] | tuple[float, float, float, float],
) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (
        max(x1, min(x2, float(point_xy[0]))),
        max(y1, min(y2, float(point_xy[1]))),
    )


def resolve_upper_anchor_from_points(
    *,
    points_source: list[tuple[float, float, float]],
    bbox_source_xyxy: list[float] | tuple[float, float, float, float],
    min_conf: float,
) -> UpperAnchorEvidence | None:
    nose = get_valid_detection_point(points_source, 0, min_conf)
    eye_center = mean_valid_detection_points(points_source, [1, 2], min_conf)
    ear_center = mean_valid_detection_points(points_source, [3, 4], min_conf)
    head_mean = mean_valid_detection_points(points_source, HEAD_KEYPOINT_INDICES, min_conf)
    left_shoulder = get_valid_detection_point(points_source, LEFT_SHOULDER_IDX, min_conf)
    right_shoulder = get_valid_detection_point(points_source, RIGHT_SHOULDER_IDX, min_conf)

    candidates: list[UpperAnchorEvidence] = []
    if nose is not None:
        candidates.append(
            UpperAnchorEvidence(
                point_xy=clamp_point_to_box(nose, bbox_source_xyxy),
                source="head_like_keypoint",
                kind="nose",
                confidence=float(points_source[0][2]),
            )
        )
    if eye_center is not None:
        eye_confs = [float(points_source[idx][2]) for idx in [1, 2] if idx < len(points_source)]
        candidates.append(
            UpperAnchorEvidence(
                point_xy=clamp_point_to_box(eye_center, bbox_source_xyxy),
                source="head_like_keypoint",
                kind="eye_center",
                confidence=sum(eye_confs) / max(1, len(eye_confs)),
            )
        )
    if ear_center is not None:
        ear_confs = [float(points_source[idx][2]) for idx in [3, 4] if idx < len(points_source)]
        candidates.append(
            UpperAnchorEvidence(
                point_xy=clamp_point_to_box(ear_center, bbox_source_xyxy),
                source="head_like_keypoint",
                kind="ear_center",
                confidence=sum(ear_confs) / max(1, len(ear_confs)),
            )
        )
    if head_mean is not None:
        head_confs = [float(points_source[idx][2]) for idx in HEAD_KEYPOINT_INDICES if idx < len(points_source)]
        candidates.append(
            UpperAnchorEvidence(
                point_xy=clamp_point_to_box(head_mean, bbox_source_xyxy),
                source="head_like_keypoint",
                kind="head_mean",
                confidence=sum(head_confs) / max(1, len(head_confs)),
            )
        )
    if candidates:
        return max(candidates, key=lambda item: item.confidence)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_mid = (0.5 * (left_shoulder[0] + right_shoulder[0]), 0.5 * (left_shoulder[1] + right_shoulder[1]))
        shoulder_conf = 0.5 * (float(points_source[LEFT_SHOULDER_IDX][2]) + float(points_source[RIGHT_SHOULDER_IDX][2]))
        return UpperAnchorEvidence(
            point_xy=clamp_point_to_box(shoulder_mid, bbox_source_xyxy),
            source="shoulder_fallback",
            kind="shoulder_center",
            confidence=shoulder_conf,
        )
    if left_shoulder is not None:
        return UpperAnchorEvidence(
            point_xy=clamp_point_to_box(left_shoulder, bbox_source_xyxy),
            source="shoulder_fallback",
            kind="left_shoulder",
            confidence=float(points_source[LEFT_SHOULDER_IDX][2]),
        )
    if right_shoulder is not None:
        return UpperAnchorEvidence(
            point_xy=clamp_point_to_box(right_shoulder, bbox_source_xyxy),
            source="shoulder_fallback",
            kind="right_shoulder",
            confidence=float(points_source[RIGHT_SHOULDER_IDX][2]),
        )
    return None


def is_headlike_source(anchor_source: str) -> bool:
    return anchor_source.startswith("head_like_keypoint")


def is_shoulder_source(anchor_source: str) -> bool:
    return anchor_source.startswith("shoulder_fallback")


def compute_upper_anchor_score(
    *,
    anchor: UpperAnchorEvidence | None,
    kp_score: float,
    box_conf: float,
) -> float:
    if anchor is None:
        return 0.0
    base = (0.50 * clamp01(anchor.confidence)) + (0.30 * clamp01(kp_score)) + (0.20 * clamp01(box_conf))
    if is_headlike_source(anchor.source):
        return clamp01(base + 0.18)
    if is_shoulder_source(anchor.source):
        return clamp01(base + 0.08)
    return clamp01(base)


def has_strong_kp_refresh(memory: BoundaryTrackMemory, cfg: BoundaryReacquireConfig) -> bool:
    if is_headlike_source(memory.last_kp_anchor_source):
        return (
            float(memory.last_kp_anchor_conf) >= cfg.headlike_refresh_conf_thr
            and float(memory.last_kp_anchor_score) >= cfg.headlike_refresh_score_thr
        )
    if is_shoulder_source(memory.last_kp_anchor_source):
        return (
            float(memory.last_kp_anchor_conf) >= cfg.shoulder_refresh_conf_thr
            and float(memory.last_kp_anchor_score) >= cfg.shoulder_refresh_score_thr
        )
    return False


def support_window_frames(memory: BoundaryTrackMemory, cfg: BoundaryReacquireConfig) -> int:
    if has_strong_kp_refresh(memory, cfg):
        if is_headlike_source(memory.last_kp_anchor_source):
            return cfg.headlike_support_window_frames
        if is_shoulder_source(memory.last_kp_anchor_source):
            return cfg.shoulder_support_window_frames
    return cfg.memory_frames


def boundary_band_scale_for_memory(memory: BoundaryTrackMemory, cfg: BoundaryReacquireConfig) -> float:
    if has_strong_kp_refresh(memory, cfg):
        if is_headlike_source(memory.last_kp_anchor_source):
            return cfg.headlike_boundary_band_scale
        if is_shoulder_source(memory.last_kp_anchor_source):
            return cfg.shoulder_boundary_band_scale
    return 1.0


def hold_missing_limit(memory: BoundaryTrackMemory, cfg: BoundaryReacquireConfig) -> int:
    if has_strong_kp_refresh(memory, cfg):
        if is_headlike_source(memory.last_kp_anchor_source):
            return cfg.headlike_hold_missing_frames
        if is_shoulder_source(memory.last_kp_anchor_source):
            return cfg.shoulder_hold_missing_frames
    return 0


def format_sidecar_float(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def box_has_area(box_xyxy: list[float] | tuple[float, float, float, float] | None) -> bool:
    if box_xyxy is None:
        return False
    if len(box_xyxy) != 4:
        return False
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (x2 - x1) > 1.0 and (y2 - y1) > 1.0


def shift_box_xyxy(
    box_xyxy: list[float] | tuple[float, float, float, float],
    dx: float,
    dy: float,
    frame_w: int,
    frame_h: int,
) -> list[float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    return clamp_box_xyxy([x1 + float(dx), y1 + float(dy), x2 + float(dx), y2 + float(dy)], frame_w, frame_h)


def expand_box_around_center(
    box_xyxy: list[float] | tuple[float, float, float, float],
    scale: float,
    frame_w: int,
    frame_h: int,
) -> list[float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    cx, cy = bbox_center_xy(box_xyxy)
    half_w = max(4.0, (x2 - x1) * 0.5 * float(scale))
    half_h = max(4.0, (y2 - y1) * 0.5 * float(scale))
    return clamp_box_xyxy([cx - half_w, cy - half_h, cx + half_w, cy + half_h], frame_w, frame_h)


def compute_point_boundary_distance(point_xy: tuple[float, float], overlay: stage0403.OverlaySpec) -> float:
    if cv2 is None or np is None or len(overlay.roi_polygon_source) < 3:
        return float("inf")
    contour = np.asarray(overlay.roi_polygon_source, dtype=np.float32)
    return float(cv2.pointPolygonTest(contour, (float(point_xy[0]), float(point_xy[1])), True))


def is_boundary_relevant_for_klt(
    *,
    bbox_xyxy: list[float],
    patch_xyxy: list[float] | None,
    overlay: stage0403.OverlaySpec,
    params: DecisionParams,
) -> bool:
    band_px = max(float(params.klt_confirm_boundary_max_distance_px), float(params.cand_distance_sustain_px)) * 2.0
    if patch_xyxy is not None and box_has_area(patch_xyxy):
        if abs(compute_point_boundary_distance(bbox_center_xy(patch_xyxy), overlay)) <= band_px:
            return True
    return abs(compute_boundary_distance(bbox_xyxy, overlay)) <= band_px


def klt_proxy_bonus_age_allowed(
    *,
    miss_frames: int,
    state: KltContinuityTrackState,
    tracked_points: int,
    params: DecisionParams,
    min_good_points: int,
) -> bool:
    base_age = max(1, int(params.klt_continuity_max_proxy_age_frames))
    bonus_age = max(base_age, int(params.klt_continuity_bonus_proxy_age_frames))
    if miss_frames <= base_age:
        return True
    if miss_frames > bonus_age:
        return False
    anchor_valid = bool(
        is_headlike_source(state.pose_anchor_source) or is_shoulder_source(state.pose_anchor_source)
    )
    tracked_points_ok = bool(
        tracked_points >= max(int(min_good_points), int(params.klt_continuity_bonus_min_tracked_points))
    )
    recent_real_ok = bool(
        miss_frames <= max(base_age, int(params.klt_continuity_bonus_recent_real_max_frames))
    )
    return bool(anchor_valid and tracked_points_ok and recent_real_ok)


def make_pose_seed_crop_asset(
    *,
    source_id: int,
    clip_label: str,
    frame_shape: tuple[int, int],
    crop_xyxy: list[float],
    source_clip_path: Path,
) -> CropAssetMeta:
    frame_h, frame_w = frame_shape
    crop_x1, crop_y1, crop_x2, crop_y2 = map(int, map(round, crop_xyxy))
    return CropAssetMeta(
        source_id=int(source_id),
        clip_label=str(clip_label),
        source_clip_path=source_clip_path,
        crop_clip_path=source_clip_path,
        crop_metadata_path=source_clip_path,
        source_width=int(frame_w),
        source_height=int(frame_h),
        source_fps=0.0,
        source_frame_count=0,
        crop_x=max(0, crop_x1),
        crop_y=max(0, crop_y1),
        crop_width=max(1, crop_x2 - crop_x1),
        crop_height=max(1, crop_y2 - crop_y1),
        roi_polygon_source=tuple(),
        roi_polygon_crop_local=tuple(),
        resize_applied=False,
        letterbox_applied=False,
        frame_index_mapping="source_frame_index",
    )


def build_pose_seed_crop(
    *,
    frame: np.ndarray,
    bbox_xyxy: list[float],
    source_id: int,
    clip_label: str,
    source_clip_path: Path,
) -> tuple[np.ndarray | None, CropAssetMeta | None, list[float] | None]:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    crop_xyxy = clamp_box_xyxy(
        [
            x1 - (bbox_w * 0.24),
            y1 - (bbox_h * 0.22),
            x2 + (bbox_w * 0.24),
            y2 + (bbox_h * 0.14),
        ],
        frame_w,
        frame_h,
    )
    if not box_has_area(crop_xyxy):
        return None, None, None
    crop_x1, crop_y1, crop_x2, crop_y2 = map(int, map(round, crop_xyxy))
    crop_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_frame.size == 0:
        return None, None, None
    crop_asset = make_pose_seed_crop_asset(
        source_id=source_id,
        clip_label=clip_label,
        frame_shape=(frame_h, frame_w),
        crop_xyxy=crop_xyxy,
        source_clip_path=source_clip_path,
    )
    expected_box = source_box_to_crop_local(list(bbox_xyxy), crop_asset)
    return crop_frame, crop_asset, expected_box


def build_upper_patch_xyxy(
    *,
    detection: ReacquireDetection,
    frame_w: int,
    frame_h: int,
) -> list[float] | None:
    if detection.upper_anchor_xy_source is None:
        return None
    bbox_xyxy = list(detection.bbox_source_xyxy)
    bbox_w = max(1.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
    bbox_h = max(1.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
    if is_headlike_source(detection.upper_anchor_source):
        patch_w = max(16.0, bbox_w * 0.46)
        patch_h = max(16.0, bbox_h * 0.26)
    elif is_shoulder_source(detection.upper_anchor_source):
        patch_w = max(18.0, bbox_w * 0.54)
        patch_h = max(18.0, bbox_h * 0.30)
    else:
        return None
    anchor_x, anchor_y = map(float, detection.upper_anchor_xy_source)
    patch_xyxy = clamp_box_xyxy(
        [
            anchor_x - (patch_w * 0.5),
            anchor_y - (patch_h * 0.5),
            anchor_x + (patch_w * 0.5),
            anchor_y + (patch_h * 0.5),
        ],
        frame_w,
        frame_h,
    )
    if not box_has_area(patch_xyxy):
        return None
    return patch_xyxy


def seed_features_in_patch(
    *,
    gray: np.ndarray,
    patch_xyxy: list[float],
    min_good_points: int,
) -> tuple[list[tuple[float, float]], list[float]]:
    if cv2 is None or np is None or gray.size == 0 or not box_has_area(patch_xyxy):
        return [], list(patch_xyxy)
    frame_h, frame_w = gray.shape[:2]
    for scale in (1.0, 1.12, 1.25):
        attempt_box = expand_box_around_center(patch_xyxy, scale, frame_w, frame_h)
        if not box_has_area(attempt_box):
            continue
        x1, y1, x2, y2 = map(int, map(round, attempt_box))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            continue
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=max(40, int(min_good_points) * 6),
            qualityLevel=0.004,
            minDistance=3.0,
            mask=None,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )
        if corners is None:
            continue
        features = [(float(point[0][0]) + float(x1), float(point[0][1]) + float(y1)) for point in corners]
        if len(features) >= max(1, int(min_good_points)):
            return features, attempt_box
    return [], list(patch_xyxy)


def set_sidecar_bbox_fields(row: dict[str, str], bbox_xyxy: list[float]) -> None:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    row["proxy_left"] = format_sidecar_float(x1)
    row["proxy_top"] = format_sidecar_float(y1)
    row["proxy_width"] = format_sidecar_float(max(0.0, x2 - x1))
    row["proxy_height"] = format_sidecar_float(max(0.0, y2 - y1))


def set_sidecar_patch_fields(row: dict[str, str], patch_xyxy: list[float] | None) -> None:
    if patch_xyxy is None or not box_has_area(patch_xyxy):
        row["patch_left"] = "0"
        row["patch_top"] = "0"
        row["patch_width"] = "0"
        row["patch_height"] = "0"
        return
    x1, y1, x2, y2 = map(float, patch_xyxy)
    row["patch_left"] = format_sidecar_float(x1)
    row["patch_top"] = format_sidecar_float(y1)
    row["patch_width"] = format_sidecar_float(max(0.0, x2 - x1))
    row["patch_height"] = format_sidecar_float(max(0.0, y2 - y1))


def format_pose_anchor_source(anchor_source: str, anchor_kind: str) -> str:
    source = str(anchor_source).strip()
    kind = str(anchor_kind).strip()
    if source and kind:
        return f"{source}:{kind}"
    return source or kind


def update_real_row_with_klt_seed(
    *,
    row: dict[str, str],
    patch_xyxy: list[float],
    pose_anchor_source: str,
    seed_feature_count: int,
) -> None:
    set_sidecar_patch_fields(row, patch_xyxy)
    row["patch_source"] = "upper_klt_seed"
    row["pose_anchor_source"] = str(pose_anchor_source)
    row["tracked_points"] = str(max(0, int(seed_feature_count)))
    row["flow_dx"] = "0"
    row["flow_dy"] = "0"
    row["flow_mag"] = "0"


def robust_trimmed_component(values: list[float]) -> float:
    if np is None or not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    median = float(np.median(arr))
    abs_dev = np.abs(arr - median)
    mad = float(np.median(abs_dev))
    if mad > 1e-6:
        keep_mask = abs_dev <= max(1.0, 2.5 * mad)
    else:
        lo = float(np.percentile(arr, 20))
        hi = float(np.percentile(arr, 80))
        keep_mask = (arr >= lo) & (arr <= hi)
    kept = arr[keep_mask]
    if kept.size == 0:
        kept = arr
    return float(np.median(kept))


def update_motion_history(state: KltContinuityTrackState, dx: float, dy: float, max_len: int = 5) -> None:
    state.recent_motion_dx.append(float(dx))
    state.recent_motion_dy.append(float(dy))
    if len(state.recent_motion_dx) > max_len:
        state.recent_motion_dx = state.recent_motion_dx[-max_len:]
    if len(state.recent_motion_dy) > max_len:
        state.recent_motion_dy = state.recent_motion_dy[-max_len:]


def recent_motion_reference(state: KltContinuityTrackState) -> tuple[float, float] | None:
    if not state.recent_motion_dx or not state.recent_motion_dy:
        return None
    return (
        robust_trimmed_component(list(state.recent_motion_dx)),
        robust_trimmed_component(list(state.recent_motion_dy)),
    )


def bbox_width_height(box_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def bbox_aspect_ratio(box_xyxy: list[float] | tuple[float, float, float, float]) -> float:
    width, height = bbox_width_height(box_xyxy)
    return float(width / max(1.0, height))


def prune_recent_real_adoptability(
    state: KltContinuityTrackState,
    *,
    frame_num: int,
    max_gap_frames: int = 6,
    max_obs: int = 3,
) -> None:
    if state.recent_real_adoptable_frames:
        gap = int(frame_num) - int(state.recent_real_adoptable_frames[-1])
        if gap > max(1, int(max_gap_frames)):
            state.recent_real_adoptable_flags = []
            state.recent_real_adoptable_frames = []
    if len(state.recent_real_adoptable_flags) > max_obs:
        state.recent_real_adoptable_flags = state.recent_real_adoptable_flags[-max_obs:]
    if len(state.recent_real_adoptable_frames) > max_obs:
        state.recent_real_adoptable_frames = state.recent_real_adoptable_frames[-max_obs:]


def record_recent_real_adoptability(
    state: KltContinuityTrackState,
    *,
    frame_num: int,
    adoptable: bool,
    max_obs: int = 3,
) -> None:
    prune_recent_real_adoptability(state, frame_num=frame_num, max_obs=max_obs)
    state.recent_real_adoptable_frames.append(int(frame_num))
    state.recent_real_adoptable_flags.append(1 if adoptable else 0)
    if len(state.recent_real_adoptable_flags) > max_obs:
        state.recent_real_adoptable_flags = state.recent_real_adoptable_flags[-max_obs:]
    if len(state.recent_real_adoptable_frames) > max_obs:
        state.recent_real_adoptable_frames = state.recent_real_adoptable_frames[-max_obs:]


def reset_recent_real_adoptability(state: KltContinuityTrackState) -> None:
    state.recent_real_adoptable_flags = []
    state.recent_real_adoptable_frames = []


def sidecar_row_source_default(mode: str) -> str:
    mode_token = str(mode).strip()
    if mode_token in {"real", "proxy", "frozen_hold"}:
        return mode_token
    return mode_token or "unknown"


def annotate_sidecar_debug_fields(
    row: dict[str, str],
    *,
    row_source: str = "",
    real_reacquire_class: str = "",
    predicted_anchor_xy: tuple[float, float] | None = None,
    anchor_innovation_px: float | None = None,
    height_ratio: float | None = None,
    aspect_ratio_ratio: float | None = None,
    recent_real_adoptable_count: int | None = None,
    klt_reliable_chain_length: int | None = None,
) -> None:
    row["row_source"] = str(row_source or sidecar_row_source_default(str(row.get("mode", "")).strip()))
    row["real_reacquire_class"] = str(real_reacquire_class)
    if predicted_anchor_xy is not None:
        row["predicted_anchor_x"] = format_sidecar_float(predicted_anchor_xy[0], digits=2)
        row["predicted_anchor_y"] = format_sidecar_float(predicted_anchor_xy[1], digits=2)
    else:
        row["predicted_anchor_x"] = ""
        row["predicted_anchor_y"] = ""
    row["real_anchor_innovation_px"] = "" if anchor_innovation_px is None or not math.isfinite(float(anchor_innovation_px)) else format_sidecar_float(anchor_innovation_px, digits=2)
    row["real_height_ratio"] = "" if height_ratio is None or not math.isfinite(float(height_ratio)) else format_sidecar_float(height_ratio, digits=3)
    row["real_aspect_ratio_ratio"] = "" if aspect_ratio_ratio is None or not math.isfinite(float(aspect_ratio_ratio)) else format_sidecar_float(aspect_ratio_ratio, digits=3)
    row["recent_real_adoptable_count"] = "" if recent_real_adoptable_count is None else str(max(0, int(recent_real_adoptable_count)))
    row["klt_reliable_chain_length"] = "" if klt_reliable_chain_length is None else str(max(0, int(klt_reliable_chain_length)))


def klt_state_anchor_xy(state: KltContinuityTrackState) -> tuple[float, float] | None:
    if state.last_valid_anchor_xy is not None:
        return (float(state.last_valid_anchor_xy[0]), float(state.last_valid_anchor_xy[1]))
    if box_has_area(state.last_patch_xyxy):
        return bbox_center_xy(state.last_patch_xyxy)
    if box_has_area(state.last_bbox_xyxy):
        return compute_bbox_upper_proxy(state.last_bbox_xyxy)
    return None


def predict_klt_anchor_xy(state: KltContinuityTrackState) -> tuple[float, float] | None:
    anchor_xy = klt_state_anchor_xy(state)
    if anchor_xy is None:
        return None
    recent_ref = recent_motion_reference(state)
    if recent_ref is None:
        return anchor_xy
    return (
        float(anchor_xy[0]) + float(recent_ref[0]),
        float(anchor_xy[1]) + float(recent_ref[1]),
    )


def klt_chain_is_reliably_active(
    *,
    state: KltContinuityTrackState,
    frame_num: int,
    min_good_points: int,
    params: DecisionParams,
) -> bool:
    miss_frames = int(frame_num) - int(state.last_real_frame)
    max_age = max(
        1,
        int(params.klt_continuity_max_proxy_age_frames),
        int(params.klt_continuity_bonus_proxy_age_frames),
    )
    return bool(
        miss_frames > 0
        and miss_frames <= max_age
        and box_has_area(state.last_bbox_xyxy)
        and box_has_area(state.last_patch_xyxy)
        and bool(state.pose_anchor_source)
        and (
            int(state.klt_reliable_chain_length) > 0
            or str(state.last_valid_row_source).strip() in {"proxy", "frozen_hold"}
        )
        and len(state.features) >= max(1, int(min_good_points) // 2)
    )


def classify_returning_real_against_klt_chain(
    *,
    frame_num: int,
    row: SidecarRow,
    state: KltContinuityTrackState,
    seed_result: RealKltSeedResult,
    overlay: stage0403.OverlaySpec,
    params: DecisionParams,
) -> RealReacquireDecision:
    prune_recent_real_adoptability(state, frame_num=frame_num)
    prev_bbox_xyxy = list(state.last_bbox_xyxy)
    real_bbox_xyxy = list(row.bbox_xyxy)
    prev_w, prev_h = bbox_width_height(prev_bbox_xyxy)
    real_w, real_h = bbox_width_height(real_bbox_xyxy)
    scale = max(prev_w, prev_h, real_w, real_h)
    predicted_anchor_xy = predict_klt_anchor_xy(state)
    observed_anchor_xy = seed_result.anchor_xy_source
    if observed_anchor_xy is None and box_has_area(seed_result.patch_xyxy):
        observed_anchor_xy = bbox_center_xy(seed_result.patch_xyxy)
    if observed_anchor_xy is None:
        observed_anchor_xy = compute_bbox_upper_proxy(real_bbox_xyxy)

    anchor_innovation_px = float("inf")
    if predicted_anchor_xy is not None and observed_anchor_xy is not None:
        anchor_innovation_px = float(
            math.hypot(
                float(observed_anchor_xy[0]) - float(predicted_anchor_xy[0]),
                float(observed_anchor_xy[1]) - float(predicted_anchor_xy[1]),
            )
        )

    height_ratio = float(real_h / max(1.0, prev_h))
    aspect_ratio_ratio = float(
        bbox_aspect_ratio(real_bbox_xyxy) / max(1e-6, bbox_aspect_ratio(prev_bbox_xyxy))
    )
    iou_score = bbox_iou(real_bbox_xyxy, prev_bbox_xyxy)
    prev_center_xy = bbox_center_xy(prev_bbox_xyxy)
    real_center_xy = bbox_center_xy(real_bbox_xyxy)
    center_dist_px = float(
        math.hypot(
            float(real_center_xy[0]) - float(prev_center_xy[0]),
            float(real_center_xy[1]) - float(prev_center_xy[1]),
        )
    )
    center_consistent = bool(center_dist_px <= max(28.0, scale * 0.35))

    motion_consistent = bool(
        predicted_anchor_xy is not None
        and observed_anchor_xy is not None
        and anchor_innovation_px <= max(18.0, scale * 0.18)
    )
    strong_motion_consistent = bool(
        predicted_anchor_xy is not None
        and observed_anchor_xy is not None
        and anchor_innovation_px <= max(12.0, scale * 0.12)
    )
    motion_reject = bool(
        predicted_anchor_xy is not None
        and observed_anchor_xy is not None
        and anchor_innovation_px > max(42.0, scale * 0.42)
    )

    shape_consistent = bool(
        0.72 <= height_ratio <= 1.38
        and 0.70 <= aspect_ratio_ratio <= 1.40
    )
    shape_reject = bool(
        height_ratio < 0.60
        or height_ratio > 1.70
        or aspect_ratio_ratio < 0.52
        or aspect_ratio_ratio > 1.85
    )

    prev_boundary_dist = float(compute_boundary_distance(prev_bbox_xyxy, overlay))
    real_boundary_dist = float(compute_boundary_distance(real_bbox_xyxy, overlay))
    boundary_band_px = max(
        float(params.klt_confirm_boundary_max_distance_px),
        float(params.cand_distance_sustain_px),
    )
    roi_compatible = bool(
        abs(real_boundary_dist) <= (boundary_band_px * 1.85)
        or real_boundary_dist >= (prev_boundary_dist - max(24.0, prev_h * 0.18))
    )
    roi_reject = bool(
        real_boundary_dist < (prev_boundary_dist - max(52.0, prev_h * 0.36))
        and abs(real_boundary_dist) > (boundary_band_px * 1.95)
    )

    patch_seeded = bool(
        box_has_area(seed_result.patch_xyxy)
        and bool(seed_result.pose_anchor_source)
        and seed_result.features is not None
        and len(seed_result.features) > 0
    )
    hard_geometry_reject = bool(
        patch_seeded
        and real_row_rejected_by_geometry_consistency(
            state=state,
            next_bbox_xyxy=real_bbox_xyxy,
            next_patch_xyxy=list(seed_result.patch_xyxy) if seed_result.patch_xyxy is not None else None,
            overlay=overlay,
        )
    )
    supportable = bool(
        not shape_reject
        and not roi_reject
        and (
            iou_score >= 0.10
            or center_consistent
            or (
                predicted_anchor_xy is not None
                and observed_anchor_xy is not None
                and anchor_innovation_px <= max(30.0, scale * 0.28)
            )
        )
    )
    adoptable_now = bool(patch_seeded and motion_consistent and shape_consistent and roi_compatible)
    projected_flags = list(state.recent_real_adoptable_flags[-2:]) + [1 if adoptable_now else 0]
    recent_real_adoptable_count = int(sum(projected_flags))
    first_reacquire_probe = not state.recent_real_adoptable_flags
    temporal_consistent = bool(adoptable_now and recent_real_adoptable_count >= 2)
    clear_override = bool(
        adoptable_now
        and seed_result.seed_mode == "pose_seed"
        and float(seed_result.anchor_score) >= 0.70
        and strong_motion_consistent
        and iou_score >= 0.35
    )

    if hard_geometry_reject or (motion_reject and not center_consistent and iou_score < 0.08) or not supportable:
        classification = REAL_REJECT
        stop_reason = "real_geometry_reject"
    elif clear_override or temporal_consistent:
        classification = REAL_ADOPT
        stop_reason = ""
    elif first_reacquire_probe:
        classification = REAL_SUPPORT_ONLY
        stop_reason = "real_first_frame_provisional"
    else:
        classification = REAL_SUPPORT_ONLY
        stop_reason = "real_temporal_consistency_insufficient"

    return RealReacquireDecision(
        classification=classification,
        stop_reason=stop_reason,
        predicted_anchor_xy=predicted_anchor_xy,
        observed_anchor_xy=observed_anchor_xy,
        anchor_innovation_px=anchor_innovation_px,
        height_ratio=height_ratio,
        aspect_ratio_ratio=aspect_ratio_ratio,
        recent_real_adoptable_count=recent_real_adoptable_count,
        klt_reliable_chain_length=max(0, int(state.klt_reliable_chain_length)),
        adoptable_now=adoptable_now,
    )


def klt_step_rejected_by_motion_consistency(
    *,
    state: KltContinuityTrackState,
    next_bbox_xyxy: list[float],
    flow_dx: float,
    flow_dy: float,
    overlay: stage0403.OverlaySpec,
) -> bool:
    lateral_jump = bool(abs(float(flow_dx)) >= 8.0 and abs(float(flow_dx)) > (abs(float(flow_dy)) * 1.5))
    recent_ref = recent_motion_reference(state)
    motion_flip = False
    motion_mismatch = False
    if recent_ref is not None:
        ref_dx, _ = recent_ref
        if abs(float(ref_dx)) >= 2.0:
            motion_flip = bool(
                math.copysign(1.0, float(flow_dx)) != math.copysign(1.0, float(ref_dx))
                and abs(float(flow_dx)) > max(8.0, abs(float(ref_dx)) * 2.0)
            )
            motion_mismatch = bool(
                abs(float(flow_dx) - float(ref_dx)) > max(8.0, abs(float(ref_dx)) * 1.5 + 4.0)
            )
    prev_dist = float(compute_boundary_distance(state.last_bbox_xyxy, overlay))
    next_dist = float(compute_boundary_distance(next_bbox_xyxy, overlay))
    moving_outward = bool(next_dist > (prev_dist + 4.0))
    inward_ok = bool(next_dist <= (prev_dist + 1.0))
    return bool((moving_outward and lateral_jump) or ((motion_flip or motion_mismatch) and not inward_ok))


def real_row_rejected_by_geometry_consistency(
    *,
    state: KltContinuityTrackState,
    next_bbox_xyxy: list[float],
    next_patch_xyxy: list[float] | None,
    overlay: stage0403.OverlaySpec,
) -> bool:
    if not box_has_area(state.last_bbox_xyxy) or not box_has_area(next_bbox_xyxy):
        return False
    if not box_has_area(state.last_patch_xyxy) or not box_has_area(next_patch_xyxy):
        return False

    prev_x1, prev_y1, prev_x2, prev_y2 = map(float, state.last_bbox_xyxy)
    next_x1, next_y1, next_x2, next_y2 = map(float, next_bbox_xyxy)
    prev_w = max(1.0, prev_x2 - prev_x1)
    prev_h = max(1.0, prev_y2 - prev_y1)
    next_h = max(1.0, next_y2 - next_y1)
    height_ratio = next_h / prev_h
    bottom_lift = prev_y2 - next_y2

    prev_patch_cx, prev_patch_cy = bbox_center_xy(state.last_patch_xyxy)
    next_patch_cx, next_patch_cy = bbox_center_xy(next_patch_xyxy)
    patch_shift_x = abs(float(next_patch_cx) - float(prev_patch_cx))
    patch_shift_y = abs(float(next_patch_cy) - float(prev_patch_cy))
    upper_anchor_stable = bool(
        patch_shift_x <= max(16.0, prev_w * 0.20)
        and patch_shift_y <= max(12.0, prev_h * 0.10)
    )
    upper_anchor_lift_jump = bool(
        patch_shift_x <= max(16.0, prev_w * 0.25)
        and (prev_patch_cy - next_patch_cy) >= max(18.0, prev_h * 0.15)
    )

    sudden_bottom_collapse = bool(
        bottom_lift >= max(26.0, prev_h * 0.22)
        and height_ratio <= 0.78
    )
    return bool(sudden_bottom_collapse and (upper_anchor_stable or upper_anchor_lift_jump))


def make_proxy_sidecar_row(
    *,
    frame_num: int,
    source_id: int,
    track_id: int,
    bbox_xyxy: list[float],
    patch_xyxy: list[float],
    proxy_age: int,
    pose_anchor_source: str,
    tracked_points: int,
    flow_dx: float,
    flow_dy: float,
    flow_mag: float,
    mode: str = "proxy",
    event: str = "proxy_active",
    stop_reason: str = "",
    handoff_reason: str = "",
) -> dict[str, str]:
    row = {
        "frame_num": str(int(frame_num)),
        "source_id": str(int(source_id)),
        "track_id": str(int(track_id)),
        "mode": str(mode),
        "proxy_active": "1",
        "proxy_age": str(max(0, int(proxy_age))),
        "event": str(event),
        "stop_reason": str(stop_reason),
        "handoff_reason": str(handoff_reason),
        "proxy_left": "0",
        "proxy_top": "0",
        "proxy_width": "0",
        "proxy_height": "0",
        "patch_left": "0",
        "patch_top": "0",
        "patch_width": "0",
        "patch_height": "0",
        "patch_source": "upper_klt_proxy",
        "pose_anchor_source": str(pose_anchor_source),
        "tracked_points": str(max(0, int(tracked_points))),
        "flow_dx": format_sidecar_float(flow_dx, digits=4),
        "flow_dy": format_sidecar_float(flow_dy, digits=4),
        "flow_mag": format_sidecar_float(flow_mag, digits=4),
    }
    set_sidecar_bbox_fields(row, bbox_xyxy)
    set_sidecar_patch_fields(row, patch_xyxy)
    annotate_sidecar_debug_fields(row, row_source=mode)
    return row


def build_preserved_hold_row(
    *,
    state: KltContinuityTrackState,
    gray: np.ndarray,
    frame_num: int,
    source_id: int,
    track_id: int,
    min_good_points: int,
    proxy_age: int,
    stop_reason: str,
    handoff_reason: str,
    row_source: str,
    decision: RealReacquireDecision | None = None,
    refresh_real_context: bool = False,
) -> dict[str, str] | None:
    hold_features, hold_patch_xyxy = seed_features_in_patch(
        gray=gray,
        patch_xyxy=list(state.last_patch_xyxy),
        min_good_points=min_good_points,
    )
    if hold_features:
        state.features = list(hold_features)
    else:
        state.features = []
    if box_has_area(hold_patch_xyxy):
        state.last_patch_xyxy = list(hold_patch_xyxy)
        state.last_valid_anchor_xy = bbox_center_xy(state.last_patch_xyxy)
    elif box_has_area(state.last_patch_xyxy):
        state.last_valid_anchor_xy = bbox_center_xy(state.last_patch_xyxy)
    if not (state.pose_anchor_source and box_has_area(state.last_bbox_xyxy) and box_has_area(state.last_patch_xyxy)):
        return None
    if refresh_real_context:
        state.last_real_frame = int(frame_num)
    state.last_valid_row_source = "frozen_hold"
    state.klt_reliable_chain_length = max(1, int(state.klt_reliable_chain_length) + 1)
    hold_row = make_proxy_sidecar_row(
        frame_num=int(frame_num),
        source_id=int(source_id),
        track_id=int(track_id),
        bbox_xyxy=list(state.last_bbox_xyxy),
        patch_xyxy=list(state.last_patch_xyxy),
        proxy_age=max(1, int(proxy_age)),
        pose_anchor_source=state.pose_anchor_source,
        tracked_points=int(len(state.features)),
        flow_dx=0.0,
        flow_dy=0.0,
        flow_mag=0.0,
        mode="frozen_hold",
        event="proxy_hold",
        stop_reason=stop_reason,
        handoff_reason=handoff_reason,
    )
    annotate_sidecar_debug_fields(
        hold_row,
        row_source=row_source,
        real_reacquire_class="" if decision is None else decision.classification,
        predicted_anchor_xy=None if decision is None else decision.predicted_anchor_xy,
        anchor_innovation_px=None if decision is None else decision.anchor_innovation_px,
        height_ratio=None if decision is None else decision.height_ratio,
        aspect_ratio_ratio=None if decision is None else decision.aspect_ratio_ratio,
        recent_real_adoptable_count=None if decision is None else decision.recent_real_adoptable_count,
        klt_reliable_chain_length=None if decision is None else decision.klt_reliable_chain_length,
    )
    return hold_row


def is_plausible_proxy_handoff(candidate_box: list[float], proxy_box: list[float], max_shift_px: float) -> bool:
    candidate_center = bbox_center_xy(candidate_box)
    proxy_center = bbox_center_xy(proxy_box)
    center_dist = math.hypot(candidate_center[0] - proxy_center[0], candidate_center[1] - proxy_center[1])
    if center_dist > float(max_shift_px):
        return False
    candidate_w = max(1.0, float(candidate_box[2]) - float(candidate_box[0]))
    candidate_h = max(1.0, float(candidate_box[3]) - float(candidate_box[1]))
    proxy_w = max(1.0, float(proxy_box[2]) - float(proxy_box[0]))
    proxy_h = max(1.0, float(proxy_box[3]) - float(proxy_box[1]))
    return (
        0.5 <= (candidate_w / proxy_w) <= 2.0
        and 0.5 <= (candidate_h / proxy_h) <= 2.0
    )


def advance_klt_proxy_state(
    *,
    prev_gray: np.ndarray,
    gray: np.ndarray,
    state: KltContinuityTrackState,
    min_good_points: int,
    max_shift_px: float,
) -> dict[str, Any] | None:
    if cv2 is None or np is None:
        return None
    if not box_has_area(state.last_bbox_xyxy) or not box_has_area(state.last_patch_xyxy):
        return None

    seed_features = list(state.features)
    if len(seed_features) < max(1, int(min_good_points)):
        seed_features, _ = seed_features_in_patch(gray=prev_gray, patch_xyxy=state.last_patch_xyxy, min_good_points=min_good_points)
    if len(seed_features) < max(1, int(min_good_points)):
        return None

    prev_points = np.asarray(seed_features, dtype=np.float32).reshape(-1, 1, 2)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return None

    frame_h, frame_w = gray.shape[:2]
    prev_good: list[tuple[float, float]] = []
    next_good: list[tuple[float, float]] = []
    for idx, flag in enumerate(status.reshape(-1).tolist()):
        if not flag:
            continue
        prev_x, prev_y = map(float, prev_points[idx][0])
        next_x, next_y = map(float, next_points[idx][0])
        if not (0.0 <= next_x < float(frame_w) and 0.0 <= next_y < float(frame_h)):
            continue
        prev_good.append((prev_x, prev_y))
        next_good.append((next_x, next_y))

    good_points = len(next_good)
    if good_points < max(1, int(min_good_points)):
        return None

    dx_values = [float(next_good[idx][0]) - float(prev_good[idx][0]) for idx in range(good_points)]
    dy_values = [float(next_good[idx][1]) - float(prev_good[idx][1]) for idx in range(good_points)]
    flow_dx = float(robust_trimmed_component(dx_values))
    flow_dy = float(robust_trimmed_component(dy_values))
    flow_mag = float(math.hypot(flow_dx, flow_dy))
    if flow_mag > float(max_shift_px):
        return None

    filtered_features = [
        point
        for idx, point in enumerate(next_good)
        if abs(dx_values[idx] - flow_dx) <= max(1.0, abs(flow_dx) * 1.5 + 2.0)
        and abs(dy_values[idx] - flow_dy) <= max(1.0, abs(flow_dy) * 1.5 + 2.0)
    ]
    if len(filtered_features) >= max(1, int(min_good_points)):
        next_good = filtered_features
        good_points = len(next_good)

    next_bbox_xyxy = shift_box_xyxy(state.last_bbox_xyxy, flow_dx, flow_dy, frame_w, frame_h)
    next_patch_xyxy = shift_box_xyxy(state.last_patch_xyxy, flow_dx, flow_dy, frame_w, frame_h)
    if not box_has_area(next_bbox_xyxy) or not box_has_area(next_patch_xyxy):
        return None

    return {
        "bbox_xyxy": next_bbox_xyxy,
        "patch_xyxy": next_patch_xyxy,
        "features": next_good,
        "tracked_points": int(good_points),
        "flow_dx": float(flow_dx),
        "flow_dy": float(flow_dy),
        "flow_mag": float(flow_mag),
    }


def maybe_seed_klt_from_real_row(
    *,
    frame: np.ndarray,
    gray: np.ndarray,
    spec: stage0403.SourceSpec,
    overlay: stage0403.OverlaySpec,
    row: SidecarRow,
    row_dict: dict[str, str],
    state: KltContinuityTrackState | None,
    pose_reacquirer: CropPoseReacquirer,
    params: DecisionParams,
 ) -> RealKltSeedResult:
    if not is_boundary_relevant_for_klt(bbox_xyxy=list(row.bbox_xyxy), patch_xyxy=None, overlay=overlay, params=params):
        return RealKltSeedResult(None, None, "", "not_boundary_relevant")

    min_good_points = max(1, int(params.klt_confirm_min_tracked_points))
    carry_pose_anchor = state.pose_anchor_source if state is not None else ""
    if state is not None and box_has_area(state.last_patch_xyxy) and box_has_area(state.last_bbox_xyxy) and carry_pose_anchor:
        bbox_center_prev = bbox_center_xy(state.last_bbox_xyxy)
        bbox_center_now = bbox_center_xy(row.bbox_xyxy)
        bbox_shift = math.hypot(bbox_center_now[0] - bbox_center_prev[0], bbox_center_now[1] - bbox_center_prev[1])
        bbox_scale = max(1.0, float(row.bbox_xyxy[2]) - float(row.bbox_xyxy[0]), float(row.bbox_xyxy[3]) - float(row.bbox_xyxy[1]))
        if bbox_shift <= max(32.0, bbox_scale * 0.60):
            shifted_patch = shift_box_xyxy(
                state.last_patch_xyxy,
                bbox_center_now[0] - bbox_center_prev[0],
                bbox_center_now[1] - bbox_center_prev[1],
                gray.shape[1],
                gray.shape[0],
            )
            features, shifted_patch = seed_features_in_patch(
                gray=gray,
                patch_xyxy=shifted_patch,
                min_good_points=min_good_points,
            )
            if len(features) >= min_good_points:
                update_real_row_with_klt_seed(
                    row=row_dict,
                    patch_xyxy=shifted_patch,
                    pose_anchor_source=carry_pose_anchor,
                    seed_feature_count=len(features),
                )
                return RealKltSeedResult(
                    patch_xyxy=shifted_patch,
                    features=features,
                    pose_anchor_source=carry_pose_anchor,
                    seed_mode="carried_seed",
                    anchor_xy_source=bbox_center_xy(shifted_patch),
                    anchor_score=0.0,
                )

    crop_frame, crop_asset, expected_box = build_pose_seed_crop(
        frame=frame,
        bbox_xyxy=list(row.bbox_xyxy),
        source_id=row.source_id,
        clip_label=spec.clip_label,
        source_clip_path=spec.local_path,
    )
    if crop_frame is None or crop_asset is None:
        return RealKltSeedResult(None, None, "", "crop_unavailable")

    detection = pose_reacquirer.detect(
        crop_frame=crop_frame,
        expected_box_crop_xyxy=expected_box,
        crop_asset=crop_asset,
    )
    if detection is None or detection.upper_anchor_xy_source is None:
        return RealKltSeedResult(None, None, "", "pose_missing")
    if not (is_headlike_source(detection.upper_anchor_source) or is_shoulder_source(detection.upper_anchor_source)):
        return RealKltSeedResult(None, None, "", "upper_anchor_unusable")

    patch_xyxy = build_upper_patch_xyxy(
        detection=detection,
        frame_w=gray.shape[1],
        frame_h=gray.shape[0],
    )
    if patch_xyxy is None:
        return RealKltSeedResult(None, None, "", "patch_missing")

    features, patch_xyxy = seed_features_in_patch(
        gray=gray,
        patch_xyxy=patch_xyxy,
        min_good_points=min_good_points,
    )
    if len(features) < min_good_points:
        return RealKltSeedResult(None, None, "", "too_few_seed_features")

    pose_anchor_source = format_pose_anchor_source(detection.upper_anchor_source, detection.upper_anchor_kind)
    update_real_row_with_klt_seed(
        row=row_dict,
        patch_xyxy=patch_xyxy,
        pose_anchor_source=pose_anchor_source,
        seed_feature_count=len(features),
    )
    return RealKltSeedResult(
        patch_xyxy=patch_xyxy,
        features=features,
        pose_anchor_source=pose_anchor_source,
        seed_mode="pose_seed",
        anchor_xy_source=tuple(detection.upper_anchor_xy_source),
        anchor_score=float(detection.upper_anchor_score),
    )


def augment_split_sidecar_with_klt_continuity(
    *,
    spec: stage0403.SourceSpec,
    overlay: stage0403.OverlaySpec,
    split_sidecar_path: Path,
    pose_reacquirer: CropPoseReacquirer,
    params: DecisionParams,
    logger: logging.Logger,
) -> dict[str, Any]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV/Numpy are required for Stage 04.05 KLT sidecar augmentation")

    with split_sidecar_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        original_rows = [dict(row) for row in reader]

    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in original_rows:
        try:
            frame_num = int(float(str(row.get("frame_num", "0")).strip() or "0"))
        except ValueError:
            frame_num = 0
        rows_by_frame[frame_num].append(row)

    cap = cv2.VideoCapture(str(spec.local_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for KLT augmentation: {spec.local_path}")

    written_rows: list[dict[str, str]] = []
    track_states: dict[int, KltContinuityTrackState] = {}
    seed_track_ids: set[int] = set()
    proxy_track_ids: set[int] = set()
    stats: Counter[str] = Counter()
    max_proxy_age = 0
    prev_gray: np.ndarray | None = None
    frame_num = 0
    max_shift_px = 32.0
    min_good_points = max(1, int(params.klt_confirm_min_tracked_points))
    rejected_hold_max_frames = 6
    base_proxy_age_frames = max(
        1,
        int(params.klt_confirm_max_loss_frames),
        int(params.klt_continuity_max_proxy_age_frames),
    )
    bonus_proxy_age_frames = max(base_proxy_age_frames, int(params.klt_continuity_bonus_proxy_age_frames))

    # Display continuity: extend frozen bbox rows past the KLT proxy age limit
    # so the FSM can evaluate relaxed confirm geometry for boundary hard-cases.
    display_continuity_max_extension = max(0, int(params.grace_frames))
    display_continuity_tracks: dict[int, dict[str, Any]] = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rows = [dict(row) for row in rows_by_frame.get(frame_num, [])]
        current_real_boxes: list[list[float]] = []
        seen_real_track_ids: set[int] = set()

        for row_dict in frame_rows:
            row = SidecarRow.from_csv_row(row_dict)
            if row.mode != "real" or not row.has_valid_bbox:
                annotate_sidecar_debug_fields(row_dict, row_source=row.mode)
                written_rows.append(row_dict)
                continue

            stats["real_rows"] += 1

            track_id = int(row.track_id)
            state = track_states.get(track_id)
            seed_result = maybe_seed_klt_from_real_row(
                frame=frame,
                gray=gray,
                spec=spec,
                overlay=overlay,
                row=row,
                row_dict=row_dict,
                state=state,
                pose_reacquirer=pose_reacquirer,
                params=params,
            )
            patch_xyxy = seed_result.patch_xyxy
            features = seed_result.features
            pose_anchor_source = seed_result.pose_anchor_source
            seed_mode = seed_result.seed_mode
            if seed_mode == "pose_seed":
                stats["pose_seed_calls"] += 1
                stats["pose_seed_successes"] += 1
            elif seed_mode in {"pose_missing", "upper_anchor_unusable", "patch_missing", "too_few_seed_features"}:
                stats["pose_seed_calls"] += 1

            annotate_sidecar_debug_fields(row_dict, row_source="real")

            active_reliable_chain = bool(
                state is not None
                and klt_chain_is_reliably_active(
                    state=state,
                    frame_num=int(frame_num),
                    min_good_points=min_good_points,
                    params=params,
                )
            )
            decision: RealReacquireDecision | None = None
            if active_reliable_chain and state is not None:
                decision = classify_returning_real_against_klt_chain(
                    frame_num=int(frame_num),
                    row=row,
                    state=state,
                    seed_result=seed_result,
                    overlay=overlay,
                    params=params,
                )
                if decision.classification == REAL_SUPPORT_ONLY:
                    record_recent_real_adoptability(
                        state,
                        frame_num=int(frame_num),
                        adoptable=decision.adoptable_now,
                    )
                    state.rejected_hold_streak = 0
                    seen_real_track_ids.add(track_id)
                    hold_row = build_preserved_hold_row(
                        state=state,
                        gray=gray,
                        frame_num=int(frame_num),
                        source_id=int(spec.source_id),
                        track_id=int(track_id),
                        min_good_points=min_good_points,
                        proxy_age=1,
                        stop_reason=decision.stop_reason,
                        handoff_reason="real_support_only",
                        row_source="real_support_only",
                        decision=decision,
                        refresh_real_context=True,
                    )
                    if hold_row is not None:
                        written_rows.append(hold_row)
                        stats["proxy_hold_rows"] += 1
                        stats["real_reacquire_support_only"] += 1
                        if decision.stop_reason == "real_first_frame_provisional":
                            stats["real_reacquire_first_frame_provisional"] += 1
                        if decision.stop_reason == "real_temporal_consistency_insufficient":
                            stats["real_reacquire_temporal_waits"] += 1
                        continue
                    track_states.pop(track_id, None)
                    stats["real_reacquire_support_stops"] += 1
                    continue
                if decision.classification == REAL_REJECT:
                    reset_recent_real_adoptability(state)
                    state.rejected_hold_streak += 1
                    seen_real_track_ids.add(track_id)
                    hold_row = build_preserved_hold_row(
                        state=state,
                        gray=gray,
                        frame_num=int(frame_num),
                        source_id=int(spec.source_id),
                        track_id=int(track_id),
                        min_good_points=min_good_points,
                        proxy_age=max(1, int(frame_num) - int(state.last_real_frame)),
                        stop_reason=decision.stop_reason,
                        handoff_reason="real_reject_preserve_klt",
                        row_source="frozen_hold",
                        decision=decision,
                        refresh_real_context=False,
                    )
                    if hold_row is not None and state.rejected_hold_streak <= int(rejected_hold_max_frames):
                        written_rows.append(hold_row)
                        stats["proxy_hold_rows"] += 1
                        stats["real_geometry_rejects"] += 1
                        stats["real_reacquire_rejects"] += 1
                        continue
                    track_states.pop(track_id, None)
                    stats["real_geometry_reject_stops"] += 1
                    stats["real_reacquire_reject_stops"] += 1
                    continue
                annotate_sidecar_debug_fields(
                    row_dict,
                    row_source="real",
                    real_reacquire_class=decision.classification,
                    predicted_anchor_xy=decision.predicted_anchor_xy,
                    anchor_innovation_px=decision.anchor_innovation_px,
                    height_ratio=decision.height_ratio,
                    aspect_ratio_ratio=decision.aspect_ratio_ratio,
                    recent_real_adoptable_count=decision.recent_real_adoptable_count,
                    klt_reliable_chain_length=decision.klt_reliable_chain_length,
                )

            if (
                state is not None
                and patch_xyxy is not None
                and pose_anchor_source
                and real_row_rejected_by_geometry_consistency(
                    state=state,
                    next_bbox_xyxy=list(row.bbox_xyxy),
                    next_patch_xyxy=list(patch_xyxy),
                    overlay=overlay,
                )
            ):
                state.rejected_hold_streak += 1
                seen_real_track_ids.add(track_id)
                hold_row = build_preserved_hold_row(
                    state=state,
                    gray=gray,
                    frame_num=int(frame_num),
                    source_id=int(spec.source_id),
                    track_id=int(track_id),
                    min_good_points=min_good_points,
                    proxy_age=max(1, int(frame_num) - int(state.last_real_frame)),
                    stop_reason="real_geometry_reject",
                    handoff_reason="real_reject_preserve_klt",
                    row_source="frozen_hold",
                )
                if state.rejected_hold_streak <= int(rejected_hold_max_frames):
                    if hold_row is not None:
                        written_rows.append(hold_row)
                        stats["proxy_hold_rows"] += 1
                        stats["real_geometry_rejects"] += 1
                        continue
                track_states.pop(track_id, None)
                stats["real_geometry_reject_stops"] += 1
                continue

            # Lightweight consistency gate: when an existing KLT state has
            # active features but the chain is not yet reliably active, reject
            # new real detections that are geometrically inconsistent (large
            # center jump or extreme scale change).  This prevents bad or
            # flickering detections from corrupting the existing KLT patch.
            if (
                state is not None
                and not active_reliable_chain
                and box_has_area(state.last_bbox_xyxy)
                and len(state.features) > 0
                and bool(state.pose_anchor_source)
            ):
                _prev_cxy = bbox_center_xy(state.last_bbox_xyxy)
                _real_cxy = bbox_center_xy(row.bbox_xyxy)
                _prev_w, _prev_h = bbox_width_height(state.last_bbox_xyxy)
                _real_w, _real_h = bbox_width_height(row.bbox_xyxy)
                _gate_scale = max(_prev_w, _prev_h, _real_w, _real_h)
                _center_dist = math.hypot(
                    float(_real_cxy[0]) - float(_prev_cxy[0]),
                    float(_real_cxy[1]) - float(_prev_cxy[1]),
                )
                _height_ratio = float(_real_h) / max(1.0, float(_prev_h))
                _soft_reject = bool(
                    _center_dist > max(40.0, _gate_scale * 0.40)
                    or _height_ratio < 0.60
                    or _height_ratio > 1.65
                )
                if _soft_reject:
                    state.rejected_hold_streak += 1
                    seen_real_track_ids.add(track_id)
                    if state.rejected_hold_streak <= int(rejected_hold_max_frames):
                        hold_row = build_preserved_hold_row(
                            state=state,
                            gray=gray,
                            frame_num=int(frame_num),
                            source_id=int(spec.source_id),
                            track_id=int(track_id),
                            min_good_points=min_good_points,
                            proxy_age=max(1, int(frame_num) - int(state.last_real_frame)),
                            stop_reason="soft_geometry_reject",
                            handoff_reason="real_soft_reject_preserve_klt",
                            row_source="soft_reject_hold",
                        )
                        if hold_row is not None:
                            written_rows.append(hold_row)
                            stats["proxy_hold_rows"] += 1
                            stats["real_soft_geometry_rejects"] += 1
                            continue
                    track_states.pop(track_id, None)
                    stats["real_soft_geometry_reject_stops"] += 1
                    continue

            seen_real_track_ids.add(track_id)
            current_real_boxes.append(list(row.bbox_xyxy))

            next_state = track_states.setdefault(track_id, KltContinuityTrackState(track_id=track_id))
            if box_has_area(next_state.last_bbox_xyxy):
                prev_center = bbox_center_xy(next_state.last_bbox_xyxy)
                curr_center = bbox_center_xy(row.bbox_xyxy)
                update_motion_history(
                    next_state,
                    dx=float(curr_center[0] - prev_center[0]),
                    dy=float(curr_center[1] - prev_center[1]),
                )
            next_state.last_real_frame = int(frame_num)
            next_state.last_bbox_xyxy = list(row.bbox_xyxy)
            next_state.rejected_hold_streak = 0
            next_state.klt_reliable_chain_length = 0
            next_state.last_valid_row_source = "real"
            reset_recent_real_adoptability(next_state)
            if patch_xyxy is not None and features is not None and pose_anchor_source:
                next_state.last_patch_xyxy = list(patch_xyxy)
                next_state.features = list(features)
                next_state.pose_anchor_source = str(pose_anchor_source)
                next_state.patch_source = "upper_klt_seed"
                next_state.last_valid_anchor_xy = (
                    tuple(seed_result.anchor_xy_source)
                    if seed_result.anchor_xy_source is not None
                    else bbox_center_xy(next_state.last_patch_xyxy)
                )
                seed_track_ids.add(int(row.track_id))
                stats["real_rows_with_upper_seed"] += 1
                if seed_mode == "carried_seed":
                    stats["real_rows_with_carried_seed"] += 1
            else:
                next_state.last_patch_xyxy = []
                next_state.features = []
                next_state.pose_anchor_source = ""
                next_state.patch_source = "upper_klt_seed"
                next_state.last_valid_anchor_xy = compute_bbox_upper_proxy(next_state.last_bbox_xyxy)

            if decision is not None and decision.classification == REAL_ADOPT:
                stats["real_reacquire_adopts"] += 1

            written_rows.append(row_dict)

        if prev_gray is not None:
            for track_id, state in list(track_states.items()):
                if track_id in seen_real_track_ids:
                    continue

                miss_frames = int(frame_num) - int(state.last_real_frame)
                if miss_frames <= 0 or miss_frames > bonus_proxy_age_frames:
                    if (
                        miss_frames > 0
                        and box_has_area(state.last_bbox_xyxy)
                        and state.pose_anchor_source
                        and track_id not in display_continuity_tracks
                    ):
                        display_continuity_tracks[track_id] = {
                            "bbox_xyxy": list(state.last_bbox_xyxy),
                            "patch_xyxy": list(state.last_patch_xyxy) if box_has_area(state.last_patch_xyxy) else list(state.last_bbox_xyxy),
                            "pose_anchor_source": str(state.pose_anchor_source),
                            "last_real_frame": int(state.last_real_frame),
                        }
                    stats["proxy_age_stops"] += 1
                    track_states.pop(track_id, None)
                    continue
                if not box_has_area(state.last_bbox_xyxy) or not box_has_area(state.last_patch_xyxy) or not state.pose_anchor_source:
                    track_states.pop(track_id, None)
                    continue
                if any(is_plausible_proxy_handoff(real_box, state.last_bbox_xyxy, max_shift_px) for real_box in current_real_boxes):
                    track_states.pop(track_id, None)
                    stats["proxy_handoff_stops"] += 1
                    continue

                proxy_step = advance_klt_proxy_state(
                    prev_gray=prev_gray,
                    gray=gray,
                    state=state,
                    min_good_points=min_good_points,
                    max_shift_px=max_shift_px,
                )
                if proxy_step is None:
                    state.rejected_hold_streak += 1
                    hold_features, hold_patch_xyxy = seed_features_in_patch(
                        gray=gray,
                        patch_xyxy=list(state.last_patch_xyxy),
                        min_good_points=min_good_points,
                    )
                    if hold_features:
                        state.features = list(hold_features)
                    else:
                        state.features = []
                    if box_has_area(hold_patch_xyxy):
                        state.last_patch_xyxy = list(hold_patch_xyxy)
                    if (
                        state.rejected_hold_streak <= int(rejected_hold_max_frames)
                        and state.pose_anchor_source
                        and box_has_area(state.last_bbox_xyxy)
                        and box_has_area(state.last_patch_xyxy)
                    ):
                        hold_row = make_proxy_sidecar_row(
                            frame_num=int(frame_num),
                            source_id=int(spec.source_id),
                            track_id=int(track_id),
                            bbox_xyxy=list(state.last_bbox_xyxy),
                            patch_xyxy=list(state.last_patch_xyxy),
                            proxy_age=miss_frames,
                            pose_anchor_source=state.pose_anchor_source,
                            tracked_points=int(len(state.features)),
                            flow_dx=0.0,
                            flow_dy=0.0,
                            flow_mag=0.0,
                            mode="frozen_hold",
                            event="proxy_hold",
                            stop_reason="klt_fail_hold",
                        )
                        state.last_valid_anchor_xy = bbox_center_xy(state.last_patch_xyxy)
                        state.last_valid_row_source = "frozen_hold"
                        state.klt_reliable_chain_length = max(1, int(state.klt_reliable_chain_length) + 1)
                        written_rows.append(hold_row)
                        stats["proxy_hold_rows"] += 1
                        stats["proxy_klt_fail_holds"] += 1
                        continue
                    track_states.pop(track_id, None)
                    stats["proxy_klt_stops"] += 1
                    continue
                if not is_boundary_relevant_for_klt(
                    bbox_xyxy=list(proxy_step["bbox_xyxy"]),
                    patch_xyxy=list(proxy_step["patch_xyxy"]),
                    overlay=overlay,
                    params=params,
                ):
                    track_states.pop(track_id, None)
                    stats["proxy_boundary_stops"] += 1
                    continue
                if klt_step_rejected_by_motion_consistency(
                    state=state,
                    next_bbox_xyxy=list(proxy_step["bbox_xyxy"]),
                    flow_dx=float(proxy_step["flow_dx"]),
                    flow_dy=float(proxy_step["flow_dy"]),
                    overlay=overlay,
                ):
                    state.rejected_hold_streak += 1
                    hold_features, hold_patch_xyxy = seed_features_in_patch(
                        gray=gray,
                        patch_xyxy=list(state.last_patch_xyxy),
                        min_good_points=min_good_points,
                    )
                    if hold_features:
                        state.features = list(hold_features)
                    else:
                        state.features = []
                    if box_has_area(hold_patch_xyxy):
                        state.last_patch_xyxy = list(hold_patch_xyxy)
                    if state.rejected_hold_streak <= int(rejected_hold_max_frames):
                        hold_row = make_proxy_sidecar_row(
                            frame_num=int(frame_num),
                            source_id=int(spec.source_id),
                            track_id=int(track_id),
                            bbox_xyxy=list(state.last_bbox_xyxy),
                            patch_xyxy=list(state.last_patch_xyxy),
                            proxy_age=miss_frames,
                            pose_anchor_source=state.pose_anchor_source,
                            tracked_points=int(len(state.features)),
                            flow_dx=0.0,
                            flow_dy=0.0,
                            flow_mag=0.0,
                            mode="frozen_hold",
                            event="proxy_hold",
                            stop_reason="jump_reject",
                        )
                        state.last_valid_anchor_xy = bbox_center_xy(state.last_patch_xyxy)
                        state.last_valid_row_source = "frozen_hold"
                        state.klt_reliable_chain_length = max(1, int(state.klt_reliable_chain_length) + 1)
                        written_rows.append(hold_row)
                        stats["proxy_hold_rows"] += 1
                        stats["proxy_jump_rejects"] += 1
                        continue
                    track_states.pop(track_id, None)
                    stats["proxy_jump_reject_stops"] += 1
                    continue
                if not klt_proxy_bonus_age_allowed(
                    miss_frames=int(miss_frames),
                    state=state,
                    tracked_points=int(proxy_step["tracked_points"]),
                    params=params,
                    min_good_points=min_good_points,
                ):
                    if (
                        box_has_area(state.last_bbox_xyxy)
                        and state.pose_anchor_source
                        and track_id not in display_continuity_tracks
                    ):
                        display_continuity_tracks[track_id] = {
                            "bbox_xyxy": list(state.last_bbox_xyxy),
                            "patch_xyxy": list(state.last_patch_xyxy) if box_has_area(state.last_patch_xyxy) else list(state.last_bbox_xyxy),
                            "pose_anchor_source": str(state.pose_anchor_source),
                            "last_real_frame": int(state.last_real_frame),
                        }
                    track_states.pop(track_id, None)
                    stats["proxy_age_stops"] += 1
                    continue

                state.last_bbox_xyxy = list(proxy_step["bbox_xyxy"])
                state.last_patch_xyxy = list(proxy_step["patch_xyxy"])
                state.features = list(proxy_step["features"])
                state.rejected_hold_streak = 0
                state.last_valid_anchor_xy = bbox_center_xy(state.last_patch_xyxy)
                state.last_valid_row_source = "proxy"
                state.klt_reliable_chain_length = max(1, int(state.klt_reliable_chain_length) + 1)
                update_motion_history(
                    state,
                    dx=float(proxy_step["flow_dx"]),
                    dy=float(proxy_step["flow_dy"]),
                )
                proxy_row = make_proxy_sidecar_row(
                    frame_num=int(frame_num),
                    source_id=int(spec.source_id),
                    track_id=int(track_id),
                    bbox_xyxy=list(proxy_step["bbox_xyxy"]),
                    patch_xyxy=list(proxy_step["patch_xyxy"]),
                    proxy_age=miss_frames,
                    pose_anchor_source=state.pose_anchor_source,
                    tracked_points=int(proxy_step["tracked_points"]),
                    flow_dx=float(proxy_step["flow_dx"]),
                    flow_dy=float(proxy_step["flow_dy"]),
                    flow_mag=float(proxy_step["flow_mag"]),
                )
                written_rows.append(proxy_row)
                proxy_track_ids.add(int(track_id))
                stats["proxy_rows_added"] += 1
                if int(miss_frames) > int(base_proxy_age_frames):
                    stats["proxy_bonus_age_rows"] += 1
                max_proxy_age = max(max_proxy_age, int(miss_frames))

        # Emit display_continuity rows for tracks that exhausted their proxy age
        # but may still be visually present near the ROI boundary.
        for dc_track_id in list(display_continuity_tracks.keys()):
            if dc_track_id in seen_real_track_ids:
                del display_continuity_tracks[dc_track_id]
                continue
            dc_state = display_continuity_tracks[dc_track_id]
            dc_miss = int(frame_num) - int(dc_state["last_real_frame"])
            dc_limit = int(bonus_proxy_age_frames) + int(display_continuity_max_extension)
            if dc_miss <= 0 or dc_miss > dc_limit:
                del display_continuity_tracks[dc_track_id]
                stats["display_continuity_age_stops"] += 1
                continue
            if any(
                is_plausible_proxy_handoff(real_box, dc_state["bbox_xyxy"], max_shift_px)
                for real_box in current_real_boxes
            ):
                del display_continuity_tracks[dc_track_id]
                stats["display_continuity_handoff_stops"] += 1
                continue
            dc_row = make_proxy_sidecar_row(
                frame_num=int(frame_num),
                source_id=int(spec.source_id),
                track_id=int(dc_track_id),
                bbox_xyxy=list(dc_state["bbox_xyxy"]),
                patch_xyxy=list(dc_state["patch_xyxy"]),
                proxy_age=dc_miss,
                pose_anchor_source=str(dc_state["pose_anchor_source"]),
                tracked_points=0,
                flow_dx=0.0,
                flow_dy=0.0,
                flow_mag=0.0,
                mode="display_continuity",
                event="display_continuity",
            )
            written_rows.append(dc_row)
            stats["display_continuity_rows"] += 1

        prev_gray = gray
        frame_num += 1

    cap.release()

    for leftover_frame_num in sorted(rows_by_frame.keys()):
        if int(leftover_frame_num) < int(frame_num):
            continue
        for row_dict in rows_by_frame[leftover_frame_num]:
            written_rows.append(dict(row_dict))

    fieldnames = fieldnames or [
        "frame_num",
        "source_id",
        "track_id",
        "mode",
        "proxy_active",
        "proxy_age",
        "event",
        "stop_reason",
        "handoff_reason",
        "proxy_left",
        "proxy_top",
        "proxy_width",
        "proxy_height",
        "patch_left",
        "patch_top",
        "patch_width",
        "patch_height",
        "patch_source",
        "pose_anchor_source",
        "tracked_points",
        "flow_dx",
        "flow_dy",
        "flow_mag",
    ]
    for debug_field in SIDECAR_DEBUG_FIELDS:
        if debug_field not in fieldnames:
            fieldnames.append(debug_field)
    written_rows.sort(
        key=lambda row: (
            int(float(str(row.get("frame_num", "0")).strip() or "0")),
            int(float(str(row.get("track_id", "0")).strip() or "0")),
            0 if str(row.get("mode", "")).strip() == "real" else 1,
        )
    )
    with split_sidecar_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written_rows)

    summary = {
        "split_sidecar_path": str(split_sidecar_path),
        "source_id": int(spec.source_id),
        "clip_label": spec.clip_label,
        "source_video_path": str(spec.local_path),
        "original_rows": int(len(original_rows)),
        "rows_after_augmentation": int(len(written_rows)),
        "pose_seed_calls": int(stats["pose_seed_calls"]),
        "pose_seed_successes": int(stats["pose_seed_successes"]),
        "real_rows_with_upper_seed": int(stats["real_rows_with_upper_seed"]),
        "real_rows_with_carried_seed": int(stats["real_rows_with_carried_seed"]),
        "proxy_rows_added": int(stats["proxy_rows_added"]),
        "proxy_bonus_age_rows": int(stats["proxy_bonus_age_rows"]),
        "proxy_hold_rows": int(stats["proxy_hold_rows"]),
        "proxy_jump_rejects": int(stats["proxy_jump_rejects"]),
        "proxy_jump_reject_stops": int(stats["proxy_jump_reject_stops"]),
        "proxy_klt_fail_holds": int(stats["proxy_klt_fail_holds"]),
        "real_geometry_rejects": int(stats["real_geometry_rejects"]),
        "real_geometry_reject_stops": int(stats["real_geometry_reject_stops"]),
        "real_soft_geometry_rejects": int(stats["real_soft_geometry_rejects"]),
        "real_soft_geometry_reject_stops": int(stats["real_soft_geometry_reject_stops"]),
        "real_reacquire_adopts": int(stats["real_reacquire_adopts"]),
        "real_reacquire_support_only": int(stats["real_reacquire_support_only"]),
        "real_reacquire_rejects": int(stats["real_reacquire_rejects"]),
        "real_reacquire_first_frame_provisional": int(stats["real_reacquire_first_frame_provisional"]),
        "real_reacquire_temporal_waits": int(stats["real_reacquire_temporal_waits"]),
        "real_reacquire_support_stops": int(stats["real_reacquire_support_stops"]),
        "real_reacquire_reject_stops": int(stats["real_reacquire_reject_stops"]),
        "proxy_age_stops": int(stats["proxy_age_stops"]),
        "proxy_klt_stops": int(stats["proxy_klt_stops"]),
        "proxy_boundary_stops": int(stats["proxy_boundary_stops"]),
        "proxy_handoff_stops": int(stats["proxy_handoff_stops"]),
        "tracks_with_upper_seed": int(len(seed_track_ids)),
        "tracks_with_proxy_rows": int(len(proxy_track_ids)),
        "max_proxy_age_frames": int(max_proxy_age),
        "base_proxy_age_frames": int(base_proxy_age_frames),
        "bonus_proxy_age_frames": int(bonus_proxy_age_frames),
        "min_good_points": int(min_good_points),
        "max_shift_px": float(max_shift_px),
        "display_continuity_rows": int(stats["display_continuity_rows"]),
        "display_continuity_age_stops": int(stats["display_continuity_age_stops"]),
        "display_continuity_handoff_stops": int(stats["display_continuity_handoff_stops"]),
        "display_continuity_max_extension_frames": int(display_continuity_max_extension),
    }
    logger.info("klt_sidecar_augmentation=%s", summary)
    return summary


def resolve_crop_manifest_path(args: argparse.Namespace, run_out_dir: Path) -> Path:
    if str(getattr(args, "crop_manifest", "")).strip():
        return stage0403.project_path(args.crop_manifest)
    return (run_out_dir / DEFAULT_CROP_MANIFEST_REL).resolve()


def load_crop_assets(manifest_path: Path, source_specs: list[stage0403.SourceSpec]) -> tuple[dict[int, CropAssetMeta], dict[str, Any]]:
    stage0403.validate_file_exists(manifest_path, "Stage 04.05a crop asset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_entries = manifest.get("sources", [])
    if not isinstance(source_entries, list):
        raise SystemExit(f"Invalid crop asset manifest: {manifest_path}")
    entry_by_source: dict[int, dict[str, Any]] = {}
    for item in source_entries:
        if not isinstance(item, dict):
            continue
        try:
            source_id = int(item.get("source_id", -1))
        except (TypeError, ValueError):
            continue
        entry_by_source[source_id] = item

    assets: dict[int, CropAssetMeta] = {}
    for spec in source_specs:
        if spec.source_id not in entry_by_source:
            raise SystemExit(f"Missing crop asset entry for source_id={spec.source_id} in {manifest_path}")
        entry = entry_by_source[spec.source_id]
        crop_meta_path = stage0403.alias_mapped_path(str(entry.get("crop_metadata_path", "")))
        stage0403.validate_file_exists(crop_meta_path, f"crop metadata source_id={spec.source_id}")
        meta = json.loads(crop_meta_path.read_text(encoding="utf-8"))
        crop_clip_path = stage0403.alias_mapped_path(str(meta.get("crop_clip_path", "")))
        source_clip_path = stage0403.alias_mapped_path(str(meta.get("source_clip_path", "")))
        stage0403.validate_file_exists(crop_clip_path, f"crop clip source_id={spec.source_id}")
        stage0403.validate_file_exists(source_clip_path, f"source clip reference source_id={spec.source_id}")
        resize_info = meta.get("resize_or_letterbox", {})
        frame_sync = meta.get("frame_index_sync", {})
        assets[spec.source_id] = CropAssetMeta(
            source_id=int(meta.get("source_id", spec.source_id)),
            clip_label=str(meta.get("clip_label", spec.clip_label)),
            source_clip_path=source_clip_path,
            crop_clip_path=crop_clip_path,
            crop_metadata_path=crop_meta_path,
            source_width=int(meta.get("source_width", 0)),
            source_height=int(meta.get("source_height", 0)),
            source_fps=float(meta.get("source_fps", 0.0) or 0.0),
            source_frame_count=int(meta.get("verified_source_frame_count", meta.get("crop_frame_count", 0)) or 0),
            crop_x=int(meta.get("crop_x", 0)),
            crop_y=int(meta.get("crop_y", 0)),
            crop_width=int(meta.get("crop_width", 0)),
            crop_height=int(meta.get("crop_height", 0)),
            roi_polygon_source=tuple(tuple(map(int, point)) for point in meta.get("original_roi_polygon_source", [])),
            roi_polygon_crop_local=tuple(tuple(map(int, point)) for point in meta.get("roi_polygon_crop_local", [])),
            resize_applied=bool(resize_info.get("resize_applied", resize_info.get("applied", False))),
            letterbox_applied=bool(resize_info.get("letterbox_applied", False)),
            frame_index_mapping=str(frame_sync.get("mapping", "")).strip(),
        )

    return assets, manifest


def should_trigger_reacquire(
    *,
    memory: BoundaryTrackMemory,
    state: str,
    event_type: str,
    frame_num: int,
    cfg: BoundaryReacquireConfig,
) -> bool:
    if state not in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
        return False
    if event_type not in MISSING_EVENT_TYPES:
        return False
    support_frame = max(int(memory.last_real_frame), int(memory.last_kp_refresh_frame))
    if support_frame < 0:
        return False
    if frame_num - support_frame > support_window_frames(memory, cfg):
        return False
    if not memory.last_real_bbox_xyxy and not memory.last_kp_bbox_xyxy:
        return False
    if abs(float(memory.last_boundary_distance_px)) > (cfg.boundary_band_px * boundary_band_scale_for_memory(memory, cfg)):
        return False
    return True


def update_memory_from_real_track(
    *,
    memory: BoundaryTrackMemory,
    frame_num: int,
    state: str,
    bbox_xyxy: list[float],
    overlay: stage0403.OverlaySpec,
) -> None:
    memory.last_state = state
    memory.last_real_frame = int(frame_num)
    memory.last_real_bbox_xyxy = list(bbox_xyxy)
    bbox_w = max(1.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
    bbox_h = max(1.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
    memory.last_bbox_size_wh = (bbox_w, bbox_h)
    memory.last_boundary_distance_px = compute_boundary_distance(bbox_xyxy, overlay)
    memory.last_reacquire_bbox_xyxy = []
    memory.last_reacquire_keypoints_source = []
    memory.last_reacquire_frame = -1
    if not memory.last_kp_bbox_xyxy:
        memory.last_kp_bbox_xyxy = list(bbox_xyxy)


def expected_box_for_reacquire(memory: BoundaryTrackMemory, frame_num: int) -> list[float] | None:
    if memory.last_reacquire_frame == frame_num - 1 and memory.last_reacquire_bbox_xyxy:
        return list(memory.last_reacquire_bbox_xyxy)
    if memory.last_kp_bbox_xyxy:
        return list(memory.last_kp_bbox_xyxy)
    if memory.last_real_bbox_xyxy:
        return list(memory.last_real_bbox_xyxy)
    return None


def should_probe_real_track_for_kp(
    *,
    state: str,
    bbox_xyxy: list[float],
    overlay: stage0403.OverlaySpec,
    cfg: BoundaryReacquireConfig,
) -> bool:
    if state not in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
        return False
    return abs(compute_boundary_distance(bbox_xyxy, overlay)) <= (cfg.boundary_band_px * cfg.real_track_probe_band_scale)


def should_refresh_from_detection(
    detection: ReacquireDetection,
    cfg: BoundaryReacquireConfig,
) -> bool:
    if is_headlike_source(detection.upper_anchor_source):
        return (
            float(detection.upper_anchor_conf) >= cfg.headlike_refresh_conf_thr
            and float(detection.upper_anchor_score) >= cfg.headlike_refresh_score_thr
        )
    if is_shoulder_source(detection.upper_anchor_source):
        return (
            float(detection.upper_anchor_conf) >= cfg.shoulder_refresh_conf_thr
            and float(detection.upper_anchor_score) >= cfg.shoulder_refresh_score_thr
        )
    return False


def refresh_memory_from_detection(
    *,
    memory: BoundaryTrackMemory,
    frame_num: int,
    detection: ReacquireDetection,
    overlay: stage0403.OverlaySpec,
    mark_reacquire_frame: bool,
) -> None:
    bbox_xyxy = list(detection.bbox_source_xyxy)
    memory.last_boundary_distance_px = compute_boundary_distance(bbox_xyxy, overlay)
    if detection.upper_anchor_xy_source is not None:
        if cv2 is not None and np is not None and len(overlay.roi_polygon_source) >= 3:
            contour = np.asarray(overlay.roi_polygon_source, dtype=np.float32)
            memory.last_boundary_distance_px = float(
                cv2.pointPolygonTest(
                    contour,
                    (float(detection.upper_anchor_xy_source[0]), float(detection.upper_anchor_xy_source[1])),
                    True,
                )
            )
        memory.last_kp_anchor_xy_source = tuple(detection.upper_anchor_xy_source)
        memory.last_kp_anchor_source = str(detection.upper_anchor_source)
        memory.last_kp_anchor_kind = str(detection.upper_anchor_kind)
        memory.last_kp_anchor_conf = float(detection.upper_anchor_conf)
        memory.last_kp_anchor_score = float(detection.upper_anchor_score)
        bbox_w = max(1.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
        bbox_h = max(1.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
        memory.last_bbox_size_wh = (bbox_w, bbox_h)
        memory.last_anchor_offset_xy = (
            float(detection.upper_anchor_xy_source[0]) - float(bbox_xyxy[0]),
            float(detection.upper_anchor_xy_source[1]) - float(bbox_xyxy[1]),
        )
        memory.last_kp_refresh_frame = int(frame_num)
        memory.last_kp_bbox_xyxy = bbox_xyxy
        memory.last_kp_keypoints_source = list(detection.keypoints_source)
        memory.kp_missing_streak = 0
    if mark_reacquire_frame:
        memory.last_reacquire_frame = int(frame_num)
        memory.last_reacquire_bbox_xyxy = bbox_xyxy
        memory.last_reacquire_keypoints_source = list(detection.keypoints_source)


def hold_box_is_implausible(
    *,
    memory: BoundaryTrackMemory,
    cfg: BoundaryReacquireConfig,
) -> bool:
    if not memory.last_kp_bbox_xyxy or not memory.last_real_bbox_xyxy:
        return False
    hold_center = bbox_center_xy(memory.last_kp_bbox_xyxy)
    real_center = bbox_center_xy(memory.last_real_bbox_xyxy)
    bbox_w = max(1.0, float(memory.last_real_bbox_xyxy[2]) - float(memory.last_real_bbox_xyxy[0]))
    bbox_h = max(1.0, float(memory.last_real_bbox_xyxy[3]) - float(memory.last_real_bbox_xyxy[1]))
    scale = max(1.0, bbox_w, bbox_h)
    center_drift = math.hypot(hold_center[0] - real_center[0], hold_center[1] - real_center[1])
    if is_headlike_source(memory.last_kp_anchor_source):
        if center_drift > (scale * cfg.headlike_hold_center_drift_ratio):
            return True
    elif is_shoulder_source(memory.last_kp_anchor_source):
        if center_drift > (scale * cfg.shoulder_hold_center_drift_ratio):
            return True
    if memory.last_kp_anchor_xy_source is not None and memory.last_anchor_offset_xy is not None:
        x1, y1, _, _ = map(float, memory.last_kp_bbox_xyxy)
        expected_anchor = (x1 + float(memory.last_anchor_offset_xy[0]), y1 + float(memory.last_anchor_offset_xy[1]))
        anchor_drift = math.hypot(
            float(memory.last_kp_anchor_xy_source[0]) - expected_anchor[0],
            float(memory.last_kp_anchor_xy_source[1]) - expected_anchor[1],
        )
        if is_headlike_source(memory.last_kp_anchor_source):
            return anchor_drift > (scale * cfg.headlike_hold_anchor_drift_ratio)
        if is_shoulder_source(memory.last_kp_anchor_source):
            return anchor_drift > (scale * cfg.shoulder_hold_anchor_drift_ratio)
    return False


def should_draw_kp_hold(
    *,
    memory: BoundaryTrackMemory,
    state: str,
    event_type: str,
    frame_num: int,
    frame_shape: tuple[int, int, int],
    cfg: BoundaryReacquireConfig,
) -> bool:
    if state not in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
        return False
    if event_type not in MISSING_EVENT_TYPES:
        return False
    if not has_strong_kp_refresh(memory, cfg):
        return False
    if not memory.last_kp_bbox_xyxy or memory.last_kp_refresh_frame < 0:
        return False
    if frame_num - memory.last_kp_refresh_frame > support_window_frames(memory, cfg):
        return False
    if memory.kp_missing_streak > hold_missing_limit(memory, cfg):
        return False
    if abs(float(memory.last_boundary_distance_px)) > (cfg.boundary_band_px * boundary_band_scale_for_memory(memory, cfg)):
        return False
    if hold_box_is_implausible(memory=memory, cfg=cfg):
        return False
    frame_h, frame_w = frame_shape[:2]
    box = memory.last_kp_bbox_xyxy
    if box[0] < 0.0 or box[1] < 0.0 or box[2] > float(frame_w) or box[3] > float(frame_h):
        return False
    return True


def compute_kp_hold_strength(
    *,
    memory: BoundaryTrackMemory,
    frame_num: int,
    cfg: BoundaryReacquireConfig,
) -> float:
    if memory.last_kp_refresh_frame < 0 or not memory.last_kp_bbox_xyxy:
        return 0.0
    support_window = max(1, int(support_window_frames(memory, cfg)))
    age_frames = max(0, int(frame_num) - int(memory.last_kp_refresh_frame))
    age_score = clamp01(1.0 - (float(age_frames) / float(support_window)))
    hold_limit = max(1, int(hold_missing_limit(memory, cfg)))
    streak_score = clamp01(1.0 - (float(max(0, int(memory.kp_missing_streak))) / float(hold_limit)))
    hold_kp_score = mean_keypoint_confidence(memory.last_kp_keypoints_source, cfg.keypoint_conf)
    source_bonus = 0.0
    if is_headlike_source(memory.last_kp_anchor_source):
        source_bonus = 0.10
    elif is_shoulder_source(memory.last_kp_anchor_source):
        source_bonus = 0.04
    return clamp01(
        (0.34 * clamp01(memory.last_kp_anchor_score))
        + (0.18 * clamp01(memory.last_kp_anchor_conf))
        + (0.18 * clamp01(hold_kp_score))
        + (0.15 * age_score)
        + (0.15 * streak_score)
        + source_bonus
    )


def heavy_is_clearly_better_than_kp_hold(
    *,
    detection: ReacquireDetection,
    memory: BoundaryTrackMemory,
    frame_num: int,
    cfg: BoundaryReacquireConfig,
) -> tuple[bool, str]:
    if memory.last_kp_refresh_frame < 0 or not memory.last_kp_bbox_xyxy:
        return True, "override_no_kp_hold"
    if hold_box_is_implausible(memory=memory, cfg=cfg):
        return True, "override_implausible_kp_hold"

    hold_strength = compute_kp_hold_strength(memory=memory, frame_num=frame_num, cfg=cfg)
    if hold_strength < cfg.kp_hold_protect_min_strength:
        return True, "override_weak_kp_hold"

    hold_box = list(memory.last_kp_bbox_xyxy)
    hold_kp_score = mean_keypoint_confidence(memory.last_kp_keypoints_source, cfg.keypoint_conf)
    hold_iou = bbox_iou(detection.bbox_source_xyxy, hold_box)
    expected_box = expected_box_for_reacquire(memory, frame_num)
    expected_iou = bbox_iou(detection.bbox_source_xyxy, expected_box) if expected_box is not None else hold_iou
    hold_center = bbox_center_xy(hold_box)
    det_center = bbox_center_xy(detection.bbox_source_xyxy)
    bbox_w = max(1.0, float(hold_box[2]) - float(hold_box[0]))
    bbox_h = max(1.0, float(hold_box[3]) - float(hold_box[1]))
    hold_scale = max(1.0, bbox_w, bbox_h)
    center_score = clamp01(1.0 - (math.hypot(det_center[0] - hold_center[0], det_center[1] - hold_center[1]) / (hold_scale * 1.35)))
    spatial_score = max(float(hold_iou), float(expected_iou), float(center_score))
    if spatial_score < cfg.heavy_override_hold_center_score_min and max(float(hold_iou), float(expected_iou)) < cfg.heavy_override_hold_iou_min:
        return False, "preserve_kp_hold_spatial"

    score_gain = float(detection.selection_score) - float(hold_strength)
    anchor_gain = float(detection.upper_anchor_score) - float(memory.last_kp_anchor_score)
    kp_gain = float(detection.kp_score) - float(hold_kp_score)
    if (
        should_refresh_from_detection(detection, cfg)
        and score_gain >= (cfg.heavy_override_hold_score_gain - 0.02)
        and anchor_gain >= (cfg.heavy_override_hold_anchor_gain * 0.50)
        and spatial_score >= cfg.heavy_override_hold_center_score_min
    ):
        return True, "override_heavy_refresh_gain"
    if score_gain >= cfg.heavy_override_hold_score_gain and anchor_gain >= cfg.heavy_override_hold_anchor_gain:
        return True, "override_heavy_anchor_gain"
    if (
        score_gain >= (cfg.heavy_override_hold_score_gain + 0.04)
        and kp_gain >= cfg.heavy_override_hold_kp_gain
        and float(detection.box_conf) >= cfg.heavy_override_hold_box_conf_min
        and spatial_score >= cfg.heavy_override_hold_center_score_min
    ):
        return True, "override_heavy_pose_gain"
    return False, "preserve_kp_hold_stronger"


def heavy_trigger_reason(
    *,
    light_detection: ReacquireDetection | None,
    memory: BoundaryTrackMemory,
    cfg: BoundaryReacquireConfig,
) -> str:
    if light_detection is None:
        return "light_missing"
    if abs(float(memory.last_boundary_distance_px)) > (cfg.boundary_band_px * cfg.heavy_boundary_band_scale):
        return ""
    if float(light_detection.selection_score) < cfg.heavy_trigger_select_score:
        return "light_selection_weak"
    if float(light_detection.upper_anchor_score) < cfg.heavy_trigger_anchor_score:
        return "light_upper_anchor_weak"
    if float(light_detection.box_conf) < cfg.heavy_trigger_box_conf:
        return "light_box_conf_weak"
    if not should_refresh_from_detection(light_detection, cfg):
        return "light_not_strong_refresh"
    return ""


def choose_reacquire_detection(
    *,
    light_detection: ReacquireDetection | None,
    heavy_detection: ReacquireDetection | None,
    cfg: BoundaryReacquireConfig,
) -> tuple[ReacquireDetection | None, str]:
    if heavy_detection is None:
        return light_detection, "light"
    if light_detection is None:
        return heavy_detection, "heavy"
    heavy_refresh = should_refresh_from_detection(heavy_detection, cfg)
    light_refresh = should_refresh_from_detection(light_detection, cfg)
    if heavy_refresh and not light_refresh:
        return heavy_detection, "heavy"
    if (
        float(heavy_detection.upper_anchor_score) > (float(light_detection.upper_anchor_score) + 0.08)
        and float(heavy_detection.selection_score) >= (float(light_detection.selection_score) - 0.02)
    ):
        return heavy_detection, "heavy"
    if float(heavy_detection.selection_score) > (float(light_detection.selection_score) + 0.05):
        return heavy_detection, "heavy"
    return light_detection, "light"


def draw_dashed_line(
    frame: Any,
    pt0: tuple[int, int],
    pt1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int,
    gap_len: int,
) -> None:
    assert cv2 is not None
    x0, y0 = pt0
    x1, y1 = pt1
    total_len = int(round(math.hypot(x1 - x0, y1 - y0)))
    if total_len <= 0:
        return
    dx = (x1 - x0) / float(total_len)
    dy = (y1 - y0) / float(total_len)
    cursor = 0
    while cursor < total_len:
        seg_end = min(total_len, cursor + dash_len)
        sx = int(round(x0 + (dx * cursor)))
        sy = int(round(y0 + (dy * cursor)))
        ex = int(round(x0 + (dx * seg_end)))
        ey = int(round(y0 + (dy * seg_end)))
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        cursor += dash_len + gap_len


def draw_reacquire_overlay(
    frame: Any,
    *,
    mapped_box: tuple[int, int, int, int],
    mapped_keypoints: list[tuple[int, int]],
) -> None:
    assert cv2 is not None
    color = (255, 210, 80)
    x1, y1, x2, y2 = mapped_box
    draw_dashed_line(frame, (x1, y1), (x2, y1), color, 2, 12, 7)
    draw_dashed_line(frame, (x2, y1), (x2, y2), color, 2, 12, 7)
    draw_dashed_line(frame, (x2, y2), (x1, y2), color, 2, 12, 7)
    draw_dashed_line(frame, (x1, y2), (x1, y1), color, 2, 12, 7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)
    for px, py in mapped_keypoints:
        if px < 0 or py < 0 or px >= frame.shape[1] or py >= frame.shape[0]:
            continue
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, color, -1, cv2.LINE_AA)


def filter_draw_keypoints(
    points_source: list[tuple[float, float, float]],
    *,
    keypoint_conf_thr: float,
    box_source_xyxy: list[float] | None = None,
) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    margin_px = 0.0
    if box_source_xyxy is not None:
        box_w = max(1.0, float(box_source_xyxy[2]) - float(box_source_xyxy[0]))
        box_h = max(1.0, float(box_source_xyxy[3]) - float(box_source_xyxy[1]))
        margin_px = float(max(8, min(20, int(round(min(box_w, box_h) * 0.12)))))
    for idx in REACQUIRE_DRAW_KEYPOINTS:
        if idx >= len(points_source):
            continue
        x, y, conf = points_source[idx]
        if conf < keypoint_conf_thr:
            continue
        if box_source_xyxy is not None:
            if (
                float(x) < float(box_source_xyxy[0]) - margin_px
                or float(x) > float(box_source_xyxy[2]) + margin_px
                or float(y) < float(box_source_xyxy[1]) - margin_px
                or float(y) > float(box_source_xyxy[3]) + margin_px
            ):
                continue
        selected.append((float(x), float(y)))
    return selected


def render_tile(
    *,
    canvas: Any,
    tile_origin: tuple[int, int],
    tile_size: tuple[int, int],
    frame: Any | None,
    crop_frame: Any | None,
    ctx: RenderSourceContext,
    frame_num: int,
    reacquirer: CropPoseReacquirer,
    heavy_reacquirer: CropPoseReacquirer | None,
    cfg: BoundaryReacquireConfig,
    stats: dict[str, Counter[str] | int | str],
) -> int:
    assert cv2 is not None and np is not None
    tile_x, tile_y = tile_origin
    tile_w, tile_h = tile_size
    tile = canvas[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]
    tile[:] = 0

    if frame is not None:
        fit = stage0403.compute_fit_rect(frame.shape[1], frame.shape[0], tile_w, tile_h)
        resized = cv2.resize(frame, (fit.display_width, fit.display_height), interpolation=cv2.INTER_LINEAR)
        tile[fit.pad_y : fit.pad_y + fit.display_height, fit.pad_x : fit.pad_x + fit.display_width] = resized
    else:
        fit = stage0403.compute_fit_rect(ctx.width, ctx.height, tile_w, tile_h)

    frame_rows = ctx.sidecar_rows_by_frame.get(frame_num, {})
    frame_records = ctx.records_by_frame.get(frame_num, {})
    summary = stage0403.summarize_frame_state(frame_rows=frame_rows, frame_records=frame_records, roi_status=ctx.overlay.roi_status)
    drawn_reacquires = 0

    if ctx.overlay.roi_status == "loaded" and ctx.overlay.roi_polygon_source:
        mapped_poly = np.asarray(
            [stage0403.map_point_to_tile(point, fit, tile_origin=(0, 0)) for point in ctx.overlay.roi_polygon_source],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(tile, [mapped_poly], isClosed=True, color=(90, 220, 230), thickness=2, lineType=cv2.LINE_AA)

    for track_id in stage0403.select_track_ids_to_draw(
        frame_rows=frame_rows,
        frame_records=frame_records,
        max_tracks_to_draw=stage0403.MAX_TRACKS_TO_DRAW,
    ):
        row = frame_rows.get(track_id)
        record = frame_records.get(track_id)
        state = str(record.get("state", STATE_OUT)) if record is not None else STATE_OUT
        event_type = str(record.get("event_type", "")).strip() if record is not None else ""
        pose_debug = stage0403.resolve_pose_debug(ctx=ctx, track_id=track_id, state=state, record=record)
        memory = ctx.memories.setdefault(track_id, BoundaryTrackMemory(track_id=track_id))
        real_row = row is not None and row.has_valid_bbox and row.mode != "display_continuity"

        if real_row:
            box_xyxy = list(row.bbox_xyxy)
            mapped_box = stage0403.map_box_to_tile(
                box_xyxy=box_xyxy,
                fit=fit,
                tile_origin=(0, 0),
                tile_w=tile_w,
                tile_h=tile_h,
            )
            if mapped_box is None:
                continue

            stage0403.draw_track_box(
                tile,
                mapped_box=mapped_box,
                track_id=track_id,
                state=state,
                row=row,
                record=record,
                from_record_only=False,
            )
            stage0403.draw_pose_debug(
                tile,
                mapped_box=mapped_box,
                fit=fit,
                state=state,
                pose_debug=pose_debug,
            )
            if state == STATE_IN_CONFIRMED:
                stage0403.draw_confirm_ankle_point(
                    tile,
                    fit=fit,
                    record=record,
                )

            if state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
                update_memory_from_real_track(
                    memory=memory,
                    frame_num=frame_num,
                    state=state,
                    bbox_xyxy=box_xyxy,
                    overlay=ctx.overlay,
                )
                if has_strong_kp_refresh(memory, cfg):
                    memory.last_kp_bbox_xyxy = list(box_xyxy)
                    memory.kp_missing_streak = 0
                if (
                    crop_frame is not None
                    and should_probe_real_track_for_kp(
                        state=state,
                        bbox_xyxy=box_xyxy,
                        overlay=ctx.overlay,
                        cfg=cfg,
                    )
                ):
                    expected_box_crop = source_box_to_crop_local(box_xyxy, ctx.crop_asset)
                    if expected_box_crop is not None:
                        real_detection = reacquirer.detect(
                            crop_frame=crop_frame,
                            expected_box_crop_xyxy=expected_box_crop,
                            crop_asset=ctx.crop_asset,
                        )
                        if real_detection is not None:
                            if should_refresh_from_detection(real_detection, cfg):
                                refresh_memory_from_detection(
                                    memory=memory,
                                    frame_num=frame_num,
                                    detection=real_detection,
                                    overlay=ctx.overlay,
                                    mark_reacquire_frame=False,
                                )
                                refresh_modes = stats["kp_refresh_modes"]
                                refresh_sources = stats["kp_refresh_anchor_sources"]
                                assert isinstance(refresh_modes, Counter)
                                assert isinstance(refresh_sources, Counter)
                                refresh_modes["real_bbox_probe"] += 1
                                refresh_sources[real_detection.upper_anchor_source or "none"] += 1
                            elif memory.last_kp_refresh_frame >= 0:
                                memory.last_kp_bbox_xyxy = list(real_detection.bbox_source_xyxy)
                                memory.last_kp_keypoints_source = list(real_detection.keypoints_source)
                                memory.last_boundary_distance_px = compute_boundary_distance(real_detection.bbox_source_xyxy, ctx.overlay)
                                memory.kp_missing_streak = 0
            continue

        if (
            frame is not None
            and crop_frame is not None
            and should_trigger_reacquire(
                memory=memory,
                state=state,
                event_type=event_type,
                frame_num=frame_num,
                cfg=cfg,
            )
        ):
            hold_drawable = should_draw_kp_hold(
                memory=memory,
                state=state,
                event_type=event_type,
                frame_num=frame_num,
                frame_shape=frame.shape,
                cfg=cfg,
            )
            expected_box_source = expected_box_for_reacquire(memory, frame_num)
            expected_box_crop = source_box_to_crop_local(expected_box_source, ctx.crop_asset) if expected_box_source is not None else None
            light_detection = reacquirer.detect(
                crop_frame=crop_frame,
                expected_box_crop_xyxy=expected_box_crop,
                crop_asset=ctx.crop_asset,
            )
            heavy_detection: ReacquireDetection | None = None
            heavy_reason = ""
            if heavy_reacquirer is not None:
                heavy_reason = heavy_trigger_reason(
                    light_detection=light_detection,
                    memory=memory,
                    cfg=cfg,
                )
                if heavy_reason:
                    stats["heavy_model_invocations"] = int(stats["heavy_model_invocations"]) + 1
                    trigger_reasons = stats["heavy_model_trigger_reasons"]
                    assert isinstance(trigger_reasons, Counter)
                    trigger_reasons[heavy_reason] += 1
                    heavy_detection = heavy_reacquirer.detect(
                        crop_frame=crop_frame,
                        expected_box_crop_xyxy=expected_box_crop,
                        crop_asset=ctx.crop_asset,
                    )
                    if heavy_detection is not None:
                        stats["heavy_model_successes"] = int(stats["heavy_model_successes"]) + 1
            detection, detection_source = choose_reacquire_detection(
                light_detection=light_detection,
                heavy_detection=heavy_detection,
                cfg=cfg,
            )
            if detection is not None and detection_source == "heavy" and hold_drawable:
                heavy_override_allowed, heavy_hold_reason = heavy_is_clearly_better_than_kp_hold(
                    detection=detection,
                    memory=memory,
                    frame_num=frame_num,
                    cfg=cfg,
                )
                hold_decisions = stats["heavy_model_hold_decisions"]
                assert isinstance(hold_decisions, Counter)
                hold_decisions[heavy_hold_reason] += 1
                if heavy_override_allowed:
                    stats["heavy_model_kp_hold_overrides"] = int(stats["heavy_model_kp_hold_overrides"]) + 1
                else:
                    stats["heavy_model_kp_hold_preserved"] = int(stats["heavy_model_kp_hold_preserved"]) + 1
                    detection = None
            if detection is not None:
                memory.last_reacquire_bbox_xyxy = list(detection.bbox_source_xyxy)
                memory.last_reacquire_keypoints_source = list(detection.keypoints_source)
                memory.last_reacquire_frame = int(frame_num)
                memory.kp_missing_streak = 0
                if should_refresh_from_detection(detection, cfg):
                    refresh_memory_from_detection(
                        memory=memory,
                        frame_num=frame_num,
                        detection=detection,
                        overlay=ctx.overlay,
                        mark_reacquire_frame=True,
                    )
                    refresh_modes = stats["kp_refresh_modes"]
                    refresh_sources = stats["kp_refresh_anchor_sources"]
                    assert isinstance(refresh_modes, Counter)
                    assert isinstance(refresh_sources, Counter)
                    refresh_modes["missing_reacquire"] += 1
                    refresh_sources[detection.upper_anchor_source or "none"] += 1
                elif memory.last_kp_refresh_frame >= 0:
                    memory.last_kp_bbox_xyxy = list(detection.bbox_source_xyxy)
                    memory.last_kp_keypoints_source = list(detection.keypoints_source)
                    memory.last_boundary_distance_px = compute_boundary_distance(detection.bbox_source_xyxy, ctx.overlay)
                mapped_box = stage0403.map_box_to_tile(
                    box_xyxy=detection.bbox_source_xyxy,
                    fit=fit,
                    tile_origin=(0, 0),
                    tile_w=tile_w,
                    tile_h=tile_h,
                )
                if mapped_box is not None:
                    stage0403.draw_track_box(
                        tile,
                        mapped_box=mapped_box,
                        track_id=track_id,
                        state=state,
                        row=row,
                        record=record,
                        from_record_only=True,
                    )
                    if state == STATE_IN_CONFIRMED:
                        stage0403.draw_confirm_ankle_point(
                            tile,
                            fit=fit,
                            record=record,
                        )
                    drawn_reacquires += 1
                    cast_states = stats["reacquire_drawn_by_state"]
                    cast_sources = stats["reacquire_drawn_by_source"]
                    cast_modes = stats["reacquire_draw_modes"]
                    assert isinstance(cast_states, Counter)
                    assert isinstance(cast_sources, Counter)
                    assert isinstance(cast_modes, Counter)
                    cast_states[state] += 1
                    cast_sources[str(ctx.spec.source_id)] += 1
                    cast_modes["fresh_reacquire"] += 1
                    if detection_source == "heavy":
                        stats["heavy_model_draws"] = int(stats["heavy_model_draws"]) + 1
                    continue
            if memory.last_kp_refresh_frame >= 0:
                memory.kp_missing_streak += 1
            if hold_drawable:
                hold_box = memory.last_kp_bbox_xyxy
                mapped_hold_box = stage0403.map_box_to_tile(
                    box_xyxy=hold_box,
                    fit=fit,
                    tile_origin=(0, 0),
                    tile_w=tile_w,
                    tile_h=tile_h,
                )
                if mapped_hold_box is not None:
                    stage0403.draw_track_box(
                        tile,
                        mapped_box=mapped_hold_box,
                        track_id=track_id,
                        state=state,
                        row=row,
                        record=record,
                        from_record_only=True,
                    )
                    if state == STATE_IN_CONFIRMED:
                        stage0403.draw_confirm_ankle_point(
                            tile,
                            fit=fit,
                            record=record,
                        )
                    drawn_reacquires += 1
                    cast_states = stats["reacquire_drawn_by_state"]
                    cast_sources = stats["reacquire_drawn_by_source"]
                    cast_modes = stats["reacquire_draw_modes"]
                    assert isinstance(cast_states, Counter)
                    assert isinstance(cast_sources, Counter)
                    assert isinstance(cast_modes, Counter)
                    cast_states[state] += 1
                    cast_sources[str(ctx.spec.source_id)] += 1
                    cast_modes["kp_hold"] += 1
                    continue

        box_xyxy, from_record_only = stage0403.extract_box_for_track(row, record)
        if box_xyxy is None:
            continue
        mapped_box = stage0403.map_box_to_tile(
            box_xyxy=box_xyxy,
            fit=fit,
            tile_origin=(0, 0),
            tile_w=tile_w,
            tile_h=tile_h,
        )
        if mapped_box is None:
            continue
        stage0403.draw_track_box(
            tile,
            mapped_box=mapped_box,
            track_id=track_id,
            state=state,
            row=row,
            record=record,
            from_record_only=from_record_only,
        )
        if state == STATE_IN_CONFIRMED:
            stage0403.draw_confirm_ankle_point(
                tile,
                fit=fit,
                record=record,
            )

    if summary["global_state"] == "INTRUSION":
        basis = str(summary.get("confirm_basis", "")).strip()
        border_color = stage0403.CONFIRM_PROXY_COLOR if basis == "ankle(proxy)" else stage0403.CONFIRM_ANKLE_COLOR
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), border_color, 4, cv2.LINE_AA)

    stage0403._draw_text_chip(
        tile,
        text=ctx.overlay.channel_label,
        x=12,
        y=10,
        align="left",
        scale=0.58,
        thickness=2,
        color=(255, 255, 255),
        bg_alpha=0.62,
    )
    status_line = "ENDED" if frame is None and not ctx.active else stage0403.build_status_line(summary, ctx.overlay.roi_status)
    status_color = (220, 220, 220)
    if summary["global_state"] == "CAND":
        status_color = (0, 255, 255)
    elif summary["global_state"] == "INTRUSION":
        basis = str(summary.get("confirm_basis", "")).strip()
        if basis == "ankle(proxy)":
            status_color = stage0403.CONFIRM_TEXT_PROXY_COLOR
        else:
            status_color = stage0403.CONFIRM_TEXT_ANKLE_COLOR
    elif summary["global_state"] == "NORMAL":
        status_color = (210, 245, 220)
    stage0403._draw_text_chip(
        tile,
        text=status_line,
        x=tile_w - 12,
        y=10,
        align="right",
        scale=0.46,
        thickness=1,
        color=status_color,
        bg_alpha=0.64,
    )
    return drawn_reacquires


def render_multistream_boundary_reacquire(
    *,
    source_specs: list[stage0403.SourceSpec],
    overlay_specs: list[stage0403.OverlaySpec],
    artifacts_by_source: dict[int, stage0403.SourceArtifacts],
    crop_assets: dict[int, CropAssetMeta],
    decision_results: dict[int, dict[str, Any]],
    output_path: Path,
    tiled_size: tuple[int, int],
    logger: logging.Logger,
    reacquirer: CropPoseReacquirer,
    heavy_reacquirer: CropPoseReacquirer | None,
    heavy_request: HeavyModelRequest,
    cfg: BoundaryReacquireConfig,
) -> dict[str, Any]:
    require_render_deps()
    contexts: list[RenderSourceContext] = []

    for spec, overlay in zip(source_specs, overlay_specs):
        artifacts = artifacts_by_source[spec.source_id]
        crop_asset = crop_assets[spec.source_id]
        cap = cv2.VideoCapture(str(spec.local_path))
        crop_cap = cv2.VideoCapture(str(crop_asset.crop_clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source video for Stage 04.05 final render: {spec.local_path}")
        if not crop_cap.isOpened():
            raise RuntimeError(f"Failed to open crop clip for Stage 04.05 final render: {crop_asset.crop_clip_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        crop_fps = float(crop_cap.get(cv2.CAP_PROP_FPS) or 0.0) or fps
        if abs(crop_fps - fps) > 0.01:
            raise RuntimeError(
                f"Crop clip FPS mismatch source_id={spec.source_id}: source={fps} crop={crop_fps}. "
                "Stage 04.05 requires frame-index alignment."
            )

        sidecar_rows_by_frame, sidecar_summary = load_sidecar_rows(artifacts.split_sidecar_path)
        records_by_frame = stage0403.load_events_by_frame(artifacts.events_path)
        contexts.append(
            RenderSourceContext(
                spec=spec,
                overlay=overlay,
                artifacts=artifacts,
                crop_asset=crop_asset,
                cap=cap,
                crop_cap=crop_cap,
                fps=fps,
                width=width,
                height=height,
                frame_num=0,
                sidecar_rows_by_frame=sidecar_rows_by_frame,
                sidecar_summary=sidecar_summary,
                records_by_frame=records_by_frame,
                decision_summary=decision_results.get(spec.source_id, {}),
            )
        )

    output_fps = contexts[0].fps if contexts else 30.0
    fps_values = [round(ctx.fps, 3) for ctx in contexts]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = stage0403.create_video_writer(cv2, output_path, width=tiled_size[0], height=tiled_size[1], fps=output_fps)
    tile_w = tiled_size[0] // TILER_COLUMNS
    tile_h = tiled_size[1] // TILER_ROWS
    frames_written = 0
    stats: dict[str, Counter[str] | int | str] = {
        "reacquire_draw_count": 0,
        "frames_with_reacquire": 0,
        "reacquire_drawn_by_state": Counter(),
        "reacquire_drawn_by_source": Counter(),
        "reacquire_draw_modes": Counter(),
        "kp_refresh_modes": Counter(),
        "kp_refresh_anchor_sources": Counter(),
        "heavy_model_invocations": 0,
        "heavy_model_successes": 0,
        "heavy_model_draws": 0,
        "heavy_model_trigger_reasons": Counter(),
        "heavy_model_kp_hold_overrides": 0,
        "heavy_model_kp_hold_preserved": 0,
        "heavy_model_hold_decisions": Counter(),
        "reacquirer_status": reacquirer.model_status,
        "heavy_reacquirer_status": "disabled" if heavy_reacquirer is None else heavy_reacquirer.model_status,
    }

    try:
        while True:
            canvas = np.zeros((tiled_size[1], tiled_size[0], 3), dtype=np.uint8)
            any_frame = False
            frame_reacquires = 0

            for idx, ctx in enumerate(contexts):
                frame = None
                crop_frame = None
                frame_num = ctx.frame_num
                if ctx.active:
                    ok, next_frame = ctx.cap.read()
                    crop_ok, next_crop = ctx.crop_cap.read()
                    if ok and next_frame is not None:
                        frame = next_frame
                        any_frame = True
                        ctx.frame_num += 1
                    else:
                        ctx.active = False
                    if crop_ok and next_crop is not None:
                        crop_frame = next_crop
                    elif ok and next_frame is not None:
                        raise RuntimeError(
                            f"Crop clip frame underflow source_id={ctx.spec.source_id}; Stage 04.05 requires source/crop frame-index lockstep."
                        )

                tile_row = idx // TILER_COLUMNS
                tile_col = idx % TILER_COLUMNS
                frame_reacquires += render_tile(
                    canvas=canvas,
                    tile_origin=(tile_col * tile_w, tile_row * tile_h),
                    tile_size=(tile_w, tile_h),
                    frame=frame,
                    crop_frame=crop_frame,
                    ctx=ctx,
                    frame_num=frame_num,
                    reacquirer=reacquirer,
                    heavy_reacquirer=heavy_reacquirer,
                    cfg=cfg,
                    stats=stats,
                )

            if not any_frame:
                break

            if frame_reacquires > 0:
                stats["frames_with_reacquire"] = int(stats["frames_with_reacquire"]) + 1
                stats["reacquire_draw_count"] = int(stats["reacquire_draw_count"]) + int(frame_reacquires)

            writer.write(canvas)
            frames_written += 1
    finally:
        writer.release()
        for ctx in contexts:
            ctx.cap.release()
            ctx.crop_cap.release()

    reacquire_by_state = stats["reacquire_drawn_by_state"]
    reacquire_by_source = stats["reacquire_drawn_by_source"]
    reacquire_draw_modes = stats["reacquire_draw_modes"]
    kp_refresh_modes = stats["kp_refresh_modes"]
    kp_refresh_anchor_sources = stats["kp_refresh_anchor_sources"]
    heavy_trigger_reasons = stats["heavy_model_trigger_reasons"]
    heavy_hold_decisions = stats["heavy_model_hold_decisions"]
    assert isinstance(reacquire_by_state, Counter)
    assert isinstance(reacquire_by_source, Counter)
    assert isinstance(reacquire_draw_modes, Counter)
    assert isinstance(kp_refresh_modes, Counter)
    assert isinstance(kp_refresh_anchor_sources, Counter)
    assert isinstance(heavy_trigger_reasons, Counter)
    assert isinstance(heavy_hold_decisions, Counter)
    return {
        "overlay_path": str(output_path),
        "frames_written": int(frames_written),
        "fps": float(output_fps),
        "canvas_size": {"width": int(tiled_size[0]), "height": int(tiled_size[1])},
        "tile_size": {"width": int(tile_w), "height": int(tile_h)},
        "source_fps_values": fps_values,
        "reacquire_frames_with_overlay": int(stats["frames_with_reacquire"]),
        "reacquire_draw_count": int(stats["reacquire_draw_count"]),
        "reacquire_drawn_by_state": dict(reacquire_by_state),
        "reacquire_drawn_by_source": dict(reacquire_by_source),
        "reacquire_draw_modes": dict(reacquire_draw_modes),
        "kp_refresh_modes": dict(kp_refresh_modes),
        "kp_refresh_anchor_sources": dict(kp_refresh_anchor_sources),
        "reacquirer_status": str(stats["reacquirer_status"]),
        "heavy_model_requested": heavy_request.mode,
        "heavy_model_kind": heavy_request.model_kind,
        "heavy_model_path": str(heavy_request.model_path) if heavy_request.model_path is not None else "",
        "heavy_reacquirer_status": str(stats["heavy_reacquirer_status"]),
        "heavy_model_invocations": int(stats["heavy_model_invocations"]),
        "heavy_model_successes": int(stats["heavy_model_successes"]),
        "heavy_model_draws": int(stats["heavy_model_draws"]),
        "heavy_model_trigger_reasons": dict(heavy_trigger_reasons),
        "heavy_model_kp_hold_overrides": int(stats["heavy_model_kp_hold_overrides"]),
        "heavy_model_kp_hold_preserved": int(stats["heavy_model_kp_hold_preserved"]),
        "heavy_model_hold_decisions": dict(heavy_hold_decisions),
        "heavy_model_true_pose11m_used": bool(
            heavy_request.enabled
            and heavy_request.model_kind == "pose"
            and "11m-pose" in str(heavy_request.model_path or "")
            and str(stats["heavy_reacquirer_status"]) == "ready"
        ),
    }


def main() -> None:
    args = parse_args()
    stage0403.normalize_output_args(args)
    if not getattr(args, "out_base", ""):
        args.out_base = DEFAULT_OUT_BASE

    source_specs = stage0403.build_source_specs(args.inputs)
    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    crop_manifest_path = resolve_crop_manifest_path(args, run.out_dir)
    crop_assets, crop_manifest = load_crop_assets(crop_manifest_path, source_specs)
    crop_path_kind = "true_original_resolution_crop"
    for crop_asset in crop_assets.values():
        if crop_asset.resize_applied or crop_asset.letterbox_applied:
            crop_path_kind = "resized_or_letterboxed_crop"
            break

    template_path = stage0403.project_path(args.ds_config_template)
    plugin_lib = stage0403.project_path(args.plugin_lib)
    pose_model_path = stage0403.project_path(args.pose_model)
    heavy_request = resolve_heavy_model_request(args)
    cfg = build_reacquire_config(args)

    stage0403.validate_file_exists(template_path, "DeepStream config template")
    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")
    if not args.dry_run:
        stage0403.validate_file_exists(plugin_lib, "Stage 04.03 intrusion export plugin library")

    preflight_rendered_text, refs = stage0403.render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=Path("/tmp/ds_multistream4_boundary_reacquire_preflight.mp4"),
    )
    streammux_spec = stage0403.parse_streammux_spec(refs)
    tiled_size = stage0403.parse_tiled_size(refs)

    infer_config_local = stage0403.alias_mapped_path(refs["infer_config"])
    tracker_config_local = stage0403.alias_mapped_path(refs["tracker_config"])
    stage0403.validate_file_exists(infer_config_local, "infer config")
    stage0403.validate_file_exists(tracker_config_local, "tracker config")
    stage0403.validate_runtime_path_expectations(
        rendered_config_text=preflight_rendered_text,
        infer_config_local=infer_config_local,
        dry_run=args.dry_run,
    )

    feature_cfg, score_weights, fsm_cfg = stage0403.load_intrusion_defaults()
    if cv2 is None and args.dry_run:
        source_fps = 30.0
    else:
        source_fps = stage0403.probe_video_fps(source_specs[0].local_path)
    grace_frames = int(args.grace_frames)
    if grace_frames < 0:
        grace_sec = float(fsm_cfg.get("grace_sec", 2.0))
        grace_frames = max(0, int(round(grace_sec * source_fps)))

    params = DecisionParams(
        candidate_enter_n=max(1, int(args.candidate_enter_n)),
        confirm_enter_n=max(1, int(args.confirm_enter_n)),
        exit_n=max(1, int(args.exit_n)),
        grace_frames=max(0, int(grace_frames)),
        candidate_iou_or_overlap_thr=float(args.candidate_iou_or_overlap_thr),
        confirm_requires_ankle=True,
        candidate_score_thr=float(fsm_cfg.get("cand_thr", 0.35)),
        proxy_start_max_age_frames=max(1, min(max(0, int(grace_frames)) or 1, 3)),
    )

    overlay_specs = stage0403.load_overlay_specs(source_specs, streammux_spec, logger)
    artifacts_by_source = stage0403.build_source_artifacts(run.out_dir, source_specs)
    tracking_output_video = run.out_dir / f"{args.out_base}_tiled_tracking_export.mp4"
    final_output_video = run.out_dir / f"{args.out_base}_tiled_boundary_reacquire.mp4"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    combined_sidecar_path = run.out_dir / "tracking_sidecar_combined.csv"
    run_summary_path = run.out_dir / "boundary_reacquire_run_summary.json"

    rendered_text, _ = stage0403.render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=tracking_output_video,
    )
    stage0403.save_rendered_config(rendered_config_path, rendered_text)
    runtime_env = stage0403.build_runtime_env(
        os.environ,
        plugin_lib.parent,
        overlay_specs,
        combined_sidecar_path,
    )
    cmd = [deepstream_app or "deepstream-app", "-c", str(rendered_config_path)]
    cmd_str = stage0403.shell_join(cmd)

    crop_fields_used = [
        "source_clip_path",
        "crop_clip_path",
        "source_width",
        "source_height",
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
        "source_fps",
        "verified_source_frame_count",
        "original_roi_polygon_source",
        "roi_polygon_crop_local",
        "resize_or_letterbox",
        "frame_index_sync",
        "coordinate_mapping",
    ]
    kp_continuity_rules = {
        "anchor_priority": [
            "head_like_keypoint:{nose,eye_center,ear_center,head_mean}",
            "shoulder_fallback:{shoulder_center,left_shoulder,right_shoulder}",
            "bbox_upper_proxy:no_extended_kp_hold_bonus",
        ],
        "real_track_kp_probe": [
            f"state in {{{STATE_CANDIDATE},{STATE_IN_CONFIRMED}}}",
            f"abs(boundary_distance)<=reacquire_boundary_band_px*{cfg.real_track_probe_band_scale}",
            "probe uses crop-based pose model matched back to current real bbox",
        ],
        "strong_kp_refresh": {
            "head_like_keypoint": [
                f"anchor_conf>={cfg.headlike_refresh_conf_thr}",
                f"upper_anchor_score>={cfg.headlike_refresh_score_thr}",
            ],
            "shoulder_fallback": [
                f"anchor_conf>={cfg.shoulder_refresh_conf_thr}",
                f"upper_anchor_score>={cfg.shoulder_refresh_score_thr}",
            ],
        },
        "support_windows_frames": {
            "head_like_keypoint": cfg.headlike_support_window_frames,
            "shoulder_fallback": cfg.shoulder_support_window_frames,
            "default_bbox_only": cfg.memory_frames,
        },
        "kp_hold_missing_limits_frames": {
            "head_like_keypoint": cfg.headlike_hold_missing_frames,
            "shoulder_fallback": cfg.shoulder_hold_missing_frames,
            "bbox_upper_proxy": 0,
        },
        "selection_score": "0.33*box_conf + 0.27*iou + 0.16*center + 0.10*kp_score + 0.14*upper_anchor_score",
        "upper_anchor_score": "0.50*anchor_conf + 0.30*kp_score + 0.20*box_conf + source_bonus(head_like=0.18, shoulder=0.08)",
        "hold_stop_conditions": [
            "real bbox returns immediately",
            "support age exceeds source-specific support window",
            "kp_missing_streak exceeds source-specific hold limit",
            "assist leaves the boundary-relevant band",
            "assist box goes out of frame",
            f"head-like hold center drift>{cfg.headlike_hold_center_drift_ratio}x last real bbox scale",
            f"shoulder hold center drift>{cfg.shoulder_hold_center_drift_ratio}x last real bbox scale",
        ],
    }
    heavy_model_meta = {
        "requested_mode": heavy_request.mode,
        "enabled": heavy_request.enabled,
        "model_kind": heavy_request.model_kind,
        "weights_path": str(heavy_request.model_path) if heavy_request.model_path is not None else "",
        "explicit_path_override_supported": True,
        "explicit_path_override_requested": bool(str(getattr(args, "reacquire_heavy_model_path", "")).strip()),
        "activation_gating": [
            "only inside Stage 04.05 ROI-crop reacquire path",
            "only after the ordinary crop reacquire is missing or weak",
            "only in candidate/grace/lost-like boundary support states",
            f"only when abs(boundary_distance)<=reacquire_boundary_band_px*{cfg.heavy_boundary_band_scale}",
        ],
        "weak_light_reacquire_conditions": [
            "light_detection is missing",
            f"light_selection_score<{cfg.heavy_trigger_select_score}",
            f"light_upper_anchor_score<{cfg.heavy_trigger_anchor_score}",
            f"light_box_conf<{cfg.heavy_trigger_box_conf}",
            "light_detection is not a strong KP refresh",
        ],
        "kp_hold_protection": {
            "protect_if_hold_strength_at_least": cfg.kp_hold_protect_min_strength,
            "preserve_hold_if_heavy_spatial_score_below": cfg.heavy_override_hold_center_score_min,
            "preserve_hold_if_heavy_iou_below": cfg.heavy_override_hold_iou_min,
            "heavy_override_requires_one_of": [
                f"strong_refresh and selection_gain>={cfg.heavy_override_hold_score_gain - 0.02:.2f} and anchor_gain>={cfg.heavy_override_hold_anchor_gain * 0.50:.2f}",
                f"selection_gain>={cfg.heavy_override_hold_score_gain} and anchor_gain>={cfg.heavy_override_hold_anchor_gain}",
                f"selection_gain>={cfg.heavy_override_hold_score_gain + 0.04:.2f} and kp_gain>={cfg.heavy_override_hold_kp_gain} and box_conf>={cfg.heavy_override_hold_box_conf_min}",
            ],
        },
    }

    run_meta: dict[str, Any] = {
        "stage": STAGE,
        "stage_step": STAGE_STEP,
        "run_ts": run.run_ts,
        "dry_run": bool(args.dry_run),
        "out_dir": str(run.out_dir),
        "tracking_output_video": str(tracking_output_video),
        "final_output_video": str(final_output_video),
        "combined_sidecar_path": str(combined_sidecar_path),
        "rendered_config_path": str(rendered_config_path),
        "crop_manifest_path": str(crop_manifest_path),
        "crop_path_kind": crop_path_kind,
        "crop_metadata_fields_used": crop_fields_used,
        "frame_index_sync_statement": "04_05a writes one crop frame for every source frame with no resize or frame dropping; 04_05 reads the source clip and crop clip sequentially in lockstep so crop frame n corresponds to source frame n.",
        "coordinate_mapping_statement": {
            "crop_local_to_source": "source_x = crop_x + crop_local_x, source_y = crop_y + crop_local_y",
            "source_to_main_display": "stage0403.compute_fit_rect + stage0403.map_box_to_tile/map_point_to_tile",
        },
        "confirmed_intrusion_definition": "real_ankle_in_roi_or_klt_lower_body_proxy_confirm",
        "candidate_definition": "overlap_or_(near_roi_boundary_and_moving_toward_roi)",
        "reacquire_changes_truth_semantics": True,
        "kp_continuity_rules": kp_continuity_rules,
        "heavy_model_experiment": heavy_model_meta,
        "crop_manifest": crop_manifest,
    }
    dump_run_meta(run.out_dir, run_meta)

    logger.info("crop_manifest_path=%s", crop_manifest_path)
    logger.info("crop_path_kind=%s", crop_path_kind)
    logger.info("rendered_config_path=%s", rendered_config_path)
    logger.info("tracking_output_video=%s", tracking_output_video)
    logger.info("final_output_video=%s", final_output_video)
    logger.info(
        "reacquire=boundary_band_px=%s memory_frames=%s pose_imgsz=%s pose_conf=%s keypoint_conf=%s",
        cfg.boundary_band_px,
        cfg.memory_frames,
        cfg.pose_imgsz,
        cfg.pose_conf,
        cfg.keypoint_conf,
    )
    logger.info(
        "heavy_model=requested=%s kind=%s path=%s",
        heavy_request.mode,
        heavy_request.model_kind,
        heavy_model_meta["weights_path"],
    )

    print(f"crop manifest: {crop_manifest_path}")
    print(f"crop path kind: {crop_path_kind}")
    print("frame index sync: crop_frame_index == source_frame_index")
    print(f"tracking export video: {tracking_output_video}")
    print(f"final boundary reacquire video: {final_output_video}")
    print(f"heavy model requested: {heavy_request.mode}")
    print("main output stays the existing 4-tile monitoring view")
    print("Stage 04.05 uses ankle-first plus conservative KLT lower-body proxy confirm semantics")
    if run.log_path is not None:
        print(f"log saved: {run.log_path}")
    if run.cmd_path is not None:
        print(f"wrapper cmd saved: {run.cmd_path}")

    if args.dry_run:
        logger.info("dry_run requested; not invoking DeepStream, decision, or final boundary-reacquire render stages")
        return

    exit_code = stage0403.stream_process_output(cmd, logger, runtime_env, prefix="deepstream-app")
    if exit_code != 0:
        logger.error("deepstream-app exited with code %s", exit_code)
        raise SystemExit(exit_code)
    logger.info("DeepStream tracking/export pass completed successfully for Stage 04.05")

    split_summary = stage0403.split_sidecar_by_source(combined_sidecar_path, source_specs, overlay_specs, artifacts_by_source)
    logger.info("split_sidecar_summary=%s", split_summary)

    reacquirer = CropPoseReacquirer(
        settings=CropReacquireSettings(
            model_path=str(pose_model_path),
            input_size=cfg.pose_imgsz,
            conf=cfg.pose_conf,
            keypoint_conf=cfg.keypoint_conf,
            min_box_conf=cfg.min_box_conf,
            max_center_dist_norm=cfg.max_center_dist_norm,
            min_select_score=cfg.min_select_score,
            label="light",
        )
    )
    klt_sidecar_summaries: dict[int, dict[str, Any]] = {}
    for spec, overlay in zip(source_specs, overlay_specs):
        artifacts = artifacts_by_source[spec.source_id]
        klt_sidecar_summaries[spec.source_id] = augment_split_sidecar_with_klt_continuity(
            spec=spec,
            overlay=overlay,
            split_sidecar_path=artifacts.split_sidecar_path,
            pose_reacquirer=reacquirer,
            params=params,
            logger=logger,
        )

    decision_results: dict[int, dict[str, Any]] = {}
    per_source_meta: list[dict[str, Any]] = []
    pose_probe_settings = PoseProbeSettings(model_path=str(pose_model_path))
    for spec, overlay in zip(source_specs, overlay_specs):
        artifacts = artifacts_by_source[spec.source_id]
        split_sidecar_path = artifacts.split_sidecar_path
        if overlay.roi_status != "loaded":
            logger.warning(
                "source_id=%s clip_label=%s has roi_status=%s; skipping ankle-confirm decision pass and keeping stream in NO ROI mode.",
                spec.source_id,
                spec.clip_label,
                overlay.roi_status,
            )
            decision_summary = stage0403.make_skip_decision_summary(
                spec=spec,
                overlay=overlay,
                artifacts=artifacts,
                split_sidecar_path=split_sidecar_path,
                reason=f"roi_{overlay.roi_status}",
                params=params,
                pose_model_path=pose_model_path,
            )
        else:
            artifacts.work_dir.mkdir(parents=True, exist_ok=True)
            decision_summary = run_intrusion_decision_pass(
                video_path=spec.local_path,
                roi_json=overlay.roi_json,
                sidecar_csv=split_sidecar_path,
                events_path=artifacts.events_path,
                params=params,
                feature_cfg=feature_cfg,
                score_weights=score_weights,
                pose_probe_settings=pose_probe_settings,
            )
            decision_summary.update(
                {
                    "source_id": int(spec.source_id),
                    "clip_label": spec.clip_label,
                    "roi_status": overlay.roi_status,
                    "confirmed_intrusion_definition": "real_ankle_in_roi_or_klt_lower_body_proxy_confirm",
                    "candidate_definition": "overlap_or_(near_roi_boundary_and_moving_toward_roi)",
                    "reacquire_changes_truth_semantics": True,
                }
            )
            write_json(artifacts.summary_path, decision_summary)

        decision_results[spec.source_id] = decision_summary
        per_source_meta.append(
            {
                "source_id": spec.source_id,
                "clip_label": spec.clip_label,
                "roi_status": overlay.roi_status,
                "split_sidecar_path": str(split_sidecar_path),
                "events_path": str(artifacts.events_path),
                "summary_path": str(artifacts.summary_path),
                "crop_clip_path": str(crop_assets[spec.source_id].crop_clip_path),
                "crop_metadata_path": str(crop_assets[spec.source_id].crop_metadata_path),
                "confirmed_events": int(decision_summary.get("confirmed_events", 0)),
                "pose_probe_status": str(decision_summary.get("pose_probe_status", "")),
                "skip_reason": str(decision_summary.get("skip_reason", "")),
                "klt_sidecar_augmentation": klt_sidecar_summaries.get(spec.source_id, {}),
            }
        )
    heavy_reacquirer: CropPoseReacquirer | None = None
    if heavy_request.enabled and heavy_request.model_kind == "pose" and heavy_request.model_path is not None:
        heavy_reacquirer = CropPoseReacquirer(
            settings=CropReacquireSettings(
                model_path=str(heavy_request.model_path),
                input_size=max(cfg.pose_imgsz, int(args.reacquire_heavy_pose_imgsz)),
                conf=min(cfg.pose_conf, float(args.reacquire_heavy_pose_conf)),
                keypoint_conf=cfg.keypoint_conf,
                min_box_conf=min(cfg.min_box_conf, float(args.reacquire_heavy_pose_conf)),
                max_center_dist_norm=cfg.max_center_dist_norm,
                min_select_score=cfg.min_select_score,
                label=f"heavy:{heavy_request.mode}",
            )
        )
    render_summary = render_multistream_boundary_reacquire(
        source_specs=source_specs,
        overlay_specs=overlay_specs,
        artifacts_by_source=artifacts_by_source,
        crop_assets=crop_assets,
        decision_results=decision_results,
        output_path=final_output_video,
        tiled_size=tiled_size,
        logger=logger,
        reacquirer=reacquirer,
        heavy_reacquirer=heavy_reacquirer,
        heavy_request=heavy_request,
        cfg=cfg,
    )

    run_summary = {
        "tracking_export_video": str(tracking_output_video),
        "final_boundary_reacquire_video": str(final_output_video),
        "combined_sidecar_path": str(combined_sidecar_path),
        "split_sidecar_summary": split_summary,
        "per_source": per_source_meta,
        "crop_manifest_path": str(crop_manifest_path),
        "crop_path_kind": crop_path_kind,
        "crop_metadata_fields_used": crop_fields_used,
        "frame_index_sync_statement": "Stage 04.05a writes every source frame into the crop clip at the same FPS with no dropping, and Stage 04.05 reads the source video and crop video sequentially in lockstep, so frame n in the source clip uses frame n in the crop clip.",
        "coordinate_mapping_statement": {
            "crop_local_to_source": "source_x = crop_x + crop_local_x, source_y = crop_y + crop_local_y",
            "source_to_main_display": "stage0403.compute_fit_rect + stage0403.map_box_to_tile/map_point_to_tile",
        },
        "confirmed_intrusion_definition": "real_ankle_in_roi_or_klt_lower_body_proxy_confirm",
        "candidate_definition": "overlap_or_(near_roi_boundary_and_moving_toward_roi)",
        "klt_sidecar_augmentation": klt_sidecar_summaries,
        "boundary_reacquire": {
            "activation_conditions": [
                "state_is_candidate_or_confirmed",
                "event_type in {candidate_grace,candidate_lost,in_grace,in_lost}",
                "recent_support_exists_from_real_bbox_or_strong_upper_kp_refresh",
                f"boundary_relevant_band_base={cfg.boundary_band_px}px",
                "main_bbox_missing_or_invalid",
                "crop_manifest_and_crop_clip_available",
            ],
            "crop_path_kind": crop_path_kind,
            "kp_continuity_rules": kp_continuity_rules,
            "heavy_model_experiment": heavy_model_meta,
            "truth_semantics_changed": True,
        },
        "render_summary": render_summary,
        "confirmed_events_total": int(sum(int(item.get("confirmed_events", 0)) for item in per_source_meta)),
        "confirmed_truth_semantics_changed": True,
    }
    write_json(run_summary_path, run_summary)

    run_meta["split_sidecar_summary"] = split_summary
    run_meta["per_source"] = per_source_meta
    run_meta["render_summary"] = render_summary
    run_meta["confirmed_events_total"] = run_summary["confirmed_events_total"]
    dump_run_meta(run.out_dir, run_meta)

    logger.info(
        "Stage 04.05 final render completed output=%s confirmed_events_total=%s reacquire_draw_count=%s",
        final_output_video,
        run_summary["confirmed_events_total"],
        render_summary.get("reacquire_draw_count", 0),
    )


if __name__ == "__main__":
    main()
