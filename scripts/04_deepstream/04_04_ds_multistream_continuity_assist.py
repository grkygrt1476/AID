#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import shutil
import sys
from collections import Counter, deque
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
STAGE_STEP = "04.04"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_TEMPLATE = stage0403.DEFAULT_TEMPLATE
DEFAULT_PLUGIN_LIB = stage0403.DEFAULT_PLUGIN_LIB
DEFAULT_POSE_MODEL = stage0403.DEFAULT_POSE_MODEL
DEFAULT_OUT_BASE = "multistream4_continuity_assist"
SOURCE_COUNT = stage0403.SOURCE_COUNT
TILER_ROWS = stage0403.TILER_ROWS
TILER_COLUMNS = stage0403.TILER_COLUMNS
MISSING_EVENT_TYPES = {"candidate_grace", "candidate_lost", "in_grace", "in_lost"}
HEAD_KEYPOINT_INDICES = [0, 1, 2, 3, 4]
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6


@dataclass(frozen=True)
class ContinuityAssistConfig:
    boundary_band_px: float
    internal_memory_frames: int
    visible_min_frames: int
    visible_max_frames: int
    motion_fast_px: float
    shape_instability_thr: float
    klt_max_corners: int
    klt_quality_level: float
    klt_min_distance: int
    klt_block_size: int
    klt_patch_width_ratio: float
    klt_patch_height_ratio: float
    klt_patch_y_offset_ratio: float
    strong_patch_quality_thr: float
    strong_klt_quality_thr: float
    strong_motion_score_min: float
    headlike_min_visible_floor: int
    shoulder_min_visible_floor: int
    headlike_activation_band_scale: float
    shoulder_activation_band_scale: float
    proxy_activation_band_scale: float
    experiment_uncapped_headlike: bool
    experiment_headlike_min_patch_quality: float
    experiment_headlike_min_klt_quality: float
    experiment_headlike_max_missing_anchor_frames: int
    experiment_headlike_max_center_drift_ratio: float
    experiment_headlike_max_anchor_drift_ratio: float


@dataclass(frozen=True)
class UpperAnchorProbeSettings:
    model_path: str
    input_size: int = 640
    conf: float = 0.25
    keypoint_conf: float = 0.35
    pad_x_ratio: float = 0.12
    pad_top_ratio: float = 0.08
    pad_bottom_ratio: float = 0.05


@dataclass(frozen=True)
class UpperAnchorResult:
    point_xy: tuple[float, float]
    source: str
    kind: str
    confidence: float


@dataclass
class TrackAssistMemory:
    track_id: int
    last_state: str = STATE_OUT
    last_real_frame: int = -1
    last_bbox_xyxy: list[float] = field(default_factory=list)
    assist_bbox_xyxy: list[float] = field(default_factory=list)
    last_anchor_xy: tuple[float, float] | None = None
    assist_anchor_xy: tuple[float, float] | None = None
    last_anchor_source: str = ""
    last_anchor_kind: str = ""
    last_anchor_confidence: float = 0.0
    last_boundary_distance_px: float = float("inf")
    last_bbox_overlap: float = 0.0
    last_motion_mag_px: float = 0.0
    last_shape_instability: float = 1.0
    motion_score: float = 0.0
    shape_score: float = 0.0
    last_seed_quality: float = 0.0
    last_patch_quality: float = 0.0
    last_klt_quality: float = 0.0
    last_stability_score: float = 0.0
    last_visible_limit_frames: int = 0
    memory_until_frame: int = -1
    anchor_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=6))
    bbox_history: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=6))
    klt_points: Any = None
    klt_point_count: int = 0
    missing_anchor_evidence_streak: int = 0
    experiment_uncapped_headlike_active: bool = False


@dataclass(frozen=True)
class AssistDrawSpec:
    bbox_xyxy: list[float]
    anchor_xy: tuple[float, float]
    anchor_source: str
    anchor_kind: str
    klt_quality: float
    patch_quality: float
    motion_score: float
    shape_score: float
    stability_score: float
    visible_limit_frames: int
    missing_age_frames: int
    boundary_distance_px: float
    experiment_uncapped_headlike: bool


@dataclass
class RenderSourceContext:
    spec: stage0403.SourceSpec
    overlay: stage0403.OverlaySpec
    artifacts: stage0403.SourceArtifacts
    cap: Any
    fps: float
    width: int
    height: int
    frame_num: int
    sidecar_rows_by_frame: dict[int, dict[int, SidecarRow]]
    sidecar_summary: dict[str, Any]
    records_by_frame: dict[int, dict[int, dict[str, Any]]]
    decision_summary: dict[str, Any]
    pose_debug_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    assist_memories: dict[int, TrackAssistMemory] = field(default_factory=dict)
    prev_gray: Any = None
    active: bool = True


class UpperAnchorProbe:
    def __init__(self, settings: UpperAnchorProbeSettings):
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
        except Exception as exc:  # pragma: no cover - runtime import depends on env
            self._load_status = f"ultralytics_import_failed:{exc.__class__.__name__}"
            return

        try:
            self._model = YOLO(str(model_path_obj))
        except Exception as exc:  # pragma: no cover - runtime model load depends on env
            self._load_status = f"pose_model_load_failed:{exc.__class__.__name__}"
            return

        self._load_status = "ready"

    def _select_pose_candidate(self, result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        if np is None:
            return None, None
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None, None

        keypoints_xy = to_numpy(getattr(keypoints, "xy", None))
        if keypoints_xy is None or keypoints_xy.ndim != 3 or keypoints_xy.shape[0] == 0:
            return None, None

        keypoints_conf = to_numpy(getattr(keypoints, "conf", None))
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
            box_conf = to_numpy(boxes.conf)
            if box_conf is not None and len(box_conf) >= keypoints_xy.shape[0]:
                best_idx = int(np.argmax(box_conf.reshape(-1)[: keypoints_xy.shape[0]]))
        elif keypoints_conf.ndim == 2:
            conf_score = np.nanmean(keypoints_conf, axis=1)
            best_idx = int(np.argmax(conf_score))

        return keypoints_xy[best_idx].astype(np.float32), keypoints_conf[best_idx].astype(np.float32).reshape(-1)

    def probe(self, frame: np.ndarray, bbox_xyxy: list[float]) -> UpperAnchorResult:
        fallback_xy = compute_bbox_upper_proxy(bbox_xyxy)
        fallback = UpperAnchorResult(
            point_xy=fallback_xy,
            source="bbox_upper_proxy",
            kind="bbox_upper_proxy",
            confidence=0.0,
        )

        self._ensure_model()
        if self._model is None or np is None:
            return fallback

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        crop_x1 = x1 - bbox_w * float(self.settings.pad_x_ratio)
        crop_x2 = x2 + bbox_w * float(self.settings.pad_x_ratio)
        crop_y1 = y1 - bbox_h * float(self.settings.pad_top_ratio)
        crop_y2 = y2 + bbox_h * float(self.settings.pad_bottom_ratio)
        x1i, y1i, x2i, y2i = clamp_xyxy(crop_x1, crop_y1, crop_x2, crop_y2, frame_w, frame_h)
        crop = frame[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            return fallback

        try:
            results = self._model.predict(
                source=crop,
                imgsz=int(self.settings.input_size),
                conf=float(self.settings.conf),
                verbose=False,
                stream=False,
            )
        except Exception:  # pragma: no cover - runtime inference depends on env
            return fallback

        result = results[0] if results else None
        if result is None:
            return fallback

        keypoints_xy, keypoints_conf = self._select_pose_candidate(result)
        if keypoints_xy is None or keypoints_conf is None:
            return fallback

        keypoints_xy[:, 0] += float(x1i)
        keypoints_xy[:, 1] += float(y1i)
        resolved = resolve_upper_anchor(
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            bbox_xyxy=bbox_xyxy,
            min_conf=float(self.settings.keypoint_conf),
        )
        return resolved or fallback


def require_render_deps() -> None:
    stage0403.require_render_deps()


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
    parser.add_argument("--pose_model", default=str(DEFAULT_POSE_MODEL), help="Pose model path reused for 04.03 truth semantics and 04.04 upper-anchor probing.")
    parser.add_argument("--candidate_enter_n", type=int, default=2, help="Consecutive candidate frames required to enter CANDIDATE.")
    parser.add_argument("--confirm_enter_n", type=int, default=1, help="Consecutive ankle-in-ROI frames required to enter IN_CONFIRMED.")
    parser.add_argument("--exit_n", type=int, default=5, help="Sustained no-evidence frames required to return to OUT after grace.")
    parser.add_argument("--grace_frames", type=int, default=-1, help="Evidence-loss grace in frames. Default is derived from configs/intrusion/mvp_v1.yaml and source FPS.")
    parser.add_argument("--candidate_iou_or_overlap_thr", type=float, default=0.05, help="ROI overlap threshold used for weak candidate geometry.")
    parser.add_argument("--out_dir", default="", help="Alias for --out_root; output root directory for this stage.")
    parser.add_argument("--assist_boundary_band_px", type=float, default=72.0, help="Boundary-near band in source pixels where 04.04 assist is allowed.")
    parser.add_argument("--assist_internal_memory_frames", type=int, default=120, help="Internal memory window retained for possible short relink handling.")
    parser.add_argument("--assist_visible_min_frames", type=int, default=4, help="Minimum visible assist hold when stability is weak.")
    parser.add_argument("--assist_visible_max_frames", type=int, default=18, help="Maximum visible assist hold when stability is strong.")
    parser.add_argument("--assist_motion_fast_px", type=float, default=24.0, help="Recent anchor motion scale where old assist becomes stale quickly.")
    parser.add_argument("--assist_shape_instability_thr", type=float, default=0.35, help="Mean relative bbox shape-change level that collapses shape stability.")
    parser.add_argument("--assist_klt_max_corners", type=int, default=18, help="Max Shi-Tomasi corners seeded in the upper-body assist patch.")
    parser.add_argument("--assist_klt_quality_level", type=float, default=0.01, help="Shi-Tomasi quality level for assist KLT seeding.")
    parser.add_argument("--assist_klt_min_distance", type=int, default=5, help="Minimum distance between assist KLT corners.")
    parser.add_argument("--assist_klt_patch_width_ratio", type=float, default=0.70, help="Upper-body assist patch width as a ratio of bbox width.")
    parser.add_argument("--assist_klt_patch_height_ratio", type=float, default=0.52, help="Upper-body assist patch height as a ratio of bbox height.")
    parser.add_argument("--assist_klt_patch_y_offset_ratio", type=float, default=0.10, help="Vertical offset applied below the chosen upper anchor for KLT seeding.")
    parser.add_argument(
        "--assist_experiment_uncapped_headlike",
        action="store_true",
        help="Experiment/debug mode: remove the fixed visible-frame cap for strong head-like assist and stop only on confidence-based termination rules.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs, write the runtime config, and print the planned commands without executing DeepStream, decision, or continuity-assist render stages.")
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


def clamp_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1i = max(0, min(int(width) - 1, int(round(x1))))
    y1i = max(0, min(int(height) - 1, int(round(y1))))
    x2i = max(0, min(int(width) - 1, int(round(x2))))
    y2i = max(0, min(int(height) - 1, int(round(y2))))
    if x2i <= x1i:
        x2i = min(int(width) - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(int(height) - 1, y1i + 1)
    return x1i, y1i, x2i, y2i


def get_valid_keypoint(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, idx: int, min_conf: float) -> np.ndarray | None:
    if idx >= len(keypoints_xy) or idx >= len(keypoints_conf):
        return None
    conf = float(keypoints_conf[idx])
    point = keypoints_xy[idx]
    if conf < min_conf or not np.all(np.isfinite(point)):
        return None
    return point.astype(np.float32)


def mean_valid_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, indices: list[int], min_conf: float) -> np.ndarray | None:
    if np is None:
        return None
    points = [get_valid_keypoint(keypoints_xy, keypoints_conf, idx, min_conf) for idx in indices]
    points = [point for point in points if point is not None]
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0).astype(np.float32)


def compute_bbox_upper_proxy(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return (0.5 * (x1 + x2), y1 + ((y2 - y1) * 0.2))


def resolve_upper_anchor(
    *,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    min_conf: float,
) -> UpperAnchorResult | None:
    x1, y1, x2, y2 = map(float, bbox_xyxy)

    nose = get_valid_keypoint(keypoints_xy, keypoints_conf, 0, min_conf)
    eye_center = mean_valid_keypoints(keypoints_xy, keypoints_conf, [1, 2], min_conf)
    ear_center = mean_valid_keypoints(keypoints_xy, keypoints_conf, [3, 4], min_conf)
    head_point = mean_valid_keypoints(keypoints_xy, keypoints_conf, HEAD_KEYPOINT_INDICES, min_conf)
    left_shoulder = get_valid_keypoint(keypoints_xy, keypoints_conf, LEFT_SHOULDER_IDX, min_conf)
    right_shoulder = get_valid_keypoint(keypoints_xy, keypoints_conf, RIGHT_SHOULDER_IDX, min_conf)

    candidates: list[tuple[np.ndarray, str, str, float]] = []
    if nose is not None:
        candidates.append((nose, "head_like_keypoint", "nose", float(keypoints_conf[0])))
    if eye_center is not None:
        eye_conf = float(np.nanmean([float(keypoints_conf[idx]) for idx in [1, 2] if idx < len(keypoints_conf)]))
        candidates.append((eye_center, "head_like_keypoint", "eye_center", eye_conf))
    if ear_center is not None:
        ear_conf = float(np.nanmean([float(keypoints_conf[idx]) for idx in [3, 4] if idx < len(keypoints_conf)]))
        candidates.append((ear_center, "head_like_keypoint", "ear_center", ear_conf))
    if head_point is not None:
        head_conf = float(np.nanmean([float(keypoints_conf[idx]) for idx in HEAD_KEYPOINT_INDICES if idx < len(keypoints_conf)]))
        candidates.append((head_point, "head_like_keypoint", "head_mean", head_conf))

    if candidates:
        point, source, kind, conf = max(candidates, key=lambda item: item[3])
        point = point.astype(np.float32)
        point[0] = float(np.clip(point[0], x1, x2))
        point[1] = float(np.clip(point[1], y1, y2))
        return UpperAnchorResult(point_xy=(float(point[0]), float(point[1])), source=source, kind=kind, confidence=conf)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
        shoulder_mid[0] = float(np.clip(shoulder_mid[0], x1, x2))
        shoulder_mid[1] = float(np.clip(shoulder_mid[1], y1, y2))
        shoulder_conf = float(np.nanmean([float(keypoints_conf[LEFT_SHOULDER_IDX]), float(keypoints_conf[RIGHT_SHOULDER_IDX])]))
        return UpperAnchorResult(
            point_xy=(float(shoulder_mid[0]), float(shoulder_mid[1])),
            source="shoulder_fallback",
            kind="shoulder_center",
            confidence=shoulder_conf,
        )

    if left_shoulder is not None:
        return UpperAnchorResult(
            point_xy=(float(left_shoulder[0]), float(left_shoulder[1])),
            source="shoulder_fallback",
            kind="left_shoulder",
            confidence=float(keypoints_conf[LEFT_SHOULDER_IDX]),
        )
    if right_shoulder is not None:
        return UpperAnchorResult(
            point_xy=(float(right_shoulder[0]), float(right_shoulder[1])),
            source="shoulder_fallback",
            kind="right_shoulder",
            confidence=float(keypoints_conf[RIGHT_SHOULDER_IDX]),
        )
    return None


def build_assist_config(args: argparse.Namespace) -> ContinuityAssistConfig:
    visible_min = max(1, int(args.assist_visible_min_frames))
    visible_max = max(visible_min, int(args.assist_visible_max_frames))
    return ContinuityAssistConfig(
        boundary_band_px=max(1.0, float(args.assist_boundary_band_px)),
        internal_memory_frames=max(1, int(args.assist_internal_memory_frames)),
        visible_min_frames=visible_min,
        visible_max_frames=visible_max,
        motion_fast_px=max(1.0, float(args.assist_motion_fast_px)),
        shape_instability_thr=max(0.05, float(args.assist_shape_instability_thr)),
        klt_max_corners=max(4, int(args.assist_klt_max_corners)),
        klt_quality_level=max(1e-4, float(args.assist_klt_quality_level)),
        klt_min_distance=max(1, int(args.assist_klt_min_distance)),
        klt_block_size=3,
        klt_patch_width_ratio=max(0.2, float(args.assist_klt_patch_width_ratio)),
        klt_patch_height_ratio=max(0.2, float(args.assist_klt_patch_height_ratio)),
        klt_patch_y_offset_ratio=float(args.assist_klt_patch_y_offset_ratio),
        strong_patch_quality_thr=0.72,
        strong_klt_quality_thr=0.45,
        strong_motion_score_min=0.28,
        headlike_min_visible_floor=max(visible_min, min(visible_max, 10)),
        shoulder_min_visible_floor=max(visible_min, min(visible_max, 8)),
        headlike_activation_band_scale=1.35,
        shoulder_activation_band_scale=1.18,
        proxy_activation_band_scale=1.0,
        experiment_uncapped_headlike=bool(args.assist_experiment_uncapped_headlike),
        experiment_headlike_min_patch_quality=0.42,
        experiment_headlike_min_klt_quality=0.18,
        experiment_headlike_max_missing_anchor_frames=6,
        experiment_headlike_max_center_drift_ratio=1.85,
        experiment_headlike_max_anchor_drift_ratio=1.45,
    )


def is_relevant_state(state: str) -> bool:
    return state in {STATE_CANDIDATE, STATE_IN_CONFIRMED}


def bbox_dimensions(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / max(1.0, height)
    return width, height, aspect


def compute_motion_mag(anchor_history: deque[tuple[float, float]]) -> float:
    if len(anchor_history) < 2:
        return 0.0
    total = 0.0
    count = 0
    prev = None
    for point in anchor_history:
        if prev is not None:
            total += math.hypot(point[0] - prev[0], point[1] - prev[1])
            count += 1
        prev = point
    return total / max(1, count)


def compute_shape_instability(bbox_history: deque[tuple[float, float, float]]) -> float:
    if len(bbox_history) < 2:
        return 0.0
    diffs: list[float] = []
    prev = None
    for dims in bbox_history:
        if prev is not None:
            wd = abs(dims[0] - prev[0]) / max(1.0, prev[0])
            hd = abs(dims[1] - prev[1]) / max(1.0, prev[1])
            ad = abs(dims[2] - prev[2]) / max(0.25, prev[2])
            diffs.append((wd + hd + ad) / 3.0)
        prev = dims
    return sum(diffs) / max(1, len(diffs))


def is_headlike_source(anchor_source: str) -> bool:
    return anchor_source == "head_like_keypoint"


def is_shoulder_source(anchor_source: str) -> bool:
    return anchor_source == "shoulder_fallback"


def compute_seed_quality(point_count: int, assist_cfg: ContinuityAssistConfig) -> float:
    return clamp01(float(point_count) / float(max(1, assist_cfg.klt_max_corners)))


def compute_patch_quality(
    *,
    anchor_source: str,
    anchor_confidence: float,
    seed_quality: float,
    klt_quality: float,
) -> float:
    conf_score = clamp01(anchor_confidence)
    seed_score = clamp01(seed_quality)
    klt_score = clamp01(klt_quality)
    if is_headlike_source(anchor_source):
        return clamp01((0.40 * klt_score) + (0.35 * seed_score) + (0.25 * conf_score))
    if is_shoulder_source(anchor_source):
        return clamp01((0.42 * klt_score) + (0.33 * seed_score) + (0.25 * conf_score))
    return min(0.68, clamp01((0.60 * klt_score) + (0.40 * seed_score)))


def compute_activation_band_scale(
    *,
    anchor_source: str,
    patch_quality: float,
    assist_cfg: ContinuityAssistConfig,
) -> float:
    patch_quality = clamp01(patch_quality)
    if is_headlike_source(anchor_source):
        if patch_quality >= assist_cfg.strong_patch_quality_thr:
            return assist_cfg.headlike_activation_band_scale
        if patch_quality >= 0.58:
            return 1.22
        return 1.10
    if is_shoulder_source(anchor_source):
        if patch_quality >= assist_cfg.strong_patch_quality_thr:
            return assist_cfg.shoulder_activation_band_scale
        if patch_quality >= 0.58:
            return 1.10
        return 1.04
    return assist_cfg.proxy_activation_band_scale


def bbox_center_xy(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def is_strong_headlike_case(memory: TrackAssistMemory, assist_cfg: ContinuityAssistConfig) -> bool:
    return (
        assist_cfg.experiment_uncapped_headlike
        and is_headlike_source(memory.last_anchor_source)
        and clamp01(memory.last_patch_quality) >= assist_cfg.strong_patch_quality_thr
        and clamp01(memory.last_klt_quality) >= assist_cfg.strong_klt_quality_thr
        and clamp01(memory.motion_score) >= assist_cfg.strong_motion_score_min
    )


def is_uncapped_headlike_quality_lost(memory: TrackAssistMemory, assist_cfg: ContinuityAssistConfig) -> bool:
    return (
        clamp01(memory.last_patch_quality) < assist_cfg.experiment_headlike_min_patch_quality
        or clamp01(memory.last_klt_quality) < assist_cfg.experiment_headlike_min_klt_quality
        or int(memory.klt_point_count) <= 0
    )


def raw_shift_bbox(bbox_xyxy: list[float], dx: float, dy: float) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def is_prediction_out_of_frame(
    *,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    anchor_xy: tuple[float, float],
    frame_w: int,
    frame_h: int,
) -> bool:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    ax, ay = map(float, anchor_xy)
    return (
        x1 < 0.0
        or y1 < 0.0
        or x2 > float(frame_w - 1)
        or y2 > float(frame_h - 1)
        or ax < 0.0
        or ay < 0.0
        or ax > float(frame_w - 1)
        or ay > float(frame_h - 1)
    )


def is_implausible_uncapped_headlike(
    *,
    memory: TrackAssistMemory,
    assist_bbox_xyxy: list[float],
    assist_anchor_xy: tuple[float, float],
    assist_cfg: ContinuityAssistConfig,
) -> bool:
    if not memory.last_bbox_xyxy or memory.last_anchor_xy is None:
        return False
    last_center = bbox_center_xy(memory.last_bbox_xyxy)
    assist_center = bbox_center_xy(assist_bbox_xyxy)
    bbox_w, bbox_h, _ = bbox_dimensions(memory.last_bbox_xyxy)
    scale = max(1.0, bbox_w, bbox_h)
    center_drift = math.hypot(assist_center[0] - last_center[0], assist_center[1] - last_center[1])
    anchor_drift = math.hypot(assist_anchor_xy[0] - memory.last_anchor_xy[0], assist_anchor_xy[1] - memory.last_anchor_xy[1])
    return (
        center_drift > (scale * assist_cfg.experiment_headlike_max_center_drift_ratio)
        or anchor_drift > (scale * assist_cfg.experiment_headlike_max_anchor_drift_ratio)
    )


def compute_stability(memory: TrackAssistMemory, assist_cfg: ContinuityAssistConfig) -> tuple[float, int]:
    motion_score = clamp01(1.0 - (memory.last_motion_mag_px / max(1.0, assist_cfg.motion_fast_px)))
    shape_score = clamp01(1.0 - (memory.last_shape_instability / max(0.05, assist_cfg.shape_instability_thr)))
    klt_score = clamp01(memory.last_klt_quality)
    patch_score = clamp01(memory.last_patch_quality)
    stability_score = clamp01((0.40 * klt_score) + (0.28 * patch_score) + (0.18 * shape_score) + (0.14 * motion_score))
    visible_limit = assist_cfg.visible_min_frames + int(
        round((assist_cfg.visible_max_frames - assist_cfg.visible_min_frames) * stability_score)
    )
    if (
        is_headlike_source(memory.last_anchor_source)
        and patch_score >= assist_cfg.strong_patch_quality_thr
        and klt_score >= assist_cfg.strong_klt_quality_thr
        and motion_score >= assist_cfg.strong_motion_score_min
    ):
        visible_limit = max(visible_limit, assist_cfg.headlike_min_visible_floor)
    elif (
        is_shoulder_source(memory.last_anchor_source)
        and patch_score >= assist_cfg.strong_patch_quality_thr
        and klt_score >= assist_cfg.strong_klt_quality_thr
        and motion_score >= assist_cfg.strong_motion_score_min
    ):
        visible_limit = max(visible_limit, assist_cfg.shoulder_min_visible_floor)
    return stability_score, max(assist_cfg.visible_min_frames, min(assist_cfg.visible_max_frames, visible_limit))


def compute_proxy_boundary_distance(point_xy: tuple[float, float], overlay: stage0403.OverlaySpec) -> float:
    if cv2 is None or np is None or len(overlay.roi_polygon_source) < 3:
        return float("inf")
    contour = np.asarray(overlay.roi_polygon_source, dtype=np.float32)
    return float(cv2.pointPolygonTest(contour, (float(point_xy[0]), float(point_xy[1])), True))


def shift_bbox(bbox_xyxy: list[float], dx: float, dy: float, frame_w: int, frame_h: int) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    new_x1 = max(0.0, min(float(frame_w) - width, x1 + dx))
    new_y1 = max(0.0, min(float(frame_h) - height, y1 + dy))
    new_x2 = min(float(frame_w), new_x1 + width)
    new_y2 = min(float(frame_h), new_y1 + height)
    return [new_x1, new_y1, new_x2, new_y2]


def clamp_point(point_xy: tuple[float, float], frame_w: int, frame_h: int) -> tuple[float, float]:
    return (
        max(0.0, min(float(frame_w - 1), float(point_xy[0]))),
        max(0.0, min(float(frame_h - 1), float(point_xy[1]))),
    )


def build_klt_patch_rect(
    bbox_xyxy: list[float],
    anchor_xy: tuple[float, float],
    frame_w: int,
    frame_h: int,
    assist_cfg: ContinuityAssistConfig,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    patch_w = max(24.0, bbox_w * assist_cfg.klt_patch_width_ratio)
    patch_h = max(24.0, bbox_h * assist_cfg.klt_patch_height_ratio)
    center_x = float(anchor_xy[0])
    center_y = float(anchor_xy[1]) + (bbox_h * assist_cfg.klt_patch_y_offset_ratio)
    left = center_x - (patch_w * 0.5)
    top = center_y - (patch_h * 0.35)
    return clamp_xyxy(left, top, left + patch_w, top + patch_h, frame_w, frame_h)


def init_upper_klt_points(
    gray_frame: np.ndarray,
    bbox_xyxy: list[float],
    anchor_xy: tuple[float, float],
    assist_cfg: ContinuityAssistConfig,
) -> tuple[Any, int]:
    if cv2 is None:
        return None, 0
    frame_h, frame_w = gray_frame.shape[:2]
    px1, py1, px2, py2 = build_klt_patch_rect(
        bbox_xyxy=bbox_xyxy,
        anchor_xy=anchor_xy,
        frame_w=frame_w,
        frame_h=frame_h,
        assist_cfg=assist_cfg,
    )
    patch = gray_frame[py1:py2, px1:px2]
    if patch.size == 0:
        return None, 0
    points = cv2.goodFeaturesToTrack(
        patch,
        maxCorners=assist_cfg.klt_max_corners,
        qualityLevel=assist_cfg.klt_quality_level,
        minDistance=assist_cfg.klt_min_distance,
        blockSize=assist_cfg.klt_block_size,
    )
    if points is None or len(points) == 0:
        return None, 0
    points = points.astype(np.float32)
    points[:, 0, 0] += float(px1)
    points[:, 0, 1] += float(py1)
    return points, int(len(points))


def run_fullframe_klt_shift(
    prev_gray: np.ndarray | None,
    curr_gray: np.ndarray | None,
    prev_points: np.ndarray | None,
    prev_point_count: int,
) -> tuple[np.ndarray | None, tuple[float, float], float, int]:
    if cv2 is None or prev_gray is None or curr_gray is None or prev_points is None or len(prev_points) == 0:
        return None, (0.0, 0.0), 0.0, 0
    try:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    except cv2.error:
        return None, (0.0, 0.0), 0.0, 0

    if next_points is None or status is None:
        return None, (0.0, 0.0), 0.0, 0

    good_mask = status.reshape(-1) == 1
    good_prev = prev_points.reshape(-1, 2)[good_mask]
    good_next = next_points.reshape(-1, 2)[good_mask]
    if len(good_prev) == 0 or len(good_next) == 0:
        return None, (0.0, 0.0), 0.0, 0

    diffs = good_next - good_prev
    dx = float(np.median(diffs[:, 0]))
    dy = float(np.median(diffs[:, 1]))
    tracked_ratio = float(len(good_next)) / float(max(1, prev_point_count))
    return good_next.reshape(-1, 1, 2).astype(np.float32), (dx, dy), clamp01(tracked_ratio), int(len(good_next))


def extract_bbox_overlap(record: dict[str, Any] | None) -> float:
    if record is None:
        return 0.0
    metrics = record.get("candidate_metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    try:
        return float(metrics.get("bbox_overlap", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def refresh_assist_memory_from_visible_track(
    *,
    memory: TrackAssistMemory,
    frame_num: int,
    bbox_xyxy: list[float],
    anchor: UpperAnchorResult,
    gray_frame: np.ndarray,
    overlay: stage0403.OverlaySpec,
    record: dict[str, Any] | None,
    assist_cfg: ContinuityAssistConfig,
) -> None:
    upper_proxy = compute_bbox_upper_proxy(bbox_xyxy)
    memory.last_state = str(record.get("state", STATE_OUT)) if record is not None else STATE_OUT
    memory.last_real_frame = int(frame_num)
    memory.last_bbox_xyxy = list(bbox_xyxy)
    memory.assist_bbox_xyxy = list(bbox_xyxy)
    memory.last_anchor_xy = tuple(anchor.point_xy)
    memory.assist_anchor_xy = tuple(anchor.point_xy)
    memory.last_anchor_source = anchor.source
    memory.last_anchor_kind = anchor.kind
    memory.last_anchor_confidence = clamp01(anchor.confidence)
    memory.last_boundary_distance_px = compute_proxy_boundary_distance(upper_proxy, overlay)
    memory.last_bbox_overlap = extract_bbox_overlap(record)
    memory.anchor_history.append(tuple(anchor.point_xy))
    memory.bbox_history.append(bbox_dimensions(bbox_xyxy))
    memory.last_motion_mag_px = compute_motion_mag(memory.anchor_history)
    memory.last_shape_instability = compute_shape_instability(memory.bbox_history)
    memory.motion_score = clamp01(1.0 - (memory.last_motion_mag_px / max(1.0, assist_cfg.motion_fast_px)))
    memory.shape_score = clamp01(1.0 - (memory.last_shape_instability / max(0.05, assist_cfg.shape_instability_thr)))
    memory.klt_points, memory.klt_point_count = init_upper_klt_points(
        gray_frame=gray_frame,
        bbox_xyxy=bbox_xyxy,
        anchor_xy=anchor.point_xy,
        assist_cfg=assist_cfg,
    )
    memory.last_klt_quality = 1.0 if memory.klt_point_count > 0 else 0.0
    memory.last_seed_quality = compute_seed_quality(memory.klt_point_count, assist_cfg)
    memory.last_patch_quality = compute_patch_quality(
        anchor_source=memory.last_anchor_source,
        anchor_confidence=memory.last_anchor_confidence,
        seed_quality=memory.last_seed_quality,
        klt_quality=memory.last_klt_quality,
    )
    memory.last_stability_score, memory.last_visible_limit_frames = compute_stability(memory, assist_cfg)
    memory.memory_until_frame = int(frame_num) + int(assist_cfg.internal_memory_frames)
    memory.missing_anchor_evidence_streak = 0
    memory.experiment_uncapped_headlike_active = is_strong_headlike_case(memory, assist_cfg)


def build_missing_track_assist(
    *,
    ctx: RenderSourceContext,
    track_id: int,
    state: str,
    frame_num: int,
    event_type: str,
    gray_frame: np.ndarray | None,
    assist_cfg: ContinuityAssistConfig,
) -> AssistDrawSpec | None:
    if event_type not in MISSING_EVENT_TYPES:
        return None

    memory = ctx.assist_memories.get(track_id)
    if memory is None or memory.last_real_frame < 0:
        return None

    missing_age = int(frame_num) - int(memory.last_real_frame)
    if missing_age <= 0:
        return None
    if frame_num > memory.memory_until_frame:
        return None
    if not memory.assist_bbox_xyxy or memory.assist_anchor_xy is None:
        return None

    predicted_out_of_frame = False
    if gray_frame is not None and ctx.prev_gray is not None and memory.klt_points is not None and len(memory.klt_points) > 0:
        next_points, shift_xy, klt_quality, point_count = run_fullframe_klt_shift(
            prev_gray=ctx.prev_gray,
            curr_gray=gray_frame,
            prev_points=memory.klt_points,
            prev_point_count=memory.klt_point_count,
        )
        memory.last_klt_quality = klt_quality
        if next_points is not None:
            memory.klt_points = next_points
            memory.klt_point_count = point_count
            frame_h, frame_w = gray_frame.shape[:2]
            raw_bbox = raw_shift_bbox(memory.assist_bbox_xyxy, shift_xy[0], shift_xy[1])
            raw_anchor = (
                memory.assist_anchor_xy[0] + shift_xy[0],
                memory.assist_anchor_xy[1] + shift_xy[1],
            )
            predicted_out_of_frame = is_prediction_out_of_frame(
                bbox_xyxy=raw_bbox,
                anchor_xy=raw_anchor,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            memory.assist_bbox_xyxy = shift_bbox(memory.assist_bbox_xyxy, shift_xy[0], shift_xy[1], frame_w, frame_h)
            memory.assist_anchor_xy = clamp_point(
                (memory.assist_anchor_xy[0] + shift_xy[0], memory.assist_anchor_xy[1] + shift_xy[1]),
                frame_w,
                frame_h,
            )
        else:
            memory.klt_points = None
            memory.klt_point_count = 0
    else:
        memory.last_klt_quality = 0.0

    memory.last_seed_quality = compute_seed_quality(memory.klt_point_count, assist_cfg)
    memory.last_patch_quality = compute_patch_quality(
        anchor_source=memory.last_anchor_source,
        anchor_confidence=memory.last_anchor_confidence,
        seed_quality=memory.last_seed_quality,
        klt_quality=memory.last_klt_quality,
    )
    memory.last_stability_score, memory.last_visible_limit_frames = compute_stability(memory, assist_cfg)
    if memory.experiment_uncapped_headlike_active:
        if predicted_out_of_frame:
            memory.experiment_uncapped_headlike_active = False
            return None
        if is_uncapped_headlike_quality_lost(memory, assist_cfg):
            memory.missing_anchor_evidence_streak += 1
        else:
            memory.missing_anchor_evidence_streak = 0
        if memory.missing_anchor_evidence_streak > assist_cfg.experiment_headlike_max_missing_anchor_frames:
            memory.experiment_uncapped_headlike_active = False
            return None
    elif missing_age > memory.last_visible_limit_frames:
        return None

    assist_bbox = list(memory.assist_bbox_xyxy)
    assist_anchor = tuple(memory.assist_anchor_xy or compute_bbox_upper_proxy(assist_bbox))
    upper_proxy = compute_bbox_upper_proxy(assist_bbox)
    boundary_distance = compute_proxy_boundary_distance(upper_proxy, ctx.overlay)
    memory.last_boundary_distance_px = boundary_distance
    allowed_band = assist_cfg.boundary_band_px * compute_activation_band_scale(
        anchor_source=memory.last_anchor_source,
        patch_quality=memory.last_patch_quality,
        assist_cfg=assist_cfg,
    )
    if abs(boundary_distance) > allowed_band:
        memory.experiment_uncapped_headlike_active = False
        return None

    if memory.experiment_uncapped_headlike_active and is_implausible_uncapped_headlike(
        memory=memory,
        assist_bbox_xyxy=assist_bbox,
        assist_anchor_xy=assist_anchor,
        assist_cfg=assist_cfg,
    ):
        memory.experiment_uncapped_headlike_active = False
        return None

    return AssistDrawSpec(
        bbox_xyxy=assist_bbox,
        anchor_xy=assist_anchor,
        anchor_source=memory.last_anchor_source or "bbox_upper_proxy",
        anchor_kind=memory.last_anchor_kind or "bbox_upper_proxy",
        klt_quality=float(memory.last_klt_quality),
        patch_quality=float(memory.last_patch_quality),
        motion_score=float(memory.motion_score),
        shape_score=float(memory.shape_score),
        stability_score=float(memory.last_stability_score),
        visible_limit_frames=int(memory.last_visible_limit_frames),
        missing_age_frames=missing_age,
        boundary_distance_px=float(boundary_distance),
        experiment_uncapped_headlike=bool(memory.experiment_uncapped_headlike_active),
    )


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


def draw_assist_overlay(
    frame: Any,
    *,
    mapped_box: tuple[int, int, int, int],
    mapped_anchor: tuple[int, int] | None,
) -> None:
    assist_color = (235, 220, 90)
    x1, y1, x2, y2 = mapped_box
    draw_dashed_line(frame, (x1, y1), (x2, y1), assist_color, thickness=2, dash_len=12, gap_len=7)
    draw_dashed_line(frame, (x2, y1), (x2, y2), assist_color, thickness=2, dash_len=12, gap_len=7)
    draw_dashed_line(frame, (x2, y2), (x1, y2), assist_color, thickness=2, dash_len=12, gap_len=7)
    draw_dashed_line(frame, (x1, y2), (x1, y1), assist_color, thickness=2, dash_len=12, gap_len=7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1, cv2.LINE_AA)

    if mapped_anchor is None:
        return
    px, py = mapped_anchor
    if px < 0 or py < 0 or px >= frame.shape[1] or py >= frame.shape[0]:
        return
    cv2.circle(frame, (px, py), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 5, assist_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 9, (40, 40, 40), 1, cv2.LINE_AA)


def cleanup_assist_memories(ctx: RenderSourceContext, frame_num: int) -> None:
    expired = [track_id for track_id, memory in ctx.assist_memories.items() if frame_num > memory.memory_until_frame]
    for track_id in expired:
        ctx.assist_memories.pop(track_id, None)


def render_tile(
    *,
    canvas: Any,
    tile_origin: tuple[int, int],
    tile_size: tuple[int, int],
    frame: Any | None,
    ctx: RenderSourceContext,
    frame_num: int,
    anchor_probe: UpperAnchorProbe,
    assist_cfg: ContinuityAssistConfig,
    assist_stats: dict[str, Counter[str] | int | str],
) -> int:
    assert cv2 is not None
    tile_x, tile_y = tile_origin
    tile_w, tile_h = tile_size
    tile = canvas[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]
    tile[:] = 0
    frame_gray = None

    if frame is not None:
        fit = stage0403.compute_fit_rect(frame.shape[1], frame.shape[0], tile_w, tile_h)
        resized = cv2.resize(frame, (fit.display_width, fit.display_height), interpolation=cv2.INTER_LINEAR)
        tile[fit.pad_y : fit.pad_y + fit.display_height, fit.pad_x : fit.pad_x + fit.display_width] = resized
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        fit = stage0403.compute_fit_rect(ctx.width, ctx.height, tile_w, tile_h)

    frame_rows = ctx.sidecar_rows_by_frame.get(frame_num, {})
    frame_records = ctx.records_by_frame.get(frame_num, {})
    summary = stage0403.summarize_frame_state(frame_rows=frame_rows, frame_records=frame_records, roi_status=ctx.overlay.roi_status)
    drawn_assists = 0

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
        real_row = row is not None and row.has_valid_bbox

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

            if frame_gray is not None and is_relevant_state(state):
                upper_proxy = compute_bbox_upper_proxy(box_xyxy)
                boundary_distance = compute_proxy_boundary_distance(upper_proxy, ctx.overlay)
                memory = ctx.assist_memories.setdefault(track_id, TrackAssistMemory(track_id=track_id))
                memory.last_boundary_distance_px = boundary_distance
                near_boundary_limit = assist_cfg.boundary_band_px * 1.15
                if abs(boundary_distance) <= near_boundary_limit:
                    anchor = anchor_probe.probe(frame=frame, bbox_xyxy=box_xyxy)
                    pre_patch_quality = clamp01(anchor.confidence) if anchor.source != "bbox_upper_proxy" else 0.0
                    activation_band = assist_cfg.boundary_band_px * compute_activation_band_scale(
                        anchor_source=anchor.source,
                        patch_quality=pre_patch_quality,
                        assist_cfg=assist_cfg,
                    )
                    if abs(boundary_distance) <= activation_band:
                        refresh_assist_memory_from_visible_track(
                            memory=memory,
                            frame_num=frame_num,
                            bbox_xyxy=box_xyxy,
                            anchor=anchor,
                            gray_frame=frame_gray,
                            overlay=ctx.overlay,
                            record=record,
                            assist_cfg=assist_cfg,
                        )
            continue

        if is_relevant_state(state):
            assist_spec = build_missing_track_assist(
                ctx=ctx,
                track_id=track_id,
                state=state,
                frame_num=frame_num,
                event_type=event_type,
                gray_frame=frame_gray,
                assist_cfg=assist_cfg,
            )
            if assist_spec is not None:
                mapped_assist_box = stage0403.map_box_to_tile(
                    box_xyxy=assist_spec.bbox_xyxy,
                    fit=fit,
                    tile_origin=(0, 0),
                    tile_w=tile_w,
                    tile_h=tile_h,
                )
                if mapped_assist_box is not None:
                    mapped_anchor = stage0403.map_point_to_tile(assist_spec.anchor_xy, fit, tile_origin=(0, 0))
                    draw_assist_overlay(tile, mapped_box=mapped_assist_box, mapped_anchor=mapped_anchor)
                    drawn_assists += 1
                    cast_drawn = assist_stats["assist_drawn_by_state"]
                    cast_sources = assist_stats["assist_anchor_sources"]
                    assert isinstance(cast_drawn, Counter)
                    assert isinstance(cast_sources, Counter)
                    cast_drawn[state] += 1
                    cast_sources[assist_spec.anchor_source] += 1
                    if assist_spec.experiment_uncapped_headlike:
                        assist_stats["uncapped_headlike_draw_count"] = int(assist_stats["uncapped_headlike_draw_count"]) + 1
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

    if summary["confirmed_count"] > 0:
        cv2.rectangle(tile, (1, 1), (tile_w - 2, tile_h - 2), (0, 0, 255), 3, cv2.LINE_AA)

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
        status_color = (170, 220, 255)
    elif summary["global_state"] == "INTRUSION":
        status_color = (200, 205, 255)
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

    if frame_gray is not None:
        ctx.prev_gray = frame_gray
    cleanup_assist_memories(ctx, frame_num)
    return drawn_assists


def render_multistream_continuity_assist(
    *,
    source_specs: list[stage0403.SourceSpec],
    overlay_specs: list[stage0403.OverlaySpec],
    artifacts_by_source: dict[int, stage0403.SourceArtifacts],
    decision_results: dict[int, dict[str, Any]],
    output_path: Path,
    tiled_size: tuple[int, int],
    logger: logging.Logger,
    anchor_probe: UpperAnchorProbe,
    assist_cfg: ContinuityAssistConfig,
) -> dict[str, Any]:
    require_render_deps()

    contexts: list[RenderSourceContext] = []
    for spec, overlay in zip(source_specs, overlay_specs):
        artifacts = artifacts_by_source[spec.source_id]
        cap = cv2.VideoCapture(str(spec.local_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source video for Stage 04.04 final render: {spec.local_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                cap.release()
                raise RuntimeError(f"Could not decode frames from video: {spec.local_path}")
            height, width = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        sidecar_rows_by_frame, sidecar_summary = load_sidecar_rows(artifacts.split_sidecar_path)
        records_by_frame = stage0403.load_events_by_frame(artifacts.events_path)
        contexts.append(
            RenderSourceContext(
                spec=spec,
                overlay=overlay,
                artifacts=artifacts,
                cap=cap,
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
    if any(abs(ctx.fps - output_fps) > 0.01 for ctx in contexts[1:]):
        logger.warning("Source FPS values differ across streams %s; using first-source fps=%s for Stage 04.04 render.", fps_values, output_fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = stage0403.create_video_writer(cv2, output_path, width=tiled_size[0], height=tiled_size[1], fps=output_fps)
    tile_w = tiled_size[0] // TILER_COLUMNS
    tile_h = tiled_size[1] // TILER_ROWS
    frames_written = 0
    assist_stats: dict[str, Counter[str] | int | str] = {
        "assist_draw_count": 0,
        "frames_with_assist": 0,
        "uncapped_headlike_draw_count": 0,
        "assist_drawn_by_state": Counter(),
        "assist_anchor_sources": Counter(),
        "upper_anchor_probe_status": anchor_probe.model_status,
    }

    try:
        while True:
            canvas = np.zeros((tiled_size[1], tiled_size[0], 3), dtype=np.uint8)
            any_frame = False
            frame_assists = 0

            for idx, ctx in enumerate(contexts):
                frame = None
                frame_num = ctx.frame_num
                if ctx.active:
                    ok, next_frame = ctx.cap.read()
                    if ok and next_frame is not None:
                        frame = next_frame
                        ctx.frame_num += 1
                        any_frame = True
                    else:
                        ctx.active = False

                tile_row = idx // TILER_COLUMNS
                tile_col = idx % TILER_COLUMNS
                frame_assists += render_tile(
                    canvas=canvas,
                    tile_origin=(tile_col * tile_w, tile_row * tile_h),
                    tile_size=(tile_w, tile_h),
                    frame=frame,
                    ctx=ctx,
                    frame_num=frame_num,
                    anchor_probe=anchor_probe,
                    assist_cfg=assist_cfg,
                    assist_stats=assist_stats,
                )

            if not any_frame:
                break

            if frame_assists > 0:
                assist_stats["frames_with_assist"] = int(assist_stats["frames_with_assist"]) + 1
                assist_stats["assist_draw_count"] = int(assist_stats["assist_draw_count"]) + int(frame_assists)

            writer.write(canvas)
            frames_written += 1
    finally:
        writer.release()
        for ctx in contexts:
            ctx.cap.release()

    assist_drawn_by_state = assist_stats["assist_drawn_by_state"]
    assist_anchor_sources = assist_stats["assist_anchor_sources"]
    assert isinstance(assist_drawn_by_state, Counter)
    assert isinstance(assist_anchor_sources, Counter)
    return {
        "overlay_path": str(output_path),
        "frames_written": int(frames_written),
        "fps": float(output_fps),
        "canvas_size": {"width": int(tiled_size[0]), "height": int(tiled_size[1])},
        "tile_size": {"width": int(tile_w), "height": int(tile_h)},
        "source_fps_values": fps_values,
        "assist_frames_with_overlay": int(assist_stats["frames_with_assist"]),
        "assist_draw_count": int(assist_stats["assist_draw_count"]),
        "uncapped_headlike_draw_count": int(assist_stats["uncapped_headlike_draw_count"]),
        "assist_drawn_by_state": dict(assist_drawn_by_state),
        "assist_anchor_sources": dict(assist_anchor_sources),
        "upper_anchor_probe_status": str(assist_stats["upper_anchor_probe_status"]),
    }


def main() -> None:
    args = parse_args()
    stage0403.normalize_output_args(args)

    if args.no_outputs:
        raise SystemExit("--no_outputs is not supported for the Stage 04.04 continuity-assist wrapper.")

    source_specs = stage0403.build_source_specs(args.inputs)
    template_path = stage0403.project_path(args.ds_config_template)
    plugin_lib = stage0403.project_path(args.plugin_lib)
    pose_model_path = stage0403.project_path(args.pose_model)
    assist_cfg = build_assist_config(args)

    stage0403.validate_file_exists(template_path, "DeepStream config template")
    if not getattr(args, "out_base", ""):
        args.out_base = DEFAULT_OUT_BASE

    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")
    if not args.dry_run:
        stage0403.validate_file_exists(plugin_lib, "Stage 04.03 intrusion export plugin library")

    preflight_rendered_text, refs = stage0403.render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=Path("/tmp/ds_multistream4_continuity_assist_preflight.mp4"),
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

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)
    if args.dry_run and not deepstream_app:
        logger.warning("deepstream-app not found; continuing because --dry_run was requested.")
    if args.dry_run and not plugin_lib.exists():
        logger.warning("Stage 04.03 plugin library not found; continuing because --dry_run was requested.")
    if args.dry_run and (cv2 is None or np is None):
        logger.warning("opencv-python/numpy not available; continuing because --dry_run was requested.")

    overlay_specs = stage0403.load_overlay_specs(source_specs, streammux_spec, logger)
    artifacts_by_source = stage0403.build_source_artifacts(run.out_dir, source_specs)
    anchor_probe = UpperAnchorProbe(settings=UpperAnchorProbeSettings(model_path=str(pose_model_path)))

    tracking_output_video = run.out_dir / f"{args.out_base}_tiled_tracking_export.mp4"
    final_output_video = run.out_dir / f"{args.out_base}_tiled_continuity_assist.mp4"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    combined_sidecar_path = run.out_dir / "tracking_sidecar_combined.csv"
    run_summary_path = run.out_dir / "continuity_assist_run_summary.json"

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

    source_meta = [
        {
            "source_id": spec.source_id,
            "clip_label": spec.clip_label,
            "input_arg": spec.input_arg,
            "resolved_local_path": str(spec.local_path),
            "runtime_path": str(spec.runtime_path),
            "runtime_uri": spec.runtime_uri,
        }
        for spec in source_specs
    ]
    overlay_meta = [
        {
            "source_id": overlay.source_id,
            "channel_label": overlay.channel_label,
            "roi_json": str(overlay.roi_json),
            "roi_status": overlay.roi_status,
            "roi_source_size": list(overlay.roi_source_size),
            "roi_polygon_source": [list(pt) for pt in overlay.roi_polygon_source],
            "roi_polygon_frame": [list(pt) for pt in overlay.roi_polygon_frame],
            "frame_transform": overlay.frame_transform,
            "warning": overlay.warning,
        }
        for overlay in overlay_specs
    ]

    continuity_meta = {
        "enabled": True,
        "assist_type": "visual_only_boundary_near_continuity_assist",
        "anchor_priority": [
            "head_like_keypoint",
            "shoulder_fallback",
            "bbox_upper_proxy",
        ],
        "activation_requires": [
            "state_is_candidate_or_confirmed",
            "current_detector_or_tracker_bbox_missing",
            "track_recently_seen",
            "upper_body_anchor_available",
            "near_roi_boundary_band",
        ],
        "visible_lifetime_rule": {
            "stability_score": "0.40*klt_quality + 0.28*patch_quality + 0.18*shape_score + 0.14*motion_score",
            "patch_quality": {
                "head_like_keypoint": "0.40*klt_quality + 0.35*seed_quality + 0.25*anchor_confidence",
                "shoulder_fallback": "0.42*klt_quality + 0.33*seed_quality + 0.25*anchor_confidence",
                "bbox_upper_proxy": "min(0.68, 0.60*klt_quality + 0.40*seed_quality)",
            },
            "motion_score": f"clamp01(1 - recent_anchor_motion_px / {assist_cfg.motion_fast_px})",
            "shape_score": f"clamp01(1 - mean_bbox_shape_change / {assist_cfg.shape_instability_thr})",
            "klt_quality": "tracked_good_points / seeded_points",
            "visible_frames": f"{assist_cfg.visible_min_frames} + round(({assist_cfg.visible_max_frames} - {assist_cfg.visible_min_frames}) * stability_score)",
            "strong_head_like_floor": {
                "requires": [
                    "anchor_source=head_like_keypoint",
                    f"patch_quality>={assist_cfg.strong_patch_quality_thr}",
                    f"klt_quality>={assist_cfg.strong_klt_quality_thr}",
                    f"motion_score>={assist_cfg.strong_motion_score_min}",
                ],
                "min_visible_frames": assist_cfg.headlike_min_visible_floor,
            },
            "strong_shoulder_floor": {
                "requires": [
                    "anchor_source=shoulder_fallback",
                    f"patch_quality>={assist_cfg.strong_patch_quality_thr}",
                    f"klt_quality>={assist_cfg.strong_klt_quality_thr}",
                    f"motion_score>={assist_cfg.strong_motion_score_min}",
                ],
                "min_visible_frames": assist_cfg.shoulder_min_visible_floor,
            },
        },
        "visible_assist_lifetime_frames": {
            "min": assist_cfg.visible_min_frames,
            "max": assist_cfg.visible_max_frames,
        },
        "internal_memory_window_frames": assist_cfg.internal_memory_frames,
        "boundary_band_px": assist_cfg.boundary_band_px,
        "activation_band_scale": {
            "head_like_keypoint": assist_cfg.headlike_activation_band_scale,
            "shoulder_fallback": assist_cfg.shoulder_activation_band_scale,
            "bbox_upper_proxy": assist_cfg.proxy_activation_band_scale,
        },
        "experiment_mode": {
            "enabled": assist_cfg.experiment_uncapped_headlike,
            "mode_name": "uncapped_headlike_visible_hold",
            "debug_only": True,
            "default_behavior_changed": False,
            "applies_only_to": ["strong_head_like_keypoint"],
            "more_conservative_for": ["shoulder_fallback", "bbox_upper_proxy"],
            "uncapped_activation_requires": [
                "assist_experiment_uncapped_headlike flag enabled",
                "last anchor source is head_like_keypoint",
                f"patch_quality>={assist_cfg.strong_patch_quality_thr}",
                f"klt_quality>={assist_cfg.strong_klt_quality_thr}",
                f"motion_score>={assist_cfg.strong_motion_score_min}",
            ],
            "uncapped_stop_conditions": [
                "detector_or_tracker_bbox_returns",
                "assist_leaves_boundary_relevant_band",
                f"patch_quality<{assist_cfg.experiment_headlike_min_patch_quality}",
                f"klt_quality<{assist_cfg.experiment_headlike_min_klt_quality}",
                f"missing_anchor_evidence_streak>{assist_cfg.experiment_headlike_max_missing_anchor_frames}",
                "predicted_assist_leaves_frame",
                f"center_drift>{assist_cfg.experiment_headlike_max_center_drift_ratio}x_last_bbox_scale",
                f"anchor_drift>{assist_cfg.experiment_headlike_max_anchor_drift_ratio}x_last_bbox_scale",
            ],
        },
        "klt_patch": {
            "width_ratio": assist_cfg.klt_patch_width_ratio,
            "height_ratio": assist_cfg.klt_patch_height_ratio,
            "y_offset_ratio": assist_cfg.klt_patch_y_offset_ratio,
            "max_corners": assist_cfg.klt_max_corners,
        },
        "confirmed_truth_semantics_changed": False,
        "candidate_truth_semantics_changed": False,
        "assist_generates_confirmed_intrusion": False,
        "assist_uses_klt_for_truth": False,
    }

    run_meta: dict[str, Any] = {
        "stage": STAGE,
        "stage_step": STAGE_STEP,
        "run_ts": run.run_ts,
        "dry_run": bool(args.dry_run),
        "source_count": SOURCE_COUNT,
        "batch_size": SOURCE_COUNT,
        "tiler_rows": TILER_ROWS,
        "tiler_columns": TILER_COLUMNS,
        "tiled_output_size": {"width": tiled_size[0], "height": tiled_size[1]},
        "streammux_width": streammux_spec.width,
        "streammux_height": streammux_spec.height,
        "streammux_enable_padding": streammux_spec.enable_padding,
        "sources": source_meta,
        "overlay_specs": overlay_meta,
        "tracking_output_video": str(tracking_output_video),
        "final_output_video": str(final_output_video),
        "combined_sidecar_path": str(combined_sidecar_path),
        "rendered_config_path": str(rendered_config_path),
        "run_summary_path": str(run_summary_path),
        "plugin_lib": str(plugin_lib),
        "pose_model": str(pose_model_path),
        "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
        "candidate_definition": "bbox_roi_geometry_candidate_only",
        "confirm_requires_ankle": True,
        "klt_included_for_assist_only": True,
        "continuity_assist": continuity_meta,
        "decision_params": {
            "candidate_enter_n": params.candidate_enter_n,
            "confirm_enter_n": params.confirm_enter_n,
            "exit_n": params.exit_n,
            "grace_frames": params.grace_frames,
            "candidate_iou_or_overlap_thr": params.candidate_iou_or_overlap_thr,
            "candidate_score_thr": params.candidate_score_thr,
            "proxy_start_max_age_frames": params.proxy_start_max_age_frames,
        },
        "deepstream_app": deepstream_app or "",
        "deepstream_app_cmd": cmd,
        "ds_config_template": str(template_path),
        "infer_config_source_of_truth": str(infer_config_local),
        "tracker_config_source_of_truth": str(tracker_config_local),
    }

    if run.outputs_enabled:
        dump_run_meta(run.out_dir, run_meta)

    logger.info("ds_config_template=%s", template_path)
    logger.info("plugin_lib=%s", plugin_lib)
    logger.info("rendered_config_path=%s", rendered_config_path)
    logger.info("tracking_output_video=%s", tracking_output_video)
    logger.info("final_output_video=%s", final_output_video)
    logger.info("combined_sidecar_path=%s", combined_sidecar_path)
    logger.info("deepstream_cmd=%s", cmd_str)
    logger.info(
        "fsm=candidate_enter_n=%s confirm_enter_n=%s exit_n=%s grace_frames=%s candidate_iou_or_overlap_thr=%s",
        params.candidate_enter_n,
        params.confirm_enter_n,
        params.exit_n,
        params.grace_frames,
        params.candidate_iou_or_overlap_thr,
    )
    logger.info(
        "assist=boundary_band_px=%s visible_frames=%s-%s internal_memory_frames=%s motion_fast_px=%s shape_instability_thr=%s",
        assist_cfg.boundary_band_px,
        assist_cfg.visible_min_frames,
        assist_cfg.visible_max_frames,
        assist_cfg.internal_memory_frames,
        assist_cfg.motion_fast_px,
        assist_cfg.shape_instability_thr,
    )

    print(f"rendered config saved: {rendered_config_path}")
    print(f"tracking export video: {tracking_output_video}")
    print(f"final continuity assist video: {final_output_video}")
    print(f"combined sidecar path: {combined_sidecar_path}")
    print(f"deepstream-app command: {cmd_str}")
    print("confirmed intrusion definition: at least one ankle enters the ROI")
    print("candidate formation: bbox-vs-ROI geometry only")
    print("Stage 04.04 continuity assist: visual-only, boundary-near, short-loss support")
    print("Stage 04.04 continuity assist does NOT change final confirmed-intrusion truth semantics")
    if run.log_path is not None:
        print(f"log saved: {run.log_path}")
    if run.cmd_path is not None:
        print(f"wrapper cmd saved: {run.cmd_path}")

    if args.dry_run:
        logger.info("dry_run requested; not invoking DeepStream, decision, or final continuity-assist render stages")
        return

    exit_code = stage0403.stream_process_output(cmd, logger, runtime_env, prefix="deepstream-app")
    if exit_code != 0:
        logger.error("deepstream-app exited with code %s", exit_code)
        raise SystemExit(exit_code)
    logger.info("DeepStream tracking/export pass completed successfully for Stage 04.04")

    split_summary = stage0403.split_sidecar_by_source(combined_sidecar_path, source_specs, overlay_specs, artifacts_by_source)
    logger.info("split_sidecar_summary=%s", split_summary)

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
                    "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
                    "candidate_definition": "bbox_roi_geometry_candidate_only",
                    "klt_included": False,
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
                "confirmed_events": int(decision_summary.get("confirmed_events", 0)),
                "pose_probe_status": str(decision_summary.get("pose_probe_status", "")),
                "skip_reason": str(decision_summary.get("skip_reason", "")),
            }
        )

    render_summary = render_multistream_continuity_assist(
        source_specs=source_specs,
        overlay_specs=overlay_specs,
        artifacts_by_source=artifacts_by_source,
        decision_results=decision_results,
        output_path=final_output_video,
        tiled_size=tiled_size,
        logger=logger,
        anchor_probe=anchor_probe,
        assist_cfg=assist_cfg,
    )

    run_summary = {
        "tracking_export_video": str(tracking_output_video),
        "final_continuity_assist_video": str(final_output_video),
        "combined_sidecar_path": str(combined_sidecar_path),
        "split_sidecar_summary": split_summary,
        "per_source": per_source_meta,
        "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
        "candidate_definition": "bbox_roi_geometry_candidate_only",
        "continuity_assist": continuity_meta,
        "render_summary": render_summary,
        "confirmed_events_total": int(sum(int(item.get("confirmed_events", 0)) for item in per_source_meta)),
        "confirmed_truth_semantics_changed": False,
    }
    write_json(run_summary_path, run_summary)

    run_meta["split_sidecar_summary"] = split_summary
    run_meta["per_source"] = per_source_meta
    run_meta["render_summary"] = render_summary
    run_meta["confirmed_events_total"] = run_summary["confirmed_events_total"]
    if run.outputs_enabled:
        dump_run_meta(run.out_dir, run_meta)

    logger.info(
        "Stage 04.04 final render completed output=%s confirmed_events_total=%s assist_draw_count=%s",
        final_output_video,
        run_summary["confirmed_events_total"],
        render_summary["assist_draw_count"],
    )


if __name__ == "__main__":
    main()
