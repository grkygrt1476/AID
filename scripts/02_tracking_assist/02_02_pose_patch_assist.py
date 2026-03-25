#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_OUT_ROOT = PROJECT_ROOT / "outputs"
LOCAL_LOG_ROOT = LOCAL_OUT_ROOT / "logs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.run_utils import common_argparser, init_run


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_roi_polygon(roi_json_path: str | Path) -> np.ndarray:
    data = load_json(roi_json_path)

    if "vertices_px" not in data:
        raise ValueError(
            f"ROI json does not contain 'vertices_px': {roi_json_path}, "
            f"keys={list(data.keys())}"
        )

    pts = data["vertices_px"]

    if not isinstance(pts, list) or len(pts) == 0:
        raise ValueError(f"'vertices_px' is empty or invalid: {roi_json_path}")

    return np.array([[float(p[0]), float(p[1])] for p in pts], dtype=np.float32)


def draw_roi(frame: np.ndarray, polygon_xy: np.ndarray) -> None:
    cv2.polylines(frame, [polygon_xy.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)


def clamp_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1i = max(0, min(w - 1, int(round(x1))))
    y1i = max(0, min(h - 1, int(round(y1))))
    x2i = max(0, min(w - 1, int(round(x2))))
    y2i = max(0, min(h - 1, int(round(y2))))
    if x2i <= x1i:
        x2i = min(w - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(h - 1, y1i + 1)
    return x1i, y1i, x2i, y2i


def bbox_roi_overlap_ratio(x1: float, y1: float, x2: float, y2: float, roi_mask: np.ndarray) -> float:
    h, w = roi_mask.shape[:2]
    x1i, y1i, x2i, y2i = clamp_xyxy(x1, y1, x2, y2, w, h)
    patch = roi_mask[y1i:y2i, x1i:x2i]
    if patch.size == 0:
        return 0.0
    inter = float(np.count_nonzero(patch))
    area = float(max(1, (x2i - x1i) * (y2i - y1i)))
    return inter / area


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_klt_patch(
    frame: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    patch_scale: float = 1.15,
    center_ratio: float = 0.5,
    anchor: str = "center",
    top_anchor_ratio: float = 0.30,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)

    if anchor == "top":
        cy = y1 + bh * top_anchor_ratio
    else:
        cy = 0.5 * (y1 + y2)

    pw = bw * center_ratio * patch_scale
    ph = bh * center_ratio * patch_scale

    px1 = cx - pw / 2.0
    py1 = cy - ph / 2.0
    px2 = cx + pw / 2.0
    py2 = cy + ph / 2.0

    px1i, py1i, px2i, py2i = clamp_xyxy(px1, py1, px2, py2, w, h)
    patch = frame[py1i:py2i, px1i:px2i].copy()
    return patch, (px1i, py1i, px2i, py2i)


def init_klt_points(
    gray_patch: np.ndarray,
    max_corners: int = 80,
    quality_level: float = 0.003,
    min_distance: int = 2,
    block_size: int = 3,
):
    if gray_patch.size == 0:
        return None
    return cv2.goodFeaturesToTrack(
        gray_patch,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )


def run_klt_shift(prev_gray_patch: np.ndarray, curr_gray_patch: np.ndarray, prev_points: np.ndarray):
    if prev_gray_patch is None or curr_gray_patch is None or prev_points is None or len(prev_points) == 0:
        return None, None, 0.0, 0, "no_prev_points"

    if prev_gray_patch.shape[:2] != curr_gray_patch.shape[:2]:
        return None, None, 0.0, 0, f"patch_size_mismatch:{prev_gray_patch.shape[:2]}->{curr_gray_patch.shape[:2]}"

    try:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray_patch,
            curr_gray_patch,
            prev_points,
            None,
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    except cv2.error:
        return None, None, 0.0, 0, "opencv_lk_error"

    if next_pts is None or status is None:
        return None, None, 0.0, 0, "klt_failed"

    good_prev = prev_points[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]

    if len(good_prev) < 2 or len(good_next) < 2:
        return None, None, 0.0, int(len(good_next)), "too_few_good_points"

    diffs = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)
    dx = float(np.median(diffs[:, 0]))
    dy = float(np.median(diffs[:, 1]))
    median_motion_px = float(np.median(np.linalg.norm(diffs, axis=1)))
    return good_next.reshape(-1, 1, 2), (dx, dy), median_motion_px, int(len(good_next)), ""


def to_numpy(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    return np.asarray(data)


def get_valid_keypoint(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    idx: int,
    min_conf: float,
) -> np.ndarray | None:
    if idx >= len(keypoints_xy) or idx >= len(keypoints_conf):
        return None
    conf = float(keypoints_conf[idx])
    pt = keypoints_xy[idx]
    if conf < min_conf or not np.all(np.isfinite(pt)):
        return None
    return pt.astype(np.float32)


def mean_valid_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    indices: list[int],
    min_conf: float,
) -> np.ndarray | None:
    pts = [get_valid_keypoint(keypoints_xy, keypoints_conf, idx, min_conf) for idx in indices]
    pts = [pt for pt in pts if pt is not None]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)


def count_points(points: np.ndarray | None) -> int:
    if points is None:
        return 0
    return int(len(points))


def bbox_is_clearly_invalid_or_offscreen(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> bool:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return x2 <= 0.0 or y2 <= 0.0 or x1 >= float(frame_w) or y1 >= float(frame_h) or x2 <= x1 or y2 <= y1


def reset_proxy_fail_state(mem: dict[str, Any]) -> None:
    mem["proxy_fail_count"] = 0
    mem["pose_patch_dead"] = False


def register_proxy_failure(mem: dict[str, Any], reason: str, proxy_fail_tolerance: int) -> bool:
    mem["proxy_fail_count"] = int(mem.get("proxy_fail_count", 0)) + 1
    mem["pose_patch_dead"] = int(mem["proxy_fail_count"]) >= int(proxy_fail_tolerance)
    if reason:
        mem["pose_reason"] = reason
    return bool(mem["pose_patch_dead"])


def retire_track(mem: dict[str, Any], expire_reason: str) -> None:
    mem["proxy_expire_reason"] = expire_reason or mem.get("proxy_expire_reason", "") or "retired_cleanup"
    mem["is_retired"] = True
    mem["bbox_xyxy"] = None
    mem["render_bbox"] = None
    mem["render_center_x"] = None
    mem["render_center_y"] = None
    mem["render_w"] = None
    mem["render_h"] = None
    mem["render_last_update_frame"] = None
    mem["render_frozen"] = False
    mem["hold_until_sec"] = -1.0
    mem["miss_count"] = 0
    mem["proxy_fail_count"] = 0
    mem["proxy_stale_count"] = 0
    mem["median_klt_motion_px"] = 0.0
    mem["proxy_center_motion_px"] = 0.0
    mem["good_point_count"] = 0
    mem["pose_patch_dead"] = False
    mem["prev_gray_patch_center"] = None
    mem["prev_points_center"] = None
    mem["patch_bbox_center"] = None
    mem["prev_gray_patch_top"] = None
    mem["prev_points_top"] = None
    mem["patch_bbox_top"] = None
    mem["prev_gray_patch_pose"] = None
    mem["prev_points_pose"] = None
    mem["patch_bbox_pose"] = None
    mem["pose_anchor_xy"] = []
    mem["last_state"] = "lost"


def bbox_area_xyxy(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
) -> float:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return float(max(1.0, x2 - x1) * max(1.0, y2 - y1))


def bbox_center_motion_px(
    prev_bbox_xyxy: list[float] | tuple[float, float, float, float],
    curr_bbox_xyxy: list[float] | tuple[float, float, float, float],
) -> float:
    px1, py1, px2, py2 = map(float, prev_bbox_xyxy)
    cx1, cy1 = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
    x1, y1, x2, y2 = map(float, curr_bbox_xyxy)
    cx2, cy2 = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return float(np.hypot(cx2 - cx1, cy2 - cy1))


def bbox_iou_xyxy(
    bbox_a_xyxy: list[float] | tuple[float, float, float, float],
    bbox_b_xyxy: list[float] | tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = map(float, bbox_a_xyxy)
    bx1, by1, bx2, by2 = map(float, bbox_b_xyxy)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def should_overlap_early_retire(
    mem: dict[str, Any],
    proxy_bbox_xyxy: list[float] | tuple[float, float, float, float],
    proxy_state: str,
    real_tracks: list[dict[str, Any]],
    enabled: bool,
    iou_thr: float,
    center_px_thr: float,
    consecutive_frames: int,
) -> bool:
    mem["overlap_retire_triggered"] = False
    if not enabled or proxy_state not in {"hold", "pose_patch"} or bool(mem.get("is_retired", False)):
        mem["overlap_retire_count"] = 0
        return False

    matched_real_ids: list[int] = []
    for real_track in real_tracks:
        real_bbox = real_track["bbox_xyxy"]
        overlap_iou = bbox_iou_xyxy(proxy_bbox_xyxy, real_bbox)
        if overlap_iou < iou_thr:
            continue
        center_dist_px = bbox_center_motion_px(proxy_bbox_xyxy, real_bbox)
        if center_dist_px > center_px_thr:
            continue
        matched_real_ids.append(int(real_track["track_id"]))
        if len(matched_real_ids) > 1:
            break

    if len(matched_real_ids) != 1:
        mem["overlap_retire_count"] = 0
        return False

    mem["overlap_retire_count"] = int(mem.get("overlap_retire_count", 0)) + 1
    if int(mem["overlap_retire_count"]) < max(1, int(consecutive_frames)):
        return False

    mem["overlap_retire_triggered"] = True
    mem["proxy_expire_reason"] = "proxy_overlapped_by_real"
    return True


def clear_render_hide_state(mem: dict[str, Any]) -> None:
    mem["render_hidden_by_real"] = False
    mem["render_hidden_match_real_id"] = None


def find_render_hide_match_real_id(
    mem: dict[str, Any],
    proxy_bbox_xyxy: list[float] | tuple[float, float, float, float],
    proxy_state: str,
    real_tracks: list[dict[str, Any]],
    frame_idx: int,
    enabled: bool,
    recent_frames: int,
    center_px_thr: float,
    iou_thr: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> int | None:
    if not enabled or proxy_state not in {"hold", "pose_patch"} or bool(mem.get("is_retired", False)):
        clear_render_hide_state(mem)
        return None

    last_real_frame_idx = int(mem.get("last_real_frame_idx", -10**9))
    if (frame_idx - last_real_frame_idx) > max(0, int(recent_frames)):
        clear_render_hide_state(mem)
        return None

    proxy_area = bbox_area_xyxy(proxy_bbox_xyxy)
    matched_real_ids: list[int] = []
    for real_track in real_tracks:
        real_bbox = real_track["bbox_xyxy"]
        center_dist_px = bbox_center_motion_px(proxy_bbox_xyxy, real_bbox)
        if center_dist_px > center_px_thr:
            continue

        overlap_iou = bbox_iou_xyxy(proxy_bbox_xyxy, real_bbox)
        if overlap_iou < iou_thr:
            continue

        area_ratio = bbox_area_xyxy(real_bbox) / max(1.0, proxy_area)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        matched_real_ids.append(int(real_track["track_id"]))
        if len(matched_real_ids) > 1:
            break

    if len(matched_real_ids) != 1:
        clear_render_hide_state(mem)
        return None

    matched_real_id = int(matched_real_ids[0])
    mem["render_hidden_by_real"] = True
    mem["render_hidden_match_real_id"] = matched_real_id
    return matched_real_id


def bbox_center_size_xyxy(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy, w, h


def center_size_to_bbox_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    half_w = 0.5 * max(1.0, float(w))
    half_h = 0.5 * max(1.0, float(h))
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def stabilize_render_bbox(
    mem: dict[str, Any],
    logic_bbox: list[float] | tuple[float, float, float, float],
    state: str,
    frame_idx: int,
    render_deadband_center_px: float,
    render_deadband_size_px: float,
    render_smoothing_alpha: float,
    stabilize_real_boxes: bool,
    stabilize_proxy_boxes: bool,
) -> tuple[list[float], bool]:
    use_stabilization = (state == "real" and stabilize_real_boxes) or (
        state in {"hold", "pose_patch"} and stabilize_proxy_boxes
    )
    logic_bbox_list = [float(v) for v in logic_bbox]
    logic_cx, logic_cy, logic_w, logic_h = bbox_center_size_xyxy(logic_bbox_list)
    prev_render_bbox = mem.get("render_bbox")

    if not use_stabilization or prev_render_bbox is None:
        render_bbox = logic_bbox_list
        render_frozen = False
    else:
        prev_cx = float(mem.get("render_center_x", logic_cx))
        prev_cy = float(mem.get("render_center_y", logic_cy))
        prev_w = float(mem.get("render_w", logic_w))
        prev_h = float(mem.get("render_h", logic_h))
        center_delta = float(np.hypot(logic_cx - prev_cx, logic_cy - prev_cy))
        size_delta = max(abs(logic_w - prev_w), abs(logic_h - prev_h))

        if center_delta <= render_deadband_center_px and size_delta <= render_deadband_size_px:
            render_bbox = [float(v) for v in prev_render_bbox]
            render_frozen = True
        else:
            alpha = float(np.clip(render_smoothing_alpha, 0.0, 1.0))
            render_cx = alpha * logic_cx + (1.0 - alpha) * prev_cx
            render_cy = alpha * logic_cy + (1.0 - alpha) * prev_cy
            render_w = alpha * logic_w + (1.0 - alpha) * prev_w
            render_h = alpha * logic_h + (1.0 - alpha) * prev_h
            render_bbox = center_size_to_bbox_xyxy(render_cx, render_cy, render_w, render_h)
            render_frozen = False

    render_cx, render_cy, render_w, render_h = bbox_center_size_xyxy(render_bbox)
    mem["render_bbox"] = [float(v) for v in render_bbox]
    mem["render_center_x"] = float(render_cx)
    mem["render_center_y"] = float(render_cy)
    mem["render_w"] = float(render_w)
    mem["render_h"] = float(render_h)
    mem["render_last_update_frame"] = int(frame_idx)
    mem["render_frozen"] = bool(render_frozen)
    return mem["render_bbox"], bool(render_frozen)


def draw_klt_points(
    frame: np.ndarray,
    points: np.ndarray | None,
    patch_bbox: list[float] | tuple[float, float, float, float] | None,
    color: tuple[int, int, int],
) -> None:
    if points is None or patch_bbox is None:
        return
    x1, y1, _, _ = map(float, patch_bbox)
    for pt in points.reshape(-1, 2):
        px = int(round(x1 + float(pt[0])))
        py = int(round(y1 + float(pt[1])))
        cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)


def select_pose_candidate(result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
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

    xy = keypoints_xy[best_idx].astype(np.float32)
    conf = keypoints_conf[best_idx].astype(np.float32).reshape(-1)
    return xy, conf


def resolve_pose_anchor(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    min_conf: float,
    anchor_preference: str,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    head_point = mean_valid_keypoints(keypoints_xy, keypoints_conf, [0, 1, 2, 3, 4], min_conf)
    left_shoulder = get_valid_keypoint(keypoints_xy, keypoints_conf, 5, min_conf)
    right_shoulder = get_valid_keypoint(keypoints_xy, keypoints_conf, 6, min_conf)
    shoulder_mid = None
    shoulder_width = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

    hip_mid = mean_valid_keypoints(keypoints_xy, keypoints_conf, [11, 12], min_conf)
    upper_body_valid = int(np.count_nonzero(keypoints_conf[[idx for idx in [0, 1, 2, 3, 4, 5, 6] if idx < len(keypoints_conf)]] >= min_conf))

    anchor = None
    anchor_kind = "weak_keypoints"

    if anchor_preference == "shoulder_mid":
        if shoulder_mid is not None:
            anchor = shoulder_mid
            anchor_kind = "shoulder_mid"
        elif head_point is not None and left_shoulder is not None:
            anchor = 0.5 * (head_point + left_shoulder)
            anchor_kind = "head_left_shoulder"
        elif head_point is not None and right_shoulder is not None:
            anchor = 0.5 * (head_point + right_shoulder)
            anchor_kind = "head_right_shoulder"
    else:
        if shoulder_mid is not None and head_point is not None:
            anchor = 0.5 * (shoulder_mid + head_point)
            anchor_kind = "head_shoulder_center"
        elif shoulder_mid is not None:
            anchor = shoulder_mid
            anchor_kind = "shoulder_mid"

    if anchor is None and head_point is not None and left_shoulder is not None:
        anchor = 0.5 * (head_point + left_shoulder)
        anchor_kind = "head_left_shoulder"
    if anchor is None and head_point is not None and right_shoulder is not None:
        anchor = 0.5 * (head_point + right_shoulder)
        anchor_kind = "head_right_shoulder"
    if anchor is None and head_point is not None:
        anchor = head_point
        anchor_kind = "head_only"
    if anchor is None and left_shoulder is not None:
        anchor = left_shoulder
        anchor_kind = "left_shoulder"
    if anchor is None and right_shoulder is not None:
        anchor = right_shoulder
        anchor_kind = "right_shoulder"

    if anchor is not None:
        anchor = anchor.astype(np.float32)
        anchor[0] = float(np.clip(anchor[0], x1, x2))
        anchor[1] = float(np.clip(anchor[1], y1, y2))

    return anchor, anchor_kind, {
        "head_point": head_point,
        "shoulder_mid": shoulder_mid,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "hip_mid": hip_mid,
        "shoulder_width_px": shoulder_width,
        "upper_body_valid_points": upper_body_valid,
    }


def derive_pose_patch_ratios(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    pose_parts: dict[str, Any],
    pose_cfg: dict[str, Any],
) -> tuple[float, float, float]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    width_ratio = float(pose_cfg.get("patch_width_ratio", 0.55))
    height_ratio = float(pose_cfg.get("patch_height_ratio", 0.48))

    shoulder_width_px = pose_parts.get("shoulder_width_px")
    if shoulder_width_px is not None and bw > 1.0:
        width_ratio = max(width_ratio, float(pose_cfg.get("shoulder_width_scale", 1.9)) * float(shoulder_width_px) / bw)
    if shoulder_width_px is not None and bh > 1.0:
        height_ratio = max(height_ratio, float(pose_cfg.get("shoulder_height_scale", 2.2)) * float(shoulder_width_px) / bh)

    head_point = pose_parts.get("head_point")
    hip_mid = pose_parts.get("hip_mid")
    torso_height_px = 0.0
    if head_point is not None and hip_mid is not None and bh > 1.0:
        torso_height_px = max(1.0, float(hip_mid[1] - head_point[1]))
        height_ratio = max(height_ratio, float(pose_cfg.get("torso_height_scale", 0.65)) * torso_height_px / bh)

    width_ratio = float(np.clip(
        width_ratio,
        float(pose_cfg.get("patch_width_ratio_min", 0.35)),
        float(pose_cfg.get("patch_width_ratio_max", 0.9)),
    ))
    height_ratio = float(np.clip(
        height_ratio,
        float(pose_cfg.get("patch_height_ratio_min", 0.32)),
        float(pose_cfg.get("patch_height_ratio_max", 0.85)),
    ))
    return width_ratio, height_ratio, torso_height_px


def build_pose_patch(
    frame: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    anchor_rel_xy: list[float] | tuple[float, float],
    patch_width_ratio: float,
    patch_height_ratio: float,
    patch_y_offset_ratio: float,
    min_patch_size_px: int,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, tuple[float, float] | None, str]:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pw = max(float(min_patch_size_px), bw * float(patch_width_ratio))
    ph = max(float(min_patch_size_px), bh * float(patch_height_ratio))
    return build_pose_patch_fixed_size(
        frame=frame,
        bbox_xyxy=bbox_xyxy,
        anchor_rel_xy=anchor_rel_xy,
        patch_width_px=pw,
        patch_height_px=ph,
        patch_y_offset_ratio=patch_y_offset_ratio,
        min_patch_size_px=min_patch_size_px,
    )


def clamp_fixed_span(start_px: float, size_px: float, limit_px: int) -> tuple[int, int]:
    size_i = max(1, min(int(limit_px), int(round(size_px))))
    start_i = int(round(start_px))
    end_i = start_i + size_i

    if start_i < 0:
        end_i -= start_i
        start_i = 0
    if end_i > int(limit_px):
        start_i -= end_i - int(limit_px)
        end_i = int(limit_px)

    start_i = max(0, min(start_i, max(0, int(limit_px) - size_i)))
    end_i = min(int(limit_px), start_i + size_i)
    return start_i, end_i


def build_pose_patch_fixed_size(
    frame: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    anchor_rel_xy: list[float] | tuple[float, float],
    patch_width_px: float,
    patch_height_px: float,
    patch_y_offset_ratio: float,
    min_patch_size_px: int,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, tuple[float, float] | None, str]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ax = x1 + bw * float(anchor_rel_xy[0])
    ay = y1 + bh * float(anchor_rel_xy[1])
    pw = max(float(min_patch_size_px), float(patch_width_px))
    ph = max(float(min_patch_size_px), float(patch_height_px))
    cx = ax
    cy = ay + ph * float(patch_y_offset_ratio)

    px1i, px2i = clamp_fixed_span(cx - pw / 2.0, pw, w)
    py1i, py2i = clamp_fixed_span(cy - ph / 2.0, ph, h)

    if (px2i - px1i) < int(min_patch_size_px) or (py2i - py1i) < int(min_patch_size_px):
        return None, None, None, "patch_size_mismatch"

    patch = frame[py1i:py2i, px1i:px2i].copy()
    return patch, (px1i, py1i, px2i, py2i), (float(ax), float(ay)), ""


def estimate_pose_patch_state(
    frame: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    pose_model: YOLO,
    pose_cfg: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    x1i, y1i, x2i, y2i = clamp_xyxy(x1, y1, x2, y2, w, h)
    if (x2i - x1i) < int(pose_cfg.get("min_crop_size_px", 32)) or (y2i - y1i) < int(pose_cfg.get("min_crop_size_px", 32)):
        return {}, "patch_size_mismatch"

    crop = frame[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        return {}, "patch_size_mismatch"

    try:
        results = pose_model.predict(
            source=crop,
            imgsz=int(pose_cfg.get("input_size", 640)),
            conf=float(pose_cfg.get("conf", 0.25)),
            verbose=False,
            stream=False,
        )
    except Exception:
        return {}, "pose_missing"

    result = results[0] if results else None
    if result is None:
        return {}, "pose_missing"

    keypoints_xy, keypoints_conf = select_pose_candidate(result)
    if keypoints_xy is None or keypoints_conf is None:
        return {}, "pose_missing"

    keypoints_xy[:, 0] += float(x1i)
    keypoints_xy[:, 1] += float(y1i)

    anchor_xy, anchor_kind, pose_parts = resolve_pose_anchor(
        keypoints_xy,
        keypoints_conf,
        bbox_xyxy,
        min_conf=float(pose_cfg.get("keypoint_conf", 0.35)),
        anchor_preference=str(pose_cfg.get("anchor", "head_shoulder_center")),
    )
    if anchor_xy is None:
        return {
            "pose_keypoints_xy": keypoints_xy,
            "pose_keypoints_conf": keypoints_conf,
            "pose_anchor_type": anchor_kind,
            "pose_upper_body_valid_points": int(pose_parts.get("upper_body_valid_points", 0)),
        }, "weak_keypoints"

    patch_width_ratio, patch_height_ratio, torso_height_px = derive_pose_patch_ratios(
        bbox_xyxy=bbox_xyxy,
        pose_parts=pose_parts,
        pose_cfg=pose_cfg,
    )

    x1f, y1f, x2f, y2f = map(float, bbox_xyxy)
    bw = max(1.0, x2f - x1f)
    bh = max(1.0, y2f - y1f)
    anchor_rel_xy = [
        float(np.clip((float(anchor_xy[0]) - x1f) / bw, 0.0, 1.0)),
        float(np.clip((float(anchor_xy[1]) - y1f) / bh, 0.0, 1.0)),
    ]

    return {
        "pose_anchor_rel": anchor_rel_xy,
        "pose_anchor_xy": [float(anchor_xy[0]), float(anchor_xy[1])],
        "pose_anchor_type": anchor_kind,
        "pose_patch_width_ratio": float(patch_width_ratio),
        "pose_patch_height_ratio": float(patch_height_ratio),
        "pose_keypoints_xy": keypoints_xy,
        "pose_keypoints_conf": keypoints_conf,
        "pose_shoulder_width_px": float(pose_parts.get("shoulder_width_px") or 0.0),
        "pose_torso_height_px": float(torso_height_px),
        "pose_upper_body_valid_points": int(pose_parts.get("upper_body_valid_points", 0)),
    }, ""


def pose_debug_fields(mem: dict[str, Any]) -> dict[str, Any]:
    return {
        "pose_anchor_type": mem.get("pose_anchor_type", ""),
        "pose_anchor_xy": [round(float(v), 2) for v in mem.get("pose_anchor_xy", [])] if mem.get("pose_anchor_xy") else [],
        "pose_patch_bbox": [round(float(v), 2) for v in mem.get("patch_bbox_pose", [])] if mem.get("patch_bbox_pose") else [],
        "pose_patch_width_px": int(mem.get("pose_patch_width_px", 0)),
        "pose_patch_height_px": int(mem.get("pose_patch_height_px", 0)),
        "pose_patch_point_count": int(count_points(mem.get("prev_points_pose"))),
        "pose_reason": mem.get("pose_reason", ""),
    }


def proxy_debug_fields(mem: dict[str, Any], proxy_alive: bool) -> dict[str, Any]:
    last_real = mem.get("last_real_time_sec")
    last_real_rounded = round(float(last_real), 4) if last_real is not None else None
    return {
        "proxy_survival_mode": mem.get("proxy_survival_mode", ""),
        "proxy_fail_count": int(mem.get("proxy_fail_count", 0)),
        "proxy_fail_tolerance": int(mem.get("proxy_fail_tolerance", 0)),
        "proxy_stale_count": int(mem.get("proxy_stale_count", 0)),
        "median_klt_motion_px": round(float(mem.get("median_klt_motion_px", 0.0)), 4),
        "proxy_center_motion_px": round(float(mem.get("proxy_center_motion_px", 0.0)), 4),
        "good_point_count": int(mem.get("good_point_count", 0)),
        "last_real_time_sec": last_real_rounded,
        "proxy_alive": bool(proxy_alive),
        "is_retired": bool(mem.get("is_retired", False)),
        "overlap_retire_count": int(mem.get("overlap_retire_count", 0)),
        "overlap_retire_triggered": bool(mem.get("overlap_retire_triggered", False)),
        "render_hidden_by_real": bool(mem.get("render_hidden_by_real", False)),
        "render_hidden_match_real_id": mem.get("render_hidden_match_real_id"),
        "render_bbox": [round(float(v), 2) for v in mem.get("render_bbox", [])] if mem.get("render_bbox") else [],
        "render_frozen": bool(mem.get("render_frozen", False)),
        "proxy_expire_reason": mem.get("proxy_expire_reason", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(
        out_root=str(LOCAL_OUT_ROOT),
        log_root=str(LOCAL_LOG_ROOT),
    )
    parser.add_argument("--cfg", type=str, required=True, help="experiment yaml path")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker yaml")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "hold", "klt", "pose_patch"])
    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--end_sec", type=float, default=None)
    parser.add_argument("--save_jsonl", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pose_model", type=str, default="", help="Override pose model path for pose_patch mode")
    parser.add_argument("--pose_refresh_frames", type=int, default=None, help="Override pose refresh interval")
    parser.add_argument(
        "--proxy_survival_mode",
        type=str,
        default="",
        choices=["fixed_hold", "track_alive"],
        help="Pose proxy lifetime policy; CLI overrides YAML.",
    )
    parser.add_argument(
        "--proxy_fail_tolerance",
        type=int,
        default=None,
        help="Override pose.proxy_fail_tolerance for track_alive mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.cfg)

    name = cfg.get("name", "baseline_bytetrack")
    if not getattr(args, "out_base", ""):
        args.out_base = name

    run = init_run(
        stage="02_tracking_assist",
        script_file=__file__,
        args=args,
    )
    logger = logging.getLogger(__name__)

    video_path = cfg["video"]
    roi_path = cfg["roi"]

    det = cfg["detector"]
    model_path = det["model"]
    imgsz = int(det.get("input_size", 960))
    conf = float(det.get("conf", 0.35))
    iou = float(det.get("iou", 0.45))
    classes = det.get("class_ids", [0])
    mode = args.mode
    display_cfg = cfg.get("display", {})
    hold_seconds = float(display_cfg.get("hold_seconds", 0.3))
    render_deadband_center_px = float(display_cfg.get("render_deadband_center_px", 3.0))
    render_deadband_size_px = float(display_cfg.get("render_deadband_size_px", 2.0))
    render_smoothing_alpha = float(display_cfg.get("render_smoothing_alpha", 0.35))
    stabilize_real_boxes = bool(display_cfg.get("stabilize_real_boxes", True))
    stabilize_proxy_boxes = bool(display_cfg.get("stabilize_proxy_boxes", True))
    enable_overlap_early_retire = bool(display_cfg.get("enable_overlap_early_retire", True))
    overlap_retire_iou_thr = float(display_cfg.get("overlap_retire_iou_thr", 0.60))
    overlap_retire_center_px_thr = float(display_cfg.get("overlap_retire_center_px_thr", 20.0))
    overlap_retire_consecutive_frames = int(display_cfg.get("overlap_retire_consecutive_frames", 2))
    enable_render_hide_old_proxy = bool(display_cfg.get("enable_render_hide_old_proxy", True))
    hide_recent_frames = int(display_cfg.get("hide_recent_frames", 12))
    hide_center_px_thr = float(display_cfg.get("hide_center_px_thr", 20.0))
    hide_iou_thr = float(display_cfg.get("hide_iou_thr", 0.45))
    hide_min_area_ratio = float(display_cfg.get("hide_min_area_ratio", 0.5))
    hide_max_area_ratio = float(display_cfg.get("hide_max_area_ratio", 2.0))
    hide_klt_points_with_box = bool(display_cfg.get("hide_klt_points_with_box", True))
    candidate_overlap_thr = float(cfg.get("candidate_overlap_thr", 0.05))
    klt_cfg = cfg.get("klt", {})
    klt_max_miss_frames = int(klt_cfg.get("max_miss_frames", 3))
    klt_patch_scale = float(klt_cfg.get("patch_scale", 1.15))
    klt_center_ratio = float(klt_cfg.get("center_ratio", 0.5))
    klt_top_center_ratio = float(klt_cfg.get("top_center_ratio", 0.45))
    klt_top_anchor_ratio = float(klt_cfg.get("top_anchor_ratio", 0.30))
    klt_max_translation_px = float(klt_cfg.get("max_translation_px", 20.0))

    pose_cfg = cfg.get("pose", {})
    pose_model_path = args.pose_model or str(pose_cfg.get("model", "yolo11s-pose.pt"))
    pose_refresh_frames = int(
        pose_cfg.get("refresh_frames", 3)
        if args.pose_refresh_frames is None
        else args.pose_refresh_frames
    )
    proxy_survival_mode = str(args.proxy_survival_mode or pose_cfg.get("proxy_survival_mode", "fixed_hold")).strip().lower() or "fixed_hold"
    proxy_fail_tolerance = int(
        pose_cfg.get("proxy_fail_tolerance", 3)
        if args.proxy_fail_tolerance is None
        else args.proxy_fail_tolerance
    )
    stale_motion_px_thr = float(pose_cfg.get("stale_motion_px_thr", 0.4))
    stale_center_px_thr = float(pose_cfg.get("stale_center_px_thr", 0.6))
    stale_seconds = float(pose_cfg.get("stale_seconds", 2.0))
    proxy_min_good_points = int(pose_cfg.get("proxy_min_good_points", 4))
    pose_max_miss_frames = int(pose_cfg.get("max_miss_frames", klt_max_miss_frames))
    pose_max_translation_px = float(pose_cfg.get("max_translation_px", klt_max_translation_px))
    pose_patch_y_offset_ratio = float(pose_cfg.get("patch_y_offset_ratio", 0.08))
    pose_min_patch_size_px = int(pose_cfg.get("min_patch_size_px", 24))
    if proxy_survival_mode not in {"fixed_hold", "track_alive"}:
        raise ValueError(f"Unsupported proxy_survival_mode: {proxy_survival_mode}")
    if proxy_fail_tolerance < 1:
        raise ValueError(f"proxy_fail_tolerance must be >= 1, got {proxy_fail_tolerance}")
    if render_deadband_center_px < 0.0:
        raise ValueError(f"render_deadband_center_px must be >= 0, got {render_deadband_center_px}")
    if render_deadband_size_px < 0.0:
        raise ValueError(f"render_deadband_size_px must be >= 0, got {render_deadband_size_px}")
    if not 0.0 <= render_smoothing_alpha <= 1.0:
        raise ValueError(f"render_smoothing_alpha must be in [0, 1], got {render_smoothing_alpha}")
    if not 0.0 <= overlap_retire_iou_thr <= 1.0:
        raise ValueError(f"overlap_retire_iou_thr must be in [0, 1], got {overlap_retire_iou_thr}")
    if overlap_retire_center_px_thr < 0.0:
        raise ValueError(
            f"overlap_retire_center_px_thr must be >= 0, got {overlap_retire_center_px_thr}"
        )
    if overlap_retire_consecutive_frames < 1:
        raise ValueError(
            "overlap_retire_consecutive_frames must be >= 1, "
            f"got {overlap_retire_consecutive_frames}"
        )
    if hide_recent_frames < 0:
        raise ValueError(f"hide_recent_frames must be >= 0, got {hide_recent_frames}")
    if hide_center_px_thr < 0.0:
        raise ValueError(f"hide_center_px_thr must be >= 0, got {hide_center_px_thr}")
    if not 0.0 <= hide_iou_thr <= 1.0:
        raise ValueError(f"hide_iou_thr must be in [0, 1], got {hide_iou_thr}")
    if hide_min_area_ratio <= 0.0:
        raise ValueError(
            f"hide_min_area_ratio must be > 0, got {hide_min_area_ratio}"
        )
    if hide_max_area_ratio <= 0.0:
        raise ValueError(
            f"hide_max_area_ratio must be > 0, got {hide_max_area_ratio}"
        )
    if hide_min_area_ratio > hide_max_area_ratio:
        raise ValueError(
            "hide_min_area_ratio must be <= hide_max_area_ratio, "
            f"got {hide_min_area_ratio}>{hide_max_area_ratio}"
        )
    if stale_motion_px_thr < 0.0:
        raise ValueError(f"stale_motion_px_thr must be >= 0, got {stale_motion_px_thr}")
    if stale_center_px_thr < 0.0:
        raise ValueError(f"stale_center_px_thr must be >= 0, got {stale_center_px_thr}")
    if stale_seconds <= 0.0:
        raise ValueError(f"stale_seconds must be > 0, got {stale_seconds}")
    if proxy_min_good_points < 1:
        raise ValueError(f"proxy_min_good_points must be >= 1, got {proxy_min_good_points}")

    logger.info("cfg=%s", args.cfg)
    logger.info("video=%s", video_path)
    logger.info("roi=%s", roi_path)
    logger.info("tracker=%s", args.tracker)
    logger.info("mode=%s", mode)
    logger.info("hold_seconds=%s", hold_seconds)
    logger.info("render_deadband_center_px=%s", render_deadband_center_px)
    logger.info("render_deadband_size_px=%s", render_deadband_size_px)
    logger.info("render_smoothing_alpha=%s", render_smoothing_alpha)
    logger.info("stabilize_real_boxes=%s", stabilize_real_boxes)
    logger.info("stabilize_proxy_boxes=%s", stabilize_proxy_boxes)
    logger.info("enable_overlap_early_retire=%s", enable_overlap_early_retire)
    logger.info("overlap_retire_iou_thr=%s", overlap_retire_iou_thr)
    logger.info("overlap_retire_center_px_thr=%s", overlap_retire_center_px_thr)
    logger.info("overlap_retire_consecutive_frames=%s", overlap_retire_consecutive_frames)
    logger.info("enable_render_hide_old_proxy=%s", enable_render_hide_old_proxy)
    logger.info("hide_recent_frames=%s", hide_recent_frames)
    logger.info("hide_center_px_thr=%s", hide_center_px_thr)
    logger.info("hide_iou_thr=%s", hide_iou_thr)
    logger.info("hide_min_area_ratio=%s", hide_min_area_ratio)
    logger.info("hide_max_area_ratio=%s", hide_max_area_ratio)
    logger.info("hide_klt_points_with_box=%s", hide_klt_points_with_box)
    logger.info("candidate_overlap_thr=%s", candidate_overlap_thr)
    logger.info("klt_max_miss_frames=%s", klt_max_miss_frames)
    logger.info("klt_patch_scale=%s", klt_patch_scale)
    logger.info("klt_center_ratio=%s", klt_center_ratio)
    logger.info("klt_top_center_ratio=%s", klt_top_center_ratio)
    logger.info("klt_top_anchor_ratio=%s", klt_top_anchor_ratio)
    logger.info("klt_max_translation_px=%s", klt_max_translation_px)
    if mode == "pose_patch":
        logger.info("pose_model=%s", pose_model_path)
        logger.info("pose_refresh_frames=%s", pose_refresh_frames)
        logger.info("pose_max_miss_frames=%s", pose_max_miss_frames)
        logger.info("pose_anchor=%s", pose_cfg.get("anchor", "head_shoulder_center"))
        logger.info("proxy_survival_mode=%s", proxy_survival_mode)
        logger.info("proxy_fail_tolerance=%s", proxy_fail_tolerance)
        logger.info("stale_motion_px_thr=%s", stale_motion_px_thr)
        logger.info("stale_center_px_thr=%s", stale_center_px_thr)
        logger.info("stale_seconds=%s", stale_seconds)
        logger.info("proxy_min_good_points=%s", proxy_min_good_points)
        logger.info("pose_patch_y_offset_ratio=%s", pose_patch_y_offset_ratio)
        logger.info("pose_max_translation_px=%s", pose_max_translation_px)
    logger.info("out_root=%s", args.out_root)
    logger.info("log_root=%s", args.log_root)
    logger.info("run_out_dir=%s", run.out_dir)
    if run.cmd_path is not None:
        logger.info("run_cmd_path=%s", run.cmd_path)
    if run.log_path is not None:
        logger.info("run_log_path=%s", run.log_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    stale_frames = max(1, int(round(stale_seconds * fps)))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if mode == "pose_patch":
        logger.info("stale_frames=%s", stale_frames)

    preview_w = 960
    preview_h = int(round(frame_h * (preview_w / frame_w)))
    preview_h = preview_h if preview_h % 2 == 0 else preview_h - 1

    start_sec = float(cfg.get("start_sec", 0.0) if args.start_sec is None else args.start_sec)
    end_sec = float(
        (total_frames / fps)
        if args.end_sec is None and "end_sec" not in cfg
        else (cfg.get("end_sec") if args.end_sec is None else args.end_sec)
    )
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, int(round(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    roi_polygon = load_roi_polygon(roi_path)
    roi_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_polygon.astype(np.int32)], 255)

    model = YOLO(model_path)
    pose_model = YOLO(pose_model_path) if mode == "pose_patch" else None

    out_video_path_full = run.out_dir / f"{Path(video_path).stem}_{mode}_bytetrack.mp4"
    out_video_path_preview = run.out_dir / f"{Path(video_path).stem}_{mode}_bytetrack_preview.mp4"

    writer_full = cv2.VideoWriter(
        str(out_video_path_full),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer_full.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_video_path_full}")

    writer_preview = cv2.VideoWriter(
        str(out_video_path_preview),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (preview_w, preview_h),
    )
    if not writer_preview.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_video_path_preview}")

    frame_logs: list[dict[str, Any]] = []
    track_memory: dict[int, dict[str, Any]] = {}
    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        now_sec = frame_idx / fps
        if frame_idx % 30 == 0:
            logger.info("frame_idx=%d now_sec=%.2f", frame_idx, now_sec)

        results = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=classes,
            verbose=False,
            stream=False,
        )

        vis = frame.copy()
        draw_roi(vis, roi_polygon)

        seen_ids: set[int] = set()
        real_tracks_this_frame: list[dict[str, Any]] = []
        result = results[0] if results else None
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4), dtype=np.float32)
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=np.int32)
            if boxes.id is None:
                ids = np.empty((0,), dtype=np.int32)
                seen_ids = set()
            else:
                ids = boxes.id.cpu().numpy().astype(int)
                seen_ids = set(ids.tolist())

            for i in range(len(ids)):
                prev_mem = track_memory.get(int(ids[i]), {})
                x1, y1, x2, y2 = map(float, xyxy[i])
                track_id = int(ids[i])
                cls_id = int(clss[i])
                score = float(confs[i])
                overlap = bbox_roi_overlap_ratio(x1, y1, x2, y2, roi_mask)

                log_row: dict[str, Any] = {
                    "frame_idx": frame_idx,
                    "time_sec": round(now_sec, 4),
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "conf": round(score, 4),
                    "bbox_xyxy": [round(v, 2) for v in [x1, y1, x2, y2]],
                    "roi_overlap": round(overlap, 4),
                    "state": "real",
                    "render_active": True,
                }

                mem: dict[str, Any] = {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "hold_until_sec": now_sec + hold_seconds,
                    "last_real_time_sec": now_sec,
                    "last_real_frame_idx": frame_idx,
                    "roi_overlap": overlap,
                    "miss_count": 0,
                    "klt_success_count": prev_mem.get("klt_success_count", 0),
                    "pose_patch_success_count": prev_mem.get("pose_patch_success_count", 0),
                    "klt_fail_reason": "",
                    "pose_reason": prev_mem.get("pose_reason", ""),
                    "pose_patch_dead": False,
                    "proxy_survival_mode": proxy_survival_mode if mode == "pose_patch" else "",
                    "proxy_fail_count": 0,
                    "proxy_fail_tolerance": proxy_fail_tolerance if mode == "pose_patch" else 0,
                    "proxy_stale_count": 0,
                    "median_klt_motion_px": 0.0,
                    "proxy_center_motion_px": 0.0,
                    "good_point_count": 0,
                    "overlap_retire_count": 0,
                    "overlap_retire_triggered": False,
                    "render_hidden_by_real": False,
                    "render_hidden_match_real_id": None,
                    "proxy_expire_reason": "",
                    "is_retired": False,
                    "pose_anchor_type": prev_mem.get("pose_anchor_type", ""),
                    "pose_anchor_xy": prev_mem.get("pose_anchor_xy", []),
                    "pose_anchor_rel": prev_mem.get("pose_anchor_rel"),
                    "pose_patch_width_ratio": prev_mem.get("pose_patch_width_ratio"),
                    "pose_patch_height_ratio": prev_mem.get("pose_patch_height_ratio"),
                    "pose_patch_width_px": prev_mem.get("pose_patch_width_px", 0),
                    "pose_patch_height_px": prev_mem.get("pose_patch_height_px", 0),
                    "pose_last_update_frame": prev_mem.get("pose_last_update_frame", -10**9),
                    "patch_bbox_pose": prev_mem.get("patch_bbox_pose"),
                    "prev_gray_patch_pose": prev_mem.get("prev_gray_patch_pose"),
                    "prev_points_pose": prev_mem.get("prev_points_pose"),
                    "render_bbox": prev_mem.get("render_bbox"),
                    "render_center_x": prev_mem.get("render_center_x"),
                    "render_center_y": prev_mem.get("render_center_y"),
                    "render_w": prev_mem.get("render_w"),
                    "render_h": prev_mem.get("render_h"),
                    "render_last_update_frame": prev_mem.get("render_last_update_frame"),
                    "render_frozen": prev_mem.get("render_frozen", False),
                }

                if mode == "klt":
                    patch_bgr_center, patch_bbox_center = make_klt_patch(
                        frame,
                        [x1, y1, x2, y2],
                        patch_scale=klt_patch_scale,
                        center_ratio=klt_center_ratio,
                        anchor="center",
                        top_anchor_ratio=klt_top_anchor_ratio,
                    )
                    patch_gray_center = cv2.cvtColor(patch_bgr_center, cv2.COLOR_BGR2GRAY) if patch_bgr_center.size > 0 else None
                    patch_pts_center = init_klt_points(patch_gray_center) if patch_gray_center is not None else None

                    patch_bgr_top, patch_bbox_top = make_klt_patch(
                        frame,
                        [x1, y1, x2, y2],
                        patch_scale=klt_patch_scale,
                        center_ratio=klt_top_center_ratio,
                        anchor="top",
                        top_anchor_ratio=klt_top_anchor_ratio,
                    )
                    patch_gray_top = cv2.cvtColor(patch_bgr_top, cv2.COLOR_BGR2GRAY) if patch_bgr_top.size > 0 else None
                    patch_pts_top = init_klt_points(patch_gray_top) if patch_gray_top is not None else None

                    mem.update(
                        {
                            "prev_gray_patch_center": patch_gray_center,
                            "prev_points_center": patch_pts_center,
                            "patch_bbox_center": list(patch_bbox_center),
                            "prev_gray_patch_top": patch_gray_top,
                            "prev_points_top": patch_pts_top,
                            "patch_bbox_top": list(patch_bbox_top),
                        }
                    )

                if mode == "pose_patch":
                    assert pose_model is not None
                    pose_should_refresh = (
                        mem.get("pose_anchor_rel") is None
                        or (frame_idx - int(mem.get("pose_last_update_frame", -10**9))) >= pose_refresh_frames
                    )
                    pose_refresh_reason = ""
                    pose_refreshed = False
                    if pose_should_refresh:
                        pose_state, pose_refresh_reason = estimate_pose_patch_state(
                            frame=frame,
                            bbox_xyxy=[x1, y1, x2, y2],
                            pose_model=pose_model,
                            pose_cfg=pose_cfg,
                        )
                        if pose_refresh_reason == "":
                            mem.update(pose_state)
                            mem["pose_reason"] = ""
                            mem["pose_last_update_frame"] = frame_idx
                            pose_refreshed = True
                        else:
                            mem["pose_reason"] = pose_refresh_reason
                            mem["pose_keypoints_xy"] = pose_state.get("pose_keypoints_xy")
                            mem["pose_keypoints_conf"] = pose_state.get("pose_keypoints_conf")
                            if mem.get("pose_anchor_rel") is None:
                                mem["pose_anchor_type"] = pose_state.get("pose_anchor_type", "")

                    if mem.get("pose_anchor_rel") is not None and mem.get("pose_patch_width_ratio") is not None and mem.get("pose_patch_height_ratio") is not None:
                        patch_bgr_pose, patch_bbox_pose, pose_anchor_xy, pose_patch_reason = build_pose_patch(
                            frame=frame,
                            bbox_xyxy=[x1, y1, x2, y2],
                            anchor_rel_xy=mem["pose_anchor_rel"],
                            patch_width_ratio=float(mem["pose_patch_width_ratio"]),
                            patch_height_ratio=float(mem["pose_patch_height_ratio"]),
                            patch_y_offset_ratio=pose_patch_y_offset_ratio,
                            min_patch_size_px=pose_min_patch_size_px,
                        )
                        if pose_patch_reason == "":
                            patch_gray_pose = cv2.cvtColor(patch_bgr_pose, cv2.COLOR_BGR2GRAY) if patch_bgr_pose is not None else None
                            patch_pts_pose = init_klt_points(patch_gray_pose) if patch_gray_pose is not None else None
                            mem["prev_gray_patch_pose"] = patch_gray_pose
                            mem["prev_points_pose"] = patch_pts_pose
                            mem["patch_bbox_pose"] = list(patch_bbox_pose) if patch_bbox_pose is not None else None
                            mem["pose_patch_width_px"] = int(patch_gray_pose.shape[1]) if patch_gray_pose is not None else 0
                            mem["pose_patch_height_px"] = int(patch_gray_pose.shape[0]) if patch_gray_pose is not None else 0
                            mem["pose_anchor_xy"] = [float(pose_anchor_xy[0]), float(pose_anchor_xy[1])] if pose_anchor_xy is not None else []
                            if count_points(patch_pts_pose) >= 2:
                                mem["pose_reason"] = ""
                            elif mem.get("pose_reason", "") == "":
                                mem["pose_reason"] = "klt_failed"
                        else:
                            mem["prev_gray_patch_pose"] = None
                            mem["prev_points_pose"] = None
                            mem["patch_bbox_pose"] = None
                            mem["pose_anchor_xy"] = []
                            if mem.get("pose_reason", "") == "":
                                mem["pose_reason"] = pose_patch_reason
                    else:
                        mem["prev_gray_patch_pose"] = None
                        mem["prev_points_pose"] = None
                        mem["patch_bbox_pose"] = None
                        mem["pose_anchor_xy"] = []

                    if mem.get("patch_bbox_pose") is not None:
                        px1, py1, px2, py2 = map(int, mem["patch_bbox_pose"])
                        cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 0, 255), 2, cv2.LINE_AA)
                    if mem.get("pose_anchor_xy"):
                        ax, ay = mem["pose_anchor_xy"]
                        cv2.circle(vis, (int(round(ax)), int(round(ay))), 4, (255, 0, 255), -1, cv2.LINE_AA)
                    draw_klt_points(vis, mem.get("prev_points_pose"), mem.get("patch_bbox_pose"), (0, 255, 255))

                    log_row.update(
                        {
                            "assist_reason": mem.get("pose_reason", ""),
                            "pose_refreshed": pose_refreshed,
                            "pose_refresh_reason": pose_refresh_reason,
                            **proxy_debug_fields(mem, proxy_alive=False),
                            **pose_debug_fields(mem),
                        }
                    )
                else:
                    log_row.update(proxy_debug_fields(mem, proxy_alive=False))

                render_bbox, render_frozen = stabilize_render_bbox(
                    mem=mem,
                    logic_bbox=[x1, y1, x2, y2],
                    state="real",
                    frame_idx=frame_idx,
                    render_deadband_center_px=render_deadband_center_px,
                    render_deadband_size_px=render_deadband_size_px,
                    render_smoothing_alpha=render_smoothing_alpha,
                    stabilize_real_boxes=stabilize_real_boxes,
                    stabilize_proxy_boxes=stabilize_proxy_boxes,
                )
                rx1i, ry1i, rx2i, ry2i = clamp_xyxy(*render_bbox, frame_w, frame_h)
                label = f"id={track_id} conf={score:.2f} ov={overlap:.2f}"
                cv2.rectangle(vis, (rx1i, ry1i), (rx2i, ry2i), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    label,
                    (rx1i, max(15, ry1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                log_row["logic_bbox"] = [round(v, 2) for v in [x1, y1, x2, y2]]
                log_row["render_bbox"] = [round(v, 2) for v in render_bbox]
                log_row["render_frozen"] = bool(render_frozen)
                mem["last_state"] = "real"

                frame_logs.append(log_row)
                track_memory[track_id] = mem
                real_tracks_this_frame.append(
                    {
                        "track_id": track_id,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )

        if mode in ("hold", "klt", "pose_patch"):
            expired_ids = []
            for track_id, mem in list(track_memory.items()):
                if bool(mem.get("is_retired", False)):
                    expired_ids.append(track_id)
                    continue
                if track_id in seen_ids:
                    continue

                state = "hold"
                draw_bbox = list(mem["bbox_xyxy"])
                assist_reason = mem.get("pose_reason", "")
                pose_overlay_bbox = mem.get("patch_bbox_pose")
                pose_overlay_points = mem.get("prev_points_pose")
                proxy_mode = str(mem.get("proxy_survival_mode", "") or proxy_survival_mode)
                allow_track_alive = mode == "pose_patch" and proxy_mode == "track_alive"
                proxy_alive = True
                proxy_expire_reason = ""
                current_overlap = bbox_roi_overlap_ratio(*draw_bbox, roi_mask)
                mem["roi_overlap"] = current_overlap

                if allow_track_alive and bbox_is_clearly_invalid_or_offscreen(draw_bbox, frame_w, frame_h):
                    state = "lost"
                    proxy_alive = False
                    proxy_expire_reason = "bbox_invalid_or_offscreen"
                    assist_reason = proxy_expire_reason
                    mem["proxy_expire_reason"] = proxy_expire_reason
                elif now_sec > float(mem["hold_until_sec"]) and not allow_track_alive:
                    state = "lost"
                    proxy_alive = False
                    proxy_expire_reason = "fixed_hold_timeout" if mode == "pose_patch" else "hold_timeout"
                    assist_reason = proxy_expire_reason
                    mem["proxy_expire_reason"] = proxy_expire_reason
                elif current_overlap < candidate_overlap_thr:
                    if allow_track_alive:
                        pass
                    else:
                        state = "lost"
                        proxy_alive = False
                        proxy_expire_reason = "no_real_no_proxy"
                        assist_reason = proxy_expire_reason
                        mem["proxy_expire_reason"] = proxy_expire_reason

                if state != "lost":
                    mem["proxy_expire_reason"] = ""
                    mem["miss_count"] = int(mem.get("miss_count", 0)) + 1

                if state != "lost" and mode == "klt" and mem["miss_count"] <= klt_max_miss_frames:
                    candidates = []
                    fail_reasons = []

                    patch_bgr_center, patch_bbox_center = make_klt_patch(
                        frame,
                        mem["bbox_xyxy"],
                        patch_scale=klt_patch_scale,
                        center_ratio=klt_center_ratio,
                        anchor="center",
                        top_anchor_ratio=klt_top_anchor_ratio,
                    )
                    patch_gray_center = cv2.cvtColor(patch_bgr_center, cv2.COLOR_BGR2GRAY) if patch_bgr_center.size > 0 else None
                    next_pts_center, delta_xy_center, _, _, fail_reason_center = run_klt_shift(
                        mem.get("prev_gray_patch_center"),
                        patch_gray_center,
                        mem.get("prev_points_center"),
                    )
                    if delta_xy_center is not None:
                        candidates.append(("center", next_pts_center, patch_gray_center, patch_bbox_center, delta_xy_center))
                    else:
                        fail_reasons.append(f"center:{fail_reason_center}")

                    patch_bgr_top, patch_bbox_top = make_klt_patch(
                        frame,
                        mem["bbox_xyxy"],
                        patch_scale=klt_patch_scale,
                        center_ratio=klt_top_center_ratio,
                        anchor="top",
                        top_anchor_ratio=klt_top_anchor_ratio,
                    )
                    patch_gray_top = cv2.cvtColor(patch_bgr_top, cv2.COLOR_BGR2GRAY) if patch_bgr_top.size > 0 else None
                    next_pts_top, delta_xy_top, _, _, fail_reason_top = run_klt_shift(
                        mem.get("prev_gray_patch_top"),
                        patch_gray_top,
                        mem.get("prev_points_top"),
                    )
                    if delta_xy_top is not None:
                        candidates.append(("top", next_pts_top, patch_gray_top, patch_bbox_top, delta_xy_top))
                    else:
                        fail_reasons.append(f"top:{fail_reason_top}")

                    mem["prev_gray_patch_center"] = patch_gray_center
                    mem["prev_points_center"] = next_pts_center if next_pts_center is not None else (init_klt_points(patch_gray_center) if patch_gray_center is not None else None)
                    mem["patch_bbox_center"] = list(patch_bbox_center)
                    mem["prev_gray_patch_top"] = patch_gray_top
                    mem["prev_points_top"] = next_pts_top if next_pts_top is not None else (init_klt_points(patch_gray_top) if patch_gray_top is not None else None)
                    mem["patch_bbox_top"] = list(patch_bbox_top)

                    if candidates:
                        best_name, best_pts, best_gray, best_bbox, best_delta = max(candidates, key=lambda x: len(x[1]))
                        dx, dy = best_delta
                        dx = max(-klt_max_translation_px, min(klt_max_translation_px, dx))
                        dy = max(-klt_max_translation_px, min(klt_max_translation_px, dy))

                        x1, y1, x2, y2 = map(float, mem["bbox_xyxy"])
                        draw_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                        mem["bbox_xyxy"] = draw_bbox
                        mem["klt_success_count"] = int(mem.get("klt_success_count", 0)) + 1
                        mem["klt_fail_reason"] = ""
                        state = "klt"
                        assist_reason = ""
                    else:
                        mem["klt_fail_reason"] = "|".join(fail_reasons) if fail_reasons else "klt_failed"
                        assist_reason = mem["klt_fail_reason"]

                elif state != "lost" and mode == "pose_patch":
                    can_try_pose_proxy = allow_track_alive or mem["miss_count"] <= pose_max_miss_frames
                    if bool(mem.get("pose_patch_dead", False)):
                        assist_reason = mem.get("klt_fail_reason", "") or mem.get("pose_reason", "") or "klt_failed"
                        if allow_track_alive:
                            state = "lost"
                            proxy_alive = False
                            proxy_expire_reason = assist_reason
                            mem["proxy_expire_reason"] = proxy_expire_reason
                    elif can_try_pose_proxy:
                        if mem.get("pose_anchor_rel") is None or mem.get("pose_patch_width_ratio") is None or mem.get("pose_patch_height_ratio") is None:
                            mem["median_klt_motion_px"] = 0.0
                            mem["proxy_center_motion_px"] = 0.0
                            mem["good_point_count"] = 0
                            assist_reason = mem.get("pose_reason", "") or "pose_missing"
                            if allow_track_alive:
                                if register_proxy_failure(mem, assist_reason, proxy_fail_tolerance):
                                    state = "lost"
                                    proxy_alive = False
                                    proxy_expire_reason = f"proxy_fail_tolerance_reached:{assist_reason}"
                                    assist_reason = proxy_expire_reason
                                    mem["proxy_expire_reason"] = proxy_expire_reason
                        else:
                            if int(mem.get("pose_patch_width_px", 0)) > 0 and int(mem.get("pose_patch_height_px", 0)) > 0:
                                patch_bgr_pose, patch_bbox_pose, pose_anchor_xy, pose_patch_reason = build_pose_patch_fixed_size(
                                    frame=frame,
                                    bbox_xyxy=mem["bbox_xyxy"],
                                    anchor_rel_xy=mem["pose_anchor_rel"],
                                    patch_width_px=float(mem["pose_patch_width_px"]),
                                    patch_height_px=float(mem["pose_patch_height_px"]),
                                    patch_y_offset_ratio=pose_patch_y_offset_ratio,
                                    min_patch_size_px=pose_min_patch_size_px,
                                )
                            else:
                                patch_bgr_pose, patch_bbox_pose, pose_anchor_xy, pose_patch_reason = build_pose_patch(
                                    frame=frame,
                                    bbox_xyxy=mem["bbox_xyxy"],
                                    anchor_rel_xy=mem["pose_anchor_rel"],
                                    patch_width_ratio=float(mem["pose_patch_width_ratio"]),
                                    patch_height_ratio=float(mem["pose_patch_height_ratio"]),
                                    patch_y_offset_ratio=pose_patch_y_offset_ratio,
                                    min_patch_size_px=pose_min_patch_size_px,
                                )
                            if pose_patch_reason != "":
                                mem["median_klt_motion_px"] = 0.0
                                mem["proxy_center_motion_px"] = 0.0
                                mem["good_point_count"] = 0
                                mem["klt_fail_reason"] = pose_patch_reason
                                assist_reason = pose_patch_reason
                                if allow_track_alive:
                                    if register_proxy_failure(mem, assist_reason, proxy_fail_tolerance):
                                        state = "lost"
                                        proxy_alive = False
                                        proxy_expire_reason = f"proxy_fail_tolerance_reached:{assist_reason}"
                                        assist_reason = proxy_expire_reason
                                        mem["proxy_expire_reason"] = proxy_expire_reason
                                else:
                                    mem["pose_patch_dead"] = True
                            else:
                                patch_gray_pose = cv2.cvtColor(patch_bgr_pose, cv2.COLOR_BGR2GRAY) if patch_bgr_pose is not None else None
                                next_pts_pose, delta_xy_pose, median_klt_motion_px, good_point_count, fail_reason_pose = run_klt_shift(
                                    mem.get("prev_gray_patch_pose"),
                                    patch_gray_pose,
                                    mem.get("prev_points_pose"),
                                )
                                mem["prev_gray_patch_pose"] = patch_gray_pose
                                mem["patch_bbox_pose"] = list(patch_bbox_pose) if patch_bbox_pose is not None else None
                                mem["pose_patch_width_px"] = int(patch_gray_pose.shape[1]) if patch_gray_pose is not None else 0
                                mem["pose_patch_height_px"] = int(patch_gray_pose.shape[0]) if patch_gray_pose is not None else 0
                                mem["pose_anchor_xy"] = [float(pose_anchor_xy[0]), float(pose_anchor_xy[1])] if pose_anchor_xy is not None else []

                                if delta_xy_pose is not None and next_pts_pose is not None:
                                    prev_bbox = list(mem["bbox_xyxy"])
                                    dx, dy = delta_xy_pose
                                    dx = max(-pose_max_translation_px, min(pose_max_translation_px, dx))
                                    dy = max(-pose_max_translation_px, min(pose_max_translation_px, dy))
                                    x1, y1, x2, y2 = map(float, prev_bbox)
                                    candidate_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                                    mem["median_klt_motion_px"] = float(median_klt_motion_px)
                                    mem["proxy_center_motion_px"] = bbox_center_motion_px(prev_bbox, candidate_bbox)
                                    mem["good_point_count"] = int(good_point_count)
                                    if mem["good_point_count"] < proxy_min_good_points:
                                        mem["proxy_stale_count"] = 0
                                        mem["klt_fail_reason"] = (
                                            f"too_few_good_points:{int(mem['good_point_count'])}<{proxy_min_good_points}"
                                        )
                                        mem["prev_points_pose"] = None
                                        assist_reason = mem["klt_fail_reason"]
                                        if allow_track_alive:
                                            if register_proxy_failure(mem, assist_reason, proxy_fail_tolerance):
                                                state = "lost"
                                                proxy_alive = False
                                                proxy_expire_reason = f"proxy_fail_tolerance_reached:{assist_reason}"
                                                assist_reason = proxy_expire_reason
                                                mem["proxy_expire_reason"] = proxy_expire_reason
                                        else:
                                            mem["pose_patch_dead"] = True
                                    else:
                                        mem["proxy_stale_count"] = 0
                                        draw_bbox = candidate_bbox
                                        mem["bbox_xyxy"] = draw_bbox
                                        mem["prev_points_pose"] = next_pts_pose
                                        mem["pose_patch_success_count"] = int(mem.get("pose_patch_success_count", 0)) + 1
                                        mem["klt_fail_reason"] = ""
                                        mem["pose_reason"] = ""
                                        reset_proxy_fail_state(mem)
                                        mem["proxy_expire_reason"] = ""
                                        state = "pose_patch"
                                        assist_reason = ""
                                        pose_overlay_bbox = mem.get("patch_bbox_pose")
                                        pose_overlay_points = mem.get("prev_points_pose")
                                else:
                                    mem["median_klt_motion_px"] = 0.0
                                    mem["proxy_center_motion_px"] = 0.0
                                    mem["good_point_count"] = int(good_point_count)
                                    mem["klt_fail_reason"] = fail_reason_pose or "klt_failed"
                                    mem["prev_points_pose"] = None
                                    assist_reason = mem["klt_fail_reason"]
                                    if allow_track_alive:
                                        if register_proxy_failure(mem, assist_reason, proxy_fail_tolerance):
                                            state = "lost"
                                            proxy_alive = False
                                            proxy_expire_reason = f"proxy_fail_tolerance_reached:{assist_reason}"
                                            assist_reason = proxy_expire_reason
                                            mem["proxy_expire_reason"] = proxy_expire_reason
                                    else:
                                        mem["pose_patch_dead"] = True

                x1, y1, x2, y2 = draw_bbox
                if state != "lost" and should_overlap_early_retire(
                    mem=mem,
                    proxy_bbox_xyxy=draw_bbox,
                    proxy_state=state,
                    real_tracks=real_tracks_this_frame,
                    enabled=enable_overlap_early_retire,
                    iou_thr=overlap_retire_iou_thr,
                    center_px_thr=overlap_retire_center_px_thr,
                    consecutive_frames=overlap_retire_consecutive_frames,
                ):
                    state = "lost"
                    proxy_alive = False
                    proxy_expire_reason = mem.get("proxy_expire_reason", "") or "proxy_overlapped_by_real"
                    assist_reason = proxy_expire_reason

                current_overlap = bbox_roi_overlap_ratio(x1, y1, x2, y2, roi_mask)
                mem["roi_overlap"] = current_overlap
                if state == "lost":
                    expire_reason = mem.get("proxy_expire_reason", "") or assist_reason or "no_real_no_proxy"
                    retire_track(mem, expire_reason)
                    frame_logs.append(
                        {
                            "frame_idx": frame_idx,
                            "time_sec": round(now_sec, 4),
                            "track_id": track_id,
                            "bbox_xyxy": [round(v, 2) for v in [x1, y1, x2, y2]],
                            "roi_overlap": round(float(current_overlap), 4),
                            "state": "lost",
                            "render_active": False,
                            "assist_reason": assist_reason,
                            "miss_count": int(mem.get("miss_count", 0)),
                            "klt_success_count": int(mem.get("klt_success_count", 0)),
                            "pose_patch_success_count": int(mem.get("pose_patch_success_count", 0)),
                            "klt_fail_reason": mem.get("klt_fail_reason", ""),
                            **proxy_debug_fields(mem, proxy_alive=False),
                            **pose_debug_fields(mem),
                        }
                    )
                    expired_ids.append(track_id)
                    continue

                logic_bbox = [x1, y1, x2, y2]
                mem["last_state"] = state
                render_bbox, render_frozen = stabilize_render_bbox(
                    mem=mem,
                    logic_bbox=logic_bbox,
                    state=state,
                    frame_idx=frame_idx,
                    render_deadband_center_px=render_deadband_center_px,
                    render_deadband_size_px=render_deadband_size_px,
                    render_smoothing_alpha=render_smoothing_alpha,
                    stabilize_real_boxes=stabilize_real_boxes,
                    stabilize_proxy_boxes=stabilize_proxy_boxes,
                )
                x1i, y1i, x2i, y2i = clamp_xyxy(*render_bbox, frame_w, frame_h)

                color = (0, 165, 255)
                if state == "klt":
                    color = (255, 255, 0)
                elif state == "pose_patch":
                    color = (255, 0, 255)
                elif state == "lost":
                    color = (0, 0, 255)
                label = f"id={track_id} {state}"
                if assist_reason:
                    label = f"{label} {assist_reason}"
                if mode == "pose_patch":
                    proxy_alive_label = int(proxy_alive and state != "lost")
                    label = (
                        f"{label} proxy={proxy_mode} alive={proxy_alive_label} "
                        f"fail={int(mem.get('proxy_fail_count', 0))}/{int(mem.get('proxy_fail_tolerance', proxy_fail_tolerance))}"
                    )
                    if mem.get("proxy_expire_reason", ""):
                        label = f"{label} expire={mem['proxy_expire_reason']}"

                hidden_match_real_id = find_render_hide_match_real_id(
                    mem=mem,
                    proxy_bbox_xyxy=draw_bbox,
                    proxy_state=state,
                    real_tracks=real_tracks_this_frame,
                    frame_idx=frame_idx,
                    enabled=enable_render_hide_old_proxy,
                    recent_frames=hide_recent_frames,
                    center_px_thr=hide_center_px_thr,
                    iou_thr=hide_iou_thr,
                    min_area_ratio=hide_min_area_ratio,
                    max_area_ratio=hide_max_area_ratio,
                )
                draw_proxy_bbox = hidden_match_real_id is None
                draw_proxy_label = hidden_match_real_id is None
                draw_proxy_overlay = hidden_match_real_id is None
                draw_proxy_klt_points = hidden_match_real_id is None or not hide_klt_points_with_box
                draw_color = color
                draw_thickness = 2
                draw_text_thickness = 1

                if draw_proxy_bbox:
                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), draw_color, draw_thickness, cv2.LINE_AA)
                if draw_proxy_label:
                    cv2.putText(
                        vis,
                        label,
                        (x1i, max(15, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        draw_color,
                        draw_text_thickness,
                        cv2.LINE_AA,
                    )

                if mode == "pose_patch":
                    if draw_proxy_overlay and pose_overlay_bbox is not None:
                        px1, py1, px2, py2 = map(int, pose_overlay_bbox)
                        cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 0, 255), 2, cv2.LINE_AA)
                    if draw_proxy_overlay and mem.get("pose_anchor_xy"):
                        ax, ay = mem["pose_anchor_xy"]
                        cv2.circle(vis, (int(round(ax)), int(round(ay))), 4, (255, 0, 255), -1, cv2.LINE_AA)
                    if draw_proxy_klt_points:
                        draw_klt_points(vis, pose_overlay_points, pose_overlay_bbox, (0, 255, 255))

                frame_logs.append(
                    {
                        "frame_idx": frame_idx,
                        "time_sec": round(now_sec, 4),
                        "track_id": track_id,
                        "bbox_xyxy": [round(v, 2) for v in logic_bbox],
                        "logic_bbox": [round(v, 2) for v in logic_bbox],
                        "render_bbox": [round(v, 2) for v in render_bbox],
                        "render_frozen": bool(render_frozen),
                        "roi_overlap": round(float(current_overlap), 4),
                        "state": state,
                        "render_active": bool(draw_proxy_bbox),
                        "assist_reason": assist_reason,
                        "miss_count": int(mem.get("miss_count", 0)),
                        "klt_success_count": int(mem.get("klt_success_count", 0)),
                        "pose_patch_success_count": int(mem.get("pose_patch_success_count", 0)),
                        "klt_fail_reason": mem.get("klt_fail_reason", ""),
                        **proxy_debug_fields(mem, proxy_alive=proxy_alive and state != "lost"),
                        **pose_debug_fields(mem),
                    }
                )
            for track_id in expired_ids:
                track_memory.pop(track_id, None)

        hud_text = f"mode={mode} frame={frame_idx} t={now_sec:.2f}s tracker={args.tracker}"
        if mode == "pose_patch":
            hud_text = f"{hud_text} proxy={proxy_survival_mode}"
        cv2.putText(
            vis,
            hud_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer_full.write(vis)
        vis_out = cv2.resize(vis, (preview_w, preview_h))
        writer_preview.write(vis_out)

        if args.show:
            cv2.imshow(f"{mode}_bytetrack", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        frame_idx += 1

    cap.release()
    writer_full.release()
    writer_preview.release()
    if args.show:
        cv2.destroyAllWindows()

    shutil.copy2(args.cfg, run.out_dir / Path(args.cfg).name)

    meta = {
        "name": name,
        "mode": mode,
        "video": video_path,
        "roi": roi_path,
        "tracker": args.tracker,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "fps": fps,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "detector": det,
        "pose": pose_cfg if mode == "pose_patch" else {},
        "proxy_survival_mode": proxy_survival_mode if mode == "pose_patch" else "",
        "proxy_fail_tolerance": proxy_fail_tolerance if mode == "pose_patch" else 0,
        "output_video_full": str(out_video_path_full),
        "output_video_preview": str(out_video_path_preview),
        "run_out_dir": str(run.out_dir),
        "run_cmd_path": str(run.cmd_path) if run.cmd_path is not None else "",
        "run_log_path": str(run.log_path) if run.log_path is not None else "",
        "run_ts": run.run_ts,
    }
    with open(run.out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.save_jsonl:
        save_jsonl(run.out_dir / f"{Path(video_path).stem}_{mode}_bytetrack.jsonl", frame_logs)

    logger.info("[DONE] %s bytetrack", mode)
    logger.info("out_dir=%s", run.out_dir)
    logger.info("video_full=%s", out_video_path_full)
    logger.info("video_preview=%s", out_video_path_preview)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("fatal error in main()")
        raise
