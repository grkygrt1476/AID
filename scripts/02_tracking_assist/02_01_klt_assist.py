#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from aidlib.run_utils import common_argparser, init_run
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def init_klt_points(gray_patch: np.ndarray, max_corners: int = 80, quality_level: float = 0.003, min_distance: int = 2, block_size: int = 3):
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
        return None, None, "no_prev_points"

    if prev_gray_patch.shape[:2] != curr_gray_patch.shape[:2]:
        return None, None, f"patch_size_mismatch:{prev_gray_patch.shape[:2]}->{curr_gray_patch.shape[:2]}"

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
        return None, None, "opencv_lk_error"

    if next_pts is None or status is None:
        return None, None, "klt_failed"

    good_prev = prev_points[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]

    if len(good_prev) < 2 or len(good_next) < 2:
        return None, None, "too_few_good_points"

    diffs = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)
    dx = float(np.median(diffs[:, 0]))
    dy = float(np.median(diffs[:, 1]))
    return good_next.reshape(-1, 1, 2), (dx, dy), ""


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.add_argument("--cfg", type=str, required=True, help="experiment yaml path")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker yaml")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "hold", "klt"])
    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--end_sec", type=float, default=None)
    parser.add_argument("--save_jsonl", action="store_true")
    parser.add_argument("--show", action="store_true")
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
    hold_seconds = float(cfg.get("display", {}).get("hold_seconds", 0.3))
    candidate_overlap_thr = float(cfg.get("candidate_overlap_thr", 0.05))
    klt_cfg = cfg.get("klt", {})
    klt_max_miss_frames = int(klt_cfg.get("max_miss_frames", 3))
    klt_patch_scale = float(klt_cfg.get("patch_scale", 1.15))
    klt_center_ratio = float(klt_cfg.get("center_ratio", 0.5))
    klt_top_center_ratio = float(klt_cfg.get("top_center_ratio", 0.45))
    klt_top_anchor_ratio = float(klt_cfg.get("top_anchor_ratio", 0.30))
    klt_max_translation_px = float(klt_cfg.get("max_translation_px", 20.0))

    logger.info("cfg=%s", args.cfg)
    logger.info("video=%s", video_path)
    logger.info("roi=%s", roi_path)
    logger.info("tracker=%s", args.tracker)
    logger.info("mode=%s", mode)
    logger.info("hold_seconds=%s", hold_seconds)
    logger.info("candidate_overlap_thr=%s", candidate_overlap_thr)
    logger.info("klt_max_miss_frames=%s", klt_max_miss_frames)
    logger.info("klt_patch_scale=%s", klt_patch_scale)
    logger.info("klt_center_ratio=%s", klt_center_ratio)
    logger.info("klt_top_center_ratio=%s", klt_top_center_ratio)
    logger.info("klt_top_anchor_ratio=%s", klt_top_anchor_ratio)
    logger.info("klt_max_translation_px=%s", klt_max_translation_px)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                x1, y1, x2, y2 = map(float, xyxy[i])
                track_id = int(ids[i])
                cls_id = int(clss[i])
                score = float(confs[i])
                overlap = bbox_roi_overlap_ratio(x1, y1, x2, y2, roi_mask)

                x1i, y1i, x2i, y2i = clamp_xyxy(x1, y1, x2, y2, frame_w, frame_h)
                cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2, cv2.LINE_AA)
                label = f"id={track_id} conf={score:.2f} ov={overlap:.2f}"
                cv2.putText(
                    vis,
                    label,
                    (x1i, max(15, y1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                frame_logs.append(
                    {
                        "frame_idx": frame_idx,
                        "time_sec": round(now_sec, 4),
                        "track_id": track_id,
                        "cls_id": cls_id,
                        "conf": round(score, 4),
                        "bbox_xyxy": [round(v, 2) for v in [x1, y1, x2, y2]],
                        "roi_overlap": round(overlap, 4),
                        "state": "real",
                    }
                )

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

                track_memory[track_id] = {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "hold_until_sec": now_sec + hold_seconds,
                    "roi_overlap": overlap,
                    "prev_gray_patch_center": patch_gray_center,
                    "prev_points_center": patch_pts_center,
                    "patch_bbox_center": list(patch_bbox_center),
                    "prev_gray_patch_top": patch_gray_top,
                    "prev_points_top": patch_pts_top,
                    "patch_bbox_top": list(patch_bbox_top),
                    "miss_count": 0,
                    "klt_success_count": track_memory.get(track_id, {}).get("klt_success_count", 0),
                    "klt_fail_reason": "",
                }

        if mode in ("hold", "klt"):
            expired_ids = []
            for track_id, mem in track_memory.items():
                if track_id in seen_ids:
                    continue
                if now_sec > float(mem["hold_until_sec"]):
                    expired_ids.append(track_id)
                    continue
                if float(mem.get("roi_overlap", 0.0)) < candidate_overlap_thr:
                    continue

                state = "hold"
                draw_bbox = list(mem["bbox_xyxy"])

                mem["miss_count"] = int(mem.get("miss_count", 0)) + 1

                if mode == "klt" and mem["miss_count"] <= klt_max_miss_frames:
                    candidates = []
                    fail_reasons = []

                    # center patch
                    patch_bgr_center, patch_bbox_center = make_klt_patch(
                        frame,
                        mem["bbox_xyxy"],
                        patch_scale=klt_patch_scale,
                        center_ratio=klt_center_ratio,
                        anchor="center",
                        top_anchor_ratio=klt_top_anchor_ratio,
                    )
                    patch_gray_center = cv2.cvtColor(patch_bgr_center, cv2.COLOR_BGR2GRAY) if patch_bgr_center.size > 0 else None
                    next_pts_center, delta_xy_center, fail_reason_center = run_klt_shift(
                        mem.get("prev_gray_patch_center"),
                        patch_gray_center,
                        mem.get("prev_points_center"),
                    )
                    if delta_xy_center is not None:
                        candidates.append(("center", next_pts_center, patch_gray_center, patch_bbox_center, delta_xy_center))
                    else:
                        fail_reasons.append(f"center:{fail_reason_center}")

                    # top patch
                    patch_bgr_top, patch_bbox_top = make_klt_patch(
                        frame,
                        mem["bbox_xyxy"],
                        patch_scale=klt_patch_scale,
                        center_ratio=klt_top_center_ratio,
                        anchor="top",
                        top_anchor_ratio=klt_top_anchor_ratio,
                    )
                    patch_gray_top = cv2.cvtColor(patch_bgr_top, cv2.COLOR_BGR2GRAY) if patch_bgr_top.size > 0 else None
                    next_pts_top, delta_xy_top, fail_reason_top = run_klt_shift(
                        mem.get("prev_gray_patch_top"),
                        patch_gray_top,
                        mem.get("prev_points_top"),
                    )
                    if delta_xy_top is not None:
                        candidates.append(("top", next_pts_top, patch_gray_top, patch_bbox_top, delta_xy_top))
                    else:
                        fail_reasons.append(f"top:{fail_reason_top}")

                    # 다음 프레임용 patch state는 둘 다 업데이트
                    mem["prev_gray_patch_center"] = patch_gray_center
                    mem["prev_points_center"] = next_pts_center if next_pts_center is not None else (init_klt_points(patch_gray_center) if patch_gray_center is not None else None)
                    mem["patch_bbox_center"] = list(patch_bbox_center)

                    mem["prev_gray_patch_top"] = patch_gray_top
                    mem["prev_points_top"] = next_pts_top if next_pts_top is not None else (init_klt_points(patch_gray_top) if patch_gray_top is not None else None)
                    mem["patch_bbox_top"] = list(patch_bbox_top)

                    if candidates:
                        # 점이 더 많이 살아남은 patch 우선
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
                    else:
                        mem["klt_fail_reason"] = "|".join(fail_reasons)

                x1, y1, x2, y2 = draw_bbox
                x1i, y1i, x2i, y2i = clamp_xyxy(x1, y1, x2, y2, frame_w, frame_h)

                color = (0, 165, 255) if state == "hold" else (255, 255, 0)
                label = f"id={track_id} {state}"

                cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    label,
                    (x1i, max(15, y1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                frame_logs.append(
                    {
                        "frame_idx": frame_idx,
                        "time_sec": round(now_sec, 4),
                        "track_id": track_id,
                        "bbox_xyxy": [round(v, 2) for v in [x1, y1, x2, y2]],
                        "roi_overlap": round(float(mem.get("roi_overlap", 0.0)), 4),
                        "state": state,
                        "miss_count": int(mem.get("miss_count", 0)),
                        "klt_success_count": int(mem.get("klt_success_count", 0)),
                        "klt_fail_reason": mem.get("klt_fail_reason", ""),
                    }
                )

            for track_id in expired_ids:
                track_memory.pop(track_id, None)

        cv2.putText(
            vis,
            f"mode={mode} frame={frame_idx} t={now_sec:.2f}s tracker={args.tracker}",
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
        "output_video_full": str(out_video_path_full),
        "output_video_preview": str(out_video_path_preview),
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
