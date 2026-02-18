#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aidlib import run_utils

STAGE = "01_runtime"


# -------------------------
# Data structures
# -------------------------
@dataclass
class SourceInfo:
    source: str
    video_id: str
    is_rtsp: bool


@dataclass
class GhostState:
    last_bbox: List[float]         # xyxy
    last_ts: float
    v: List[float]                 # bbox velocity (dx1,dy1,dx2,dy2) per sec
    ghost_until_ts: float


# -------------------------
# Utils
# -------------------------
def resolve_source(args) -> SourceInfo:
    source = (args.source or "").strip()
    video_id = (args.video_id or "").strip()

    if not source:
        if not video_id:
            raise ValueError("Either --source or --video_id must be provided.")
        source = str(Path("data/videos") / f"{video_id}.mp4")
    elif not video_id:
        p = Path(source)
        if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            video_id = p.stem

    is_rtsp = bool(args.rtsp) or source.lower().startswith("rtsp://")
    return SourceInfo(source=source, video_id=video_id, is_rtsp=is_rtsp)


def get_versions() -> Dict[str, Optional[str]]:
    v = {"python": platform.python_version(), "cv2": None, "ultralytics": None, "torch": None, "boxmot": None}
    try:
        import cv2
        v["cv2"] = getattr(cv2, "__version__", None)
    except Exception:
        pass
    try:
        import ultralytics
        v["ultralytics"] = getattr(ultralytics, "__version__", None)
    except Exception:
        pass
    try:
        import torch
        v["torch"] = getattr(torch, "__version__", None)
    except Exception:
        pass
    try:
        import boxmot
        v["boxmot"] = getattr(boxmot, "__version__", None)
    except Exception:
        pass
    return v


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clip_xyxy(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


# -------------------------
# BoxMOT wrapper (robust to API variations)
# -------------------------
def _find_tracker_class(boxmot_module, tracker_name: str):
    name = tracker_name.lower().replace("-", "").replace("_", "")
    candidates = {
        "deepocsort": ["DeepOCSORT", "DeepOCSort", "DeepOcSort"],
        "strongsort": ["StrongSORT", "StrongSort"],
        "botsort": ["BoTSORT", "BoTSort"],
        "bytetrack": ["BYTETracker", "ByteTrack", "ByteTracker"],
        "ocsort": ["OCSORT", "OcSort"],
        "boosttrack": ["BoostTrack", "BoostTRACK"],
        "hybridsort": ["HybridSORT", "HybridSort"],
        "sfsort": ["SFSORT", "SfSort"],
    }.get(name, [])
    for cls_name in candidates:
        if hasattr(boxmot_module, cls_name):
            return getattr(boxmot_module, cls_name)
    # fallback: try attribute by upper-ish
    for attr in dir(boxmot_module):
        if attr.lower().replace("_", "") == name:
            return getattr(boxmot_module, attr)
    raise ValueError(f"Unknown tracker '{tracker_name}' or not found in boxmot module.")


def make_boxmot_tracker(
    tracker_name: str,
    reid_weights: str,
    device: str,
    fp16: bool,
    per_class: bool,
):
    import inspect
    import boxmot
    cls = _find_tracker_class(boxmot, tracker_name)

    # Try multiple common kw names (boxmot versions differ)
    kw_variants = [
        {"reid_weights": Path(reid_weights) if reid_weights else None, "device": device, "half": fp16, "per_class": per_class},
        {"model_weights": Path(reid_weights) if reid_weights else None, "device": device, "fp16": fp16, "per_class": per_class},
        {"weights": Path(reid_weights) if reid_weights else None, "device": device, "half": fp16, "per_class": per_class},
        {"device": device, "half": fp16, "per_class": per_class},
        {"device": device, "fp16": fp16, "per_class": per_class},
        {"device": device},
    ]

    sig = None
    try:
        sig = inspect.signature(cls)
    except Exception:
        sig = None

    last_err = None
    for kw in kw_variants:
        # drop None-valued keys
        kw2 = {k: v for k, v in kw.items() if v is not None}
        if sig is not None:
            allowed = set(sig.parameters.keys())
            kw2 = {k: v for k, v in kw2.items() if k in allowed}
        try:
            return cls(**kw2)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to instantiate BoxMOT tracker {tracker_name}: {last_err}")


def tracker_update(tracker, dets_xyxy_conf_cls, frame):
    # dets: Nx6 float32 [x1,y1,x2,y2,conf,cls]
    try:
        return tracker.update(dets_xyxy_conf_cls, frame)
    except TypeError:
        return tracker.update(dets=dets_xyxy_conf_cls, img=frame)


def parse_tracker_outputs(out) -> List[Dict[str, Any]]:
    import numpy as np

    if out is None:
        return []
    arr = np.asarray(out)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    # Typical: [x1,y1,x2,y2,id,conf,cls,ind] OR [x1,y1,x2,y2,id,conf,cls]
    tracks = []
    for row in arr:
        row = row.tolist()
        if len(row) < 5:
            continue
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        tid = int(row[4])
        conf = float(row[5]) if len(row) >= 6 else None
        cls = int(row[6]) if len(row) >= 7 else None
        tracks.append({"track_id": tid, "bbox_xyxy": [x1, y1, x2, y2], "conf": conf, "cls": cls})
    return tracks


# -------------------------
# Pose helpers
# -------------------------
COCO_EDGES = [
    (5, 7), (7, 9),   # L shoulder-elbow-wrist
    (6, 8), (8, 10),  # R shoulder-elbow-wrist
    (5, 6),           # shoulders
    (5, 11), (6, 12), # torso
    (11, 12),         # hips
    (11, 13), (13, 15), # L hip-knee-ankle
    (12, 14), (14, 16), # R hip-knee-ankle
    (0, 1), (0, 2), (1, 3), (2, 4), # head
    (3, 5), (4, 6),   # head to shoulders
]


def pick_anchor(kxy, kcf, mode: str, thr: float) -> Tuple[Optional[Tuple[float, float]], str]:
    # COCO keypoint indices: 11/12 hips, 15/16 ankles
    def _pt(i):
        return (float(kxy[i][0]), float(kxy[i][1]))

    def _ok(i):
        return float(kcf[i]) >= thr

    mode = (mode or "auto").lower()
    if mode == "ankle":
        # prefer mid of two ankles if both ok, else single
        if _ok(15) and _ok(16):
            ax = 0.5 * (_pt(15)[0] + _pt(16)[0])
            ay = 0.5 * (_pt(15)[1] + _pt(16)[1])
            return (ax, ay), "ankle_mid"
        if _ok(15):
            return _pt(15), "l_ankle"
        if _ok(16):
            return _pt(16), "r_ankle"
        return None, "ankle_none"

    if mode == "pelvis":
        if _ok(11) and _ok(12):
            px = 0.5 * (_pt(11)[0] + _pt(12)[0])
            py = 0.5 * (_pt(11)[1] + _pt(12)[1])
            return (px, py), "pelvis_mid"
        if _ok(11):
            return _pt(11), "l_hip"
        if _ok(12):
            return _pt(12), "r_hip"
        return None, "pelvis_none"

    # auto: ankles -> pelvis -> none
    pt, tag = pick_anchor(kxy, kcf, "ankle", thr)
    if pt is not None:
        return pt, tag
    pt, tag = pick_anchor(kxy, kcf, "pelvis", thr)
    return pt, tag


def draw_skeleton_cv2(cv2, img, kxy, kcf, thr: float, draw_edges: bool, draw_points: bool) -> None:
    if kxy is None or kcf is None:
        return
    if draw_edges:
        for a, b in COCO_EDGES:
            if float(kcf[a]) >= thr and float(kcf[b]) >= thr:
                x1, y1 = int(kxy[a][0]), int(kxy[a][1])
                x2, y2 = int(kxy[b][0]), int(kxy[b][1])
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    if draw_points:
        for i in range(len(kxy)):
            if float(kcf[i]) >= thr:
                x, y = int(kxy[i][0]), int(kxy[i][1])
                cv2.circle(img, (x, y), 3, (0, 255, 255), -1)


def pose_predict_full(pose_model, frame, imgsz: int, conf: float, device: str):
    res = pose_model.predict(source=frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
    r0 = res[0] if res else None
    if r0 is None or getattr(r0, "boxes", None) is None:
        return [], None, None
    boxes = r0.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
    confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else []
    dets = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
        dets.append({"bbox_xyxy": [x1, y1, x2, y2], "conf": float(confs[i]) if i < len(confs) else 0.0})
    kxy = None
    kcf = None
    if getattr(r0, "keypoints", None) is not None:
        kxy = r0.keypoints.xy.detach().cpu().numpy()      # (N,17,2)
        kcf = r0.keypoints.conf.detach().cpu().numpy()    # (N,17)
    return dets, kxy, kcf


def pose_predict_crop(pose_model, frame, crop_xyxy: List[float], imgsz: int, conf: float, device: str):
    # returns (kxy_full, kcf) in FULL-frame coords if 1st person found in crop
    x1, y1, x2, y2 = [int(v) for v in crop_xyxy]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2].copy()
    res = pose_model.predict(source=crop, imgsz=imgsz, conf=conf, device=device, verbose=False)
    r0 = res[0] if res else None
    if r0 is None or getattr(r0, "keypoints", None) is None:
        return None, None
    kxy = r0.keypoints.xy.detach().cpu().numpy()
    kcf = r0.keypoints.conf.detach().cpu().numpy()
    if kxy is None or len(kxy) == 0:
        return None, None

    # pick best by box conf if available, else first
    idx = 0
    if getattr(r0, "boxes", None) is not None and getattr(r0.boxes, "conf", None) is not None:
        confs = r0.boxes.conf.detach().cpu().numpy()
        if len(confs) > 0:
            idx = int(confs.argmax())
    kxy0 = kxy[idx]
    kcf0 = kcf[idx]

    # map to full coords
    kxy_full = kxy0.copy()
    kxy_full[:, 0] += float(x1)
    kxy_full[:, 1] += float(y1)
    return kxy_full, kcf0


# -------------------------
# Ghost logic
# -------------------------
def update_ghost(memory: Dict[int, GhostState], tid: int, bbox: List[float], ts: float, ttl: float, vel_alpha: float):
    if tid not in memory:
        memory[tid] = GhostState(last_bbox=bbox, last_ts=ts, v=[0.0, 0.0, 0.0, 0.0], ghost_until_ts=ts + ttl)
        return
    st = memory[tid]
    dt = max(1e-6, ts - st.last_ts)
    raw_v = [(bbox[i] - st.last_bbox[i]) / dt for i in range(4)]
    st.v = [vel_alpha * raw_v[i] + (1.0 - vel_alpha) * st.v[i] for i in range(4)]
    st.last_bbox = bbox
    st.last_ts = ts
    st.ghost_until_ts = ts + ttl


def predict_ghost(st: GhostState, ts: float) -> List[float]:
    dt = max(0.0, ts - st.last_ts)
    return [st.last_bbox[i] + st.v[i] * dt for i in range(4)]


# -------------------------
# CLI
# -------------------------
def build_parser():
    p = run_utils.common_argparser()
    # IMPORTANT: 서버에서 /home/serdic 권한 문제 터지기 쉬워서 기본을 상대경로로 덮어씀
    p.set_defaults(out_root="outputs", log_root="outputs/logs")

    p.add_argument("--source", default="", help="mp4 path OR rtsp url; if empty, use --video_id")
    p.add_argument("--video_id", default="", help="e.g., E01_006 -> data/videos/E01_006.mp4")
    p.add_argument("--rtsp", action="store_true", help="force treat source as RTSP")

    # Detector for tracking (bbox)
    p.add_argument("--det_model", default="yolov8m.pt", help="Ultralytics YOLO det weights for BoxMOT detections")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", type=int, nargs="*", default=[0], help="COCO class indices to keep (default: 0=person).")

    # BoxMOT tracker
    p.add_argument("--tracker", default="deepocsort", help="deepocsort|strongsort|botsort|bytetrack|ocsort|...")
    p.add_argument("--reid_model", default="osnet_x0_25_msmt17.pt", help="ReID weights (if tracker uses it)")
    p.add_argument("--device", default="0", help="cuda device index like 0, or 'cpu'")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--per_class", action="store_true")

    # Pose overlay
    p.add_argument("--pose_model", default="yolov8n-pose.pt", help="Ultralytics pose model weights")
    p.add_argument("--pose_conf", type=float, default=0.25)
    p.add_argument("--draw_skeleton", action="store_true")
    p.add_argument("--draw_kp", action="store_true")
    p.add_argument("--kp_thr", type=float, default=0.35)
    p.add_argument("--anchor_mode", default="pelvis", help="pelvis|ankle|auto")
    p.add_argument("--anchor_thr", type=float, default=0.35)

    # Ghost
    p.add_argument("--ghost_ttl_sec", type=float, default=0.5, help="how long to keep ghost bbox after track missing")
    p.add_argument("--ghost_vel_alpha", type=float, default=0.6, help="velocity EMA alpha (higher=faster react)")
    p.add_argument("--ghost_pose", action="store_true", help="try pose on ghost bbox crop (cost↑, 복구↑)")
    p.add_argument("--ghost_pose_pad", type=float, default=0.15, help="crop padding ratio for ghost pose")

    # IO / display
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--save_jsonl", action="store_true")
    p.add_argument("--no_show", action="store_true", help="headless mode (server)")

    return p


def _device_str(args_device: str) -> str:
    d = str(args_device).strip()
    if d.lower() == "cpu":
        return "cpu"
    if d.isdigit():
        return f"cuda:{d}"
    return d


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # auto headless if no DISPLAY and user didn't explicitly set
    if not args.no_show:
        if os.environ.get("DISPLAY", "") == "":
            args.no_show = True

    run_paths = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    try:
        import cv2
    except Exception as e:
        logger.exception("Failed to import cv2: %s", e)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as e:
        logger.exception("Failed to import ultralytics: %s", e)
        return 2

    try:
        import numpy as np
    except Exception as e:
        logger.exception("Failed to import numpy: %s", e)
        return 2

    try:
        import boxmot  # noqa: F401
    except Exception as e:
        logger.exception("Failed to import boxmot: %s", e)
        return 2

    # Resolve source
    try:
        src = resolve_source(args)
    except ValueError as e:
        logger.error(str(e))
        logger.error("Usage: %s", parser.format_usage().strip())
        return 2

    if (not src.is_rtsp) and (not Path(src.source).exists()):
        logger.error("Input file not found: %s", src.source)
        return 2

    device = _device_str(args.device)

    # Models
    logger.info("Loading det model: %s", args.det_model)
    det_model = YOLO(args.det_model)

    logger.info("Loading pose model: %s", args.pose_model)
    pose_model = YOLO(args.pose_model)

    logger.info("Creating BoxMOT tracker: %s (reid=%s) device=%s fp16=%s", args.tracker, args.reid_model, device, args.fp16)
    tracker = make_boxmot_tracker(
        tracker_name=args.tracker,
        reid_weights=args.reid_model,
        device=device,
        fp16=bool(args.fp16),
        per_class=bool(args.per_class),
    )

    # Video IO
    cap = cv2.VideoCapture(src.source)
    if not cap.isOpened():
        logger.error("Failed to open source: %s", src.source)
        return 2

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        ok, fr = cap.read()
        if not ok or fr is None:
            logger.error("Failed to read first frame for size.")
            return 2
        h, w = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_mp4 = run_paths.out_dir / f"{args.out_base}_tracked_pose_ghost.mp4"
    out_fps = in_fps if in_fps > 0 else 30.0
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    if not vw.isOpened():
        logger.error("Failed to open VideoWriter: %s", out_mp4)
        return 2

    events_fp = None
    if args.save_jsonl:
        events_fp = (run_paths.out_dir / "tracks_pose_ghost.jsonl").open("w", encoding="utf-8")

    params = vars(args).copy()
    params["source"] = src.source
    params["video_id"] = src.video_id
    params["is_rtsp"] = src.is_rtsp
    params["in_fps"] = in_fps
    params["width"] = w
    params["height"] = h
    params["versions"] = get_versions()
    write_json(run_paths.out_dir / "params.json", params)

    logger.info("Input: %s (%dx%d fps=%.3f)", src.source, w, h, in_fps)
    logger.info("Output: %s", out_mp4)

    # State
    ghost_mem: Dict[int, GhostState] = {}
    last_active_ids: set[int] = set()

    keep_cls = set(int(x) for x in (args.classes or []))
    frame_idx = -1
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            # ts
            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            ts_sec = (frame_idx / in_fps) if in_fps > 0 else (pos_ms / 1000.0 if pos_ms > 0 else 0.0)

            # --- Detection for tracking (bbox) ---
            det_res = det_model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            dr0 = det_res[0] if det_res else None

            dets = []
            if dr0 is not None and getattr(dr0, "boxes", None) is not None:
                b = dr0.boxes
                xyxy = b.xyxy.detach().cpu().numpy() if getattr(b, "xyxy", None) is not None else []
                confs = b.conf.detach().cpu().numpy() if getattr(b, "conf", None) is not None else []
                cls = b.cls.detach().cpu().numpy() if getattr(b, "cls", None) is not None else None

                for i in range(len(xyxy)):
                    c = int(cls[i]) if cls is not None else 0
                    if keep_cls and c not in keep_cls:
                        continue
                    sc = float(confs[i]) if i < len(confs) else 0.0
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                    dets.append([x1, y1, x2, y2, sc, float(c)])

            det_arr = np.asarray(dets, dtype=np.float32) if len(dets) else np.zeros((0, 6), dtype=np.float32)

            # --- BoxMOT update ---
            tracks_out = tracker_update(tracker, det_arr, frame)
            tracks = parse_tracker_outputs(tracks_out)
            active_ids = set(int(t["track_id"]) for t in tracks)

            # --- Pose inference (full frame) ---
            pose_dets, kpts_xy, kpts_conf = pose_predict_full(
                pose_model=pose_model,
                frame=frame,
                imgsz=args.imgsz,
                conf=args.pose_conf,
                device=args.device,
            )

            # track -> pose det matching (IoU)
            track_to_pose_idx: Dict[int, int] = {}
            if len(tracks) and len(pose_dets):
                for tr in tracks:
                    tid = int(tr["track_id"])
                    tb = tr["bbox_xyxy"]
                    best_iou, best_j = 0.0, -1
                    for j, pd in enumerate(pose_dets):
                        v = iou_xyxy(tb, pd["bbox_xyxy"])
                        if v > best_iou:
                            best_iou, best_j = v, j
                    if best_j >= 0 and best_iou >= 0.2:
                        track_to_pose_idx[tid] = best_j

            # --- Update ghost memory from ACTIVE tracks ---
            for tr in tracks:
                tid = int(tr["track_id"])
                bbox = clip_xyxy(tr["bbox_xyxy"], w, h)
                update_ghost(ghost_mem, tid, bbox, ts_sec, args.ghost_ttl_sec, args.ghost_vel_alpha)

            # IDs that disappeared this frame (candidate for ghost)
            vanished = last_active_ids - active_ids
            last_active_ids = active_ids

            # --- Draw ---
            vis = frame.copy()

            # Draw active tracks (green)
            for tr in tracks:
                tid = int(tr["track_id"])
                bbox = clip_xyxy(tr["bbox_xyxy"], w, h)
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 220, 60), 2)
                cv2.putText(vis, f"id={tid}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # pose overlay if matched
                pidx = track_to_pose_idx.get(tid)
                if pidx is not None and kpts_xy is not None and kpts_conf is not None:
                    kxy = kpts_xy[pidx]
                    kcf = kpts_conf[pidx]
                    if args.draw_skeleton or args.draw_kp:
                        draw_skeleton_cv2(cv2, vis, kxy, kcf, float(args.kp_thr), args.draw_skeleton, args.draw_kp)
                    anchor_pt, anchor_tag = pick_anchor(kxy, kcf, args.anchor_mode, float(args.anchor_thr))
                    if anchor_pt is not None:
                        ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
                        cv2.circle(vis, (ax, ay), 6, (0, 165, 255), -1)
                        cv2.putText(vis, anchor_tag, (ax + 6, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    if events_fp is not None:
                        rec = {
                            "frame_idx": frame_idx,
                            "ts_sec": ts_sec,
                            "track_id": tid,
                            "bbox_xyxy": [float(v) for v in bbox],
                            "pose_det_idx": int(pidx),
                            "anchor_tag": anchor_tag,
                            "anchor_xy": [float(anchor_pt[0]), float(anchor_pt[1])] if anchor_pt is not None else None,
                        }
                        events_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Draw ghost for vanished + still-alive mem (orange thin)
            # (단, active로 돌아온 건 이미 위에서 update_ghost로 갱신되므로 자연히 ghost로 안 그림)
            for tid, st in list(ghost_mem.items()):
                if tid in active_ids:
                    continue
                if ts_sec > st.ghost_until_ts:
                    ghost_mem.pop(tid, None)
                    continue

                gb = clip_xyxy(predict_ghost(st, ts_sec), w, h)
                x1, y1, x2, y2 = [int(v) for v in gb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 1)
                cv2.putText(vis, f"id={tid} GHOST", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # optional: pose on ghost crop (복구 목적)
                if args.ghost_pose:
                    pad_x = int((x2 - x1) * float(args.ghost_pose_pad))
                    pad_y = int((y2 - y1) * float(args.ghost_pose_pad))
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w - 1, x2 + pad_x)
                    cy2 = min(h - 1, y2 + pad_y)
                    kxy_g, kcf_g = pose_predict_crop(
                        pose_model=pose_model,
                        frame=frame,
                        crop_xyxy=[cx1, cy1, cx2, cy2],
                        imgsz=args.imgsz,
                        conf=max(0.10, float(args.pose_conf) * 0.8),
                        device=args.device,
                    )
                    if kxy_g is not None and kcf_g is not None:
                        # "키포인트 하나라도 잡히면" 사용자 체감상 복구에 도움 -> 시각화
                        draw_skeleton_cv2(cv2, vis, kxy_g, kcf_g, float(args.kp_thr), args.draw_skeleton, args.draw_kp)
                        anchor_pt, anchor_tag = pick_anchor(kxy_g, kcf_g, args.anchor_mode, float(args.anchor_thr))
                        if anchor_pt is not None:
                            ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
                            cv2.circle(vis, (ax, ay), 6, (0, 165, 255), -1)
                            cv2.putText(vis, f"{anchor_tag}(G)", (ax + 6, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # HUD
            elapsed = max(1e-6, time.time() - t0)
            fps = (frame_idx + 1) / elapsed
            cv2.putText(vis, f"frame={frame_idx} fps={fps:.1f} active={len(active_ids)} ghost={len(ghost_mem)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            vw.write(vis)

            if frame_idx % 30 == 0:
                logger.info("progress frame=%d fps=%.2f active=%d ghost=%d dets=%d", frame_idx, fps, len(active_ids), len(ghost_mem), len(dets))

            if not args.no_show:
                cv2.imshow("AID BoxMOT Pose Ghost", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        cap.release()
        vw.release()
        if events_fp is not None:
            events_fp.close()
        if not args.no_show:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    logger.info("Done. saved=%s", out_mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
