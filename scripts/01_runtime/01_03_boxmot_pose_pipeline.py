#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aidlib import run_utils

STAGE = "01_runtime"


@dataclass
class SourceInfo:
    source: str
    video_id: str
    is_rtsp: bool


# COCO-17 keypoint indices
KP_NOSE = 0
KP_LHIP, KP_RHIP = 11, 12
KP_LANKLE, KP_RANKLE = 15, 16

# COCO skeleton edges (index pairs)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def build_parser():
    p = run_utils.common_argparser()
    p.add_argument("--source", default="", help="mp4 path OR rtsp url; if empty, use --video_id")
    p.add_argument("--video_id", default="", help="e.g., E01_006 -> data/videos/E01_006.mp4")
    p.add_argument("--rtsp", action="store_true", help="force treat source as RTSP (seek/clip not supported)")

    # Pose + detector
    p.add_argument("--pose_model", default="yolov8m-pose.pt", help="Ultralytics pose weights (e.g., yolov8m-pose.pt)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", type=int, nargs="*", default=[0], help="class ids to keep (default: person=0)")
    p.add_argument("--device", default="0", help="Ultralytics device (e.g., 0, 0,1, cpu)")

    # BoxMOT tracker
    p.add_argument("--tracker", default="deepocsort", choices=["deepocsort", "strongsort"],
                   help="BoxMOT tracker backend")
    p.add_argument("--reid_model", default="osnet_x0_25_msmt17.pt",
                   help="ReID weights for BoxMOT (BoxMOT may auto-download if missing)")
    p.add_argument("--fp16", action="store_true", help="try fp16/half if supported by tracker")
    p.add_argument("--per_class", action="store_true", help="track per class (usually False for person-only)")

    # Matching det <-> track for keypoints
    p.add_argument("--match_iou", type=float, default=0.3, help="IoU threshold to attach keypoints to a track")

    # Drawing
    p.add_argument("--kp_thr", type=float, default=0.30, help="min keypoint conf to draw")
    p.add_argument("--draw_skeleton", action="store_true", help="draw skeleton edges (default off if not set)")
    p.add_argument("--draw_kp", action="store_true", help="draw keypoint dots (default off if not set)")
    p.add_argument("--draw_anchor", default="pelvis", choices=["pelvis", "ankle", "auto"],
                   help="anchor point mode for highlighting")
    p.add_argument("--anchor_thr", type=float, default=0.30, help="min kp conf for anchor selection")
    p.add_argument("--no_show", action="store_true", help="do not cv2.imshow (recommended on server)")
    p.add_argument("--save_jsonl", action="store_true", help="save per-frame track+anchor jsonl")
    p.add_argument("--max_frames", type=int, default=0, help="0=all")

    p.add_argument("--occl_pose_fallback", action="store_true",
               help="when track missing, run pose on last bbox crop to bridge occlusion")
    p.add_argument("--occl_max_sec", type=float, default=0.6,
                help="max seconds to keep trying fallback after last seen")
    p.add_argument("--occl_expand", type=float, default=0.25,
                help="bbox expand ratio for crop (e.g., 0.25 = 25%)")
    p.add_argument("--occl_pose_conf", type=float, default=0.15,
                help="pose conf for fallback crop inference")
    p.add_argument("--occl_kp_thr", type=float, default=0.25,
                help="keypoint conf threshold for accepting fallback")
    p.add_argument("--occl_min_kps", type=int, default=4,
                help="min #keypoints above thr to accept fallback")
    p.add_argument("--occl_max_tracks", type=int, default=4,
                help="max fallback crops per frame (compute guard)")


    return p


def resolve_source(args) -> SourceInfo:
    source = args.source.strip()
    video_id = args.video_id.strip()

    if not source:
        if not video_id:
            raise ValueError("Either --source or --video_id must be provided.")
        source = str(Path("data/videos") / f"{video_id}.mp4")
    elif not video_id:
        sp = Path(source)
        if sp.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]:
            video_id = sp.stem

    is_rtsp = bool(args.rtsp) or source.lower().startswith("rtsp://")
    return SourceInfo(source=source, video_id=video_id, is_rtsp=is_rtsp)


def get_versions(cv2_module, ultralytics_module, torch_module, boxmot_module) -> dict:
    return {
        "python": platform.python_version(),
        "cv2": getattr(cv2_module, "__version__", None),
        "ultralytics": getattr(ultralytics_module, "__version__", None),
        "torch": getattr(torch_module, "__version__", None) if torch_module is not None else None,
        "boxmot": getattr(boxmot_module, "__version__", None),
    }


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    if union <= 0:
        return 0.0
    return inter / union


def pick_anchor(
    kxy, kconf, mode: str, thr: float
) -> Tuple[Optional[Tuple[float, float]], str]:
    # kxy: (17,2) , kconf: (17,)
    def _best_of(idx_a: int, idx_b: int, tag: str):
        a_ok = float(kconf[idx_a]) >= thr
        b_ok = float(kconf[idx_b]) >= thr
        if not (a_ok or b_ok):
            return None, "NONE"
        if a_ok and b_ok:
            # pick higher-conf point
            ia = idx_a if float(kconf[idx_a]) >= float(kconf[idx_b]) else idx_b
            return (float(kxy[ia][0]), float(kxy[ia][1])), tag
        ia = idx_a if a_ok else idx_b
        return (float(kxy[ia][0]), float(kxy[ia][1])), tag

    if mode == "ankle":
        return _best_of(KP_LANKLE, KP_RANKLE, "ANKLE")
    if mode == "pelvis":
        lh_ok = float(kconf[KP_LHIP]) >= thr
        rh_ok = float(kconf[KP_RHIP]) >= thr
        if lh_ok and rh_ok:
            x = (float(kxy[KP_LHIP][0]) + float(kxy[KP_RHIP][0])) / 2.0
            y = (float(kxy[KP_LHIP][1]) + float(kxy[KP_RHIP][1])) / 2.0
            return (x, y), "PELVIS"
        if lh_ok or rh_ok:
            idx = KP_LHIP if float(kconf[KP_LHIP]) >= float(kconf[KP_RHIP]) else KP_RHIP
            return (float(kxy[idx][0]), float(kxy[idx][1])), "HIP1"
        return None, "NONE"

    # auto: ankle first, then pelvis
    pt, tag = _best_of(KP_LANKLE, KP_RANKLE, "ANKLE")
    if pt is not None:
        return pt, tag
    return pick_anchor(kxy, kconf, "pelvis", thr)


def draw_skeleton_cv2(cv2, img, kxy, kconf, thr: float, draw_edges: bool, draw_points: bool):
    if draw_edges:
        for a, b in COCO_SKELETON:
            if float(kconf[a]) < thr or float(kconf[b]) < thr:
                continue
            x1, y1 = int(kxy[a][0]), int(kxy[a][1])
            x2, y2 = int(kxy[b][0]), int(kxy[b][1])
            cv2.line(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
    if draw_points:
        for i in range(len(kxy)):
            if float(kconf[i]) < thr:
                continue
            x, y = int(kxy[i][0]), int(kxy[i][1])
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)


def draw_both_ankles_cv2(cv2, img, kxy, kconf, thr: float):
    if float(kconf[KP_LANKLE]) >= thr:
        lx, ly = int(kxy[KP_LANKLE][0]), int(kxy[KP_LANKLE][1])
        cv2.circle(img, (lx, ly), 6, (0, 165, 255), -1)
        cv2.putText(
            img, "L", (lx + 6, ly - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )
    if float(kconf[KP_RANKLE]) >= thr:
        rx, ry = int(kxy[KP_RANKLE][0]), int(kxy[KP_RANKLE][1])
        cv2.circle(img, (rx, ry), 6, (0, 165, 255), -1)
        cv2.putText(
            img, "R", (rx + 6, ry - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )


def parse_tracker_outputs(tracks_out) -> List[Dict[str, Any]]:
    """
    Try to normalize tracker outputs to:
    [{'track_id': int, 'bbox_xyxy': [x1,y1,x2,y2], 'conf': float|None, 'cls': int|None}, ...]
    """
    import numpy as np

    if tracks_out is None:
        return []
    if isinstance(tracks_out, list):
        arr = np.array(tracks_out) if len(tracks_out) > 0 else np.zeros((0, 0))
    else:
        arr = np.array(tracks_out)

    if arr.size == 0:
        return []

    # common patterns:
    # (N, 5): x1,y1,x2,y2,track_id
    # (N, 6): x1,y1,x2,y2,track_id,conf OR x1,y1,x2,y2,conf,track_id (varies)
    # (N, 7+): x1,y1,x2,y2,track_id,conf,cls ...
    out: List[Dict[str, Any]] = []
    for row in arr:
        r = [float(x) for x in row.tolist()]
        if len(r) < 5:
            continue

        x1, y1, x2, y2 = r[0], r[1], r[2], r[3]

        # Heuristic: track_id often near last columns and is integer-ish
        cand_ids = []
        for idx in range(4, min(len(r), 9)):
            v = r[idx]
            if abs(v - round(v)) < 1e-3 and v >= 0:
                cand_ids.append((idx, int(round(v))))
        if not cand_ids:
            continue
        # Prefer the first integer-ish after bbox (common: index 4)
        tid_idx, tid = cand_ids[0]

        conf = None
        cls = None
        # If there's a float conf somewhere (0~1), try to pick it from remaining cols
        for j in range(4, len(r)):
            if j == tid_idx:
                continue
            v = r[j]
            if 0.0 <= v <= 1.0:
                conf = float(v)
                break
        # cls guess: if any remaining integer-ish small number
        for j in range(4, len(r)):
            if j == tid_idx:
                continue
            v = r[j]
            if abs(v - round(v)) < 1e-3 and 0 <= v <= 80:
                cls = int(round(v))
                break

        out.append(
            {"track_id": tid, "bbox_xyxy": [x1, y1, x2, y2], "conf": conf, "cls": cls}
        )
    return out


def make_boxmot_tracker(tracker_name: str, reid_weights: str, device: str, fp16: bool, per_class: bool):
    """
    Robust-ish constructor across BoxMOT versions.
    """
    import inspect

    try:
        import boxmot  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import boxmot: {e}")

    cls = None
    # common exports: DeepOCSORT, StrongSORT
    if tracker_name.lower() == "deepocsort":
        cls = getattr(boxmot, "DeepOCSORT", None) or getattr(boxmot, "DeepOcSort", None)
    elif tracker_name.lower() == "strongsort":
        cls = getattr(boxmot, "StrongSORT", None) or getattr(boxmot, "StrongSort", None)

    if cls is None:
        raise RuntimeError(f"BoxMOT tracker class not found for '{tracker_name}'. dir(boxmot)={dir(boxmot)[:50]}...")

    # Build kwargs by inspecting signature
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name in sig.parameters.keys():
        if name in ("self",):
            continue
        if name in ("model_weights", "reid_weights", "weights", "model"):
            kwargs[name] = reid_weights
        elif name in ("device", "dev"):
            kwargs[name] = device
        elif name in ("half", "fp16"):
            kwargs[name] = fp16
        elif name in ("per_class",):
            kwargs[name] = per_class

    try:
        return cls(**kwargs)
    except TypeError:
        # Last resort: try positional (weights, device)
        try:
            return cls(reid_weights, device)
        except Exception as e:
            raise RuntimeError(f"Failed to construct {cls} with kwargs={kwargs}: {e}")


def write_jsonl(fp, rec: Dict[str, Any]):
    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_xyxy(b, w, h, r):
    x1, y1, x2, y2 = b
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw2 = bw * (1.0 + r)
    bh2 = bh * (1.0 + r)
    nx1 = clamp(int(cx - bw2 * 0.5), 0, w - 1)
    ny1 = clamp(int(cy - bh2 * 0.5), 0, h - 1)
    nx2 = clamp(int(cx + bw2 * 0.5), 0, w - 1)
    ny2 = clamp(int(cy + bh2 * 0.5), 0, h - 1)
    if nx2 <= nx1: nx2 = min(w - 1, nx1 + 1)
    if ny2 <= ny1: ny2 = min(h - 1, ny1 + 1)
    return [nx1, ny1, nx2, ny2]

def kp_bbox_xyxy(kxy, kcf, thr, pad=6):
    # kxy: (K,2), kcf: (K,)
    xs, ys = [], []
    for (x, y), c in zip(kxy, kcf):
        if c >= thr:
            xs.append(float(x)); ys.append(float(y))
    if len(xs) == 0:
        return None
    x1 = min(xs) - pad; y1 = min(ys) - pad
    x2 = max(xs) + pad; y2 = max(ys) + pad
    return [x1, y1, x2, y2]

def count_good_kps(kcf, thr):
    return int(sum(float(c) >= thr for c in kcf))

def run_pose_on_crop(pose_model, crop_bgr, imgsz, device, conf):
    # ultralytics YOLO pose
    results = pose_model.predict(crop_bgr, imgsz=imgsz, device=device, conf=conf, verbose=False)
    if not results:
        return None
    r = results[0]
    kps = getattr(r, "keypoints", None)
    if kps is None or kps.xy is None:
        return None
    # 여러 사람 나오면 "kp 평균 conf" 제일 높은 것 선택
    xy = kps.xy.cpu().numpy()      # (N,K,2)
    cf = kps.conf.cpu().numpy()    # (N,K)
    if len(xy) == 0:
        return None
    best = max(range(len(xy)), key=lambda i: float(cf[i].mean()))
    return xy[best], cf[best]

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_paths = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    # ---- local helpers (self-contained: paste-only main) ----
    from typing import Dict, Tuple, Optional
    import numpy as np

    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def _expand_xyxy(b: list[float], W: int, H: int, r: float) -> list[int]:
        x1, y1, x2, y2 = b
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw2 = bw * (1.0 + r)
        bh2 = bh * (1.0 + r)
        nx1 = _clamp(int(cx - 0.5 * bw2), 0, W - 1)
        ny1 = _clamp(int(cy - 0.5 * bh2), 0, H - 1)
        nx2 = _clamp(int(cx + 0.5 * bw2), 0, W - 1)
        ny2 = _clamp(int(cy + 0.5 * bh2), 0, H - 1)
        if nx2 <= nx1:
            nx2 = min(W - 1, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(H - 1, ny1 + 1)
        return [nx1, ny1, nx2, ny2]

    def _count_good_kps(kcf: np.ndarray, thr: float) -> int:
        # kcf: (K,)
        return int(np.sum(kcf >= thr))

    def _kp_bbox_xyxy(kxy: np.ndarray, kcf: np.ndarray, thr: float, pad: int = 8) -> Optional[list[float]]:
        # kxy: (K,2), kcf: (K,)
        m = kcf >= thr
        if not np.any(m):
            return None
        xs = kxy[m, 0].astype(np.float32)
        ys = kxy[m, 1].astype(np.float32)
        x1 = float(xs.min() - pad)
        y1 = float(ys.min() - pad)
        x2 = float(xs.max() + pad)
        y2 = float(ys.max() + pad)
        return [x1, y1, x2, y2]

    def _run_pose_best(pose_model, img_bgr, imgsz: int, device, conf: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # returns (kxy(K,2), kcf(K,))
        res = pose_model.predict(
            source=img_bgr,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        r0 = res[0] if res else None
        if r0 is None or getattr(r0, "keypoints", None) is None:
            return None
        kpts = r0.keypoints
        if getattr(kpts, "xy", None) is None or getattr(kpts, "conf", None) is None:
            return None
        xy = kpts.xy.detach().cpu().numpy()    # (N,K,2)
        cf = kpts.conf.detach().cpu().numpy()  # (N,K)
        if xy is None or cf is None or len(xy) == 0:
            return None
        # choose best person by mean kp conf
        best_i = int(np.argmax(cf.mean(axis=1)))
        return xy[best_i], cf[best_i]

    # Fallback knobs: works even if parser doesn't have these args (getattr-safe)
    occl_pose_fallback = bool(getattr(args, "occl_pose_fallback", True))  # default ON if arg missing
    occl_max_sec = float(getattr(args, "occl_max_sec", 0.6))
    occl_expand = float(getattr(args, "occl_expand", 0.30))
    occl_pose_conf = float(getattr(args, "occl_pose_conf", max(0.05, min(0.25, float(args.conf) * 0.5))))
    occl_kp_thr = float(getattr(args, "occl_kp_thr", float(getattr(args, "kp_thr", 0.25))))
    occl_min_kps = int(getattr(args, "occl_min_kps", 4))
    occl_max_tracks = int(getattr(args, "occl_max_tracks", 4))
    # --------------------------------------------------------

    # Imports
    cv2 = None
    ultralytics = None
    torch = None
    boxmot = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
    except Exception as e:
        logger.exception("Failed to import cv2: %s", e)
        return 2

    try:
        import ultralytics as _ultralytics
        from ultralytics import YOLO as _YOLO
        ultralytics = _ultralytics
    except Exception as e:
        logger.exception("Failed to import ultralytics: %s", e)
        return 2

    try:
        import torch as _torch
        torch = _torch
    except Exception:
        torch = None

    try:
        import boxmot as _boxmot
        boxmot = _boxmot
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

    # Init models
    logger.info("Loading pose model: %s", args.pose_model)
    pose_model = _YOLO(args.pose_model)

    logger.info("Creating BoxMOT tracker: %s (reid=%s)", args.tracker, args.reid_model)
    tracker = make_boxmot_tracker(
        tracker_name=args.tracker,
        reid_weights=args.reid_model,
        device=f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device),
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

    out_mp4 = run_paths.out_dir / f"{args.out_base}_tracked_pose.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = in_fps if in_fps > 0 else 30.0
    vw = cv2.VideoWriter(str(out_mp4), fourcc, out_fps, (w, h))
    if not vw.isOpened():
        logger.error("Failed to open VideoWriter: %s", out_mp4)
        return 2

    events_fp = None
    if args.save_jsonl:
        events_fp = (run_paths.out_dir / "tracks_pose.jsonl").open("w", encoding="utf-8")
        logger.info("Saving jsonl: %s", run_paths.out_dir / "tracks_pose.jsonl")

    versions = get_versions(cv2, ultralytics, torch, boxmot)
    params = vars(args).copy()
    params["source"] = src.source
    params["video_id"] = src.video_id
    params["is_rtsp"] = src.is_rtsp
    params["in_fps"] = in_fps
    params["width"] = w
    params["height"] = h
    params["versions"] = versions
    params["occl_pose_fallback"] = occl_pose_fallback
    params["occl_max_sec"] = occl_max_sec
    params["occl_expand"] = occl_expand
    params["occl_pose_conf"] = occl_pose_conf
    params["occl_kp_thr"] = occl_kp_thr
    params["occl_min_kps"] = occl_min_kps
    params["occl_max_tracks"] = occl_max_tracks
    (run_paths.out_dir / "params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Input: %s (%dx%d fps=%.3f)", src.source, w, h, in_fps)
    logger.info("Output video: %s", out_mp4)
    logger.info(
        "OcclPoseFallback: enabled=%s max_sec=%.2f expand=%.2f pose_conf=%.2f kp_thr=%.2f min_kps=%d max_tracks=%d",
        occl_pose_fallback, occl_max_sec, occl_expand, occl_pose_conf, occl_kp_thr, occl_min_kps, occl_max_tracks
    )

    frame_idx = -1
    t0 = time.time()

    # memory for occlusion fallback
    # tid -> {"last_bbox": [..], "last_ts": float, "last_kxy": np(K,2), "last_kcf": np(K), "last_kts": float}
    occl_mem: Dict[int, dict] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            # timestamp
            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if in_fps > 0:
                ts_sec = frame_idx / in_fps
            elif pos_ms > 0:
                ts_sec = pos_ms / 1000.0
            else:
                ts_sec = 0.0

            # Pose inference (full-frame)
            res = pose_model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            r0 = res[0] if res else None

            dets = []
            kpts_xy = None
            kpts_conf = None

            if r0 is not None and getattr(r0, "boxes", None) is not None:
                boxes = r0.boxes
                xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
                confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else []
                cls = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None

                if getattr(r0, "keypoints", None) is not None:
                    kpts_xy = r0.keypoints.xy.detach().cpu().numpy()    # (N,17,2)
                    kpts_conf = r0.keypoints.conf.detach().cpu().numpy()  # (N,17)

                keep_cls = set(int(x) for x in (args.classes or []))

                for i in range(len(xyxy)):
                    c = int(cls[i]) if cls is not None else 0
                    if keep_cls and c not in keep_cls:
                        continue
                    sc = float(confs[i]) if i < len(confs) else 0.0
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                    dets.append([x1, y1, x2, y2, sc, float(c)])

            # BoxMOT tracking
            det_arr = np.asarray(dets, dtype=np.float32) if len(dets) else np.zeros((0, 6), dtype=np.float32)
            try:
                tracks_out = tracker.update(det_arr, frame)
            except TypeError:
                tracks_out = tracker.update(dets=det_arr, img=frame)
            tracks = parse_tracker_outputs(tracks_out)

            # Match track -> det index for keypoints (IoU)
            track_to_det: Dict[int, int] = {}
            if len(tracks) and len(dets):
                for tr in tracks:
                    tid = int(tr["track_id"])
                    tb = tr["bbox_xyxy"]
                    best_iou, best_j = 0.0, -1
                    for j, dj in enumerate(dets):
                        db = dj[:4]
                        v = iou_xyxy(tb, db)
                        if v > best_iou:
                            best_iou, best_j = v, j
                    if best_j >= 0 and best_iou >= float(args.match_iou):
                        track_to_det[tid] = best_j

            # Update occl memory for seen tracks
            seen_tids = set()
            for tr in tracks:
                tid = int(tr["track_id"])
                seen_tids.add(tid)
                st = occl_mem.get(tid, {})
                st["last_bbox"] = [float(v) for v in tr["bbox_xyxy"]]
                st["last_ts"] = float(ts_sec)

                det_idx = track_to_det.get(tid, None)
                if det_idx is not None and kpts_xy is not None and kpts_conf is not None:
                    st["last_kxy"] = kpts_xy[det_idx].copy()
                    st["last_kcf"] = kpts_conf[det_idx].copy()
                    st["last_kts"] = float(ts_sec)
                occl_mem[tid] = st

            # Occlusion fallback: missing tracks -> crop pose -> draw hold
            fallback: Dict[int, dict] = {}
            if occl_pose_fallback and len(occl_mem):
                missing = [tid for tid in occl_mem.keys() if tid not in seen_tids]
                missing = sorted(missing, key=lambda t: float(occl_mem[t].get("last_ts", -1e9)), reverse=True)
                missing = missing[: max(0, occl_max_tracks)]

                for tid in missing:
                    st = occl_mem.get(tid, {})
                    last_ts = float(st.get("last_ts", -1e9))
                    if ts_sec - last_ts > occl_max_sec:
                        continue
                    last_bbox = st.get("last_bbox", None)
                    if last_bbox is None:
                        continue

                    x1, y1, x2, y2 = _expand_xyxy(last_bbox, w, h, occl_expand)
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue

                    out = _run_pose_best(pose_model, crop, imgsz=args.imgsz, device=args.device, conf=occl_pose_conf)
                    if out is None:
                        continue
                    kxy, kcf = out  # crop coords
                    kxy = kxy.astype(np.float32)
                    kcf = kcf.astype(np.float32)

                    # shift to full-frame coords
                    kxy[:, 0] += float(x1)
                    kxy[:, 1] += float(y1)

                    if _count_good_kps(kcf, occl_kp_thr) < occl_min_kps:
                        continue

                    kb = _kp_bbox_xyxy(kxy, kcf, thr=occl_kp_thr, pad=10)
                    if kb is None:
                        kb = [float(x1), float(y1), float(x2), float(y2)]

                    # refresh memory so we can bridge multiple frames
                    st["last_bbox"] = [float(v) for v in kb]
                    st["last_ts"] = float(ts_sec)
                    st["last_kxy"] = kxy.copy()
                    st["last_kcf"] = kcf.copy()
                    st["last_kts"] = float(ts_sec)
                    occl_mem[tid] = st

                    fallback[tid] = {"bbox": kb, "kxy": kxy, "kcf": kcf, "age": float(ts_sec - last_ts)}

            # Draw overlay
            vis = frame.copy()

            # draw tracked
            for tr in tracks:
                tid = int(tr["track_id"])
                x1, y1, x2, y2 = [int(v) for v in tr["bbox_xyxy"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 220, 60), 2)
                cv2.putText(
                    vis, f"id={tid}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
                )

                det_idx = track_to_det.get(tid, None)
                if det_idx is None or kpts_xy is None or kpts_conf is None:
                    continue

                kxy = kpts_xy[det_idx]    # (17,2)
                kcf = kpts_conf[det_idx]  # (17,)

                if args.draw_skeleton or args.draw_kp:
                    draw_skeleton_cv2(
                        cv2, vis, kxy, kcf, float(args.kp_thr),
                        draw_edges=bool(args.draw_skeleton),
                        draw_points=bool(args.draw_kp),
                    )

                if args.draw_anchor in ("ankle", "auto"):
                    draw_both_ankles_cv2(cv2, vis, kxy, kcf, float(args.anchor_thr))

                anchor_pt, anchor_tag = pick_anchor(kxy, kcf, args.draw_anchor, float(args.anchor_thr))
                skip_anchor_draw = anchor_tag == "ANKLE" and args.draw_anchor in ("ankle", "auto")
                if anchor_pt is not None and not skip_anchor_draw:
                    ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
                    cv2.circle(vis, (ax, ay), 6, (0, 165, 255), -1)
                    cv2.putText(
                        vis, f"{anchor_tag}", (ax + 6, ay - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
                    )

                if events_fp is not None:
                    rec = {
                        "frame_idx": frame_idx,
                        "ts_sec": ts_sec,
                        "track_id": tid,
                        "bbox_xyxy": [float(v) for v in tr["bbox_xyxy"]],
                        "det_idx": int(det_idx),
                        "occl_fallback": False,
                        "anchor_tag": anchor_tag,
                        "anchor_xy": [float(anchor_pt[0]), float(anchor_pt[1])] if anchor_pt is not None else None,
                        "kpts_xy": kxy.tolist(),
                        "kpts_conf": [float(x) for x in kcf.tolist()],
                    }
                    write_jsonl(events_fp, rec)

            # draw occl fallback (orange thin)
            for tid, fb in fallback.items():
                kb = fb["bbox"]
                x1, y1, x2, y2 = [int(v) for v in kb]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 1)
                cv2.putText(
                    vis, f"id={tid} OCCL {fb['age']:.1f}s",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
                )

                kxy = fb["kxy"]
                kcf = fb["kcf"]
                if args.draw_skeleton or args.draw_kp:
                    draw_skeleton_cv2(
                        cv2, vis, kxy, kcf, float(args.kp_thr),
                        draw_edges=bool(args.draw_skeleton),
                        draw_points=bool(args.draw_kp),
                    )

                if args.draw_anchor in ("ankle", "auto"):
                    draw_both_ankles_cv2(cv2, vis, kxy, kcf, float(args.anchor_thr))

                anchor_pt, anchor_tag = pick_anchor(kxy, kcf, args.draw_anchor, float(args.anchor_thr))
                skip_anchor_draw = anchor_tag == "ANKLE" and args.draw_anchor in ("ankle", "auto")
                if anchor_pt is not None and not skip_anchor_draw:
                    ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
                    cv2.circle(vis, (ax, ay), 6, (0, 165, 255), -1)
                    cv2.putText(
                        vis, f"{anchor_tag}", (ax + 6, ay - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
                    )

                if events_fp is not None:
                    rec = {
                        "frame_idx": frame_idx,
                        "ts_sec": ts_sec,
                        "track_id": int(tid),
                        "bbox_xyxy": [float(v) for v in kb],
                        "det_idx": None,
                        "occl_fallback": True,
                        "anchor_tag": anchor_tag,
                        "anchor_xy": [float(anchor_pt[0]), float(anchor_pt[1])] if anchor_pt is not None else None,
                        "kpts_xy": kxy.tolist(),
                        "kpts_conf": [float(x) for x in kcf.tolist()],
                    }
                    write_jsonl(events_fp, rec)

            # HUD
            elapsed = max(1e-6, time.time() - t0)
            fps = (frame_idx + 1) / elapsed
            cv2.putText(
                vis, f"frame={frame_idx} fps={fps:.1f} tracks={len(tracks)} dets={len(dets)} fb={len(fallback)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            vw.write(vis)

            if (frame_idx % 30) == 0:
                logger.info(
                    "progress frame=%d fps=%.2f tracks=%d dets=%d fb=%d",
                    frame_idx, fps, len(tracks), len(dets), len(fallback)
                )

            if not args.no_show:
                cv2.imshow("AID BoxMOT Pose", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        cap.release()
        vw.release()
        if events_fp is not None:
            events_fp.close()
        if not args.no_show:
            cv2.destroyAllWindows()

    logger.info("Done. saved=%s", out_mp4)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
