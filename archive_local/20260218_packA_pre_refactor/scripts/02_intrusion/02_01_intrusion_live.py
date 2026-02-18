#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import platform
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aidlib import run_utils


STAGE = "02_intrusion"


@dataclass
class SourceInfo:
    source: str
    video_id: str
    is_rtsp: bool


@dataclass
class ROIConfig:
    path: Path
    vertices_px: list[list[float]]
    image_width: int
    image_height: int


def build_parser():
    parser = run_utils.common_argparser()

    parser.add_argument("--source", default="", help="mp4 path OR rtsp url; if empty, use --video_id")
    parser.add_argument("--video_id", default="", help="e.g., E01_009 -> data/videos/E01_009.mp4")
    parser.add_argument("--rtsp", action="store_true", help="force treat source as RTSP")
    parser.add_argument("--fps_sample", type=float, default=10.0, help="target processing fps via frame skipping")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--duration_sec", type=float, default=0.0, help="0 means no limit")

    parser.add_argument("--model", required=True, help="local YOLO weights path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="if empty: cuda if available else cpu")
    parser.add_argument("--tracker_cfg", default="bytetrack.yaml")
    parser.add_argument("--track_conf", type=float, default=0.15, help="low-confidence feed for tracker")
    parser.add_argument("--conf_enter", type=float, default=0.35)
    parser.add_argument("--min_h", type=float, default=40.0, help="fallback to center if bbox height < min_h")

    parser.add_argument("--t_enter", type=float, default=0.15, help="ioa enter threshold")
    parser.add_argument("--t_exit", type=float, default=0.05, help="ioa exit threshold")
    parser.add_argument("--dwell_sec", type=float, default=1.0)
    parser.add_argument("--enter_k", type=int, default=2)
    parser.add_argument("--enter_m", type=int, default=3)
    parser.add_argument("--exit_k", type=int, default=2)
    parser.add_argument("--exit_m", type=int, default=3)

    parser.add_argument("--exit_dilate", type=int, default=15, help="exit-mask dilation pixels")
    parser.add_argument("--roi_path", default="", help="optional ROI json override")

    parser.add_argument("--window_name", default="AID Intrusion Live")
    parser.add_argument("--realtime_sim", action="store_true")
    parser.add_argument("--jump_sec", type=float, default=5.0)
    parser.add_argument("--save_overlay_video", action="store_true", help="optional debug overlay mp4")
    return parser


def resolve_source(args) -> SourceInfo:
    source = args.source.strip()
    video_id = args.video_id.strip()
    if not source:
        if not video_id:
            raise ValueError("Either --source or --video_id must be provided.")
        source = str(Path("data/videos") / f"{video_id}.mp4")
    elif not video_id:
        src_path = Path(source)
        if src_path.suffix.lower() == ".mp4":
            video_id = src_path.stem
    is_rtsp = bool(args.rtsp) or source.lower().startswith("rtsp://")
    return SourceInfo(source=source, video_id=video_id, is_rtsp=is_rtsp)


def choose_device(arg_device: str, torch_module) -> str:
    if arg_device.strip():
        return arg_device.strip()
    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def get_versions(cv2_module, ultralytics_module, torch_module) -> dict[str, Optional[str]]:
    return {
        "python": platform.python_version(),
        "cv2": getattr(cv2_module, "__version__", None),
        "ultralytics": getattr(ultralytics_module, "__version__", None),
        "torch": getattr(torch_module, "__version__", None) if torch_module is not None else None,
    }


def _parse_roi_version(path: Path) -> int:
    m = re.search(r"_v(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def load_roi_config(video_id: str, roi_path_arg: str) -> ROIConfig:
    if roi_path_arg.strip():
        roi_path = Path(roi_path_arg.strip())
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI json not found: {roi_path}")
    else:
        if not video_id:
            raise ValueError("video_id is required for ROI auto-discovery when --roi_path is not provided.")
        roi_dir = Path("configs/rois") / video_id
        if not roi_dir.exists():
            raise FileNotFoundError(f"ROI directory not found: {roi_dir}")
        candidates = list(roi_dir.glob("roi_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No ROI json found under: {roi_dir}")
        roi_path = max(candidates, key=lambda p: (_parse_roi_version(p), p.stat().st_mtime))

    obj = json.loads(roi_path.read_text(encoding="utf-8"))
    vertices = obj.get("vertices_px", [])
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError(f"Invalid vertices_px in {roi_path}")

    img_size = obj.get("image_size", {}) or {}
    if isinstance(img_size, dict):
        iw = int(img_size.get("width", 0))
        ih = int(img_size.get("height", 0))
    elif isinstance(img_size, list) and len(img_size) >= 2:
        iw = int(img_size[0])
        ih = int(img_size[1])
    else:
        iw = 0
        ih = 0
    if iw <= 0 or ih <= 0:
        raise ValueError(f"Invalid image_size in {roi_path}")

    return ROIConfig(path=roi_path, vertices_px=vertices, image_width=iw, image_height=ih)


def scale_vertices(vertices_px: list[list[float]], src_w: int, src_h: int, dst_w: int, dst_h: int):
    sx = dst_w / float(max(1, src_w))
    sy = dst_h / float(max(1, src_h))
    pts = []
    for v in vertices_px:
        x = int(round(float(v[0]) * sx))
        y = int(round(float(v[1]) * sy))
        pts.append([max(0, min(dst_w - 1, x)), max(0, min(dst_h - 1, y))])
    return np.array(pts, dtype=np.int32)


def build_roi_and_exit_masks(frame_w: int, frame_h: int, roi_poly, exit_dilate: int):
    roi_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly], 1)

    if exit_dilate > 0:
        k = int(exit_dilate)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        exit_mask = cv2.dilate(roi_mask, kernel, iterations=1)
    else:
        exit_mask = roi_mask.copy()

    exit_contours, _ = cv2.findContours((exit_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return roi_mask, exit_mask, exit_contours


def clamp_point(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def point_in_mask(mask, x: int, y: int) -> int:
    h, w = mask.shape[:2]
    cx, cy = clamp_point(x, y, w, h)
    return int(mask[cy, cx] > 0)


def bbox_ioa_roi(mask, bbox_xyxy: list[float]) -> float:
    h, w = mask.shape[:2]
    x1 = max(0, min(w, int(round(bbox_xyxy[0]))))
    y1 = max(0, min(h, int(round(bbox_xyxy[1]))))
    x2 = max(0, min(w, int(round(bbox_xyxy[2]))))
    y2 = max(0, min(h, int(round(bbox_xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = float((x2 - x1) * (y2 - y1))
    inside = float(mask[y1:y2, x1:x2].sum())
    return inside / area if area > 0 else 0.0


def stable_k_of_m(hist: deque, k: int, m: int) -> bool:
    return len(hist) >= max(1, m) and sum(hist) >= max(1, k)


def init_track_fsm(ts_sec: float, enter_m: int, exit_m: int) -> dict[str, Any]:
    return {
        "state": "OUT",
        "dwell_sec": 0.0,
        "last_seen_ts": ts_sec,
        "enter_hist": deque(maxlen=max(1, enter_m)),
        "exit_hist": deque(maxlen=max(1, exit_m)),
    }


def maybe_transition_state(st: dict[str, Any], in_candidate: bool, out_candidate: bool, args, dt_sec: float) -> tuple[str, str]:
    prev_state = str(st["state"])

    st["enter_hist"].append(1 if in_candidate else 0)
    st["exit_hist"].append(1 if out_candidate else 0)
    enter_ok = stable_k_of_m(st["enter_hist"], args.enter_k, args.enter_m)
    exit_ok = stable_k_of_m(st["exit_hist"], args.exit_k, args.exit_m)

    if in_candidate:
        st["dwell_sec"] += max(0.0, dt_sec)
    else:
        st["dwell_sec"] = 0.0

    intrusion_confirmed = (st["dwell_sec"] >= args.dwell_sec) and enter_ok

    if prev_state == "OUT":
        if in_candidate:
            st["state"] = "IN_CANDIDATE"
    elif prev_state == "IN_CANDIDATE":
        if intrusion_confirmed:
            st["state"] = "IN_ACTIVE"
        elif out_candidate and exit_ok:
            st["state"] = "OUT"
    elif prev_state == "IN_ACTIVE":
        if out_candidate and exit_ok:
            st["state"] = "OUT"
            st["dwell_sec"] = 0.0

    return prev_state, str(st["state"])


def extract_person_tracks(result, logger: logging.Logger, cls_warned: list[bool]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return tracks

    xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
    confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []

    ids = None
    has_ids = getattr(boxes, "id", None) is not None
    if has_ids:
        ids = boxes.id.cpu().numpy()

    has_cls = getattr(boxes, "cls", None) is not None
    cls = boxes.cls.cpu().numpy() if has_cls else None
    if not has_cls and not cls_warned[0]:
        logger.warning("boxes.cls is not available; tracking all classes.")
        cls_warned[0] = True

    for i in range(len(xyxy)):
        if has_cls and int(cls[i]) != 0:
            continue
        tid = None
        if ids is not None and i < len(ids):
            raw = ids[i]
            if raw == raw:
                tid = int(raw)
        tracks.append(
            {
                "track_id": tid,
                "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                "conf": float(confs[i]) if i < len(confs) else None,
            }
        )
    if has_ids:
        tracks = [t for t in tracks if t["track_id"] is not None]
    return tracks


def seek_file_capture(cap, current_ts_sec: float, delta_sec: float, input_fps: float, total_frames: int, logger: logging.Logger) -> None:
    target_ts_sec = max(0.0, current_ts_sec + delta_sec)
    logger.info(
        "Seek request: current_ts_sec=%.3f delta_sec=%+.3f target_ts_sec=%.3f",
        current_ts_sec,
        delta_sec,
        target_ts_sec,
    )
    if input_fps > 0:
        target_frame = int(round(target_ts_sec * input_fps))
        if total_frames > 0:
            target_frame = min(max(target_frame, 0), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        logger.info("Seek via CAP_PROP_POS_FRAMES: target_frame=%d total_frames=%d", target_frame, total_frames)
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ts_sec * 1000.0)
        logger.info("Seek via CAP_PROP_POS_MSEC: target_ts_sec=%.3f", target_ts_sec)


def write_jsonl(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def handle_key(
    key: int,
    is_rtsp: bool,
    paused: bool,
    cap,
    model,
    yolo_cls,
    model_path: str,
    input_fps: float,
    total_frames: int,
    jump_sec: float,
    current_ts_sec: float,
    logger: logging.Logger,
    track_fsm: dict[int, dict[str, Any]],
):
    if key == ord("q"):
        return True, paused, model, False
    if key == ord(" "):
        paused = not paused
        logger.info("paused=%s", paused)
        return False, paused, model, False
    if key == ord("r"):
        logger.info("manual reset: reload model and clear intrusion state")
        model = yolo_cls(model_path)
        track_fsm.clear()
        return False, paused, model, True

    jump_map = {ord("d"): +jump_sec, ord("a"): -jump_sec}
    if key in jump_map:
        if is_rtsp:
            logger.info("Seek not supported on RTSP; jump ignored.")
            return False, paused, model, False
        delta = jump_map[key]
        seek_file_capture(cap, current_ts_sec, delta, input_fps, total_frames, logger)
        model = yolo_cls(model_path)
        track_fsm.clear()
        logger.info("tracker/model reset due to jump %+0.1fs", delta)
        return False, paused, model, True
    return False, paused, model, False


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_paths = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    cv2_module = None
    ultralytics_module = None
    torch_module = None
    yolo_cls = None
    try:
        import cv2 as _cv2
        import numpy as _np

        cv2_module = _cv2
        globals()["cv2"] = _cv2
        globals()["np"] = _np
    except Exception as e:
        logger.exception("Failed to import cv2/numpy: %s", e)
        return 2

    try:
        import ultralytics as _ultralytics
        from ultralytics import YOLO as _YOLO

        ultralytics_module = _ultralytics
        yolo_cls = _YOLO
    except Exception as e:
        logger.exception("Failed to import ultralytics/YOLO: %s", e)
        return 2

    try:
        import torch as _torch

        torch_module = _torch
    except Exception:
        torch_module = None

    versions = get_versions(cv2_module, ultralytics_module, torch_module)
    logger.info(
        "versions | python=%s cv2=%s ultralytics=%s torch=%s",
        versions["python"],
        versions["cv2"],
        versions["ultralytics"],
        versions["torch"],
    )

    try:
        source_info = resolve_source(args)
    except ValueError as e:
        logger.error(str(e))
        logger.error("Usage: %s", parser.format_usage().strip())
        return 2

    source = source_info.source
    is_rtsp = source_info.is_rtsp
    if not is_rtsp:
        src_path = Path(source)
        if not src_path.exists():
            logger.error("Input file not found: %s", src_path)
            return 2

    try:
        roi_cfg = load_roi_config(source_info.video_id, args.roi_path)
    except Exception as e:
        logger.exception("Failed to load ROI config: %s", e)
        return 2

    device = choose_device(args.device, torch_module)
    logger.info("device=%s", device)

    try:
        model = yolo_cls(args.model)
    except Exception as e:
        logger.exception("Failed to load YOLO model from %s: %s", args.model, e)
        return 2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open source: %s", source)
        return 2

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not is_rtsp else 0
    stride = max(1, int(round(input_fps / args.fps_sample))) if (input_fps > 0 and args.fps_sample > 0) else 1
    output_fps = (input_fps / stride) if input_fps > 0 else (args.fps_sample if args.fps_sample > 0 else 10.0)

    roi_scaled = scale_vertices(
        roi_cfg.vertices_px,
        roi_cfg.image_width,
        roi_cfg.image_height,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or roi_cfg.image_width),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or roi_cfg.image_height),
    )
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or roi_cfg.image_width)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or roi_cfg.image_height)
    roi_mask, exit_mask, exit_contours = build_roi_and_exit_masks(frame_w, frame_h, roi_scaled, args.exit_dilate)

    params = {
        "source": source,
        "video_id": source_info.video_id,
        "is_rtsp": is_rtsp,
        "roi_path": str(roi_cfg.path),
        "roi_vertices_px": roi_cfg.vertices_px,
        "roi_image_size": {"width": roi_cfg.image_width, "height": roi_cfg.image_height},
        "model": args.model,
        "imgsz": args.imgsz,
        "device": device,
        "tracker_cfg": args.tracker_cfg,
        "fps_sample": args.fps_sample,
        "stride": stride,
        "output_fps": output_fps,
        "track_conf": args.track_conf,
        "conf_enter": args.conf_enter,
        "min_h": args.min_h,
        "t_enter": args.t_enter,
        "t_exit": args.t_exit,
        "dwell_sec": args.dwell_sec,
        "enter_k": args.enter_k,
        "enter_m": args.enter_m,
        "exit_k": args.exit_k,
        "exit_m": args.exit_m,
        "exit_dilate": args.exit_dilate,
        "realtime_sim": bool(args.realtime_sim),
        "jump_sec": args.jump_sec,
        "max_frames": args.max_frames,
        "duration_sec": args.duration_sec,
        "save_overlay_video": bool(args.save_overlay_video),
        "versions": versions,
    }
    with (run_paths.out_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
        f.write("\n")

    events_path = run_paths.out_dir / "events.jsonl"
    events_fp = events_path.open("w", encoding="utf-8")
    logger.info("events: %s", events_path)

    writer = None
    writer_path = run_paths.out_dir / "overlay.mp4"
    stale_timeout_sec = max(2.0, args.dwell_sec + 1.0)

    frame_idx_processed = 0
    frame_idx_input = -1
    start_wall = time.time()
    paused = False
    should_quit = False
    last_overlay = None
    last_ts_sec = 0.0
    cls_warned = [False]
    track_fsm: dict[int, dict[str, Any]] = {}

    try:
        while not should_quit:
            loop_t0 = time.time()
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key != 255:
                    cur_ts_for_seek = (frame_idx_input / input_fps) if (input_fps > 0 and frame_idx_input >= 0) else last_ts_sec
                    should_quit, paused, model, clear_overlay = handle_key(
                        key=key,
                        is_rtsp=is_rtsp,
                        paused=paused,
                        cap=cap,
                        model=model,
                        yolo_cls=yolo_cls,
                        model_path=args.model,
                        input_fps=input_fps,
                        total_frames=total_frames,
                        jump_sec=args.jump_sec,
                        current_ts_sec=cur_ts_for_seek,
                        logger=logger,
                        track_fsm=track_fsm,
                    )
                    if clear_overlay:
                        last_overlay = None
                if last_overlay is not None:
                    cv2.imshow(args.window_name, last_overlay)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                logger.info("End of stream or frame read failure; stopping.")
                break

            cap_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
            if cap_frame_idx < 0:
                cap_frame_idx = frame_idx_input + 1
            frame_idx_input = cap_frame_idx
            if stride > 1 and (frame_idx_input % stride) != 0:
                continue

            raw_pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if input_fps > 0:
                ts_sec = frame_idx_input / input_fps
            elif raw_pos_ms > 0:
                ts_sec = raw_pos_ms / 1000.0
            elif output_fps > 0:
                ts_sec = frame_idx_processed / output_fps
            else:
                ts_sec = last_ts_sec
            last_ts_sec = ts_sec

            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=args.tracker_cfg,
                    conf=args.track_conf,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )
                result = results[0] if results else None
            except Exception as e:
                logger.exception("Tracking failed at frame_idx_input=%d: %s", frame_idx_input, e)
                break

            tracks = extract_person_tracks(result, logger=logger, cls_warned=cls_warned) if result is not None else []
            seen_ids: set[int] = set()
            overlay_rows: list[dict[str, Any]] = []

            for t in tracks:
                tid = t.get("track_id")
                if tid is None:
                    continue
                seen_ids.add(tid)
                x1, y1, x2, y2 = t["bbox_xyxy"]
                cx = int(round((x1 + x2) * 0.5))
                cy = int(round((y1 + y2) * 0.5))
                foot = (cx, int(round(y2)))
                center = (cx, cy)

                bbox_h = float(y2 - y1)
                foot_in_frame = (0 <= foot[0] < frame_w) and (0 <= foot[1] < frame_h)
                use_center = (not foot_in_frame) or (bbox_h < args.min_h)
                active = center if use_center else foot
                active_type = "center" if use_center else "foot"

                pip_foot = point_in_mask(roi_mask, foot[0], foot[1])
                pip_center = point_in_mask(roi_mask, center[0], center[1])
                active_in_roi = point_in_mask(roi_mask, active[0], active[1]) == 1
                active_in_exit = point_in_mask(exit_mask, active[0], active[1]) == 1
                ioa = bbox_ioa_roi(roi_mask, t["bbox_xyxy"])

                in_candidate = active_in_roi or (ioa >= args.t_enter)
                out_candidate = (not active_in_exit) and (ioa < args.t_exit)

                st = track_fsm.get(tid)
                if st is None:
                    st = init_track_fsm(ts_sec, args.enter_m, args.exit_m)
                dt_sec = max(0.0, ts_sec - float(st["last_seen_ts"]))
                prev_state, cur_state = maybe_transition_state(st, in_candidate, out_candidate, args, dt_sec)
                st["last_seen_ts"] = ts_sec
                track_fsm[tid] = st

                enter_votes = int(sum(st["enter_hist"]))
                exit_votes = int(sum(st["exit_hist"]))
                frame_rec = {
                    "event": "frame",
                    "ts_sec": ts_sec,
                    "frame_idx": frame_idx_input,
                    "track_id": tid,
                    "state": cur_state,
                    "ioa": ioa,
                    "pip_foot": pip_foot,
                    "pip_center": pip_center,
                    "active_point_type": active_type,
                    "dwell_sec": float(st["dwell_sec"]),
                    "enter_votes": {"sum": enter_votes, "len": len(st["enter_hist"])},
                    "exit_votes": {"sum": exit_votes, "len": len(st["exit_hist"])},
                }
                write_jsonl(events_fp, frame_rec)

                if prev_state != cur_state:
                    ev = "state_change"
                    if prev_state != "IN_ACTIVE" and cur_state == "IN_ACTIVE":
                        ev = "intrusion_start"
                    elif prev_state == "IN_ACTIVE" and cur_state != "IN_ACTIVE":
                        ev = "intrusion_end"
                    write_jsonl(
                        events_fp,
                        {
                            "event": ev,
                            "ts_sec": ts_sec,
                            "frame_idx": frame_idx_input,
                            "track_id": tid,
                            "prev_state": prev_state,
                            "state": cur_state,
                            "ioa": ioa,
                            "dwell_sec": float(st["dwell_sec"]),
                        },
                    )

                overlay_rows.append(
                    {
                        "track_id": tid,
                        "bbox_xyxy": t["bbox_xyxy"],
                        "conf": t.get("conf"),
                        "state": cur_state,
                        "ioa": ioa,
                        "dwell_sec": float(st["dwell_sec"]),
                        "foot": foot,
                        "center": center,
                        "active": active,
                        "active_type": active_type,
                    }
                )

            for tid in list(track_fsm.keys()):
                if tid in seen_ids:
                    continue
                st = track_fsm[tid]
                st["enter_hist"].append(0)
                st["exit_hist"].append(1)
                exit_ok = stable_k_of_m(st["exit_hist"], args.exit_k, args.exit_m)
                prev_state = str(st["state"])
                if prev_state in ("IN_CANDIDATE", "IN_ACTIVE") and exit_ok:
                    st["state"] = "OUT"
                    st["dwell_sec"] = 0.0
                    if prev_state == "IN_ACTIVE":
                        write_jsonl(
                            events_fp,
                            {
                                "event": "intrusion_end",
                                "ts_sec": ts_sec,
                                "frame_idx": frame_idx_input,
                                "track_id": tid,
                                "prev_state": prev_state,
                                "state": "OUT",
                                "ioa": None,
                                "dwell_sec": 0.0,
                            },
                        )
                if (ts_sec - float(st["last_seen_ts"])) > stale_timeout_sec:
                    track_fsm.pop(tid, None)

            disp = frame.copy()
            cv2.polylines(disp, [roi_scaled.reshape((-1, 1, 2))], isClosed=True, color=(80, 240, 80), thickness=2)
            if args.exit_dilate > 0:
                cv2.drawContours(disp, exit_contours, -1, (0, 170, 255), 1)

            active_count = 0
            for row in overlay_rows:
                x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
                state = row["state"]
                if state == "IN_ACTIVE":
                    color = (0, 0, 255)
                    active_count += 1
                elif state == "IN_CANDIDATE":
                    color = (0, 220, 255)
                else:
                    color = (60, 220, 60)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

                label = (
                    f"id={row['track_id']} {state} "
                    f"ioa={row['ioa']:.2f} dwell={row['dwell_sec']:.1f} {row['active_type']}"
                )
                cv2.putText(
                    disp,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                fx, fy = clamp_point(int(row["foot"][0]), int(row["foot"][1]), frame_w, frame_h)
                cx, cy = clamp_point(int(row["center"][0]), int(row["center"][1]), frame_w, frame_h)
                ax, ay = clamp_point(int(row["active"][0]), int(row["active"][1]), frame_w, frame_h)
                cv2.circle(disp, (fx, fy), 3, (255, 0, 0), -1)
                cv2.circle(disp, (cx, cy), 3, (255, 255, 0), -1)
                cv2.circle(disp, (ax, ay), 4, (0, 0, 255), -1)

            elapsed = max(1e-6, time.time() - start_wall)
            effective_fps = (frame_idx_processed + 1) / elapsed
            hud = [
                f"frame_idx_processed: {frame_idx_processed}",
                f"effective_fps: {effective_fps:.2f}",
                f"stride: {stride}",
                f"mode: {'RTSP' if is_rtsp else 'FILE'}",
                f"paused: {paused}",
                f"intrusion_active: {active_count}",
            ]
            y = 24
            for line in hud:
                cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y += 24

            if args.save_overlay_video and writer is None and not is_rtsp:
                writer = cv2.VideoWriter(
                    str(writer_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(output_fps if output_fps > 0 else 10.0),
                    (disp.shape[1], disp.shape[0]),
                )
                logger.info("overlay video enabled: %s", writer_path)
            if writer is not None:
                writer.write(disp)

            last_overlay = disp
            cv2.imshow(args.window_name, disp)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                cur_ts_for_seek = (frame_idx_input / input_fps) if (input_fps > 0 and frame_idx_input >= 0) else last_ts_sec
                should_quit, paused, model, clear_overlay = handle_key(
                    key=key,
                    is_rtsp=is_rtsp,
                    paused=paused,
                    cap=cap,
                    model=model,
                    yolo_cls=yolo_cls,
                    model_path=args.model,
                    input_fps=input_fps,
                    total_frames=total_frames,
                    jump_sec=args.jump_sec,
                    current_ts_sec=cur_ts_for_seek,
                    logger=logger,
                    track_fsm=track_fsm,
                )
                if clear_overlay:
                    last_overlay = None

            frame_idx_processed += 1

            if args.max_frames > 0 and frame_idx_processed >= args.max_frames:
                logger.info("Reached max_frames=%d", args.max_frames)
                break

            if args.duration_sec > 0 and output_fps > 0:
                proc_time_sec = frame_idx_processed / output_fps
                if proc_time_sec >= args.duration_sec:
                    logger.info("Reached duration_sec=%.3f", args.duration_sec)
                    break

            if args.realtime_sim and not is_rtsp and output_fps > 0:
                target_dt = 1.0 / output_fps
                loop_elapsed = time.time() - loop_t0
                to_sleep = max(0.0, target_dt - loop_elapsed)
                if to_sleep > 0:
                    time.sleep(to_sleep)

    finally:
        events_fp.close()
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
