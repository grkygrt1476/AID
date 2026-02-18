#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidlib import run_utils

STAGE = "02_intrusion"
STATE_OUT = "OUT"
STATE_CAND = "CAND"
STATE_IN = "IN"
EVIDENCE_TYPE = "ankle"

# COCO keypoint indices
KP_LANKLE = 15
KP_RANKLE = 16


@dataclass
class SourceInfo:
    source: str
    video_id: str


@dataclass
class ROIConfig:
    path: Path
    video_id: str
    roi_id: str
    roi_version: int
    image_width: int
    image_height: int
    vertices_orig: list[list[float]]


@dataclass
class ZoneFSM:
    state: str = STATE_OUT
    candidate_start_frame: Optional[int] = None
    candidate_start_ts: Optional[float] = None
    dwell_frames: int = 0
    enter_streak: int = 0
    exit_streak: int = 0
    missing_streak: int = 0
    current_event: Optional[dict[str, Any]] = None


def build_parser():
    p = run_utils.common_argparser()
    p.add_argument("--source", default="", help="Input mp4 path; if empty, use --video_id.")
    p.add_argument("--video_id", default="", help="Video id like E01_007 -> data/videos/E01_007.mp4")
    p.add_argument("--roi_json", default="", help="ROI polygon json path. If empty, auto-resolve from configs/rois/<video_id>/")
    p.add_argument("--pose_model", default="yolo11s-pose.pt", help="Ultralytics pose model path/name.")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--det_conf", type=float, default=0.25)
    p.add_argument("--kp_conf", type=float, default=0.35, help="Min ankle keypoint confidence.")
    p.add_argument("--device", default="0", help="Ultralytics device (e.g. 0, cpu)")
    p.add_argument("--enter_n", type=int, default=3)
    p.add_argument("--exit_n", type=int, default=5)
    p.add_argument("--dwell_s", type=float, default=1.0)
    p.add_argument("--grace_s", type=float, default=2.0)
    p.add_argument("--max_frames", type=int, default=0, help="0 = full video")
    p.add_argument("--draw_ankles", action="store_true", default=True)
    p.add_argument("--no_show", action="store_true", default=True, help="Do not preview window (default on)")
    p.add_argument("--show", dest="no_show", action="store_false", help="Enable preview window")
    return p


def resolve_source(args) -> SourceInfo:
    source = str(args.source).strip()
    video_id = str(args.video_id).strip()
    if not source:
        if not video_id:
            raise ValueError("Either --source or --video_id must be provided.")
        source = str(Path("data/videos") / f"{video_id}.mp4")
    elif not video_id:
        stem = Path(source).stem
        m = re.search(r"(E\d{2}_\d{3})", stem)
        video_id = m.group(1) if m else stem
    return SourceInfo(source=source, video_id=video_id)


def _read_roi_obj(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read ROI json {path}: {e}")


def _parse_version_from_name(path: Path) -> int:
    m = re.search(r"_v(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def _extract_vertices_original(obj: dict, roi_path: Path) -> list[list[float]]:
    vertices = obj.get("vertices_px", [])
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError(f"Invalid vertices_px in {roi_path}")

    labeled_on = str(obj.get("labeled_on", "")).strip().lower()
    disp_scale_used = obj.get("disp_scale_used", None)
    if labeled_on == "disp" and disp_scale_used is not None:
        scale = float(disp_scale_used)
        if scale <= 0:
            raise ValueError(f"Invalid disp_scale_used in {roi_path}: {disp_scale_used}")
        return [[float(v[0]) / scale, float(v[1]) / scale] for v in vertices]
    return [[float(v[0]), float(v[1])] for v in vertices]


def load_roi_config(args, source_video_id: str) -> ROIConfig:
    roi_json = str(args.roi_json).strip()
    if roi_json:
        roi_path = Path(roi_json)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI json not found: {roi_path}")
        obj = _read_roi_obj(roi_path)
    else:
        if not source_video_id:
            raise ValueError("--roi_json is required when --video_id cannot be inferred.")
        roi_root = Path("configs/rois") / source_video_id
        if not roi_root.exists():
            raise FileNotFoundError(f"ROI folder not found: {roi_root}")
        candidates = sorted([p for p in roi_root.glob("*.json") if p.is_file()])
        if not candidates:
            raise FileNotFoundError(f"No ROI json found under: {roi_root}")
        scored: list[tuple[int, float, Path, dict]] = []
        for p in candidates:
            obj_try = _read_roi_obj(p)
            ver = int(obj_try.get("roi_version", _parse_version_from_name(p)))
            scored.append((ver, float(p.stat().st_mtime), p, obj_try))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, roi_path, obj = scored[0]

    img_size = obj.get("image_size", {}) or {}
    if isinstance(img_size, dict):
        iw = int(img_size.get("width", 0))
        ih = int(img_size.get("height", 0))
    elif isinstance(img_size, list) and len(img_size) >= 2:
        iw = int(img_size[0])
        ih = int(img_size[1])
    else:
        iw, ih = 0, 0
    if iw <= 0 or ih <= 0:
        raise ValueError(f"Invalid image_size in {roi_path}")

    return ROIConfig(
        path=roi_path,
        video_id=str(obj.get("video_id", "")).strip() or source_video_id,
        roi_id=str(obj.get("roi_id", "")).strip(),
        roi_version=int(obj.get("roi_version", _parse_version_from_name(roi_path))),
        image_width=iw,
        image_height=ih,
        vertices_orig=_extract_vertices_original(obj, roi_path),
    )


def scale_vertices_to_frame(vertices_orig: list[list[float]], src_w: int, src_h: int, dst_w: int, dst_h: int):
    sx = float(dst_w) / float(max(1, src_w))
    sy = float(dst_h) / float(max(1, src_h))
    out: list[tuple[float, float]] = []
    for x, y in vertices_orig:
        xx = max(0.0, min(float(dst_w - 1), float(x) * sx))
        yy = max(0.0, min(float(dst_h - 1), float(y) * sy))
        out.append((xx, yy))
    return out


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-6) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    # Ray-casting with explicit boundary handling.
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if _point_on_segment(px, py, ax, ay, bx, by):
            return True
        intersects = ((ay > py) != (by > py)) and (px < (bx - ax) * (py - ay) / ((by - ay) + 1e-12) + ax)
        if intersects:
            inside = not inside
    return inside


def extract_ankles(result, kp_conf_thr: float, frame_w: int, frame_h: int) -> list[dict[str, float]]:
    import numpy as np

    out: list[dict[str, float]] = []
    if result is None:
        return out
    kobj = getattr(result, "keypoints", None)
    if kobj is None or getattr(kobj, "xy", None) is None:
        return out

    xy_obj = kobj.xy
    xy = xy_obj.detach().cpu().numpy() if hasattr(xy_obj, "detach") else np.asarray(xy_obj)
    if xy.size == 0:
        return out

    cf = None
    if getattr(kobj, "conf", None) is not None:
        cf_obj = kobj.conf
        cf = cf_obj.detach().cpu().numpy() if hasattr(cf_obj, "detach") else np.asarray(cf_obj)

    for pi in range(len(xy)):
        for kp_idx in (KP_LANKLE, KP_RANKLE):
            if kp_idx >= len(xy[pi]):
                continue
            px = float(xy[pi][kp_idx][0])
            py = float(xy[pi][kp_idx][1])
            conf = 1.0
            if cf is not None and pi < len(cf) and kp_idx < len(cf[pi]):
                conf = float(cf[pi][kp_idx])
            if conf < float(kp_conf_thr):
                continue
            if not (math.isfinite(px) and math.isfinite(py)):
                continue
            px = max(0.0, min(float(frame_w - 1), px))
            py = max(0.0, min(float(frame_h - 1), py))
            out.append({"x": px, "y": py, "conf": conf})
    return out


def _reset_to_out(fsm: ZoneFSM):
    fsm.state = STATE_OUT
    fsm.candidate_start_frame = None
    fsm.candidate_start_ts = None
    fsm.dwell_frames = 0
    fsm.enter_streak = 0
    fsm.exit_streak = 0
    fsm.current_event = None


def _start_candidate(fsm: ZoneFSM, frame_idx: int, ts_sec: float):
    fsm.state = STATE_CAND
    fsm.candidate_start_frame = int(frame_idx)
    fsm.candidate_start_ts = float(ts_sec)
    fsm.dwell_frames = 1
    fsm.enter_streak = 1
    fsm.exit_streak = 0
    fsm.current_event = None


def _confirm_in(fsm: ZoneFSM, frame_idx: int, ts_sec: float, event_id: int):
    fsm.state = STATE_IN
    fsm.exit_streak = 0
    fsm.current_event = {
        "event_id": int(event_id),
        "evidence_type": EVIDENCE_TYPE,
        "enter_frame": int(fsm.candidate_start_frame if fsm.candidate_start_frame is not None else frame_idx),
        "enter_ts": float(fsm.candidate_start_ts if fsm.candidate_start_ts is not None else ts_sec),
        "confirm_frame": int(frame_idx),
        "confirm_ts": float(ts_sec),
        "exit_frame": None,
        "exit_ts": None,
        "duration_sec": None,
    }


def _finalize_in_event(fsm: ZoneFSM, frame_idx: int, ts_sec: float, events: list[dict[str, Any]]):
    if fsm.current_event is None:
        return
    ev = dict(fsm.current_event)
    ev["exit_frame"] = int(frame_idx)
    ev["exit_ts"] = float(ts_sec)
    ev["duration_sec"] = max(0.0, float(ev["exit_ts"]) - float(ev["confirm_ts"]))
    events.append(ev)
    _reset_to_out(fsm)


def update_zone_fsm(
    fsm: ZoneFSM,
    *,
    frame_idx: int,
    ts_sec: float,
    ankle_valid_any: bool,
    inside_any: bool,
    dwell_frames_req: int,
    grace_frames: int,
    enter_n: int,
    exit_n: int,
    events: list[dict[str, Any]],
):
    if ankle_valid_any:
        fsm.missing_streak = 0
    else:
        fsm.missing_streak += 1

    positive = bool(inside_any)
    missing_in_grace = fsm.missing_streak <= int(grace_frames)
    negative = (not bool(inside_any)) and (bool(ankle_valid_any) or (not missing_in_grace))

    if fsm.state == STATE_OUT:
        if positive:
            _start_candidate(fsm, frame_idx, ts_sec)
        return

    if fsm.state == STATE_CAND:
        if positive:
            fsm.dwell_frames += 1
            fsm.enter_streak += 1
            fsm.exit_streak = 0
            if fsm.enter_streak >= int(enter_n) and fsm.dwell_frames >= int(dwell_frames_req):
                _confirm_in(fsm, frame_idx, ts_sec, event_id=len(events) + 1)
        elif negative:
            fsm.exit_streak += 1
            if fsm.exit_streak >= int(exit_n):
                _reset_to_out(fsm)
        return

    if fsm.state == STATE_IN:
        if positive:
            fsm.exit_streak = 0
        elif negative:
            fsm.exit_streak += 1
            if fsm.exit_streak >= int(exit_n):
                _finalize_in_event(fsm, frame_idx, ts_sec, events)
        return


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        src = resolve_source(args)
    except ValueError as e:
        print(f"[ERROR] {e}")
        print(parser.format_usage().strip())
        return 2

    if not args.out_base:
        args.out_base = src.video_id if src.video_id else Path(src.source).stem

    run_paths = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except Exception as e:
        logger.exception("Missing dependency: %s", e)
        return 2

    in_path = Path(src.source)
    if not in_path.exists():
        logger.error("Input file not found: %s", in_path)
        return 2

    try:
        roi_cfg = load_roi_config(args, source_video_id=src.video_id)
    except Exception as e:
        logger.exception("Failed to load ROI: %s", e)
        return 2

    logger.info("Loading pose model: %s", args.pose_model)
    try:
        pose_model = YOLO(args.pose_model)
    except Exception as e:
        logger.exception("Failed to load pose model: %s", e)
        return 2

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        logger.error("Failed to open source: %s", in_path)
        return 2

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = in_fps if in_fps > 0 else 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        ok, fr = cap.read()
        if not ok or fr is None:
            logger.error("Failed to read first frame.")
            return 2
        frame_h, frame_w = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    roi_poly_f = scale_vertices_to_frame(
        vertices_orig=roi_cfg.vertices_orig,
        src_w=roi_cfg.image_width,
        src_h=roi_cfg.image_height,
        dst_w=frame_w,
        dst_h=frame_h,
    )
    roi_poly_i = np.asarray([[int(round(x)), int(round(y))] for x, y in roi_poly_f], dtype=np.int32)

    dwell_frames_req = max(1, int(math.ceil(float(args.dwell_s) * float(fps))))
    grace_frames = max(1, int(math.ceil(float(args.grace_s) * float(fps))))

    out_overlay = run_paths.out_dir / "overlay.mp4"
    out_events = run_paths.out_dir / "events.json"
    out_params = run_paths.out_dir / "params.json"

    vw = cv2.VideoWriter(
        str(out_overlay),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not vw.isOpened():
        logger.error("Failed to open VideoWriter: %s", out_overlay)
        return 2

    params = vars(args).copy()
    params.update(
        {
            "source": str(in_path),
            "video_id": src.video_id,
            "roi_path_resolved": str(roi_cfg.path),
            "roi_version": int(roi_cfg.roi_version),
            "frame_size": {"width": frame_w, "height": frame_h},
            "fps_used": float(fps),
            "evidence_type": EVIDENCE_TYPE,
            "dwell_frames_req": int(dwell_frames_req),
            "grace_frames": int(grace_frames),
            "fsm_states": [STATE_OUT, STATE_CAND, STATE_IN],
        }
    )
    out_params.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("Input: %s", in_path)
    logger.info("ROI: %s (id=%s v=%d)", roi_cfg.path, roi_cfg.roi_id, roi_cfg.roi_version)
    logger.info("Output overlay: %s", out_overlay)
    logger.info("Output events : %s", out_events)

    frame_idx = -1
    t0 = time.time()
    fsm = ZoneFSM()
    finished_events: list[dict[str, Any]] = []
    saw_keypoint_field = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            next_idx = frame_idx + 1
            if int(args.max_frames) > 0 and next_idx >= int(args.max_frames):
                break
            frame_idx = next_idx
            ts_sec = float(frame_idx) / float(fps)

            res = pose_model.predict(
                source=frame,
                imgsz=int(args.imgsz),
                conf=float(args.det_conf),
                classes=[0],
                device=str(args.device),
                verbose=False,
            )
            r0 = res[0] if res else None
            if r0 is not None and getattr(r0, "keypoints", None) is not None:
                saw_keypoint_field = True

            ankles = extract_ankles(
                result=r0,
                kp_conf_thr=float(args.kp_conf),
                frame_w=frame_w,
                frame_h=frame_h,
            )
            ankle_valid_any = len(ankles) > 0
            inside_any = False
            for a in ankles:
                inside = point_in_polygon(float(a["x"]), float(a["y"]), roi_poly_f)
                a["inside"] = 1.0 if inside else 0.0
                if inside:
                    inside_any = True

            update_zone_fsm(
                fsm,
                frame_idx=frame_idx,
                ts_sec=ts_sec,
                ankle_valid_any=ankle_valid_any,
                inside_any=inside_any,
                dwell_frames_req=dwell_frames_req,
                grace_frames=grace_frames,
                enter_n=int(args.enter_n),
                exit_n=int(args.exit_n),
                events=finished_events,
            )

            vis = frame.copy()
            cv2.polylines(vis, [roi_poly_i.reshape((-1, 1, 2))], isClosed=True, color=(80, 240, 80), thickness=2)

            if bool(args.draw_ankles):
                for a in ankles:
                    x = int(round(float(a["x"])))
                    y = int(round(float(a["y"])))
                    in_roi = bool(int(a.get("inside", 0)) == 1)
                    color = (0, 0, 255) if in_roi else (0, 255, 255)
                    cv2.circle(vis, (x, y), 4, color, -1)

            state_color = (60, 220, 60)
            if fsm.state == STATE_CAND:
                state_color = (0, 165, 255)
            elif fsm.state == STATE_IN:
                state_color = (0, 0, 255)
            cv2.putText(vis, f"STATE: {fsm.state}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"inside_any={int(bool(inside_any))} ankle_valid_any={int(bool(ankle_valid_any))}",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"evidence_type={EVIDENCE_TYPE} enter={fsm.enter_streak}/{int(args.enter_n)} exit={fsm.exit_streak}/{int(args.exit_n)}",
                (16, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"dwell_frames={fsm.dwell_frames}/{dwell_frames_req} missing={fsm.missing_streak}/{grace_frames}",
                (16, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )

            vw.write(vis)

            if not args.no_show:
                cv2.imshow("AID Baseline Intrusion", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if frame_idx % 30 == 0:
                elapsed = max(1e-6, time.time() - t0)
                fps_eff = float(frame_idx + 1) / elapsed
                logger.info(
                    "progress frame=%d fps_eff=%.2f state=%s inside_any=%d ankle_valid_any=%d events=%d",
                    frame_idx,
                    fps_eff,
                    fsm.state,
                    int(bool(inside_any)),
                    int(bool(ankle_valid_any)),
                    len(finished_events),
                )

    finally:
        cap.release()
        vw.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    if frame_idx >= 0 and fsm.state == STATE_IN and fsm.current_event is not None:
        ev = dict(fsm.current_event)
        ev["exit_frame"] = None
        ev["exit_ts"] = None
        ev["duration_sec"] = max(0.0, (float(frame_idx) / float(fps)) - float(ev["confirm_ts"]))
        ev["end_reason"] = "video_end"
        finished_events.append(ev)

    frames_processed = max(0, frame_idx + 1)
    elapsed = max(1e-6, time.time() - t0)
    fps_eff = float(frames_processed) / elapsed if frames_processed > 0 else 0.0

    if not saw_keypoint_field:
        logger.error("Model output did not include keypoints. Use a pose model (e.g., yolo11s-pose.pt).")
        return 2

    events_doc = {
        "meta": {
            "video_id": src.video_id,
            "source": str(in_path),
            "roi_path": str(roi_cfg.path),
            "roi_id": roi_cfg.roi_id,
            "roi_version": int(roi_cfg.roi_version),
            "evidence_type": EVIDENCE_TYPE,
            "states": [STATE_OUT, STATE_CAND, STATE_IN],
            "fps_in": float(in_fps),
            "fps_used": float(fps),
            "fps_eff": float(fps_eff),
            "frame_size": {"width": frame_w, "height": frame_h},
            "thresholds": {
                "enter_n": int(args.enter_n),
                "exit_n": int(args.exit_n),
                "dwell_s": float(args.dwell_s),
                "grace_s": float(args.grace_s),
                "dwell_frames_req": int(dwell_frames_req),
                "grace_frames": int(grace_frames),
                "kp_conf": float(args.kp_conf),
            },
        },
        "events": finished_events,
        "stats": {
            "frames_processed": int(frames_processed),
            "events_count": int(len(finished_events)),
            "overlay_mp4": str(out_overlay),
        },
    }
    out_events.write_text(json.dumps(events_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_meta = {
        "run_ts": run_paths.run_ts,
        "stage": STAGE,
        "script": run_paths.script_stem,
        "git_commit": run_utils.get_git_commit(),
        "git_dirty": run_utils.get_git_dirty(),
        "input": {"source": str(in_path), "video_id": src.video_id},
        "roi_path": str(roi_cfg.path),
        "roi_version": int(roi_cfg.roi_version),
        "fps_eff": float(fps_eff),
        "params": params,
        "outputs": {
            "overlay_mp4": str(out_overlay),
            "events_json": str(out_events),
            "params_json": str(out_params),
            "cmd_path": str(run_paths.cmd_path),
            "log_path": str(run_paths.log_path),
        },
    }
    run_utils.dump_run_meta(run_paths.out_dir, run_meta)

    logger.info("Done. overlay=%s", out_overlay)
    logger.info("Done. events=%s", out_events)
    logger.info("Done. run_meta=%s", run_paths.out_dir / "run_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
