#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aidlib import run_utils
from aidlib.intrusion import (
    FeatureConfig,
    FsmParams,
    RoiFsm,
    STATE_CAND,
    STATE_IN,
    STATE_OUT,
    append_jsonl,
    build_roi_cache,
    compute_bbox_factors,
    compute_score,
    create_video_writer,
    draw_roi_view,
    init_io,
    load_roi_polygon,
    load_yaml_config,
    save_yaml,
    write_json,
)
from aidlib.intrusion.score import ScoreWeights


STAGE = "02_fsm"


@dataclass
class RoiRuntime:
    roi_id: str
    source_path: str
    cache: Any
    fsm: RoiFsm


@dataclass
class LabelModeContext:
    label_json_path: str
    event_idx: int
    s: int
    e: int
    fps_src: float
    fps_clip: float
    clip_start_sec: float
    cand_window_frames: int


@dataclass
class ServiceFsmRuntime:
    state: str = STATE_OUT
    enter_cnt: int = 0
    in_cnt: int = 0
    exit_cnt: int = 0
    grace_cnt: int = 0
    event_index: int = 0
    active_event: Optional[dict[str, Any]] = None


@dataclass
class ServiceFsmContext:
    enabled: bool
    enter_n: int
    enter_in_n: int
    exit_n: int
    grace_n: int
    cand_thr: float
    in_thr: float
    exit_thr: float
    det_fps: float
    fps_src: float
    fps_clip: float
    clip_start_sec: float
    video_id: str
    clip_path: str
    params_for_event: dict[str, Any]
    runtimes: dict[str, ServiceFsmRuntime]
    events_fp: Any
    num_events: int = 0
    total_in_time_sec: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AID intrusion MVP scorer (ROI-level, no tracking).")
    parser.add_argument("--cfg", default="configs/intrusion/mvp_v1.yaml", help="YAML config path.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Run synthetic self-contained dry test.")
    parser.add_argument("--video", default="", help="Input video path for real mode.")
    parser.add_argument("--roi_json", nargs="+", default=[], help="One or more ROI json files for real mode.")
    parser.add_argument("--det_jsonl", default="", help="Optional detections jsonl for real mode.")
    parser.add_argument("--yolo_model", default="", help="Optional Ultralytics YOLO model path for real mode.")
    parser.add_argument("--device", default="0", help="Ultralytics device (e.g. 0, cpu).")
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = full length.")
    parser.add_argument("--det_fps", type=float, default=10.0, help="Detection update fps target for real mode.")
    parser.set_defaults(hold_det=True)
    parser.add_argument("--hold_det", dest="hold_det", action="store_true", help="Hold last detections between sampled updates.")
    parser.add_argument("--no_hold_det", dest="hold_det", action="store_false", help="Do not hold detections between sampled updates.")
    parser.add_argument("--label_mode", action="store_true", default=False, help="Overlay state from label json instead of score/FSM state.")
    parser.add_argument("--label_json", default="", help="Label json path used by --label_mode.")
    parser.add_argument("--label_event_idx", type=int, default=0, help="Event index in label json for --label_mode.")
    parser.add_argument("--clip_start_sec", type=float, default=None, help="Clip start time in source video seconds (label-mode mapping).")
    parser.add_argument("--cand_window_sec", type=float, default=0.0, help="Candidate window seconds around event boundaries in label mode.")
    parser.add_argument("--label_fps", type=float, default=None, help="Override source fps for label mapping; else metadata.fps or clip fps.")
    parser.add_argument("--band_ratio", type=float, default=0.20, help="Bottom band ratio for f_ov computation.")
    parser.add_argument("--w1", type=float, default=None, help="Override weight for f_dist (score = w1*f_dist + w2*f_ov - w3*p_gap).")
    parser.add_argument("--w2", type=float, default=None, help="Override weight for f_ov (score = w1*f_dist + w2*f_ov - w3*p_gap).")
    parser.add_argument("--w3", type=float, default=None, help="Override weight for p_gap penalty (score = w1*f_dist + w2*f_ov - w3*p_gap).")
    parser.add_argument("--ov_min_in", type=float, default=0.0, help="Minimum f_ov required for IN state (0.0 disables gate).")
    parser.add_argument("--draw_all_bbox", action="store_true", default=False, help="Draw all bboxes including OUT(gray).")
    parser.add_argument("--viz_rep_only", action="store_true", default=False, help="Draw only representative(best-score) bbox per frame.")
    parser.add_argument("--fsm_enable", action="store_true", default=False, help="Enable service-level FSM on update frames.")
    parser.add_argument("--enter_n", type=int, default=3, help="Consecutive cand-threshold frames required to enter CAND.")
    parser.add_argument("--enter_in_n", type=int, default=2, help="Consecutive in-gate frames required to confirm IN.")
    parser.add_argument("--exit_n", type=int, default=8, help="Consecutive low-score frames required to exit after grace.")
    parser.add_argument("--grace_sec", type=float, default=2.0, help="Grace duration in seconds before exit counting starts.")
    parser.add_argument("--exit_thr", type=float, default=None, help="Exit threshold (defaults to cand_thr when omitted).")
    parser.add_argument("--events_jsonl", default="", help="Path to events.jsonl (default: <out_dir>/events.jsonl).")
    parser.add_argument("--dry_frames", type=int, default=60, help="Number of frames for dry run.")
    parser.add_argument("--dry_fps", type=float, default=10.0, help="FPS for dry run.")
    parser.add_argument("--dry_wh", default="640x360", help="Dry frame size. Example: 640x360")
    parser.add_argument("--dry_rois", type=int, default=1, help="Number of synthetic ROIs for dry run.")
    parser.add_argument("--draw_out_bbox", action="store_true", default=False, help="Draw gray representative bbox in OUT.")

    parser.add_argument("--out_root", default="outputs", help="Output root.")
    parser.add_argument("--log_root", default="outputs/logs", help="Log root.")
    parser.add_argument("--out_base", default="intrusion_mvp", help="Output basename.")
    parser.add_argument("--run_ts", default="", help="Run timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    parser.add_argument("--codec", default="mp4v", help="VideoWriter codec (default: mp4v).")
    return parser


def _parse_wh(raw: str) -> tuple[int, int]:
    tokens = [t for t in raw.replace(",", "x").replace("X", "x").split("x") if t.strip()]
    if len(tokens) != 2:
        raise ValueError(f"Invalid size '{raw}'. Use WIDTHxHEIGHT, e.g. 640x360")
    w = int(tokens[0])
    h = int(tokens[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size '{raw}'. WIDTH/HEIGHT must be > 0")
    return w, h


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _effective_score_weights(args, base: ScoreWeights, logger: logging.Logger) -> tuple[ScoreWeights, bool]:
    wd = float(base.wd)
    wo = float(base.wo)
    wg = float(base.wg)
    override = False

    if args.w1 is not None:
        wd = float(args.w1)
        override = True
    if args.w2 is not None:
        wo = float(args.w2)
        override = True
    if args.w3 is not None:
        wg = float(args.w3)
        override = True

    out = ScoreWeights(wd=wd, wo=wo, wg=wg)
    if override:
        logger.info("Score weight override active: w1=%.6f w2=%.6f w3=%.6f", out.wd, out.wo, out.wg)
    return out, override


def _normalize_bbox(item: Any) -> Optional[list[float]]:
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        conf = float(item[4]) if len(item) >= 5 else 1.0
        return [float(item[0]), float(item[1]), float(item[2]), float(item[3]), conf]
    if isinstance(item, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(k in item for k in keys):
            conf = float(item.get("conf", 1.0))
            return [float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]), conf]
        if "bbox" in item:
            return _normalize_bbox(item["bbox"])
    return None


def _parse_bbox_list(raw: Any) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    out: list[list[float]] = []
    for item in raw:
        b = _normalize_bbox(item)
        if b is not None:
            out.append(b)
    return out


def load_det_jsonl(path: str | Path) -> dict[int, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"det_jsonl not found: '{p}'")
    frame_map: dict[int, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as fp:
        for ln, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON at {p}:{ln}: {exc}") from exc
            if not isinstance(row, dict) or "frame_idx" not in row:
                continue
            frame_idx = int(row["frame_idx"])
            slot = frame_map.setdefault(frame_idx, {"all": [], "by_roi": {}})

            by_roi_raw = row.get("rois", None)
            if isinstance(by_roi_raw, dict):
                for roi_id, items in by_roi_raw.items():
                    roi_key = str(roi_id)
                    slot["by_roi"].setdefault(roi_key, [])
                    slot["by_roi"][roi_key].extend(_parse_bbox_list(items))
            else:
                items = row.get("bboxes", row.get("detections", []))
                slot["all"].extend(_parse_bbox_list(items))
    return frame_map


def build_det_map_yolo(
    video_path,
    model_path,
    device,
    conf,
    imgsz,
    det_stride: int = 1,
    max_frames: int = 0,
) -> dict[int, dict[str, Any]]:
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for YOLO inference: {video_path}")

    model = YOLO(str(model_path))
    det_map: dict[int, dict[str, Any]] = {}
    frame_idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if int(max_frames) > 0 and frame_idx >= int(max_frames):
                break
            if int(det_stride) > 1 and (frame_idx % int(det_stride)) != 0:
                continue

            h, w = frame.shape[:2]
            bboxes: list[list[float]] = []
            results = model.predict(
                source=frame,
                classes=[0],
                conf=float(conf),
                imgsz=int(imgsz),
                device=str(device),
                verbose=False,
            )
            if results:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().tolist()
                    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [1.0] * len(xyxy)
                    for xy, c in zip(xyxy, confs):
                        if len(xy) < 4:
                            continue
                        x1 = _clamp(float(xy[0]), 0.0, float(max(0, w - 1)))
                        y1 = _clamp(float(xy[1]), 0.0, float(max(0, h - 1)))
                        x2 = _clamp(float(xy[2]), 0.0, float(max(0, w - 1)))
                        y2 = _clamp(float(xy[3]), 0.0, float(max(0, h - 1)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        bboxes.append([x1, y1, x2, y2, float(c)])
            det_map[frame_idx] = {"all": bboxes, "by_roi": {}, "bboxes": bboxes}
    finally:
        cap.release()

    return det_map


def _poly_bounds(poly) -> tuple[int, int, int, int]:
    x1 = int(poly[:, 0].min())
    y1 = int(poly[:, 1].min())
    x2 = int(poly[:, 0].max())
    y2 = int(poly[:, 1].max())
    return x1, y1, x2, y2


def _make_rect_poly(x1: int, y1: int, x2: int, y2: int):
    import numpy as np

    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def build_dry_rois(width: int, height: int, count: int) -> list[tuple[str, str, Any]]:
    import numpy as np

    n = max(1, int(count))
    centers = np.linspace(int(width * 0.25), int(width * 0.75), n)
    roi_w = max(100, int(width * 0.26))
    roi_h = max(120, int(height * 0.52))
    y2 = int(height * 0.90)
    y1 = max(0, y2 - roi_h)

    out: list[tuple[str, str, Any]] = []
    for i, cx in enumerate(centers, start=1):
        x1 = int(round(cx)) - roi_w // 2
        x2 = int(round(cx)) + roi_w // 2
        x1 = max(8, min(width - 8, x1))
        x2 = max(8, min(width - 8, x2))
        if x2 <= x1:
            x2 = min(width - 8, x1 + 10)
        poly = _make_rect_poly(x1, y1, x2, y2)
        roi_id = f"dry_roi_{i:02d}"
        source_path = f"__dry_run__/{roi_id}.json"
        cache = build_roi_cache(roi_id=roi_id, poly=poly, width=width, height=height)
        out.append((roi_id, source_path, cache))
    return out


def synth_bboxes_for_roi(frame_idx: int, nframes: int, roi_cache) -> list[list[float]]:
    x1, y1, x2, y2 = _poly_bounds(roi_cache.poly)
    roi_w = max(20, x2 - x1)
    roi_h = max(20, y2 - y1)
    box_w = max(28, int(roi_w * 0.28))
    box_h = max(48, int(roi_h * 0.68))
    # Keep the bottom-center farther from ROI boundary so f_dist can exceed in_thr region.
    y_bottom_inside = float(y2 - max(8, int(round(roi_h * 0.06))))

    phase1_end = max(2, int(round(nframes * 0.20)))
    phase2_end = max(phase1_end + 2, int(round(nframes * 0.42)))

    if frame_idx < phase1_end:
        center_x = float(x1 - int(1.5 * box_w))
        y_bottom = y_bottom_inside
    elif frame_idx < phase2_end:
        center_x = float(0.5 * (x1 + x2))
        y_bottom = y_bottom_inside
    else:
        center_x = float(x2 + int(1.5 * box_w))
        y_bottom = float(y2 - max(10, int(roi_h * 0.15)))

    main = [
        float(center_x - box_w / 2.0),
        float(y_bottom - box_h),
        float(center_x + box_w / 2.0),
        float(y_bottom),
        0.95,
    ]
    # Distractor is intentionally weak/non-overlapping.
    noise = [
        float(x1 + int(0.10 * roi_w)),
        float(max(0, y1 - int(0.35 * roi_h))),
        float(x1 + int(0.28 * roi_w)),
        float(max(1, y1 - int(0.08 * roi_h))),
        0.25,
    ]
    return [main, noise]


def _select_best_bbox(
    *,
    bboxes: list[list[float]],
    roi_cache,
    feature_cfg: FeatureConfig,
    score_weights: ScoreWeights,
    width: int,
    height: int,
) -> tuple[float, Optional[list[float]], Optional[dict[str, Any]]]:
    best_score = 0.0
    best_bbox: Optional[list[float]] = None
    best_factors: Optional[dict[str, Any]] = None

    for bbox in bboxes:
        try:
            factors = compute_bbox_factors(
                bbox=bbox,
                roi_cache=roi_cache,
                cfg_norms=feature_cfg,
                image_w=width,
                image_h=height,
            )
        except Exception:
            continue
        s = compute_score(factors, score_weights)
        if (best_bbox is None) or (s > best_score):
            best_score = float(s)
            best_bbox = list(factors["bbox_clamped"])
            best_factors = factors
    return float(best_score), best_bbox, best_factors


def _has_transition_sequence(states: list[str], target: list[str]) -> bool:
    if not states:
        return False
    t = 0
    for state in states:
        if state == target[t]:
            t += 1
            if t >= len(target):
                return True
    return False


def _state_counts(states: list[str]) -> dict[str, int]:
    out = {STATE_OUT: 0, STATE_CAND: 0, STATE_IN: 0}
    for s in states:
        if s in out:
            out[s] += 1
    return out


def _read_roi_id(path: Path, fallback: str) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rid = str(obj.get("roi_id", "")).strip()
        return rid if rid else fallback
    except Exception:
        return fallback


def _build_real_rois(roi_paths: list[str], width: int, height: int) -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    used: dict[str, int] = {}
    for i, roi_path in enumerate(roi_paths, start=1):
        p = Path(roi_path)
        if not p.exists():
            raise FileNotFoundError(f"ROI json not found: '{p}'")
        poly = load_roi_polygon(p)
        fallback_id = f"roi_{i:02d}"
        roi_id_raw = _read_roi_id(p, fallback=fallback_id)
        seq = used.get(roi_id_raw, 0) + 1
        used[roi_id_raw] = seq
        roi_id = roi_id_raw if seq == 1 else f"{roi_id_raw}_{seq}"
        cache = build_roi_cache(roi_id=roi_id, poly=poly, width=width, height=height)
        out.append((roi_id, str(p), cache))
    return out


def _extract_event_frames(obj: dict[str, Any]) -> list[tuple[int, int]]:
    ann = obj.get("annotations", {})
    raw = ann.get("event_frame", None) if isinstance(ann, dict) else None
    if raw is None:
        raw = obj.get("event_frame", None)
    out: list[tuple[int, int]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            s = int(item[0])
            e = int(item[1])
        except Exception:
            continue
        if e < s:
            s, e = e, s
        out.append((s, e))
    return out


def _infer_label_json_path(video_path: Path) -> Optional[Path]:
    p = video_path.resolve()
    raw = str(p)
    # Works for posix paths used in this repo.
    if "/clips/" in raw:
        left, right = raw.split("/clips/", 1)
        parts = Path(right).parts
        if parts:
            video_id = str(parts[0]).strip()
            if video_id:
                return Path(left) / "labels_user" / f"{video_id}.json"
    # Fallback guess: .../<root>/clips/<video_id>/<clip>.mp4
    if p.parent.name and p.parent.parent.exists():
        return p.parent.parent / "labels_user" / f"{p.parent.name}.json"
    return None


def _infer_clip_start_sec_from_name(video_path: Path, fps_src: float) -> Optional[float]:
    m = re.search(r"(?:event\d+_)?padded_(\d+)_(\d+)", video_path.name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        start_frame = int(m.group(1))
    except Exception:
        return None
    if fps_src <= 0:
        return None
    return float(start_frame) / float(fps_src)


def _build_label_mode_context(
    *,
    args,
    logger: logging.Logger,
    video_path: Path,
    fps_clip: float,
) -> Optional[LabelModeContext]:
    if not bool(args.label_mode):
        return None

    label_json_raw = str(args.label_json).strip()
    label_path: Optional[Path] = Path(label_json_raw) if label_json_raw else None
    if label_path is None:
        label_path = _infer_label_json_path(video_path)
        if label_path is None:
            logger.warning("label_mode on but label_json missing and not inferable; label_mode disabled.")
            return None
        logger.info("label_mode inferred label_json: %s", label_path)
    if not label_path.exists():
        logger.warning("label_mode on but label_json not found: %s; label_mode disabled.", label_path)
        return None

    try:
        obj = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("label_mode failed to parse label_json (%s): %s; label_mode disabled.", label_path, exc)
        return None
    if not isinstance(obj, dict):
        logger.warning("label_mode label_json is not an object: %s; label_mode disabled.", label_path)
        return None

    events = _extract_event_frames(obj)
    if not events:
        logger.warning("label_mode: no event_frame found in %s; label_mode disabled.", label_path)
        return None

    idx_raw = int(args.label_event_idx)
    idx = max(0, min(idx_raw, len(events) - 1))
    if idx != idx_raw:
        logger.warning("label_mode: label_event_idx=%d out of range, clamped to %d", idx_raw, idx)
    s, e = events[idx]

    fps_src = float(args.label_fps) if args.label_fps is not None else 0.0
    if fps_src <= 0:
        meta = obj.get("metadata", {})
        fps_meta = 0.0
        if isinstance(meta, dict) and meta.get("fps", None) is not None:
            try:
                fps_meta = float(meta.get("fps", 0.0))
            except Exception:
                fps_meta = 0.0
        fps_src = fps_meta if fps_meta > 0 else float(fps_clip if fps_clip > 0 else 30.0)
    if fps_src <= 0:
        fps_src = 30.0

    if args.clip_start_sec is not None:
        clip_start_sec = float(args.clip_start_sec)
    else:
        clip_start_inferred = _infer_clip_start_sec_from_name(video_path=video_path, fps_src=fps_src)
        if clip_start_inferred is None:
            clip_start_sec = 0.0
            logger.warning("label_mode: clip_start_sec could not be inferred; defaulting to 0.0")
        else:
            clip_start_sec = float(clip_start_inferred)

    cand_window_frames = max(0, int(round(float(args.cand_window_sec) * float(fps_src))))
    logger.info(
        "label_mode enabled: label_json=%s event_idx=%d event=[%d,%d] fps_src=%.6f clip_start_sec=%.6f cand_window_frames=%d",
        label_path,
        idx,
        s,
        e,
        fps_src,
        clip_start_sec,
        cand_window_frames,
    )
    return LabelModeContext(
        label_json_path=str(label_path),
        event_idx=idx,
        s=int(s),
        e=int(e),
        fps_src=float(fps_src),
        fps_clip=float(fps_clip if fps_clip > 0 else 30.0),
        clip_start_sec=float(clip_start_sec),
        cand_window_frames=int(cand_window_frames),
    )


def _label_state_for_frame(label_ctx: LabelModeContext, frame_idx: int) -> tuple[str, int]:
    fps_clip = float(label_ctx.fps_clip if label_ctx.fps_clip > 0 else 30.0)
    t_sec = float(frame_idx) / fps_clip
    orig_frame = int(round((float(t_sec) + float(label_ctx.clip_start_sec)) * float(label_ctx.fps_src)))
    s = int(label_ctx.s)
    e = int(label_ctx.e)
    if s <= orig_frame <= e:
        return STATE_IN, orig_frame
    c = int(label_ctx.cand_window_frames)
    if c > 0:
        if (s - c) <= orig_frame <= (s - 1):
            return STATE_CAND, orig_frame
        if (e + 1) <= orig_frame <= (e + c):
            return STATE_CAND, orig_frame
    return STATE_OUT, orig_frame


def _infer_video_id_from_clip(video_path: Path) -> str:
    raw = str(video_path.resolve())
    if "/clips/" in raw:
        right = raw.split("/clips/", 1)[1]
        parts = Path(right).parts
        if parts:
            video_id = str(parts[0]).strip()
            if video_id:
                return video_id
    if video_path.parent.name:
        return str(video_path.parent.name)
    return str(video_path.stem)


def _infer_clip_start_sec_generic(
    *,
    args,
    logger: logging.Logger,
    video_path: Path,
    fps_src: float,
    warn_prefix: str,
) -> float:
    if args.clip_start_sec is not None:
        return float(args.clip_start_sec)
    clip_start_inferred = _infer_clip_start_sec_from_name(video_path=video_path, fps_src=float(fps_src))
    if clip_start_inferred is None:
        logger.warning("%s: clip_start_sec could not be inferred; defaulting to 0.0", warn_prefix)
        return 0.0
    return float(clip_start_inferred)


def _service_event_peak_from_row(orig_frame: int, roi_score: float, row: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "orig_frame": int(orig_frame),
            "score": float(roi_score),
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "conf": 0.0,
            "f_dist": 0.0,
            "f_ov": 0.0,
            "p_gap": 1.0,
        }
    return {
        "orig_frame": int(orig_frame),
        "score": float(roi_score),
        "bbox": [
            float(row.get("x1", 0.0)),
            float(row.get("y1", 0.0)),
            float(row.get("x2", 0.0)),
            float(row.get("y2", 0.0)),
        ],
        "conf": float(row.get("conf", 0.0)),
        "f_dist": float(row.get("f_dist", 0.0)),
        "f_ov": float(row.get("f_ov", 0.0)),
        "p_gap": float(row.get("p_gap", 1.0)),
    }


def _service_event_update_peak(active_event: dict[str, Any], *, orig_frame: int, roi_score: float, row: Optional[dict[str, Any]]) -> None:
    prev = float(active_event.get("peak", {}).get("score", -1e9))
    if float(roi_score) >= prev:
        active_event["peak"] = _service_event_peak_from_row(orig_frame=int(orig_frame), roi_score=float(roi_score), row=row)


def _service_event_finalize(
    *,
    ctx: ServiceFsmContext,
    runtime: ServiceFsmRuntime,
    roi_id: str,
    end_orig_frame: int,
) -> Optional[dict[str, Any]]:
    if not isinstance(runtime.active_event, dict):
        return None
    start_orig_frame = int(runtime.active_event.get("start_orig_frame", end_orig_frame))
    enter_confirm_frame = int(runtime.active_event.get("enter_confirm_frame", start_orig_frame))
    exit_confirm_frame = int(end_orig_frame)
    duration_sec = 0.0
    if float(ctx.fps_src) > 0:
        duration_sec = max(0.0, float(end_orig_frame - start_orig_frame) / float(ctx.fps_src))
    idx = int(runtime.event_index)
    event_id = f"{ctx.video_id}_{roi_id}_{idx:04d}"
    payload = {
        "event_id": event_id,
        "video_id": str(ctx.video_id),
        "clip_path": str(ctx.clip_path),
        "roi_id": str(roi_id),
        "start_orig_frame": int(start_orig_frame),
        "end_orig_frame": int(end_orig_frame),
        "enter_confirm_frame": int(enter_confirm_frame),
        "exit_confirm_frame": int(exit_confirm_frame),
        "duration_sec": float(duration_sec),
        "peak": runtime.active_event.get("peak", _service_event_peak_from_row(orig_frame=end_orig_frame, roi_score=0.0, row=None)),
        "params": dict(ctx.params_for_event),
    }
    runtime.event_index += 1
    runtime.active_event = None
    ctx.num_events += 1
    ctx.total_in_time_sec += float(duration_sec)
    return payload


def _service_fsm_step(
    *,
    ctx: ServiceFsmContext,
    runtime: ServiceFsmRuntime,
    roi_id: str,
    roi_score: float,
    roi_in_any: bool,
    orig_frame: int,
    best_row: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    finalized_event: Optional[dict[str, Any]] = None

    if runtime.state == STATE_OUT:
        if float(roi_score) >= float(ctx.cand_thr):
            runtime.enter_cnt += 1
        else:
            runtime.enter_cnt = 0
        runtime.in_cnt = 0
        runtime.exit_cnt = 0
        runtime.grace_cnt = 0
        if runtime.enter_cnt >= int(ctx.enter_n):
            runtime.state = STATE_CAND
            runtime.enter_cnt = 0
            runtime.in_cnt = 0
            runtime.exit_cnt = 0
            runtime.grace_cnt = 0

    elif runtime.state == STATE_CAND:
        if bool(roi_in_any):
            runtime.in_cnt += 1
        else:
            runtime.in_cnt = 0
        if float(roi_score) < float(ctx.cand_thr):
            runtime.exit_cnt += 1
        else:
            runtime.exit_cnt = 0
        runtime.grace_cnt = 0
        if runtime.in_cnt >= int(ctx.enter_in_n):
            runtime.state = STATE_IN
            runtime.in_cnt = 0
            runtime.exit_cnt = 0
            runtime.grace_cnt = 0
            runtime.active_event = {
                "start_orig_frame": int(orig_frame),
                "enter_confirm_frame": int(orig_frame),
                "peak": _service_event_peak_from_row(orig_frame=int(orig_frame), roi_score=float(roi_score), row=best_row),
            }
        elif runtime.exit_cnt >= int(ctx.exit_n):
            runtime.state = STATE_OUT
            runtime.enter_cnt = 0
            runtime.in_cnt = 0
            runtime.exit_cnt = 0
            runtime.grace_cnt = 0
            runtime.active_event = None

    elif runtime.state == STATE_IN:
        if bool(roi_in_any):
            runtime.grace_cnt = 0
            runtime.exit_cnt = 0
        else:
            runtime.grace_cnt += 1
            if runtime.grace_cnt > int(ctx.grace_n):
                if float(roi_score) < float(ctx.exit_thr):
                    runtime.exit_cnt += 1
                else:
                    runtime.exit_cnt = 0
                if runtime.exit_cnt >= int(ctx.exit_n):
                    finalized_event = _service_event_finalize(
                        ctx=ctx,
                        runtime=runtime,
                        roi_id=roi_id,
                        end_orig_frame=int(orig_frame),
                    )
                    runtime.state = STATE_OUT
                    runtime.enter_cnt = 0
                    runtime.in_cnt = 0
                    runtime.exit_cnt = 0
                    runtime.grace_cnt = 0

    if runtime.state == STATE_IN:
        if runtime.active_event is None:
            runtime.active_event = {
                "start_orig_frame": int(orig_frame),
                "enter_confirm_frame": int(orig_frame),
                "peak": _service_event_peak_from_row(orig_frame=int(orig_frame), roi_score=float(roi_score), row=best_row),
            }
        _service_event_update_peak(runtime.active_event, orig_frame=int(orig_frame), roi_score=float(roi_score), row=best_row)

    return finalized_event


def _draw_out_bboxes_gray(frame, bbox_rows: list[dict[str, Any]]) -> None:
    import cv2

    h, w = frame.shape[:2]
    for row in bbox_rows:
        if str(row.get("state", "")) != STATE_OUT:
            continue
        x1 = int(round(float(row.get("x1", 0.0))))
        y1 = int(round(float(row.get("y1", 0.0))))
        x2 = int(round(float(row.get("x2", 0.0))))
        y2 = int(round(float(row.get("y2", 0.0))))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        conf = float(row.get("conf", 0.0))
        score = float(row.get("score", 0.0))
        c_text = f"c={conf:.2f}"
        s_text = f"s={score:.2f}"
        scale = 0.50
        th = 2
        pad = 4
        c_sz = cv2.getTextSize(c_text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0]
        s_sz = cv2.getTextSize(s_text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0]

        cx = max(0, min(w - c_sz[0] - 1, x2 - c_sz[0] - pad))
        cy = max(12, y1 + c_sz[1] + 2)
        cv2.rectangle(frame, (max(0, cx - 2), max(0, cy - c_sz[1] - 2)), (min(w - 1, cx + c_sz[0] + 2), min(h - 1, cy + 2)), (0, 0, 0), -1)
        cv2.putText(frame, c_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

        sx = max(0, min(w - s_sz[0] - 1, x2 - s_sz[0] - pad))
        sy = min(h - 6, y2 - 4)
        cv2.rectangle(frame, (max(0, sx - 2), max(0, sy - s_sz[1] - 2)), (min(w - 1, sx + s_sz[0] + 2), min(h - 1, sy + 2)), (0, 0, 0), -1)
        cv2.putText(frame, s_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def _build_service_fsm_context(
    *,
    args,
    rois: list[RoiRuntime],
    events_fp,
    cand_thr: float,
    in_thr: float,
    fps_src: float,
    fps_clip: float,
    clip_start_sec: float,
    video_id: str,
    clip_path: str,
    score_weights: ScoreWeights,
) -> ServiceFsmContext:
    exit_thr = float(args.exit_thr) if args.exit_thr is not None else float(cand_thr)
    det_fps = float(args.det_fps if float(args.det_fps) > 0 else 1.0)
    grace_n = max(0, int(round(det_fps * float(args.grace_sec))))
    runtimes = {rr.roi_id: ServiceFsmRuntime() for rr in rois}
    params_for_event = {
        "det_fps": float(det_fps),
        "enter_n": int(max(1, int(args.enter_n))),
        "enter_in_n": int(max(1, int(args.enter_in_n))),
        "exit_n": int(max(1, int(args.exit_n))),
        "grace_sec": float(args.grace_sec),
        "cand_thr": float(cand_thr),
        "in_thr": float(in_thr),
        "exit_thr": float(exit_thr),
        "w1": float(score_weights.wd),
        "w2": float(score_weights.wo),
        "w3": float(score_weights.wg),
        "band_ratio": float(args.band_ratio),
        "ov_min_in": float(max(0.0, float(args.ov_min_in))),
    }
    return ServiceFsmContext(
        enabled=bool(args.fsm_enable),
        enter_n=int(max(1, int(args.enter_n))),
        enter_in_n=int(max(1, int(args.enter_in_n))),
        exit_n=int(max(1, int(args.exit_n))),
        grace_n=int(grace_n),
        cand_thr=float(cand_thr),
        in_thr=float(in_thr),
        exit_thr=float(exit_thr),
        det_fps=float(det_fps),
        fps_src=float(fps_src),
        fps_clip=float(fps_clip),
        clip_start_sec=float(clip_start_sec),
        video_id=str(video_id),
        clip_path=str(clip_path),
        params_for_event=params_for_event,
        runtimes=runtimes,
        events_fp=events_fp,
    )


def _finalize_open_service_events(ctx: Optional[ServiceFsmContext], *, end_orig_frame: int) -> int:
    if ctx is None or not bool(ctx.enabled) or ctx.events_fp is None:
        return 0
    n = 0
    for roi_id, runtime in ctx.runtimes.items():
        if runtime.state != STATE_IN or runtime.active_event is None:
            continue
        payload = _service_event_finalize(
            ctx=ctx,
            runtime=runtime,
            roi_id=str(roi_id),
            end_orig_frame=int(end_orig_frame),
        )
        runtime.state = STATE_OUT
        runtime.enter_cnt = 0
        runtime.in_cnt = 0
        runtime.exit_cnt = 0
        runtime.grace_cnt = 0
        if payload is not None:
            append_jsonl(ctx.events_fp, payload)
            n += 1
    return n


def _build_dry_base_frame(width: int, height: int):
    import numpy as np

    y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, width)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.clip(20 + 35 * y, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(28 + 24 * x, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(18 + 28 * (1.0 - y), 0, 255).astype(np.uint8)
    return frame


def _process_frame(
    *,
    frame,
    frame_idx: int,
    ts_sec: float,
    rois: list[RoiRuntime],
    det_all: list[list[float]],
    det_by_roi: dict[str, list[list[float]]],
    feature_cfg: FeatureConfig,
    score_weights: ScoreWeights,
    scores_fp,
    state_history: dict[str, list[str]],
    draw_out_bbox: bool,
    label_ctx: Optional[LabelModeContext] = None,
    draw_all_bbox: bool = False,
    viz_rep_only: bool = False,
    weights_override_active: bool = False,
    weights_debug_text: str = "",
    ov_min_in: float = 0.0,
    band_ratio: float = 0.20,
    is_update_frame: bool = True,
    orig_frame: Optional[int] = None,
    service_ctx: Optional[ServiceFsmContext] = None,
) -> None:
    roi_rows: list[dict[str, Any]] = []
    h, w = frame.shape[:2]
    roi_draw_rows: list[dict[str, Any]] = []
    gt_state: Optional[str] = None
    gt_orig_frame: Optional[int] = None
    orig_frame_eff = int(frame_idx if orig_frame is None else orig_frame)
    if label_ctx is not None:
        gt_state, gt_orig_frame = _label_state_for_frame(label_ctx=label_ctx, frame_idx=frame_idx)

    params = rois[0].fsm.params if rois else FsmParams()
    cand_thr = float(params.cand_thr)
    in_thr = float(params.in_thr)

    for idx, rr in enumerate(rois):
        bboxes = det_by_roi.get(rr.roi_id, det_all)
        score_t = 0.0
        best_bbox: Optional[list[float]] = None
        best_factors: Optional[dict[str, Any]] = None
        best_peak_row: Optional[dict[str, Any]] = None
        roi_in_any = False
        has_any = False
        for bbox in bboxes:
            try:
                factors = compute_bbox_factors(
                    bbox=bbox,
                    roi_cache=rr.cache,
                    cfg_norms=feature_cfg,
                    image_w=w,
                    image_h=h,
                )
            except Exception:
                continue
            s = float(compute_score(factors, score_weights))
            f_ov = float(factors.get("f_ov", 0.0))
            in_gate = (s >= in_thr) and (f_ov >= float(ov_min_in))
            if in_gate:
                roi_in_any = True
            if (not has_any) or (s > score_t):
                bbox_clamped = factors.get("bbox_clamped", [])
                conf_best = float(bbox_clamped[4]) if isinstance(bbox_clamped, (list, tuple)) and len(bbox_clamped) >= 5 else (float(bbox[4]) if len(bbox) >= 5 else 1.0)
                score_t = float(s)
                best_bbox = list(bbox_clamped) if isinstance(bbox_clamped, (list, tuple)) else list(bbox)
                best_factors = factors
                best_peak_row = {
                    "x1": float(bbox_clamped[0]) if isinstance(bbox_clamped, (list, tuple)) and len(bbox_clamped) >= 4 else float(bbox[0]),
                    "y1": float(bbox_clamped[1]) if isinstance(bbox_clamped, (list, tuple)) and len(bbox_clamped) >= 4 else float(bbox[1]),
                    "x2": float(bbox_clamped[2]) if isinstance(bbox_clamped, (list, tuple)) and len(bbox_clamped) >= 4 else float(bbox[2]),
                    "y2": float(bbox_clamped[3]) if isinstance(bbox_clamped, (list, tuple)) and len(bbox_clamped) >= 4 else float(bbox[3]),
                    "conf": float(conf_best),
                    "f_dist": float(factors.get("f_dist", 0.0)),
                    "f_ov": float(factors.get("f_ov", 0.0)),
                    "p_gap": float(factors.get("p_gap", 1.0)),
                }
                has_any = True
        snap = rr.fsm.update(score_t=score_t, frame_idx=frame_idx, ts_sec=ts_sec)
        state_history[rr.roi_id].append(snap.state)
        draw_state = gt_state if gt_state is not None else snap.state
        roi_draw_rows.append(
            {
                "roi_index": idx,
                "rr": rr,
                "state": draw_state,
                "score_t": float(score_t),
                "best_bbox": best_bbox,
                "best_factors": best_factors,
                "roi_score": float(score_t),
                "roi_in_any": bool(roi_in_any),
            }
        )

        service_rt: Optional[ServiceFsmRuntime] = None
        if service_ctx is not None and bool(service_ctx.enabled):
            service_rt = service_ctx.runtimes.setdefault(rr.roi_id, ServiceFsmRuntime())
            if bool(is_update_frame):
                finalized = _service_fsm_step(
                    ctx=service_ctx,
                    runtime=service_rt,
                    roi_id=rr.roi_id,
                    roi_score=float(score_t),
                    roi_in_any=bool(roi_in_any),
                    orig_frame=int(orig_frame_eff),
                    best_row=best_peak_row,
                )
                if finalized is not None and service_ctx.events_fp is not None:
                    append_jsonl(service_ctx.events_fp, finalized)

        row_payload = {
            "roi_id": rr.roi_id,
            "state": snap.state,
            "score_t": float(score_t),
            "best_bbox": best_bbox,
            "best_factors": {
                "f_dist": float(best_factors["f_dist"]) if best_factors else 0.0,
                "f_ov": float(best_factors["f_ov"]) if best_factors else 0.0,
                "p_gap": float(best_factors["p_gap"]) if best_factors else 1.0,
                "sd": float(best_factors["sd"]) if best_factors else None,
                "ov": float(best_factors["ov"]) if best_factors else None,
                "gap_up": best_factors["gap_up"] if best_factors else None,
                "bc": list(best_factors["bc"]) if best_factors else None,
            },
            "num_bboxes": len(bboxes),
            "fsm": snap.to_dict(),
            "transition": snap.transition,
            "roi_score": float(score_t),
            "roi_in_any": bool(roi_in_any),
        }
        if service_rt is not None:
            row_payload.update(
                {
                    "fsm_state": str(service_rt.state),
                    "fsm_enter_cnt": int(service_rt.enter_cnt),
                    "fsm_in_cnt": int(service_rt.in_cnt),
                    "fsm_exit_cnt": int(service_rt.exit_cnt),
                    "fsm_grace_cnt": int(service_rt.grace_cnt),
                    "fsm_event_index": int(service_rt.event_index),
                    "fsm_event_active": bool(service_rt.active_event is not None),
                }
            )
        roi_rows.append(
            {
                **row_payload,
            }
        )

    table_rows: list[dict[str, Any]] = []
    for row in roi_draw_rows:
        best_factors = row["best_factors"] if isinstance(row["best_factors"], dict) else {}
        bbox_clamped = best_factors.get("bbox_clamped", []) if isinstance(best_factors, dict) else []
        conf = float(bbox_clamped[4]) if isinstance(bbox_clamped, (list, tuple)) and len(bbox_clamped) >= 5 else 0.0
        f_dist = float(best_factors.get("f_dist", 0.0)) if isinstance(best_factors, dict) else 0.0
        f_ov = float(best_factors.get("f_ov", 0.0)) if isinstance(best_factors, dict) else 0.0
        p_gap = float(best_factors.get("p_gap", 1.0)) if isinstance(best_factors, dict) else 1.0
        table_rows.append(
            {
                "roi_id": row["rr"].roi_id,
                "conf": conf,
                "state": row["state"],
                "score": float(row["score_t"]),
                "dist": float(score_weights.wd) * f_dist,
                "ov": float(score_weights.wo) * f_ov,
                "gap": -float(score_weights.wg) * p_gap,
            }
        )

    ann_source_bboxes: list[list[float]] = []
    if det_all:
        ann_source_bboxes = [list(b) for b in det_all if isinstance(b, (list, tuple)) and len(b) >= 4]
    else:
        seen: set[tuple[float, float, float, float, float]] = set()
        for boxes in det_by_roi.values():
            if not isinstance(boxes, list):
                continue
            for b in boxes:
                if not isinstance(b, (list, tuple)) or len(b) < 4:
                    continue
                conf_b = float(b[4]) if len(b) >= 5 else 1.0
                key = (
                    round(float(b[0]), 3),
                    round(float(b[1]), 3),
                    round(float(b[2]), 3),
                    round(float(b[3]), 3),
                    round(conf_b, 3),
                )
                if key in seen:
                    continue
                seen.add(key)
                ann_source_bboxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), conf_b])

    bbox_draw: list[dict[str, Any]] = []
    for bbox in ann_source_bboxes:
        score_max = 0.0
        has_any = False
        roi_id_max = ""
        factors_max: dict[str, Any] | None = None
        for rr in rois:
            try:
                factors = compute_bbox_factors(
                    bbox=bbox,
                    roi_cache=rr.cache,
                    cfg_norms=feature_cfg,
                    image_w=w,
                    image_h=h,
                )
            except Exception:
                continue
            s = float(compute_score(factors, score_weights))
            if (not has_any) or (s > score_max):
                score_max = s
                has_any = True
                roi_id_max = rr.roi_id
                factors_max = factors
        score_val = float(score_max) if has_any else 0.0
        conf_b = float(bbox[4]) if len(bbox) >= 5 else 1.0
        x1 = _clamp(float(bbox[0]), 0.0, float(max(0, w - 1)))
        y1 = _clamp(float(bbox[1]), 0.0, float(max(0, h - 1)))
        x2 = _clamp(float(bbox[2]), 0.0, float(max(0, w - 1)))
        y2 = _clamp(float(bbox[3]), 0.0, float(max(0, h - 1)))
        if x2 <= x1 or y2 <= y1:
            continue
        f_dist = float(factors_max.get("f_dist", 0.0)) if isinstance(factors_max, dict) else 0.0
        f_ov = float(factors_max.get("f_ov", 0.0)) if isinstance(factors_max, dict) else 0.0
        p_gap = float(factors_max.get("p_gap", 1.0)) if isinstance(factors_max, dict) else 1.0
        if score_val >= in_thr and f_ov >= float(ov_min_in):
            bbox_state = STATE_IN
        elif score_val >= cand_thr:
            bbox_state = STATE_CAND
        else:
            bbox_state = STATE_OUT
        bbox_draw.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": conf_b,
                "score": score_val,
                "state": bbox_state,
                "roi_id_max": roi_id_max,
                "f_dist": f_dist,
                "f_ov": f_ov,
                "p_gap": p_gap,
                "wd": float(score_weights.wd),
                "wo": float(score_weights.wo),
                "wg": float(score_weights.wg),
                "cand_thr": cand_thr,
                "in_thr": in_thr,
            }
        )

    bbox_draw_viz = list(bbox_draw)
    if bool(viz_rep_only) and bbox_draw:
        rep_idx = max(range(len(bbox_draw)), key=lambda i: float(bbox_draw[i].get("score", 0.0)))
        masked: list[dict[str, Any]] = []
        for i, row in enumerate(bbox_draw):
            if i == rep_idx:
                masked.append(row)
                continue
            row_mask = dict(row)
            x1 = float(row_mask.get("x1", 0.0))
            y1 = float(row_mask.get("y1", 0.0))
            row_mask["x2"] = x1
            row_mask["y2"] = y1
            masked.append(row_mask)
        bbox_draw_viz = masked

    for row in roi_draw_rows:
        rr = row["rr"]
        draw_roi_view(
            frame,
            rr.cache,
            state=row["state"],
            score_t=row["score_t"],
            best_bbox=row["best_bbox"],
            best_factors=row["best_factors"],
            roi_index=row["roi_index"],
            draw_out_bbox=draw_out_bbox,
            draw_global=(int(row["roi_index"]) == 0),
            top_table_rows=table_rows,
            bbox_draw=bbox_draw_viz,
        )

    if bool(draw_all_bbox) and bbox_draw_viz:
        _draw_out_bboxes_gray(frame, bbox_rows=bbox_draw_viz)

    if gt_state is not None and gt_orig_frame is not None:
        import cv2

        cv2.putText(
            frame,
            f"GT: {gt_state} (orig_frame={int(gt_orig_frame)})",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )
    if weights_override_active and weights_debug_text:
        import cv2

        y = 56 if (gt_state is not None and gt_orig_frame is not None) else 30
        cv2.putText(
            frame,
            weights_debug_text,
            (14, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 210, 120),
            2,
            cv2.LINE_AA,
        )

    if weights_override_active and roi_rows:
        roi_rows[0]["weights"] = {
            "w1": float(score_weights.wd),
            "w2": float(score_weights.wo),
            "w3": float(score_weights.wg),
        }
    if roi_rows:
        roi_rows[0]["ov_min_in"] = float(ov_min_in)
    if float(ov_min_in) > 0:
        import cv2

        y_gate = 82
        if gt_state is None and not (weights_override_active and weights_debug_text):
            y_gate = 30
        elif gt_state is not None and not (weights_override_active and weights_debug_text):
            y_gate = 56
        cv2.putText(
            frame,
            f"IN gate: f_ov >= {float(ov_min_in):.3f}",
            (14, int(y_gate)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 255, 180),
            2,
            cv2.LINE_AA,
        )
    if abs(float(band_ratio) - 0.20) > 1e-9:
        import cv2

        y_band = 108
        if gt_state is None and not (weights_override_active and weights_debug_text) and float(ov_min_in) <= 0:
            y_band = 30
        cv2.putText(
            frame,
            f"band_ratio={float(band_ratio):.2f}",
            (14, int(y_band)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (210, 210, 255),
            2,
            cv2.LINE_AA,
        )

    if service_ctx is not None and bool(service_ctx.enabled) and rois:
        import cv2

        rt0 = service_ctx.runtimes.get(rois[0].roi_id)
        if rt0 is not None:
            y_fsm = 134
            if gt_state is None and not weights_override_active and float(ov_min_in) <= 0 and abs(float(band_ratio) - 0.20) <= 1e-9:
                y_fsm = 30
            roi0 = roi_rows[0] if roi_rows else {}
            roi_score_hud = float(roi0.get("roi_score", 0.0))
            roi_in_any_hud = 1 if bool(roi0.get("roi_in_any", False)) else 0
            cv2.putText(
                frame,
                f"FSM: {rt0.state}  enter={int(rt0.enter_cnt)} in={int(rt0.in_cnt)} grace={int(rt0.grace_cnt)} exit={int(rt0.exit_cnt)}  roi_score={roi_score_hud:.3f} in_any={roi_in_any_hud}",
                (14, int(y_fsm)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 255, 220),
                2,
                cv2.LINE_AA,
            )

    append_jsonl(
        scores_fp,
        {
            "frame_idx": int(frame_idx),
            "ts_sec": float(ts_sec),
            "orig_frame": int(orig_frame_eff),
            "is_update_frame": bool(is_update_frame),
            "rois": roi_rows,
        },
    )


def run_dry(
    *,
    args,
    io_ctx,
    logger: logging.Logger,
    cfg: dict[str, Any],
    feature_cfg: FeatureConfig,
    score_weights: ScoreWeights,
) -> int:
    try:
        import cv2
    except Exception as exc:
        logger.error("Missing dependency: cv2 is required (%s)", exc)
        return 2

    width, height = _parse_wh(args.dry_wh)
    nframes = max(1, int(args.dry_frames))
    fps = float(args.dry_fps)
    if fps <= 0:
        logger.error("--dry_fps must be > 0")
        return 2

    rois_raw = build_dry_rois(width=width, height=height, count=int(args.dry_rois))
    fsm_params = FsmParams.from_cfg(cfg.get("fsm", {}), fps=fps)
    rois = [RoiRuntime(roi_id=rid, source_path=src, cache=cache, fsm=RoiFsm(params=fsm_params)) for (rid, src, cache) in rois_raw]
    state_history = {r.roi_id: [] for r in rois}

    writer = create_video_writer(cv2, io_ctx.video_path, width=width, height=height, fps=fps, codec=args.codec)
    scores_fp = io_ctx.scores_path.open("w", encoding="utf-8")
    events_path = Path(str(args.events_jsonl).strip()) if str(args.events_jsonl).strip() else (io_ctx.out_dir / "events.jsonl")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_fp = events_path.open("w", encoding="utf-8")
    params = rois[0].fsm.params if rois else FsmParams()
    service_ctx = _build_service_fsm_context(
        args=args,
        rois=rois,
        events_fp=events_fp,
        cand_thr=float(params.cand_thr),
        in_thr=float(params.in_thr),
        fps_src=float(fps),
        fps_clip=float(fps),
        clip_start_sec=0.0,
        video_id="dry_run",
        clip_path="__dry_run__",
        score_weights=score_weights,
    )
    base = _build_dry_base_frame(width=width, height=height)
    weights_override_active = (args.w1 is not None) or (args.w2 is not None) or (args.w3 is not None)
    weights_debug_text = f"W: w1={float(score_weights.wd):.3f} w2={float(score_weights.wo):.3f} w3={float(score_weights.wg):.3f}"

    logger.info("Dry run started: %d frames, %dx%d, fps=%.3f, rois=%d", nframes, width, height, fps, len(rois))
    try:
        for frame_idx in range(nframes):
            frame = base.copy()
            cv2.putText(
                frame,
                f"DRY RUN frame={frame_idx}",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            det_by_roi = {rr.roi_id: synth_bboxes_for_roi(frame_idx, nframes=nframes, roi_cache=rr.cache) for rr in rois}
            _process_frame(
                frame=frame,
                frame_idx=frame_idx,
                ts_sec=float(frame_idx) / fps,
                rois=rois,
                det_all=[],
                det_by_roi=det_by_roi,
                feature_cfg=feature_cfg,
                score_weights=score_weights,
                scores_fp=scores_fp,
                state_history=state_history,
                draw_out_bbox=bool(args.draw_out_bbox),
                draw_all_bbox=bool(args.draw_all_bbox),
                viz_rep_only=bool(args.viz_rep_only),
                weights_override_active=bool(weights_override_active),
                weights_debug_text=weights_debug_text,
                ov_min_in=float(max(0.0, float(args.ov_min_in))),
                band_ratio=float(args.band_ratio),
                is_update_frame=True,
                orig_frame=int(frame_idx),
                service_ctx=service_ctx,
            )
            writer.write(frame)
            if frame_idx % 15 == 0:
                logger.info("Dry progress frame=%d/%d", frame_idx, nframes - 1)
    finally:
        _finalize_open_service_events(service_ctx, end_orig_frame=max(0, nframes - 1))
        writer.release()
        scores_fp.close()
        events_fp.close()

    target = [STATE_OUT, STATE_CAND, STATE_IN, STATE_OUT]
    dry_ok = any(_has_transition_sequence(state_history[roi_id], target) for roi_id in state_history)
    if dry_ok:
        logger.info("Dry transition check passed: found OUT->CAND->IN->OUT sequence")
    else:
        logger.error("Dry transition check failed: no OUT->CAND->IN->OUT sequence found")
        return 3

    meta = {
        "stage": STAGE,
        "mode": "dry_run",
        "W": int(width),
        "H": int(height),
        "fps": float(fps),
        "nframes": int(nframes),
        "cmd": " ".join(sys.argv),
        "cfg_path": str(Path(args.cfg)),
        "roi_paths": [r.source_path for r in rois],
        "roi_ids": [r.roi_id for r in rois],
        "git_commit": run_utils.get_git_commit(),
        "outputs": {
            "video": str(io_ctx.video_path),
            "scores_jsonl": str(io_ctx.scores_path),
            "events_jsonl": str(events_path),
            "meta_json": str(io_ctx.meta_path),
            "params_used_yaml": str(io_ctx.params_path),
            "cmd_path": str(io_ctx.cmd_path),
            "log_path": str(io_ctx.log_path),
        },
        "fsm_service": {
            "enabled": bool(args.fsm_enable),
            "num_events": int(service_ctx.num_events),
            "total_in_time_sec": float(service_ctx.total_in_time_sec),
        },
        "state_summary": {roi_id: _state_counts(states) for (roi_id, states) in state_history.items()},
    }
    write_json(io_ctx.meta_path, meta)
    logger.info(
        "FSM summary (dry): enabled=%s num_events=%d total_in_time_sec=%.3f events_jsonl=%s",
        bool(args.fsm_enable),
        int(service_ctx.num_events),
        float(service_ctx.total_in_time_sec),
        events_path,
    )
    logger.info("Dry run completed: %s", io_ctx.out_dir)
    return 0


def run_real(
    *,
    args,
    io_ctx,
    logger: logging.Logger,
    cfg: dict[str, Any],
    feature_cfg: FeatureConfig,
    score_weights: ScoreWeights,
) -> int:
    if not str(args.video).strip():
        logger.error("Real mode requires --video")
        return 2
    if not args.roi_json:
        logger.error("Real mode requires one or more --roi_json files")
        return 2

    try:
        import cv2
    except Exception as exc:
        logger.error("Missing dependency: cv2 is required (%s)", exc)
        return 2

    video_path = Path(str(args.video))
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return 2

    fps_in_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_in = fps_in_raw if fps_in_raw > 0 else 30.0
    fps = fps_in
    det_fps = float(args.det_fps)
    if det_fps <= 0:
        cap.release()
        logger.error("--det_fps must be > 0")
        return 2
    det_stride = max(1, int(round(float(fps_in) / float(det_fps))))
    hold_det = bool(args.hold_det)
    logger.info("det_fps=%.3f fps_in=%.3f stride=%d hold_det=%s", det_fps, fps_in, det_stride, hold_det)
    weights_override_active = (args.w1 is not None) or (args.w2 is not None) or (args.w3 is not None)
    weights_debug_text = f"W: w1={float(score_weights.wd):.3f} w2={float(score_weights.wo):.3f} w3={float(score_weights.wg):.3f}"
    label_ctx = _build_label_mode_context(
        args=args,
        logger=logger,
        video_path=video_path,
        fps_clip=fps_in,
    )

    det_jsonl = str(args.det_jsonl).strip()
    yolo_model = str(args.yolo_model).strip()
    if det_jsonl and yolo_model:
        logger.warning("Both --det_jsonl and --yolo_model provided; using --det_jsonl and ignoring YOLO.")

    det_source = ""
    if det_jsonl:
        try:
            det_map = load_det_jsonl(det_jsonl)
        except Exception as exc:
            cap.release()
            logger.error("Failed to load --det_jsonl: %s", exc)
            return 2
        det_source = "det_jsonl"
    elif yolo_model:
        try:
            det_map = build_det_map_yolo(
                video_path=video_path,
                model_path=yolo_model,
                device=args.device,
                conf=args.conf,
                imgsz=args.imgsz,
                det_stride=det_stride,
                max_frames=int(args.max_frames),
            )
        except Exception as exc:
            cap.release()
            logger.error("Failed to build detections via --yolo_model: %s", exc)
            return 2
        det_source = "yolo"
    else:
        cap.release()
        logger.error("Real mode requires either --det_jsonl or --yolo_model")
        return 2
    logger.info("det_source=%s", det_source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            logger.error("Failed to read first frame from video")
            return 2
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        rois_raw = _build_real_rois(args.roi_json, width=width, height=height)
    except Exception as exc:
        cap.release()
        logger.error("Failed to load ROI(s): %s", exc)
        return 2

    fsm_params = FsmParams.from_cfg(cfg.get("fsm", {}), fps=fps)
    rois = [RoiRuntime(roi_id=rid, source_path=src, cache=cache, fsm=RoiFsm(params=fsm_params)) for (rid, src, cache) in rois_raw]
    state_history = {r.roi_id: [] for r in rois}
    fsm_params_eff = rois[0].fsm.params if rois else FsmParams()
    video_id = _infer_video_id_from_clip(video_path)
    if label_ctx is not None:
        fps_src_event = float(label_ctx.fps_src)
        clip_start_sec_event = float(label_ctx.clip_start_sec)
    else:
        fps_src_event = float(args.label_fps) if args.label_fps is not None and float(args.label_fps) > 0 else float(fps_in if fps_in > 0 else 30.0)
        clip_start_sec_event = _infer_clip_start_sec_generic(
            args=args,
            logger=logger,
            video_path=video_path,
            fps_src=float(fps_src_event),
            warn_prefix="fsm",
        )
    events_path = Path(str(args.events_jsonl).strip()) if str(args.events_jsonl).strip() else (io_ctx.out_dir / "events.jsonl")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_fp = events_path.open("w", encoding="utf-8")
    service_ctx = _build_service_fsm_context(
        args=args,
        rois=rois,
        events_fp=events_fp,
        cand_thr=float(fsm_params_eff.cand_thr),
        in_thr=float(fsm_params_eff.in_thr),
        fps_src=float(fps_src_event),
        fps_clip=float(fps_in),
        clip_start_sec=float(clip_start_sec_event),
        video_id=str(video_id),
        clip_path=str(video_path),
        score_weights=score_weights,
    )
    logger.info(
        "service_fsm enabled=%s enter_n=%d enter_in_n=%d exit_n=%d grace_n=%d exit_thr=%.6f fps_src=%.6f clip_start_sec=%.6f",
        bool(service_ctx.enabled),
        int(service_ctx.enter_n),
        int(service_ctx.enter_in_n),
        int(service_ctx.exit_n),
        int(service_ctx.grace_n),
        float(service_ctx.exit_thr),
        float(service_ctx.fps_src),
        float(service_ctx.clip_start_sec),
    )

    writer = None
    scores_fp = None
    frame_idx = -1
    last_orig_frame = 0
    last_det_all: list[list[float]] = []
    last_det_by_roi: dict[str, list[list[float]]] = {}
    try:
        writer = create_video_writer(cv2, io_ctx.video_path, width=width, height=height, fps=fps, codec=args.codec)
        scores_fp = io_ctx.scores_path.open("w", encoding="utf-8")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if int(args.max_frames) > 0 and frame_idx >= int(args.max_frames):
                break

            is_update_frame = (frame_idx % det_stride) == 0
            if is_update_frame:
                det_slot = det_map.get(frame_idx, {"all": [], "by_roi": {}})
                if isinstance(det_slot, dict):
                    slot_all = det_slot.get("all", [])
                    slot_by_roi = det_slot.get("by_roi", {})
                    last_det_all = list(slot_all) if isinstance(slot_all, list) else []
                    if isinstance(slot_by_roi, dict):
                        clean_by_roi: dict[str, list[list[float]]] = {}
                        for roi_id, items in slot_by_roi.items():
                            clean_by_roi[str(roi_id)] = list(items) if isinstance(items, list) else []
                        last_det_by_roi = clean_by_roi
                    else:
                        last_det_by_roi = {}
                else:
                    last_det_all = []
                    last_det_by_roi = {}

            if hold_det or is_update_frame:
                det_all = last_det_all
                det_by_roi = last_det_by_roi
            else:
                det_all = []
                det_by_roi = {}
            fps_clip_eff = float(fps_in if fps_in > 0 else 30.0)
            t_sec = float(frame_idx) / fps_clip_eff
            orig_frame = int(round((float(t_sec) + float(clip_start_sec_event)) * float(fps_src_event)))
            last_orig_frame = int(orig_frame)
            _process_frame(
                frame=frame,
                frame_idx=frame_idx,
                ts_sec=float(frame_idx) / fps,
                rois=rois,
                det_all=det_all,
                det_by_roi=det_by_roi,
                feature_cfg=feature_cfg,
                score_weights=score_weights,
                scores_fp=scores_fp,
                state_history=state_history,
                draw_out_bbox=bool(args.draw_out_bbox),
                label_ctx=label_ctx,
                draw_all_bbox=bool(args.draw_all_bbox),
                viz_rep_only=bool(args.viz_rep_only),
                weights_override_active=bool(weights_override_active),
                weights_debug_text=weights_debug_text,
                ov_min_in=float(max(0.0, float(args.ov_min_in))),
                band_ratio=float(args.band_ratio),
                is_update_frame=bool(is_update_frame),
                orig_frame=int(orig_frame),
                service_ctx=service_ctx,
            )
            writer.write(frame)
            if frame_idx % 120 == 0:
                logger.info("Real progress frame=%d", frame_idx)
    finally:
        _finalize_open_service_events(service_ctx, end_orig_frame=int(last_orig_frame))
        cap.release()
        if writer is not None:
            writer.release()
        if scores_fp is not None:
            scores_fp.close()
        events_fp.close()

    nframes = max(0, frame_idx + 1)
    meta = {
        "stage": STAGE,
        "mode": "real_det_jsonl",
        "det_source": det_source,
        "det_fps": float(det_fps),
        "det_stride": int(det_stride),
        "hold_det": bool(hold_det),
        "W": int(width),
        "H": int(height),
        "fps": float(fps),
        "nframes": int(nframes),
        "cmd": " ".join(sys.argv),
        "cfg_path": str(Path(args.cfg)),
        "video_path": str(video_path),
        "roi_paths": [r.source_path for r in rois],
        "roi_ids": [r.roi_id for r in rois],
        "det_jsonl": str(Path(args.det_jsonl)) if det_jsonl else "",
        "yolo_model": yolo_model,
        "git_commit": run_utils.get_git_commit(),
        "outputs": {
            "video": str(io_ctx.video_path),
            "scores_jsonl": str(io_ctx.scores_path),
            "events_jsonl": str(events_path),
            "meta_json": str(io_ctx.meta_path),
            "params_used_yaml": str(io_ctx.params_path),
            "cmd_path": str(io_ctx.cmd_path),
            "log_path": str(io_ctx.log_path),
        },
        "fsm_service": {
            "enabled": bool(args.fsm_enable),
            "video_id": str(video_id),
            "fps_src": float(fps_src_event),
            "clip_start_sec": float(clip_start_sec_event),
            "num_events": int(service_ctx.num_events),
            "total_in_time_sec": float(service_ctx.total_in_time_sec),
        },
        "state_summary": {roi_id: _state_counts(states) for (roi_id, states) in state_history.items()},
    }
    write_json(io_ctx.meta_path, meta)
    logger.info(
        "FSM summary (real): enabled=%s num_events=%d total_in_time_sec=%.3f events_jsonl=%s",
        bool(args.fsm_enable),
        int(service_ctx.num_events),
        float(service_ctx.total_in_time_sec),
        events_path,
    )
    logger.info("Real mode completed: %s", io_ctx.out_dir)
    return 0


def main() -> int:
    args = build_parser().parse_args()

    try:
        cfg = load_yaml_config(args.cfg)
    except Exception as exc:
        print(f"[ERROR] Failed to load --cfg: {exc}", file=sys.stderr)
        return 2

    score_cfg = cfg.get("score", {})
    if not isinstance(score_cfg, dict):
        print("[ERROR] Config 'score' must be a mapping", file=sys.stderr)
        return 2
    fsm_cfg = cfg.get("fsm", {})
    if not isinstance(fsm_cfg, dict):
        print("[ERROR] Config 'fsm' must be a mapping", file=sys.stderr)
        return 2

    io_ctx = init_io(
        stage=STAGE,
        out_root=args.out_root,
        log_root=args.log_root,
        out_base=args.out_base,
        run_ts=args.run_ts,
        argv=sys.argv,
        log_level=args.log_level,
    )
    logger = logging.getLogger(__name__)
    logger.info("cmd saved: %s", io_ctx.cmd_path)
    logger.info("log saved: %s", io_ctx.log_path)
    logger.info("output dir: %s", io_ctx.out_dir)

    try:
        save_yaml(io_ctx.params_path, cfg)
    except Exception as exc:
        logger.error("Failed to save params_used.yaml: %s", exc)
        return 2

    try:
        feature_cfg = FeatureConfig.from_score_cfg(score_cfg)
        feature_cfg = FeatureConfig(
            d0_ratio=float(feature_cfg.d0_ratio),
            ov0=float(feature_cfg.ov0),
            g0_ratio=float(feature_cfg.g0_ratio),
            g1_ratio=float(feature_cfg.g1_ratio),
            lower_ratio=float(args.band_ratio),
        )
        if abs(float(args.band_ratio) - 0.20) > 1e-9:
            logger.info("band_ratio override active: %.6f", float(args.band_ratio))
        score_weights_cfg = ScoreWeights.from_score_cfg(score_cfg)
        score_weights, _weights_override_active = _effective_score_weights(args=args, base=score_weights_cfg, logger=logger)
    except Exception as exc:
        logger.error("Failed to parse score config: %s", exc)
        return 2

    if args.dry_run:
        return run_dry(
            args=args,
            io_ctx=io_ctx,
            logger=logger,
            cfg={"score": score_cfg, "fsm": fsm_cfg},
            feature_cfg=feature_cfg,
            score_weights=score_weights,
        )

    return run_real(
        args=args,
        io_ctx=io_ctx,
        logger=logger,
        cfg={"score": score_cfg, "fsm": fsm_cfg},
        feature_cfg=feature_cfg,
        score_weights=score_weights,
    )


if __name__ == "__main__":
    raise SystemExit(main())

# Usage examples:
# 1) Clean demo (no bboxes)
# python scripts/02_fsm/02_01_score_mvp_fsm.py --video path/to/input.mp4 --roi_json path/to/roi.json --det_jsonl path/to/dets.jsonl --fsm_enable --enter_n 3 --enter_in_n 2 --exit_n 8 --grace_sec 2.0
# 2) Debug (all bboxes)
# python scripts/02_fsm/02_01_score_mvp_fsm.py --video path/to/input.mp4 --roi_json path/to/roi.json --det_jsonl path/to/dets.jsonl --fsm_enable
# 3) Only show red bbox when IN
# python scripts/02_fsm/02_01_score_mvp_fsm.py --video path/to/input.mp4 --roi_json path/to/roi.json --det_jsonl path/to/dets.jsonl --fsm_enable --viz_rep_only
