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
from typing import Any, Optional, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aidlib import run_utils


STAGE = "01_runtime"


@dataclass
class SourceInfo:
    source: str
    video_id: str
    is_rtsp: bool


def build_parser():
    parser = run_utils.common_argparser()

    parser.add_argument("--source", default="", help="mp4 path OR rtsp url; if empty, use --video_id")
    parser.add_argument("--video_id", default="", help="e.g., E01_009 -> data/videos/E01_009.mp4")
    parser.add_argument("--rtsp", action="store_true", help="force treat source as RTSP")
    parser.add_argument("--fps_sample", type=float, default=10.0, help="target processing fps via frame skipping")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--duration_sec", type=float, default=0.0, help="0 means no limit")

    parser.add_argument("--model", required=True, help="local YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--track_conf", type=float, default=-1.0)
    parser.add_argument("--conf_enter", type=float, default=0.35)
    parser.add_argument("--conf_keep", type=float, default=0.20)
    parser.add_argument("--hold_sec", type=float, default=1.5)
    parser.add_argument("--hold_min_sec", type=float, default=0.3)
    parser.add_argument("--hold_max_sec", type=float, default=2.5)
    parser.add_argument("--hold_age_alpha", type=float, default=0.6)
    parser.add_argument("--conf_floor", type=float, default=0.10)
    parser.add_argument("--conf_ema_alpha", type=float, default=0.7)
    parser.add_argument("--low_conf_drop_frames", type=int, default=8)
    parser.add_argument("--min_confirmed_hold_sec", type=float, default=0.7)
    parser.add_argument("--hold_suppress_iou", type=float, default=0.4)
    parser.add_argument("--hold_suppress_contain", type=float, default=0.7)
    parser.add_argument("--hold_dedupe_iou", type=float, default=0.6)
    parser.add_argument("--normal_dedupe_iou", type=float, default=0.7)
    parser.add_argument("--overlap_cluster_iou", type=float, default=0.70)
    parser.add_argument("--overlap_new_track_enter_boost", type=float, default=0.15)
    parser.add_argument("--normal_merge_iou", type=float, default=0.6)
    parser.add_argument("--normal_merge_contain", type=float, default=0.7)
    parser.add_argument("--confirm_k", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="if empty: cuda if available else cpu")
    parser.add_argument("--tracker_cfg", default="bytetrack.yaml", help="Ultralytics tracker config name/path")

    parser.add_argument("--window_name", default="AID Track Live")
    parser.add_argument("--save_video", action="store_true", help="save overlay output video")
    parser.add_argument("--no_show", action="store_true", help="disable cv2.imshow/waitKey for headless runs")
    parser.add_argument(
        "--save_video_fps",
        type=float,
        default=0.0,
        help="output video fps; 0 means use computed output_fps",
    )
    parser.add_argument("--video_codec", default="mp4v", help="VideoWriter codec (e.g., mp4v, XVID)")
    parser.add_argument("--video_ext", default="mp4", help="overlay video extension (e.g., mp4, avi)")
    parser.add_argument("--realtime_sim", action="store_true", help="throttle FILE input to output_fps")
    parser.add_argument("--jump_sec", type=float, default=5.0, help="jump step in seconds for FILE input")

    parser.add_argument("--save_events", action="store_true", help="write events.jsonl")
    parser.add_argument("--save_snapshot", action="store_true", help="enable key 's' to save snapshots")
    parser.add_argument("--show_hold_when_overlapped", action="store_true")
    return parser


def resolve_source(args) -> SourceInfo:
    source = args.source.strip()
    video_id = args.video_id.strip()

    if not source:
        if not video_id:
            raise ValueError("Either --source or --video_id must be provided.")
        source = str(Path("data/videos") / f"{video_id}.mp4")
    elif not video_id:
        source_path = Path(source)
        if source_path.suffix.lower() == ".mp4":
            video_id = source_path.stem

    is_rtsp = bool(args.rtsp) or source.lower().startswith("rtsp://")
    return SourceInfo(source=source, video_id=video_id, is_rtsp=is_rtsp)


def choose_device(arg_device: str, torch_module) -> str:
    if arg_device.strip():
        return arg_device.strip()
    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def init_events_writer(enabled: bool, out_dir: Path) -> Optional[TextIO]:
    if not enabled:
        return None
    path = out_dir / "events.jsonl"
    return path.open("w", encoding="utf-8")


def get_versions(cv2_module, ultralytics_module, torch_module) -> dict[str, Optional[str]]:
    return {
        "python": platform.python_version(),
        "cv2": getattr(cv2_module, "__version__", None),
        "ultralytics": getattr(ultralytics_module, "__version__", None),
        "torch": getattr(torch_module, "__version__", None) if torch_module is not None else None,
    }


def load_yolo_model(yolo_cls, model_path: str):
    return yolo_cls(model_path)


def reset_tracker(yolo_cls, model_path: str, logger: logging.Logger, reason: str):
    logger.info("Reset tracker (%s): reloading YOLO model.", reason)
    return load_yolo_model(yolo_cls, model_path)


def reset_hysteresis_state(
    state_dict: dict[int, dict[str, Any]],
    reason: str,
    logger: logging.Logger,
    hold_skip_logged_ids: Optional[set[int]] = None,
) -> None:
    state_dict.clear()
    if hold_skip_logged_ids is not None:
        hold_skip_logged_ids.clear()
    logger.info("hysteresis state cleared due to %s", reason)


def format_track_label(
    track_id: Optional[int], conf: Optional[float], is_hold: bool = False, hold_left_sec: Optional[float] = None
) -> str:
    tid = "-" if track_id is None else str(track_id)
    if is_hold:
        if hold_left_sec is None:
            return f"id={tid} HOLD"
        return f"id={tid} HOLD {max(0.0, hold_left_sec):.1f}s"
    c = 0.0 if conf is None else conf
    return f"id={tid} conf={c:.2f}"


def draw_overlay(
    frame: Any,
    tracks: list[dict[str, Any]],
    frame_idx_processed: int,
    effective_fps: float,
    stride: int,
    mode: str,
    paused: bool,
) -> Any:
    disp = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox_xyxy"]]
        conf = t.get("conf")
        tid = t.get("track_id")
        is_hold = bool(t.get("is_hold", False))
        is_overlap_debug_hold = bool(t.get("debug_overlap", False))
        hold_left_sec = t.get("hold_left_sec")
        color = (0, 165, 255) if is_hold else (60, 220, 60)
        thickness = 1 if (is_hold and is_overlap_debug_hold) else 2
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            disp,
            format_track_label(tid, conf, is_hold=is_hold, hold_left_sec=hold_left_sec),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    hud_lines = [
        f"frame_idx_processed: {frame_idx_processed}",
        f"effective_fps: {effective_fps:.2f}",
        f"stride: {stride}",
        f"mode: {mode}",
        f"paused: {paused}",
    ]
    y = 24
    for line in hud_lines:
        cv2.putText(
            disp,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return disp


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
        track_id = None
        if ids is not None and i < len(ids):
            raw_id = ids[i]
            if raw_id == raw_id:  # NaN-safe check without extra dependency
                track_id = int(raw_id)
        tracks.append(
            {
                "track_id": track_id,
                "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                "conf": float(confs[i]) if i < len(confs) else None,
            }
        )
    if has_ids:
        tracks = [t for t in tracks if t["track_id"] is not None]
    return tracks


def build_overlay_tracks_with_hysteresis(
    tracks: list[dict[str, Any]],
    track_state: dict[int, dict[str, Any]],
    ts_sec: float,
    conf_enter: float,
    conf_keep: float,
    hold_sec: float,
    hold_min_sec: float,
    hold_max_sec: float,
    hold_age_alpha: float,
    conf_floor: float,
    conf_ema_alpha: float,
    low_conf_drop_frames: int,
    min_confirmed_hold_sec: float,
    confirm_k: int,
    overlap_cluster_iou: float,
    overlap_new_track_enter_boost: float,
    logger: logging.Logger,
    hold_skip_logged_ids: set[int],
) -> list[dict[str, Any]]:
    overlay_tracks: list[dict[str, Any]] = []
    seen_confirmed_ids: set[int] = set()
    confirmed_ids_before = {tid for tid, st in track_state.items() if st.get("confirmed", False)}
    overlap_conflict_new_ids: set[int] = set()

    for i in range(len(tracks)):
        tid_i = tracks[i].get("track_id")
        if tid_i is None:
            continue
        st_i = track_state.get(tid_i, {})
        if st_i.get("confirmed", False):
            continue
        box_i = tracks[i]["bbox_xyxy"]
        for j in range(len(tracks)):
            if i == j:
                continue
            tid_j = tracks[j].get("track_id")
            if tid_j is None:
                continue
            if tid_j not in confirmed_ids_before:
                continue
            if iou_xyxy(box_i, tracks[j]["bbox_xyxy"]) >= overlap_cluster_iou:
                overlap_conflict_new_ids.add(tid_i)
                break

    for t in tracks:
        tid = t.get("track_id")
        conf = t.get("conf")
        if tid is None:
            overlay_tracks.append(
                {
                    "track_id": None,
                    "bbox_xyxy": t["bbox_xyxy"],
                    "conf": t.get("conf"),
                    "is_hold": False,
                }
            )
            continue

        state = track_state.get(tid)
        if state is None:
            state = {
                "last_bbox_xyxy": t["bbox_xyxy"],
                "last_seen_ts": ts_sec,
                "first_seen_ts": ts_sec,
                "confirmed_at_ts": None,
                "conf_ema": None,
                "low_conf_count": 0,
                "confirm_count": 0,
                "confirmed": False,
            }
        else:
            state.setdefault("first_seen_ts", ts_sec)
            state.setdefault("conf_ema", None)
            state.setdefault("low_conf_count", 0)

        if conf is not None:
            prev_ema = state.get("conf_ema")
            if prev_ema is None:
                state["conf_ema"] = float(conf)
            else:
                state["conf_ema"] = (conf_ema_alpha * float(conf)) + ((1.0 - conf_ema_alpha) * float(prev_ema))
        conf_ema_val = state.get("conf_ema")
        if conf_ema_val is not None and conf_ema_val < conf_floor:
            state["low_conf_count"] = int(state.get("low_conf_count", 0)) + 1
        else:
            state["low_conf_count"] = 0

        if not state["confirmed"]:
            enter_thr = conf_enter + (overlap_new_track_enter_boost if tid in overlap_conflict_new_ids else 0.0)
            if conf is not None and conf >= enter_thr:
                state["confirm_count"] += 1
            else:
                state["confirm_count"] = 0
            if state["confirm_count"] >= max(1, confirm_k):
                state["confirmed"] = True
                if state.get("confirmed_at_ts") is None:
                    state["confirmed_at_ts"] = ts_sec

        # Always draw current-frame normal tracks, even before confirmation.
        overlay_tracks.append(
            {
                "track_id": tid,
                "bbox_xyxy": t["bbox_xyxy"],
                "conf": t.get("conf"),
                "is_hold": False,
            }
        )

        if state["confirmed"]:
            hold_skip_logged_ids.discard(tid)
            seen_confirmed_ids.add(tid)
            confirmed_duration = ts_sec - float(state.get("confirmed_at_ts", ts_sec))
            age_sec = max(0.0, ts_sec - float(state.get("first_seen_ts", ts_sec)))
            hold_dyn = min(hold_max_sec, hold_min_sec + hold_age_alpha * math.log1p(age_sec))
            conf_ema_for_hold = state.get("conf_ema")
            if conf_ema_for_hold is None:
                conf_factor = 1.0
            else:
                denom = max(1e-6, 0.8 - conf_floor)
                conf_factor = (float(conf_ema_for_hold) - conf_floor) / denom
                conf_factor = max(0.0, min(1.0, conf_factor))
            hold_eff = hold_min_sec + (hold_dyn - hold_min_sec) * conf_factor
            conf_keep_dyn = max(conf_floor, conf_keep - 0.05 * math.log1p(age_sec))
            if conf is None or conf >= conf_keep_dyn:
                state["last_bbox_xyxy"] = t["bbox_xyxy"]
                state["last_seen_ts"] = ts_sec
            else:
                hold_left = hold_eff - max(0.0, ts_sec - float(state["last_seen_ts"]))
                if (
                    hold_left > 0
                    and confirmed_duration >= min_confirmed_hold_sec
                    and int(state.get("low_conf_count", 0)) < max(1, low_conf_drop_frames)
                ):
                    overlay_tracks.append(
                        {
                            "track_id": tid,
                            "bbox_xyxy": state["last_bbox_xyxy"],
                            "conf": None,
                            "is_hold": True,
                            "hold_left_sec": hold_left,
                        }
                    )

        track_state[tid] = state

    for tid, state in list(track_state.items()):
        if tid in seen_confirmed_ids:
            continue
        if not state.get("confirmed", False):
            if tid not in hold_skip_logged_ids:
                logger.debug("HOLD skipped for unconfirmed track_id=%s", tid)
                hold_skip_logged_ids.add(tid)
            track_state.pop(tid, None)
            continue
        confirmed_duration = ts_sec - float(state.get("confirmed_at_ts", ts_sec))
        if confirmed_duration < min_confirmed_hold_sec:
            continue
        age_sec = max(0.0, ts_sec - float(state.get("first_seen_ts", ts_sec)))
        hold_dyn = min(hold_max_sec, hold_min_sec + hold_age_alpha * math.log1p(age_sec))
        conf_ema_for_hold = state.get("conf_ema")
        if conf_ema_for_hold is None:
            conf_factor = 1.0
        else:
            denom = max(1e-6, 0.8 - conf_floor)
            conf_factor = (float(conf_ema_for_hold) - conf_floor) / denom
            conf_factor = max(0.0, min(1.0, conf_factor))
        hold_eff = hold_min_sec + (hold_dyn - hold_min_sec) * conf_factor
        conf_ema_val = state.get("conf_ema")
        if conf_ema_val is not None and conf_ema_val < conf_floor:
            state["low_conf_count"] = int(state.get("low_conf_count", 0)) + 1
        else:
            state["low_conf_count"] = 0
        if int(state.get("low_conf_count", 0)) >= max(1, low_conf_drop_frames):
            track_state.pop(tid, None)
            continue
        age = max(0.0, ts_sec - float(state["last_seen_ts"]))
        hold_left = hold_eff - age
        if hold_left > 0:
            overlay_tracks.append(
                {
                    "track_id": tid,
                    "bbox_xyxy": state["last_bbox_xyxy"],
                    "conf": None,
                    "is_hold": True,
                    "hold_left_sec": hold_left,
                }
            )
        else:
            track_state.pop(tid, None)

    for tid, state in list(track_state.items()):
        if state.get("confirmed", False):
            continue
        if max(0.0, ts_sec - float(state["last_seen_ts"])) > hold_sec:
            track_state.pop(tid, None)

    return overlay_tracks


def intersection_area_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_area = intersection_area_xyxy(box_a, box_b)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def containment_ratio_xyxy(small_box: list[float], big_box: list[float]) -> float:
    sx1, sy1, sx2, sy2 = small_box
    small_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
    if small_area <= 0.0:
        return 0.0
    return intersection_area_xyxy(small_box, big_box) / small_area


def dedupe_normal_tracks(normal_tracks: list[dict[str, Any]], iou_thr: float) -> list[dict[str, Any]]:
    ordered = sorted(normal_tracks, key=lambda t: (t.get("conf") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in ordered:
        cand_box = cand["bbox_xyxy"]
        overlapped = False
        for kept_t in kept:
            if iou_xyxy(cand_box, kept_t["bbox_xyxy"]) >= iou_thr:
                overlapped = True
                break
        if not overlapped:
            kept.append(cand)
    return kept


def dedupe_hold_tracks(
    hold_tracks: list[dict[str, Any]],
    track_state: dict[int, dict[str, Any]],
    iou_thr: float,
    contain_thr: float,
) -> list[dict[str, Any]]:
    if len(hold_tracks) <= 1:
        return hold_tracks

    n = len(hold_tracks)
    visited = [False] * n
    kept: list[dict[str, Any]] = []

    def _hold_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
        box_a = a["bbox_xyxy"]
        box_b = b["bbox_xyxy"]
        if iou_xyxy(box_a, box_b) >= iou_thr:
            return True
        if containment_ratio_xyxy(box_a, box_b) >= contain_thr:
            return True
        if containment_ratio_xyxy(box_b, box_a) >= contain_thr:
            return True
        return False

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        cluster_idx: list[int] = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            cluster_idx.append(cur)
            for j in range(n):
                if visited[j]:
                    continue
                if _hold_overlap(hold_tracks[cur], hold_tracks[j]):
                    visited[j] = True
                    stack.append(j)

        def _hold_sort_key(idx: int):
            t = hold_tracks[idx]
            last_seen = float(track_state.get(t.get("track_id"), {}).get("last_seen_ts", float("-inf")))
            return last_seen

        best_idx = max(cluster_idx, key=_hold_sort_key)
        kept.append(hold_tracks[best_idx])

    kept.sort(
        key=lambda t: float(track_state.get(t.get("track_id"), {}).get("last_seen_ts", float("-inf"))),
        reverse=True,
    )
    return kept


def normalize_overlay_overlaps(
    overlay_tracks: list[dict[str, Any]],
    track_state: dict[int, dict[str, Any]],
    iou_thr: float,
    show_hold_when_overlapped: bool = False,
) -> list[dict[str, Any]]:
    if len(overlay_tracks) <= 1:
        return overlay_tracks

    n = len(overlay_tracks)
    visited = [False] * n
    selected: list[dict[str, Any]] = []

    def _priority(t: dict[str, Any]):
        is_hold = bool(t.get("is_hold", False))
        tid = t.get("track_id")
        confirmed = bool(track_state.get(tid, {}).get("confirmed", False)) if tid is not None else False
        if (not is_hold) and confirmed:
            class_rank = 2
        elif not is_hold:
            class_rank = 1
        else:
            class_rank = 0
        conf = float(t.get("conf") or 0.0)
        x1, y1, x2, y2 = t["bbox_xyxy"]
        area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        last_seen = float(track_state.get(tid, {}).get("last_seen_ts", float("-inf"))) if tid is not None else float("-inf")
        return (class_rank, conf, area, last_seen)

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        cluster_idx: list[int] = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            cluster_idx.append(cur)
            cur_box = overlay_tracks[cur]["bbox_xyxy"]
            for j in range(n):
                if visited[j]:
                    continue
                if iou_xyxy(cur_box, overlay_tracks[j]["bbox_xyxy"]) >= iou_thr:
                    visited[j] = True
                    stack.append(j)

        best_idx = max(cluster_idx, key=lambda idx: _priority(overlay_tracks[idx]))
        best_track = overlay_tracks[best_idx]
        selected.append(best_track)
        if show_hold_when_overlapped and not bool(best_track.get("is_hold", False)):
            for idx in cluster_idx:
                if idx == best_idx:
                    continue
                t = overlay_tracks[idx]
                if bool(t.get("is_hold", False)):
                    debug_hold = dict(t)
                    debug_hold["debug_overlap"] = True
                    selected.append(debug_hold)

    return selected


def _normal_tracks_overlap(a: dict[str, Any], b: dict[str, Any], iou_thr: float, contain_thr: float) -> bool:
    box_a = a["bbox_xyxy"]
    box_b = b["bbox_xyxy"]
    if iou_xyxy(box_a, box_b) >= iou_thr:
        return True
    if containment_ratio_xyxy(box_a, box_b) >= contain_thr:
        return True
    if containment_ratio_xyxy(box_b, box_a) >= contain_thr:
        return True
    return False


def merge_normal_tracks(
    normal_tracks: list[dict[str, Any]],
    track_state: dict[int, dict[str, Any]],
    iou_thr: float,
    contain_thr: float,
) -> list[dict[str, Any]]:
    if len(normal_tracks) <= 1:
        return normal_tracks

    n = len(normal_tracks)
    visited = [False] * n
    merged: list[dict[str, Any]] = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        cluster_idx: list[int] = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            cluster_idx.append(cur)
            for j in range(n):
                if visited[j]:
                    continue
                if _normal_tracks_overlap(normal_tracks[cur], normal_tracks[j], iou_thr, contain_thr):
                    visited[j] = True
                    stack.append(j)

        cluster = [normal_tracks[k] for k in cluster_idx]
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue

        def _primary_sort_key(t: dict[str, Any]):
            tid = t.get("track_id")
            st = track_state.get(tid, {}) if tid is not None else {}
            confirmed_rank = 0 if st.get("confirmed", False) else 1
            first_seen = float(st.get("first_seen_ts", float("inf")))
            conf_rank = -(t.get("conf") or 0.0)
            return (confirmed_rank, first_seen, conf_rank)

        primary = sorted(cluster, key=_primary_sort_key)[0]
        x1 = min(t["bbox_xyxy"][0] for t in cluster)
        y1 = min(t["bbox_xyxy"][1] for t in cluster)
        x2 = max(t["bbox_xyxy"][2] for t in cluster)
        y2 = max(t["bbox_xyxy"][3] for t in cluster)
        merged.append(
            {
                "track_id": primary.get("track_id"),
                "bbox_xyxy": [x1, y1, x2, y2],
                "conf": primary.get("conf"),
                "is_hold": False,
            }
        )

    return merged


def seek_file_capture(
    cap,
    current_ts_sec: float,
    delta_sec: float,
    input_fps: float,
    total_frames: int,
    logger: logging.Logger,
) -> None:
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
        logger.info(
            "Seek via CAP_PROP_POS_FRAMES: target_frame=%d total_frames=%d target_ts_sec=%.3f",
            target_frame,
            total_frames,
            target_ts_sec,
        )
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ts_sec * 1000.0)
        logger.info("Seek via CAP_PROP_POS_MSEC: target_ts_sec=%.3f", target_ts_sec)


def maybe_save_snapshot(
    enabled: bool,
    run_out_dir: Path,
    overlay_frame: Optional[Any],
    frame_idx_processed: int,
    logger: logging.Logger,
) -> None:
    if not enabled:
        logger.info("Snapshot ignored: --save_snapshot not enabled.")
        return
    if overlay_frame is None:
        logger.info("Snapshot ignored: no frame is currently available.")
        return
    snap_dir = run_out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"frame_{frame_idx_processed:06d}.jpg"
    ok = cv2.imwrite(str(snap_path), overlay_frame)
    if ok:
        logger.info("Snapshot saved: %s", snap_path)
    else:
        logger.warning("Failed to save snapshot: %s", snap_path)


def write_params_json(out_dir: Path, params: dict[str, Any]) -> None:
    path = out_dir / "params.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
        f.write("\n")


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

        cv2_module = _cv2
        globals()["cv2"] = _cv2
    except Exception as e:
        logger.exception("Failed to import cv2: %s", e)
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
    mode_name = "RTSP" if is_rtsp else "FILE"
    if not is_rtsp:
        src_path = Path(source)
        if not src_path.exists():
            logger.error("Input file not found: %s", src_path)
            return 2

    device = choose_device(args.device, torch_module)
    logger.info("device=%s", device)

    try:
        model = load_yolo_model(yolo_cls, args.model)
    except Exception as e:
        logger.exception("Failed to load YOLO model from %s: %s", args.model, e)
        return 2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open source: %s", source)
        return 2

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not is_rtsp else 0
    if input_fps > 0 and args.fps_sample > 0:
        stride = max(1, int(round(input_fps / args.fps_sample)))
    else:
        stride = 1
    output_fps = (input_fps / stride) if input_fps > 0 else (args.fps_sample if args.fps_sample > 0 else 10.0)
    video_ext = str(args.video_ext).strip().lstrip(".") or "mp4"
    overlay_video_path = run_paths.out_dir / f"{args.out_base}_overlay.{video_ext}"
    track_conf = args.track_conf if args.track_conf >= 0.0 else (args.conf_keep if args.conf_keep > 0 else 0.15)

    params = {
        "source": source,
        "video_id": source_info.video_id,
        "is_rtsp": is_rtsp,
        "model": args.model,
        "conf": args.conf,
        "track_conf": track_conf,
        "conf_enter": args.conf_enter,
        "conf_keep": args.conf_keep,
        "hold_sec": args.hold_sec,
        "hold_min_sec": args.hold_min_sec,
        "hold_max_sec": args.hold_max_sec,
        "hold_age_alpha": args.hold_age_alpha,
        "conf_floor": args.conf_floor,
        "conf_ema_alpha": args.conf_ema_alpha,
        "low_conf_drop_frames": args.low_conf_drop_frames,
        "min_confirmed_hold_sec": args.min_confirmed_hold_sec,
        "hold_suppress_iou": args.hold_suppress_iou,
        "hold_suppress_contain": args.hold_suppress_contain,
        "hold_dedupe_iou": args.hold_dedupe_iou,
        "normal_dedupe_iou": args.normal_dedupe_iou,
        "overlap_cluster_iou": args.overlap_cluster_iou,
        "overlap_new_track_enter_boost": args.overlap_new_track_enter_boost,
        "show_hold_when_overlapped": bool(args.show_hold_when_overlapped),
        "normal_merge_iou": args.normal_merge_iou,
        "normal_merge_contain": args.normal_merge_contain,
        "confirm_k": args.confirm_k,
        "imgsz": args.imgsz,
        "device": device,
        "fps_sample": args.fps_sample,
        "stride": stride,
        "output_fps": output_fps,
        "realtime_sim": bool(args.realtime_sim),
        "jump_sec": args.jump_sec,
        "max_frames": args.max_frames,
        "duration_sec": args.duration_sec,
        "tracker_cfg": args.tracker_cfg,
        "save_video": bool(args.save_video),
        "no_show": bool(args.no_show),
        "save_video_fps": args.save_video_fps,
        "video_codec": args.video_codec,
        "video_ext": video_ext,
        "overlay_video_path": str(overlay_video_path),
        "versions": versions,
    }
    write_params_json(run_paths.out_dir, params)
    logger.info("Saved params: %s", run_paths.out_dir / "params.json")
    logger.info(
        "source=%s mode=%s input_fps=%.3f total_frames=%d stride=%d output_fps=%.3f",
        source,
        mode_name,
        input_fps,
        total_frames,
        stride,
        output_fps,
    )

    events_fp = init_events_writer(args.save_events, run_paths.out_dir)
    if events_fp is not None:
        logger.info("events.jsonl enabled: %s", run_paths.out_dir / "events.jsonl")

    frame_idx_processed = 0
    frame_idx_input = -1
    start_wall = time.time()
    paused = False
    should_quit = False
    last_overlay: Optional[Any] = None
    last_ts_sec = 0.0
    cls_warned = [False]
    track_state: dict[int, dict[str, Any]] = {}
    hold_skip_logged_ids: set[int] = set()
    video_writer = None
    video_save_enabled = bool(args.save_video)
    wrote_video_frame = False

    try:
        while not should_quit:
            loop_t0 = time.time()

            if paused:
                if args.no_show:
                    key = 255
                else:
                    key = cv2.waitKey(30) & 0xFF
                if key != 255:
                    # Seek base should follow input timeline of the displayed frame, not sampled output timeline.
                    if input_fps > 0 and frame_idx_input >= 0:
                        cur_ts_for_seek = frame_idx_input / input_fps
                    else:
                        cur_ts_for_seek = last_ts_sec
                    should_quit, paused, model, clear_overlay_cache = handle_key(
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
                        save_snapshot=args.save_snapshot,
                        out_dir=run_paths.out_dir,
                        overlay=last_overlay,
                        frame_idx_processed=frame_idx_processed,
                        track_state=track_state,
                        hold_skip_logged_ids=hold_skip_logged_ids,
                    )
                    if clear_overlay_cache:
                        last_overlay = None
                if (not args.no_show) and (last_overlay is not None):
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
            else:
                ts_sec = last_ts_sec
            last_ts_sec = ts_sec

            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=args.tracker_cfg,
                    conf=track_conf,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )
                result = results[0] if results else None
            except Exception as e:
                logger.exception("Tracking failed at frame_idx_input=%d: %s", frame_idx_input, e)
                break

            tracks = extract_person_tracks(result, logger=logger, cls_warned=cls_warned) if result is not None else []
            overlay_tracks = build_overlay_tracks_with_hysteresis(
                tracks=tracks,
                track_state=track_state,
                ts_sec=ts_sec,
                conf_enter=args.conf_enter,
                conf_keep=args.conf_keep,
                hold_sec=args.hold_sec,
                hold_min_sec=args.hold_min_sec,
                hold_max_sec=args.hold_max_sec,
                hold_age_alpha=args.hold_age_alpha,
                conf_floor=args.conf_floor,
                conf_ema_alpha=args.conf_ema_alpha,
                low_conf_drop_frames=args.low_conf_drop_frames,
                min_confirmed_hold_sec=args.min_confirmed_hold_sec,
                confirm_k=args.confirm_k,
                overlap_cluster_iou=args.overlap_cluster_iou,
                overlap_new_track_enter_boost=args.overlap_new_track_enter_boost,
                logger=logger,
                hold_skip_logged_ids=hold_skip_logged_ids,
            )
            normal_tracks = [t for t in overlay_tracks if not bool(t.get("is_hold", False))]
            hold_tracks = [t for t in overlay_tracks if bool(t.get("is_hold", False))]
            normal_tracks = dedupe_normal_tracks(normal_tracks, args.normal_dedupe_iou)
            filtered_hold_tracks: list[dict[str, Any]] = []
            for ht in hold_tracks:
                max_iou = 0.0
                max_contain = 0.0
                for nt in normal_tracks:
                    max_iou = max(max_iou, iou_xyxy(ht["bbox_xyxy"], nt["bbox_xyxy"]))
                    max_contain = max(max_contain, containment_ratio_xyxy(ht["bbox_xyxy"], nt["bbox_xyxy"]))
                if (max_iou < args.hold_suppress_iou) and (max_contain < args.hold_suppress_contain):
                    filtered_hold_tracks.append(ht)
            filtered_hold_tracks = dedupe_hold_tracks(
                hold_tracks=filtered_hold_tracks,
                track_state=track_state,
                iou_thr=args.hold_dedupe_iou,
                contain_thr=args.hold_suppress_contain,
            )
            overlay_tracks = normal_tracks + filtered_hold_tracks
            overlay_tracks = normalize_overlay_overlaps(
                overlay_tracks=overlay_tracks,
                track_state=track_state,
                iou_thr=args.overlap_cluster_iou,
                show_hold_when_overlapped=bool(args.show_hold_when_overlapped),
            )

            for t in tracks:
                if events_fp is None:
                    break
                rec = {
                    "frame_idx": frame_idx_input,
                    "track_id": t["track_id"],
                    "bbox_xyxy": t["bbox_xyxy"],
                    "conf": t["conf"],
                    "ts_sec": ts_sec,
                }
                events_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elapsed = max(1e-6, time.time() - start_wall)
            effective_fps = (frame_idx_processed + 1) / elapsed
            overlay = draw_overlay(
                frame=frame,
                tracks=overlay_tracks,
                frame_idx_processed=frame_idx_processed,
                effective_fps=effective_fps,
                stride=stride,
                mode=mode_name,
                paused=paused,
            )
            last_overlay = overlay

            if video_save_enabled:
                if video_writer is None:
                    fps_to_use = args.save_video_fps if args.save_video_fps > 0 else output_fps
                    fps_to_use = max(1.0, fps_to_use)
                    h, w = overlay.shape[:2]
                    codec = str(args.video_codec).strip() or "mp4v"
                    if len(codec) != 4:
                        logger.warning("Invalid --video_codec '%s'; fallback to 'mp4v'.", codec)
                        codec = "mp4v"
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps_to_use, (w, h))
                        if not video_writer.isOpened():
                            logger.warning("Failed to open VideoWriter: %s", overlay_video_path)
                            video_writer.release()
                            video_writer = None
                            video_save_enabled = False
                        else:
                            logger.info(
                                "Overlay video writer opened: path=%s codec=%s fps=%.3f size=%dx%d",
                                overlay_video_path,
                                codec,
                                fps_to_use,
                                w,
                                h,
                            )
                    except Exception as e:
                        logger.warning("VideoWriter init failed; disabling save_video: %s", e)
                        video_writer = None
                        video_save_enabled = False
                if video_writer is not None:
                    try:
                        video_writer.write(overlay)
                        wrote_video_frame = True
                    except Exception as e:
                        logger.warning("VideoWriter write failed; disabling save_video: %s", e)
                        video_writer.release()
                        video_writer = None
                        video_save_enabled = False

            if args.no_show:
                key = 255
            else:
                cv2.imshow(args.window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
            if key != 255:
                # Seek base should follow input timeline of the displayed frame, not sampled output timeline.
                if input_fps > 0 and frame_idx_input >= 0:
                    cur_ts_for_seek = frame_idx_input / input_fps
                else:
                    cur_ts_for_seek = last_ts_sec
                should_quit, paused, model, clear_overlay_cache = handle_key(
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
                    save_snapshot=args.save_snapshot,
                    out_dir=run_paths.out_dir,
                    overlay=last_overlay,
                    frame_idx_processed=frame_idx_processed,
                    track_state=track_state,
                    hold_skip_logged_ids=hold_skip_logged_ids,
                )
                if clear_overlay_cache:
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
        if events_fp is not None:
            events_fp.close()
        if video_writer is not None:
            video_writer.release()
            if wrote_video_frame:
                logger.info("Saved overlay video: %s", overlay_video_path)
            else:
                logger.info("Overlay video writer closed without frames: %s", overlay_video_path)
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        logger.info("Resources released.")

    return 0


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
    save_snapshot: bool,
    out_dir: Path,
    overlay: Optional[Any],
    frame_idx_processed: int,
    track_state: dict[int, dict[str, Any]],
    hold_skip_logged_ids: set[int],
):
    if key == ord("q"):
        return True, paused, model, False

    if key == ord(" "):
        paused = not paused
        logger.info("paused=%s", paused)
        return False, paused, model, False

    if key == ord("s"):
        maybe_save_snapshot(save_snapshot, out_dir, overlay, frame_idx_processed, logger)
        return False, paused, model, False

    if key == ord("r"):
        model = reset_tracker(yolo_cls, model_path, logger, reason="manual reset key")
        reset_hysteresis_state(
            track_state,
            reason="manual reset key",
            logger=logger,
            hold_skip_logged_ids=hold_skip_logged_ids,
        )
        return False, paused, model, True

    jump_map = {
        ord("d"): +jump_sec,
        ord("a"): -jump_sec,
        ord("D"): +30.0,
        ord("A"): -30.0,
    }
    if key in jump_map:
        if is_rtsp:
            logger.info("Seek not supported on RTSP; jump key ignored.")
            return False, paused, model, False
        delta = jump_map[key]
        seek_file_capture(
            cap=cap,
            current_ts_sec=current_ts_sec,
            delta_sec=delta,
            input_fps=input_fps,
            total_frames=total_frames,
            logger=logger,
        )
        model = reset_tracker(yolo_cls, model_path, logger, reason=f"jump {delta:+.1f}s")
        reset_hysteresis_state(
            track_state,
            reason=f"jump {delta:+.1f}s",
            logger=logger,
            hold_skip_logged_ids=hold_skip_logged_ids,
        )
        return False, paused, model, True

    return False, paused, model, False


if __name__ == "__main__":
    raise SystemExit(main())
