#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aidlib import run_utils

STAGE = "02_intrusion"

STATE_OUT = "OUT"
STATE_CAND = "CANDIDATE"
STATE_IN = "IN"

# COCO-17 keypoint indices
KP_LHIP, KP_RHIP = 11, 12
KP_LANKLE, KP_RANKLE = 15, 16


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
    vertices_file: List[List[float]]
    vertices_orig: List[List[float]]
    labeled_on: str
    disp_scale_used: Optional[float]


@dataclass
class TrackState:
    state: str = STATE_OUT
    actor_id: int = -1
    last_track_id: int = -1
    last_track_supported: bool = False
    last_conf: Optional[float] = None
    last_foot_xy: Optional[List[float]] = None
    last_foot_method: str = "BBOX"
    last_foot_in_roi: bool = False
    last_ankles: List[Dict[str, Any]] = field(default_factory=list)
    last_seen_frame: int = -1
    last_seen_ts: float = 0.0
    last_bbox: Optional[List[float]] = None
    last_ioa: float = 0.0
    dwell_frames: int = 0
    enter_streak: int = 0
    exit_streak: int = 0
    candidate_enter_frame: Optional[int] = None
    candidate_enter_ts: Optional[float] = None
    confirm_frame: Optional[int] = None
    confirm_ts: Optional[float] = None
    confirm_reason: Optional[str] = None
    ioa_at_confirm: Optional[float] = None
    max_ioa_current: float = 0.0
    current_event: Optional[Dict[str, Any]] = field(default=None)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"invalid bool value: {v}")


def build_parser():
    p = run_utils.common_argparser()
    p.add_argument("--source", default="", help="input mp4 path; if empty, use --video_id")
    p.add_argument("--video_id", default="", help="e.g., E01_007 -> data/videos/E01_007.mp4")

    # ROI selection
    p.add_argument("--roi_path", default="", help="explicit ROI json path")
    p.add_argument("--roi_video_id", default="", help="ROI namespace video id (configs/rois/<roi_video_id>/...)")
    p.add_argument("--roi_id", default="", help="ROI id (e.g., area01)")
    p.add_argument("--roi_version", type=int, default=0, help="ROI version (>0 exact, <=0 latest)")

    # Intrusion algorithm
    p.add_argument(
        "--algo",
        default="_01_simple_IoA",
        choices=["_01_simple_IoA", "_02_use_keypoint"],
        help="intrusion algorithm stage",
    )
    p.add_argument("--cand_ioa_thr", type=float, default=0.08, help="candidate threshold (IoA)")
    p.add_argument("--in_ioa_thr", type=float, default=0.15, help="confirm threshold (IoA)")
    p.add_argument("--kp_conf", type=float, default=0.35, help="min keypoint confidence for footpoint")
    p.add_argument("--foot_ioa_thr", type=float, default=0.02, help="candidate by foot-in-roi with weak IoA")
    p.add_argument("--foot_confirm_ioa_thr", type=float, default=0.03, help="confirm by foot-in-roi with weak IoA")
    p.add_argument("--pose_draw", type=_str2bool, nargs="?", const=True, default=None, help="draw footpoint marker")
    p.add_argument("--enter_n", type=int, default=3, help="consecutive frames for IN confirm")
    p.add_argument("--exit_n", type=int, default=5, help="consecutive frames for exit")
    p.add_argument("--dwell_sec", type=float, default=1.0, help="minimum dwell sec before IN confirm")
    p.add_argument("--grace_sec", type=float, default=2.0, help="missing-track grace sec before force exit")
    p.add_argument(
        "--missing_policy",
        default="exit",
        choices=["exit", "hold_in"],
        help="missing track handling policy (exit: finalize after grace, hold_in: keep IN until explicit exit)",
    )

    # Detection + tracking
    p.add_argument("--det_model", default="yolov8m.pt", help="Ultralytics detector weights")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--det_conf", type=float, default=0.25)
    p.add_argument("--det_dedup_enable", action="store_true", help="deduplicate overlapping detections before tracker")
    p.add_argument("--det_dedup_iou", type=float, default=0.85, help="IoU threshold for duplicate suppression")
    p.add_argument("--det_dedup_contain_ioa", type=float, default=0.90,
                   help="containment IoA threshold (inter / area(candidate))")
    p.add_argument("--det_dedup_keep", default="max_conf", choices=["max_conf", "max_area"],
                   help="which duplicate candidate to keep")
    p.add_argument("--device", default="0", help="Ultralytics device (e.g., 0, cpu)")
    p.add_argument("--classes", type=int, nargs="*", default=[0], help="class ids to keep (default: person=0)")
    p.add_argument("--tracker", default="deepocsort", choices=["deepocsort", "strongsort"])
    p.add_argument("--reid_model", default="osnet_x0_25_msmt17.pt")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--per_class", action="store_true")
    p.add_argument("--track_max_age", type=int, default=0, help="tracker max_age (0=auto -> grace_frames)")
    p.add_argument("--track_min_hits", type=int, default=3, help="tracker min_hits")
    p.add_argument("--track_max_obs", type=int, default=0, help="tracker max_obs (0=auto)")
    p.add_argument("--track_support_iou", type=float, default=0.10, help="track supported if max IoU(det) >= this")
    p.add_argument("--track_cleanup_enable", action="store_true", help="merge/suppress duplicate tracker boxes")
    p.add_argument("--track_merge_iou", type=float, default=0.80, help="track merge IoU threshold")
    p.add_argument("--track_contain_ioa", type=float, default=0.90, help="track contain-IoA merge threshold")
    p.add_argument("--actor_enable", type=_str2bool, nargs="?", const=True, default=True, help="enable actor-id layer")
    p.add_argument("--actor_use_for_fsm", type=_str2bool, nargs="?", const=True, default=True,
                   help="use actor_id as FSM identity key")
    p.add_argument("--actor_max_gap_sec", type=float, default=2.5, help="max relink gap sec")
    p.add_argument("--actor_iou_thr", type=float, default=0.20, help="relink IoU gate")
    p.add_argument("--actor_dist_thr", type=float, default=0.12, help="relink normalized center-distance gate")
    p.add_argument("--actor_size_thr", type=float, default=0.60,
                   help="relink size similarity gate: min(area_ratio,1/area_ratio)")
    p.add_argument("--actor_dup_iou", type=float, default=0.90, help="same-frame duplicate IoU threshold")
    p.add_argument("--actor_takeover_iou", type=float, default=0.25, help="takeover IoU threshold")
    p.add_argument("--viz_mode", default="debug", choices=["debug", "clean"], help="overlay verbosity mode")

    p.add_argument("--max_frames", type=int, default=0, help="0=all")
    p.add_argument("--no_show", action="store_true", default=True, help="do not cv2.imshow (default: on)")
    p.add_argument("--show", dest="no_show", action="store_false", help="enable cv2.imshow preview")
    return p


def resolve_source(args) -> SourceInfo:
    source = args.source.strip()
    video_id = args.video_id.strip()
    if not source:
        if not video_id:
            raise ValueError("Either --source or --video_id must be provided.")
        source = str(Path("data/videos") / f"{video_id}.mp4")
    elif not video_id:
        stem = Path(source).stem
        m = re.search(r"(E\d{2}_\d{3})", stem)
        video_id = m.group(1) if m else stem
    return SourceInfo(source=source, video_id=video_id)


def _parse_version_from_name(path: Path) -> int:
    m = re.search(r"_v(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def _extract_vertices_original(obj: dict, roi_path: Path) -> List[List[float]]:
    vertices = obj.get("vertices_px", [])
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError(f"Invalid vertices_px in {roi_path}")

    labeled_on = str(obj.get("labeled_on", "")).strip().lower()
    disp_scale_used = obj.get("disp_scale_used", None)
    if labeled_on == "disp" and disp_scale_used is not None:
        scale = float(disp_scale_used)
        if scale <= 0:
            raise ValueError(f"Invalid disp_scale_used in {roi_path}: {disp_scale_used}")
        out = []
        for v in vertices:
            x = float(v[0]) / scale
            y = float(v[1]) / scale
            out.append([x, y])
        return out

    out = []
    for v in vertices:
        out.append([float(v[0]), float(v[1])])
    return out


def _read_roi_obj(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read ROI json {path}: {e}")


def _collect_roi_candidates(roi_root: Path) -> List[Path]:
    if not roi_root.exists():
        return []
    return sorted([p for p in roi_root.rglob("*.json") if p.is_file()])


def load_roi_config(args, source_video_id: str) -> ROIConfig:
    if args.roi_path.strip():
        roi_path = Path(args.roi_path.strip())
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI json not found: {roi_path}")
        obj = _read_roi_obj(roi_path)
    else:
        roi_video_id = args.roi_video_id.strip() or source_video_id
        if not roi_video_id:
            raise ValueError("--roi_video_id is required when source video_id is unavailable.")

        roi_root = Path("configs/rois") / roi_video_id
        candidates = _collect_roi_candidates(roi_root)
        if not candidates:
            raise FileNotFoundError(f"No ROI json found under: {roi_root}")

        req_roi_id = args.roi_id.strip()
        req_ver = int(args.roi_version)

        # Keep auto-find permissive: folder namespace is authoritative.
        # json.video_id mismatch should not reject an otherwise valid ROI file.
        matched: List[tuple[int, int, float, Path, dict]] = []
        for p in candidates:
            obj_try = _read_roi_obj(p)
            obj_video_id = str(obj_try.get("video_id", "")).strip()
            obj_roi_id = str(obj_try.get("roi_id", "")).strip()
            obj_ver = int(obj_try.get("roi_version", _parse_version_from_name(p)))
            if req_roi_id and obj_roi_id and obj_roi_id != req_roi_id:
                continue
            if req_roi_id and (not obj_roi_id):
                stem = p.stem.lower()
                if req_roi_id.lower() not in stem:
                    continue
            if req_ver > 0 and obj_ver != req_ver:
                continue
            vid_match = 1 if (obj_video_id and obj_video_id == roi_video_id) else 0
            matched.append((obj_ver, vid_match, p.stat().st_mtime, p, obj_try))

        if not matched:
            available = [str(x) for x in candidates[:20]]
            raise FileNotFoundError(
                f"ROI auto-find failed under {roi_root} "
                f"(roi_id={req_roi_id or '*'}, roi_version={req_ver}). "
                f"Available examples: {available}"
            )

        matched.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        _, _, _, roi_path, obj = matched[0]

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

    vertices_file = []
    for v in obj.get("vertices_px", []):
        vertices_file.append([float(v[0]), float(v[1])])

    vertices_orig = _extract_vertices_original(obj, roi_path)
    return ROIConfig(
        path=roi_path,
        video_id=str(obj.get("video_id", "")).strip() or (args.roi_video_id.strip() or source_video_id),
        roi_id=str(obj.get("roi_id", "")).strip() or args.roi_id.strip(),
        roi_version=int(obj.get("roi_version", _parse_version_from_name(roi_path))),
        image_width=iw,
        image_height=ih,
        vertices_file=vertices_file,
        vertices_orig=vertices_orig,
        labeled_on=str(obj.get("labeled_on", "")).strip(),
        disp_scale_used=float(obj["disp_scale_used"]) if obj.get("disp_scale_used", None) is not None else None,
    )


def get_versions(cv2_module, ultralytics_module, torch_module, boxmot_module) -> dict:
    return {
        "python": platform.python_version(),
        "cv2": getattr(cv2_module, "__version__", None),
        "ultralytics": getattr(ultralytics_module, "__version__", None),
        "torch": getattr(torch_module, "__version__", None) if torch_module is not None else None,
        "boxmot": getattr(boxmot_module, "__version__", None),
    }


def scale_vertices_to_frame(vertices_orig, src_w: int, src_h: int, dst_w: int, dst_h: int):
    import numpy as np

    sx = float(dst_w) / float(max(1, src_w))
    sy = float(dst_h) / float(max(1, src_h))
    pts = []
    for v in vertices_orig:
        x = int(round(float(v[0]) * sx))
        y = int(round(float(v[1]) * sy))
        x = max(0, min(dst_w - 1, x))
        y = max(0, min(dst_h - 1, y))
        pts.append([x, y])
    return np.asarray(pts, dtype=np.int32)


def build_roi_mask(np_module, cv2_module, frame_w: int, frame_h: int, roi_poly):
    mask = np_module.zeros((frame_h, frame_w), dtype=np_module.uint8)
    cv2_module.fillPoly(mask, [roi_poly], 1)
    return mask


def bbox_ioa_roi(mask, bbox_xyxy: List[float]) -> float:
    h, w = mask.shape[:2]
    x1 = max(0, min(w, int(round(float(bbox_xyxy[0])))))
    y1 = max(0, min(h, int(round(float(bbox_xyxy[1])))))
    x2 = max(0, min(w, int(round(float(bbox_xyxy[2])))))
    y2 = max(0, min(h, int(round(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = float((x2 - x1) * (y2 - y1))
    inside = float(mask[y1:y2, x1:x2].sum())
    return inside / area if area > 0 else 0.0


def compute_footpoint(
    bbox: List[float],
    kps_xy,
    kps_conf,
    kp_conf: float,
) -> Tuple[float, float, str, List[Dict[str, Any]]]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_pt = (0.5 * (x1 + x2), y2, "BBOX", [])

    if kps_xy is None:
        return bbox_pt

    def _get_kp(idx: int) -> Optional[Tuple[float, float, float]]:
        try:
            if idx >= len(kps_xy):
                return None
            pt = kps_xy[idx]
            x = float(pt[0])
            y = float(pt[1])
        except Exception:
            return None

        conf_v = 1.0
        if kps_conf is not None:
            try:
                if idx >= len(kps_conf):
                    return None
                conf_v = float(kps_conf[idx])
            except Exception:
                return None
        if conf_v < float(kp_conf):
            return None
        return x, y, conf_v

    lank = _get_kp(KP_LANKLE)
    rank = _get_kp(KP_RANKLE)
    ankles: List[Dict[str, Any]] = []
    if lank is not None:
        ankles.append({"name": "LA", "xy": [float(lank[0]), float(lank[1])], "conf": float(lank[2])})
    if rank is not None:
        ankles.append({"name": "RA", "xy": [float(rank[0]), float(rank[1])], "conf": float(rank[2])})
    if len(ankles) >= 2:
        ax = 0.5 * (float(ankles[0]["xy"][0]) + float(ankles[1]["xy"][0]))
        ay = 0.5 * (float(ankles[0]["xy"][1]) + float(ankles[1]["xy"][1]))
        return float(ax), float(ay), "ANKLE", ankles
    if len(ankles) == 1:
        axy = ankles[0]["xy"]
        return float(axy[0]), float(axy[1]), "ANKLE", ankles

    lhip = _get_kp(KP_LHIP)
    rhip = _get_kp(KP_RHIP)
    hips = [p for p in (lhip, rhip) if p is not None]
    if len(hips) == 2:
        return float((hips[0][0] + hips[1][0]) * 0.5), float((hips[0][1] + hips[1][1]) * 0.5), "HIP", []
    if len(hips) == 1:
        return float(hips[0][0]), float(hips[0][1]), "HIP", []
    return bbox_pt


def point_in_roi_mask(mask, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return False
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    return bool(mask[yi, xi] == 1)


def _foot_reason_tag(foot_method: str) -> str:
    m = str(foot_method).strip().upper()
    if m == "ANKLE":
        return "ANKLE"
    if m == "HIP":
        return "HIP"
    return "BBOX_FOOT"


def parse_tracker_outputs(tracks_out) -> List[Dict[str, Any]]:
    import numpy as np

    if tracks_out is None:
        return []
    if isinstance(tracks_out, list):
        arr = np.array(tracks_out) if len(tracks_out) > 0 else np.zeros((0, 0))
    else:
        arr = np.array(tracks_out)
    if arr.size == 0:
        return []

    out: List[Dict[str, Any]] = []
    for row in arr:
        r = [float(x) for x in row.tolist()]
        if len(r) < 5:
            continue
        x1, y1, x2, y2 = r[0], r[1], r[2], r[3]

        cand_ids = []
        for idx in range(4, min(len(r), 9)):
            v = r[idx]
            if abs(v - round(v)) < 1e-3 and v >= 0:
                cand_ids.append((idx, int(round(v))))
        if not cand_ids:
            continue
        tid_idx, tid = cand_ids[0]

        conf = None
        cls = None
        for j in range(4, len(r)):
            if j == tid_idx:
                continue
            v = r[j]
            if 0.0 <= v <= 1.0:
                conf = float(v)
                break
        for j in range(4, len(r)):
            if j == tid_idx:
                continue
            v = r[j]
            if abs(v - round(v)) < 1e-3 and 0 <= v <= 80:
                cls = int(round(v))
                break

        out.append({"track_id": tid, "bbox_xyxy": [x1, y1, x2, y2], "conf": conf, "cls": cls})
    return out


def make_boxmot_tracker(
    tracker_name: str,
    reid_weights: str,
    device: str,
    fp16: bool,
    per_class: bool,
    track_max_age: int = 0,
    track_min_hits: int = 0,
    track_max_obs: int = 0,
):
    import inspect
    import boxmot  # type: ignore

    cls = None
    if tracker_name.lower() == "deepocsort":
        cls = getattr(boxmot, "DeepOCSORT", None) or getattr(boxmot, "DeepOcSort", None)
    elif tracker_name.lower() == "strongsort":
        cls = getattr(boxmot, "StrongSORT", None) or getattr(boxmot, "StrongSort", None)
    if cls is None:
        raise RuntimeError(f"BoxMOT tracker class not found for '{tracker_name}'.")

    sig = inspect.signature(cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    kwargs = {}
    for name in sig.parameters.keys():
        if name == "self":
            continue
        if name in ("model_weights", "reid_weights", "weights", "model"):
            kwargs[name] = reid_weights
        elif name in ("device", "dev"):
            kwargs[name] = device
        elif name in ("half", "fp16"):
            kwargs[name] = fp16
        elif name in ("per_class",):
            kwargs[name] = per_class
    if int(track_max_age) > 0 and ("max_age" in sig.parameters or has_var_kw):
        kwargs["max_age"] = int(track_max_age)
    if int(track_min_hits) > 0 and ("min_hits" in sig.parameters or has_var_kw):
        kwargs["min_hits"] = int(track_min_hits)
    if int(track_max_obs) > 0 and ("max_obs" in sig.parameters or has_var_kw):
        kwargs["max_obs"] = int(track_max_obs)
    try:
        return cls(**kwargs)
    except TypeError:
        return cls(reid_weights, device)


def bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def box_area(b: List[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in b]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def inter_area_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def iou_xyxy(a: List[float], b: List[float]) -> float:
    inter = inter_area_xyxy(a, b)
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def ioa_xyxy(a: List[float], b: List[float]) -> float:
    # Intersection-over-area(A)
    aa = box_area(a)
    if aa <= 0:
        return 0.0
    inter = inter_area_xyxy(a, b)
    return inter / aa


def dedup_detections_greedy(
    dets: List[List[float]],
    *,
    iou_thr: float,
    contain_ioa_thr: float,
    keep_mode: str,
) -> List[List[float]]:
    if len(dets) <= 1:
        return dets

    by_cls: Dict[int, List[List[float]]] = {}
    for d in dets:
        cls_id = int(round(float(d[5]))) if len(d) > 5 else 0
        by_cls.setdefault(cls_id, []).append(d)

    kept_all: List[List[float]] = []
    for cls_id in sorted(by_cls.keys()):
        items = by_cls[cls_id]
        if keep_mode == "max_area":
            items = sorted(items, key=lambda d: (-box_area(d[:4]), -float(d[4])))
        else:
            items = sorted(items, key=lambda d: (-float(d[4]), -box_area(d[:4])))

        kept_cls: List[List[float]] = []
        for cand in items:
            cbox = cand[:4]
            is_dup = False
            for kept in kept_cls:
                kbox = kept[:4]
                if iou_xyxy(cbox, kbox) >= float(iou_thr):
                    is_dup = True
                    break
                if ioa_xyxy(cbox, kbox) >= float(contain_ioa_thr):
                    is_dup = True
                    break
            if not is_dup:
                kept_cls.append(cand)
        kept_all.extend(kept_cls)

    return kept_all


def track_cleanup_greedy(
    tracks: List[Dict[str, Any]],
    *,
    merge_iou_thr: float,
    contain_ioa_thr: float,
) -> List[Dict[str, Any]]:
    if len(tracks) <= 1:
        return tracks

    def _rank(tr: Dict[str, Any]):
        supported = 1 if bool(tr.get("track_supported", False)) else 0
        conf = tr.get("conf", None)
        has_conf = 1 if conf is not None else 0
        conf_val = float(conf) if conf is not None else -1.0
        area = float(box_area(tr["bbox_xyxy"]))
        return (supported, has_conf, conf_val, area)

    sorted_tracks = sorted(tracks, key=_rank, reverse=True)
    kept: List[Dict[str, Any]] = []
    for tr in sorted_tracks:
        tb = tr["bbox_xyxy"]
        is_dup = False
        for kr in kept:
            kb = kr["bbox_xyxy"]
            if iou_xyxy(tb, kb) >= float(merge_iou_thr):
                is_dup = True
                break
            if ioa_xyxy(tb, kb) >= float(contain_ioa_thr):
                is_dup = True
                break
            if ioa_xyxy(kb, tb) >= float(contain_ioa_thr):
                is_dup = True
                break
        if not is_dup:
            kept.append(tr)
    return kept


def bbox_center_xyxy(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def bbox_area_xyxy(b: List[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in b]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_size_similarity(a: List[float], b: List[float]) -> float:
    aa = bbox_area_xyxy(a)
    bb = bbox_area_xyxy(b)
    if aa <= 0 or bb <= 0:
        return 0.0
    r = aa / bb
    return min(r, 1.0 / r)


@dataclass
class ActorSnapshot:
    actor_id: int
    last_bbox: Optional[List[float]] = None
    last_seen_frame: int = -1
    last_seen_ts: float = -1.0
    prev_center: Optional[Tuple[float, float]] = None
    prev_ts: Optional[float] = None
    vel_xy: Tuple[float, float] = (0.0, 0.0)  # px/sec
    last_track_id: Optional[int] = None


class ActorManager:
    def __init__(
        self,
        *,
        enable: bool,
        enable_takeover: bool,
        max_gap_sec: float,
        iou_thr: float,
        dist_thr: float,
        size_thr: float,
        dup_iou: float,
        takeover_iou: float,
        frame_w: int,
        frame_h: int,
        logger: logging.Logger,
    ):
        self.enable = bool(enable)
        self.enable_takeover = bool(enable_takeover)
        self.max_gap_sec = float(max_gap_sec)
        self.iou_thr = float(iou_thr)
        self.dist_thr = float(dist_thr)
        self.size_thr = float(size_thr)
        self.dup_iou = float(dup_iou)
        self.takeover_iou = float(takeover_iou)
        self.diag = max(1e-6, math.hypot(float(frame_w), float(frame_h)))
        self.logger = logger

        self.next_actor_id = 1
        self.track_to_actor: Dict[int, int] = {}
        self.track_last_seen_ts: Dict[int, float] = {}
        self.actors: Dict[int, ActorSnapshot] = {}

        self.frame_actor_to_tid: Dict[int, int] = {}
        self.frame_actor_to_bbox: Dict[int, List[float]] = {}
        self.frame_actor_supported: Dict[int, bool] = {}

    def begin_frame(self):
        self.frame_actor_to_tid.clear()
        self.frame_actor_to_bbox.clear()
        self.frame_actor_supported.clear()

    def end_frame(self, seen_tids: set[int], ts_sec: float):
        if not self.enable:
            return
        stale = []
        for tid, last_ts in self.track_last_seen_ts.items():
            if tid in seen_tids:
                continue
            if float(ts_sec) - float(last_ts) > max(0.5, 2.0 * self.max_gap_sec):
                stale.append(tid)
        for tid in stale:
            self.track_last_seen_ts.pop(tid, None)
            self.track_to_actor.pop(tid, None)

    def _new_actor(self) -> int:
        aid = int(self.next_actor_id)
        self.next_actor_id += 1
        self.actors[aid] = ActorSnapshot(actor_id=aid)
        return aid

    def _predict_bbox(self, actor: ActorSnapshot, ts_sec: float) -> Optional[List[float]]:
        if actor.last_bbox is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in actor.last_bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx, cy = bbox_center_xyxy(actor.last_bbox)
        dt = max(0.0, float(ts_sec) - float(actor.last_seen_ts))
        vx, vy = actor.vel_xy
        pcx = cx + vx * dt
        pcy = cy + vy * dt
        return [pcx - 0.5 * w, pcy - 0.5 * h, pcx + 0.5 * w, pcy + 0.5 * h]

    def _update_actor(
        self,
        actor_id: int,
        track_id: int,
        bbox: List[float],
        frame_idx: int,
        ts_sec: float,
        track_supported: bool,
    ):
        actor = self.actors.get(actor_id)
        if actor is None:
            actor = ActorSnapshot(actor_id=actor_id)
            self.actors[actor_id] = actor
        cx, cy = bbox_center_xyxy(bbox)
        if actor.prev_center is not None and actor.prev_ts is not None:
            dt = float(ts_sec) - float(actor.prev_ts)
            if dt > 1e-6:
                vx = (cx - actor.prev_center[0]) / dt
                vy = (cy - actor.prev_center[1]) / dt
                actor.vel_xy = (vx, vy)
        actor.prev_center = (cx, cy)
        actor.prev_ts = float(ts_sec)
        actor.last_bbox = [float(v) for v in bbox]
        actor.last_seen_frame = int(frame_idx)
        actor.last_seen_ts = float(ts_sec)
        actor.last_track_id = int(track_id)

        self.track_to_actor[int(track_id)] = int(actor_id)
        self.track_last_seen_ts[int(track_id)] = float(ts_sec)
        self.frame_actor_to_tid[int(actor_id)] = int(track_id)
        self.frame_actor_to_bbox[int(actor_id)] = [float(v) for v in bbox]
        self.frame_actor_supported[int(actor_id)] = bool(track_supported)

    def assign(self, track_id: int, bbox: List[float], frame_idx: int, ts_sec: float, track_supported: bool) -> int:
        tid = int(track_id)
        bb = [float(v) for v in bbox]

        if not self.enable:
            return tid

        # existing mapping
        if tid in self.track_to_actor:
            aid = int(self.track_to_actor[tid])
            frame_tid = self.frame_actor_to_tid.get(aid, None)
            if frame_tid is not None and frame_tid != tid:
                frame_bbox = self.frame_actor_to_bbox.get(aid, bb)
                frame_supported = bool(self.frame_actor_supported.get(aid, False))
                dup_iou = bbox_iou_xyxy(frame_bbox, bb)
                pcx, pcy = bbox_center_xyxy(frame_bbox)
                ncx, ncy = bbox_center_xyxy(bb)
                takeover_dist = float(math.hypot(pcx - ncx, pcy - ncy) / self.diag)
                if (
                    self.enable_takeover
                    and (not frame_supported)
                    and bool(track_supported)
                    and ((dup_iou >= self.takeover_iou) or (takeover_dist <= self.dist_thr))
                ):
                    self.logger.info(
                        "takeover: actor=%d old_tid=%d -> new_tid=%d iou=%.2f dist=%.3f",
                        aid,
                        frame_tid,
                        tid,
                        dup_iou,
                        takeover_dist,
                    )
                elif dup_iou >= self.dup_iou:
                    self.logger.info("duplicate: tid1=%d tid2=%d -> actor=%d", frame_tid, tid, aid)
                else:
                    # avoid collapsing two visible persons
                    aid = self._new_actor()
            self._update_actor(aid, tid, bb, frame_idx, ts_sec, track_supported=bool(track_supported))
            return aid

        # new track_id: relink or new actor
        best = None
        for aid, actor in self.actors.items():
            if actor.last_bbox is None:
                continue
            gap = float(ts_sec) - float(actor.last_seen_ts)
            if gap < 0.0 or gap > self.max_gap_sec:
                continue

            pred = self._predict_bbox(actor, ts_sec)
            if pred is None:
                continue
            iou = float(bbox_iou_xyxy(pred, bb))
            pcx, pcy = bbox_center_xyxy(pred)
            ncx, ncy = bbox_center_xyxy(bb)
            dist = float(math.hypot(pcx - ncx, pcy - ncy) / self.diag)
            size_sim = float(bbox_size_similarity(pred, bb))

            if not ((iou >= self.iou_thr) or (dist <= self.dist_thr)):
                continue
            if size_sim < self.size_thr:
                continue

            assigned_tid = self.frame_actor_to_tid.get(aid, None)
            is_dup = False
            is_takeover = False
            if assigned_tid is not None and assigned_tid != tid:
                frame_bbox = self.frame_actor_to_bbox.get(aid, bb)
                frame_supported = bool(self.frame_actor_supported.get(aid, False))
                dup_iou = float(bbox_iou_xyxy(frame_bbox, bb))
                fcx, fcy = bbox_center_xyxy(frame_bbox)
                ncx2, ncy2 = bbox_center_xyxy(bb)
                takeover_dist = float(math.hypot(fcx - ncx2, fcy - ncy2) / self.diag)
                if (
                    self.enable_takeover
                    and (not frame_supported)
                    and bool(track_supported)
                    and ((dup_iou >= self.takeover_iou) or (takeover_dist <= self.dist_thr))
                ):
                    is_takeover = True
                elif dup_iou >= self.dup_iou:
                    is_dup = True
                else:
                    continue

            gap_norm = gap / max(1e-6, self.max_gap_sec)
            cost = (1.0 - iou) + dist + (1.0 - size_sim) + 0.05 * gap_norm
            rec = {
                "actor_id": int(aid),
                "cost": float(cost),
                "iou": float(iou),
                "dist": float(dist),
                "gap": float(gap),
                "is_dup": bool(is_dup),
                "is_takeover": bool(is_takeover),
                "assigned_tid": int(assigned_tid) if assigned_tid is not None else None,
                "prev_tid": int(actor.last_track_id) if actor.last_track_id is not None else None,
            }
            if best is None or rec["cost"] < best["cost"]:
                best = rec

        if best is None:
            aid = self._new_actor()
        else:
            aid = int(best["actor_id"])
            if best.get("is_takeover", False) and best["assigned_tid"] is not None:
                self.logger.info(
                    "takeover: actor=%d old_tid=%d -> new_tid=%d iou=%.2f dist=%.3f",
                    aid,
                    best["assigned_tid"],
                    tid,
                    best["iou"],
                    best["dist"],
                )
            elif best["is_dup"] and best["assigned_tid"] is not None:
                self.logger.info("duplicate: tid1=%d tid2=%d -> actor=%d", best["assigned_tid"], tid, aid)
            else:
                self.logger.info(
                    "relink: new_tid=%d -> actor=%d (prev_tid=%s) iou=%.2f dist=%.3f gap=%.2f",
                    tid,
                    aid,
                    str(best["prev_tid"]),
                    best["iou"],
                    best["dist"],
                    best["gap"],
                )

        self._update_actor(aid, tid, bb, frame_idx, ts_sec, track_supported=bool(track_supported))
        return aid


def color_for_state(state: str) -> tuple[int, int, int]:
    if state == STATE_IN:
        return (0, 0, 255)  # red
    if state == STATE_CAND:
        return (0, 165, 255)  # orange
    return (60, 220, 60)  # green


def _reset_candidate(st: TrackState):
    st.dwell_frames = 0
    st.enter_streak = 0
    st.exit_streak = 0
    st.candidate_enter_frame = None
    st.candidate_enter_ts = None
    st.confirm_frame = None
    st.confirm_ts = None
    st.confirm_reason = None
    st.ioa_at_confirm = None
    st.max_ioa_current = 0.0
    st.current_event = None


def _start_candidate(st: TrackState, frame_idx: int, ts_sec: float, ioa: float):
    st.state = STATE_CAND
    st.candidate_enter_frame = int(frame_idx)
    st.candidate_enter_ts = float(ts_sec)
    st.dwell_frames = 1
    st.enter_streak = 1
    st.exit_streak = 0
    st.max_ioa_current = float(ioa)
    st.confirm_reason = None
    st.current_event = None


def _confirm_intrusion(
    st: TrackState,
    actor_id: int,
    track_id: int,
    frame_idx: int,
    ts_sec: float,
    ioa: float,
    reason: str,
):
    st.state = STATE_IN
    st.actor_id = int(actor_id)
    st.last_track_id = int(track_id)
    st.confirm_frame = int(frame_idx)
    st.confirm_ts = float(ts_sec)
    st.confirm_reason = str(reason)
    st.ioa_at_confirm = float(ioa)
    st.max_ioa_current = max(float(st.max_ioa_current), float(ioa))
    st.exit_streak = 0
    st.current_event = {
        "actor_id": int(actor_id),
        "track_id": int(track_id),
        "enter_frame": int(st.candidate_enter_frame) if st.candidate_enter_frame is not None else int(frame_idx),
        "enter_ts": float(st.candidate_enter_ts) if st.candidate_enter_ts is not None else float(ts_sec),
        "confirm_frame": int(frame_idx),
        "confirm_ts": float(ts_sec),
        "exit_frame": None,
        "exit_ts": None,
        "duration_sec": None,
        "confirm_reason": str(reason),
        "max_ioa": float(st.max_ioa_current),
        "ioa_at_confirm": float(ioa),
    }


def _finalize_event_if_needed(st: TrackState, frame_idx: Optional[int], ts_sec: Optional[float]) -> Optional[Dict[str, Any]]:
    if st.current_event is None:
        return None
    ev = dict(st.current_event)
    if frame_idx is not None:
        ev["exit_frame"] = int(frame_idx)
    else:
        ev["exit_frame"] = None
    if ts_sec is not None:
        ev["exit_ts"] = float(ts_sec)
    else:
        ev["exit_ts"] = None

    confirm_ts = float(ev.get("confirm_ts") or 0.0)
    if ev["exit_ts"] is not None:
        ev["duration_sec"] = max(0.0, float(ev["exit_ts"]) - confirm_ts)
    else:
        ev["duration_sec"] = None
    ev["max_ioa"] = max(float(ev.get("max_ioa", 0.0)), float(st.max_ioa_current))
    st.current_event = None
    return ev


def _confirm_reason_hook(
    args,
    ioa: float,
    foot_in_roi: bool,
    foot_method: str,
    enter_streak: int,
    dwell_frames: int,
    dwell_frames_req: int,
) -> Optional[str]:
    bbox_ok = bool(
        ioa >= float(args.in_ioa_thr) and enter_streak >= int(args.enter_n) and dwell_frames >= dwell_frames_req
    )
    if str(args.algo) != "_02_use_keypoint":
        if bbox_ok:
            return "BBOX_IOA"
        return None

    foot_ok = bool(
        foot_in_roi
        and ioa >= float(args.foot_confirm_ioa_thr)
        and enter_streak >= int(args.enter_n)
        and dwell_frames >= dwell_frames_req
    )
    if bbox_ok and foot_ok:
        return f"{_foot_reason_tag(foot_method)}+BBOX"
    if foot_ok:
        return _foot_reason_tag(foot_method)
    if bbox_ok:
        return "BBOX_IOA"
    return None


def update_track_state_seen(
    args,
    st: TrackState,
    actor_id: int,
    track_id: int,
    track_supported: bool,
    track_conf: Optional[float],
    frame_idx: int,
    ts_sec: float,
    ioa: float,
    foot_in_roi: bool,
    foot_xy: Optional[List[float]],
    foot_method: str,
    foot_ankles: Optional[List[Dict[str, Any]]],
    dwell_frames_req: int,
    events: List[Dict[str, Any]],
):
    st.actor_id = int(actor_id)
    st.last_track_id = int(track_id)
    st.last_track_supported = bool(track_supported)
    st.last_conf = float(track_conf) if track_conf is not None else None
    st.last_seen_frame = int(frame_idx)
    st.last_seen_ts = float(ts_sec)
    st.last_ioa = float(ioa)
    st.last_foot_in_roi = bool(foot_in_roi)
    st.last_foot_xy = [float(foot_xy[0]), float(foot_xy[1])] if foot_xy is not None else None
    st.last_foot_method = str(foot_method)
    st.last_ankles = list(foot_ankles) if foot_ankles is not None else []
    if st.current_event is not None:
        st.current_event["actor_id"] = int(actor_id)
        st.current_event["track_id"] = int(track_id)

    use_foot = str(args.algo) == "_02_use_keypoint"
    in_cand = float(ioa) >= float(args.cand_ioa_thr)
    if use_foot and bool(foot_in_roi) and float(ioa) >= float(args.foot_ioa_thr):
        in_cand = True

    in_confirm_zone = float(ioa) >= float(args.in_ioa_thr)
    if use_foot and bool(foot_in_roi) and float(ioa) >= float(args.foot_confirm_ioa_thr):
        in_confirm_zone = True

    if st.state == STATE_OUT:
        if in_cand:
            _start_candidate(st, frame_idx, ts_sec, ioa)
            if not in_confirm_zone:
                st.enter_streak = 0
        return

    if st.state == STATE_CAND:
        if in_cand:
            st.dwell_frames += 1
            st.max_ioa_current = max(float(st.max_ioa_current), float(ioa))
            if in_confirm_zone:
                st.enter_streak += 1
            else:
                st.enter_streak = 0
            st.exit_streak = 0

            reason = _confirm_reason_hook(
                args,
                ioa,
                foot_in_roi,
                foot_method,
                st.enter_streak,
                st.dwell_frames,
                dwell_frames_req,
            )
            if reason is not None:
                _confirm_intrusion(st, actor_id, track_id, frame_idx, ts_sec, ioa, reason)
        else:
            st.exit_streak += 1
            if st.exit_streak >= int(args.exit_n):
                st.state = STATE_OUT
                _reset_candidate(st)
        return

    if st.state == STATE_IN:
        st.max_ioa_current = max(float(st.max_ioa_current), float(ioa))
        if st.current_event is not None:
            st.current_event["max_ioa"] = max(float(st.current_event.get("max_ioa", 0.0)), float(ioa))

        if in_cand:
            st.exit_streak = 0
        else:
            st.exit_streak += 1
            if st.exit_streak >= int(args.exit_n):
                ev = _finalize_event_if_needed(st, frame_idx, ts_sec)
                if ev is not None:
                    events.append(ev)
                st.state = STATE_OUT
                _reset_candidate(st)
        return


def update_track_state_missing(
    args,
    st: TrackState,
    frame_idx: int,
    ts_sec: float,
    grace_frames: int,
    events: List[Dict[str, Any]],
):
    if st.state not in (STATE_CAND, STATE_IN):
        return
    miss_frames = max(0, int(frame_idx) - int(st.last_seen_frame))
    if miss_frames <= int(grace_frames):
        return
    if st.state == STATE_IN:
        if str(args.missing_policy) == "hold_in":
            return
        ev = _finalize_event_if_needed(st, frame_idx, ts_sec)
        if ev is not None:
            events.append(ev)
    st.state = STATE_OUT
    _reset_candidate(st)


def draw_hud_top_right(
    cv2_module,
    img,
    lines: List[str],
    title: str = "Active CANDIDATE/IN",
):
    h, w = img.shape[:2]
    panel_w = max(360, min(920, w - 20))
    panel_x1 = max(10, w - panel_w - 10)
    panel_y1 = 10
    line_h = 22
    panel_h = 36 + max(1, len(lines)) * line_h
    panel_x2 = min(w - 10, panel_x1 + panel_w)
    panel_y2 = min(h - 10, panel_y1 + panel_h)

    overlay = img.copy()
    cv2_module.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2_module.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2_module.rectangle(img, (panel_x1, panel_y1), (panel_x2, panel_y2), (200, 200, 200), 1)

    cv2_module.putText(
        img,
        title,
        (panel_x1 + 10, panel_y1 + 22),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )

    if not lines:
        lines = ["(none)"]
    y = panel_y1 + 22 + line_h
    for ln in lines:
        cv2_module.putText(
            img,
            ln,
            (panel_x1 + 10, y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2_module.LINE_AA,
        )
        y += line_h


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.pose_draw is None:
        args.pose_draw = bool(str(args.algo) == "_02_use_keypoint")

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

    cv2 = None
    np = None
    ultralytics = None
    torch = None
    boxmot = None
    YOLO = None

    try:
        import cv2 as _cv2
        import numpy as _np

        cv2 = _cv2
        np = _np
    except Exception as e:
        logger.exception("Failed to import cv2/numpy: %s", e)
        return 2

    try:
        import ultralytics as _ultralytics
        from ultralytics import YOLO as _YOLO

        ultralytics = _ultralytics
        YOLO = _YOLO
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

    in_path = Path(src.source)
    if not in_path.exists():
        logger.error("Input file not found: %s", in_path)
        return 2

    try:
        roi_cfg = load_roi_config(args, source_video_id=src.video_id)
    except Exception as e:
        logger.exception("Failed to load ROI: %s", e)
        return 2
    if not args.roi_path.strip():
        roi_namespace = args.roi_video_id.strip() or src.video_id
        if roi_cfg.video_id and roi_namespace and roi_cfg.video_id != roi_namespace:
            logger.warning(
                "ROI json.video_id(%s) != roi namespace(%s); accepted by permissive auto-find.",
                roi_cfg.video_id,
                roi_namespace,
            )

    logger.info("Loading detector: %s", args.det_model)
    det_model = YOLO(args.det_model)

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
            logger.error("Failed to read first frame for shape.")
            return 2
        frame_h, frame_w = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    dwell_frames_req = max(1, int(math.ceil(float(args.dwell_sec) * float(fps))))
    grace_frames = max(1, int(math.ceil(float(args.grace_sec) * float(fps))))

    track_max_age = int(args.track_max_age)
    track_min_hits = int(args.track_min_hits)
    track_max_obs = int(args.track_max_obs)
    if track_max_age == 0:
        track_max_age = int(grace_frames)
        if track_max_obs == 0:
            track_max_obs = int(track_max_age)
    elif track_max_obs == 0:
        track_max_obs = int(track_max_age)

    tracker_device = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    logger.info("Tracker tune: max_age=%d, min_hits=%d, max_obs=%d", track_max_age, track_min_hits, track_max_obs)
    logger.info("Creating BoxMOT tracker: %s (reid=%s, device=%s)", args.tracker, args.reid_model, tracker_device)
    try:
        tracker = make_boxmot_tracker(
            tracker_name=args.tracker,
            reid_weights=args.reid_model,
            device=tracker_device,
            fp16=bool(args.fp16),
            per_class=bool(args.per_class),
            track_max_age=track_max_age,
            track_min_hits=track_min_hits,
            track_max_obs=track_max_obs,
        )
    except Exception as e:
        logger.exception("Failed to create BoxMOT tracker: %s", e)
        return 2

    roi_poly = scale_vertices_to_frame(
        roi_cfg.vertices_orig,
        src_w=roi_cfg.image_width,
        src_h=roi_cfg.image_height,
        dst_w=frame_w,
        dst_h=frame_h,
    )
    roi_mask = build_roi_mask(np, cv2, frame_w, frame_h, roi_poly)

    out_overlay = run_paths.out_dir / f"{args.out_base}_intrusion_overlay.mp4"
    out_events = run_paths.out_dir / f"{args.out_base}_intrusion_events.json"

    vw = cv2.VideoWriter(
        str(out_overlay),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not vw.isOpened():
        logger.error("Failed to open VideoWriter: %s", out_overlay)
        return 2

    versions = get_versions(cv2, ultralytics, torch, boxmot)
    params = vars(args).copy()
    params.update(
        {
            "source": str(in_path),
            "video_id": src.video_id,
            "roi_path_resolved": str(roi_cfg.path),
            "fps_used": fps,
            "frame_size": {"width": frame_w, "height": frame_h},
            "dwell_frames_req": dwell_frames_req,
            "grace_frames": grace_frames,
            "track_tune_resolved": {
                "max_age": int(track_max_age),
                "min_hits": int(track_min_hits),
                "max_obs": int(track_max_obs),
            },
            "versions": versions,
        }
    )
    (run_paths.out_dir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Input: %s (%dx%d fps=%.3f)", in_path, frame_w, frame_h, fps)
    logger.info("ROI: %s (id=%s v=%d labeled_on=%s)", roi_cfg.path, roi_cfg.roi_id, roi_cfg.roi_version, roi_cfg.labeled_on)
    logger.info("Output overlay: %s", out_overlay)
    logger.info("Output events : %s", out_events)
    logger.info(
        "Algo=%s cand_ioa_thr=%.3f in_ioa_thr=%.3f enter_n=%d exit_n=%d dwell_sec=%.2f grace_sec=%.2f missing_policy=%s",
        args.algo,
        args.cand_ioa_thr,
        args.in_ioa_thr,
        args.enter_n,
        args.exit_n,
        args.dwell_sec,
        args.grace_sec,
        str(args.missing_policy),
    )
    logger.info(
        "Actor layer: enable=%s use_for_fsm=%s max_gap_sec=%.2f iou_thr=%.2f dist_thr=%.3f size_thr=%.2f dup_iou=%.2f",
        bool(args.actor_enable),
        bool(args.actor_use_for_fsm),
        float(args.actor_max_gap_sec),
        float(args.actor_iou_thr),
        float(args.actor_dist_thr),
        float(args.actor_size_thr),
        float(args.actor_dup_iou),
    )
    logger.info(
        "Track cleanup: enable=%s support_iou=%.2f merge_iou=%.2f contain_ioa=%.2f viz_mode=%s",
        bool(args.track_cleanup_enable),
        float(args.track_support_iou),
        float(args.track_merge_iou),
        float(args.track_contain_ioa),
        str(args.viz_mode),
    )

    actor_manager = ActorManager(
        enable=bool(args.actor_enable),
        enable_takeover=bool(args.track_cleanup_enable) and bool(args.actor_enable),
        max_gap_sec=float(args.actor_max_gap_sec),
        iou_thr=float(args.actor_iou_thr),
        dist_thr=float(args.actor_dist_thr),
        size_thr=float(args.actor_size_thr),
        dup_iou=float(args.actor_dup_iou),
        takeover_iou=float(args.actor_takeover_iou),
        frame_w=frame_w,
        frame_h=frame_h,
        logger=logger,
    )

    track_states: Dict[int, TrackState] = {}
    finished_events: List[Dict[str, Any]] = []
    track_stats = {
        "raw_sum": 0,
        "kept_sum": 0,
        "removed_sum": 0,
    }
    det_stats = {
        "raw_sum": 0,
        "kept_sum": 0,
        "removed_sum": 0,
    }

    frame_idx = -1
    t0 = time.time()
    keep_cls = set(int(x) for x in (args.classes or []))
    stale_drop_frames = max(int(3 * grace_frames), int(2 * fps))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            next_frame_idx = frame_idx + 1
            if args.max_frames > 0 and next_frame_idx >= int(args.max_frames):
                break
            frame_idx = next_frame_idx

            ts_sec = float(frame_idx) / float(fps)

            res = det_model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.det_conf,
                classes=list(keep_cls) if keep_cls else None,
                device=args.device,
                verbose=False,
            )
            r0 = res[0] if res else None

            dets = []
            det_footpoints: List[Dict[str, Any]] = []
            kp_xy_all = None
            kp_cf_all = None
            if (
                str(args.algo) == "_02_use_keypoint"
                and r0 is not None
                and getattr(r0, "keypoints", None) is not None
            ):
                kobj = r0.keypoints
                if getattr(kobj, "xy", None) is not None:
                    xy_obj = kobj.xy
                    kp_xy_all = xy_obj.detach().cpu().numpy() if hasattr(xy_obj, "detach") else np.asarray(xy_obj)
                if getattr(kobj, "conf", None) is not None:
                    cf_obj = kobj.conf
                    kp_cf_all = cf_obj.detach().cpu().numpy() if hasattr(cf_obj, "detach") else np.asarray(cf_obj)
            if r0 is not None and getattr(r0, "boxes", None) is not None:
                boxes = r0.boxes
                xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
                confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else []
                cls = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None

                for i in range(len(xyxy)):
                    c = int(cls[i]) if cls is not None else 0
                    if keep_cls and c not in keep_cls:
                        continue
                    sc = float(confs[i]) if i < len(confs) else 0.0
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                    bbox = [x1, y1, x2, y2]
                    dets.append([x1, y1, x2, y2, sc, float(c)])

                    kxy = kp_xy_all[i] if kp_xy_all is not None and i < len(kp_xy_all) else None
                    kcf = kp_cf_all[i] if kp_cf_all is not None and i < len(kp_cf_all) else None
                    fx, fy, fmethod, ankles = compute_footpoint(
                        bbox=bbox,
                        kps_xy=kxy,
                        kps_conf=kcf,
                        kp_conf=float(args.kp_conf),
                    )
                    det_footpoints.append(
                        {
                            "xy": [float(fx), float(fy)],
                            "method": str(fmethod),
                            "ankles": list(ankles),
                        }
                    )

            raw_det_count = int(len(dets))
            if bool(args.det_dedup_enable) and raw_det_count > 1:
                det_pairs = list(zip(dets, det_footpoints))
                dedup_dets = dedup_detections_greedy(
                    dets,
                    iou_thr=float(args.det_dedup_iou),
                    contain_ioa_thr=float(args.det_dedup_contain_ioa),
                    keep_mode=str(args.det_dedup_keep),
                )
                used = [False] * len(det_pairs)
                dedup_fp: List[Dict[str, Any]] = []
                for kd in dedup_dets:
                    best_idx = -1
                    best_score = float("inf")
                    kcls = int(round(float(kd[5]))) if len(kd) > 5 else 0
                    for j, (od, _) in enumerate(det_pairs):
                        if used[j]:
                            continue
                        ocls = int(round(float(od[5]))) if len(od) > 5 else 0
                        if ocls != kcls:
                            continue
                        score = 0.0
                        for vi in range(min(len(od), len(kd), 6)):
                            score += abs(float(od[vi]) - float(kd[vi]))
                        if score < best_score:
                            best_score = score
                            best_idx = j
                    if best_idx >= 0:
                        used[best_idx] = True
                        dedup_fp.append(det_pairs[best_idx][1])
                    else:
                        fx, fy, fmethod, ankles = compute_footpoint(
                            bbox=kd[:4],
                            kps_xy=None,
                            kps_conf=None,
                            kp_conf=float(args.kp_conf),
                        )
                        dedup_fp.append({"xy": [float(fx), float(fy)], "method": str(fmethod), "ankles": list(ankles)})
                dets = dedup_dets
                det_footpoints = dedup_fp
            dedup_det_count = int(len(dets))
            removed_det_count = int(max(0, raw_det_count - dedup_det_count))
            det_stats["raw_sum"] += raw_det_count
            det_stats["kept_sum"] += dedup_det_count
            det_stats["removed_sum"] += removed_det_count

            det_arr = np.asarray(dets, dtype=np.float32) if len(dets) else np.zeros((0, 6), dtype=np.float32)
            try:
                tracks_out = tracker.update(det_arr, frame)
            except TypeError:
                tracks_out = tracker.update(dets=det_arr, img=frame)
            tracks = sorted(parse_tracker_outputs(tracks_out), key=lambda x: int(x["track_id"]))

            det_boxes = [d[:4] for d in dets]
            for tr in tracks:
                tb = tr["bbox_xyxy"]
                best_iou = 0.0
                best_det_idx = -1
                best_det_conf = None
                for di, db in enumerate(det_boxes):
                    v = float(iou_xyxy(tb, db))
                    if v > best_iou:
                        best_iou = v
                        best_det_idx = di
                        if di < len(dets):
                            best_det_conf = float(dets[di][4])
                tr["support_iou"] = float(best_iou)
                tr["track_supported"] = bool(best_iou >= float(args.track_support_iou))
                tr["support_det_idx"] = int(best_det_idx)
                tr["det_conf"] = float(best_det_conf) if best_det_conf is not None else None

            raw_track_count = int(len(tracks))
            if bool(args.track_cleanup_enable) and raw_track_count > 1:
                tracks = track_cleanup_greedy(
                    tracks,
                    merge_iou_thr=float(args.track_merge_iou),
                    contain_ioa_thr=float(args.track_contain_ioa),
                )
            kept_track_count = int(len(tracks))
            removed_track_count = int(max(0, raw_track_count - kept_track_count))
            if bool(args.track_cleanup_enable):
                track_stats["raw_sum"] += raw_track_count
                track_stats["kept_sum"] += kept_track_count
                track_stats["removed_sum"] += removed_track_count

            actor_manager.begin_frame()
            seen_tids = set()
            seen_state_ids = set()
            seen_actor_ids = set()
            updated_state_ids = set()
            draw_rows: List[Dict[str, Any]] = []

            for tr in tracks:
                tid = int(tr["track_id"])
                seen_tids.add(tid)
                bbox = [float(v) for v in tr["bbox_xyxy"]]
                ioa = float(bbox_ioa_roi(roi_mask, bbox))
                track_supported = bool(tr.get("track_supported", False))
                det_idx = int(tr.get("support_det_idx", -1))
                track_conf = tr.get("conf", None)
                if track_conf is None:
                    track_conf = tr.get("det_conf", None)
                if 0 <= det_idx < len(det_footpoints):
                    foot_xy = [float(det_footpoints[det_idx]["xy"][0]), float(det_footpoints[det_idx]["xy"][1])]
                    foot_method = str(det_footpoints[det_idx].get("method", "BBOX"))
                    foot_ankles = list(det_footpoints[det_idx].get("ankles", []) or [])
                else:
                    fx, fy, fmethod, ankles = compute_footpoint(
                        bbox=bbox,
                        kps_xy=None,
                        kps_conf=None,
                        kp_conf=float(args.kp_conf),
                    )
                    foot_xy = [float(fx), float(fy)]
                    foot_method = str(fmethod)
                    foot_ankles = list(ankles)
                foot_in_roi = point_in_roi_mask(roi_mask, foot_xy[0], foot_xy[1])
                actor_id = int(
                    actor_manager.assign(
                        tid,
                        bbox,
                        frame_idx=frame_idx,
                        ts_sec=ts_sec,
                        track_supported=track_supported,
                    )
                )
                state_id = int(actor_id) if bool(args.actor_enable) and bool(args.actor_use_for_fsm) else int(tid)

                st = track_states.get(state_id)
                if st is None:
                    st = TrackState()
                    track_states[state_id] = st

                st.actor_id = int(actor_id)
                st.last_track_id = int(tid)
                st.last_track_supported = bool(track_supported)
                seen_state_ids.add(state_id)
                seen_actor_ids.add(actor_id)

                # If duplicate tracks map to same actor in one frame, update FSM once.
                if state_id not in updated_state_ids:
                    st.last_bbox = bbox
                    update_track_state_seen(
                        args=args,
                        st=st,
                        actor_id=actor_id,
                        track_id=tid,
                        track_supported=track_supported,
                        track_conf=float(track_conf) if track_conf is not None else None,
                        frame_idx=frame_idx,
                        ts_sec=ts_sec,
                        ioa=ioa,
                        foot_in_roi=foot_in_roi,
                        foot_xy=foot_xy,
                        foot_method=foot_method,
                        foot_ankles=foot_ankles,
                        dwell_frames_req=dwell_frames_req,
                        events=finished_events,
                    )
                    updated_state_ids.add(state_id)
                else:
                    # Keep the most overlapping instance as representative debug bbox.
                    if ioa >= float(st.last_ioa):
                        st.last_bbox = bbox
                        st.last_ioa = float(ioa)
                        st.last_track_id = int(tid)
                        st.last_track_supported = bool(track_supported)
                        st.last_conf = float(track_conf) if track_conf is not None else None
                        st.last_foot_in_roi = bool(foot_in_roi)
                        st.last_foot_xy = [float(foot_xy[0]), float(foot_xy[1])]
                        st.last_foot_method = str(foot_method)
                        st.last_ankles = list(foot_ankles)
                        if st.current_event is not None:
                            st.current_event["actor_id"] = int(actor_id)
                            st.current_event["track_id"] = int(tid)

                draw_rows.append(
                    {
                        "actor_id": int(actor_id),
                        "track_id": int(tid),
                        "bbox": bbox,
                        "state": st.state,
                        "track_supported": bool(track_supported),
                        "conf": float(st.last_conf) if st.last_conf is not None else None,
                        "foot_in_roi": bool(st.last_foot_in_roi),
                        "foot_xy": [float(st.last_foot_xy[0]), float(st.last_foot_xy[1])]
                        if st.last_foot_xy is not None
                        else None,
                        "foot_method": str(st.last_foot_method),
                        "ankles": list(st.last_ankles),
                    }
                )

            actor_manager.end_frame(seen_tids, ts_sec)

            for sid, st in list(track_states.items()):
                if sid in seen_state_ids:
                    continue
                st.last_track_supported = False
                update_track_state_missing(
                    args=args,
                    st=st,
                    frame_idx=frame_idx,
                    ts_sec=ts_sec,
                    grace_frames=grace_frames,
                    events=finished_events,
                )
                miss_frames = max(0, frame_idx - st.last_seen_frame)
                draw_ghost = bool(st.state in (STATE_CAND, STATE_IN) and st.last_bbox is not None)
                if draw_ghost and miss_frames > grace_frames:
                    draw_ghost = bool(str(args.missing_policy) == "hold_in" and st.state == STATE_IN)
                if draw_ghost:
                    draw_rows.append(
                        {
                            "actor_id": int(st.actor_id if st.actor_id > 0 else sid),
                            "track_id": int(st.last_track_id),
                            "bbox": [float(v) for v in st.last_bbox],
                            "state": st.state,
                            "track_supported": False,
                            "conf": float(st.last_conf) if st.last_conf is not None else None,
                            "foot_in_roi": bool(st.last_foot_in_roi),
                            "foot_xy": [float(st.last_foot_xy[0]), float(st.last_foot_xy[1])]
                            if st.last_foot_xy is not None
                            else None,
                            "foot_method": str(st.last_foot_method),
                            "ankles": list(st.last_ankles),
                        }
                    )
                if st.state == STATE_OUT and miss_frames > stale_drop_frames:
                    track_states.pop(sid, None)

            vis = frame.copy()
            cv2.polylines(vis, [roi_poly.reshape((-1, 1, 2))], isClosed=True, color=(80, 240, 80), thickness=1)

            # sort for deterministic overlay
            draw_rows = sorted(draw_rows, key=lambda x: (int(x["actor_id"]), int(x["track_id"])))
            rows_for_viz = draw_rows
            if str(args.viz_mode) == "clean":
                actor_primary: Dict[int, Dict[str, Any]] = {}
                for row in draw_rows:
                    if row["state"] != STATE_IN:
                        continue
                    if not bool(row.get("track_supported", False)):
                        continue
                    aid = int(row["actor_id"])
                    prev = actor_primary.get(aid, None)
                    if prev is None or float(box_area(row["bbox"])) > float(box_area(prev["bbox"])):
                        actor_primary[aid] = row
                rows_for_viz = sorted(actor_primary.values(), key=lambda x: (int(x["actor_id"]), int(x["track_id"])))

            for row in rows_for_viz:
                actor_id = int(row["actor_id"])
                track_id = int(row["track_id"])
                bbox = row["bbox"]
                state = row["state"]
                row_conf = row.get("conf", None)
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                color = color_for_state(state)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                if bool(args.actor_enable):
                    base_lbl = f"A{actor_id} (T{track_id})"
                else:
                    base_lbl = f"T{track_id}"
                if row_conf is not None:
                    base_lbl = f"{base_lbl} c={float(row_conf):.2f}"
                cv2.putText(
                    vis,
                    base_lbl,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                if bool(args.pose_draw) and str(args.algo) == "_02_use_keypoint":
                    for ak in row.get("ankles", []) or []:
                        axy = ak.get("xy", None)
                        if axy is None or len(axy) < 2:
                            continue
                        ax = int(round(float(axy[0])))
                        ay = int(round(float(axy[1])))
                        ax = max(0, min(frame_w - 1, ax))
                        ay = max(0, min(frame_h - 1, ay))
                        an = str(ak.get("name", "A"))
                        cv2.circle(vis, (ax, ay), 4, color, -1)
                        cv2.putText(
                            vis,
                            an,
                            (ax + 5, max(14, ay - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    fxy = row.get("foot_xy", None)
                    if fxy is not None and len(fxy) >= 2:
                        fx = int(round(float(fxy[0])))
                        fy = int(round(float(fxy[1])))
                        fx = max(0, min(frame_w - 1, fx))
                        fy = max(0, min(frame_h - 1, fy))
                        cv2.circle(vis, (fx, fy), 4, color, -1)
                        fmethod = str(row.get("foot_method", "BBOX"))
                        cv2.putText(
                            vis,
                            fmethod,
                            (fx + 5, max(14, fy - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            hud_lines = []
            if bool(args.det_dedup_enable):
                hud_lines.append(f"det_dedup: raw={raw_det_count} kept={dedup_det_count}")
            if bool(args.track_cleanup_enable):
                hud_lines.append(f"track_cleanup: raw={raw_track_count} kept={kept_track_count}")
            if bool(args.actor_enable):
                active_actor_map: Dict[int, TrackState] = {}
                for sid, st in track_states.items():
                    if st.state not in (STATE_CAND, STATE_IN):
                        continue
                    aid = int(st.actor_id if st.actor_id > 0 else sid)
                    prev = active_actor_map.get(aid, None)
                    if prev is None or int(st.last_seen_frame) >= int(prev.last_seen_frame):
                        active_actor_map[aid] = st
                if str(args.viz_mode) == "clean":
                    visible_actor_ids = set(int(r["actor_id"]) for r in rows_for_viz)
                    unique_actors_visible = len(visible_actor_ids)
                    unique_actors_in_or_cand = len(visible_actor_ids)
                else:
                    unique_actors_visible = len(set(int(a) for a in seen_actor_ids))
                    unique_actors_in_or_cand = len(active_actor_map)
                hud_lines.append(
                    f"unique_actors_visible={unique_actors_visible} | unique_actors_in_or_cand={unique_actors_in_or_cand}"
                )

                active_actor_ids = sorted(active_actor_map.keys())
                for actor_id in active_actor_ids:
                    st = active_actor_map[actor_id]
                    if str(args.viz_mode) == "clean":
                        if st.state != STATE_IN:
                            continue
                        if not bool(st.last_track_supported):
                            continue
                    miss_frames = max(0, frame_idx - st.last_seen_frame)
                    grace_left = max(0.0, float(grace_frames - miss_frames) / float(fps))
                    if st.state == STATE_CAND:
                        dwell = float(st.dwell_frames) / float(fps)
                        reason = "PENDING"
                    else:
                        if st.current_event is not None:
                            enter_f = int(st.current_event.get("enter_frame", frame_idx))
                            dwell = max(0.0, float(frame_idx - enter_f + 1) / float(fps))
                            reason = str(st.current_event.get("confirm_reason", st.confirm_reason or "BBOX_IOA"))
                        else:
                            dwell = float(st.dwell_frames) / float(fps)
                            reason = str(st.confirm_reason or "BBOX_IOA")
                    hud_lines.append(
                        f"A{actor_id} (T{int(st.last_track_id)}) | {'IN' if st.state == STATE_IN else 'CAND'} | "
                        f"dwell {dwell:.1f}s | grace {grace_left:.1f}s | reason {reason} | ioa {st.last_ioa:.2f}"
                    )
            else:
                active_ids = sorted([sid for sid, st in track_states.items() if st.state in (STATE_CAND, STATE_IN)])
                unique_actors_in_or_cand = len(active_ids)
                for tid in active_ids:
                    st = track_states[tid]
                    miss_frames = max(0, frame_idx - st.last_seen_frame)
                    grace_left = max(0.0, float(grace_frames - miss_frames) / float(fps))
                    if st.state == STATE_CAND:
                        dwell = float(st.dwell_frames) / float(fps)
                        reason = "PENDING"
                    else:
                        if st.current_event is not None:
                            enter_f = int(st.current_event.get("enter_frame", frame_idx))
                            dwell = max(0.0, float(frame_idx - enter_f + 1) / float(fps))
                            reason = str(st.current_event.get("confirm_reason", st.confirm_reason or "BBOX_IOA"))
                        else:
                            dwell = float(st.dwell_frames) / float(fps)
                            reason = str(st.confirm_reason or "BBOX_IOA")
                    hud_lines.append(
                        f"ID {tid} | {'IN' if st.state == STATE_IN else 'CAND'} | "
                        f"dwell {dwell:.1f}s | grace {grace_left:.1f}s | reason {reason} | ioa {st.last_ioa:.2f}"
                    )

            draw_hud_top_right(cv2, vis, hud_lines)
            vw.write(vis)

            if not args.no_show:
                cv2.imshow("AID Intrusion Offline", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if frame_idx % 30 == 0:
                elapsed = max(1e-6, time.time() - t0)
                eff_fps = (frame_idx + 1) / elapsed
                logger.info(
                    "progress frame=%d fps=%.2f tracks=%d active=%d events=%d",
                    frame_idx,
                    eff_fps,
                    len(tracks),
                    unique_actors_in_or_cand,
                    len(finished_events),
                )

    finally:
        cap.release()
        vw.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    # finalize still-open IN events at end-of-video
    final_ts = float(frame_idx) / float(fps) if frame_idx >= 0 else 0.0
    for tid, st in sorted(track_states.items(), key=lambda x: int(x[0])):
        if st.state == STATE_IN and st.current_event is not None:
            ev = _finalize_event_if_needed(st, None, None)
            if ev is not None:
                # if video ended while IN, duration up to last seen timestamp only.
                confirm_ts = float(ev.get("confirm_ts") or 0.0)
                ev["duration_sec"] = max(0.0, float(st.last_seen_ts) - confirm_ts)
                finished_events.append(ev)

    summary = {
        "meta": {
            "video_id": src.video_id,
            "source": str(in_path),
            "roi_id": roi_cfg.roi_id,
            "roi_version": roi_cfg.roi_version,
            "roi_video_id": roi_cfg.video_id,
            "roi_path": str(roi_cfg.path),
            "roi_labeled_on": roi_cfg.labeled_on,
            "roi_disp_scale_used": roi_cfg.disp_scale_used,
            "roi_image_size_original": {"width": roi_cfg.image_width, "height": roi_cfg.image_height},
            "roi_vertices_original_px": roi_cfg.vertices_orig,
            "fps": float(fps),
            "frame_size": {"width": frame_w, "height": frame_h},
            "thresholds": {
                "cand_ioa_thr": float(args.cand_ioa_thr),
                "in_ioa_thr": float(args.in_ioa_thr),
                "kp_conf": float(args.kp_conf),
                "foot_ioa_thr": float(args.foot_ioa_thr),
                "foot_confirm_ioa_thr": float(args.foot_confirm_ioa_thr),
                "enter_n": int(args.enter_n),
                "exit_n": int(args.exit_n),
                "dwell_sec": float(args.dwell_sec),
                "grace_sec": float(args.grace_sec),
                "missing_policy": str(args.missing_policy),
                "dwell_frames_req": int(dwell_frames_req),
                "grace_frames": int(grace_frames),
            },
            "algo": args.algo,
            "detector_tracker": {
                "det_model": args.det_model,
                "det_conf": float(args.det_conf),
                "det_dedup_enable": bool(args.det_dedup_enable),
                "det_dedup_iou": float(args.det_dedup_iou),
                "det_dedup_contain_ioa": float(args.det_dedup_contain_ioa),
                "det_dedup_keep": str(args.det_dedup_keep),
                "track_support_iou": float(args.track_support_iou),
                "track_cleanup_enable": bool(args.track_cleanup_enable),
                "track_merge_iou": float(args.track_merge_iou),
                "track_contain_ioa": float(args.track_contain_ioa),
                "imgsz": int(args.imgsz),
                "device": str(args.device),
                "classes": [int(x) for x in (args.classes or [])],
                "tracker": args.tracker,
                "reid_model": args.reid_model,
                "fp16": bool(args.fp16),
                "per_class": bool(args.per_class),
            },
            "det_stats": {
                "raw_sum": int(det_stats["raw_sum"]),
                "kept_sum": int(det_stats["kept_sum"]),
                "removed_sum": int(det_stats["removed_sum"]),
            },
            "track_stats": {
                "raw_sum": int(track_stats["raw_sum"]),
                "kept_sum": int(track_stats["kept_sum"]),
                "removed_sum": int(track_stats["removed_sum"]),
            },
            "versions": versions,
        },
        "events": finished_events,
        "stats": {
            "frames_processed": int(max(0, frame_idx + 1)),
            "events_count": int(len(finished_events)),
            "last_ts_sec": float(final_ts),
            "overlay_mp4": str(out_overlay),
        },
    }
    out_events.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("Done. overlay=%s", out_overlay)
    logger.info("Done. events=%s", out_events)
    logger.info("Saved under: %s", run_paths.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
