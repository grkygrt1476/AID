#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None


WINDOW_NAME = "Intrusion ROI Label Tool"
STATUS_DONE = "done"
STATUS_DIRTY = "dirty"
STATUS_TODO = "todo"


@dataclass
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    resized_w: int
    resized_h: int
    dst_w: int
    dst_h: int


@dataclass
class VideoJob:
    video_id: str
    label_path: Path
    video_path: Path
    width: int
    height: int
    frame_count: int
    fps: float
    event_start: int
    event_end: int
    start_use: int
    end_use: int
    roi_dir: Path
    roi_json_path: Path
    roi_fix_path: Path
    overlay_path: Path
    overlay_disp_path: Path
    snap_start_path: Path
    snap_end_path: Path
    points: list[tuple[int, int]] = field(default_factory=list)
    saved_valid: bool = False
    dirty: bool = False
    start_img: Any = None
    end_img: Any = None

    def status(self) -> str:
        if self.dirty:
            return STATUS_DIRTY
        if self.saved_valid:
            return STATUS_DONE
        return STATUS_TODO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive ROI labeling tool for intrusion videos.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Label all videos found in --label_root.")
    group.add_argument("--choice", nargs="+", default=None, help="Specific video IDs or mp4 filenames.")

    p.add_argument("--label_root", default="data/videos/labels")
    p.add_argument("--video_root", default="data/videos")
    p.add_argument("--roi_root", default="data/videos/rois")
    p.add_argument("--event_idx", type=int, default=0)
    p.add_argument("--pad_frames", type=int, default=0)
    p.add_argument("--disp_wh", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"))
    p.add_argument("--roi_name", default="roi_area01")
    p.add_argument("--roi_version", type=int, default=1)
    p.add_argument("--force_overwrite", action="store_true", default=False)
    return p


def _normalize_video_id(token: str) -> str:
    raw = str(token).strip()
    if not raw:
        return ""
    if raw.lower().endswith(".mp4"):
        return Path(raw).stem
    return Path(raw).stem if "." in Path(raw).name else raw


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_event_pairs(label_obj: dict[str, Any]) -> list[tuple[int, int]]:
    ann = label_obj.get("annotations", {})
    event_raw = None
    if isinstance(ann, dict):
        event_raw = ann.get("event_frame", None)
    if event_raw is None:
        event_raw = label_obj.get("event_frame", None)

    out: list[tuple[int, int]] = []
    if not isinstance(event_raw, list):
        return out
    for item in event_raw:
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_video_path(video_root: Path, video_id: str, file_name: str) -> Optional[Path]:
    candidates = [
        video_root / f"{video_id}.mp4",
        video_root / "validation" / f"{video_id}.mp4",
    ]
    if file_name:
        candidates.extend(
            [
                video_root / file_name,
                video_root / "validation" / file_name,
            ]
        )
    for c in candidates:
        if c.exists():
            return c
    return None


def _infer_frame_count(video_path: Path) -> int:
    if cv2 is None:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            count += 1
    finally:
        cap.release()
    return count


def _probe_video_info(video_path: Path) -> tuple[int, int, int, float]:
    if cv2 is None:
        return 0, 0, 0, 0.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0, 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if (w <= 0 or h <= 0) and cap.isOpened():
            ok, frame0 = cap.read()
            if ok and frame0 is not None:
                h, w = frame0.shape[:2]
    finally:
        cap.release()
    if n <= 0:
        n = _infer_frame_count(video_path)
    return int(w), int(h), int(n), float(fps if fps > 0 else 30.0)


def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _is_valid_vertices(vertices: list[tuple[int, int]], width: int, height: int) -> bool:
    if len(vertices) < 3:
        return False
    if width <= 0 or height <= 0:
        return False
    for x, y in vertices:
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return False
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
    return True


def _read_existing_roi(path: Path, width: int, height: int) -> Optional[list[tuple[int, int]]]:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return None
    raw = obj.get("vertices_px", None)
    if not isinstance(raw, list):
        return None
    vertices: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        try:
            x = int(round(float(item[0])))
            y = int(round(float(item[1])))
        except Exception:
            return None
        vertices.append((x, y))
    if not _is_valid_vertices(vertices, width=width, height=height):
        return None
    return vertices


def _order_points_clockwise(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(points) < 3:
        return list(points)
    arr = np.asarray(points, dtype=np.float32)
    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = arr[order]
    area2 = 0.0
    n = int(ordered.shape[0])
    for i in range(n):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % n]
        area2 += float(x1 * y2 - x2 * y1)
    if area2 < 0:
        ordered = ordered[::-1]
    return [(int(round(p[0])), int(round(p[1]))) for p in ordered]


def _compute_letterbox(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Letterbox:
    if src_w <= 0 or src_h <= 0:
        return Letterbox(scale=1.0, pad_x=0, pad_y=0, resized_w=max(1, dst_w), resized_h=max(1, dst_h), dst_w=dst_w, dst_h=dst_h)
    scale = min(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
    resized_w = max(1, int(round(float(src_w) * scale)))
    resized_h = max(1, int(round(float(src_h) * scale)))
    pad_x = max(0, (dst_w - resized_w) // 2)
    pad_y = max(0, (dst_h - resized_h) // 2)
    return Letterbox(
        scale=float(scale),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        resized_w=int(resized_w),
        resized_h=int(resized_h),
        dst_w=int(dst_w),
        dst_h=int(dst_h),
    )


def _render_letterbox(img, tf: Letterbox):
    canvas = np.zeros((tf.dst_h, tf.dst_w, 3), dtype=np.uint8)
    if img is None:
        return canvas
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return canvas
    resized = cv2.resize(img, (tf.resized_w, tf.resized_h), interpolation=cv2.INTER_LINEAR)
    x0 = tf.pad_x
    y0 = tf.pad_y
    x1 = min(tf.dst_w, x0 + tf.resized_w)
    y1 = min(tf.dst_h, y0 + tf.resized_h)
    canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return canvas


def _disp_to_orig(
    x: int,
    y: int,
    pane_x: int,
    pane_y: int,
    tf: Letterbox,
    width: int,
    height: int,
) -> Optional[tuple[int, int]]:
    lx = float(x - pane_x - tf.pad_x)
    ly = float(y - pane_y - tf.pad_y)
    if lx < 0 or ly < 0 or lx >= float(tf.resized_w) or ly >= float(tf.resized_h):
        return None
    ox = int(round(lx / max(1e-6, tf.scale)))
    oy = int(round(ly / max(1e-6, tf.scale)))
    ox = _clamp(ox, 0, max(0, width - 1))
    oy = _clamp(oy, 0, max(0, height - 1))
    return ox, oy


def _orig_to_disp(
    x: int,
    y: int,
    pane_x: int,
    pane_y: int,
    tf: Letterbox,
) -> tuple[int, int]:
    dx = int(round(float(x) * tf.scale)) + pane_x + tf.pad_x
    dy = int(round(float(y) * tf.scale)) + pane_y + tf.pad_y
    return dx, dy


def _draw_overlay_on_image(img, points: list[tuple[int, int]]) -> Any:
    out = img.copy()
    if len(points) >= 3:
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    for i, (x, y) in enumerate(points, start=1):
        cv2.circle(out, (int(x), int(y)), 4, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def _extract_and_cache_snapshots(job: VideoJob, logger: logging.Logger) -> bool:
    job.roi_dir.mkdir(parents=True, exist_ok=True)
    start_img = cv2.imread(str(job.snap_start_path)) if job.snap_start_path.exists() else None
    end_img = cv2.imread(str(job.snap_end_path)) if job.snap_end_path.exists() else None

    if start_img is None or end_img is None:
        cap = cv2.VideoCapture(str(job.video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video for snapshots: %s", job.video_path)
            return False
        try:
            if start_img is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(job.start_use))
                ok, frame = cap.read()
                if ok and frame is not None:
                    start_img = frame
                    cv2.imwrite(str(job.snap_start_path), start_img)
            if end_img is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(job.end_use))
                ok, frame = cap.read()
                if ok and frame is not None:
                    end_img = frame
                    cv2.imwrite(str(job.snap_end_path), end_img)
        finally:
            cap.release()

    if start_img is None and job.snap_start_path.exists():
        start_img = cv2.imread(str(job.snap_start_path))
    if end_img is None and job.snap_end_path.exists():
        end_img = cv2.imread(str(job.snap_end_path))

    if start_img is None or end_img is None:
        logger.warning("Snapshot missing for %s (start=%s end=%s)", job.video_id, job.snap_start_path, job.snap_end_path)
        return False

    job.start_img = start_img
    job.end_img = end_img
    return True


def _collect_target_ids(args, label_root: Path) -> list[str]:
    if args.all:
        return sorted([p.stem for p in label_root.glob("*.json") if p.is_file()])
    raw = args.choice or []
    out: list[str] = []
    for token in raw:
        vid = _normalize_video_id(token)
        if vid and vid not in out:
            out.append(vid)
    return out


def build_jobs(args, logger: logging.Logger) -> list[VideoJob]:
    label_root = Path(args.label_root)
    video_root = Path(args.video_root)
    roi_root = Path(args.roi_root)
    event_idx = int(args.event_idx)
    pad_frames = max(0, int(args.pad_frames))
    roi_prefix = f"{args.roi_name}_v{int(args.roi_version)}"

    ids = _collect_target_ids(args, label_root=label_root)
    jobs: list[VideoJob] = []

    for video_id in ids:
        label_path = label_root / f"{video_id}.json"
        if not label_path.exists():
            logger.warning("Skip %s: missing label json (%s)", video_id, label_path)
            continue

        label_obj = _load_json(label_path)
        if label_obj is None:
            logger.warning("Skip %s: failed to parse label json (%s)", video_id, label_path)
            continue

        pairs = _extract_event_pairs(label_obj)
        if not pairs:
            logger.warning("Skip %s: no annotations.event_frame", video_id)
            continue
        if event_idx < 0 or event_idx >= len(pairs):
            logger.warning("Skip %s: event_idx=%d out of range (n=%d)", video_id, event_idx, len(pairs))
            continue
        event_start, event_end = pairs[event_idx]

        meta = label_obj.get("metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        file_name = str(meta.get("file_name", "")).strip()
        video_path = _resolve_video_path(video_root=video_root, video_id=video_id, file_name=file_name)
        if video_path is None:
            logger.warning("Skip %s: video not found under %s (or validation)", video_id, video_root)
            continue

        w0 = _safe_int(meta.get("width", 0), 0)
        h0 = _safe_int(meta.get("height", 0), 0)
        n0 = _safe_int(meta.get("frame_count", 0), 0)
        wv, hv, nv, fps = _probe_video_info(video_path)
        width = wv if wv > 0 else w0
        height = hv if hv > 0 else h0
        frame_count = nv if nv > 0 else n0
        if width <= 0 or height <= 0:
            logger.warning("Skip %s: invalid video size (%dx%d)", video_id, width, height)
            continue

        start_use = max(0, int(event_start) - pad_frames)
        if frame_count > 0:
            end_use = min(frame_count - 1, int(event_end) + pad_frames)
            start_use = min(start_use, frame_count - 1)
        else:
            end_use = max(start_use, int(event_end) + pad_frames)
        if end_use < start_use:
            end_use = start_use

        roi_dir = roi_root / video_id
        roi_json_path = roi_dir / f"{roi_prefix}.json"
        roi_fix_path = roi_dir / f"{roi_prefix}_fix.json"
        overlay_path = roi_dir / f"{roi_prefix}_overlay.jpg"
        overlay_disp_path = roi_dir / f"{roi_prefix}_overlay_disp.jpg"
        snap_start_path = roi_dir / f"snap_start_f{start_use}.jpg"
        snap_end_path = roi_dir / f"snap_end_f{end_use}.jpg"

        job = VideoJob(
            video_id=video_id,
            label_path=label_path,
            video_path=video_path,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps if fps > 0 else 30.0,
            event_start=int(event_start),
            event_end=int(event_end),
            start_use=int(start_use),
            end_use=int(end_use),
            roi_dir=roi_dir,
            roi_json_path=roi_json_path,
            roi_fix_path=roi_fix_path,
            overlay_path=overlay_path,
            overlay_disp_path=overlay_disp_path,
            snap_start_path=snap_start_path,
            snap_end_path=snap_end_path,
        )

        for cand in (job.roi_json_path, job.roi_fix_path):
            if not cand.exists():
                continue
            existing = _read_existing_roi(cand, width=job.width, height=job.height)
            if existing is not None:
                job.points = existing
                job.saved_valid = True
                job.dirty = False
                break
        jobs.append(job)

    return jobs


class RoiLabelApp:
    def __init__(self, args, jobs: list[VideoJob], logger: logging.Logger) -> None:
        self.args = args
        self.jobs = jobs
        self.logger = logger
        self.current_idx = 0
        self.point_n = 4
        self.last_q_ts = 0.0
        self.note_text = ""
        self.note_deadline = 0.0
        self.disp_w = max(320, int(args.disp_wh[0]))
        self.disp_h = max(240, int(args.disp_wh[1]))
        self.left_w = self.disp_w // 2
        self.right_w = self.disp_w - self.left_w

    def set_note(self, msg: str, duration: float = 1.4) -> None:
        self.note_text = str(msg)
        self.note_deadline = time.time() + float(duration)

    def note(self) -> str:
        if time.time() <= self.note_deadline:
            return self.note_text
        return ""

    def current_job(self) -> Optional[VideoJob]:
        if self.current_idx < 0 or self.current_idx >= len(self.jobs):
            return None
        return self.jobs[self.current_idx]

    def _ensure_job_loaded(self, job: VideoJob) -> bool:
        if job.start_img is not None and job.end_img is not None:
            return True
        ok = _extract_and_cache_snapshots(job, self.logger)
        if not ok:
            self.set_note(f"Snapshot load failed: {job.video_id}", 2.0)
            return False
        return True

    def _pane_transforms(self, job: VideoJob) -> tuple[Letterbox, Letterbox]:
        tf_left = _compute_letterbox(job.width, job.height, self.left_w, self.disp_h)
        tf_right = _compute_letterbox(job.width, job.height, self.right_w, self.disp_h)
        return tf_left, tf_right

    def _compose_canvas(self, job: VideoJob, with_hud: bool = True):
        tf_left, tf_right = self._pane_transforms(job)
        canvas = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

        left_img = _render_letterbox(job.start_img, tf_left)
        right_img = _render_letterbox(job.end_img, tf_right)
        canvas[:, 0 : self.left_w] = left_img
        canvas[:, self.left_w : self.disp_w] = right_img

        cv2.line(canvas, (self.left_w, 0), (self.left_w, self.disp_h - 1), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"START f={job.start_use}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"END f={job.end_use}",
            (self.left_w + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        draw_points = list(job.points)
        if len(draw_points) == self.point_n and self.point_n >= 3:
            draw_points = _order_points_clockwise(draw_points)

        for pane_idx, pane_x, tf in ((0, 0, tf_left), (1, self.left_w, tf_right)):
            mapped: list[tuple[int, int]] = []
            for i, (ox, oy) in enumerate(draw_points, start=1):
                dx, dy = _orig_to_disp(ox, oy, pane_x=pane_x, pane_y=0, tf=tf)
                mapped.append((dx, dy))
                cv2.circle(canvas, (dx, dy), 4, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, str(i), (dx + 5, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            if len(mapped) == self.point_n and self.point_n >= 3:
                pts = np.asarray(mapped, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if with_hud:
            self._draw_hud(canvas, job)
        return canvas

    def _draw_hud(self, canvas, job: VideoJob) -> None:
        idx_txt = f"[{self.current_idx + 1}/{len(self.jobs)}] {job.video_id}  (event: {job.event_start}-{job.event_end})  points: {len(job.points)}/{self.point_n}"
        status = job.status()
        status_color = (180, 180, 180)
        if status == STATUS_DONE:
            status_color = (80, 220, 80)
        elif status == STATUS_DIRTY:
            status_color = (255, 0, 255)
        bar_h = 62
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.disp_w - 1, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0, canvas)
        cv2.putText(canvas, idx_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"N={self.point_n}  status={status.upper()}  keys: click u r 3-9 s a/d q",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            status_color,
            2,
            cv2.LINE_AA,
        )
        note = self.note()
        if note:
            cv2.putText(canvas, note, (10, self.disp_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    def _map_click_to_orig(self, job: VideoJob, x: int, y: int) -> Optional[tuple[int, int]]:
        tf_left, tf_right = self._pane_transforms(job)
        p = _disp_to_orig(x, y, pane_x=0, pane_y=0, tf=tf_left, width=job.width, height=job.height)
        if p is not None:
            return p
        return _disp_to_orig(x, y, pane_x=self.left_w, pane_y=0, tf=tf_right, width=job.width, height=job.height)

    def on_mouse(self, event, x, y, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        job = self.current_job()
        if job is None:
            return
        if not self._ensure_job_loaded(job):
            return
        p = self._map_click_to_orig(job, x=int(x), y=int(y))
        if p is None:
            return
        if len(job.points) >= self.point_n:
            self.set_note(f"N={self.point_n} points already set. Use u/r or change N.")
            return
        ox = _clamp(int(p[0]), 0, max(0, job.width - 1))
        oy = _clamp(int(p[1]), 0, max(0, job.height - 1))
        job.points.append((ox, oy))
        if len(job.points) == self.point_n and self.point_n >= 3:
            job.points = _order_points_clockwise(job.points)
        job.dirty = True
        self.set_note(f"Point added ({len(job.points)}/{self.point_n})")

    def _save_job(self, job: VideoJob) -> bool:
        if len(job.points) < self.point_n:
            self.set_note(f"Need {self.point_n} points before save.")
            return False

        points = _order_points_clockwise(job.points[: self.point_n])
        if not _is_valid_vertices(points, width=job.width, height=job.height):
            self.set_note("Invalid ROI points. Check bounds/point count.")
            return False

        if job.roi_json_path.exists() and not bool(self.args.force_overwrite) and not job.dirty:
            self.set_note("Already saved. Move to next.")
            return True

        job.roi_dir.mkdir(parents=True, exist_ok=True)
        tf_left = _compute_letterbox(job.width, job.height, self.left_w, self.disp_h)
        payload = {
            "video_id": job.video_id,
            "roi_id": str(self.args.roi_name),
            "roi_version": int(self.args.roi_version),
            "vertices_px": [[int(x), int(y)] for (x, y) in points],
            "image_size": {"width": int(job.width), "height": int(job.height)},
            "labeled_on": "disp",
            "disp_wh": [int(self.disp_w), int(self.disp_h)],
            "transform": {
                "scale": float(tf_left.scale),
                "pad_x": int(tf_left.pad_x),
                "pad_y": int(tf_left.pad_y),
                "resized_w": int(tf_left.resized_w),
                "resized_h": int(tf_left.resized_h),
            },
            "source_label": str(job.label_path),
            "event_frame_used": [int(job.event_start), int(job.event_end)],
            "event_frame_used_padded": [int(job.start_use), int(job.end_use)],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        for out_path in (job.roi_json_path, job.roi_fix_path):
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.write("\n")

        if job.start_img is None and not self._ensure_job_loaded(job):
            self.set_note("Saved ROI json but overlay snapshot load failed.")
            job.points = points
            job.saved_valid = True
            job.dirty = False
            return True

        overlay_orig = _draw_overlay_on_image(job.start_img, points)
        cv2.imwrite(str(job.overlay_path), overlay_orig)
        render_disp = self._compose_canvas(job, with_hud=False)
        cv2.imwrite(str(job.overlay_disp_path), render_disp)

        job.points = points
        job.saved_valid = True
        job.dirty = False
        self.set_note(f"Saved: {job.roi_json_path.name}", 1.6)
        return True

    def _handle_set_n(self, n: int) -> None:
        self.point_n = int(n)
        job = self.current_job()
        if job is not None:
            if len(job.points) > self.point_n:
                job.points = list(job.points[: self.point_n])
                job.dirty = True
            if len(job.points) == self.point_n and self.point_n >= 3:
                job.points = _order_points_clockwise(job.points)
        self.set_note(f"Point count N set to {self.point_n}")

    def _draw_done_screen(self):
        canvas = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
        cv2.putText(canvas, "DONE", (max(30, self.disp_w // 2 - 90), max(80, self.disp_h // 2 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 220, 80), 4, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "q: quit   a: previous",
            (max(20, self.disp_w // 2 - 170), max(120, self.disp_h // 2 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, canvas)

    def run(self) -> int:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.disp_w, self.disp_h)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        try:
            while True:
                job = self.current_job()
                if job is None:
                    self._draw_done_screen()
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("a") and self.jobs:
                        self.current_idx = len(self.jobs) - 1
                    continue

                self._ensure_job_loaded(job)
                canvas = self._compose_canvas(job, with_hud=True)
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key == 255:
                    continue
                if ord("3") <= key <= ord("9"):
                    self._handle_set_n(int(chr(key)))
                    continue
                if key == ord("u"):
                    if job.points:
                        job.points.pop()
                        job.dirty = True
                        self.set_note(f"Undo -> {len(job.points)}/{self.point_n}")
                    else:
                        self.set_note("No points to undo.")
                    continue
                if key == ord("r"):
                    if job.points:
                        job.points = []
                        job.dirty = True
                    self.set_note("Points reset.")
                    continue
                if key == ord("s"):
                    ok = self._save_job(job)
                    if ok:
                        self.current_idx += 1
                    continue
                if key == ord("d"):
                    self.current_idx = min(len(self.jobs), self.current_idx + 1)
                    continue
                if key == ord("a"):
                    self.current_idx = max(0, self.current_idx - 1)
                    continue
                if key == ord("q"):
                    if job.dirty:
                        now = time.monotonic()
                        if now - self.last_q_ts <= 1.0:
                            break
                        self.last_q_ts = now
                        self.set_note("Current video is dirty. Press q again within 1s to exit.", 1.1)
                        continue
                    break
        finally:
            cv2.destroyAllWindows()
        return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    if cv2 is None or np is None:
        print("ERROR: This tool requires opencv-python and numpy in the current environment.")
        return 1

    jobs = build_jobs(args, logger=logger)
    if not jobs:
        logger.error("No valid videos found to label.")
        return 1

    app = RoiLabelApp(args=args, jobs=jobs, logger=logger)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
