#!/usr/bin/env python3
from __future__ import annotations

# ---------------------------------------------------------------------------
# 이 도구는 manifest(기본: data/clips/manifest.csv)에 있는 50초 이벤트 클립을 순회하며,
# 1) ROI(다각형) 편집
# 2) transition / intrusion 구간 라벨링
# 을 한 화면(OpenCV)에서 수행한다.
#
# 출력 경로/저장 방식
# - ROI: data/videos/rois/<VIDEO_ID>/ 내 기존 ROI JSON을 같은 경로에 덮어쓰기
#   (저장 전 .bak 백업 생성)
# - 라벨: data/labels_user/<VIDEO_ID>.json 에 저장
#   (events 맵에서 event_idx 키 단위로 관리)
#
# 주요 키 바인딩
# - 재생/이동: space, 좌/우 화살표, Shift+좌/우(10프레임), j/l(±1), J/L(±10), [ ](1초), n/p, q(더티 시 1초 내 2회)
# - 모드 전환: 1 ROI, 2 transition, 3 intrusion
# - 마킹: a/s(transition 시작/끝), i/o(intrusion 시작/끝)
# - 저장/실행취소/리셋: w / u / r
# - 소스 라벨 오버레이 토글: t
# - 키코드 디버그: k (마지막 키 정수 코드 출력)
# - ROI 꼭짓점 목표 개수 N: 4~9
# - 타임라인 시킹: 하단 타임라인 바를 클릭/드래그하여 프레임 이동
#
# 구간 규칙
# - intrusion 구간은 저장 필수
# - transition 구간은 선택 사항
# - transition은 intrusion보다 반드시 앞서야 하며, 겹치면 저장 시 자동 보정
# ---------------------------------------------------------------------------

import argparse
import copy
import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


WINDOW_NAME = "Transition + Intrusion Annotator"
TOOL_NAME = "02_04_roi_transition_intrusion_annotator"
POLICY_TEXT = "transition=floating; intrusion=feet_on_ground_ROI"
HUD_HELP_LINE = (
    "space play/pause | arrows/jl step | shift+arrows or J/L x10 | [ ] +/-1sec | 1 ROI 2 transition 3 intrusion | "
    "4-9 set ROI N | a/s transition start/end | i/o intrusion start/end | u undo | r reset | w save | "
    "n/p nav | t src | k keydbg | h help | q quit | mouse: timeline seek"
)

MODE_ROI = "roi"
MODE_TRANSITION = "transition"
MODE_INTRUSION = "intrusion"

DEFAULT_DISP_W = 1280
DEFAULT_DISP_H = 720
DEFAULT_ROI_TARGET_N = 4

LEFT_CODES = {
    81,  # GTK low byte
    2424832,  # Win32
    65361,  # X11
    1113937,  # Qt (variant)
    16777234,  # Qt::Key_Left
}
RIGHT_CODES = {
    83,  # GTK low byte
    2555904,  # Win32
    65363,  # X11
    1113939,  # Qt (variant)
    16777236,  # Qt::Key_Right
}
# These can vary by backend; explicit sets plus pattern fallback are used.
SHIFT_LEFT_CODES = {
    2162688,
    3014656,
    3407872,
    3932160,
    33587234,
}
SHIFT_RIGHT_CODES = {
    2293760,
    3145728,
    3538944,
    4063232,
    33587236,
}


@dataclass
class ManifestEvent:
    manifest_index: int
    video_id: str
    event_idx: int
    clip_path: Path
    clip_path_raw: str
    fps_src: float
    clip_start_sec: float
    ev_start_frame: int
    ev_end_frame: int
    row: dict[str, str]


@dataclass
class IntervalMark:
    start: Optional[int] = None
    end: Optional[int] = None

    def clear(self) -> None:
        self.start = None
        self.end = None

    def as_closed_pair(self) -> Optional[tuple[int, int]]:
        if self.start is None or self.end is None:
            return None
        s = int(self.start)
        e = int(self.end)
        if e < s:
            s, e = e, s
        return s, e

    def as_display_pair(self, current_idx: int) -> Optional[tuple[int, int]]:
        if self.start is None:
            return None
        s = int(self.start)
        e = int(current_idx if self.end is None else self.end)
        if e < s:
            s, e = e, s
        return s, e

    def status_text(self) -> str:
        if self.start is None:
            return "none"
        if self.end is None:
            return f"open [{self.start}, ...]"
        s, e = self.as_closed_pair() or (self.start, self.end)
        return f"[{s}, {e}]"


@dataclass
class EventState:
    event: ManifestEvent
    cap: Any
    frame_count: int
    fps_clip: float
    frame_w: int
    frame_h: int
    disp_w: int
    disp_h: int
    frame_idx: int = 0
    frame_cache_idx: int = -1
    frame_cache: Any = None
    mode: str = MODE_ROI
    roi_path: Path = Path(".")
    roi_primary_path: Path = Path(".")
    roi_fallback_path: Path = Path(".")
    roi_file_found: bool = False
    roi_obj: dict[str, Any] = field(default_factory=dict)
    roi_points: list[tuple[int, int]] = field(default_factory=list)
    roi_target_n: int = DEFAULT_ROI_TARGET_N
    roi_warning: str = ""
    roi_warning_detail: str = ""
    roi_dirty: bool = False
    labels_dirty: bool = False
    transition: IntervalMark = field(default_factory=IntervalMark)
    intrusion: IntervalMark = field(default_factory=IntervalMark)
    show_source_interval: bool = True
    source_interval_orig: Optional[tuple[int, int]] = None
    dragging_vertex: Optional[int] = None
    dragging_timeline: bool = False
    selected_vertex: Optional[int] = None
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    note_text: str = ""
    note_deadline: float = 0.0
    last_saved_at: Optional[str] = None
    loaded_user_data: bool = False

    def is_dirty(self) -> bool:
        return bool(self.roi_dirty or self.labels_dirty)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def normalize_video_id(token: str) -> str:
    raw = str(token or "").strip()
    if not raw:
        return ""
    if raw.lower().endswith(".mp4"):
        return Path(raw).stem
    if "/" in raw or "\\" in raw:
        return Path(raw).stem
    p = Path(raw)
    if p.suffix:
        return p.stem
    return raw


def load_json_dict(path: Path) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def parse_pair(raw: Any) -> Optional[tuple[int, int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        # [[s,e], ...]
        if len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            first = raw[0]
            if len(first) >= 2:
                s = safe_int(first[0], 0)
                e = safe_int(first[1], 0)
                if e < s:
                    s, e = e, s
                return s, e
        # [s,e]
        if len(raw) >= 2 and not isinstance(raw[0], (list, tuple)):
            s = safe_int(raw[0], 0)
            e = safe_int(raw[1], 0)
            if e < s:
                s, e = e, s
            return s, e
    return None


def pair_to_nested(pair: Optional[tuple[int, int]]) -> list[list[int]]:
    if pair is None:
        return []
    s, e = pair
    if e < s:
        s, e = e, s
    return [[int(s), int(e)]]


def extract_event_pairs(label_obj: dict[str, Any]) -> list[tuple[int, int]]:
    ann = label_obj.get("annotations")
    raw = None
    if isinstance(ann, dict):
        raw = ann.get("event_frame")
    if raw is None:
        raw = label_obj.get("event_frame")
    if not isinstance(raw, list):
        return []
    pairs: list[tuple[int, int]] = []
    for item in raw:
        p = parse_pair(item)
        if p is not None:
            pairs.append(p)
    return pairs


def make_canvas(width: int, height: int, color: tuple[int, int, int] = (0, 0, 0)):
    w = max(1, int(width))
    h = max(1, int(height))
    img = cv2.UMat(h, w, cv2.CV_8UC3).get()
    img[:, :] = color
    return img


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def write_json_with_backup(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bak_path = Path(str(path) + ".bak")
        shutil.copy2(path, bak_path)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive annotation tool for transition/intrusion intervals with editable ROI."
    )
    p.add_argument("--manifest", default="data/clips/manifest.csv")
    p.add_argument("--labels_user_dir", default="data/labels_user")
    p.add_argument("--labels_src_dir", default="data/labels")
    p.add_argument("--rois_dir", default="data/videos/rois")
    p.add_argument("--roi_name", default="area01")
    p.add_argument("--roi_version", default="v1")
    p.add_argument("--roi_n", type=int, default=0, help="ROI target vertex count override (>0).")
    p.add_argument("--choice", nargs="+", default=None, help="Filter by VIDEO_ID or .mp4 stem.")
    p.add_argument("--start_index", type=int, default=0, help="Start index in filtered/sorted manifest.")
    p.add_argument("--disp_wh", nargs=2, type=int, default=[DEFAULT_DISP_W, DEFAULT_DISP_H], metavar=("W", "H"))
    p.add_argument("--fps_override", type=float, default=None)
    return p


def load_manifest_events(manifest_path: Path, choice: Optional[list[str]]) -> list[ManifestEvent]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    choice_set: set[str] = set()
    if choice:
        for token in choice:
            vid = normalize_video_id(token)
            if vid:
                choice_set.add(vid)

    events: list[ManifestEvent] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            video_id = normalize_video_id(row.get("video_id", ""))
            if not video_id:
                continue
            if choice_set and video_id not in choice_set:
                continue

            clip_raw = str(row.get("clip_path", "")).strip()
            if not clip_raw:
                print(f"[WARN] skip manifest row {idx}: empty clip_path")
                continue
            clip_path = Path(clip_raw)
            if not clip_path.is_absolute():
                clip_path = (Path.cwd() / clip_path).resolve()
            if not clip_path.exists():
                print(f"[WARN] skip manifest row {idx}: clip missing -> {clip_raw}")
                continue

            event_idx = safe_int(row.get("event_idx"), 0)
            fps_src = safe_float(row.get("fps"), 30.0)
            if fps_src <= 0:
                fps_src = 30.0
            clip_start_sec = safe_float(row.get("clip_start_sec"), 0.0)

            ev_start = safe_int(row.get("ev_start_frame"), 0)
            ev_end = safe_int(row.get("ev_end_frame"), ev_start)
            if ev_end < ev_start:
                ev_start, ev_end = ev_end, ev_start

            events.append(
                ManifestEvent(
                    manifest_index=idx,
                    video_id=video_id,
                    event_idx=event_idx,
                    clip_path=clip_path,
                    clip_path_raw=clip_raw,
                    fps_src=fps_src,
                    clip_start_sec=clip_start_sec,
                    ev_start_frame=ev_start,
                    ev_end_frame=ev_end,
                    row={str(k): str(v) for k, v in row.items()},
                )
            )

    events.sort(key=lambda e: (e.video_id, e.event_idx, e.manifest_index))
    return events


class AnnotatorApp:
    def __init__(self, args, events: list[ManifestEvent]) -> None:
        self.args = args
        self.events = events
        self.current_index = clamp(int(args.start_index), 0, max(0, len(events) - 1))
        self.state: Optional[EventState] = None
        self.playing = False
        self.last_tick = time.monotonic()
        self.last_quit_press = 0.0
        self.last_key_code: Optional[int] = None
        self.pending_nav: dict[str, float] = {}
        self.source_label_cache: dict[Path, list[tuple[int, int]]] = {}
        self.source_label_fail: set[Path] = set()

    def set_note(self, text: str, duration: float = 1.4) -> None:
        if self.state is None:
            return
        self.state.note_text = str(text)
        self.state.note_deadline = time.monotonic() + float(duration)

    def current_note(self) -> str:
        if self.state is None:
            return ""
        if time.monotonic() <= self.state.note_deadline:
            return self.state.note_text
        return ""

    def close_state(self) -> None:
        if self.state is None:
            return
        try:
            self.state.cap.release()
        except Exception:
            pass
        self.state = None

    def _clip_to_orig(self, state: EventState, clip_idx: int) -> int:
        fps_clip = max(1e-6, float(state.fps_clip))
        fps_src = max(1e-6, float(state.event.fps_src))
        t = float(clip_idx) / fps_clip
        orig = round((t + float(state.event.clip_start_sec)) * fps_src)
        return int(orig)

    def _orig_to_clip(self, state: EventState, orig_idx: int) -> int:
        fps_src = max(1e-6, float(state.event.fps_src))
        fps_clip = max(1e-6, float(state.fps_clip))
        t = float(orig_idx) / fps_src
        local_t = t - float(state.event.clip_start_sec)
        k = int(round(local_t * fps_clip))
        return clamp(k, 0, max(0, state.frame_count - 1))

    def _interval_clip_to_orig(self, state: EventState, pair: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        if pair is None:
            return None
        s, e = pair
        s2 = self._clip_to_orig(state, int(s))
        e2 = self._clip_to_orig(state, int(e))
        if e2 < s2:
            s2, e2 = e2, s2
        return s2, e2

    def _first_frame_and_info(self, event: ManifestEvent) -> tuple[Any, Any, int, float, int, int]:
        cap = cv2.VideoCapture(str(event.clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open clip: {event.clip_path}")

        fps_cap = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
        frame_count = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            raise RuntimeError(f"Failed to read first frame: {event.clip_path}")
        frame_h, frame_w = frame0.shape[:2]

        if frame_count <= 0:
            # Fallback: count frames by decoding once.
            count = 1
            while True:
                ok_more, _ = cap.read()
                if not ok_more:
                    break
                count += 1
            cap.release()

            cap = cv2.VideoCapture(str(event.clip_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to reopen clip: {event.clip_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame0 = cap.read()
            if not ok or frame0 is None:
                cap.release()
                raise RuntimeError(f"Failed to reread first frame: {event.clip_path}")
            frame_count = count

        if frame_count <= 0:
            frame_count = 1

        fps_clip = float(self.args.fps_override) if self.args.fps_override else float(fps_cap)
        if fps_clip <= 0:
            fps_clip = float(event.fps_src) if event.fps_src > 0 else 30.0
        if fps_clip <= 0:
            fps_clip = 30.0

        return cap, frame0, int(frame_count), float(fps_clip), int(frame_w), int(frame_h)

    def _roi_paths(self, video_id: str) -> tuple[Path, Path]:
        roi_dir = Path(self.args.rois_dir) / video_id
        base = f"roi_{self.args.roi_name}_{self.args.roi_version}"
        primary = roi_dir / f"{base}.json"
        fallback = roi_dir / f"{base}_fix.json"
        return primary, fallback

    def _extract_image_size_from_roi_obj(self, obj: dict[str, Any], frame_w: int, frame_h: int) -> tuple[int, int]:
        width = int(frame_w)
        height = int(frame_h)
        if not isinstance(obj, dict):
            return width, height

        candidates: list[Any] = [
            obj.get("image_size"),
            obj.get("roi", {}).get("image_size") if isinstance(obj.get("roi"), dict) else None,
        ]
        for c in candidates:
            if not isinstance(c, dict):
                continue
            w = safe_int(c.get("width"), 0)
            h = safe_int(c.get("height"), 0)
            if w > 0 and h > 0:
                return int(w), int(h)

        w2 = safe_int(obj.get("width"), 0)
        h2 = safe_int(obj.get("height"), 0)
        if w2 > 0 and h2 > 0:
            return int(w2), int(h2)
        return width, height

    def _parse_point_like(self, item: Any) -> Optional[tuple[float, float]]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                return float(item[0]), float(item[1])
            except Exception:
                return None
        if isinstance(item, dict):
            x = item.get("x", item.get("X"))
            y = item.get("y", item.get("Y"))
            if x is None:
                x = item.get("px", item.get("u"))
            if y is None:
                y = item.get("py", item.get("v"))
            if x is None or y is None:
                return None
            try:
                return float(x), float(y)
            except Exception:
                return None
        return None

    def _extract_points_from_candidate(self, raw: Any) -> list[tuple[float, float]]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            for key in (
                "vertices_px",
                "vertices",
                "points",
                "polygon",
                "poly",
                "roi_vertices",
                "vertices_xy",
                "vertices_abs",
            ):
                if key in raw:
                    pts = self._extract_points_from_candidate(raw.get(key))
                    if pts:
                        return pts
            p = self._parse_point_like(raw)
            return [p] if p is not None else []
        if isinstance(raw, list):
            # [x1, y1, x2, y2, ...] 형태 지원
            if raw and all(isinstance(v, (int, float)) for v in raw) and len(raw) >= 4:
                out_flat: list[tuple[float, float]] = []
                i = 0
                while i + 1 < len(raw):
                    out_flat.append((float(raw[i]), float(raw[i + 1])))
                    i += 2
                return out_flat
            out: list[tuple[float, float]] = []
            for item in raw:
                p = self._parse_point_like(item)
                if p is not None:
                    out.append(p)
            return out
        return []

    def _extract_roi_points(self, obj: Optional[dict[str, Any]], frame_w: int, frame_h: int) -> list[tuple[int, int]]:
        if not isinstance(obj, dict):
            return []
        key_variants = (
            "vertices_px",
            "vertices",
            "points",
            "polygon",
            "poly",
            "roi",
            "roi_vertices",
            "vertices_xy",
            "vertices_abs",
        )

        candidates: list[Any] = []
        for key in key_variants:
            if key in obj:
                candidates.append(obj.get(key))
        roi_node = obj.get("roi")
        if isinstance(roi_node, dict):
            for key in key_variants:
                if key in roi_node:
                    candidates.append(roi_node.get(key))
        elif isinstance(roi_node, list):
            candidates.append(roi_node)

        raw_points: list[tuple[float, float]] = []
        for c in candidates:
            pts = self._extract_points_from_candidate(c)
            if pts:
                raw_points = pts
                break
        if not raw_points:
            return []

        # 정규화 좌표(0~1) 추정: 모든 점이 [0, 1.5] 범위면 정규화로 처리.
        all_norm_like = all(0.0 <= float(x) <= 1.5 and 0.0 <= float(y) <= 1.5 for (x, y) in raw_points)
        ref_w, ref_h = self._extract_image_size_from_roi_obj(obj=obj, frame_w=frame_w, frame_h=frame_h)
        ref_w = max(1, int(ref_w))
        ref_h = max(1, int(ref_h))

        points: list[tuple[int, int]] = []
        for x_raw, y_raw in raw_points:
            x_f = float(x_raw)
            y_f = float(y_raw)
            if all_norm_like:
                x_f *= float(ref_w)
                y_f *= float(ref_h)
            x = clamp(safe_int(round(x_f)), 0, max(0, frame_w - 1))
            y = clamp(safe_int(round(y_f)), 0, max(0, frame_h - 1))
            points.append((x, y))
        return points

    def _load_roi_for_event(
        self, video_id: str, frame_w: int, frame_h: int
    ) -> tuple[Path, Path, Path, bool, dict[str, Any], list[tuple[int, int]]]:
        primary, fallback = self._roi_paths(video_id)
        chosen = primary
        obj: dict[str, Any] = {}
        found = False

        if primary.exists():
            found = True
            chosen = primary
            parsed = load_json_dict(primary)
            if parsed:
                obj = parsed
        elif fallback.exists():
            found = True
            chosen = fallback
            parsed = load_json_dict(fallback)
            if parsed:
                obj = parsed
        else:
            chosen = primary
            obj = {}

        points = self._extract_roi_points(obj, frame_w=frame_w, frame_h=frame_h)
        return primary, fallback, chosen, found, obj, points

    def _source_label_paths(self, event: ManifestEvent) -> list[Path]:
        out: list[Path] = []
        src_dir = Path(self.args.labels_src_dir)
        out.append(src_dir / f"{event.video_id}.json")

        raw = str(event.row.get("label_json", "")).strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p not in out:
                out.append(p)
        return out

    def _load_source_pairs(self, path: Path) -> list[tuple[int, int]]:
        if path in self.source_label_cache:
            return self.source_label_cache[path]
        if path in self.source_label_fail:
            return []
        obj = load_json_dict(path)
        if obj is None:
            self.source_label_fail.add(path)
            return []
        pairs = extract_event_pairs(obj)
        self.source_label_cache[path] = pairs
        return pairs

    def _load_source_interval(self, event: ManifestEvent) -> Optional[tuple[int, int]]:
        for path in self._source_label_paths(event):
            if not path.exists():
                continue
            pairs = self._load_source_pairs(path)
            if event.event_idx < 0 or event.event_idx >= len(pairs):
                continue
            return pairs[event.event_idx]
        return None

    def _labels_user_path(self, video_id: str) -> Path:
        return Path(self.args.labels_user_dir) / f"{video_id}.json"

    def _load_existing_user_event(self, state: EventState) -> None:
        path = self._labels_user_path(state.event.video_id)
        obj = load_json_dict(path)
        if obj is None:
            return

        events_obj = obj.get("events")
        if not isinstance(events_obj, dict):
            return
        payload = events_obj.get(str(state.event.event_idx))
        if not isinstance(payload, dict):
            return

        ann_clip = payload.get("annotations_clip")
        ann_orig = payload.get("annotations")
        ann_clip = ann_clip if isinstance(ann_clip, dict) else {}
        ann_orig = ann_orig if isinstance(ann_orig, dict) else {}

        t_pair = parse_pair(ann_clip.get("transition_frame"))
        i_pair = parse_pair(ann_clip.get("intrusion_frame"))

        if t_pair is None:
            t_orig = parse_pair(ann_orig.get("transition_frame"))
            if t_orig is not None:
                t_pair = (self._orig_to_clip(state, t_orig[0]), self._orig_to_clip(state, t_orig[1]))
                if t_pair[1] < t_pair[0]:
                    t_pair = (t_pair[1], t_pair[0])
        if i_pair is None:
            i_orig = parse_pair(ann_orig.get("intrusion_frame"))
            if i_orig is not None:
                i_pair = (self._orig_to_clip(state, i_orig[0]), self._orig_to_clip(state, i_orig[1]))
                if i_pair[1] < i_pair[0]:
                    i_pair = (i_pair[1], i_pair[0])

        if t_pair is not None:
            state.transition.start = clamp(t_pair[0], 0, max(0, state.frame_count - 1))
            state.transition.end = clamp(t_pair[1], 0, max(0, state.frame_count - 1))
            state.loaded_user_data = True
        if i_pair is not None:
            state.intrusion.start = clamp(i_pair[0], 0, max(0, state.frame_count - 1))
            state.intrusion.end = clamp(i_pair[1], 0, max(0, state.frame_count - 1))
            state.loaded_user_data = True

    def _read_frame(self, state: EventState, idx: int):
        i = clamp(int(idx), 0, max(0, state.frame_count - 1))
        if state.frame_cache_idx == i and state.frame_cache is not None:
            state.frame_idx = i
            return state.frame_cache
        state.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = state.cap.read()
        if not ok or frame is None:
            return None
        state.frame_cache_idx = i
        state.frame_cache = frame
        state.frame_idx = i
        return frame

    def load_event(self, index: int) -> bool:
        self.close_state()
        if index < 0 or index >= len(self.events):
            return False

        event = self.events[index]
        cap, frame0, frame_count, fps_clip, frame_w, frame_h = self._first_frame_and_info(event)

        if self.args.disp_wh and len(self.args.disp_wh) == 2:
            disp_w = max(320, int(self.args.disp_wh[0]))
            disp_h = max(240, int(self.args.disp_wh[1]))
        else:
            disp_w = frame_w
            disp_h = frame_h

        roi_primary_path, roi_fallback_path, roi_path, roi_file_found, roi_obj, roi_points = self._load_roi_for_event(
            event.video_id, frame_w=frame_w, frame_h=frame_h
        )

        loaded_n = len(roi_points)
        if loaded_n >= 3:
            roi_target_n = loaded_n
        else:
            roi_target_n = DEFAULT_ROI_TARGET_N
        if safe_int(self.args.roi_n, 0) > 0:
            roi_target_n = max(3, int(self.args.roi_n))

        state = EventState(
            event=event,
            cap=cap,
            frame_count=frame_count,
            fps_clip=fps_clip,
            frame_w=frame_w,
            frame_h=frame_h,
            disp_w=disp_w,
            disp_h=disp_h,
            frame_idx=0,
            frame_cache_idx=0,
            frame_cache=frame0,
            roi_path=roi_path,
            roi_primary_path=roi_primary_path,
            roi_fallback_path=roi_fallback_path,
            roi_file_found=roi_file_found,
            roi_obj=roi_obj,
            roi_points=roi_points,
            roi_target_n=roi_target_n,
            source_interval_orig=self._load_source_interval(event),
        )
        if not roi_file_found:
            state.roi_warning = f"ROI not found. Expected: {roi_primary_path} or {roi_fallback_path}"
            state.roi_warning_detail = (
                f"rois_dir={self.args.rois_dir} roi_name={self.args.roi_name} roi_version={self.args.roi_version}"
            )
        elif len(roi_points) == 0:
            state.roi_warning = "ROI loaded but no vertices parsed. Check ROI JSON schema."
            state.roi_warning_detail = f"loaded={roi_path}"
        state.last_saved_at = "loaded"
        self._load_existing_user_event(state)

        self.state = state
        self.current_index = index
        self.playing = False
        self.last_tick = time.monotonic()
        cv2.resizeWindow(WINDOW_NAME, state.disp_w, state.disp_h)
        self.set_note(
            f"Loaded {event.video_id} ev{event.event_idx} | ROI file: {state.roi_path.name} | "
            f"ROI verts={len(state.roi_points)} | "
            f"ROI N={state.roi_target_n} | "
            f"{'prefill from labels_user' if state.loaded_user_data else 'no labels_user prefill'}",
            duration=2.2,
        )
        return True

    def _push_undo(self, state: EventState) -> None:
        snap = {
            "roi_points": list(state.roi_points),
            "transition": (state.transition.start, state.transition.end),
            "intrusion": (state.intrusion.start, state.intrusion.end),
            "roi_dirty": bool(state.roi_dirty),
            "labels_dirty": bool(state.labels_dirty),
        }
        state.undo_stack.append(snap)
        if len(state.undo_stack) > 300:
            state.undo_stack = state.undo_stack[-300:]

    def _undo(self, state: EventState) -> None:
        if not state.undo_stack:
            self.set_note("No action to undo.")
            return
        snap = state.undo_stack.pop()
        state.roi_points = list(snap.get("roi_points", []))

        tr = snap.get("transition", (None, None))
        it = snap.get("intrusion", (None, None))
        state.transition.start = tr[0]
        state.transition.end = tr[1]
        state.intrusion.start = it[0]
        state.intrusion.end = it[1]

        state.roi_dirty = bool(snap.get("roi_dirty", False))
        state.labels_dirty = bool(snap.get("labels_dirty", False))
        state.dragging_vertex = None
        state.selected_vertex = None
        self.set_note("Undo applied.")

    def _frame_to_disp(self, state: EventState, x: int, y: int) -> tuple[int, int]:
        sx = float(state.disp_w) / max(1.0, float(state.frame_w))
        sy = float(state.disp_h) / max(1.0, float(state.frame_h))
        dx = clamp(int(round(float(x) * sx)), 0, max(0, state.disp_w - 1))
        dy = clamp(int(round(float(y) * sy)), 0, max(0, state.disp_h - 1))
        return dx, dy

    def _disp_to_frame(self, state: EventState, x: int, y: int) -> tuple[int, int]:
        fx = int(round(float(x) * float(state.frame_w) / max(1.0, float(state.disp_w))))
        fy = int(round(float(y) * float(state.frame_h) / max(1.0, float(state.disp_h))))
        fx = clamp(fx, 0, max(0, state.frame_w - 1))
        fy = clamp(fy, 0, max(0, state.frame_h - 1))
        return fx, fy

    def _nearest_vertex(self, state: EventState, x_disp: int, y_disp: int) -> tuple[Optional[int], float]:
        if not state.roi_points:
            return None, math.inf
        best_i: Optional[int] = None
        best_d2 = math.inf
        for i, (fx, fy) in enumerate(state.roi_points):
            dx, dy = self._frame_to_disp(state, fx, fy)
            d2 = float((dx - x_disp) ** 2 + (dy - y_disp) ** 2)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i, best_d2

    def _fill_polygon_scanline(self, img, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
        if len(points) < 3:
            return
        h, w = img.shape[:2]
        min_y = max(0, min(p[1] for p in points))
        max_y = min(h - 1, max(p[1] for p in points))
        n = len(points)
        for y in range(min_y, max_y + 1):
            xs: list[float] = []
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]
                if y1 == y2:
                    continue
                y_low = min(y1, y2)
                y_high = max(y1, y2)
                if y < y_low or y >= y_high:
                    continue
                ratio = float(y - y1) / float(y2 - y1)
                x = float(x1) + ratio * float(x2 - x1)
                xs.append(x)
            if len(xs) < 2:
                continue
            xs.sort()
            j = 0
            while j + 1 < len(xs):
                x_start = max(0, int(round(xs[j])))
                x_end = min(w - 1, int(round(xs[j + 1])))
                if x_end >= x_start:
                    img[y, x_start : x_end + 1] = color
                j += 2

    def _draw_roi_overlay(self, state: EventState, disp) -> None:
        if not state.roi_points:
            return
        pts_disp = [self._frame_to_disp(state, x, y) for (x, y) in state.roi_points]

        if len(pts_disp) >= 3:
            overlay = disp.copy()
            self._fill_polygon_scanline(overlay, pts_disp, color=(0, 180, 255))
            cv2.addWeighted(overlay, 0.20, disp, 0.80, 0.0, disp)

            for i in range(len(pts_disp)):
                p1 = pts_disp[i]
                p2 = pts_disp[(i + 1) % len(pts_disp)]
                cv2.line(disp, p1, p2, (0, 220, 255), 3, cv2.LINE_AA)

        for i, (dx, dy) in enumerate(pts_disp):
            color = (0, 255, 255) if i == state.selected_vertex else (0, 150, 255)
            cv2.circle(disp, (dx, dy), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                disp,
                str(i),
                (dx + 6, dy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_interval_segment(
        self,
        disp,
        pair: Optional[tuple[int, int]],
        frame_count: int,
        x0: int,
        y0: int,
        w: int,
        h: int,
        color: tuple[int, int, int],
    ) -> None:
        if pair is None or frame_count <= 0:
            return
        s, e = pair
        s = clamp(int(s), 0, max(0, frame_count - 1))
        e = clamp(int(e), 0, max(0, frame_count - 1))
        if e < s:
            s, e = e, s
        if frame_count <= 1:
            x_s = x0
            x_e = x0 + w
        else:
            x_s = x0 + int(round((float(s) / float(frame_count - 1)) * float(w)))
            x_e = x0 + int(round((float(e) / float(frame_count - 1)) * float(w)))
        x_s = clamp(x_s, x0, x0 + w)
        x_e = clamp(x_e, x0, x0 + w)
        cv2.rectangle(disp, (x_s, y0), (x_e, y0 + h), color, -1)

    def _timeline_geometry(self, disp_w: int, disp_h: int) -> tuple[int, int, int, int]:
        bar_x = 18
        bar_w = max(80, int(disp_w) - 36)
        bar_h = 14
        bar_y = int(disp_h) - 34
        return bar_x, bar_w, bar_h, bar_y

    def _timeline_hit_test(self, state: EventState, x: int, y: int, pad: int = 8) -> bool:
        bar_x, bar_w, bar_h, bar_y = self._timeline_geometry(state.disp_w, state.disp_h)
        x0 = bar_x - int(pad)
        x1 = bar_x + bar_w + int(pad)
        y0 = bar_y - int(pad)
        y1 = bar_y + bar_h + int(pad)
        return x0 <= int(x) <= x1 and y0 <= int(y) <= y1

    def _seek_timeline_x(self, state: EventState, x: int) -> None:
        bar_x, bar_w, _bar_h, _bar_y = self._timeline_geometry(state.disp_w, state.disp_h)
        if state.frame_count <= 1:
            target = 0
        else:
            x_clamped = clamp(int(x), bar_x, bar_x + bar_w)
            ratio = float(x_clamped - bar_x) / float(max(1, bar_w))
            target = int(round(ratio * float(state.frame_count - 1)))
        target = clamp(target, 0, max(0, state.frame_count - 1))
        state.frame_idx = target
        _ = self._read_frame(state, target)

    def _draw_timeline(self, state: EventState, disp) -> None:
        h, w = disp.shape[:2]
        bar_x, bar_w, bar_h, bar_y = self._timeline_geometry(w, h)

        overlay = disp.copy()
        cv2.rectangle(overlay, (bar_x - 4, bar_y - 14), (bar_x + bar_w + 4, bar_y + bar_h + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, disp, 0.55, 0.0, disp)

        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)

        t_pair = state.transition.as_display_pair(state.frame_idx)
        i_pair = state.intrusion.as_display_pair(state.frame_idx)

        self._draw_interval_segment(
            disp,
            t_pair,
            frame_count=state.frame_count,
            x0=bar_x,
            y0=bar_y + 1,
            w=bar_w,
            h=bar_h - 2,
            color=(0, 220, 220),  # yellow-ish
        )
        self._draw_interval_segment(
            disp,
            i_pair,
            frame_count=state.frame_count,
            x0=bar_x,
            y0=bar_y + 1,
            w=bar_w,
            h=bar_h - 2,
            color=(0, 80, 255),  # red-ish
        )

        if state.show_source_interval and state.source_interval_orig is not None:
            src_c = (
                self._orig_to_clip(state, state.source_interval_orig[0]),
                self._orig_to_clip(state, state.source_interval_orig[1]),
            )
            self._draw_interval_segment(
                disp,
                src_c,
                frame_count=state.frame_count,
                x0=bar_x,
                y0=bar_y - 8,
                w=bar_w,
                h=4,
                color=(255, 200, 0),
            )

        if state.frame_count > 1:
            x_cur = bar_x + int(round((float(state.frame_idx) / float(state.frame_count - 1)) * float(bar_w)))
        else:
            x_cur = bar_x
        x_cur = clamp(x_cur, bar_x, bar_x + bar_w)
        cv2.line(disp, (x_cur, bar_y - 2), (x_cur, bar_y + bar_h + 2), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(
            disp,
            "timeline: transition(yellow) intrusion(red) source(cyan-top)",
            (bar_x, bar_y - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _draw_roi_warning(self, state: EventState, disp) -> None:
        if not state.roi_warning:
            return
        h, w = disp.shape[:2]
        box_x = 20
        box_w = max(200, w - 40)
        box_h = 96
        box_y = max(188, h // 2 - box_h // 2)

        overlay = disp.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.55, disp, 0.45, 0.0, disp)
        cv2.rectangle(disp, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)

        cv2.putText(
            disp,
            "WARNING",
            (box_x + 12, box_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            state.roi_warning,
            (box_x + 12, box_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        if state.roi_warning_detail:
            cv2.putText(
                disp,
                state.roi_warning_detail,
                (box_x + 12, box_y + 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    def _draw_hud(self, state: EventState, disp) -> None:
        h, w = disp.shape[:2]
        overlay = disp.copy()
        hud_h = 188
        cv2.rectangle(overlay, (0, 0), (w - 1, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.48, disp, 0.52, 0.0, disp)

        idx_txt = f"[{self.current_index + 1}/{len(self.events)}] {state.event.video_id} event_idx={state.event.event_idx}"
        cv2.putText(disp, idx_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)

        orig_idx = self._clip_to_orig(state, state.frame_idx)
        frame_txt = (
            f"clip_frame={state.frame_idx}/{max(0, state.frame_count - 1)}  "
            f"orig_frame={orig_idx}  mode={state.mode.upper()}  "
            f"fps_clip={state.fps_clip:.3f} fps_src={state.event.fps_src:.3f}"
        )
        cv2.putText(disp, frame_txt, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1, cv2.LINE_AA)

        state_txt = (
            f"transition={state.transition.status_text()}  intrusion={state.intrusion.status_text()}  "
            f"show_src={'ON' if state.show_source_interval else 'OFF'}"
        )
        cv2.putText(disp, state_txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 255), 1, cv2.LINE_AA)

        roi_info_txt = (
            f"roi_file={state.roi_path.name}  roi_vertices_count={len(state.roi_points)}  roi_target_n={state.roi_target_n}"
        )
        cv2.putText(disp, roi_info_txt, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 255), 1, cv2.LINE_AA)

        dirty_txt = (
            f"dirty_roi={'Y' if state.roi_dirty else 'N'}  dirty_labels={'Y' if state.labels_dirty else 'N'}  "
            f"save_status={'UNSAVED' if state.is_dirty() else 'SAVED'}"
        )
        cv2.putText(disp, dirty_txt, (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 190, 120), 1, cv2.LINE_AA)

        cv2.putText(disp, HUD_HELP_LINE, (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)

        tip_txt = "Tip: if clip start alignment looks off, regenerate clips with --reencode."
        cv2.putText(disp, tip_txt, (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

        note = self.current_note()
        if note:
            cv2.putText(disp, note, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2, cv2.LINE_AA)

    def _render(self, state: EventState):
        frame = self._read_frame(state, state.frame_idx)
        if frame is None:
            disp = make_canvas(state.disp_w, state.disp_h, color=(0, 0, 0))
            cv2.putText(disp, "Frame read failed", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return disp

        if state.disp_w != state.frame_w or state.disp_h != state.frame_h:
            disp = cv2.resize(frame, (state.disp_w, state.disp_h), interpolation=cv2.INTER_LINEAR)
        else:
            disp = frame.copy()

        self._draw_roi_overlay(state, disp)
        self._draw_timeline(state, disp)
        self._draw_hud(state, disp)
        self._draw_roi_warning(state, disp)
        return disp

    def _save_roi(self, state: EventState) -> None:
        if len(state.roi_points) < 3:
            raise RuntimeError("ROI requires at least 3 vertices.")

        payload = copy.deepcopy(state.roi_obj) if isinstance(state.roi_obj, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        payload["video_id"] = state.event.video_id
        payload["roi_id"] = payload.get("roi_id", f"roi_{self.args.roi_name}")
        payload["roi_version"] = payload.get("roi_version", self.args.roi_version)
        payload["vertices_px"] = [[int(x), int(y)] for (x, y) in state.roi_points]
        image_size = payload.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {}
        image_size["width"] = int(state.frame_w)
        image_size["height"] = int(state.frame_h)
        payload["image_size"] = image_size
        payload["updated_at"] = now_iso()

        write_json_with_backup(state.roi_path, payload)
        state.roi_obj = payload
        state.roi_dirty = False

    def _build_event_payload(self, state: EventState) -> dict[str, Any]:
        trans_clip = state.transition.as_closed_pair()
        intr_clip = state.intrusion.as_closed_pair()
        trans_orig = self._interval_clip_to_orig(state, trans_clip)
        intr_orig = self._interval_clip_to_orig(state, intr_clip)

        payload = {
            "event_idx": int(state.event.event_idx),
            "source": {
                "manifest_path": str(self.args.manifest),
                "event_idx": int(state.event.event_idx),
                "clip_path": state.event.clip_path_raw,
                "fps": float(state.event.fps_src),
                "clip_start_sec": float(state.event.clip_start_sec),
            },
            "annotations": {
                "intrusion_frame": pair_to_nested(intr_orig),
                "transition_frame": pair_to_nested(trans_orig),
                "event_frame": pair_to_nested(intr_orig),
            },
            "annotations_clip": {
                "intrusion_frame": pair_to_nested(intr_clip),
                "transition_frame": pair_to_nested(trans_clip),
            },
            "meta": {
                "tool": TOOL_NAME,
                "updated_at": now_iso(),
                "policy": POLICY_TEXT,
            },
        }
        return payload

    def _save_labels(self, state: EventState) -> None:
        path = self._labels_user_path(state.event.video_id)
        root = load_json_dict(path)
        if root is None:
            root = {}
        if not isinstance(root, dict):
            root = {}

        events_obj = root.get("events")
        if not isinstance(events_obj, dict):
            events_obj = {}

        events_obj[str(state.event.event_idx)] = self._build_event_payload(state)
        root["video_id"] = state.event.video_id
        root["events"] = events_obj

        meta_obj = root.get("meta")
        if not isinstance(meta_obj, dict):
            meta_obj = {}
        meta_obj["tool"] = TOOL_NAME
        meta_obj["updated_at"] = now_iso()
        meta_obj["policy"] = POLICY_TEXT
        root["meta"] = meta_obj

        atomic_write_json(path, root)
        state.labels_dirty = False

    def save_current(self) -> bool:
        state = self.state
        if state is None:
            return False
        if state.intrusion.as_closed_pair() is None:
            self.set_note("Intrusion interval required (use i to start, o to end).", duration=2.0)
            return False

        adjust_note = self._enforce_transition_before_intrusion(state)

        try:
            self._save_roi(state)
            self._save_labels(state)
        except Exception as e:
            self.set_note(f"Save failed: {e}", duration=2.2)
            return False
        state.last_saved_at = now_iso()
        if adjust_note:
            self.set_note(
                f"Saved ROI ({state.roi_path.name}) and labels_user ({state.event.video_id}.json) | {adjust_note}",
                duration=2.6,
            )
        else:
            self.set_note(
                f"Saved ROI ({state.roi_path.name}) and labels_user ({state.event.video_id}.json)",
                duration=2.0,
            )
        return True

    def _enforce_transition_before_intrusion(self, state: EventState) -> Optional[str]:
        trans = state.transition.as_closed_pair()
        intr = state.intrusion.as_closed_pair()
        if trans is None or intr is None:
            return None

        t_start, t_end = trans
        i_start, _i_end = intr

        if t_start > i_start:
            state.transition.clear()
            state.labels_dirty = True
            return "Transition cleared (must precede intrusion)."

        if t_end >= i_start:
            state.transition.end = max(t_start, i_start - 1)
            if state.transition.start is None or state.transition.end is None:
                state.transition.clear()
                state.labels_dirty = True
                return "Transition cleared after overlap adjustment."
            if state.transition.end < state.transition.start:
                state.transition.clear()
                state.labels_dirty = True
                return "Transition cleared after overlap adjustment."
            state.labels_dirty = True
            return f"Transition end auto-adjusted to {state.transition.end}."

        return None

    def _set_roi_target_n(self, n: int) -> None:
        state = self.state
        if state is None:
            return
        new_n = max(3, int(n))
        state.roi_target_n = new_n
        self.set_note(f"ROI N set to {new_n}")

    def _digit_from_key(self, key: int) -> Optional[int]:
        low = key & 0xFF
        if 48 <= low <= 57:
            return low - 48
        if 96 <= low <= 105:
            return low - 96
        if 65456 <= key <= 65465:
            return key - 65456
        if 1114112 <= key <= 1114121:
            return key - 1114112
        return None

    def _print_help_summary(self) -> None:
        print("")
        print("=" * 84)
        print("Transition/Intrusion Annotator Keybindings")
        print("- Playback/Seek: space, left/right, shift+left/right, j/l, J/L, [, ], n/p, q (dirty 시 1초 내 2회)")
        print("- Modes: 1 ROI, 2 transition, 3 intrusion")
        print("- Marking: a/s (transition start/end), i/o (intrusion start/end)")
        print("- Save/Undo/Reset: w / u / r")
        print("- Toggle source label: t")
        print("- Key code debug: k")
        print("- ROI vertex target N: 4-9")
        print("- Mouse timeline seek: click/drag bottom timeline bar")
        print("- Notes: intrusion required to save; transition optional and auto-adjusted before intrusion.")
        print("- HUD help: " + HUD_HELP_LINE)
        print("=" * 84)
        print("")

    def _has_shift_modifier_variant(self, key: int, base_codes: set[int]) -> bool:
        # Heuristic: if the lower 24 bits match a known arrow key but full key differs,
        # treat as a modified arrow (often shift on some backends).
        masked = key & 0x00FFFFFF
        for c in base_codes:
            if (c & 0x00FFFFFF) == masked and key != c:
                return True
        return False

    def _classify_arrow(self, key: int) -> Optional[str]:
        if key in SHIFT_LEFT_CODES or self._has_shift_modifier_variant(key, LEFT_CODES):
            return "shift_left"
        if key in SHIFT_RIGHT_CODES or self._has_shift_modifier_variant(key, RIGHT_CODES):
            return "shift_right"
        if key in LEFT_CODES:
            return "left"
        if key in RIGHT_CODES:
            return "right"
        return None

    def _step_frames(self, delta: int) -> None:
        state = self.state
        if state is None:
            return
        prev = state.frame_idx
        tgt = clamp(prev + int(delta), 0, max(0, state.frame_count - 1))
        state.frame_idx = tgt
        _ = self._read_frame(state, tgt)
        if tgt == prev and delta > 0:
            self.playing = False

    def _jump_seconds(self, sign: int) -> None:
        state = self.state
        if state is None:
            return
        jump = max(1, int(round(float(state.fps_clip))))
        self._step_frames(sign * jump)

    def _mark_start(self, kind: str) -> None:
        state = self.state
        if state is None:
            return
        if kind == MODE_TRANSITION:
            if state.mode != MODE_TRANSITION:
                self.set_note("Switch to transition mode (key 2) before using a/s.")
                return
            self._push_undo(state)
            state.transition.start = int(state.frame_idx)
            state.transition.end = None
        elif kind == MODE_INTRUSION:
            if state.mode != MODE_INTRUSION:
                self.set_note("Switch to intrusion mode (key 3) before using i/o.")
                return
            self._push_undo(state)
            state.intrusion.start = int(state.frame_idx)
            state.intrusion.end = None
        else:
            return
        state.labels_dirty = True
        self.set_note(f"{kind} start @ clip frame {state.frame_idx}")

    def _mark_end(self, kind: str) -> None:
        state = self.state
        if state is None:
            return
        if kind == MODE_TRANSITION:
            if state.mode != MODE_TRANSITION:
                self.set_note("Switch to transition mode (key 2) before using a/s.")
                return
            if state.transition.start is None:
                self.set_note("Transition start not set. Use a first.")
                return
            self._push_undo(state)
            state.transition.end = int(state.frame_idx)
            if state.transition.end < state.transition.start:
                state.transition.start, state.transition.end = state.transition.end, state.transition.start
        elif kind == MODE_INTRUSION:
            if state.mode != MODE_INTRUSION:
                self.set_note("Switch to intrusion mode (key 3) before using i/o.")
                return
            if state.intrusion.start is None:
                self.set_note("Intrusion start not set. Use i first.")
                return
            self._push_undo(state)
            state.intrusion.end = int(state.frame_idx)
            if state.intrusion.end < state.intrusion.start:
                state.intrusion.start, state.intrusion.end = state.intrusion.end, state.intrusion.start
        else:
            return
        state.labels_dirty = True
        self.set_note(f"{kind} end @ clip frame {state.frame_idx}")

    def _reset_intervals(self) -> None:
        state = self.state
        if state is None:
            return
        if (
            state.transition.start is None
            and state.transition.end is None
            and state.intrusion.start is None
            and state.intrusion.end is None
        ):
            self.set_note("No intervals to reset.")
            return
        self._push_undo(state)
        state.transition.clear()
        state.intrusion.clear()
        state.labels_dirty = True
        self.set_note("Transition + intrusion intervals reset.")

    def _confirm_if_dirty(self, key: str, message: str) -> bool:
        state = self.state
        if state is None:
            return False
        if not state.is_dirty():
            return False
        now = time.monotonic()
        prev = self.pending_nav.get(key, 0.0)
        if now - prev <= 1.0:
            self.pending_nav[key] = 0.0
            return False
        self.pending_nav[key] = now
        self.set_note(message, duration=1.2)
        return True

    def _goto_index(self, index: int) -> None:
        if index < 0 or index >= len(self.events):
            self.set_note("No more clips in that direction.")
            return
        self.load_event(index)

    def on_mouse(self, event, x, y, flags, _userdata) -> None:
        state = self.state
        if state is None:
            return

        x_i = clamp(int(x), 0, max(0, state.disp_w - 1))
        y_i = clamp(int(y), 0, max(0, state.disp_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN and self._timeline_hit_test(state, x_i, y_i, pad=8):
            state.dragging_timeline = True
            state.dragging_vertex = None
            self.playing = False
            self._seek_timeline_x(state, x_i)
            return

        if event == cv2.EVENT_MOUSEMOVE and state.dragging_timeline:
            if int(flags) & int(cv2.EVENT_FLAG_LBUTTON):
                self.playing = False
                self._seek_timeline_x(state, x_i)
            return

        if event == cv2.EVENT_LBUTTONUP and state.dragging_timeline:
            state.dragging_timeline = False
            return

        if state.mode != MODE_ROI:
            return

        fx, fy = self._disp_to_frame(state, x_i, y_i)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state.roi_points) < state.roi_target_n:
                self._push_undo(state)
                state.roi_points.append((fx, fy))
                state.selected_vertex = len(state.roi_points) - 1
                state.roi_dirty = True
                self.set_note(f"ROI vertex added ({len(state.roi_points)}/{state.roi_target_n})")
                return

            idx, _d2 = self._nearest_vertex(state, x_i, y_i)
            if idx is not None:
                self._push_undo(state)
                state.dragging_vertex = idx
                state.selected_vertex = idx
                state.roi_points[idx] = (fx, fy)
                state.roi_dirty = True
            return

        if event == cv2.EVENT_MOUSEMOVE and state.dragging_vertex is not None:
            if int(flags) & int(cv2.EVENT_FLAG_LBUTTON):
                idx = int(state.dragging_vertex)
                if 0 <= idx < len(state.roi_points):
                    state.roi_points[idx] = (fx, fy)
                    state.roi_dirty = True
            return

        if event == cv2.EVENT_LBUTTONUP:
            state.dragging_vertex = None
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            idx, d2 = self._nearest_vertex(state, x_i, y_i)
            if idx is None:
                return
            # Avoid accidental deletion when far away.
            if d2 > float(24 * 24):
                return
            self._push_undo(state)
            state.roi_points.pop(idx)
            state.selected_vertex = None
            state.dragging_vertex = None
            state.roi_dirty = True
            self.set_note("Deleted nearest ROI vertex.")
            return

    def _handle_key(self, key: int) -> Optional[str]:
        state = self.state
        if state is None:
            return None

        arrow = self._classify_arrow(key)
        if arrow == "shift_left":
            self._step_frames(-10)
            return None
        if arrow == "shift_right":
            self._step_frames(10)
            return None
        if arrow == "left":
            self._step_frames(-1)
            return None
        if arrow == "right":
            self._step_frames(1)
            return None

        low = key & 0xFF
        char_raw = chr(low) if 32 <= low <= 126 else ""
        char = char_raw.lower()
        digit = self._digit_from_key(key)

        if key == 32 or char == " ":
            self.playing = not self.playing
            self.last_tick = time.monotonic()
            self.set_note("Play" if self.playing else "Pause", duration=0.6)
            return None

        if char_raw == "j":
            self._step_frames(-1)
            return None
        if char_raw == "l":
            self._step_frames(1)
            return None
        if char_raw == "J":
            self._step_frames(-10)
            return None
        if char_raw == "L":
            self._step_frames(10)
            return None

        if digit is not None and 4 <= digit <= 9:
            self._set_roi_target_n(digit)
            return None

        if char == "[":
            self._jump_seconds(-1)
            return None
        if char == "]":
            self._jump_seconds(1)
            return None

        if char == "1":
            state.mode = MODE_ROI
            self.set_note("Mode: ROI edit")
            return None
        if char == "2":
            state.mode = MODE_TRANSITION
            self.set_note("Mode: Mark transition")
            return None
        if char == "3":
            state.mode = MODE_INTRUSION
            self.set_note("Mode: Mark intrusion")
            return None

        if char == "t":
            state.show_source_interval = not state.show_source_interval
            self.set_note(f"Source label overlay {'ON' if state.show_source_interval else 'OFF'}")
            return None

        if char == "k":
            print(f"[KEYDBG] last_key_code={self.last_key_code} current_key_code={key}")
            self.set_note("Key code printed to console.", duration=1.0)
            return None

        if char == "h":
            self._print_help_summary()
            self.set_note("Help printed to console.", duration=1.2)
            return None

        if char == "a":
            self._mark_start(MODE_TRANSITION)
            return None
        if char == "s":
            self._mark_end(MODE_TRANSITION)
            return None

        if char == "i":
            self._mark_start(MODE_INTRUSION)
            return None
        if char == "o":
            self._mark_end(MODE_INTRUSION)
            return None

        if char == "u":
            self._undo(state)
            return None

        if char == "r":
            self._reset_intervals()
            return None

        if char == "w":
            self.save_current()
            return None

        if char == "n":
            if self._confirm_if_dirty("n", "Unsaved changes. Press n again within 1s to go next."):
                return None
            self._goto_index(self.current_index + 1)
            return None

        if char == "p":
            if self._confirm_if_dirty("p", "Unsaved changes. Press p again within 1s to go previous."):
                return None
            self._goto_index(self.current_index - 1)
            return None

        if char == "q":
            if state.is_dirty():
                now = time.monotonic()
                if now - self.last_quit_press <= 1.0:
                    return "quit"
                self.last_quit_press = now
                self.set_note("Unsaved changes. Press q again within 1s to quit.", duration=1.2)
                return None
            return "quit"

        if not char_raw:
            print(f"[KEYDBG] unknown non-char key code={key}")

        return None

    def run(self) -> int:
        if not self.events:
            print("ERROR: No manifest clips to annotate.")
            return 1

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        if not self.load_event(self.current_index):
            print("ERROR: Failed to load initial event.")
            cv2.destroyAllWindows()
            return 1

        try:
            while True:
                state = self.state
                if state is None:
                    break

                if self.playing:
                    now = time.monotonic()
                    frame_dt = 1.0 / max(1e-6, state.fps_clip)
                    if now - self.last_tick >= frame_dt:
                        self._step_frames(1)
                        self.last_tick = now

                canvas = self._render(state)
                cv2.imshow(WINDOW_NAME, canvas)

                key = cv2.waitKeyEx(10)
                if key != -1 and key != 255:
                    action = self._handle_key(key)
                    self.last_key_code = int(key)
                    if action == "quit":
                        break
        finally:
            self.close_state()
            cv2.destroyAllWindows()
        return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if cv2 is None:
        print("ERROR: OpenCV is required for this tool.")
        return 1

    manifest_path = Path(args.manifest)
    try:
        events = load_manifest_events(manifest_path=manifest_path, choice=args.choice)
    except Exception as e:
        print(f"ERROR: Failed to load manifest: {e}")
        return 1

    if not events:
        print("ERROR: No usable events found in manifest after filtering.")
        return 1

    app = AnnotatorApp(args=args, events=events)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
