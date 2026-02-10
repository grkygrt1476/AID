#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

STAGE = "00_prep"

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def save_cmd(cmd_path: Path, argv: List[str], project_root: Path) -> None:
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    # best-effort reproducible cmd
    line = "cd " + str(project_root) + " && " + " ".join(argv) + "\n"
    cmd_path.write_text(line, encoding="utf-8")

def read_label_event_start(label_json: Path, event_index: int = 0) -> Tuple[Optional[int], Dict[str, Any]]:
    meta: Dict[str, Any] = {"label_path": str(label_json)}
    if not label_json.exists():
        return None, {**meta, "label_exists": False}

    try:
        obj = json.loads(label_json.read_text(encoding="utf-8"))
    except Exception as e:
        return None, {**meta, "label_exists": True, "label_parse_ok": False, "label_error": str(e)}

    an = obj.get("annotations", {}) or {}
    md = obj.get("metadata", {}) or {}
    event_frames = an.get("event_frame", []) or []

    meta.update({
        "label_exists": True,
        "label_parse_ok": True,
        "event_class": an.get("event_class", ""),
        "event_frames": event_frames,
        "event_length": an.get("event_length", None),
        "metadata": {
            "file_name": md.get("file_name", ""),
            "width": md.get("width", None),
            "height": md.get("height", None),
            "frame_count": md.get("frame_count", None),
            "night": md.get("night", None),
            "date": md.get("date", None),
        },
    })

    if not event_frames or event_index >= len(event_frames):
        return None, meta

    try:
        start = int(event_frames[event_index][0])
        return start, meta
    except Exception:
        return None, meta

def pick_frame_index(
    cap: cv2.VideoCapture,
    mode: str,
    label_start: Optional[int],
    offset: int,
) -> int:
    # Try to get frame count; if unreliable, fallback to 0.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if mode == "first":
        base = 0
    elif mode == "middle":
        base = max(0, frame_count // 2) if frame_count > 0 else 0
    else:  # event_start
        base = label_start if label_start is not None else (max(0, frame_count // 2) if frame_count > 0 else 0)

    idx = base + int(offset)
    if frame_count > 0:
        idx = max(0, min(idx, frame_count - 1))
    else:
        idx = max(0, idx)
    return idx

def maybe_resize(img, max_side: int):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    scale = max_side / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to mp4 (e.g., data/videos/E01_004.mp4)")
    p.add_argument("--label_json", default="", help="optional label json (e.g., data/labels/E01_004.json)")
    p.add_argument("--frame_mode", default="event_start", choices=["event_start", "middle", "first"])
    p.add_argument("--event_index", type=int, default=0, help="which event_frame to use if multiple")
    p.add_argument("--offset", type=int, default=0, help="frame offset from chosen base frame")
    p.add_argument("--video_id", default="", help="override video id (default: stem of video filename)")
    p.add_argument("--out_image", default="", help="output jpg path (default: data/previews/{video_id}.jpg)")
    p.add_argument("--out_meta", default="", help="output meta json path (default: data/previews/{video_id}.meta.json)")
    p.add_argument("--max_side", type=int, default=0, help="if >0, also save a resized display image *and* keep original; display image ends with _disp.jpg")
    p.add_argument("--run_ts", default="", help="run timestamp (default: now)")
    return p.parse_args()

def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)

    run_ts = args.run_ts or now_ts()

    script_stem = Path(__file__).stem
    log_dir = project_root / "outputs" / "logs" / STAGE
    out_dir = project_root / "outputs" / STAGE / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_path = log_dir / f"{script_stem}_{run_ts}.cmd.txt"
    log_path = log_dir / f"{script_stem}_{run_ts}.log"
    save_cmd(cmd_path, ["python", str(Path(__file__).as_posix())] + [a for a in os.sys.argv[1:]], project_root)
    setup_logger(log_path)

    logging.info(f"cmd saved: {cmd_path}")
    logging.info(f"log saved: {log_path}")
    logging.info(f"out_dir: {out_dir}")

    video_path = Path(args.video)
    if not video_path.exists():
        logging.error(f"video not found: {video_path}")
        return 2

    video_id = args.video_id.strip() or video_path.stem

    label_meta: Dict[str, Any] = {}
    label_start: Optional[int] = None
    if args.label_json:
        label_start, label_meta = read_label_event_start(Path(args.label_json), event_index=args.event_index)
        logging.info(f"label_start(frame): {label_start}")
    else:
        label_meta = {"label_path": None, "label_exists": False}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"failed to open video: {video_path}")
        return 3

    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_idx = pick_frame_index(cap, args.frame_mode, label_start, args.offset)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        logging.error(f"failed to read frame at idx={frame_idx}")
        return 4

    out_image = Path(args.out_image) if args.out_image else Path("data/previews") / f"{video_id}.jpg"
    out_meta = Path(args.out_meta) if args.out_meta else Path("data/previews") / f"{video_id}.meta.json"
    out_image.parent.mkdir(parents=True, exist_ok=True)

    # Save original resolution snapshot
    cv2.imwrite(str(out_image), frame)
    logging.info(f"saved snapshot: {out_image} (orig {width}x{height}, frame_idx={frame_idx})")

    disp_path = None
    disp_scale = 1.0
    if args.max_side and args.max_side > 0:
        disp, disp_scale = maybe_resize(frame, args.max_side)
        disp_path = out_image.with_name(out_image.stem + "_disp.jpg")
        cv2.imwrite(str(disp_path), disp)
        logging.info(f"saved display snapshot: {disp_path} (scale={disp_scale:.4f})")

    meta: Dict[str, Any] = {
        "video_id": video_id,
        "video_path": str(video_path),
        "snapshot_frame": frame_idx,
        "frame_mode": args.frame_mode,
        "offset": args.offset,
        "fps": float(fps) if fps else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "image_size": [width, height],
        "snapshot_path": str(out_image),
        "snapshot_disp_path": str(disp_path) if disp_path else None,
        "snapshot_disp_scale": disp_scale,
        "label": label_meta,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"saved meta: {out_meta}")

    # Small artifact for quick look
    (out_dir / f"snapshot_{video_id}.txt").write_text(
        f"video_id={video_id}\nvideo={video_path}\nframe_idx={frame_idx}\nimage={out_image}\n",
        encoding="utf-8"
    )

    logging.info("[DONE]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
