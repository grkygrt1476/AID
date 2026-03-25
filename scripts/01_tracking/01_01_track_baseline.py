#!/usr/bin/env python3
import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CV2_MOD = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO person-only tracking on one clip and write tracked.mp4 via ffmpeg (libx264)."
    )
    parser.add_argument("--clip", required=True, help="Input mp4 path")
    parser.add_argument("--weights", required=True, help="YOLO weights path (e.g., weights/person/yolo11s.pt)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker yaml name or path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--device", default="0", help='CUDA device index (e.g., "0") or "cpu"')
    parser.add_argument(
        "--out_root",
        default="outputs/01_tracking",
        help="relative output directory",
    )
    parser.add_argument(
        "--out_name",
        default="",
        help="Output folder name; if empty, auto-generates as RUN_TS_clipstem",
    )
    parser.add_argument("--crf", type=int, default=24, help="x264 CRF")
    parser.add_argument("--preset", default="veryfast", help="x264 preset")
    parser.add_argument("--max_frames", type=int, default=-1, help="If >0, stop after N frames")
    parser.add_argument("--roi_json", default="", help="Path to ROI json file")
    parser.add_argument("--roi_video_id", default="", help="Video ID used to infer ROI json")
    parser.add_argument("--roi_glob", default="roi_*.json", help="ROI filename glob under configs/rois/<video_id>/")
    parser.add_argument("--roi_color", default="0,255,0", help="ROI color in BGR format, e.g. 0,255,0")
    parser.add_argument("--roi_thickness", type=int, default=2, help="ROI polyline thickness")
    parser.add_argument("--roi_draw_fill", action="store_true", help="Draw semi-transparent ROI fill")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--smooth_alpha", type=float, default=0.0, help="EMA alpha for bbox smoothing; 0 disables")
    return parser.parse_args()

def ensure_paths(args):
    clip_path = Path(args.clip).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()

    if not clip_path.exists():
        print(f"Error: clip not found: {clip_path}")
        raise SystemExit(1)
    if not weights_path.exists():
        print(f"Error: weights not found: {weights_path}")
        raise SystemExit(1)

    repo_root = Path(__file__).resolve().parents[2]

    out_root_arg = Path(args.out_root).expanduser()
    out_root = out_root_arg if out_root_arg.is_absolute() else (repo_root / out_root_arg)
    out_root = out_root.resolve()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name if args.out_name else f"{run_ts}_{clip_path.stem}"

    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return clip_path, weights_path, out_dir


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_cmd(path):
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    path.write_text(cmd + "\n", encoding="utf-8")


def infer_roi_json(clip_path, repo_root, roi_json, roi_video_id, roi_glob):
    if roi_json:
        roi_path = Path(roi_json).expanduser().resolve()
        if roi_path.exists():
            return roi_path, ""

    inferred_video_id = roi_video_id.strip()
    if not inferred_video_id:
        parts = clip_path.resolve().parts
        if "clips" in parts:
            idx = parts.index("clips")
            if idx + 1 < len(parts):
                inferred_video_id = parts[idx + 1]

    if not inferred_video_id:
        return None, ""

    roi_dir = (repo_root / "configs" / "rois" / inferred_video_id).resolve()
    if not roi_dir.exists() or not roi_dir.is_dir():
        return None, inferred_video_id

    matches = sorted([p for p in roi_dir.glob(roi_glob) if p.is_file()])
    if not matches:
        return None, inferred_video_id

    fix_candidates = [p for p in matches if "_fix" in p.name]
    suffix_fix = [p for p in fix_candidates if p.name.endswith("_fix.json")]
    if suffix_fix:
        return suffix_fix[0], inferred_video_id
    if fix_candidates:
        return fix_candidates[0], inferred_video_id
    return matches[0], inferred_video_id


def load_roi_poly(roi_json_path, frame_width, frame_height):
    if roi_json_path is None:
        return None, ""

    try:
        data = json.loads(roi_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None, ""

    points = None
    key_used = ""
    used_vertices_px = False
    if isinstance(data, dict):
        vertices_px = data.get("vertices_px")
        if isinstance(vertices_px, list) and len(vertices_px) > 0:
            points = vertices_px
            key_used = "vertices_px"
            used_vertices_px = True
        for key in ("poly", "vertices", "points"):
            if points is not None:
                break
            value = data.get(key)
            if isinstance(value, list):
                points = value
                key_used = key
                break
        if points is None:
            roi_obj = data.get("roi")
            if isinstance(roi_obj, dict) and isinstance(roi_obj.get("vertices"), list):
                points = roi_obj.get("vertices")
                key_used = "roi.vertices"
    if not isinstance(points, list) or len(points) < 3:
        return None, ""

    parsed = []
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            parsed.append([float(pt[0]), float(pt[1])])
        elif isinstance(pt, dict) and "x" in pt and "y" in pt:
            parsed.append([float(pt["x"]), float(pt["y"])])

    if len(parsed) < 3:
        return None, ""

    poly = np.array(parsed, dtype=np.float32)

    src_width = data.get("width") if isinstance(data, dict) else None
    src_height = data.get("height") if isinstance(data, dict) else None
    if isinstance(data, dict):
        image_size = data.get("image_size")
        if isinstance(image_size, dict):
            src_width = image_size.get("width", src_width)
            src_height = image_size.get("height", src_height)

    if used_vertices_px:
        if src_width and src_height and float(src_width) > 0 and float(src_height) > 0:
            if int(float(src_width)) != int(frame_width) or int(float(src_height)) != int(frame_height):
                sx = float(frame_width) / float(src_width)
                sy = float(frame_height) / float(src_height)
                poly[:, 0] *= sx
                poly[:, 1] *= sy
    else:
        if np.max(poly) <= 1.5:
            poly[:, 0] *= float(frame_width)
            poly[:, 1] *= float(frame_height)
        elif src_width and src_height and float(src_width) > 0 and float(src_height) > 0:
            sx = float(frame_width) / float(src_width)
            sy = float(frame_height) / float(src_height)
            poly[:, 0] *= sx
            poly[:, 1] *= sy

    return np.round(poly).astype(np.int32), key_used


def draw_roi_overlay(frame, poly, bgr, thickness, draw_fill, cv2_mod):
    if poly is None:
        return frame
    if draw_fill:
        overlay = frame.copy()
        cv2_mod.fillPoly(overlay, [poly], bgr)
        cv2_mod.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2_mod.polylines(frame, [poly], isClosed=True, color=bgr, thickness=thickness)
    return frame


def draw_boxes(frame, results, smooth_alpha, ema_bbox, ema_seen, frame_idx):
    prune_every = 120
    ttl = 120

    if frame_idx > 0 and frame_idx % prune_every == 0:
        stale = [tid for tid, last_seen in ema_seen.items() if frame_idx - last_seen > ttl]
        for tid in stale:
            ema_seen.pop(tid, None)
            ema_bbox.pop(tid, None)

    boxes = results.boxes
    if boxes is None or boxes.id is None or boxes.xyxy is None:
        return frame

    xyxy = boxes.xyxy.detach().cpu().numpy()
    track_ids = boxes.id.detach().cpu().numpy().astype(int)
    h, w = frame.shape[:2]

    for box, track_id in zip(xyxy, track_ids):
        box = np.asarray(box, dtype=np.float32)
        prev = ema_bbox.get(track_id)
        if prev is None:
            smoothed = box
        else:
            smoothed = smooth_alpha * box + (1.0 - smooth_alpha) * prev
        ema_bbox[track_id] = smoothed
        ema_seen[track_id] = frame_idx

        x1, y1, x2, y2 = np.round(smoothed).astype(int)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        CV2_MOD.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = y1 - 8 if y1 > 18 else y1 + 18
        CV2_MOD.putText(
            frame,
            f"id {track_id}",
            (x1, text_y),
            CV2_MOD.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            CV2_MOD.LINE_AA,
        )
    return frame


def open_ffmpeg_writer(out_path, width, height, fps, preset, crf):
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Install it with: sudo apt-get update && sudo apt-get install -y ffmpeg")
        sys.exit(1)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    global CV2_MOD
    args = parse_args()
    clip_path, weights_path, out_dir = ensure_paths(args)
    tracked_path = out_dir / "tracked.mp4"
    repo_root = Path(__file__).resolve().parents[2]

    try:
        import cv2
    except Exception as e:
        print("Error: failed to import OpenCV (cv2). Install it with: pip install opencv-python")
        print(f"Details: {e}")
        sys.exit(1)
    CV2_MOD = cv2

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"Error: cannot open input video: {clip_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 30.0

    roi_json_path, roi_inferred_video_id = infer_roi_json(
        clip_path=clip_path,
        repo_root=repo_root,
        roi_json=args.roi_json,
        roi_video_id=args.roi_video_id,
        roi_glob=args.roi_glob,
    )
    roi_poly, roi_key_used = load_roi_poly(
        roi_json_path=roi_json_path,
        frame_width=width,
        frame_height=height,
    )
    roi_enabled = roi_poly is not None

    try:
        roi_color = tuple(int(x.strip()) for x in args.roi_color.split(","))
        if len(roi_color) != 3:
            raise ValueError("roi_color needs three values")
    except Exception:
        roi_color = (0, 255, 0)

    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as e:
        print("Error: failed to import ultralytics. Install it with: pip install ultralytics")
        print(f"Details: {e}")
        sys.exit(1)

    ffmpeg_proc = open_ffmpeg_writer(
        out_path=tracked_path,
        width=width,
        height=height,
        fps=fps,
        preset=args.preset,
        crf=args.crf,
    )

    model = YOLO(str(weights_path))
    stream = model.track(
        source=str(clip_path),
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        classes=0,
        save=False,
        verbose=False,
    )

    total_frames_processed = 0
    ema_bbox = {}
    ema_seen = {}
    try:
        for r in stream:
            if args.smooth_alpha > 0.0:
                frame = r.orig_img.copy() if r.orig_img is not None else r.plot()
                frame = draw_boxes(
                    frame=frame,
                    results=r,
                    smooth_alpha=args.smooth_alpha,
                    ema_bbox=ema_bbox,
                    ema_seen=ema_seen,
                    frame_idx=total_frames_processed,
                )
            else:
                frame = r.plot()
            if roi_enabled:
                frame = draw_roi_overlay(
                    frame=frame,
                    poly=roi_poly,
                    bgr=roi_color,
                    thickness=args.roi_thickness,
                    draw_fill=args.roi_draw_fill,
                    cv2_mod=cv2,
                )
            ffmpeg_proc.stdin.write(frame.tobytes())
            total_frames_processed += 1
            if args.max_frames > 0 and total_frames_processed >= args.max_frames:
                break
    finally:
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        err = ""
        if ffmpeg_proc.stderr:
            err = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace").strip()
        print("Error: ffmpeg encoding failed.")
        if err:
            print(err)
        sys.exit(1)

    write_json(out_dir / "params.json", vars(args))
    write_json(
        out_dir / "meta.json",
        {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames_processed,
            "ultralytics_version": ultralytics.__version__,
            "roi_enabled": roi_enabled,
            "roi_json_used": str(roi_json_path) if roi_json_path else "",
            "roi_inferred_video_id": roi_inferred_video_id,
            "roi_key_used": roi_key_used,
            "smooth_alpha": args.smooth_alpha,
        },
    )
    write_cmd(out_dir / "cmd.txt")

    print(f"Output directory: {out_dir}")
    print(f"Final video: {tracked_path}")


if __name__ == "__main__":
    main()
