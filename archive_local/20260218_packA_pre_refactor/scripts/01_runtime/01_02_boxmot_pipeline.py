#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aidlib import run_utils  # noqa: E402

'''
python3 scripts/01_runtime/01_02_boxmot_pipeline.py \
  --out_base all_boxmot \
  --clip_sec 30 --clip_mode center \
  --detector yolov8m --reid osnet_x0_25_msmt17 --tracker deepocsort \
  --device 0 --imgsz 960 --conf 0.35 \
  --vid_stride 2 \
  --skip_existing

'''
STAGE = "01_runtime"


@dataclass
class EventItem:
    video_id: str
    video_path: Path
    label_path: Path
    event_idx: int
    s_frame: int
    e_frame: int


def build_parser():
    p = run_utils.common_argparser()

    # Inputs
    p.add_argument("--labels_dir", default="data/labels", help="label json directory")
    p.add_argument("--videos_dir", default="data/videos", help="video directory")
    p.add_argument("--label_glob", default="E01_*.json", help="glob pattern under labels_dir")
    p.add_argument("--video_id", default="", help="if set, only process this video_id (e.g., E01_009)")
    p.add_argument("--event_index", type=int, default=-1, help="-1 = all events, else only that index")

    # Clip
    p.add_argument("--clip_sec", type=float, default=30.0, help="clip duration in seconds")
    p.add_argument("--clip_mode", choices=["center", "start"], default="center")
    p.add_argument("--pre_sec", type=float, default=5.0, help="used when clip_mode=start (start = s_frame - pre_sec)")
    p.add_argument("--reencode", action="store_true", help="reencode with libx264 (robust). default=True")
    p.add_argument("--copy", action="store_true", help="use stream copy (-c copy), faster but keyframe-bound")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--clip_dir", default="", help="if empty, use run_out/clips")

    # BoxMOT
    p.add_argument("--detector", default="yolov8m", help="boxmot detector name (e.g., yolov8m)")
    p.add_argument("--reid", default="osnet_x0_25_msmt17", help="boxmot reid weights name")
    p.add_argument("--tracker", default="deepocsort", help="tracker name (deepocsort/strongsort/...)")
    p.add_argument("--classes", default="0", help="class ids string for boxmot (e.g., '0')")
    p.add_argument("--device", default="0", help="device for boxmot (e.g., 0 for CUDA:0)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--vid_stride", type=int, default=1, help="frame stride for faster preview (1=all)")
    p.add_argument("--show", action="store_true", help="try to show window (requires GUI/X11)")
    p.add_argument("--save_lost", action="store_true", help="add --show-lost to save lost tracks")

    # Behavior
    p.add_argument("--skip_existing", action="store_true", help="skip if tracked output already exists")
    p.add_argument("--dry_run", action="store_true", help="print commands only")
    return p


def _run(cmd: list[str], logger: logging.Logger, dry_run: bool = False) -> None:
    logger.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def ffprobe_fps(video_path: Path) -> float:
    r = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        text=True,
    ).strip()
    if "/" in r:
        a, b = r.split("/")
        return float(a) / float(b)
    return float(r)


def load_events(args, logger: logging.Logger) -> list[EventItem]:
    labels_dir = Path(args.labels_dir)
    videos_dir = Path(args.videos_dir)

    label_paths = sorted(labels_dir.glob(args.label_glob))
    if args.video_id.strip():
        want = f"{args.video_id.strip()}.json"
        label_paths = [p for p in label_paths if p.name == want]

    items: list[EventItem] = []
    for lp in label_paths:
        try:
            j = json.loads(lp.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("skip label (read fail): %s (%s)", lp, e)
            continue

        file_name = j.get("metadata", {}).get("file_name", "")
        if not file_name:
            logger.warning("skip label (no metadata.file_name): %s", lp)
            continue

        video_path = videos_dir / file_name
        if not video_path.exists():
            logger.warning("skip label (missing video): %s -> %s", lp, video_path)
            continue

        video_id = Path(file_name).stem
        event_frames = j.get("annotations", {}).get("event_frame", [])
        if not event_frames:
            logger.info("skip (no event_frame): %s", lp.name)
            continue

        if args.event_index >= 0:
            if args.event_index >= len(event_frames):
                logger.warning("skip (event_index out of range): %s idx=%d", lp.name, args.event_index)
                continue
            event_frames = [event_frames[args.event_index]]

        for idx, (s, e) in enumerate(event_frames):
            items.append(
                EventItem(
                    video_id=video_id,
                    video_path=video_path,
                    label_path=lp,
                    event_idx=(args.event_index if args.event_index >= 0 else idx),
                    s_frame=int(s),
                    e_frame=int(e),
                )
            )
    return items


def make_clip(
    item: EventItem,
    fps: float,
    clip_sec: float,
    clip_mode: str,
    pre_sec: float,
    clip_dir: Path,
    crf: int,
    preset: str,
    do_copy: bool,
    do_reencode: bool,
    logger: logging.Logger,
    dry_run: bool,
) -> tuple[Path, float]:
    clip_dir.mkdir(parents=True, exist_ok=True)

    s, e = item.s_frame, item.e_frame
    if clip_mode == "center":
        mid = (s + e) / 2.0
        start_sec = max(0.0, mid / fps - (clip_sec / 2.0))
    else:  # start
        start_sec = max(0.0, (s / fps) - pre_sec)

    out = clip_dir / f"{item.video_id}_ev{item.event_idx:02d}_f{s}_{e}_{int(clip_sec)}s.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out, start_sec

    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-t", f"{clip_sec:.3f}", "-i", str(item.video_path), "-an"]
    if do_copy and not do_reencode:
        cmd += ["-c", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    cmd += [str(out)]

    _run(cmd, logger, dry_run=dry_run)
    return out, start_sec


def locate_boxmot_runs_dir() -> Optional[Path]:
    # BoxMOT logs showed: <venv>/lib/pythonX.Y/site-packages/runs/exp/...
    # Try to find that in common locations.
    # 1) Relative to current python executable
    py = Path(sys.executable).resolve()
    venv = py.parents[1]  # .../.venv/bin/python -> .../.venv
    cand1 = venv / "lib"
    if cand1.exists():
        for sp in cand1.glob("python*/site-packages"):
            r = sp / "runs" / "exp"
            if r.exists():
                return r
            # fallback: runs/*
            rr = sp / "runs"
            if rr.exists():
                return rr
    return None


def find_latest_tracked(runs_dir: Path, key: str) -> Optional[Path]:
    # Prefer exact stem match, else fallback to any tracked mp4 (latest)
    cands = sorted(runs_dir.rglob(f"*{key}*_tracked*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    cands = sorted(runs_dir.rglob("*_tracked*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def run_boxmot_on_clip(
    args,
    clip_path: Path,
    tracked_dir: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> Optional[Path]:
    tracked_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{clip_path.stem}_{args.tracker}_tracked.mp4"
    out_path = tracked_dir / out_name
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        logger.info("skip existing tracked: %s", out_path)
        return out_path

    cmd = [
        "boxmot",
        "track",
        args.detector,
        args.reid,
        args.tracker,
        "--source",
        str(clip_path),
        "--classes",
        str(args.classes),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--save",
    ]
    if args.save_lost:
        cmd.append("--show-lost")
    if args.show:
        cmd.append("--show")
    if int(args.vid_stride) > 1:
        cmd += ["--vid-stride", str(args.vid_stride)]

    _run(cmd, logger, dry_run=dry_run)

    # Collect produced tracked mp4 into our outputs dir
    runs_dir = locate_boxmot_runs_dir()
    if runs_dir is None or not runs_dir.exists():
        logger.warning("Cannot locate boxmot runs dir; tracked video may remain in site-packages runs.")
        return None

    produced = find_latest_tracked(runs_dir, clip_path.stem)
    if produced is None or not produced.exists():
        logger.warning("Tracked mp4 not found under %s", runs_dir)
        return None

    if dry_run:
        logger.info("Would copy: %s -> %s", produced, out_path)
        return out_path

    # copy (keep original)
    shutil.copy2(produced, out_path)
    logger.info("tracked saved: %s", out_path)
    return out_path


def append_manifest(fp, rec: dict[str, Any]) -> None:
    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fp.flush()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # default behavior: reencode unless --copy is explicitly requested
    if not args.reencode and not args.copy:
        args.reencode = True

    run_paths = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    logger.info("python=%s platform=%s", platform.python_version(), platform.platform())
    logger.info("run_out_dir=%s", run_paths.out_dir)

    clip_dir = Path(args.clip_dir) if args.clip_dir.strip() else (run_paths.out_dir / "clips")
    tracked_dir = run_paths.out_dir / "tracked"
    manifest_path = run_paths.out_dir / "manifest.jsonl"

    items = load_events(args, logger)
    if not items:
        logger.error("No events found. Check labels_dir/label_glob/video_id.")
        return 2

    runs_dir = locate_boxmot_runs_dir()
    logger.info("boxmot runs dir guess: %s", runs_dir)

    with manifest_path.open("w", encoding="utf-8") as mf:
        for item in items:
            t0 = time.time()
            rec: dict[str, Any] = {
                "video_id": item.video_id,
                "label_path": str(item.label_path),
                "video_path": str(item.video_path),
                "event_idx": item.event_idx,
                "s_frame": item.s_frame,
                "e_frame": item.e_frame,
                "clip_mode": args.clip_mode,
                "clip_sec": args.clip_sec,
                "status": "ok",
                "error": None,
            }
            try:
                fps = ffprobe_fps(item.video_path)
                rec["fps"] = fps

                clip_path, start_sec = make_clip(
                    item=item,
                    fps=fps,
                    clip_sec=float(args.clip_sec),
                    clip_mode=str(args.clip_mode),
                    pre_sec=float(args.pre_sec),
                    clip_dir=clip_dir,
                    crf=int(args.crf),
                    preset=str(args.preset),
                    do_copy=bool(args.copy),
                    do_reencode=bool(args.reencode),
                    logger=logger,
                    dry_run=bool(args.dry_run),
                )
                rec["clip_path"] = str(clip_path)
                rec["clip_start_sec"] = start_sec

                tracked = run_boxmot_on_clip(args, clip_path, tracked_dir, logger, dry_run=bool(args.dry_run))
                rec["tracked_path"] = str(tracked) if tracked else None
            except Exception as e:
                rec["status"] = "fail"
                rec["error"] = repr(e)
                logger.exception("fail: %s ev=%d", item.video_id, item.event_idx)
            finally:
                rec["elapsed_sec"] = round(time.time() - t0, 3)
                append_manifest(mf, rec)

    logger.info("DONE. clips=%s tracked=%s manifest=%s", clip_dir, tracked_dir, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
