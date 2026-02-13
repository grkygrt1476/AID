#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aidlib import run_utils

STAGE = "00_prep"


@dataclass
class ClipJob:
    video_id: str
    video_path: Path
    label_path: Path


@dataclass
class ClipResult:
    status: str  # ok | skip | err
    video_id: str
    video_path: str
    label_path: str
    event_frame: Optional[List[int]]
    fps: Optional[float]
    total_frames: Optional[int]
    clip_start_frame: Optional[int]
    clip_end_frame: Optional[int]
    clip_start_sec: Optional[float]
    clip_end_sec: Optional[float]
    output_clip_path: Optional[str]
    duration_sec: float
    message: str


def build_parser():
    p = run_utils.common_argparser()

    p.add_argument("--video_id", default="", help="single mode: e.g., E01_001")
    p.add_argument("--video_path", default="", help="single mode: explicit video path")
    p.add_argument("--video_ids", nargs="*", default=None, help="batch mode: e.g., E01_001 E01_004 E01_007")

    p.add_argument("--label_path", default="", help="single mode: explicit label json path")
    p.add_argument("--duration_sec", type=float, default=50.0, help="target clip duration in seconds")
    p.add_argument("--out_dir", default="data/clips", help="dataset clip output directory")
    p.add_argument("--suffix", default="", help="optional suffix appended before .mp4")
    p.add_argument("--event_index", type=int, default=0, help="which event_frame pair to use")
    p.add_argument("--dry_run", action="store_true", help="print computed range only; do not write clip")
    return p


def _extract_video_id_from_stem(stem: str) -> str:
    m = re.search(r"(E\d{2}_\d{3})", stem)
    return m.group(1) if m else stem


def _duration_tag(duration_sec: float) -> str:
    if float(duration_sec).is_integer():
        return f"{int(duration_sec)}"
    # Keep filename stable if non-integer duration is used.
    txt = f"{duration_sec:.3f}".rstrip("0").rstrip(".")
    return txt.replace(".", "p")


def _normalize_suffix(s: str) -> str:
    x = (s or "").strip()
    if not x:
        return ""
    return x if x.startswith("_") else f"_{x}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_event_frames(obj: Dict[str, Any]) -> List[List[Any]]:
    ann = obj.get("annotations", {}) or {}
    event_frames = ann.get("event_frame", None)
    if not event_frames:
        event_frames = obj.get("event_frame", None)
    if not event_frames:
        return []
    if isinstance(event_frames, list):
        return event_frames
    return []


def _pick_event_pair(label_obj: Dict[str, Any], event_index: int) -> Optional[Tuple[int, int]]:
    event_frames = _extract_event_frames(label_obj)
    if not event_frames:
        return None
    if event_index < 0 or event_index >= len(event_frames):
        return None
    pair = event_frames[event_index]
    if not isinstance(pair, list) or len(pair) < 2:
        return None
    try:
        return int(pair[0]), int(pair[1])
    except Exception:
        return None


def _compute_clip_range(
    event_start: int,
    event_end: int,
    fps: float,
    total_frames: int,
    duration_sec: float,
) -> Tuple[int, int, int]:
    dur_frames = max(1, int(round(float(duration_sec) * float(fps))))
    mid = (float(event_start) + float(event_end)) / 2.0
    half = dur_frames // 2
    raw_start = int(round(mid - half))

    # If video length is known, clamp robustly and handle overlong duration.
    if total_frames > 0:
        if dur_frames >= total_frames:
            return 0, total_frames - 1, total_frames
        max_start = max(0, total_frames - dur_frames)
        clip_start = max(0, min(raw_start, max_start))
        clip_end = clip_start + dur_frames - 1
        return clip_start, clip_end, dur_frames

    # Unknown total frame count: use best-effort target window.
    clip_start = max(0, raw_start)
    clip_end = clip_start + dur_frames - 1
    return clip_start, clip_end, dur_frames


def _validate_output_clip(cv2_module, out_path: Path, expect_w: int, expect_h: int) -> Tuple[bool, str]:
    cap = cv2_module.VideoCapture(str(out_path))
    if not cap.isOpened():
        return False, "failed to re-open output clip"
    try:
        w = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if w <= 0 or h <= 0 or n <= 0:
        return False, f"invalid output properties (w={w}, h={h}, frames={n})"
    if w != int(expect_w) or h != int(expect_h):
        return False, f"dimension mismatch (expected {expect_w}x{expect_h}, got {w}x{h})"
    return True, "ok"


def _resolve_single_job(args) -> Tuple[Optional[ClipJob], Optional[str]]:
    video_id = args.video_id.strip()
    video_path = args.video_path.strip()

    if bool(video_id) == bool(video_path):
        return None, "single mode requires exactly one of --video_id or --video_path"

    if video_path:
        vp = Path(video_path)
        vid = video_id or _extract_video_id_from_stem(vp.stem)
    else:
        vid = video_id
        vp = Path("data/videos") / f"{vid}.mp4"

    lp = Path(args.label_path.strip()) if args.label_path.strip() else Path("data/labels") / f"{vid}.json"
    return ClipJob(video_id=vid, video_path=vp, label_path=lp), None


def _resolve_batch_jobs(args) -> Tuple[List[ClipJob], Optional[str]]:
    raw_ids = [x.strip() for x in (args.video_ids or []) if str(x).strip()]
    if not raw_ids:
        return [], "batch mode requires at least one id in --video_ids"
    if args.video_id.strip() or args.video_path.strip():
        return [], "do not mix --video_ids with --video_id/--video_path"
    if args.label_path.strip() and len(raw_ids) > 1:
        return [], "--label_path with batch is only allowed when one --video_ids item is provided"

    jobs: List[ClipJob] = []
    for vid in raw_ids:
        vp = Path("data/videos") / f"{vid}.mp4"
        if args.label_path.strip():
            lp = Path(args.label_path.strip())
        else:
            lp = Path("data/labels") / f"{vid}.json"
        jobs.append(ClipJob(video_id=vid, video_path=vp, label_path=lp))
    return jobs, None


def process_one_job(args, cv2_module, logger: logging.Logger, job: ClipJob, out_dir: Path) -> ClipResult:
    duration_sec = float(args.duration_sec)
    suffix = _normalize_suffix(args.suffix)

    base = {
        "status": "err",
        "video_id": job.video_id,
        "video_path": str(job.video_path),
        "label_path": str(job.label_path),
        "event_frame": None,
        "fps": None,
        "total_frames": None,
        "clip_start_frame": None,
        "clip_end_frame": None,
        "clip_start_sec": None,
        "clip_end_sec": None,
        "output_clip_path": None,
        "duration_sec": duration_sec,
        "message": "",
    }

    if not job.video_path.exists():
        msg = f"[ERR] video not found: {job.video_path}"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)
    if not job.label_path.exists():
        msg = f"[ERR] label not found: {job.label_path}"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)

    try:
        label_obj = _load_json(job.label_path)
    except Exception as e:
        msg = f"[ERR] failed to parse label json: {job.label_path} ({e})"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)

    pair = _pick_event_pair(label_obj, int(args.event_index))
    if pair is None:
        msg = f"[SKIP] no event_frame in {job.label_path}"
        print(msg)
        logger.info(msg)
        base["status"] = "skip"
        base["message"] = msg
        return ClipResult(**base)

    event_start, event_end = pair
    base["event_frame"] = [event_start, event_end]

    cap = cv2_module.VideoCapture(str(job.video_path))
    if not cap.isOpened():
        msg = f"[ERR] failed to open video: {job.video_path}"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)

    fps = float(cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
        logger.warning("fps unavailable -> fallback to 30.0 for %s", job.video_path)

    total_frames = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        ok, fr = cap.read()
        if not ok or fr is None:
            cap.release()
            msg = f"[ERR] failed to read first frame for dimensions: {job.video_path}"
            logger.error(msg)
            base["message"] = msg
            return ClipResult(**base)
        h, w = fr.shape[:2]
        cap.set(cv2_module.CAP_PROP_POS_FRAMES, 0)

    clip_start, clip_end, target_frames = _compute_clip_range(
        event_start=event_start,
        event_end=event_end,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
    )

    dur_tag = _duration_tag(duration_sec)
    out_name = f"{job.video_id}_f{event_start}_{event_end}_{dur_tag}s{suffix}.mp4"
    out_path = out_dir / out_name

    base["fps"] = fps
    base["total_frames"] = total_frames if total_frames > 0 else None
    base["clip_start_frame"] = int(clip_start)
    base["clip_end_frame"] = int(clip_end)
    base["clip_start_sec"] = float(clip_start / fps)
    base["clip_end_sec"] = float(clip_end / fps)
    base["output_clip_path"] = str(out_path)

    logger.info(
        "clip plan | video_id=%s event=[%d,%d] fps=%.3f total=%s start=%d end=%d out=%s",
        job.video_id,
        event_start,
        event_end,
        fps,
        str(total_frames if total_frames > 0 else "unknown"),
        clip_start,
        clip_end,
        out_path,
    )

    if args.dry_run:
        base["status"] = "ok"
        base["message"] = "[DRY_RUN] computed only"
        return ClipResult(**base)

    cap.set(cv2_module.CAP_PROP_POS_FRAMES, int(clip_start))
    writer = cv2_module.VideoWriter(
        str(out_path),
        cv2_module.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        cap.release()
        msg = f"[ERR] failed to open VideoWriter: {out_path}"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)

    written = 0
    try:
        for _ in range(int(target_frames)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    if written <= 0:
        msg = f"[ERR] no frames written: {out_path}"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)

    ok_play, msg_play = _validate_output_clip(cv2_module, out_path, expect_w=w, expect_h=h)
    if not ok_play:
        msg = f"[ERR] output validation failed: {out_path} ({msg_play})"
        logger.error(msg)
        base["message"] = msg
        return ClipResult(**base)

    # Actual end frame may be shorter if video ended early.
    actual_end = int(clip_start + written - 1)
    base["clip_end_frame"] = actual_end
    base["clip_end_sec"] = float(actual_end / fps)
    base["status"] = "ok"
    base["message"] = f"written={written}"
    return ClipResult(**base)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.duration_sec <= 0:
        print("[ERR] --duration_sec must be > 0")
        return 2
    if args.event_index < 0:
        print("[ERR] --event_index must be >= 0")
        return 2

    batch_ids = [x for x in (args.video_ids or []) if str(x).strip()]
    is_batch = len(batch_ids) > 0

    if not args.out_base:
        if is_batch:
            args.out_base = f"batch_clip{_duration_tag(float(args.duration_sec))}s"
        else:
            vid_hint = args.video_id.strip()
            if not vid_hint and args.video_path.strip():
                vid_hint = _extract_video_id_from_stem(Path(args.video_path.strip()).stem)
            args.out_base = f"{vid_hint or 'single'}_clip{_duration_tag(float(args.duration_sec))}s"

    run = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    try:
        import cv2  # type: ignore
    except Exception as e:
        logger.exception("Failed to import cv2: %s", e)
        return 2

    out_dir = Path(args.out_dir)
    run_utils.safe_mkdir(out_dir)

    params = vars(args).copy()
    params.update({"is_batch": is_batch, "resolved_out_dir": str(out_dir)})
    (run.out_dir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if is_batch:
        jobs, err = _resolve_batch_jobs(args)
    else:
        one, err = _resolve_single_job(args)
        jobs = [one] if one is not None else []
    if err:
        logger.error(err)
        logger.error("Usage: %s", parser.format_usage().strip())
        return 2

    results: List[ClipResult] = []
    for job in jobs:
        try:
            res = process_one_job(args, cv2, logger, job, out_dir=out_dir)
        except Exception as e:
            msg = f"[ERR] unexpected failure for {job.video_id}: {e}"
            logger.exception(msg)
            res = ClipResult(
                status="err",
                video_id=job.video_id,
                video_path=str(job.video_path),
                label_path=str(job.label_path),
                event_frame=None,
                fps=None,
                total_frames=None,
                clip_start_frame=None,
                clip_end_frame=None,
                clip_start_sec=None,
                clip_end_sec=None,
                output_clip_path=None,
                duration_sec=float(args.duration_sec),
                message=msg,
            )
        results.append(res)

    summary = {
        "stage": STAGE,
        "run_ts": run.run_ts,
        "out_base": args.out_base,
        "out_dir": str(run.out_dir),
        "clips_out_dir": str(out_dir),
        "results": [asdict(r) for r in results],
        "stats": {
            "total": len(results),
            "ok": sum(1 for r in results if r.status == "ok"),
            "skip": sum(1 for r in results if r.status == "skip"),
            "err": sum(1 for r in results if r.status == "err"),
        },
    }
    (run.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "done | total=%d ok=%d skip=%d err=%d summary=%s",
        summary["stats"]["total"],
        summary["stats"]["ok"],
        summary["stats"]["skip"],
        summary["stats"]["err"],
        run.out_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
