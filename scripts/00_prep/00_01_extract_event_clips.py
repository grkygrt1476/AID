#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aidlib import run_utils

STAGE = "00_prep"


@dataclass
class VideoMeta:
    fps: float
    duration_sec: Optional[float]
    frame_count: Optional[int]
    source: str


@dataclass
class ManifestRow:
    video_id: str
    event_idx: int
    src_video: str
    label_json: str
    ev_start_frame: Optional[int]
    ev_end_frame: Optional[int]
    fps: Optional[float]
    clip_start_sec: Optional[float]
    clip_dur_sec: Optional[float]
    clip_path: str
    status: str
    message: str


def build_parser() -> Any:
    p = run_utils.common_argparser()
    p.set_defaults(out_root="outputs", log_root="outputs/logs", out_base="extract_event_clips")

    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Process all label JSON files in --labels_dir")
    sel.add_argument("--choice", nargs="+", default=None, help="Specific video IDs or mp4 names")

    p.add_argument("--labels_dir", default="data/videos/labels")
    p.add_argument("--videos_dir", default="data/videos")
    p.add_argument("--out_dir", default="data/clips")
    p.add_argument("--dur", type=float, default=50.0)
    p.add_argument("--mode", choices=["mid", "start"], default="mid")
    p.add_argument("--pre_sec", type=float, default=25.0)
    p.add_argument("--post_sec", type=float, default=25.0)
    p.add_argument("--reencode", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--crf", type=int, default=23)
    return p


def _norm_video_id(token: str) -> str:
    raw = str(token).strip()
    if not raw:
        return ""
    if raw.lower().endswith(".mp4"):
        return Path(raw).stem
    return Path(raw).stem if "." in Path(raw).name else raw


def _collect_ids(args, labels_dir: Path) -> list[str]:
    if args.all:
        return sorted([p.stem for p in labels_dir.glob("*.json") if p.is_file()])
    out: list[str] = []
    for t in args.choice or []:
        vid = _norm_video_id(t)
        if vid and vid not in out:
            out.append(vid)
    return out


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_event_frames(label_obj: dict[str, Any]) -> list[tuple[int, int]]:
    ann = label_obj.get("annotations", {})
    raw = ann.get("event_frame", None) if isinstance(ann, dict) else None
    if raw is None:
        raw = label_obj.get("event_frame", None)

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


def _resolve_video_path(videos_dir: Path, video_id: str) -> Optional[Path]:
    primary = videos_dir / f"{video_id}.mp4"
    if primary.exists():
        return primary
    fallback = videos_dir / "validation" / f"{video_id}.mp4"
    if fallback.exists():
        return fallback
    return None


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode), str(proc.stdout), str(proc.stderr)


def _parse_ratio(text: str) -> Optional[float]:
    t = str(text).strip()
    if not t or t in {"0", "0/0", "N/A"}:
        return None
    if "/" in t:
        a, b = t.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den == 0:
                return None
            v = num / den
            return v if v > 0 else None
        except Exception:
            return None
    try:
        v2 = float(t)
        return v2 if v2 > 0 else None
    except Exception:
        return None


def _probe_meta_ffprobe(video_path: Path) -> tuple[Optional[VideoMeta], str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(video_path),
    ]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        return None, (err.strip() or out.strip() or f"ffprobe failed rc={rc}")

    try:
        obj = json.loads(out)
    except Exception as exc:
        return None, f"ffprobe json parse failed: {exc}"

    streams = obj.get("streams", []) if isinstance(obj, dict) else []
    fmt = obj.get("format", {}) if isinstance(obj, dict) else {}
    fmt = fmt if isinstance(fmt, dict) else {}

    fps: Optional[float] = None
    frame_count: Optional[int] = None
    for st in streams:
        if not isinstance(st, dict):
            continue
        if st.get("codec_type") != "video":
            continue
        fps = _parse_ratio(str(st.get("avg_frame_rate", ""))) or _parse_ratio(str(st.get("r_frame_rate", "")))
        try:
            nb = st.get("nb_frames", None)
            if nb not in (None, "", "N/A"):
                frame_count = int(nb)
        except Exception:
            frame_count = None
        break

    duration: Optional[float] = None
    try:
        d = fmt.get("duration", None)
        if d not in (None, "", "N/A"):
            dv = float(d)
            duration = dv if dv > 0 else None
    except Exception:
        duration = None

    if fps is None or fps <= 0:
        return None, "ffprobe did not provide valid fps"

    return VideoMeta(fps=float(fps), duration_sec=duration, frame_count=frame_count, source="ffprobe"), "ok"


def _probe_meta_cv2(video_path: Path) -> tuple[Optional[VideoMeta], str]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None, "cv2 unavailable"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "cv2 failed to open video"
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if fps <= 0:
        return None, "cv2 returned invalid fps"
    duration = None
    if n > 0:
        duration = float(n) / float(fps)
    return VideoMeta(fps=float(fps), duration_sec=duration, frame_count=(n if n > 0 else None), source="cv2"), "ok"


def _probe_video_meta(video_path: Path, logger: logging.Logger) -> Optional[VideoMeta]:
    meta, msg = _probe_meta_ffprobe(video_path)
    if meta is not None:
        return meta
    logger.warning("ffprobe metadata failed for %s: %s", video_path, msg)

    meta2, msg2 = _probe_meta_cv2(video_path)
    if meta2 is not None:
        logger.info("Using cv2 fallback metadata for %s", video_path)
        return meta2
    logger.warning("cv2 metadata fallback failed for %s: %s", video_path, msg2)
    return None


def _compute_clip_window(
    *,
    s: int,
    e: int,
    fps: float,
    dur: float,
    mode: str,
    pre_sec: float,
    post_sec: float,
    video_dur: Optional[float],
) -> tuple[float, float]:
    ev_len_sec = max(0.0, float(e - s + 1)) / float(fps)

    if mode == "mid":
        if ev_len_sec <= float(dur):
            mid_frame = (float(s) + float(e)) / 2.0
            start = (mid_frame / float(fps)) - (float(dur) / 2.0)
        else:
            start = float(s) / float(fps)
        req_dur = float(dur)
    else:
        start = (float(s) / float(fps)) - float(pre_sec)
        req_dur = float(pre_sec) + float(post_sec)

    start = max(0.0, float(start))
    clip_dur = float(req_dur)

    if video_dur is not None and video_dur > 0:
        max_start = max(0.0, float(video_dur) - float(req_dur))
        start = min(start, max_start)
        remain = max(0.0, float(video_dur) - start)
        clip_dur = min(float(req_dur), remain)

    return float(start), float(clip_dur)


def _dur_tag(dur: float) -> str:
    if float(dur).is_integer():
        return f"{int(dur)}s"
    return f"{dur:g}s"


def _run_ffmpeg_extract(
    *,
    src: Path,
    out_path: Path,
    start_sec: float,
    dur_sec: float,
    reencode: bool,
    preset: str,
    crf: int,
) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_sec:.3f}", "-i", str(src), "-t", f"{dur_sec:.3f}"]
    if reencode:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    else:
        cmd += ["-c", "copy", "-movflags", "+faststart", str(out_path)]

    rc, _out, err = _run_cmd(cmd)
    if rc != 0:
        return False, (err.strip() or f"ffmpeg failed rc={rc}")
    if not out_path.exists():
        return False, "ffmpeg returned success but output missing"
    return True, "ok"


def _ensure_ffmpeg_available() -> Optional[str]:
    rc, out, err = _run_cmd(["ffmpeg", "-version"])
    if rc != 0:
        return err.strip() or out.strip() or "ffmpeg not available"
    return None


def _row_to_csv(row: ManifestRow) -> dict[str, str]:
    def _fmt_float(v: Optional[float]) -> str:
        if v is None:
            return ""
        return f"{float(v):.6f}"

    return {
        "video_id": row.video_id,
        "event_idx": str(row.event_idx),
        "src_video": row.src_video,
        "label_json": row.label_json,
        "ev_start_frame": "" if row.ev_start_frame is None else str(int(row.ev_start_frame)),
        "ev_end_frame": "" if row.ev_end_frame is None else str(int(row.ev_end_frame)),
        "fps": _fmt_float(row.fps),
        "clip_start_sec": _fmt_float(row.clip_start_sec),
        "clip_dur_sec": _fmt_float(row.clip_dur_sec),
        "clip_path": row.clip_path,
        "status": row.status,
        "message": row.message,
    }


def _write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "event_idx",
        "src_video",
        "label_json",
        "ev_start_frame",
        "ev_end_frame",
        "fps",
        "clip_start_sec",
        "clip_dur_sec",
        "clip_path",
        "status",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(_row_to_csv(row))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if float(args.dur) <= 0:
        print("[ERR] --dur must be > 0", file=sys.stderr)
        return 2
    if float(args.pre_sec) < 0 or float(args.post_sec) < 0:
        print("[ERR] --pre_sec/--post_sec must be >= 0", file=sys.stderr)
        return 2

    ffmpeg_err = _ensure_ffmpeg_available()
    if ffmpeg_err is not None:
        print(f"[ERR] {ffmpeg_err}", file=sys.stderr)
        return 2

    run = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    labels_dir = Path(args.labels_dir)
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _collect_ids(args, labels_dir=labels_dir)
    rows: list[ManifestRow] = []

    videos_seen = 0
    total_events = 0
    created = 0
    skipped = 0
    errors = 0

    dur_tag = _dur_tag(float(args.dur))

    for video_id in ids:
        videos_seen += 1
        label_path = labels_dir / f"{video_id}.json"
        if not label_path.exists():
            rows.append(
                ManifestRow(
                    video_id=video_id,
                    event_idx=-1,
                    src_video="",
                    label_json=str(label_path),
                    ev_start_frame=None,
                    ev_end_frame=None,
                    fps=None,
                    clip_start_sec=None,
                    clip_dur_sec=None,
                    clip_path="",
                    status="error",
                    message=f"label not found: {label_path}",
                )
            )
            errors += 1
            logger.error("label not found: %s", label_path)
            continue

        label_obj = _load_json(label_path)
        if label_obj is None:
            rows.append(
                ManifestRow(
                    video_id=video_id,
                    event_idx=-1,
                    src_video="",
                    label_json=str(label_path),
                    ev_start_frame=None,
                    ev_end_frame=None,
                    fps=None,
                    clip_start_sec=None,
                    clip_dur_sec=None,
                    clip_path="",
                    status="error",
                    message=f"label parse failed: {label_path}",
                )
            )
            errors += 1
            logger.error("label parse failed: %s", label_path)
            continue

        events = _extract_event_frames(label_obj)
        if not events:
            rows.append(
                ManifestRow(
                    video_id=video_id,
                    event_idx=-1,
                    src_video="",
                    label_json=str(label_path),
                    ev_start_frame=None,
                    ev_end_frame=None,
                    fps=None,
                    clip_start_sec=None,
                    clip_dur_sec=None,
                    clip_path="",
                    status="error",
                    message="no annotations.event_frame",
                )
            )
            errors += 1
            logger.error("no annotations.event_frame: %s", label_path)
            continue

        video_path = _resolve_video_path(videos_dir=videos_dir, video_id=video_id)
        if video_path is None:
            for i, (s, e) in enumerate(events):
                rows.append(
                    ManifestRow(
                        video_id=video_id,
                        event_idx=i,
                        src_video="",
                        label_json=str(label_path),
                        ev_start_frame=int(s),
                        ev_end_frame=int(e),
                        fps=None,
                        clip_start_sec=None,
                        clip_dur_sec=None,
                        clip_path="",
                        status="error",
                        message="video not found (primary and validation)",
                    )
                )
                total_events += 1
                errors += 1
            logger.error("video not found: %s", video_id)
            continue

        meta = _probe_video_meta(video_path, logger=logger)
        if meta is None:
            for i, (s, e) in enumerate(events):
                rows.append(
                    ManifestRow(
                        video_id=video_id,
                        event_idx=i,
                        src_video=str(video_path),
                        label_json=str(label_path),
                        ev_start_frame=int(s),
                        ev_end_frame=int(e),
                        fps=None,
                        clip_start_sec=None,
                        clip_dur_sec=None,
                        clip_path="",
                        status="error",
                        message="failed to probe fps/duration",
                    )
                )
                total_events += 1
                errors += 1
            logger.error("metadata probe failed: %s", video_path)
            continue

        for i, (s, e) in enumerate(events):
            total_events += 1
            clip_dir = out_dir / video_id
            clip_name = f"ev{i:02d}_f{int(s)}-{int(e)}_{dur_tag}.mp4"
            clip_path = clip_dir / clip_name

            start_sec, clip_dur_sec = _compute_clip_window(
                s=int(s),
                e=int(e),
                fps=float(meta.fps),
                dur=float(args.dur),
                mode=str(args.mode),
                pre_sec=float(args.pre_sec),
                post_sec=float(args.post_sec),
                video_dur=meta.duration_sec,
            )

            if clip_dur_sec <= 0.01:
                rows.append(
                    ManifestRow(
                        video_id=video_id,
                        event_idx=i,
                        src_video=str(video_path),
                        label_json=str(label_path),
                        ev_start_frame=int(s),
                        ev_end_frame=int(e),
                        fps=float(meta.fps),
                        clip_start_sec=float(start_sec),
                        clip_dur_sec=float(clip_dur_sec),
                        clip_path=str(clip_path),
                        status="error",
                        message="computed clip duration <= 0",
                    )
                )
                errors += 1
                continue

            if clip_path.exists() and not bool(args.overwrite):
                rows.append(
                    ManifestRow(
                        video_id=video_id,
                        event_idx=i,
                        src_video=str(video_path),
                        label_json=str(label_path),
                        ev_start_frame=int(s),
                        ev_end_frame=int(e),
                        fps=float(meta.fps),
                        clip_start_sec=float(start_sec),
                        clip_dur_sec=float(clip_dur_sec),
                        clip_path=str(clip_path),
                        status="skipped",
                        message="exists (use --overwrite)",
                    )
                )
                skipped += 1
                continue

            ok, msg = _run_ffmpeg_extract(
                src=video_path,
                out_path=clip_path,
                start_sec=float(start_sec),
                dur_sec=float(clip_dur_sec),
                reencode=bool(args.reencode),
                preset=str(args.preset),
                crf=int(args.crf),
            )
            if not ok:
                rows.append(
                    ManifestRow(
                        video_id=video_id,
                        event_idx=i,
                        src_video=str(video_path),
                        label_json=str(label_path),
                        ev_start_frame=int(s),
                        ev_end_frame=int(e),
                        fps=float(meta.fps),
                        clip_start_sec=float(start_sec),
                        clip_dur_sec=float(clip_dur_sec),
                        clip_path=str(clip_path),
                        status="error",
                        message=msg,
                    )
                )
                errors += 1
                logger.error("ffmpeg failed | video=%s ev=%d msg=%s", video_id, i, msg)
                continue

            rows.append(
                ManifestRow(
                    video_id=video_id,
                    event_idx=i,
                    src_video=str(video_path),
                    label_json=str(label_path),
                    ev_start_frame=int(s),
                    ev_end_frame=int(e),
                    fps=float(meta.fps),
                    clip_start_sec=float(start_sec),
                    clip_dur_sec=float(clip_dur_sec),
                    clip_path=str(clip_path),
                    status="created",
                    message=(
                        "reencode" if bool(args.reencode) else "copy mode (keyframe-aligned start may shift slightly)"
                    ),
                )
            )
            created += 1

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(manifest_path, rows)

    logger.info("manifest written: %s", manifest_path)
    logger.info(
        "summary | videos=%d events=%d created=%d skipped=%d error=%d",
        videos_seen,
        total_events,
        created,
        skipped,
        errors,
    )

    print(
        f"videos={videos_seen} events={total_events} created={created} skipped={skipped} error={errors} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
