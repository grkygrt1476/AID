#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess collector for Stage 03 intrusion FSM outputs.

Scans existing run folders under outputs/03_deepstream/ and produces:
  1. A per-clip CSV metrics table   -> outputs/eval/stage03_metrics_<TS>.csv
  2. A Markdown manifest of runs    -> outputs/eval/stage03_manifest_<TS>.md

This script is read-only with respect to Stage 03 outputs.  It never modifies,
overwrites, or deletes any existing file.  It only writes to --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pattern: outputs/03_deepstream/<run_ts>/<out_base>/
# run_ts is YYYYMMDD_HHMMSS
RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")

# Log line timestamp format: "2026-03-16 19:54:50 | INFO | ..."
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|")
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"

# CSV columns -- always emitted (empty string when unavailable)
REQUIRED_COLUMNS = [
    "run_ts",
    "scene_id",
    "clip_id",
    "run_dir",
    "duration_sec",
    "wall_clock_runtime_sec",
    "confirmed_events",
    "pose_probe_status_runtime",
    "pose_probe_status_preflight",
    "frame_count",
    "fps",
    "tracking_mode",
    "overlay_video_path",
    "tracking_video_path",
    "summary_json_path",
    "events_jsonl_path",
    "sidecar_csv_path",
    "ds_app_runtime_path",
    "notes",
]

# Optional columns -- only populated when the source data contains them
OPTIONAL_COLUMNS = [
    "tracks_seen",
    "sidecar_row_count",
    "sidecar_modes",
    "records_emitted",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON dict or None on any read/parse failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def extract_scene_clip(video_path_str: str) -> tuple[str, str]:
    """Derive scene_id and clip_id from an input_video path.

    Expected pattern: .../data/clips/<scene_id>/<clip_filename>.mp4
    Falls back to empty strings when the path doesn't match.
    """
    p = Path(video_path_str)
    clip_id = p.stem  # e.g. ev00_f1826-2854_50s
    parent = p.parent.name  # e.g. E01_001
    if re.match(r"^E\d+_\d+$", parent):
        return parent, clip_id
    return "", clip_id


def find_artifact(run_dir: Path, glob_pattern: str) -> str:
    """Return the first match or empty string."""
    matches = sorted(run_dir.glob(glob_pattern))
    return str(matches[0]) if matches else ""


def parse_log_wall_clock(log_path: Path) -> float | None:
    """Extract wall-clock seconds from first and last timestamp in a log file.

    Log format: ``2026-03-16 19:54:50 | INFO | __main__ | ...``
    Returns elapsed seconds or None if timestamps cannot be parsed.
    """
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LOG_TS_RE.match(line)
                if m:
                    try:
                        ts = datetime.strptime(m.group(1), LOG_TS_FMT)
                    except ValueError:
                        continue
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
    except OSError:
        return None

    if first_ts is not None and last_ts is not None and last_ts >= first_ts:
        return (last_ts - first_ts).total_seconds()
    return None


# ---------------------------------------------------------------------------
# Per-run collection
# ---------------------------------------------------------------------------

def collect_run(run_dir: Path, run_ts: str, log_root: Path | None) -> dict[str, str]:
    """Collect metrics from a single Stage 03 run folder.

    Returns a dict keyed by column name; every column is always present.
    """
    notes: list[str] = []

    row: dict[str, str] = {c: "" for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
    row["run_ts"] = run_ts
    row["run_dir"] = str(run_dir)

    # --- intrusion_summary.json (primary source) -------------------------
    summary_path = run_dir / "intrusion_summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        row["summary_json_path"] = str(summary_path)
        summary = load_json(summary_path)
        if summary is None:
            notes.append("intrusion_summary.json: parse error")
    else:
        notes.append("intrusion_summary.json: missing")

    # --- run_meta.json (secondary / supplementary) -----------------------
    meta_path = run_dir / "run_meta.json"
    meta: dict[str, Any] | None = None
    if meta_path.is_file():
        meta = load_json(meta_path)
        if meta is None:
            notes.append("run_meta.json: parse error")
    else:
        notes.append("run_meta.json: missing")

    # --- Extract fields from summary first, then meta fallback -----------
    # input_video -> scene_id / clip_id
    video_path_str = ""
    if summary:
        video_path_str = str(summary.get("video_path", ""))
    if not video_path_str and meta:
        video_path_str = str(meta.get("input_video", ""))
    if video_path_str:
        scene_id, clip_id = extract_scene_clip(video_path_str)
        row["scene_id"] = scene_id
        row["clip_id"] = clip_id

    # confirmed_events
    if summary and "confirmed_events" in summary:
        row["confirmed_events"] = str(summary["confirmed_events"])
    elif meta:
        ds = meta.get("decision_summary", {})
        if isinstance(ds, dict) and "confirmed_events" in ds:
            row["confirmed_events"] = str(ds["confirmed_events"])

    # pose_probe_status -- separated into runtime truth vs preflight/config
    # Runtime: only from intrusion_summary.json (the actual decision pass result)
    if summary and "pose_probe_status" in summary:
        row["pose_probe_status_runtime"] = str(summary["pose_probe_status"])
    else:
        notes.append("pose runtime status: unavailable (summary missing)")

    # Preflight: from run_meta.json top-level (config-time status before execution)
    if meta and "pose_probe_status" in meta:
        row["pose_probe_status_preflight"] = str(meta["pose_probe_status"])

    # frame_count, fps, duration_sec
    fc: int | None = None
    fps_val: float | None = None

    if summary:
        fc = summary.get("frame_count")
        fps_val = summary.get("fps")
    if fc is None and meta:
        ds = meta.get("decision_summary", {})
        if isinstance(ds, dict):
            fc = ds.get("frame_count")
            fps_val = fps_val or ds.get("fps")

    if fc is not None:
        row["frame_count"] = str(fc)
    if fps_val is not None:
        row["fps"] = str(fps_val)
    if fc and fps_val:
        row["duration_sec"] = f"{fc / fps_val:.2f}"

    # tracking_mode
    if meta and "tracking_mode" in meta:
        row["tracking_mode"] = str(meta["tracking_mode"])

    # tracks_seen (optional)
    tracks_seen = None
    if summary and "tracks_seen" in summary:
        tracks_seen = summary["tracks_seen"]
    elif meta:
        ds = meta.get("decision_summary", {})
        if isinstance(ds, dict):
            tracks_seen = ds.get("tracks_seen")
    if isinstance(tracks_seen, list):
        row["tracks_seen"] = str(len(tracks_seen))

    # sidecar stats (optional)
    sidecar_info: dict[str, Any] | None = None
    if summary and isinstance(summary.get("sidecar"), dict):
        sidecar_info = summary["sidecar"]
    elif meta:
        ds = meta.get("decision_summary", {})
        if isinstance(ds, dict) and isinstance(ds.get("sidecar"), dict):
            sidecar_info = ds["sidecar"]
    if sidecar_info:
        if "row_count" in sidecar_info:
            row["sidecar_row_count"] = str(sidecar_info["row_count"])
        if "modes" in sidecar_info and isinstance(sidecar_info["modes"], list):
            row["sidecar_modes"] = ",".join(str(m) for m in sidecar_info["modes"])

    # records_emitted (optional)
    if summary and "records_emitted" in summary:
        row["records_emitted"] = str(summary["records_emitted"])
    elif meta:
        ds = meta.get("decision_summary", {})
        if isinstance(ds, dict) and "records_emitted" in ds:
            row["records_emitted"] = str(ds["records_emitted"])

    # --- Artifact paths (check actual files on disk) ---------------------
    events_path = run_dir / "intrusion_events.jsonl"
    if events_path.is_file():
        row["events_jsonl_path"] = str(events_path)
    else:
        notes.append("intrusion_events.jsonl: missing")

    sidecar_csv_path = run_dir / "pose_patch_persistent_sidecar.csv"
    if sidecar_csv_path.is_file():
        row["sidecar_csv_path"] = str(sidecar_csv_path)
    else:
        notes.append("sidecar CSV: missing")

    overlay = find_artifact(run_dir, "*_intrusion_overlay.mp4")
    if overlay:
        row["overlay_video_path"] = overlay
    else:
        notes.append("overlay MP4: missing")

    tracking = find_artifact(run_dir, "*_nvdcf_posepatchpersistent.mp4")
    if tracking:
        row["tracking_video_path"] = tracking
    else:
        notes.append("tracking MP4: missing")

    # ds_app_runtime.txt -- rendered DeepStream config (not a timing log)
    ds_runtime_path = run_dir / "ds_app_runtime.txt"
    if ds_runtime_path.is_file():
        row["ds_app_runtime_path"] = str(ds_runtime_path)

    # wall_clock_runtime_sec -- derived from 03_03 log first/last timestamps
    if log_root is not None and log_root.is_dir():
        log_file = log_root / f"03_03_ds_intrusion_fsm_{run_ts}.log"
        if log_file.is_file():
            elapsed = parse_log_wall_clock(log_file)
            if elapsed is not None:
                row["wall_clock_runtime_sec"] = f"{elapsed:.1f}"
            else:
                notes.append("wall_clock: log timestamps unparseable")
        else:
            notes.append("wall_clock: log file not found")
    else:
        notes.append("wall_clock: log root unavailable")

    row["notes"] = "; ".join(notes) if notes else ""
    return row


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_stage03_root(root: Path, log_root: Path | None) -> list[dict[str, str]]:
    """Walk outputs/03_deepstream/<run_ts>/<out_base>/ and collect rows."""
    rows: list[dict[str, str]] = []
    if not root.is_dir():
        return rows

    for ts_dir in sorted(root.iterdir()):
        if not ts_dir.is_dir():
            continue
        if not RUN_TS_RE.match(ts_dir.name):
            continue
        run_ts = ts_dir.name

        # Each run_ts folder may contain one or more out_base sub-folders
        sub_dirs = sorted(d for d in ts_dir.iterdir() if d.is_dir())
        if not sub_dirs:
            # Some runs may place artifacts directly in the ts_dir
            if (ts_dir / "intrusion_summary.json").is_file() or (ts_dir / "run_meta.json").is_file():
                rows.append(collect_run(ts_dir, run_ts, log_root))
            continue

        for out_base_dir in sub_dirs:
            # Only consider folders that look like actual runs
            has_summary = (out_base_dir / "intrusion_summary.json").is_file()
            has_meta = (out_base_dir / "run_meta.json").is_file()
            has_events = (out_base_dir / "intrusion_events.jsonl").is_file()
            if has_summary or has_meta or has_events:
                rows.append(collect_run(out_base_dir, run_ts, log_root))

    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(rows: list[dict[str, str]], out_path: Path, eval_ts: str) -> None:
    lines: list[str] = []
    lines.append(f"# Stage 03 Evaluation Manifest")
    lines.append(f"")
    lines.append(f"Generated: {eval_ts}")
    lines.append(f"Runs collected: {len(rows)}")
    lines.append("")

    for row in rows:
        scene = row.get("scene_id") or "unknown"
        clip = row.get("clip_id") or "unknown"
        run_ts = row.get("run_ts", "")
        lines.append(f"## {scene} / {clip}")
        lines.append("")
        lines.append(f"- **run_ts**: `{run_ts}`")
        lines.append(f"- **run_dir**: `{row.get('run_dir', '')}`")

        confirmed = row.get("confirmed_events", "")
        if confirmed:
            lines.append(f"- **confirmed_events**: {confirmed}")

        duration = row.get("duration_sec", "")
        if duration:
            lines.append(f"- **duration_sec** (clip length): {duration}")

        wall = row.get("wall_clock_runtime_sec", "")
        if wall:
            lines.append(f"- **wall_clock_runtime_sec**: {wall}")

        pose_rt = row.get("pose_probe_status_runtime", "")
        pose_pf = row.get("pose_probe_status_preflight", "")
        if pose_rt:
            lines.append(f"- **pose_probe_status_runtime**: `{pose_rt}`")
        if pose_pf:
            lines.append(f"- **pose_probe_status_preflight**: `{pose_pf}`")

        lines.append("")
        lines.append("**Artifacts:**")
        lines.append("")
        artifact_keys = [
            ("summary_json_path", "Summary JSON"),
            ("events_jsonl_path", "Events JSONL"),
            ("sidecar_csv_path", "Sidecar CSV"),
            ("tracking_video_path", "Tracking video"),
            ("overlay_video_path", "Overlay video"),
        ]
        for key, label in artifact_keys:
            val = row.get(key, "")
            if val:
                lines.append(f"- {label}: `{val}`")
            else:
                lines.append(f"- {label}: *(missing)*")

        notes = row.get("notes", "")
        if notes:
            lines.append("")
            lines.append(f"**Notes:** {notes}")

        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect metrics from existing Stage 03 intrusion FSM outputs.",
    )
    parser.add_argument(
        "--stage03-root",
        default="outputs/03_deepstream",
        help="Root directory containing Stage 03 run folders (default: outputs/03_deepstream).",
    )
    parser.add_argument(
        "--log-root",
        default="outputs/logs/03_deepstream",
        help="Directory containing Stage 03 log files for wall-clock extraction (default: outputs/logs/03_deepstream).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/eval",
        help="Output directory for the CSV and manifest (default: outputs/eval).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stage03_root = Path(args.stage03_root)
    if not stage03_root.is_absolute():
        stage03_root = Path.cwd() / stage03_root
    stage03_root = stage03_root.resolve()

    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = Path.cwd() / log_root
    log_root = log_root.resolve()
    if not log_root.is_dir():
        print(f"Log root not found (wall-clock will be unavailable): {log_root}", file=sys.stderr)
        log_root = None  # type: ignore[assignment]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir = out_dir.resolve()

    if not stage03_root.is_dir():
        print(f"Stage 03 root not found: {stage03_root}", file=sys.stderr)
        raise SystemExit(1)

    rows = scan_stage03_root(stage03_root, log_root)
    if not rows:
        print(f"No valid Stage 03 runs found under {stage03_root}", file=sys.stderr)
        raise SystemExit(0)

    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"stage03_metrics_{eval_ts}.csv"
    manifest_path = out_dir / f"stage03_manifest_{eval_ts}.md"

    # Guard against overwrites (should not happen with second-level timestamps,
    # but be safe)
    for p in (csv_path, manifest_path):
        if p.exists():
            print(f"Output already exists, refusing to overwrite: {p}", file=sys.stderr)
            raise SystemExit(1)

    write_csv(rows, csv_path)
    write_manifest(rows, manifest_path, eval_ts)

    print(f"Collected {len(rows)} run(s)")
    print(f"CSV:      {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
