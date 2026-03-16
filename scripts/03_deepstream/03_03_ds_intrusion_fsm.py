#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    cv2 = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.intrusion.decision_fsm import (
    STATE_CANDIDATE,
    STATE_IN_CONFIRMED,
    STATE_OUT,
    DecisionParams,
    PoseProbeSettings,
    SidecarRow,
    load_roi_cache_from_json,
    load_sidecar_rows,
    run_intrusion_decision_pass,
)
from aidlib.intrusion.features import FeatureConfig
from aidlib.intrusion.io import create_video_writer, load_yaml_config, write_json
from aidlib.intrusion.score import ScoreWeights
from aidlib.run_utils import common_argparser, dump_run_meta, init_run


STAGE = "03_deepstream"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "deepstream" / "ds_yolo11_tracker_nvdcf_intrusionfsm.txt"
DEFAULT_INTRUSION_CFG = PROJECT_ROOT / "configs" / "intrusion" / "mvp_v1.yaml"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "yolo11s-pose.pt"
TRACKING_BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "03_deepstream" / "03_02c_ds_pose_patch_persistent.py"
AUTO_TRACKING_HARD_MAX_PROXY_AGE_FRAMES = 120

FLASHABLE_EVENT_TYPES = {"candidate_start", "in_confirmed", "exit"}
EVENT_FLASH_COLORS = {
    "candidate_start": (0, 165, 255),
    "in_confirmed": (0, 0, 255),
    "exit": (220, 220, 220),
}
MODE_BOX_COLORS = {
    "real": (80, 235, 140),
    "proxy": (0, 180, 255),
    "frozen_hold": (0, 235, 235),
}
STATE_TEXT_COLORS = {
    STATE_OUT: (215, 215, 215),
    STATE_CANDIDATE: (0, 165, 255),
    STATE_IN_CONFIRMED: (0, 0, 255),
}
STATE_PRIORITY = {
    STATE_IN_CONFIRMED: 0,
    STATE_CANDIDATE: 1,
    STATE_OUT: 2,
}


def require_cv2() -> None:
    if cv2 is None:
        raise SystemExit(
            "Missing required Python dependency: opencv-python. "
            "03_03 needs cv2 for video probing, the offline decision pass, and overlay rendering."
        )


@dataclass(frozen=True)
class ArtifactPaths:
    tracking_video_path: Path
    sidecar_path: Path
    events_path: Path
    summary_path: Path
    overlay_path: Path
    run_meta_path: Path


@dataclass(frozen=True)
class TrackingPlan:
    mode: str
    continuity_source: str
    source_sidecar_path: Path | None
    sidecar_path: Path
    tracking_video_path: Path
    tracking_script: Path | None
    tracking_cmd: list[str]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")


def expect_file(path: Path, label: str) -> Path:
    validate_file_exists(path, label)
    return path


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_root=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--input_video", required=True, help="Input video path")
    parser.add_argument("--roi_json", required=True, help="ROI json path used for candidate and ankle-in-ROI tests")
    parser.add_argument(
        "--tracking_sidecar_csv",
        default="",
        help="Optional existing 03_02c sidecar CSV to consume instead of rerunning DeepStream tracking.",
    )
    parser.add_argument(
        "--ds_config_template",
        default=str(DEFAULT_TEMPLATE),
        help="DeepStream app config template path for the 03_02c continuity baseline.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Alias for --out_root; output root directory for this stage.",
    )
    parser.add_argument("--candidate_enter_n", type=int, default=2, help="Consecutive candidate frames required to enter CANDIDATE.")
    parser.add_argument("--confirm_enter_n", type=int, default=1, help="Consecutive ankle-in-ROI frames required to confirm IN.")
    parser.add_argument("--exit_n", type=int, default=5, help="Sustained no-evidence frames required to return to OUT after grace.")
    parser.add_argument(
        "--grace_frames",
        type=int,
        default=-1,
        help="Evidence-loss grace in frames. Default is derived from configs/intrusion/mvp_v1.yaml and video FPS.",
    )
    parser.add_argument(
        "--candidate_iou_or_overlap_thr",
        type=float,
        default=0.05,
        help="ROI overlap threshold used for weak candidate geometry.",
    )
    parser.set_defaults(confirm_requires_ankle=True)
    parser.add_argument(
        "--confirm_requires_ankle",
        dest="confirm_requires_ankle",
        action="store_true",
        help="Require ankle-in-ROI evidence for IN_CONFIRMED.",
    )
    parser.add_argument(
        "--no_confirm_requires_ankle",
        dest="confirm_requires_ankle",
        action="store_false",
        help="Disable the final ankle-confirm branch and keep the run in candidate-only degraded mode.",
    )
    parser.add_argument(
        "--pose_model",
        default=str(DEFAULT_POSE_MODEL),
        help="Candidate-triggered pose model path used for ankle confirmation.",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Disable the final inspection overlay MP4 render.",
    )
    parser.add_argument(
        "--event_flash_frames",
        type=int,
        default=12,
        help="Number of frames to flash transition markers in the overlay video.",
    )
    parser.add_argument(
        "--max_tracks_to_draw",
        type=int,
        default=10,
        help="Maximum number of sidecar tracks to draw per frame in the overlay video.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs, resolve the planned stages, and print expected artifacts without running tracking, decision, or overlay stages.",
    )
    return parser.parse_args()


def cli_option_present(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def normalize_output_args(args: argparse.Namespace) -> None:
    out_dir_explicit = cli_option_present("--out_dir")
    out_root_explicit = cli_option_present("--out_root")

    if out_dir_explicit and out_root_explicit:
        raise SystemExit("Use only one of --out_dir or --out_root.")

    if out_dir_explicit:
        args.out_root = args.out_dir

    args.out_root = str(project_path(args.out_root))
    args.log_root = str(project_path(args.log_root))


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def load_intrusion_defaults() -> tuple[FeatureConfig, ScoreWeights, dict[str, Any]]:
    if not DEFAULT_INTRUSION_CFG.exists():
        return FeatureConfig(), ScoreWeights(), {}
    cfg = load_yaml_config(DEFAULT_INTRUSION_CFG)
    score_cfg = cfg.get("score", {}) if isinstance(cfg, dict) else {}
    fsm_cfg = cfg.get("fsm", {}) if isinstance(cfg, dict) else {}
    return FeatureConfig.from_score_cfg(score_cfg), ScoreWeights.from_score_cfg(score_cfg), dict(fsm_cfg)


def probe_video_fps(video_path: Path) -> float:
    require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video for FPS probe: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    cap.release()
    return fps


def build_artifact_paths(run, out_base: str) -> ArtifactPaths:
    return ArtifactPaths(
        tracking_video_path=run.out_dir / f"{out_base}_nvdcf_posepatchpersistent.mp4",
        sidecar_path=run.out_dir / "pose_patch_persistent_sidecar.csv",
        events_path=run.out_dir / "intrusion_events.jsonl",
        summary_path=run.out_dir / "intrusion_summary.json",
        overlay_path=run.out_dir / f"{out_base}_intrusion_overlay.mp4",
        run_meta_path=run.out_dir / "run_meta.json",
    )


def build_tracking_plan(
    *,
    args: argparse.Namespace,
    input_video: Path,
    template_path: Path,
    run,
    artifacts: ArtifactPaths,
) -> TrackingPlan:
    sidecar_arg = str(args.tracking_sidecar_csv).strip()
    if sidecar_arg:
        source_sidecar = project_path(sidecar_arg)
        validate_file_exists(source_sidecar, "tracking sidecar CSV")
        return TrackingPlan(
            mode="reuse_existing_sidecar",
            continuity_source="existing_sidecar_csv",
            source_sidecar_path=source_sidecar,
            sidecar_path=source_sidecar,
            tracking_video_path=artifacts.tracking_video_path,
            tracking_script=None,
            tracking_cmd=[],
        )

    validate_file_exists(template_path, "DeepStream continuity config template")
    validate_file_exists(TRACKING_BASELINE_SCRIPT, "03_02c tracking baseline script")

    cmd = [
        sys.executable,
        str(TRACKING_BASELINE_SCRIPT),
        "--input_video",
        str(input_video),
        "--ds_config_template",
        str(template_path),
        "--out_root",
        str(args.out_root),
        "--log_root",
        str(args.log_root),
        "--out_base",
        str(args.out_base),
        "--run_ts",
        str(run.run_ts),
        "--log_level",
        str(args.log_level),
        "--hard_max_proxy_age_frames",
        str(AUTO_TRACKING_HARD_MAX_PROXY_AGE_FRAMES),
    ]
    if getattr(args, "dump_env", False):
        cmd.append("--dump_env")
    if getattr(args, "no_cmdlog", False):
        cmd.append("--no_cmdlog")

    return TrackingPlan(
        mode="auto_run_03_02c",
        continuity_source="03_02c_posepatchpersistent_baseline",
        source_sidecar_path=None,
        sidecar_path=artifacts.sidecar_path,
        tracking_video_path=artifacts.tracking_video_path,
        tracking_script=TRACKING_BASELINE_SCRIPT,
        tracking_cmd=cmd,
    )


def run_streaming_command(
    cmd: list[str],
    *,
    logger: logging.Logger,
    log_prefix: str,
    env: dict[str, str] | None = None,
) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[%s] %s", log_prefix, line.rstrip())

    return proc.wait()


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events_by_frame(events_path: Path) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    records_by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    flashes_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with events_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse intrusion event jsonl line {line_no}: {events_path}") from exc

            frame_num = int(record.get("frame_num", 0))
            track_id = int(record.get("track_id", -1))
            records_by_frame[frame_num][track_id] = record

            event_type = str(record.get("event_type", "")).strip()
            if event_type in FLASHABLE_EVENT_TYPES:
                flashes_by_frame[frame_num].append(
                    {
                        "event_type": event_type,
                        "track_id": track_id,
                        "text": f"{event_type} T{track_id}",
                    }
                )

    return dict(records_by_frame), dict(flashes_by_frame)


def _clamp_xyxy(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1i = max(0, min(width - 1, int(round(x1))))
    y1i = max(0, min(height - 1, int(round(y1))))
    x2i = max(0, min(width - 1, int(round(x2))))
    y2i = max(0, min(height - 1, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i


def _draw_text_with_bg(
    frame,
    *,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    assert cv2 is not None
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x0 = max(0, x - 3)
    y0 = max(0, y - th - 5)
    x1 = min(frame.shape[1] - 1, x + tw + 3)
    y1 = min(frame.shape[0] - 1, y + baseline + 3)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_lines_panel(
    frame,
    *,
    lines: list[str],
    colors: list[tuple[int, int, int]],
    x: int,
    y: int,
    scale: float = 0.58,
    thickness: int = 2,
    alpha: float = 0.55,
) -> None:
    assert cv2 is not None
    if not lines:
        return

    text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in text_sizes) + 10
    panel_w = max(size[0] for size in text_sizes) + 20
    panel_h = line_height * len(lines) + 12
    x0 = max(0, min(frame.shape[1] - 1, x))
    y0 = max(0, min(frame.shape[0] - 1, y))
    x1 = min(frame.shape[1] - 1, x0 + panel_w)
    y1 = min(frame.shape[0] - 1, y0 + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), 1, cv2.LINE_AA)

    cursor_y = y0 + line_height
    for line, color in zip(lines, colors):
        cv2.putText(frame, line, (x0 + 8, cursor_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        cursor_y += line_height


def _frame_state_summary(
    *,
    frame_rows: dict[int, SidecarRow],
    frame_records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    candidate_count = 0
    confirmed_count = 0
    confirm_active = False
    proxy_involved = False

    for record in frame_records.values():
        state = str(record.get("state", STATE_OUT))
        if state == STATE_CANDIDATE:
            candidate_count += 1
        elif state == STATE_IN_CONFIRMED:
            confirmed_count += 1
        evidence = record.get("evidence", {})
        if isinstance(evidence, dict):
            confirm_active = confirm_active or bool(evidence.get("ankle_confirm"))
            proxy_involved = proxy_involved or bool(evidence.get("proxy_candidate"))
        if state == STATE_IN_CONFIRMED:
            confirm_active = True
        mode = str(record.get("mode", ""))
        if mode in {"proxy", "frozen_hold"}:
            proxy_involved = True

    for row in frame_rows.values():
        if row.mode in {"proxy", "frozen_hold"} and row.proxy_active:
            proxy_involved = True

    if confirmed_count > 0:
        global_state = STATE_IN_CONFIRMED
    elif candidate_count > 0:
        global_state = STATE_CANDIDATE
    else:
        global_state = STATE_OUT

    return {
        "global_state": global_state,
        "candidate_count": candidate_count,
        "confirmed_count": confirmed_count,
        "confirm_active": confirm_active,
        "proxy_involved": proxy_involved,
        "visible_tracks": len(frame_rows),
    }


def _select_track_ids_to_draw(
    *,
    frame_rows: dict[int, SidecarRow],
    frame_records: dict[int, dict[str, Any]],
    max_tracks_to_draw: int,
) -> list[int]:
    def sort_key(track_id: int) -> tuple[int, int, int]:
        record = frame_records.get(track_id, {})
        state = str(record.get("state", STATE_OUT))
        evidence = record.get("evidence", {})
        confirm_priority = 0 if isinstance(evidence, dict) and bool(evidence.get("ankle_confirm")) else 1
        return (
            STATE_PRIORITY.get(state, 99),
            confirm_priority,
            track_id,
        )

    return sorted(frame_rows.keys(), key=sort_key)[: max(1, int(max_tracks_to_draw))]


def _draw_track_box(
    frame,
    *,
    row: SidecarRow,
    record: dict[str, Any] | None,
) -> None:
    assert cv2 is not None
    clamped = _clamp_xyxy(list(row.bbox_xyxy), frame.shape[1], frame.shape[0])
    if clamped is None:
        return

    x1, y1, x2, y2 = clamped
    state = str(record.get("state", STATE_OUT)) if record is not None else STATE_OUT
    state_color = STATE_TEXT_COLORS.get(state, (255, 255, 255))
    box_color = MODE_BOX_COLORS.get(row.mode, (200, 200, 200))
    thickness = 4 if state == STATE_IN_CONFIRMED else 3 if state == STATE_CANDIDATE else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)
    if state == STATE_IN_CONFIRMED:
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), state_color, 1, cv2.LINE_AA)

    label_parts = [f"T{row.track_id}", row.mode, state]
    if row.mode != "real":
        label_parts.append(f"age={row.proxy_age}")
    if record is not None:
        evidence = record.get("evidence", {})
        if isinstance(evidence, dict) and bool(evidence.get("ankle_confirm")):
            label_parts.append("ankle confirm")
        elif isinstance(evidence, dict) and bool(evidence.get("proxy_candidate")):
            label_parts.append("proxy continuity")
    _draw_text_with_bg(
        frame,
        text=" | ".join(label_parts),
        x=x1 + 4,
        y=max(20, y1 - 8),
        color=state_color,
    )

    if record is None:
        return

    event_type = str(record.get("event_type", "")).strip()
    if event_type in FLASHABLE_EVENT_TYPES:
        _draw_text_with_bg(
            frame,
            text=event_type,
            x=x1 + 4,
            y=min(frame.shape[0] - 8, y2 + 18),
            color=EVENT_FLASH_COLORS[event_type],
            scale=0.5,
        )

    confirm = record.get("confirm", {})
    if not isinstance(confirm, dict):
        return
    ankles = confirm.get("ankles", [])
    if not isinstance(ankles, list):
        return
    for ankle in ankles:
        if not isinstance(ankle, dict):
            continue
        if "x" not in ankle or "y" not in ankle:
            continue
        ax = max(0, min(frame.shape[1] - 1, int(round(float(ankle["x"])))))
        ay = max(0, min(frame.shape[0] - 1, int(round(float(ankle["y"])))))
        color = (0, 255, 0) if bool(ankle.get("inside_roi")) else (0, 180, 255)
        cv2.circle(frame, (ax, ay), 4, color, -1, cv2.LINE_AA)


def render_intrusion_overlay(
    *,
    video_path: Path,
    roi_json: Path,
    sidecar_csv: Path,
    events_path: Path,
    overlay_path: Path,
    event_flash_frames: int,
    max_tracks_to_draw: int,
    summary_data: dict[str, Any] | None,
) -> dict[str, Any]:
    require_cv2()
    validate_file_exists(video_path, "input video")
    validate_file_exists(roi_json, "ROI json")
    validate_file_exists(sidecar_csv, "tracking sidecar CSV")
    validate_file_exists(events_path, "intrusion events jsonl")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video for overlay render: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Could not decode frames for overlay render: {video_path}")

    height, width = frame.shape[:2]
    roi_cache, roi_meta = load_roi_cache_from_json(roi_json=roi_json, width=width, height=height)
    sidecar_rows_by_frame, sidecar_summary = load_sidecar_rows(sidecar_csv)
    records_by_frame, flashes_by_frame = load_events_by_frame(events_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(cv2, overlay_path, width=width, height=height, fps=fps)

    poly = roi_cache.poly.astype("int32").reshape((-1, 1, 2))
    active_flashes: list[dict[str, Any]] = []
    frames_written = 0
    frame_num = 0
    flash_window = max(1, int(event_flash_frames))
    track_limit = max(1, int(max_tracks_to_draw))
    confirmed_total = ""
    if isinstance(summary_data, dict):
        confirmed_total = str(summary_data.get("confirmed_events", ""))

    try:
        while ok and frame is not None:
            frame_rows = sidecar_rows_by_frame.get(frame_num, {})
            frame_records = records_by_frame.get(frame_num, {})
            frame_summary = _frame_state_summary(frame_rows=frame_rows, frame_records=frame_records)

            cv2.polylines(frame, [poly], isClosed=True, color=(80, 220, 80), thickness=2, lineType=cv2.LINE_AA)

            for marker in flashes_by_frame.get(frame_num, []):
                active_flashes.append(
                    {
                        "text": str(marker["text"]),
                        "color": EVENT_FLASH_COLORS.get(str(marker["event_type"]), (220, 220, 220)),
                        "until_frame": frame_num + flash_window - 1,
                    }
                )
            active_flashes = [item for item in active_flashes if int(item["until_frame"]) >= frame_num]

            for track_id in _select_track_ids_to_draw(
                frame_rows=frame_rows,
                frame_records=frame_records,
                max_tracks_to_draw=track_limit,
            ):
                _draw_track_box(
                    frame,
                    row=frame_rows[track_id],
                    record=frame_records.get(track_id),
                )

            hud_lines = [
                f"frame {frame_num}",
                (
                    f"state {frame_summary['global_state']} | "
                    f"cand {frame_summary['candidate_count']} | in {frame_summary['confirmed_count']} | "
                    f"visible {frame_summary['visible_tracks']}"
                ),
                (
                    f"confirm_active={'yes' if frame_summary['confirm_active'] else 'no'} | "
                    f"proxy_continuity={'yes' if frame_summary['proxy_involved'] else 'no'}"
                ),
            ]
            hud_colors = [
                (235, 235, 235),
                STATE_TEXT_COLORS.get(str(frame_summary["global_state"]), (235, 235, 235)),
                (235, 235, 235),
            ]
            if confirmed_total:
                hud_lines.append(f"confirmed_events_total {confirmed_total}")
                hud_colors.append((200, 200, 200))

            _draw_lines_panel(frame, lines=hud_lines, colors=hud_colors, x=16, y=16)

            if active_flashes:
                flash_lines = [str(item["text"]) for item in active_flashes[:4]]
                flash_colors = [tuple(item["color"]) for item in active_flashes[:4]]
                _draw_lines_panel(frame, lines=flash_lines, colors=flash_colors, x=16, y=124, scale=0.52)

            writer.write(frame)
            frames_written += 1
            frame_num += 1
            ok, frame = cap.read()
    finally:
        writer.release()
        cap.release()

    return {
        "overlay_path": str(overlay_path),
        "video_path": str(video_path),
        "events_path": str(events_path),
        "sidecar_csv": str(sidecar_csv),
        "frame_count": int(frame_count) if frame_count > 0 else int(frames_written),
        "frames_written": int(frames_written),
        "fps": float(fps),
        "image_size": {"width": int(width), "height": int(height)},
        "roi": roi_meta,
        "sidecar": sidecar_summary,
        "event_flash_frames": int(flash_window),
        "max_tracks_to_draw": int(track_limit),
    }


def main() -> None:
    args = parse_args()
    normalize_output_args(args)

    if args.no_outputs:
        raise SystemExit("--no_outputs is not supported for the 03_03 intrusion FSM wrapper.")

    input_video = project_path(args.input_video)
    roi_json = project_path(args.roi_json)
    template_path = project_path(args.ds_config_template)
    pose_model_path = project_path(args.pose_model)

    validate_file_exists(input_video, "input video")
    validate_file_exists(roi_json, "ROI json")

    if not getattr(args, "out_base", ""):
        args.out_base = input_video.stem

    feature_cfg, score_weights, fsm_cfg = load_intrusion_defaults()
    if cv2 is None and args.dry_run:
        fps = 30.0
    else:
        fps = probe_video_fps(input_video)
    grace_frames = int(args.grace_frames)
    if grace_frames < 0:
        grace_sec = float(fsm_cfg.get("grace_sec", 2.0))
        grace_frames = max(0, int(round(grace_sec * fps)))

    params = DecisionParams(
        candidate_enter_n=max(1, int(args.candidate_enter_n)),
        confirm_enter_n=max(1, int(args.confirm_enter_n)),
        exit_n=max(1, int(args.exit_n)),
        grace_frames=max(0, int(grace_frames)),
        candidate_iou_or_overlap_thr=float(args.candidate_iou_or_overlap_thr),
        confirm_requires_ankle=bool(args.confirm_requires_ankle),
        candidate_score_thr=float(fsm_cfg.get("cand_thr", 0.35)),
        proxy_start_max_age_frames=max(1, min(max(0, int(grace_frames)) or 1, 3)),
    )

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    artifacts = build_artifact_paths(run, str(args.out_base))
    tracking_plan = build_tracking_plan(
        args=args,
        input_video=input_video,
        template_path=template_path,
        run=run,
        artifacts=artifacts,
    )
    logger = logging.getLogger(__name__)

    if cv2 is None and args.dry_run:
        logger.warning("opencv-python is not installed; using fps=30.0 for --dry_run only.")
    elif cv2 is None:
        require_cv2()

    overlay_requested = not bool(args.no_overlay)
    effective_sidecar_path = tracking_plan.sidecar_path
    gathered_sidecar_path: Path | None = None
    overlay_result: dict[str, Any] | None = None
    tracking_stage_meta: dict[str, Any] | None = None

    pose_branch_mode = "candidate_only_degraded"
    pose_probe_status = "pose_probe_disabled"
    pose_probe_settings = None
    if bool(args.confirm_requires_ankle):
        pose_branch_mode = "candidate_triggered_ankle_probe"
        pose_probe_settings = PoseProbeSettings(model_path=str(pose_model_path))
        if not pose_model_path.exists():
            pose_probe_status = f"pose_model_missing:{pose_model_path}"
            logger.warning(
                "Ankle-confirm branch unavailable (%s). 03_03 will remain honest and stay in candidate-only degraded mode.",
                pose_probe_status,
            )
        else:
            pose_probe_status = "pose_model_configured"

    planned_flow = [
        (
            "reuse tracking sidecar"
            if tracking_plan.mode == "reuse_existing_sidecar"
            else "auto-run 03_02c tracking baseline"
        ),
        "run intrusion FSM decision pass",
        "render overlay MP4" if overlay_requested else "skip overlay MP4",
    ]

    meta_common = {
        "stage": STAGE,
        "run_ts": run.run_ts,
        "dry_run": bool(args.dry_run),
        "planned_flow": planned_flow,
        "input_video": str(input_video),
        "roi_json": str(roi_json),
        "tracking_mode": tracking_plan.mode,
        "continuity_source": tracking_plan.continuity_source,
        "tracking_script": str(tracking_plan.tracking_script) if tracking_plan.tracking_script is not None else "",
        "tracking_stage_cmd": tracking_plan.tracking_cmd,
        "tracking_sidecar_source_path": str(tracking_plan.source_sidecar_path) if tracking_plan.source_sidecar_path is not None else "",
        "sidecar_path": str(effective_sidecar_path),
        "gathered_sidecar_path": "",
        "output_video": str(tracking_plan.tracking_video_path) if tracking_plan.mode == "auto_run_03_02c" else "",
        "tracking_output_video": str(tracking_plan.tracking_video_path) if tracking_plan.mode == "auto_run_03_02c" else "",
        "events_path": str(artifacts.events_path),
        "summary_path": str(artifacts.summary_path),
        "overlay_path": str(artifacts.overlay_path) if overlay_requested else "",
        "overlay_enabled": bool(overlay_requested),
        "overlay_event_flash_frames": max(1, int(args.event_flash_frames)),
        "overlay_max_tracks_to_draw": max(1, int(args.max_tracks_to_draw)),
        "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
        "proxy_can_confirm": False,
        "candidate_sources": ["real_track", "proxy_klt_continuity"],
        "confirm_source": "candidate_triggered_pose_ankle_probe" if bool(args.confirm_requires_ankle) else "disabled_candidate_only",
        "degraded_behavior_without_ankle": "stay_candidate_without_promoting_to_in_confirmed",
        "ds_config_template": str(template_path),
        "pose_model": str(pose_model_path),
        "pose_branch_mode": pose_branch_mode,
        "pose_probe_status": pose_probe_status,
        "decision_params": {
            "candidate_enter_n": params.candidate_enter_n,
            "confirm_enter_n": params.confirm_enter_n,
            "exit_n": params.exit_n,
            "grace_frames": params.grace_frames,
            "candidate_iou_or_overlap_thr": params.candidate_iou_or_overlap_thr,
            "confirm_requires_ankle": params.confirm_requires_ankle,
        },
        "candidate_score_defaults": {
            "score_cfg_path": str(DEFAULT_INTRUSION_CFG),
            "candidate_score_thr": params.candidate_score_thr,
            "feature_cfg": {
                "d0_ratio": feature_cfg.d0_ratio,
                "ov0": feature_cfg.ov0,
                "g0_ratio": feature_cfg.g0_ratio,
                "g1_ratio": feature_cfg.g1_ratio,
                "lower_ratio": feature_cfg.lower_ratio,
            },
            "score_weights": {
                "wd": score_weights.wd,
                "wo": score_weights.wo,
                "wg": score_weights.wg,
            },
        },
    }

    if run.outputs_enabled:
        dump_run_meta(run.out_dir, meta_common)

    logger.info("input_video=%s", input_video)
    logger.info("roi_json=%s", roi_json)
    logger.info("tracking_mode=%s", tracking_plan.mode)
    logger.info("continuity_source=%s", tracking_plan.continuity_source)
    logger.info("sidecar_path=%s", effective_sidecar_path)
    logger.info("events_path=%s", artifacts.events_path)
    logger.info("summary_path=%s", artifacts.summary_path)
    logger.info("overlay_enabled=%s", overlay_requested)
    logger.info("overlay_path=%s", artifacts.overlay_path if overlay_requested else "")
    logger.info(
        "fsm=candidate_enter_n=%s confirm_enter_n=%s exit_n=%s grace_frames=%s candidate_iou_or_overlap_thr=%s",
        params.candidate_enter_n,
        params.confirm_enter_n,
        params.exit_n,
        params.grace_frames,
        params.candidate_iou_or_overlap_thr,
    )
    if tracking_plan.tracking_cmd:
        logger.info("tracking_stage_cmd=%s", shell_join(tracking_plan.tracking_cmd))

    print(f"input video: {input_video}")
    print(f"roi json: {roi_json}")
    print(f"tracking mode: {tracking_plan.mode}")
    print(f"continuity source: {tracking_plan.continuity_source}")
    print(f"tracking sidecar path: {effective_sidecar_path}")
    print(f"events output path: {artifacts.events_path}")
    print(f"summary output path: {artifacts.summary_path}")
    if overlay_requested:
        print(f"overlay output path: {artifacts.overlay_path}")
    else:
        print("overlay output path: disabled by --no_overlay")
    print("product confirm definition: at least one ankle enters the ROI")
    print("proxy can maintain candidate continuity but cannot directly confirm final intrusion")
    print(f"pose confirm branch: {meta_common['confirm_source']}")
    print("planned flow:")
    for idx, step in enumerate(planned_flow, start=1):
        print(f"  {idx}. {step}")
    if tracking_plan.tracking_cmd:
        print(f"internal 03_02c command: {shell_join(tracking_plan.tracking_cmd)}")
    if run.log_path is not None:
        print(f"log saved: {run.log_path}")
    if run.cmd_path is not None:
        print(f"wrapper cmd saved: {run.cmd_path}")

    if args.dry_run:
        logger.info("dry_run requested; not invoking tracking, decision, or overlay stages")
        return

    if tracking_plan.mode == "auto_run_03_02c":
        exit_code = run_streaming_command(
            tracking_plan.tracking_cmd,
            logger=logger,
            log_prefix="03_02c",
        )
        if exit_code != 0:
            logger.error("03_02c tracking stage exited with code %s", exit_code)
            raise SystemExit(exit_code)

        effective_sidecar_path = expect_file(artifacts.sidecar_path, "tracking stage sidecar CSV")
        expect_file(artifacts.tracking_video_path, "tracking stage output video")
        if artifacts.run_meta_path.exists():
            try:
                tracking_stage_meta = load_json_file(artifacts.run_meta_path)
            except json.JSONDecodeError:
                logger.warning("Failed to parse 03_02c run_meta.json after tracking stage: %s", artifacts.run_meta_path)
    else:
        effective_sidecar_path = expect_file(tracking_plan.sidecar_path, "tracking sidecar CSV")
        if effective_sidecar_path != artifacts.sidecar_path:
            shutil.copy2(effective_sidecar_path, artifacts.sidecar_path)
            gathered_sidecar_path = artifacts.sidecar_path

    decision_summary = run_intrusion_decision_pass(
        video_path=input_video,
        roi_json=roi_json,
        sidecar_csv=effective_sidecar_path,
        events_path=artifacts.events_path,
        params=params,
        feature_cfg=feature_cfg,
        score_weights=score_weights,
        pose_probe_settings=pose_probe_settings,
    )
    expect_file(artifacts.events_path, "intrusion decision events jsonl")
    write_json(artifacts.summary_path, decision_summary)
    expect_file(artifacts.summary_path, "intrusion summary json")

    if overlay_requested:
        try:
            overlay_result = render_intrusion_overlay(
                video_path=input_video,
                roi_json=roi_json,
                sidecar_csv=effective_sidecar_path,
                events_path=artifacts.events_path,
                overlay_path=artifacts.overlay_path,
                event_flash_frames=max(1, int(args.event_flash_frames)),
                max_tracks_to_draw=max(1, int(args.max_tracks_to_draw)),
                summary_data=decision_summary,
            )
        except Exception as exc:
            logger.exception("overlay render failed")
            raise SystemExit(f"Overlay render failed for expected artifact {artifacts.overlay_path}: {exc}") from exc
        expect_file(artifacts.overlay_path, "overlay MP4")

    final_meta = dict(meta_common)
    final_meta.update(
        {
            "sidecar_path": str(effective_sidecar_path),
            "gathered_sidecar_path": str(gathered_sidecar_path) if gathered_sidecar_path is not None else "",
            "decision_summary": decision_summary,
            "overlay_result": overlay_result or {},
            "artifacts": {
                "tracking_output_video": str(tracking_plan.tracking_video_path) if tracking_plan.mode == "auto_run_03_02c" else "",
                "tracking_sidecar_used_for_decision": str(effective_sidecar_path),
                "tracking_sidecar_gathered_copy": str(gathered_sidecar_path) if gathered_sidecar_path is not None else "",
                "intrusion_events_jsonl": str(artifacts.events_path),
                "intrusion_summary_json": str(artifacts.summary_path),
                "overlay_mp4": str(artifacts.overlay_path) if overlay_requested else "",
            },
        }
    )
    if tracking_stage_meta is not None:
        final_meta["tracking_stage_meta_snapshot"] = tracking_stage_meta

    if run.outputs_enabled:
        dump_run_meta(run.out_dir, final_meta)

    logger.info("intrusion decision pass completed successfully")
    logger.info("events_path=%s", artifacts.events_path)
    logger.info("summary_path=%s", artifacts.summary_path)
    if overlay_requested:
        logger.info("overlay_path=%s", artifacts.overlay_path)

    print(f"intrusion events saved: {artifacts.events_path}")
    print(f"intrusion summary saved: {artifacts.summary_path}")
    if overlay_requested:
        print(f"intrusion overlay saved: {artifacts.overlay_path}")
    if tracking_plan.mode == "auto_run_03_02c":
        print(f"tracking video saved: {tracking_plan.tracking_video_path}")
        print(f"tracking sidecar produced: {effective_sidecar_path}")
    elif gathered_sidecar_path is not None:
        print(f"tracking sidecar gathered copy: {gathered_sidecar_path}")


if __name__ == "__main__":
    main()
