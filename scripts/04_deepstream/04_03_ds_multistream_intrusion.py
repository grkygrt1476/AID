#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    np = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.intrusion.decision_fsm import (  # noqa: E402
    STATE_CANDIDATE,
    STATE_IN_CONFIRMED,
    STATE_OUT,
    DecisionParams,
    PoseProbeSettings,
    SidecarRow,
    load_sidecar_rows,
    run_intrusion_decision_pass,
)
from aidlib.intrusion.features import FeatureConfig  # noqa: E402
from aidlib.intrusion.io import create_video_writer, load_yaml_config, write_json  # noqa: E402
from aidlib.intrusion.score import ScoreWeights  # noqa: E402
from aidlib.run_utils import common_argparser, dump_run_meta, init_run  # noqa: E402


STAGE = "04_deepstream"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "deepstream" / "04_ds_yolo11_tracker_nvdcf_multistream4.txt"
DEFAULT_PLUGIN_LIB = PROJECT_ROOT / "scripts" / "04_deepstream" / "gst-dsintrusionmeta" / "libnvdsgst_dsintrusionmeta.so"
DEFAULT_INTRUSION_CFG = PROJECT_ROOT / "configs" / "intrusion" / "mvp_v1.yaml"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "yolo11s-pose.pt"
REPO_ALIAS = Path("/workspace/AID")
SOURCE_COUNT = 4
TILER_ROWS = 2
TILER_COLUMNS = 2
DEFAULT_OUT_BASE = "multistream4_intrusion"
MAX_TRACKS_TO_DRAW = 16
DEFAULT_INPUTS = [
    "/workspace/AID/data/clips/E01_001/ev00_f1826-2854_50s.mp4",
    "/workspace/AID/data/clips/E01_004/ev00_f1803-1868_50s.mp4",
    "/workspace/AID/data/clips/E01_008/ev00_f1899-2191_50s.mp4",
    "/workspace/AID/data/clips/E01_011/ev00_f2031-2293_50s.mp4",
]
DEFAULT_FIXED_ROIS = {
    0: PROJECT_ROOT / "data" / "videos" / "rois" / "E01_001" / "roi_area01_v1.json",
    1: PROJECT_ROOT / "data" / "videos" / "rois" / "E01_004" / "roi_area01_v1.json",
    2: PROJECT_ROOT / "data" / "videos" / "rois" / "E01_008" / "roi_area01_v1.json",
    3: PROJECT_ROOT / "data" / "videos" / "rois" / "E01_011" / "roi_area01_v1.json",
}
DEFAULT_FIXED_LABELS = {
    0: "E01_001",
    1: "E01_004",
    2: "E01_008",
    3: "E01_011",
}
DS_EXAMPLE_SECTION = {
    "enable": "1",
    "full-frame": "1",
    "processing-width": "1920",
    "processing-height": "1080",
    "blur-objects": "0",
    "unique-id": "16",
    "gpu-id": "0",
    "batch-size": str(SOURCE_COUNT),
}
STATE_BOX_COLORS = {
    STATE_OUT: (80, 235, 140),
    STATE_CANDIDATE: (0, 165, 255),
    STATE_IN_CONFIRMED: (0, 0, 255),
}
STATE_TEXT_COLORS = {
    STATE_OUT: (210, 245, 220),
    STATE_CANDIDATE: (170, 220, 255),
    STATE_IN_CONFIRMED: (200, 205, 255),
}
STATE_TOKEN_MAP = {
    STATE_OUT: "NORMAL",
    STATE_CANDIDATE: "CAND",
    STATE_IN_CONFIRMED: "INTRUSION",
}
STATE_PRIORITY = {
    STATE_IN_CONFIRMED: 0,
    STATE_CANDIDATE: 1,
    STATE_OUT: 2,
}


class BaselineConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    source_id: int
    clip_label: str
    input_arg: str
    local_path: Path
    runtime_path: Path
    runtime_uri: str


@dataclass(frozen=True)
class OverlaySpec:
    source_id: int
    channel_label: str
    roi_json: Path
    roi_status: str
    roi_source_size: tuple[int, int]
    roi_polygon_source: tuple[tuple[int, int], ...]
    roi_polygon_frame: tuple[tuple[int, int], ...]
    frame_transform: dict[str, float | int | bool]
    warning: str


@dataclass(frozen=True)
class LoadedRoi:
    source_size: tuple[int, int]
    polygon_source: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StreammuxSpec:
    width: int
    height: int
    enable_padding: bool


@dataclass(frozen=True)
class SourceArtifacts:
    source_id: int
    clip_label: str
    work_dir: Path
    split_sidecar_path: Path
    events_path: Path
    summary_path: Path


@dataclass
class RenderSourceContext:
    spec: SourceSpec
    overlay: OverlaySpec
    artifacts: SourceArtifacts
    cap: Any
    fps: float
    width: int
    height: int
    frame_num: int
    sidecar_rows_by_frame: dict[int, dict[int, SidecarRow]]
    sidecar_summary: dict[str, Any]
    records_by_frame: dict[int, dict[int, dict[str, Any]]]
    decision_summary: dict[str, Any]
    pose_debug_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    active: bool = True


@dataclass(frozen=True)
class FitRect:
    display_width: int
    display_height: int
    pad_x: int
    pad_y: int
    scale: float


def require_render_deps() -> None:
    missing: list[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if missing:
        raise SystemExit(
            "Missing required Python dependencies for Stage 04.03 decision/render pass: "
            + ", ".join(missing)
        )


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def alias_mapped_path(raw_path: str) -> Path:
    if raw_path.startswith(f"{REPO_ALIAS}/"):
        rel = Path(raw_path).relative_to(REPO_ALIAS)
        return (PROJECT_ROOT / rel).resolve()
    return project_path(raw_path)


def to_runtime_repo_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved
    return REPO_ALIAS / rel


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_root=str(DEFAULT_LOG_ROOT))
    parser.add_argument(
        "--inputs",
        action="append",
        nargs="+",
        default=None,
        help="Exactly 4 input videos. Can be passed once with 4 paths or repeated.",
    )
    parser.add_argument(
        "--ds_config_template",
        default=str(DEFAULT_TEMPLATE),
        help="DeepStream app config template path",
    )
    parser.add_argument(
        "--plugin_lib",
        default=str(DEFAULT_PLUGIN_LIB),
        help="Path to the Stage 04.03 intrusion export plugin library.",
    )
    parser.add_argument(
        "--pose_model",
        default=str(DEFAULT_POSE_MODEL),
        help="Pose model path used for ankle-confirmed intrusion.",
    )
    parser.add_argument("--candidate_enter_n", type=int, default=2, help="Consecutive candidate frames required to enter CANDIDATE.")
    parser.add_argument("--confirm_enter_n", type=int, default=1, help="Consecutive ankle-in-ROI frames required to enter IN_CONFIRMED.")
    parser.add_argument("--exit_n", type=int, default=5, help="Sustained no-evidence frames required to return to OUT after grace.")
    parser.add_argument(
        "--grace_frames",
        type=int,
        default=-1,
        help="Evidence-loss grace in frames. Default is derived from configs/intrusion/mvp_v1.yaml and source FPS.",
    )
    parser.add_argument(
        "--candidate_iou_or_overlap_thr",
        type=float,
        default=0.05,
        help="ROI overlap threshold used for weak candidate geometry.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Alias for --out_root; output root directory for this stage.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs, write the runtime config, and print the planned commands without executing DeepStream or the decision/render passes.",
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


def flatten_inputs(raw_inputs: list[list[str]] | None) -> list[str]:
    if not raw_inputs:
        return list(DEFAULT_INPUTS)
    return [item for group in raw_inputs for item in group]


def parse_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or not parsed.path:
        raise SystemExit(f"Only file:// URIs are supported for --inputs: {uri}")
    return alias_mapped_path(unquote(parsed.path))


def resolve_input_local_path(raw_input: str) -> Path:
    if raw_input.startswith("file://"):
        return parse_file_uri(raw_input)
    return alias_mapped_path(raw_input)


def derive_clip_label(local_path: Path) -> str:
    parent_name = local_path.parent.name.strip()
    if parent_name:
        return parent_name
    return local_path.stem


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")


def build_source_specs(raw_inputs: list[list[str]] | None) -> list[SourceSpec]:
    flat_inputs = flatten_inputs(raw_inputs)
    if len(flat_inputs) != SOURCE_COUNT:
        raise SystemExit(f"--inputs requires exactly {SOURCE_COUNT} videos; got {len(flat_inputs)}")

    specs: list[SourceSpec] = []
    for source_id, raw_input in enumerate(flat_inputs):
        input_arg = str(raw_input).strip()
        if not input_arg:
            raise SystemExit(f"Input {source_id} is empty")

        local_path = resolve_input_local_path(input_arg)
        validate_file_exists(local_path, f"input video source_id={source_id}")
        runtime_path = to_runtime_repo_path(local_path)

        specs.append(
            SourceSpec(
                source_id=source_id,
                clip_label=derive_clip_label(local_path),
                input_arg=input_arg,
                local_path=local_path,
                runtime_path=runtime_path,
                runtime_uri=runtime_path.as_uri(),
            )
        )

    return specs


def append_ds_example_section(rendered_lines: list[str]) -> None:
    if rendered_lines and rendered_lines[-1].strip():
        rendered_lines.append("\n")
    rendered_lines.append("[ds-example]\n")
    for key, value in DS_EXAMPLE_SECTION.items():
        rendered_lines.append(f"{key}={value}\n")


def render_app_config(
    template_path: Path,
    source_specs: list[SourceSpec],
    output_file: Path,
) -> tuple[str, dict[str, str]]:
    current_section = ""
    source_uri_replaced = 0
    sink_output_replaced = 0
    streammux_batch_replaced = 0
    gie_batch_replaced = 0
    tiler_rows_replaced = 0
    tiler_columns_replaced = 0
    has_ds_example = False
    refs: dict[str, str] = {}
    rendered_lines: list[str] = []

    with template_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1].strip()
                if current_section == "ds-example":
                    has_ds_example = True
                rendered_lines.append(line)
                continue

            if "=" not in line or not stripped or stripped.startswith("#"):
                rendered_lines.append(line)
                continue

            prefix, value = line.split("=", 1)
            key = prefix.strip()
            raw_value = value.strip()
            newline = "\n" if line.endswith("\n") else ""

            if current_section.startswith("source") and key == "uri":
                source_idx = int(current_section.removeprefix("source"))
                if source_idx >= len(source_specs):
                    raise BaselineConfigError(f"Unexpected source section in template: [{current_section}]")
                rendered_lines.append(f"{prefix}={source_specs[source_idx].runtime_uri}{newline}")
                source_uri_replaced += 1
                continue

            if current_section == "streammux" and key == "batch-size":
                rendered_lines.append(f"{prefix}={SOURCE_COUNT}{newline}")
                streammux_batch_replaced += 1
                continue

            if current_section == "primary-gie" and key == "batch-size":
                rendered_lines.append(f"{prefix}={SOURCE_COUNT}{newline}")
                gie_batch_replaced += 1
                continue

            if current_section == "tiled-display" and key == "rows":
                rendered_lines.append(f"{prefix}={TILER_ROWS}{newline}")
                tiler_rows_replaced += 1
                continue

            if current_section == "tiled-display" and key == "columns":
                rendered_lines.append(f"{prefix}={TILER_COLUMNS}{newline}")
                tiler_columns_replaced += 1
                continue

            if current_section == "sink0" and key == "output-file":
                rendered_lines.append(f"{prefix}={output_file}{newline}")
                sink_output_replaced += 1
                continue

            if current_section == "primary-gie" and key == "config-file":
                refs["infer_config"] = raw_value

            if current_section == "tracker" and key == "ll-config-file":
                refs["tracker_config"] = raw_value

            if current_section == "streammux" and key == "width":
                refs["streammux_width"] = raw_value

            if current_section == "streammux" and key == "height":
                refs["streammux_height"] = raw_value

            if current_section == "streammux" and key == "enable-padding":
                refs["streammux_enable_padding"] = raw_value

            if current_section == "tiled-display" and key == "width":
                refs["tiled_width"] = raw_value

            if current_section == "tiled-display" and key == "height":
                refs["tiled_height"] = raw_value

            rendered_lines.append(line)

    if not has_ds_example:
        append_ds_example_section(rendered_lines)

    if source_uri_replaced != len(source_specs):
        raise BaselineConfigError(
            f"Expected exactly {len(source_specs)} [sourceN].uri entries in template, found {source_uri_replaced}: "
            f"{template_path}"
        )
    if sink_output_replaced != 1:
        raise BaselineConfigError(
            f"Expected exactly one [sink0].output-file in template, found {sink_output_replaced}: {template_path}"
        )
    if streammux_batch_replaced != 1:
        raise BaselineConfigError(
            f"Expected exactly one [streammux].batch-size in template, found {streammux_batch_replaced}: {template_path}"
        )
    if gie_batch_replaced != 1:
        raise BaselineConfigError(
            f"Expected exactly one [primary-gie].batch-size in template, found {gie_batch_replaced}: {template_path}"
        )
    if tiler_rows_replaced != 1 or tiler_columns_replaced != 1:
        raise BaselineConfigError(
            "Expected exactly one [tiled-display].rows and one [tiled-display].columns in template: "
            f"{template_path}"
        )
    if "infer_config" not in refs:
        raise BaselineConfigError(f"Missing [primary-gie].config-file in template: {template_path}")
    if "tracker_config" not in refs:
        raise BaselineConfigError(f"Missing [tracker].ll-config-file in template: {template_path}")

    return "".join(rendered_lines), refs


def validate_runtime_path_expectations(
    rendered_config_text: str,
    infer_config_local: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    if str(REPO_ALIAS) not in rendered_config_text:
        return
    if REPO_ALIAS.exists():
        return

    infer_text = infer_config_local.read_text(encoding="utf-8")
    if str(REPO_ALIAS) in infer_text or str(REPO_ALIAS) in rendered_config_text:
        raise SystemExit(
            "Current DeepStream source-of-truth configs still reference '/workspace/AID/...', "
            "but that path does not exist in this runtime environment. "
            "Run this script in the DeepStream environment where the repo is mounted at /workspace/AID, "
            "or update the source-of-truth config paths before using this wrapper."
        )


def save_rendered_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def prepend_path_env(env_value: str | None, path: Path) -> str:
    path_str = str(path)
    if not env_value:
        return path_str
    return f"{path_str}:{env_value}"


def fallback_roi_path(clip_label: str) -> Path:
    return PROJECT_ROOT / "data" / "videos" / "rois" / clip_label / "roi_area01_v1.json"


def _parse_roi_image_size(obj: dict) -> tuple[int, int]:
    img_size = obj.get("image_size", {})
    if isinstance(img_size, dict):
        width = int(img_size.get("width", 0) or obj.get("img_w", 0) or obj.get("width", 0))
        height = int(img_size.get("height", 0) or obj.get("img_h", 0) or obj.get("height", 0))
        return max(0, width), max(0, height)
    if isinstance(img_size, list) and len(img_size) >= 2:
        return max(0, int(img_size[0])), max(0, int(img_size[1]))
    width = int(obj.get("img_w", 0) or obj.get("width", 0))
    height = int(obj.get("img_h", 0) or obj.get("height", 0))
    return max(0, width), max(0, height)


def _parse_vertices(raw: object, key: str, roi_path: Path) -> list[tuple[float, float]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"Invalid '{key}' in ROI json '{roi_path}'")

    points: list[tuple[float, float]] = []
    for idx, point in enumerate(raw):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"Invalid point #{idx} in '{key}' of ROI json '{roi_path}'")
        points.append((float(point[0]), float(point[1])))
    return points


def _convert_norm_to_px(
    vertices_norm: list[tuple[float, float]],
    width: int,
    height: int,
    roi_path: Path,
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Normalized ROI requires image_size in '{roi_path}'")
    return [(x * float(width), y * float(height)) for (x, y) in vertices_norm]


def load_roi_spec(path: Path) -> LoadedRoi:
    if not path.exists():
        raise FileNotFoundError(f"ROI json not found: '{path}'")

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read ROI json '{path}': {exc}") from exc

    width, height = _parse_roi_image_size(obj)
    vertices_px = _parse_vertices(obj.get("vertices_px"), "vertices_px", path)

    if vertices_px is None:
        vertices_norm = None
        for key in ("vertices_norm", "vertices_normalized", "points_norm", "polygon_norm"):
            cand = _parse_vertices(obj.get(key), key, path)
            if cand is not None:
                vertices_norm = cand
                break

        if vertices_norm is not None:
            vertices_px = _convert_norm_to_px(vertices_norm, width=width, height=height, roi_path=path)
        else:
            for key in ("vertices", "points", "polygon"):
                cand = _parse_vertices(obj.get(key), key, path)
                if cand is None:
                    continue
                is_norm = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for (x, y) in cand)
                vertices_px = _convert_norm_to_px(cand, width=width, height=height, roi_path=path) if is_norm else cand
                break

    if vertices_px is None:
        raise ValueError(f"ROI vertices not found in '{path}'")

    labeled_on = str(obj.get("labeled_on", "")).strip().lower()
    disp_scale_used = obj.get("disp_scale_used", None)
    if labeled_on == "disp" and disp_scale_used is not None:
        scale = float(disp_scale_used)
        if scale <= 0:
            raise ValueError(f"Invalid disp_scale_used in '{path}': {disp_scale_used}")
        vertices_px = [(x / scale, y / scale) for (x, y) in vertices_px]

    points = tuple((int(round(x)), int(round(y))) for (x, y) in vertices_px)
    if len(points) < 3:
        raise ValueError(f"Invalid polygon in ROI json '{path}'")
    if width <= 0 or height <= 0:
        raise ValueError(f"ROI image_size missing or invalid in '{path}'")
    return LoadedRoi(source_size=(width, height), polygon_source=points)


def parse_streammux_spec(refs: dict[str, str]) -> StreammuxSpec:
    try:
        width = int(refs["streammux_width"])
        height = int(refs["streammux_height"])
        enable_padding = bool(int(refs.get("streammux_enable_padding", "0")))
    except (KeyError, ValueError) as exc:
        raise BaselineConfigError(f"Invalid or missing streammux geometry in config refs: {exc}") from exc
    if width <= 0 or height <= 0:
        raise BaselineConfigError(f"Invalid streammux geometry: {width}x{height}")
    return StreammuxSpec(width=width, height=height, enable_padding=enable_padding)


def parse_tiled_size(refs: dict[str, str]) -> tuple[int, int]:
    try:
        width = int(refs["tiled_width"])
        height = int(refs["tiled_height"])
    except (KeyError, ValueError) as exc:
        raise BaselineConfigError(f"Invalid or missing tiled-display geometry in config refs: {exc}") from exc
    if width <= 0 or height <= 0:
        raise BaselineConfigError(f"Invalid tiled-display geometry: {width}x{height}")
    return width, height


def transform_roi_to_frame_space(
    points: tuple[tuple[int, int], ...],
    source_size: tuple[int, int],
    streammux_spec: StreammuxSpec,
) -> tuple[tuple[tuple[int, int], ...], dict[str, float | int | bool]]:
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid ROI source size: {source_width}x{source_height}")

    mux_width = streammux_spec.width
    mux_height = streammux_spec.height
    if streammux_spec.enable_padding:
        scale = min(float(mux_width) / float(source_width), float(mux_height) / float(source_height))
        disp_width = float(source_width) * scale
        disp_height = float(source_height) * scale
        pad_x = (float(mux_width) - disp_width) * 0.5
        pad_y = (float(mux_height) - disp_height) * 0.5
        scale_x = scale
        scale_y = scale
    else:
        scale_x = float(mux_width) / float(source_width)
        scale_y = float(mux_height) / float(source_height)
        disp_width = float(mux_width)
        disp_height = float(mux_height)
        pad_x = 0.0
        pad_y = 0.0

    transformed = tuple(
        (
            int(round(float(x) * scale_x + pad_x)),
            int(round(float(y) * scale_y + pad_y)),
        )
        for (x, y) in points
    )
    return transformed, {
        "streammux_width": mux_width,
        "streammux_height": mux_height,
        "enable_padding": streammux_spec.enable_padding,
        "source_width": source_width,
        "source_height": source_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "display_width": disp_width,
        "display_height": disp_height,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def load_overlay_specs(
    source_specs: list[SourceSpec],
    streammux_spec: StreammuxSpec,
    logger: logging.Logger,
) -> list[OverlaySpec]:
    overlay_specs: list[OverlaySpec] = []

    for spec in source_specs:
        fixed_clip_label = DEFAULT_FIXED_LABELS.get(spec.source_id, "")
        channel_label = f"CH{spec.source_id} | {spec.clip_label}"
        roi_json = DEFAULT_FIXED_ROIS.get(spec.source_id, fallback_roi_path(spec.clip_label))

        if fixed_clip_label and spec.clip_label != fixed_clip_label:
            logger.warning(
                "source_id=%s expected fixed Stage 04.03 clip_label=%s but resolved=%s; using clip-label ROI fallback.",
                spec.source_id,
                fixed_clip_label,
                spec.clip_label,
            )
            roi_json = fallback_roi_path(spec.clip_label)

        warning = ""
        roi_status = "loaded"
        roi_source_size = (0, 0)
        roi_polygon_source: tuple[tuple[int, int], ...] = ()
        roi_polygon_frame: tuple[tuple[int, int], ...] = ()
        frame_transform: dict[str, float | int | bool] = {}

        if not roi_json.exists():
            roi_status = "missing"
            warning = f"ROI json missing for source_id={spec.source_id}: {roi_json}"
            logger.warning(warning)
        else:
            try:
                loaded_roi = load_roi_spec(roi_json)
                roi_source_size = loaded_roi.source_size
                roi_polygon_source = loaded_roi.polygon_source
                roi_polygon_frame, frame_transform = transform_roi_to_frame_space(
                    points=roi_polygon_source,
                    source_size=roi_source_size,
                    streammux_spec=streammux_spec,
                )
            except Exception as exc:
                roi_status = "error"
                warning = f"ROI load failed for source_id={spec.source_id} path={roi_json}: {exc}"
                logger.warning(warning)

        overlay_specs.append(
            OverlaySpec(
                source_id=spec.source_id,
                channel_label=channel_label,
                roi_json=roi_json,
                roi_status=roi_status,
                roi_source_size=roi_source_size,
                roi_polygon_source=roi_polygon_source,
                roi_polygon_frame=roi_polygon_frame,
                frame_transform=frame_transform,
                warning=warning,
            )
        )

    return overlay_specs


def serialize_roi_polygon(points: tuple[tuple[int, int], ...]) -> str:
    return ";".join(f"{int(x)},{int(y)}" for (x, y) in points)


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def convert_box_frame_to_source(
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    frame_transform: dict[str, float | int | bool],
    source_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    source_w = max(1.0, float(source_size[0]))
    source_h = max(1.0, float(source_size[1]))
    scale_x = max(1e-6, float(frame_transform.get("scale_x", 1.0) or 1.0))
    scale_y = max(1e-6, float(frame_transform.get("scale_y", 1.0) or 1.0))
    pad_x = float(frame_transform.get("pad_x", 0.0) or 0.0)
    pad_y = float(frame_transform.get("pad_y", 0.0) or 0.0)

    x1 = (float(left) - pad_x) / scale_x
    y1 = (float(top) - pad_y) / scale_y
    x2 = (float(left) + float(width) - pad_x) / scale_x
    y2 = (float(top) + float(height) - pad_y) / scale_y

    x1 = _clamp_float(x1, 0.0, source_w)
    y1 = _clamp_float(y1, 0.0, source_h)
    x2 = _clamp_float(x2, 0.0, source_w)
    y2 = _clamp_float(y2, 0.0, source_h)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def convert_sidecar_row_to_source_space(row: dict[str, str], overlay: OverlaySpec) -> dict[str, str]:
    converted = dict(row)
    if not overlay.frame_transform or overlay.roi_source_size[0] <= 0 or overlay.roi_source_size[1] <= 0:
        return converted

    try:
        left = float(row.get("proxy_left", "0") or 0.0)
        top = float(row.get("proxy_top", "0") or 0.0)
        width = float(row.get("proxy_width", "0") or 0.0)
        height = float(row.get("proxy_height", "0") or 0.0)
    except ValueError:
        return converted

    src_left, src_top, src_width, src_height = convert_box_frame_to_source(
        left=left,
        top=top,
        width=width,
        height=height,
        frame_transform=overlay.frame_transform,
        source_size=overlay.roi_source_size,
    )
    converted["proxy_left"] = f"{src_left:.2f}"
    converted["proxy_top"] = f"{src_top:.2f}"
    converted["proxy_width"] = f"{src_width:.2f}"
    converted["proxy_height"] = f"{src_height:.2f}"
    return converted


def build_runtime_env(
    base_env: dict[str, str],
    plugin_dir: Path,
    overlay_specs: list[OverlaySpec],
    sidecar_path: Path,
) -> dict[str, str]:
    env = dict(base_env)
    env["GST_PLUGIN_PATH"] = prepend_path_env(env.get("GST_PLUGIN_PATH"), plugin_dir)
    env["AID_DSINTRUSIONMETA_SOURCE_COUNT"] = str(len(overlay_specs))
    env["AID_DSINTRUSIONMETA_PERSON_CLASS_ID"] = "0"
    env["AID_DSINTRUSIONMETA_SIDECAR_PATH"] = str(sidecar_path)

    for overlay in overlay_specs:
        prefix = f"AID_DSINTRUSIONMETA_SOURCE{overlay.source_id}"
        env[f"{prefix}_LABEL"] = overlay.channel_label
        env[f"{prefix}_ROI_STATUS"] = overlay.roi_status
        if overlay.roi_polygon_frame:
            env[f"{prefix}_ROI_POLY"] = serialize_roi_polygon(overlay.roi_polygon_frame)
        else:
            env.pop(f"{prefix}_ROI_POLY", None)

    return env


def stream_process_output(cmd: list[str], logger: logging.Logger, env: dict[str, str], prefix: str) -> int:
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
        logger.info("[%s] %s", prefix, line.rstrip())
    return proc.wait()


def load_intrusion_defaults() -> tuple[FeatureConfig, ScoreWeights, dict[str, Any]]:
    if not DEFAULT_INTRUSION_CFG.exists():
        return FeatureConfig(), ScoreWeights(), {}
    cfg = load_yaml_config(DEFAULT_INTRUSION_CFG)
    score_cfg = cfg.get("score", {}) if isinstance(cfg, dict) else {}
    fsm_cfg = cfg.get("fsm", {}) if isinstance(cfg, dict) else {}
    return FeatureConfig.from_score_cfg(score_cfg), ScoreWeights.from_score_cfg(score_cfg), dict(fsm_cfg)


def probe_video_fps(video_path: Path) -> float:
    require_render_deps()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video for FPS probe: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    cap.release()
    return fps


def build_source_artifacts(out_dir: Path, source_specs: list[SourceSpec]) -> dict[int, SourceArtifacts]:
    artifacts: dict[int, SourceArtifacts] = {}
    for spec in source_specs:
        work_dir = out_dir / "per_source" / f"source{spec.source_id}_{spec.clip_label}"
        artifacts[spec.source_id] = SourceArtifacts(
            source_id=spec.source_id,
            clip_label=spec.clip_label,
            work_dir=work_dir,
            split_sidecar_path=work_dir / "tracking_sidecar.csv",
            events_path=work_dir / "intrusion_events.jsonl",
            summary_path=work_dir / "intrusion_summary.json",
        )
    return artifacts


def split_sidecar_by_source(
    combined_sidecar_path: Path,
    source_specs: list[SourceSpec],
    overlay_specs: list[OverlaySpec],
    artifacts_by_source: dict[int, SourceArtifacts],
) -> dict[str, Any]:
    validate_file_exists(combined_sidecar_path, "combined tracking sidecar CSV")

    writers: dict[int, csv.DictWriter] = {}
    outputs: dict[int, Any] = {}
    counts: dict[int, int] = {spec.source_id: 0 for spec in source_specs}
    overlay_by_source = {overlay.source_id: overlay for overlay in overlay_specs}
    header: list[str] | None = None

    try:
        with combined_sidecar_path.open("r", encoding="utf-8", newline="") as src_f:
            reader = csv.DictReader(src_f)
            header = list(reader.fieldnames or [])
            if not header:
                raise RuntimeError(f"Combined sidecar missing CSV header: {combined_sidecar_path}")

            for spec in source_specs:
                artifacts = artifacts_by_source[spec.source_id]
                artifacts.work_dir.mkdir(parents=True, exist_ok=True)
                out_f = artifacts.split_sidecar_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(out_f, fieldnames=header)
                writer.writeheader()
                writers[spec.source_id] = writer
                outputs[spec.source_id] = out_f

            for row in reader:
                try:
                    source_id = int(row.get("source_id", "-1"))
                except ValueError:
                    source_id = -1
                if source_id not in writers:
                    continue
                overlay = overlay_by_source[source_id]
                writers[source_id].writerow(convert_sidecar_row_to_source_space(row, overlay))
                counts[source_id] += 1
    finally:
        for handle in outputs.values():
            handle.close()

    return {
        "combined_sidecar_path": str(combined_sidecar_path),
        "header": header or [],
        "row_counts_by_source": {str(k): int(v) for (k, v) in counts.items()},
        "split_paths_by_source": {
            str(spec.source_id): str(artifacts_by_source[spec.source_id].split_sidecar_path)
            for spec in source_specs
        },
        "bbox_space_after_split": "source_coordinates",
    }


def load_events_by_frame(events_path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    records_by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    if not events_path.exists():
        return {}

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

    return dict(records_by_frame)


def make_skip_decision_summary(
    *,
    spec: SourceSpec,
    overlay: OverlaySpec,
    artifacts: SourceArtifacts,
    split_sidecar_path: Path,
    reason: str,
    params: DecisionParams,
    pose_model_path: Path,
) -> dict[str, Any]:
    artifacts.work_dir.mkdir(parents=True, exist_ok=True)
    artifacts.events_path.write_text("", encoding="utf-8")
    summary = {
        "source_id": int(spec.source_id),
        "clip_label": spec.clip_label,
        "video_path": str(spec.local_path),
        "roi_json": str(overlay.roi_json),
        "roi_status": overlay.roi_status,
        "events_path": str(artifacts.events_path),
        "sidecar": {
            "sidecar_csv": str(split_sidecar_path),
            "row_count": 0,
            "frame_min": None,
            "frame_max": None,
            "modes": [],
        },
        "records_emitted": 0,
        "tracks_seen": [],
        "confirmed_events": 0,
        "pose_probe_status": f"skipped:{reason}",
        "skip_reason": reason,
        "decision_params": {
            "candidate_enter_n": int(params.candidate_enter_n),
            "confirm_enter_n": int(params.confirm_enter_n),
            "exit_n": int(params.exit_n),
            "grace_frames": int(params.grace_frames),
            "candidate_iou_or_overlap_thr": float(params.candidate_iou_or_overlap_thr),
            "confirm_requires_ankle": bool(params.confirm_requires_ankle),
            "candidate_score_thr": float(params.candidate_score_thr),
            "proxy_start_max_age_frames": int(params.proxy_start_max_age_frames),
        },
        "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
        "pose_model": str(pose_model_path),
        "klt_included": False,
    }
    write_json(artifacts.summary_path, summary)
    return summary


def _draw_text_chip(
    frame: Any,
    *,
    text: str,
    x: int,
    y: int,
    align: str,
    scale: float,
    thickness: int,
    color: tuple[int, int, int],
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.58,
    pad_x: int = 8,
    pad_y: int = 6,
) -> tuple[int, int, int, int]:
    assert cv2 is not None
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    total_w = tw + (pad_x * 2)
    total_h = th + baseline + (pad_y * 2)

    if align == "right":
        x0 = max(0, int(x) - total_w)
    else:
        x0 = max(0, int(x))
    y0 = max(0, int(y))

    x1 = min(frame.shape[1], x0 + total_w)
    y1 = min(frame.shape[0], y0 + total_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1, cv2.LINE_AA)

    text_x = x0 + pad_x
    text_y = y0 + pad_y + th
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return x0, y0, x1, y1


def compute_fit_rect(source_w: int, source_h: int, target_w: int, target_h: int) -> FitRect:
    scale = min(float(target_w) / float(max(1, source_w)), float(target_h) / float(max(1, source_h)))
    display_width = max(1, int(round(float(source_w) * scale)))
    display_height = max(1, int(round(float(source_h) * scale)))
    pad_x = max(0, (int(target_w) - display_width) // 2)
    pad_y = max(0, (int(target_h) - display_height) // 2)
    return FitRect(
        display_width=display_width,
        display_height=display_height,
        pad_x=pad_x,
        pad_y=pad_y,
        scale=scale,
    )


def map_point_to_tile(point: tuple[float, float], fit: FitRect, tile_origin: tuple[int, int]) -> tuple[int, int]:
    return (
        int(round(tile_origin[0] + fit.pad_x + (float(point[0]) * fit.scale))),
        int(round(tile_origin[1] + fit.pad_y + (float(point[1]) * fit.scale))),
    )


def map_box_to_tile(
    box_xyxy: list[float],
    fit: FitRect,
    tile_origin: tuple[int, int],
    tile_w: int,
    tile_h: int,
) -> tuple[int, int, int, int] | None:
    x1 = tile_origin[0] + fit.pad_x + int(round(float(box_xyxy[0]) * fit.scale))
    y1 = tile_origin[1] + fit.pad_y + int(round(float(box_xyxy[1]) * fit.scale))
    x2 = tile_origin[0] + fit.pad_x + int(round(float(box_xyxy[2]) * fit.scale))
    y2 = tile_origin[1] + fit.pad_y + int(round(float(box_xyxy[3]) * fit.scale))

    min_x = tile_origin[0]
    min_y = tile_origin[1]
    max_x = tile_origin[0] + tile_w - 1
    max_y = tile_origin[1] + tile_h - 1

    x1 = max(min_x, min(max_x, x1))
    y1 = max(min_y, min(max_y, y1))
    x2 = max(min_x, min(max_x, x2))
    y2 = max(min_y, min(max_y, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def summarize_frame_state(
    *,
    frame_rows: dict[int, SidecarRow],
    frame_records: dict[int, dict[str, Any]],
    roi_status: str,
) -> dict[str, Any]:
    candidate_count = 0
    confirmed_count = 0

    for record in frame_records.values():
        state = str(record.get("state", STATE_OUT))
        if state == STATE_CANDIDATE:
            candidate_count += 1
        elif state == STATE_IN_CONFIRMED:
            confirmed_count += 1

    if roi_status != "loaded":
        global_state = "NO ROI"
    elif confirmed_count > 0:
        global_state = "INTRUSION"
    elif candidate_count > 0:
        global_state = "CAND"
    else:
        global_state = "NORMAL"

    return {
        "global_state": global_state,
        "object_count": len(frame_rows),
        "tracked_count": len(frame_rows),
        "candidate_count": candidate_count,
        "confirmed_count": confirmed_count,
    }


def select_track_ids_to_draw(
    *,
    frame_rows: dict[int, SidecarRow],
    frame_records: dict[int, dict[str, Any]],
    max_tracks_to_draw: int,
) -> list[int]:
    track_ids = set(frame_rows.keys()) | set(frame_records.keys())

    def sort_key(track_id: int) -> tuple[int, int]:
        record = frame_records.get(track_id, {})
        state = str(record.get("state", STATE_OUT))
        return (STATE_PRIORITY.get(state, 99), track_id)

    return sorted(track_ids, key=sort_key)[: max(1, int(max_tracks_to_draw))]


def build_status_line(summary: dict[str, Any], roi_status: str) -> str:
    if roi_status != "loaded":
        return f"NO ROI | O{summary['object_count']} T{summary['tracked_count']}"
    return (
        f"{summary['global_state']} | ROI ON | "
        f"O{summary['object_count']} T{summary['tracked_count']} "
        f"C{summary['candidate_count']} I{summary['confirmed_count']}"
    )


def extract_box_for_track(row: SidecarRow | None, record: dict[str, Any] | None) -> tuple[list[float] | None, bool]:
    if row is not None and row.has_valid_bbox:
        return list(row.bbox_xyxy), False
    if record is None:
        return None, False
    bbox = record.get("bbox", [])
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, False
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None, False
    if values[2] <= values[0] or values[3] <= values[1]:
        return None, False
    return values, True


def normalize_pose_status(raw_status: str) -> str:
    status = str(raw_status).strip()
    if not status:
        return ""
    if status == "ankle_outside_roi":
        return "ankle_not_in_roi"
    if status == "ankle_missing":
        return "pose_missing"
    return status


def resolve_pose_debug(
    *,
    ctx: RenderSourceContext,
    track_id: int,
    state: str,
    record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if state not in {STATE_CANDIDATE, STATE_IN_CONFIRMED}:
        ctx.pose_debug_cache.pop(track_id, None)
        return None

    if record is not None:
        confirm = record.get("confirm", {})
        if isinstance(confirm, dict):
            raw_status = str(confirm.get("status", "")).strip()
            ankles = confirm.get("ankles", [])
            ankles = ankles if isinstance(ankles, list) else []
            if raw_status and not raw_status.startswith("pose_not_needed"):
                debug = {
                    "status": normalize_pose_status(raw_status),
                    "status_raw": raw_status,
                    "ankles": ankles,
                    "attempted": bool(confirm.get("attempted")),
                }
                ctx.pose_debug_cache[track_id] = debug
                return debug

    return ctx.pose_debug_cache.get(track_id)


def draw_pose_debug(
    frame: Any,
    *,
    mapped_box: tuple[int, int, int, int],
    fit: FitRect,
    state: str,
    pose_debug: dict[str, Any] | None,
) -> None:
    assert cv2 is not None
    if pose_debug is None:
        return

    x1, y1, x2, y2 = mapped_box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    # Keep valid ankle markers slightly tolerant to edge jitter, but suppress stale
    # cached points once they drift outside the current track box region.
    margin_px = max(6, min(14, int(round(min(box_w, box_h) * 0.08))))
    min_x = x1 - margin_px
    min_y = y1 - margin_px
    max_x = x2 + margin_px
    max_y = y2 + margin_px

    ankles = pose_debug.get("ankles", [])
    ankles = ankles if isinstance(ankles, list) else []
    ankle_map = {
        str(ankle.get("name", "")).strip(): ankle
        for ankle in ankles
        if isinstance(ankle, dict)
    }

    for ankle_name in ("left_ankle", "right_ankle"):
        ankle = ankle_map.get(ankle_name)
        if not ankle:
            continue
        try:
            point = (float(ankle["x"]), float(ankle["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        px, py = map_point_to_tile(point, fit, tile_origin=(0, 0))
        if px < 0 or py < 0 or px >= frame.shape[1] or py >= frame.shape[0]:
            continue
        if px < min_x or px > max_x or py < min_y or py > max_y:
            continue
        inside_roi = bool(ankle.get("inside_roi"))
        color = (0, 0, 255) if inside_roi else (0, 165, 255)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 10, (20, 20, 20), 1, cv2.LINE_AA)


def draw_track_box(
    frame: Any,
    *,
    mapped_box: tuple[int, int, int, int],
    track_id: int,
    state: str,
    row: SidecarRow | None,
    record: dict[str, Any] | None,
    from_record_only: bool,
) -> None:
    assert cv2 is not None
    x1, y1, x2, y2 = mapped_box
    box_color = STATE_BOX_COLORS.get(state, STATE_BOX_COLORS[STATE_OUT])
    text_color = STATE_TEXT_COLORS.get(state, (240, 240, 240))
    thickness = 4 if state == STATE_IN_CONFIRMED else 3 if state == STATE_CANDIDATE else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)
    if state == STATE_IN_CONFIRMED:
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 1, cv2.LINE_AA)

    label_parts = [f"T{track_id}", STATE_TOKEN_MAP.get(state, "NORMAL")]
    if from_record_only:
        label_parts.append("grace")
    if record is not None:
        evidence = record.get("evidence", {})
        if isinstance(evidence, dict) and bool(evidence.get("ankle_confirm")):
            label_parts.append("ankle")
    elif row is not None:
        label_parts.append("track")

    _draw_text_chip(
        frame,
        text=" | ".join(label_parts),
        x=x1 + 4,
        y=max(4, y1 - 26),
        align="left",
        scale=0.48,
        thickness=1,
        color=text_color,
        bg_alpha=0.64,
    )


def render_tile(
    *,
    canvas: Any,
    tile_origin: tuple[int, int],
    tile_size: tuple[int, int],
    frame: Any | None,
    ctx: RenderSourceContext,
    frame_num: int,
) -> None:
    assert cv2 is not None
    tile_x, tile_y = tile_origin
    tile_w, tile_h = tile_size
    tile = canvas[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]
    tile[:] = 0

    if frame is not None:
        fit = compute_fit_rect(frame.shape[1], frame.shape[0], tile_w, tile_h)
        resized = cv2.resize(frame, (fit.display_width, fit.display_height), interpolation=cv2.INTER_LINEAR)
        tile[fit.pad_y : fit.pad_y + fit.display_height, fit.pad_x : fit.pad_x + fit.display_width] = resized
    else:
        fit = compute_fit_rect(ctx.width, ctx.height, tile_w, tile_h)

    frame_rows = ctx.sidecar_rows_by_frame.get(frame_num, {})
    frame_records = ctx.records_by_frame.get(frame_num, {})
    summary = summarize_frame_state(frame_rows=frame_rows, frame_records=frame_records, roi_status=ctx.overlay.roi_status)

    if ctx.overlay.roi_status == "loaded" and ctx.overlay.roi_polygon_source:
        mapped_poly = np.asarray(
            [map_point_to_tile(point, fit, tile_origin=(0, 0)) for point in ctx.overlay.roi_polygon_source],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(tile, [mapped_poly], isClosed=True, color=(90, 220, 230), thickness=2, lineType=cv2.LINE_AA)

    for track_id in select_track_ids_to_draw(
        frame_rows=frame_rows,
        frame_records=frame_records,
        max_tracks_to_draw=MAX_TRACKS_TO_DRAW,
    ):
        row = frame_rows.get(track_id)
        record = frame_records.get(track_id)
        box_xyxy, from_record_only = extract_box_for_track(row, record)
        if box_xyxy is None:
            continue

        state = str(record.get("state", STATE_OUT)) if record is not None else STATE_OUT
        pose_debug = resolve_pose_debug(
            ctx=ctx,
            track_id=track_id,
            state=state,
            record=record,
        )
        mapped_box = map_box_to_tile(
            box_xyxy=box_xyxy,
            fit=fit,
            tile_origin=(0, 0),
            tile_w=tile_w,
            tile_h=tile_h,
        )
        if mapped_box is None:
            continue
        draw_track_box(
            tile,
            mapped_box=mapped_box,
            track_id=track_id,
            state=state,
            row=row,
            record=record,
            from_record_only=from_record_only,
        )
        draw_pose_debug(
            tile,
            mapped_box=mapped_box,
            fit=fit,
            state=state,
            pose_debug=pose_debug,
        )

    if summary["confirmed_count"] > 0:
        cv2.rectangle(tile, (1, 1), (tile_w - 2, tile_h - 2), (0, 0, 255), 3, cv2.LINE_AA)

    _draw_text_chip(
        tile,
        text=ctx.overlay.channel_label,
        x=12,
        y=10,
        align="left",
        scale=0.58,
        thickness=2,
        color=(255, 255, 255),
        bg_alpha=0.62,
    )

    status_line = "ENDED" if frame is None and not ctx.active else build_status_line(summary, ctx.overlay.roi_status)
    status_color = (220, 220, 220)
    if summary["global_state"] == "CAND":
        status_color = (170, 220, 255)
    elif summary["global_state"] == "INTRUSION":
        status_color = (200, 205, 255)
    elif summary["global_state"] == "NORMAL":
        status_color = (210, 245, 220)

    _draw_text_chip(
        tile,
        text=status_line,
        x=tile_w - 12,
        y=10,
        align="right",
        scale=0.46,
        thickness=1,
        color=status_color,
        bg_alpha=0.64,
    )


def render_multistream_intrusion(
    *,
    source_specs: list[SourceSpec],
    overlay_specs: list[OverlaySpec],
    artifacts_by_source: dict[int, SourceArtifacts],
    decision_results: dict[int, dict[str, Any]],
    output_path: Path,
    tiled_size: tuple[int, int],
    logger: logging.Logger,
) -> dict[str, Any]:
    require_render_deps()

    contexts: list[RenderSourceContext] = []
    for spec, overlay in zip(source_specs, overlay_specs):
        artifacts = artifacts_by_source[spec.source_id]
        cap = cv2.VideoCapture(str(spec.local_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source video for final render: {spec.local_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                cap.release()
                raise RuntimeError(f"Could not decode frames from video: {spec.local_path}")
            height, width = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        sidecar_rows_by_frame, sidecar_summary = load_sidecar_rows(artifacts.split_sidecar_path)
        records_by_frame = load_events_by_frame(artifacts.events_path)

        contexts.append(
            RenderSourceContext(
                spec=spec,
                overlay=overlay,
                artifacts=artifacts,
                cap=cap,
                fps=fps,
                width=width,
                height=height,
                frame_num=0,
                sidecar_rows_by_frame=sidecar_rows_by_frame,
                sidecar_summary=sidecar_summary,
                records_by_frame=records_by_frame,
                decision_summary=decision_results.get(spec.source_id, {}),
            )
        )

    output_fps = contexts[0].fps if contexts else 30.0
    fps_values = [round(ctx.fps, 3) for ctx in contexts]
    if any(abs(ctx.fps - output_fps) > 0.01 for ctx in contexts[1:]):
        logger.warning("Source FPS values differ across streams %s; using first-source fps=%s for tiled render.", fps_values, output_fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(cv2, output_path, width=tiled_size[0], height=tiled_size[1], fps=output_fps)
    frames_written = 0
    tile_w = tiled_size[0] // TILER_COLUMNS
    tile_h = tiled_size[1] // TILER_ROWS

    try:
        while True:
            canvas = np.zeros((tiled_size[1], tiled_size[0], 3), dtype=np.uint8)
            any_frame = False

            for idx, ctx in enumerate(contexts):
                frame = None
                frame_num = ctx.frame_num
                if ctx.active:
                    ok, next_frame = ctx.cap.read()
                    if ok and next_frame is not None:
                        frame = next_frame
                        ctx.frame_num += 1
                        any_frame = True
                    else:
                        ctx.active = False

                tile_row = idx // TILER_COLUMNS
                tile_col = idx % TILER_COLUMNS
                render_tile(
                    canvas=canvas,
                    tile_origin=(tile_col * tile_w, tile_row * tile_h),
                    tile_size=(tile_w, tile_h),
                    frame=frame,
                    ctx=ctx,
                    frame_num=frame_num,
                )

            if not any_frame:
                break

            writer.write(canvas)
            frames_written += 1
    finally:
        writer.release()
        for ctx in contexts:
            ctx.cap.release()

    return {
        "overlay_path": str(output_path),
        "frames_written": int(frames_written),
        "fps": float(output_fps),
        "canvas_size": {"width": int(tiled_size[0]), "height": int(tiled_size[1])},
        "tile_size": {"width": int(tile_w), "height": int(tile_h)},
        "source_fps_values": fps_values,
    }


def main() -> None:
    args = parse_args()
    normalize_output_args(args)

    if args.no_outputs:
        raise SystemExit("--no_outputs is not supported for the Stage 04.03 multistream intrusion wrapper.")

    source_specs = build_source_specs(args.inputs)
    template_path = project_path(args.ds_config_template)
    plugin_lib = project_path(args.plugin_lib)
    pose_model_path = project_path(args.pose_model)

    validate_file_exists(template_path, "DeepStream config template")
    if not getattr(args, "out_base", ""):
        args.out_base = DEFAULT_OUT_BASE

    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")
    if not args.dry_run:
        validate_file_exists(plugin_lib, "Stage 04.03 intrusion export plugin library")

    preflight_rendered_text, refs = render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=Path("/tmp/ds_multistream4_intrusion_preflight.mp4"),
    )
    streammux_spec = parse_streammux_spec(refs)
    tiled_size = parse_tiled_size(refs)

    infer_config_local = alias_mapped_path(refs["infer_config"])
    tracker_config_local = alias_mapped_path(refs["tracker_config"])
    validate_file_exists(infer_config_local, "infer config")
    validate_file_exists(tracker_config_local, "tracker config")
    validate_runtime_path_expectations(
        rendered_config_text=preflight_rendered_text,
        infer_config_local=infer_config_local,
        dry_run=args.dry_run,
    )

    feature_cfg, score_weights, fsm_cfg = load_intrusion_defaults()
    if cv2 is None and args.dry_run:
        source_fps = 30.0
    else:
        source_fps = probe_video_fps(source_specs[0].local_path)
    grace_frames = int(args.grace_frames)
    if grace_frames < 0:
        grace_sec = float(fsm_cfg.get("grace_sec", 2.0))
        grace_frames = max(0, int(round(grace_sec * source_fps)))

    params = DecisionParams(
        candidate_enter_n=max(1, int(args.candidate_enter_n)),
        confirm_enter_n=max(1, int(args.confirm_enter_n)),
        exit_n=max(1, int(args.exit_n)),
        grace_frames=max(0, int(grace_frames)),
        candidate_iou_or_overlap_thr=float(args.candidate_iou_or_overlap_thr),
        confirm_requires_ankle=True,
        candidate_score_thr=float(fsm_cfg.get("cand_thr", 0.35)),
        proxy_start_max_age_frames=max(1, min(max(0, int(grace_frames)) or 1, 3)),
    )

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)
    if args.dry_run and not deepstream_app:
        logger.warning("deepstream-app not found; continuing because --dry_run was requested.")
    if args.dry_run and not plugin_lib.exists():
        logger.warning("Stage 04.03 plugin library not found; continuing because --dry_run was requested.")
    if args.dry_run and (cv2 is None or np is None):
        logger.warning("opencv-python/numpy not available; continuing because --dry_run was requested.")

    overlay_specs = load_overlay_specs(source_specs, streammux_spec, logger)
    artifacts_by_source = build_source_artifacts(run.out_dir, source_specs)

    tracking_output_video = run.out_dir / f"{args.out_base}_tiled_tracking_export.mp4"
    final_output_video = run.out_dir / f"{args.out_base}_tiled_intrusion.mp4"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    combined_sidecar_path = run.out_dir / "tracking_sidecar_combined.csv"
    run_summary_path = run.out_dir / "intrusion_run_summary.json"

    rendered_text, _ = render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=tracking_output_video,
    )
    save_rendered_config(rendered_config_path, rendered_text)

    runtime_env = build_runtime_env(
        os.environ,
        plugin_lib.parent,
        overlay_specs,
        combined_sidecar_path,
    )
    cmd = [deepstream_app or "deepstream-app", "-c", str(rendered_config_path)]
    cmd_str = shell_join(cmd)

    source_meta = [
        {
            "source_id": spec.source_id,
            "clip_label": spec.clip_label,
            "input_arg": spec.input_arg,
            "resolved_local_path": str(spec.local_path),
            "runtime_path": str(spec.runtime_path),
            "runtime_uri": spec.runtime_uri,
        }
        for spec in source_specs
    ]
    overlay_meta = [
        {
            "source_id": overlay.source_id,
            "channel_label": overlay.channel_label,
            "roi_json": str(overlay.roi_json),
            "roi_status": overlay.roi_status,
            "roi_source_size": list(overlay.roi_source_size),
            "roi_polygon_source": [list(pt) for pt in overlay.roi_polygon_source],
            "roi_polygon_frame": [list(pt) for pt in overlay.roi_polygon_frame],
            "frame_transform": overlay.frame_transform,
            "warning": overlay.warning,
        }
        for overlay in overlay_specs
    ]

    run_meta: dict[str, Any] = {
        "stage": STAGE,
        "stage_step": "04.03",
        "run_ts": run.run_ts,
        "dry_run": bool(args.dry_run),
        "source_count": SOURCE_COUNT,
        "batch_size": SOURCE_COUNT,
        "tiler_rows": TILER_ROWS,
        "tiler_columns": TILER_COLUMNS,
        "tiled_output_size": {"width": tiled_size[0], "height": tiled_size[1]},
        "streammux_width": streammux_spec.width,
        "streammux_height": streammux_spec.height,
        "streammux_enable_padding": streammux_spec.enable_padding,
        "sources": source_meta,
        "overlay_specs": overlay_meta,
        "tracking_output_video": str(tracking_output_video),
        "final_output_video": str(final_output_video),
        "combined_sidecar_path": str(combined_sidecar_path),
        "rendered_config_path": str(rendered_config_path),
        "run_summary_path": str(run_summary_path),
        "plugin_lib": str(plugin_lib),
        "pose_model": str(pose_model_path),
        "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
        "candidate_definition": "bbox_roi_geometry_candidate_only",
        "klt_included": False,
        "confirm_requires_ankle": True,
        "decision_params": {
            "candidate_enter_n": params.candidate_enter_n,
            "confirm_enter_n": params.confirm_enter_n,
            "exit_n": params.exit_n,
            "grace_frames": params.grace_frames,
            "candidate_iou_or_overlap_thr": params.candidate_iou_or_overlap_thr,
            "candidate_score_thr": params.candidate_score_thr,
            "proxy_start_max_age_frames": params.proxy_start_max_age_frames,
        },
        "candidate_score_defaults": {
            "score_cfg_path": str(DEFAULT_INTRUSION_CFG),
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
        "deepstream_app": deepstream_app or "",
        "deepstream_app_cmd": cmd,
        "ds_config_template": str(template_path),
        "infer_config_source_of_truth": str(infer_config_local),
        "tracker_config_source_of_truth": str(tracker_config_local),
    }

    if run.outputs_enabled:
        dump_run_meta(run.out_dir, run_meta)

    logger.info("ds_config_template=%s", template_path)
    logger.info("plugin_lib=%s", plugin_lib)
    logger.info("rendered_config_path=%s", rendered_config_path)
    logger.info("tracking_output_video=%s", tracking_output_video)
    logger.info("final_output_video=%s", final_output_video)
    logger.info("combined_sidecar_path=%s", combined_sidecar_path)
    logger.info("deepstream_cmd=%s", cmd_str)
    logger.info(
        "fsm=candidate_enter_n=%s confirm_enter_n=%s exit_n=%s grace_frames=%s candidate_iou_or_overlap_thr=%s",
        params.candidate_enter_n,
        params.confirm_enter_n,
        params.exit_n,
        params.grace_frames,
        params.candidate_iou_or_overlap_thr,
    )

    print(f"rendered config saved: {rendered_config_path}")
    print(f"tracking export video: {tracking_output_video}")
    print(f"final intrusion video: {final_output_video}")
    print(f"combined sidecar path: {combined_sidecar_path}")
    print(f"deepstream-app command: {cmd_str}")
    print("confirmed intrusion definition: at least one ankle enters the ROI")
    print("candidate formation: bbox-vs-ROI geometry only")
    print("KLT continuity: intentionally disabled in Stage 04.03")
    if run.log_path is not None:
        print(f"log saved: {run.log_path}")
    if run.cmd_path is not None:
        print(f"wrapper cmd saved: {run.cmd_path}")

    if args.dry_run:
        logger.info("dry_run requested; not invoking DeepStream, decision, or final render stages")
        return

    exit_code = stream_process_output(cmd, logger, runtime_env, prefix="deepstream-app")
    if exit_code != 0:
        logger.error("deepstream-app exited with code %s", exit_code)
        raise SystemExit(exit_code)
    logger.info("DeepStream tracking/export pass completed successfully")

    split_summary = split_sidecar_by_source(combined_sidecar_path, source_specs, overlay_specs, artifacts_by_source)
    logger.info("split_sidecar_summary=%s", split_summary)

    decision_results: dict[int, dict[str, Any]] = {}
    per_source_meta: list[dict[str, Any]] = []
    pose_probe_settings = PoseProbeSettings(model_path=str(pose_model_path))
    for spec, overlay in zip(source_specs, overlay_specs):
        artifacts = artifacts_by_source[spec.source_id]
        split_sidecar_path = artifacts.split_sidecar_path

        if overlay.roi_status != "loaded":
            logger.warning(
                "source_id=%s clip_label=%s has roi_status=%s; skipping ankle-confirm decision pass and keeping stream in NO ROI mode.",
                spec.source_id,
                spec.clip_label,
                overlay.roi_status,
            )
            decision_summary = make_skip_decision_summary(
                spec=spec,
                overlay=overlay,
                artifacts=artifacts,
                split_sidecar_path=split_sidecar_path,
                reason=f"roi_{overlay.roi_status}",
                params=params,
                pose_model_path=pose_model_path,
            )
        else:
            artifacts.work_dir.mkdir(parents=True, exist_ok=True)
            decision_summary = run_intrusion_decision_pass(
                video_path=spec.local_path,
                roi_json=overlay.roi_json,
                sidecar_csv=split_sidecar_path,
                events_path=artifacts.events_path,
                params=params,
                feature_cfg=feature_cfg,
                score_weights=score_weights,
                pose_probe_settings=pose_probe_settings,
            )
            decision_summary.update(
                {
                    "source_id": int(spec.source_id),
                    "clip_label": spec.clip_label,
                    "roi_status": overlay.roi_status,
                    "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
                    "candidate_definition": "bbox_roi_geometry_candidate_only",
                    "klt_included": False,
                }
            )
            write_json(artifacts.summary_path, decision_summary)

        decision_results[spec.source_id] = decision_summary
        per_source_meta.append(
            {
                "source_id": spec.source_id,
                "clip_label": spec.clip_label,
                "roi_status": overlay.roi_status,
                "split_sidecar_path": str(split_sidecar_path),
                "events_path": str(artifacts.events_path),
                "summary_path": str(artifacts.summary_path),
                "confirmed_events": int(decision_summary.get("confirmed_events", 0)),
                "pose_probe_status": str(decision_summary.get("pose_probe_status", "")),
                "skip_reason": str(decision_summary.get("skip_reason", "")),
            }
        )

    render_summary = render_multistream_intrusion(
        source_specs=source_specs,
        overlay_specs=overlay_specs,
        artifacts_by_source=artifacts_by_source,
        decision_results=decision_results,
        output_path=final_output_video,
        tiled_size=tiled_size,
        logger=logger,
    )

    run_summary = {
        "tracking_export_video": str(tracking_output_video),
        "final_intrusion_video": str(final_output_video),
        "combined_sidecar_path": str(combined_sidecar_path),
        "split_sidecar_summary": split_summary,
        "per_source": per_source_meta,
        "confirmed_intrusion_definition": "at_least_one_ankle_in_roi",
        "klt_included": False,
        "render_summary": render_summary,
        "confirmed_events_total": int(sum(int(item.get("confirmed_events", 0)) for item in per_source_meta)),
    }
    write_json(run_summary_path, run_summary)

    run_meta["split_sidecar_summary"] = split_summary
    run_meta["per_source"] = per_source_meta
    run_meta["render_summary"] = render_summary
    run_meta["confirmed_events_total"] = run_summary["confirmed_events_total"]
    if run.outputs_enabled:
        dump_run_meta(run.out_dir, run_meta)

    logger.info("final render completed output=%s confirmed_events_total=%s", final_output_video, run_summary["confirmed_events_total"])


if __name__ == "__main__":
    main()
