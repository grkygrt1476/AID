#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.run_utils import common_argparser, dump_run_meta, init_run


STAGE = "04_deepstream"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "deepstream" / "04_ds_yolo11_tracker_nvdcf_multistream4.txt"
DEFAULT_PLUGIN_LIB = PROJECT_ROOT / "scripts" / "04_deepstream" / "gst-dsmonitorosd" / "libnvdsgst_dsmonitorosd.so"
REPO_ALIAS = Path("/workspace/AID")
SOURCE_COUNT = 4
TILER_ROWS = 2
TILER_COLUMNS = 2
DEFAULT_OUT_BASE = "multistream4_monitor_osd"
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
    "unique-id": "15",
    "gpu-id": "0",
    "batch-size": str(SOURCE_COUNT),
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
        help="Path to the Stage 04.02 monitoring OSD plugin library.",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Alias for --out_root; output root directory for this stage.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Render config and print command without running deepstream-app",
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


def load_roi_polygon(path: Path) -> LoadedRoi:
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
    return LoadedRoi(
        source_size=(width, height),
        polygon_source=points,
    )


def parse_streammux_spec(refs: dict[str, str]) -> StreammuxSpec:
    try:
        width = int(refs["streammux_width"])
        height = int(refs["streammux_height"])
        enable_padding = bool(int(refs.get("streammux_enable_padding", "0")))
    except (KeyError, ValueError) as exc:
        raise BaselineConfigError(f"Invalid or missing streammux geometry in config refs: {exc}") from exc

    if width <= 0 or height <= 0:
        raise BaselineConfigError(f"Invalid streammux geometry: {width}x{height}")

    return StreammuxSpec(
        width=width,
        height=height,
        enable_padding=enable_padding,
    )


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
                "source_id=%s expected fixed Stage 04.02 clip_label=%s but resolved=%s; "
                "using clip-label ROI fallback.",
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
                loaded_roi = load_roi_polygon(roi_json)
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


def build_runtime_env(base_env: dict[str, str], plugin_dir: Path, overlay_specs: list[OverlaySpec]) -> dict[str, str]:
    env = dict(base_env)
    env["GST_PLUGIN_PATH"] = prepend_path_env(env.get("GST_PLUGIN_PATH"), plugin_dir)
    env["AID_DSMONITOROSD_SOURCE_COUNT"] = str(len(overlay_specs))
    env["AID_DSMONITOROSD_PERSON_CLASS_ID"] = "0"

    for overlay in overlay_specs:
        prefix = f"AID_DSMONITOROSD_SOURCE{overlay.source_id}"
        env[f"{prefix}_LABEL"] = overlay.channel_label
        env[f"{prefix}_ROI_STATUS"] = overlay.roi_status
        if overlay.roi_polygon_frame:
            env[f"{prefix}_ROI_POLY"] = serialize_roi_polygon(overlay.roi_polygon_frame)
        else:
            env.pop(f"{prefix}_ROI_POLY", None)

    return env


def stream_process_output(cmd: list[str], logger: logging.Logger, env: dict[str, str]) -> int:
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
        logger.info("[deepstream-app] %s", line.rstrip())

    return proc.wait()


def main() -> None:
    args = parse_args()
    normalize_output_args(args)

    if args.no_outputs:
        raise SystemExit("--no_outputs is not supported for this multistream monitoring wrapper.")

    source_specs = build_source_specs(args.inputs)
    template_path = project_path(args.ds_config_template)
    plugin_lib = project_path(args.plugin_lib)

    validate_file_exists(template_path, "DeepStream config template")

    if not getattr(args, "out_base", ""):
        args.out_base = DEFAULT_OUT_BASE

    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")
    if not args.dry_run:
        validate_file_exists(plugin_lib, "monitoring OSD plugin library")

    preflight_rendered_text, refs = render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=Path("/tmp/ds_multistream4_monitor_osd_preflight.mp4"),
    )
    streammux_spec = parse_streammux_spec(refs)

    infer_config_local = alias_mapped_path(refs["infer_config"])
    tracker_config_local = alias_mapped_path(refs["tracker_config"])

    validate_file_exists(infer_config_local, "infer config")
    validate_file_exists(tracker_config_local, "tracker config")
    validate_runtime_path_expectations(
        rendered_config_text=preflight_rendered_text,
        infer_config_local=infer_config_local,
        dry_run=args.dry_run,
    )

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)
    if args.dry_run and not deepstream_app:
        logger.warning("deepstream-app not found; continuing because --dry_run was requested.")
    if args.dry_run and not plugin_lib.exists():
        logger.warning("monitoring OSD plugin library not found; continuing because --dry_run was requested.")

    overlay_specs = load_overlay_specs(source_specs, streammux_spec, logger)

    output_video = run.out_dir / f"{args.out_base}_tiled_nvdcf_monitor_osd.mp4"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    rendered_text, _ = render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=output_video,
    )
    save_rendered_config(rendered_config_path, rendered_text)

    runtime_env = build_runtime_env(os.environ, plugin_lib.parent, overlay_specs)
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

    if run.outputs_enabled:
        dump_run_meta(
            run.out_dir,
            {
                "stage": STAGE,
                "source_count": SOURCE_COUNT,
                "batch_size": SOURCE_COUNT,
                "streammux_width": streammux_spec.width,
                "streammux_height": streammux_spec.height,
                "streammux_enable_padding": streammux_spec.enable_padding,
                "tiler_rows": TILER_ROWS,
                "tiler_columns": TILER_COLUMNS,
                "tiler_tile_order": [spec.source_id for spec in source_specs],
                "sources": source_meta,
                "source_id_to_clip_label": {
                    str(spec.source_id): spec.clip_label for spec in source_specs
                },
                "source_id_to_runtime_path": {
                    str(spec.source_id): str(spec.runtime_path) for spec in source_specs
                },
                "overlay_specs": overlay_meta,
                "output_video": str(output_video),
                "ds_config_template": str(template_path),
                "rendered_config_path": str(rendered_config_path),
                "plugin_lib": str(plugin_lib),
                "infer_config_source_of_truth": str(infer_config_local),
                "tracker_config_source_of_truth": str(tracker_config_local),
                "deepstream_app": deepstream_app or "",
                "deepstream_app_cmd": cmd,
                "dry_run": bool(args.dry_run),
                "run_ts": run.run_ts,
            },
        )

    logger.info("ds_config_template=%s", template_path)
    logger.info("infer_config_source_of_truth=%s", infer_config_local)
    logger.info("tracker_config_source_of_truth=%s", tracker_config_local)
    logger.info("plugin_lib=%s", plugin_lib)
    logger.info("rendered_config_path=%s", rendered_config_path)
    logger.info("output_video=%s", output_video)
    logger.info("deepstream_cmd=%s", cmd_str)
    logger.info("run_out_dir=%s", run.out_dir)
    logger.info("run_log_path=%s", run.log_path)
    logger.info("run_cmd_path=%s", run.cmd_path)
    for spec in source_specs:
        logger.info(
            "source_id=%s clip_label=%s runtime_path=%s local_path=%s",
            spec.source_id,
            spec.clip_label,
            spec.runtime_path,
            spec.local_path,
        )
    for overlay in overlay_specs:
        logger.info(
            "overlay source_id=%s channel_label=%s roi_status=%s roi_json=%s roi_points=%s",
            overlay.source_id,
            overlay.channel_label,
            overlay.roi_status,
            overlay.roi_json,
            len(overlay.roi_polygon_frame),
        )

    print(f"rendered config saved: {rendered_config_path}")
    print(f"output video path: {output_video}")
    print(f"deepstream-app command: {cmd_str}")
    if run.log_path is not None:
        print(f"log saved: {run.log_path}")
    if run.cmd_path is not None:
        print(f"wrapper cmd saved: {run.cmd_path}")

    if args.dry_run:
        logger.info("dry_run requested; not invoking deepstream-app")
        return

    exit_code = stream_process_output(cmd, logger, runtime_env)
    if exit_code != 0:
        logger.error("deepstream-app exited with code %s", exit_code)
        raise SystemExit(exit_code)

    logger.info("deepstream-app completed successfully")


if __name__ == "__main__":
    main()
