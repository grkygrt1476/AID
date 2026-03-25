#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.intrusion.io import write_json  # noqa: E402
from aidlib.run_utils import common_argparser, init_run  # noqa: E402


STAGE = "04_deepstream"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_OUT_BASE = "multistream16_runner"
DEFAULT_MANIFEST_4CH = PROJECT_ROOT / "configs" / "deepstream" / "16ch" / "sources_4ch_demo.json"
DEFAULT_MANIFEST_16CH = PROJECT_ROOT / "configs" / "deepstream" / "16ch" / "sources_16ch_demo.json"
DEFAULT_OVERRIDES = PROJECT_ROOT / "configs" / "deepstream" / "16ch" / "source_overrides.json"
DEFAULT_DS_TEMPLATE = PROJECT_ROOT / "configs" / "deepstream" / "16ch" / "ds_multistream_16ch_boundary_reacquire.txt"
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "data" / "prepared_inputs"
PREP_SCRIPT = PROJECT_ROOT / "scripts" / "04_ds_multi_stream" / "04_05a_make_roi_crop_clips.py"
CORE_SCRIPT = PROJECT_ROOT / "scripts" / "04_ds_multi_stream" / "04_05_ds_multistream_boundary_reacquire.py"
FPS_TOLERANCE = 0.5
PREPARED_CACHE_VERSION = "native16_prepare_v1"
PREPARED_CACHE_PREP_OUT_BASE = "_prepared_cache_prepare"
PREPARED_CACHE_CORE_OUT_BASE = "_prepared_cache_core"


@dataclass(frozen=True)
class SourceOverride:
    compatibility_group: str
    requires_isolated_batch: bool
    ds_config_template: Path | None
    extra_core_args: tuple[str, ...]
    tags: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class PlannedSource:
    manifest_index: int
    source_id: str
    source_key: str
    video_path: Path
    roi_json: Path
    tags: tuple[str, ...]
    notes: str
    nominal_fps: float | None
    nominal_frame_count: int | None
    frame_width: int | None
    frame_height: int | None
    metadata_backend: str | None
    override: SourceOverride
    run_signature: tuple[str, str, tuple[str, ...]]


@dataclass(frozen=True)
class NativeExecutionPlan:
    source_count: int
    tiler_rows: int
    tiler_columns: int
    ds_config_template: Path
    extra_core_args: tuple[str, ...]
    sources: tuple[PlannedSource, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    core_args: tuple[str, ...]


@dataclass(frozen=True)
class VideoMetadata:
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    backend: str | None


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def alias_mapped_path(value: str | Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    repo_alias = Path("/workspace/AID")
    if str(path).startswith(f"{repo_alias}/"):
        rel = path.relative_to(repo_alias)
        return (PROJECT_ROOT / rel).resolve()
    return project_path(path)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read {label} '{path}': {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return obj


def normalize_positive_float(raw_value: Any) -> float | None:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def normalize_positive_int(raw_value: Any) -> int | None:
    value = normalize_positive_float(raw_value)
    if value is None:
        return None
    return int(round(value))


def probe_video_metadata_ffprobe(video_path: Path, logger: logging.Logger) -> VideoMetadata | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_frames,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.warning("ffprobe metadata probe failed for '%s': %s", video_path, proc.stderr.strip() or proc.stdout.strip())
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        logger.warning("ffprobe metadata parse failed for '%s': %s", video_path, exc)
        return None
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        logger.warning("ffprobe returned no video streams for '%s'", video_path)
        return None
    stream = streams[0] if isinstance(streams[0], dict) else {}
    raw_fps = str(stream.get("r_frame_rate", "")).strip()
    fps = None
    if raw_fps:
        try:
            fps = normalize_positive_float(Fraction(raw_fps))
        except Exception:
            fps = None
    return VideoMetadata(
        fps=fps,
        frame_count=normalize_positive_int(stream.get("nb_frames")),
        width=normalize_positive_int(stream.get("width")),
        height=normalize_positive_int(stream.get("height")),
        backend="ffprobe",
    )


def probe_video_metadata_cv2(video_path: Path, logger: logging.Logger) -> VideoMetadata | None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        logger.warning("OpenCV metadata probe unavailable for '%s': %s", video_path, exc)
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.warning("OpenCV metadata probe could not open '%s'", video_path)
            return None
        return VideoMetadata(
            fps=normalize_positive_float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=normalize_positive_int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=normalize_positive_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=normalize_positive_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            backend="cv2",
        )
    finally:
        cap.release()


def probe_video_metadata(video_path: Path, logger: logging.Logger) -> VideoMetadata:
    ffprobe_meta = probe_video_metadata_ffprobe(video_path, logger)
    if ffprobe_meta is not None:
        return ffprobe_meta
    cv2_meta = probe_video_metadata_cv2(video_path, logger)
    if cv2_meta is not None:
        return cv2_meta
    logger.warning("Video metadata probe unavailable for '%s'; proceeding with degraded validation", video_path)
    return VideoMetadata(fps=None, frame_count=None, width=None, height=None, backend=None)


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_root=str(DEFAULT_LOG_ROOT), out_base=DEFAULT_OUT_BASE)
    parser.add_argument("--manifest", default="", help="Manifest JSON path. Defaults from --profile when omitted.")
    parser.add_argument("--source_overrides", default=str(DEFAULT_OVERRIDES), help="Source override JSON path.")
    parser.add_argument("--ds_config_template", default=str(DEFAULT_DS_TEMPLATE), help="Default DeepStream config template path.")
    parser.add_argument("--prepared_root", default=str(DEFAULT_PREPARED_ROOT), help="Root directory for reusable prepared-input caches.")
    parser.add_argument("--prepared_cache_key", default="", help="Human-readable prepared-input cache key, for example 16ch_10fps.")
    parser.add_argument("--reuse_prepared", action="store_true", help="Reuse a matching prepared-input cache and skip Stage 04.05a.")
    parser.add_argument("--rebuild_prepared", action="store_true", help="Rebuild the prepared-input cache for --prepared_cache_key instead of reusing it.")
    parser.add_argument("--profile", choices=["4ch", "16ch"], default="", help="Convenience manifest selector when --manifest is omitted.")
    parser.add_argument("--validate_only", action="store_true", help="Validate manifest, ROI policy, config compatibility, and native execution planning without running subprocesses.")
    parser.add_argument("--dry_run", action="store_true", help="Run Stage 04.05a and Stage 04.05 once in dry-run mode for the selected manifest.")
    parser.add_argument("--only_batch", type=int, default=-1, help="Deprecated sequential-batch option. Native execution no longer supports it.")
    parser.add_argument("--skip_batches", default="", help="Deprecated sequential-batch option. Native execution no longer supports it.")
    parser.add_argument(
        "--input_read_fps",
        type=float,
        default=0.0,
        help="Target effective input processing/read FPS. 0 keeps the original source FPS when source metadata allows it.",
    )
    parser.add_argument(
        "--reacquire_pose_imgsz",
        type=int,
        default=960,
        help="Crop-based Stage 04.05 pose/reacquire inference size. Use 640 for the quick 640 experiment path.",
    )
    parser.add_argument(
        "--experiment",
        choices=["baseline", "b1", "b2", "b1_b2", "b3_light"],
        default="baseline",
        help="Optional Stage 04.05 Pack B experiment preset for the native 16ch burst benchmark runner.",
    )
    parser.add_argument(
        "--b3_light_render_budget_max_calls_per_frame",
        type=int,
        default=4,
        help="Per-frame render model call cap used by --experiment b3_light.",
    )
    parser.add_argument("--tiler_rows", type=int, default=0, help="Optional native tiled-display row override. 0 auto-resolves from source count.")
    parser.add_argument("--tiler_columns", type=int, default=0, help="Optional native tiled-display column override. 0 auto-resolves from source count.")
    return parser.parse_args()


def resolve_manifest_path(args: argparse.Namespace) -> Path:
    manifest_arg = str(args.manifest).strip()
    if manifest_arg:
        return project_path(manifest_arg)
    profile = str(getattr(args, "profile", "")).strip().lower()
    if profile == "4ch":
        return DEFAULT_MANIFEST_4CH
    if profile == "16ch":
        return DEFAULT_MANIFEST_16CH
    raise SystemExit("Provide --manifest or --profile {4ch,16ch}.")


def validate_canonical_roi(source_id: str, roi_json: Path) -> None:
    expected_root = (PROJECT_ROOT / "configs" / "rois" / source_id).resolve()
    resolved = roi_json.resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise SystemExit(f"ROI for {source_id} must live under {expected_root}, got {resolved}") from exc
    if not resolved.name.endswith("_fix.json"):
        raise SystemExit(f"ROI for {source_id} must be a canonical *_fix.json file, got {resolved.name}")


def resolve_native_tiler_layout(source_count: int, *, requested_rows: int = 0, requested_columns: int = 0) -> tuple[int, int]:
    if source_count <= 0:
        raise SystemExit(f"source_count must be positive, got {source_count}")
    rows = max(0, int(requested_rows))
    columns = max(0, int(requested_columns))
    if rows == 0 and columns == 0:
        columns = max(1, int(math.ceil(math.sqrt(source_count))))
        rows = max(1, int(math.ceil(float(source_count) / float(columns))))
    elif rows == 0:
        rows = max(1, int(math.ceil(float(source_count) / float(columns))))
    elif columns == 0:
        columns = max(1, int(math.ceil(float(source_count) / float(rows))))
    if rows * columns < source_count:
        raise SystemExit(
            f"Invalid native tiler layout rows={rows} columns={columns} for source_count={source_count}; capacity is too small."
        )
    return rows, columns


def parse_override_entry(raw_value: dict[str, Any], *, base_path: Path | None) -> SourceOverride:
    compatibility_group = str(raw_value.get("compatibility_group", "default")).strip() or "default"
    requires_isolated_batch = bool(raw_value.get("requires_isolated_batch", False))
    ds_config_raw = str(raw_value.get("ds_config_template", "")).strip()
    ds_config_template = project_path(ds_config_raw) if ds_config_raw else base_path
    extra_core_args_raw = raw_value.get("extra_core_args", [])
    if not isinstance(extra_core_args_raw, list) or not all(isinstance(item, str) for item in extra_core_args_raw):
        raise SystemExit("override extra_core_args must be a list of strings")
    tags_raw = raw_value.get("tags", [])
    if tags_raw and (not isinstance(tags_raw, list) or not all(isinstance(item, str) for item in tags_raw)):
        raise SystemExit("override tags must be a list of strings")
    return SourceOverride(
        compatibility_group=compatibility_group,
        requires_isolated_batch=requires_isolated_batch,
        ds_config_template=ds_config_template,
        extra_core_args=tuple(str(item) for item in extra_core_args_raw),
        tags=tuple(str(item) for item in tags_raw),
        notes=str(raw_value.get("notes", "")).strip(),
    )


def load_overrides(path: Path, default_ds_template: Path) -> tuple[SourceOverride, dict[str, SourceOverride]]:
    obj = load_json_object(path, "source_overrides")
    default_policy_raw = obj.get("default_policy", {})
    if default_policy_raw and not isinstance(default_policy_raw, dict):
        raise SystemExit(f"default_policy must be an object in {path}")
    default_policy = parse_override_entry(default_policy_raw or {}, base_path=default_ds_template)
    overrides_raw = obj.get("source_overrides", {})
    if not isinstance(overrides_raw, dict):
        raise SystemExit(f"source_overrides must be an object in {path}")
    overrides: dict[str, SourceOverride] = {}
    for source_id, raw_value in overrides_raw.items():
        if not isinstance(raw_value, dict):
            raise SystemExit(f"Override for {source_id} must be an object in {path}")
        overrides[str(source_id).strip()] = parse_override_entry(raw_value, base_path=default_ds_template)
    return default_policy, overrides


def parse_manifest_sources(
    manifest: dict[str, Any],
    *,
    default_override: SourceOverride,
    overrides_by_source: dict[str, SourceOverride],
    forced_ds_config_template: Path,
    logger: logging.Logger,
) -> tuple[list[PlannedSource], list[dict[str, Any]], int]:
    manifest_version = int(manifest.get("manifest_version", 0) or 0)
    if manifest_version != 1:
        raise SystemExit(f"Unsupported manifest_version={manifest_version}; expected 1")

    source_count_target = int(manifest.get("core_batch_size", manifest.get("batch_size", 0)) or 0)
    if source_count_target <= 0:
        raise SystemExit("Manifest core_batch_size must be a positive integer for native execution planning")

    sources_raw = manifest.get("sources", [])
    if not isinstance(sources_raw, list) or not sources_raw:
        raise SystemExit("Manifest sources must be a non-empty list")

    planned_sources: list[PlannedSource] = []
    source_status_rows: list[dict[str, Any]] = []
    seen_source_keys: set[str] = set()

    for idx, raw_source in enumerate(sources_raw):
        if not isinstance(raw_source, dict):
            raise SystemExit(f"Manifest source entry #{idx} must be an object")

        source_id = str(raw_source.get("source_id", "")).strip()
        video_path_raw = str(raw_source.get("video_path", "")).strip()
        roi_json_raw = str(raw_source.get("roi_json", "")).strip()
        enabled = bool(raw_source.get("enabled", False))
        tags_raw = raw_source.get("tags", [])
        if not source_id:
            raise SystemExit(f"Manifest source entry #{idx} is missing source_id")
        if not video_path_raw:
            raise SystemExit(f"Manifest source entry #{idx} ({source_id}) is missing video_path")
        if not roi_json_raw:
            raise SystemExit(f"Manifest source entry #{idx} ({source_id}) is missing roi_json")
        if not isinstance(tags_raw, list) or not tags_raw or not all(isinstance(item, str) for item in tags_raw):
            raise SystemExit(f"Manifest source entry #{idx} ({source_id}) must provide a non-empty string tag list")

        video_path = project_path(video_path_raw)
        roi_json = project_path(roi_json_raw)
        artifact_name = str(raw_source.get("artifact_name", "")).strip() or f"{source_id}__{video_path.stem}"
        if artifact_name in seen_source_keys:
            raise SystemExit(f"Duplicate source-aware artifact_name/source_key '{artifact_name}' in manifest")
        seen_source_keys.add(artifact_name)

        if not enabled:
            source_status_rows.append(
                {
                    "manifest_index": idx,
                    "source_id": source_id,
                    "source_key": artifact_name,
                    "enabled": False,
                    "status": "disabled",
                    "video_path": str(video_path),
                    "roi_json": str(roi_json),
                    "nominal_fps": "",
                    "nominal_frame_count": "",
                    "frame_width": "",
                    "frame_height": "",
                    "metadata_backend": "",
                    "execution_mode": "",
                }
            )
            continue

        if not video_path.exists() or not video_path.is_file():
            raise SystemExit(f"Missing video_path for {source_id}: {video_path}")
        if not roi_json.exists() or not roi_json.is_file():
            raise SystemExit(f"Missing roi_json for {source_id}: {roi_json}")
        validate_canonical_roi(source_id, roi_json)

        metadata = probe_video_metadata(video_path, logger)
        override = overrides_by_source.get(source_id, default_override)
        if override.requires_isolated_batch:
            raise SystemExit(
                f"Source {source_id} requires isolated_batch, which is incompatible with the native single-run 16-stream benchmark path"
            )
        ds_config_template = forced_ds_config_template
        if ds_config_template is None or not ds_config_template.exists():
            raise SystemExit(f"DeepStream config template for {source_id} does not exist: {ds_config_template}")

        planned_sources.append(
            PlannedSource(
                manifest_index=idx,
                source_id=source_id,
                source_key=artifact_name,
                video_path=video_path,
                roi_json=roi_json,
                tags=tuple(str(item) for item in tags_raw),
                notes=str(raw_source.get("notes", "")).strip(),
                nominal_fps=metadata.fps,
                nominal_frame_count=metadata.frame_count,
                frame_width=metadata.width,
                frame_height=metadata.height,
                metadata_backend=metadata.backend,
                override=override,
                run_signature=(
                    override.compatibility_group,
                    str(ds_config_template),
                    tuple(override.extra_core_args),
                ),
            )
        )
        logger.info(
            "validated source manifest_index=%s source_id=%s source_key=%s nominal_fps=%s probe_backend=%s signature=%s",
            idx,
            source_id,
            artifact_name,
            f"{metadata.fps:.3f}" if metadata.fps is not None else "unavailable",
            metadata.backend or "unavailable",
            planned_sources[-1].run_signature,
        )

    if not planned_sources:
        raise SystemExit("Manifest has no enabled sources after filtering")
    if len(planned_sources) != source_count_target:
        raise SystemExit(
            f"Manifest core_batch_size/source_count target={source_count_target} does not match enabled source count={len(planned_sources)}"
        )
    return planned_sources, source_status_rows, source_count_target


def validate_native_run_signature(sources: list[PlannedSource]) -> tuple[Path, tuple[str, ...]]:
    signatures = {source.run_signature for source in sources}
    if len(signatures) != 1:
        summary = ", ".join(f"{source.source_key}:{source.run_signature}" for source in sources)
        raise SystemExit(
            "Native single-run execution requires one shared override/config signature across all enabled sources; "
            f"got {len(signatures)} signatures: {summary}"
        )
    signature = next(iter(signatures))
    return Path(signature[1]), signature[2]


def resolve_experiment_spec(args: argparse.Namespace) -> ExperimentSpec:
    experiment = str(getattr(args, "experiment", "baseline")).strip().lower() or "baseline"
    if experiment == "baseline":
        return ExperimentSpec(
            name="baseline",
            description="Pack B disabled; baseline burst benchmark path.",
            core_args=(),
        )
    if experiment == "b1":
        return ExperimentSpec(
            name="b1",
            description="Enable semantic-safe pose probe reuse only.",
            core_args=("--enable_pose_probe_reuse",),
        )
    if experiment == "b2":
        return ExperimentSpec(
            name="b2",
            description="Enable decision lazy decode only.",
            core_args=("--enable_decision_lazy_decode",),
        )
    if experiment == "b1_b2":
        return ExperimentSpec(
            name="b1_b2",
            description="Enable semantic-safe pose probe reuse and decision lazy decode.",
            core_args=("--enable_pose_probe_reuse", "--enable_decision_lazy_decode"),
        )
    if experiment == "b3_light":
        return ExperimentSpec(
            name="b3_light",
            description="Enable light render model budget hard cap.",
            core_args=(
                "--enable_render_model_budget",
                "--render_model_budget_max_calls_per_frame",
                str(max(1, int(getattr(args, "b3_light_render_budget_max_calls_per_frame", 4)))),
            ),
        )
    raise SystemExit(f"Unsupported experiment preset: {experiment}")


def count_source_sections(template_path: Path) -> int:
    count = 0
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[source") and stripped.endswith("]"):
            count += 1
    return count


def extract_template_infer_config(template_path: Path) -> Path | None:
    current_section = ""
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            continue
        if current_section == "primary-gie" and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == "config-file":
                return alias_mapped_path(value.strip())
    return None


def read_infer_batch_size(path: Path) -> int | None:
    if not path.exists():
        return None
    current_section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            continue
        if current_section == "property" and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == "batch-size":
                try:
                    return int(value.strip())
                except ValueError:
                    return None
    return None


def validate_native_config_stack(template_path: Path, *, expected_source_count: int) -> dict[str, Any]:
    source_sections = count_source_sections(template_path)
    if source_sections != expected_source_count:
        raise SystemExit(
            f"DeepStream template {template_path} must define exactly {expected_source_count} [sourceN] sections for native execution; found {source_sections}"
        )
    infer_config_path = extract_template_infer_config(template_path)
    infer_batch_size = None
    if infer_config_path is not None:
        if not infer_config_path.exists():
            raise SystemExit(f"Infer config referenced by {template_path} does not exist: {infer_config_path}")
        infer_batch_size = read_infer_batch_size(infer_config_path)
        if infer_batch_size is not None and infer_batch_size != expected_source_count:
            raise SystemExit(
                f"Infer config batch-size must be {expected_source_count} for native execution; got {infer_batch_size} in {infer_config_path}"
            )
    return {
        "template_path": str(template_path),
        "source_sections": source_sections,
        "infer_config_path": str(infer_config_path) if infer_config_path is not None else "",
        "infer_batch_size": infer_batch_size,
    }


def resolve_target_input_read_fps(
    requested_fps: float,
    sources: list[PlannedSource],
    *,
    logger: logging.Logger,
) -> float | None:
    requested = float(requested_fps)
    if requested < 0.0:
        raise SystemExit(f"--input_read_fps must be >= 0, got {requested}")
    if not sources:
        return 0.0
    nominal_fps_values = [float(source.nominal_fps) for source in sources if source.nominal_fps is not None]
    if requested <= 0.0:
        if not nominal_fps_values:
            logger.warning("Skipping shared nominal FPS validation for --input_read_fps 0 because no source metadata was available")
            return None
        baseline = nominal_fps_values[0]
        if any(abs(value - baseline) > FPS_TOLERANCE for value in nominal_fps_values[1:]):
            raise SystemExit(
                "--input_read_fps 0 requires all selected sources to share one nominal FPS within tolerance; "
                f"got {', '.join(f'{value:.3f}' for value in nominal_fps_values)}"
            )
        if len(nominal_fps_values) != len(sources):
            logger.warning(
                "Proceeding with degraded target FPS resolution for --input_read_fps 0 because some source metadata was unavailable"
            )
            return None
        return float(baseline)
    missing_nominal = [source.source_key for source in sources if source.nominal_fps is None]
    if missing_nominal:
        logger.warning(
            "Skipping nominal FPS ceiling validation for sources with unavailable metadata: %s",
            ", ".join(missing_nominal),
        )
    for source in sources:
        if source.nominal_fps is None:
            continue
        if requested > float(source.nominal_fps) + FPS_TOLERANCE:
            raise SystemExit(
                f"--input_read_fps={requested} exceeds nominal FPS={source.nominal_fps:.3f} for source {source.source_key}"
            )
    return requested


def build_execution_plan_payload(
    plan: NativeExecutionPlan,
    *,
    run_ts: str,
    requested_input_read_fps: float,
    resolved_target_input_read_fps: float | None,
    config_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_ts": run_ts,
        "execution_mode": "native_single_run",
        "source_count": plan.source_count,
        "tiler_rows": plan.tiler_rows,
        "tiler_columns": plan.tiler_columns,
        "requested_input_read_fps": requested_input_read_fps,
        "resolved_target_input_read_fps": resolved_target_input_read_fps,
        "ds_config_template": str(plan.ds_config_template),
        "extra_core_args": list(plan.extra_core_args),
        "config_validation": config_validation,
        "sources": [
            {
                "manifest_index": source.manifest_index,
                "source_id": source.source_id,
                "source_key": source.source_key,
                "video_path": str(source.video_path),
                "roi_json": str(source.roi_json),
                "nominal_fps": source.nominal_fps,
                "nominal_frame_count": source.nominal_frame_count,
                "frame_width": source.frame_width,
                "frame_height": source.frame_height,
                "metadata_backend": source.metadata_backend,
                "tags": list(source.tags),
                "notes": source.notes,
                "compatibility_group": source.override.compatibility_group,
            }
            for source in plan.sources
        ],
    }


def build_prepared_request_signature(
    plan: NativeExecutionPlan,
    *,
    requested_input_read_fps: float,
    reacquire_pose_imgsz: int,
) -> dict[str, Any]:
    return {
        "prep_cache_version": PREPARED_CACHE_VERSION,
        "execution_mode": "native_single_run",
        "source_count": int(plan.source_count),
        "requested_input_read_fps": float(requested_input_read_fps),
        "reacquire_pose_imgsz": int(reacquire_pose_imgsz),
        "sources": [
            {
                "manifest_index": int(source.manifest_index),
                "source_id": str(source.source_id),
                "source_key": str(source.source_key),
                "video_path": str(source.video_path),
                "roi_json": str(source.roi_json),
            }
            for source in plan.sources
        ],
    }


def validate_prepared_cache(
    cache_dir: Path,
    *,
    expected_signature: dict[str, Any],
) -> dict[str, Any]:
    prepare_meta_path = cache_dir / "prepare_meta.json"
    prepare_meta = load_json_object(prepare_meta_path, "prepared cache metadata")
    actual_signature = prepare_meta.get("request_signature")
    if actual_signature != expected_signature:
        raise SystemExit(
            f"Prepared cache does not match the current request: {cache_dir}. "
            "Use --rebuild_prepared to rebuild it explicitly."
        )
    required_paths = [
        cache_dir / "manifest_resolved.json",
        cache_dir / "resolved_roi_map.json",
        cache_dir / "source_labels.json",
        cache_dir / "crop_assets",
        cache_dir / "crop_assets" / "crop_assets_manifest.json",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise SystemExit(
            f"Prepared cache is incomplete: {cache_dir}. Missing: {', '.join(missing_paths)}. "
            "Use --rebuild_prepared to rebuild it explicitly."
        )
    return prepare_meta


def write_prepared_cache_metadata(
    *,
    cache_dir: Path,
    cache_key: str,
    crop_assets_dir: Path,
    request_signature: dict[str, Any],
    resolved_manifest_payload: dict[str, Any],
    source_labels_payload: dict[str, str],
    roi_map_payload: dict[str, str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=False)
    write_json(cache_dir / "prepare_meta.json", {
        "cache_key": cache_key,
        "prepared_root": str(cache_dir.parent),
        "prepared_cache_dir": str(cache_dir),
        "created_at": now_iso(),
        "prep_cache_version": PREPARED_CACHE_VERSION,
        "request_signature": request_signature,
        "crop_assets_dir": str(crop_assets_dir),
        "crop_assets_manifest": str(crop_assets_dir / "crop_assets_manifest.json"),
    })
    write_json(cache_dir / "manifest_resolved.json", resolved_manifest_payload)
    write_json(cache_dir / "resolved_roi_map.json", roi_map_payload)
    write_json(cache_dir / "source_labels.json", source_labels_payload)
    target = crop_assets_dir
    link_path = cache_dir / "crop_assets"
    link_path.symlink_to(target)


def write_source_status_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "manifest_index",
        "source_id",
        "source_key",
        "enabled",
        "status",
        "video_path",
        "roi_json",
        "nominal_fps",
        "nominal_frame_count",
        "frame_width",
        "frame_height",
        "metadata_backend",
        "execution_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_subprocess(cmd: list[str], logger: logging.Logger) -> int:
    logger.info("subprocess: %s", shell_join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def collect_processed_frames(core_out_dir: Path) -> int:
    run_summary_path = core_out_dir / "boundary_reacquire_run_summary.json"
    if not run_summary_path.exists():
        raise SystemExit(f"Missing Stage 04.05 run summary: {run_summary_path}")
    run_summary = load_json_object(run_summary_path, "Stage 04.05 run summary")
    per_source = run_summary.get("per_source", [])
    if not isinstance(per_source, list):
        raise SystemExit(f"Invalid per_source section in {run_summary_path}")
    total_frames = 0
    for item in per_source:
        if not isinstance(item, dict):
            continue
        summary_path = project_path(str(item.get("summary_path", "")).strip())
        summary = load_json_object(summary_path, "Stage 04.05 per-source summary")
        total_frames += int(summary.get("frame_count", 0) or 0)
    return total_frames


def collect_core_experiment_metrics(core_out_dir: Path, *, experiment_name: str) -> dict[str, Any]:
    run_summary_path = core_out_dir / "boundary_reacquire_run_summary.json"
    run_summary = load_json_object(run_summary_path, "Stage 04.05 run summary")
    phase_timings = run_summary.get("phase_timings_sec", {})
    render_summary = run_summary.get("render_summary", {})
    decision_aggregate = run_summary.get("decision_aggregate", {})
    if not isinstance(phase_timings, dict):
        phase_timings = {}
    if not isinstance(render_summary, dict):
        render_summary = {}
    if not isinstance(decision_aggregate, dict):
        decision_aggregate = {}
    budget_by_state = render_summary.get("frame_budget_exceeded_by_state", {})
    if not isinstance(budget_by_state, dict):
        budget_by_state = {}

    guardrail_warnings: list[str] = []
    if int(decision_aggregate.get("lazy_decode_miss_count", 0) or 0) > 0:
        guardrail_warnings.append("lazy_decode_miss_count>0")
    if str(experiment_name).strip().lower() == "b3_light" and int(budget_by_state.get("IN_CONFIRMED", 0) or 0) > 0:
        guardrail_warnings.append("frame_budget_exceeded_by_state.IN_CONFIRMED>0")

    return {
        "experiment": str(experiment_name),
        "stage_total": float(phase_timings.get("stage_total", 0.0) or 0.0),
        "decision_pass": float(phase_timings.get("decision_pass", 0.0) or 0.0),
        "render_boundary_reacquire": float(phase_timings.get("render_boundary_reacquire", 0.0) or 0.0),
        "klt_sidecar_augmentation": float(phase_timings.get("klt_sidecar_augmentation", 0.0) or 0.0),
        "confirmed_events_total": int(run_summary.get("confirmed_events_total", 0) or 0),
        "render_h6_probe_sec": float(render_summary.get("render_h6_probe_sec", 0.0) or 0.0),
        "render_h7_reacquire_sec": float(render_summary.get("render_h7_reacquire_sec", 0.0) or 0.0),
        "render_h6_call_count": int(render_summary.get("render_h6_call_count", 0) or 0),
        "render_h7_call_count": int(render_summary.get("render_h7_call_count", 0) or 0),
        "pose_probe_attempt_count": int(decision_aggregate.get("pose_probe_attempt_count", 0) or 0),
        "pose_probe_sec": float(decision_aggregate.get("pose_probe_sec", 0.0) or 0.0),
        "pose_probe_reuse_count": int(decision_aggregate.get("pose_probe_reuse_count", 0) or 0),
        "pose_probe_forced_fresh_count": int(decision_aggregate.get("pose_probe_forced_fresh_count", 0) or 0),
        "lazy_decode_skip_count": int(decision_aggregate.get("lazy_decode_skip_count", 0) or 0),
        "lazy_decode_miss_count": int(decision_aggregate.get("lazy_decode_miss_count", 0) or 0),
        "frame_budget_exceeded_count": int(render_summary.get("frame_budget_exceeded_count", 0) or 0),
        "frame_budget_exceeded_by_state": budget_by_state,
        "max_frame_render_sec": float(render_summary.get("max_frame_render_sec", 0.0) or 0.0),
        "p95_frame_render_sec": float(render_summary.get("p95_frame_render_sec", 0.0) or 0.0),
        "mean_frame_render_sec": float(render_summary.get("mean_frame_render_sec", 0.0) or 0.0),
        "slow_frame_count": int(render_summary.get("slow_frame_count", 0) or 0),
        "guardrail_warnings": guardrail_warnings,
    }


def compute_throughput_metrics(
    *,
    num_streams: int,
    core_wall_clock_sec: float | None,
    target_input_read_fps: float | None,
    total_processed_frames: int | None,
) -> dict[str, Any]:
    target_total_fps = None if target_input_read_fps is None else float(num_streams) * float(target_input_read_fps)
    if total_processed_frames is None or core_wall_clock_sec is None or core_wall_clock_sec <= 0.0 or num_streams <= 0:
        return {
            "num_streams": int(num_streams),
            "target_input_read_fps": None if target_input_read_fps is None else float(target_input_read_fps),
            "target_total_fps": target_total_fps,
            "total_processed_frames": None,
            "aggregate_effective_fps": None,
            "per_stream_effective_fps": None,
            "realtime_ratio": None,
        }
    aggregate_effective_fps = float(total_processed_frames) / float(core_wall_clock_sec)
    return {
        "num_streams": int(num_streams),
        "target_input_read_fps": None if target_input_read_fps is None else float(target_input_read_fps),
        "target_total_fps": target_total_fps,
        "total_processed_frames": int(total_processed_frames),
        "aggregate_effective_fps": float(aggregate_effective_fps),
        "per_stream_effective_fps": float(aggregate_effective_fps / float(num_streams)),
        "realtime_ratio": float(aggregate_effective_fps / target_total_fps)
        if target_total_fps is not None and target_total_fps > 0.0
        else None,
    }


def main() -> None:
    args = parse_args()
    if args.validate_only and args.dry_run:
        raise SystemExit("Use either --validate_only or --dry_run, not both.")
    prepared_cache_key = str(args.prepared_cache_key).strip()
    if args.reuse_prepared and args.rebuild_prepared:
        raise SystemExit("Use either --reuse_prepared or --rebuild_prepared, not both.")
    if (args.reuse_prepared or args.rebuild_prepared) and not prepared_cache_key:
        raise SystemExit("--reuse_prepared/--rebuild_prepared require --prepared_cache_key.")
    if args.dry_run and args.rebuild_prepared:
        raise SystemExit("--rebuild_prepared requires a real prep run; it cannot be combined with --dry_run.")
    if int(args.only_batch) >= 0 or str(args.skip_batches).strip():
        raise SystemExit("--only_batch/--skip_batches are not supported in the native single-run execution path.")
    if not getattr(args, "out_base", ""):
        args.out_base = DEFAULT_OUT_BASE

    manifest_path = resolve_manifest_path(args)
    source_override_path = project_path(args.source_overrides)
    default_ds_template = project_path(args.ds_config_template)
    if not default_ds_template.exists():
        raise SystemExit(f"Missing default DeepStream config template: {default_ds_template}")
    if not PREP_SCRIPT.exists():
        raise SystemExit(f"Missing Stage 04.05a script: {PREP_SCRIPT}")
    if not CORE_SCRIPT.exists():
        raise SystemExit(f"Missing Stage 04.05 core script: {CORE_SCRIPT}")

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)
    runner_dir = run.out_dir
    runner_dir.mkdir(parents=True, exist_ok=True)
    shared_out_root = project_path(args.out_root)
    shared_log_root = project_path(args.log_root)
    shared_run_root = shared_out_root / STAGE / run.run_ts

    orchestrator_start_iso = now_iso()
    orchestrator_started = time.perf_counter()

    manifest = load_json_object(manifest_path, "manifest")
    default_override, overrides_by_source = load_overrides(source_override_path, default_ds_template)
    planned_sources, source_status_rows, source_count_target = parse_manifest_sources(
        manifest,
        default_override=default_override,
        overrides_by_source=overrides_by_source,
        forced_ds_config_template=default_ds_template,
        logger=logger,
    )
    ds_config_template, extra_core_args = validate_native_run_signature(planned_sources)
    experiment = resolve_experiment_spec(args)
    effective_core_args = tuple(extra_core_args) + tuple(experiment.core_args)
    config_validation = validate_native_config_stack(ds_config_template, expected_source_count=source_count_target)

    requested_input_read_fps = float(args.input_read_fps)
    resolved_target_input_read_fps = resolve_target_input_read_fps(
        requested_input_read_fps,
        planned_sources,
        logger=logger,
    )
    tiler_rows, tiler_columns = resolve_native_tiler_layout(
        len(planned_sources),
        requested_rows=int(args.tiler_rows),
        requested_columns=int(args.tiler_columns),
    )

    plan = NativeExecutionPlan(
        source_count=len(planned_sources),
        tiler_rows=tiler_rows,
        tiler_columns=tiler_columns,
        ds_config_template=ds_config_template,
        extra_core_args=effective_core_args,
        sources=tuple(planned_sources),
    )

    source_labels_path = runner_dir / "source_labels.json"
    roi_map_path = runner_dir / "resolved_roi_map.json"
    execution_plan_path = runner_dir / "execution_plan.json"
    execution_status_path = runner_dir / "execution_status.json"

    source_labels_payload = {str(idx): source.source_key for idx, source in enumerate(plan.sources)}
    roi_map_payload = {str(idx): str(source.roi_json) for idx, source in enumerate(plan.sources)}
    execution_plan_payload = build_execution_plan_payload(
        plan,
        run_ts=run.run_ts,
        requested_input_read_fps=requested_input_read_fps,
        resolved_target_input_read_fps=resolved_target_input_read_fps,
        config_validation=config_validation,
    )
    execution_plan_payload["experiment"] = {
        "name": experiment.name,
        "description": experiment.description,
        "core_args": list(experiment.core_args),
        "effective_extra_core_args": list(plan.extra_core_args),
    }
    write_json(source_labels_path, source_labels_payload)
    write_json(roi_map_path, roi_map_payload)
    write_json(execution_plan_path, execution_plan_payload)

    for source in plan.sources:
        source_status_rows.append(
            {
                "manifest_index": source.manifest_index,
                "source_id": source.source_id,
                "source_key": source.source_key,
                "enabled": True,
                "status": "planned",
                "video_path": str(source.video_path),
                "roi_json": str(source.roi_json),
                "nominal_fps": source.nominal_fps,
                "nominal_frame_count": source.nominal_frame_count,
                "frame_width": source.frame_width,
                "frame_height": source.frame_height,
                "metadata_backend": source.metadata_backend or "",
                "execution_mode": "native_single_run",
            }
        )

    prep_out_base = f"{args.out_base}/prepare"
    core_out_base = f"{args.out_base}/core"
    prepared_root = project_path(args.prepared_root)
    prepared_cache_dir = prepared_root / prepared_cache_key if prepared_cache_key else None
    prepared_stage_root = prepared_root / STAGE / prepared_cache_key if prepared_cache_key else None
    prepared_core_dir = prepared_stage_root / PREPARED_CACHE_CORE_OUT_BASE if prepared_stage_root is not None else None
    prepared_crop_assets_dir = prepared_core_dir / "crop_assets" if prepared_core_dir is not None else None
    prepared_crop_manifest_path = prepared_crop_assets_dir / "crop_assets_manifest.json" if prepared_crop_assets_dir is not None else None
    prepared_request_signature = build_prepared_request_signature(
        plan,
        requested_input_read_fps=requested_input_read_fps,
        reacquire_pose_imgsz=int(args.reacquire_pose_imgsz),
    )
    use_prepared_cache = False
    if prepared_cache_dir is not None and prepared_cache_dir.exists():
        if args.rebuild_prepared:
            shutil.rmtree(prepared_cache_dir)
            if prepared_stage_root is not None and prepared_stage_root.exists():
                shutil.rmtree(prepared_stage_root)
        elif args.reuse_prepared:
            validate_prepared_cache(prepared_cache_dir, expected_signature=prepared_request_signature)
            use_prepared_cache = True
        else:
            raise SystemExit(
                f"Prepared cache already exists: {prepared_cache_dir}. "
                "Use --reuse_prepared to reuse it or --rebuild_prepared to rebuild it."
            )
    elif args.reuse_prepared:
        raise SystemExit(
            f"Prepared cache does not exist: {prepared_cache_dir}. "
            "Run once without --reuse_prepared or with --rebuild_prepared to create it."
        )
    elif prepared_stage_root is not None and prepared_stage_root.exists():
        if args.rebuild_prepared:
            shutil.rmtree(prepared_stage_root)
        else:
            raise SystemExit(
                f"Prepared stage directory already exists without a completed cache entry: {prepared_stage_root}. "
                "Use --rebuild_prepared to rebuild it explicitly."
            )
    execution_status: dict[str, Any] = {
        "run_ts": run.run_ts,
        "execution_mode": "native_single_run",
        "shared_run_root": str(shared_run_root),
        "source_count": plan.source_count,
        "tiler_rows": plan.tiler_rows,
        "tiler_columns": plan.tiler_columns,
        "requested_input_read_fps": requested_input_read_fps,
        "target_input_read_fps": None if resolved_target_input_read_fps is None else float(resolved_target_input_read_fps),
        "target_total_fps": None
        if resolved_target_input_read_fps is None
        else float(plan.source_count * resolved_target_input_read_fps),
        "started_at": now_iso(),
        "ended_at": None,
        "total_wall_clock_sec": None,
        "prepare_started_at": None,
        "prepare_ended_at": None,
        "prepare_wall_clock_sec": None,
        "core_started_at": None,
        "core_ended_at": None,
        "core_wall_clock_sec": None,
        "exit_code": None,
        "dry_run": bool(args.dry_run),
        "validate_only": bool(args.validate_only),
        "source_labels_path": str(source_labels_path),
        "resolved_roi_map_path": str(roi_map_path),
        "execution_plan_path": str(execution_plan_path),
        "prepare_out_base": prep_out_base,
        "core_out_base": core_out_base,
        "prepared_root": str(prepared_root),
        "prepared_cache_key": prepared_cache_key,
        "prepared_cache_dir": "" if prepared_cache_dir is None else str(prepared_cache_dir),
        "prepared_cache_reused": bool(use_prepared_cache),
        "prepared_cache_built": False,
        "experiment": {
            "name": experiment.name,
            "description": experiment.description,
            "core_args": list(experiment.core_args),
        },
        "sources": [source.source_key for source in plan.sources],
        "config_validation": config_validation,
        "total_processed_frames": None,
        "aggregate_effective_fps": None,
        "per_stream_effective_fps": None,
        "realtime_ratio": None,
        "core_experiment_metrics": {},
    }

    logger.info(
        "native execution plan source_count=%s tiler=%sx%s ds_config_template=%s",
        plan.source_count,
        plan.tiler_rows,
        plan.tiler_columns,
        plan.ds_config_template,
    )

    if args.validate_only:
        execution_status["exit_code"] = 0
        execution_status["ended_at"] = now_iso()
        execution_status["total_wall_clock_sec"] = float(time.perf_counter() - orchestrator_started)
        write_json(execution_status_path, execution_status)
        throughput_metrics = compute_throughput_metrics(
            num_streams=plan.source_count,
            core_wall_clock_sec=None,
            target_input_read_fps=resolved_target_input_read_fps,
            total_processed_frames=None,
        )
    else:
        crop_manifest_path: Path | None = None
        prep_exit_code = 0
        if use_prepared_cache:
            assert prepared_cache_dir is not None
            crop_manifest_path = prepared_cache_dir / "crop_assets" / "crop_assets_manifest.json"
            execution_status["prepare_started_at"] = now_iso()
            execution_status["prepare_ended_at"] = execution_status["prepare_started_at"]
            execution_status["prepare_wall_clock_sec"] = 0.0
            execution_status["prepared_cache_reused"] = True
            logger.info("reusing prepared-input cache cache_key=%s cache_dir=%s", prepared_cache_key, prepared_cache_dir)
        else:
            prep_run_ts = prepared_cache_key if prepared_cache_key else run.run_ts
            prep_out_root = prepared_root if prepared_cache_key else shared_out_root
            prep_stage_out_base = PREPARED_CACHE_PREP_OUT_BASE if prepared_cache_key else prep_out_base
            prep_main_out_base = PREPARED_CACHE_CORE_OUT_BASE if prepared_cache_key else core_out_base
            prep_cmd = [
                sys.executable,
                str(PREP_SCRIPT),
                "--run_ts",
                prep_run_ts,
                "--out_dir",
                str(prep_out_root),
                "--log_root",
                str(shared_log_root),
                "--out_base",
                prep_stage_out_base,
                "--main_out_base",
                prep_main_out_base,
                "--source_labels",
                str(source_labels_path),
                "--roi_map",
                str(roi_map_path),
                "--input_read_fps",
                str(requested_input_read_fps),
                "--inputs",
                *(str(source.video_path) for source in plan.sources),
            ]
            if args.dry_run:
                prep_cmd.append("--dry_run")

            execution_status["prepare_started_at"] = now_iso()
            prep_started = time.perf_counter()
            prep_exit_code = run_subprocess(prep_cmd, logger)
            execution_status["prepare_wall_clock_sec"] = float(time.perf_counter() - prep_started)
            execution_status["prepare_ended_at"] = now_iso()
            if prep_exit_code == 0:
                crop_manifest_path = (
                    prepared_crop_manifest_path
                    if prepared_crop_manifest_path is not None and prepared_cache_key
                    else shared_run_root / core_out_base / "crop_assets" / "crop_assets_manifest.json"
                )
                if prepared_cache_dir is not None and not args.dry_run:
                    assert prepared_crop_assets_dir is not None
                    if prepared_crop_manifest_path is None or not prepared_crop_manifest_path.exists():
                        raise SystemExit(f"Prepared cache crop manifest was not created: {prepared_crop_manifest_path}")
                    write_prepared_cache_metadata(
                        cache_dir=prepared_cache_dir,
                        cache_key=prepared_cache_key,
                        crop_assets_dir=prepared_crop_assets_dir,
                        request_signature=prepared_request_signature,
                        resolved_manifest_payload=execution_plan_payload,
                        source_labels_payload=source_labels_payload,
                        roi_map_payload=roi_map_payload,
                    )
                    execution_status["prepared_cache_built"] = True
                    logger.info("prepared-input cache built cache_key=%s cache_dir=%s", prepared_cache_key, prepared_cache_dir)

        if prep_exit_code != 0:
            execution_status["exit_code"] = prep_exit_code
            execution_status["ended_at"] = now_iso()
            execution_status["total_wall_clock_sec"] = float(time.perf_counter() - orchestrator_started)
            write_json(execution_status_path, execution_status)
            throughput_metrics = compute_throughput_metrics(
                num_streams=plan.source_count,
                core_wall_clock_sec=None,
                target_input_read_fps=resolved_target_input_read_fps,
                total_processed_frames=None,
            )
        else:
            assert crop_manifest_path is not None
            crop_manifest = load_json_object(crop_manifest_path, "crop asset manifest")
            source_entries = crop_manifest.get("sources", [])
            if not isinstance(source_entries, list) or len(source_entries) != plan.source_count:
                raise SystemExit(
                    f"Invalid crop asset manifest source count in {crop_manifest_path}; expected {plan.source_count}, got {len(source_entries) if isinstance(source_entries, list) else 'non-list'}"
                )
            core_inputs = [str(project_path(str(item.get("source_clip_path", "")).strip())) for item in source_entries]

            core_cmd = [
                sys.executable,
                str(CORE_SCRIPT),
                "--run_ts",
                run.run_ts,
                "--out_dir",
                str(shared_out_root),
                "--log_root",
                str(shared_log_root),
                "--out_base",
                core_out_base,
                "--ds_config_template",
                str(plan.ds_config_template),
                "--crop_manifest",
                str(crop_manifest_path),
                "--source_labels",
                str(source_labels_path),
                "--roi_map",
                str(roi_map_path),
                "--input_read_fps",
                str(requested_input_read_fps),
                "--reacquire_pose_imgsz",
                str(int(args.reacquire_pose_imgsz)),
                "--tiler_rows",
                str(plan.tiler_rows),
                "--tiler_columns",
                str(plan.tiler_columns),
                "--inputs",
                *core_inputs,
                *plan.extra_core_args,
            ]
            if args.dry_run:
                core_cmd.append("--dry_run")

            execution_status["core_started_at"] = now_iso()
            core_started = time.perf_counter()
            core_exit_code = run_subprocess(core_cmd, logger)
            execution_status["core_wall_clock_sec"] = float(time.perf_counter() - core_started)
            execution_status["core_ended_at"] = now_iso()
            execution_status["exit_code"] = core_exit_code
            execution_status["ended_at"] = now_iso()
            execution_status["total_wall_clock_sec"] = float(time.perf_counter() - orchestrator_started)

            total_processed_frames = None
            if core_exit_code == 0 and not args.dry_run:
                core_out_dir = shared_run_root / core_out_base
                total_processed_frames = collect_processed_frames(core_out_dir)
                execution_status["core_experiment_metrics"] = collect_core_experiment_metrics(
                    core_out_dir,
                    experiment_name=experiment.name,
                )

            throughput_metrics = compute_throughput_metrics(
                num_streams=plan.source_count,
                core_wall_clock_sec=execution_status["core_wall_clock_sec"],
                target_input_read_fps=resolved_target_input_read_fps,
                total_processed_frames=total_processed_frames,
            )
            execution_status.update(throughput_metrics)
            write_json(execution_status_path, execution_status)

    status_value = "validated" if args.validate_only else ("dry_run_ok" if args.dry_run else "ok")
    if int(execution_status.get("exit_code", 1) or 0) != 0:
        status_value = "validation_error" if args.validate_only else ("dry_run_error" if args.dry_run else "error")
    for row in source_status_rows:
        if bool(row.get("enabled", False)):
            row["status"] = status_value

    orchestrator_ended_iso = now_iso()
    orchestrator_meta = {
        "stage": STAGE,
        "stage_step": "04.06",
        "execution_mode": "native_single_run",
        "manifest_path": str(manifest_path),
        "manifest_name": str(manifest.get("manifest_name", "")),
        "runner_profile": str(manifest.get("runner_profile", "")),
        "source_overrides_path": str(source_override_path),
        "run_ts": run.run_ts,
        "dry_run": bool(args.dry_run),
        "validate_only": bool(args.validate_only),
        "requested_input_read_fps": requested_input_read_fps,
        "shared_run_root": str(shared_run_root),
        "started_at": orchestrator_start_iso,
        "ended_at": orchestrator_ended_iso,
        "total_wall_clock_sec": execution_status.get("total_wall_clock_sec"),
        "prepare_wall_clock_sec": execution_status.get("prepare_wall_clock_sec"),
        "core_wall_clock_sec": execution_status.get("core_wall_clock_sec"),
        "source_count": plan.source_count,
        "tiler_rows": plan.tiler_rows,
        "tiler_columns": plan.tiler_columns,
        "ds_config_template": str(plan.ds_config_template),
        "experiment": execution_status.get("experiment", {}),
        "execution_status_path": str(execution_status_path),
        "config_validation": config_validation,
        "core_experiment_metrics": execution_status.get("core_experiment_metrics", {}),
        **throughput_metrics,
    }
    write_json(runner_dir / "orchestrator_meta.json", orchestrator_meta)
    write_source_status_csv(runner_dir / "source_status.csv", source_status_rows)
    if run.outputs_enabled:
        write_json(runner_dir / "run_meta.json", orchestrator_meta)

    if int(execution_status.get("exit_code", 1) or 0) != 0:
        raise SystemExit(int(execution_status["exit_code"]))


if __name__ == "__main__":
    main()
