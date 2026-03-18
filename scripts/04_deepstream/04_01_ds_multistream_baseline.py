#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
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
REPO_ALIAS = Path("/workspace/AID")
SOURCE_COUNT = 4
TILER_ROWS = 2
TILER_COLUMNS = 2
DEFAULT_OUT_BASE = "multistream4_baseline"
DEFAULT_INPUTS = [
    "/workspace/AID/data/clips/E01_001/ev00_f1826-2854_50s.mp4",
    "/workspace/AID/data/clips/E01_004/ev00_f1803-1868_50s.mp4",
    "/workspace/AID/data/clips/E01_008/ev00_f1899-2191_50s.mp4",
    "/workspace/AID/data/clips/E01_011/ev00_f2031-2293_50s.mp4",
]


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
    refs: dict[str, str] = {}
    rendered_lines: list[str] = []

    with template_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1].strip()
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

            rendered_lines.append(line)

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


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")


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


def stream_process_output(cmd: list[str], logger: logging.Logger) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[deepstream-app] %s", line.rstrip())

    return proc.wait()


def main() -> None:
    args = parse_args()
    normalize_output_args(args)

    if args.no_outputs:
        raise SystemExit("--no_outputs is not supported for this multistream baseline wrapper.")

    source_specs = build_source_specs(args.inputs)
    template_path = project_path(args.ds_config_template)
    validate_file_exists(template_path, "DeepStream config template")

    if not getattr(args, "out_base", ""):
        args.out_base = DEFAULT_OUT_BASE

    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")

    preflight_rendered_text, refs = render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=Path("/tmp/ds_multistream4_baseline_preflight_nvdcf.mp4"),
    )

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

    output_video = run.out_dir / f"{args.out_base}_tiled_nvdcf.mp4"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    rendered_text, _ = render_app_config(
        template_path=template_path,
        source_specs=source_specs,
        output_file=output_video,
    )

    save_rendered_config(rendered_config_path, rendered_text)

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

    if run.outputs_enabled:
        dump_run_meta(
            run.out_dir,
            {
                "stage": STAGE,
                "source_count": SOURCE_COUNT,
                "batch_size": SOURCE_COUNT,
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
                "output_video": str(output_video),
                "ds_config_template": str(template_path),
                "rendered_config_path": str(rendered_config_path),
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

    exit_code = stream_process_output(cmd, logger)
    if exit_code != 0:
        logger.error("deepstream-app exited with code %s", exit_code)
        raise SystemExit(exit_code)

    logger.info("deepstream-app completed successfully")


if __name__ == "__main__":
    main()
