#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.run_utils import common_argparser, dump_run_meta, init_run


STAGE = "03_deepstream"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "deepstream" / "ds_yolo11_tracker_nvdcf.txt"
REPO_ALIAS = Path("/workspace/AID")


class BaselineConfigError(RuntimeError):
    pass


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


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_root=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--input_video", required=True, help="Input video path")
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
    parser.add_argument("--dry_run", action="store_true", help="Render config and print command without running deepstream-app")
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


def render_app_config(template_path: Path, input_uri: str, output_file: Path) -> tuple[str, dict[str, str]]:
    current_section = ""
    source_uri_replaced = 0
    sink_output_replaced = 0
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

            if current_section == "source0" and key == "uri":
                rendered_lines.append(f"{prefix}={input_uri}{newline}")
                source_uri_replaced += 1
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

    if source_uri_replaced != 1:
        raise BaselineConfigError(
            f"Expected exactly one [source0].uri in template, found {source_uri_replaced}: {template_path}"
        )
    if sink_output_replaced != 1:
        raise BaselineConfigError(
            f"Expected exactly one [sink0].output-file in template, found {sink_output_replaced}: {template_path}"
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
        raise SystemExit("--no_outputs is not supported for this baseline wrapper.")

    input_video = project_path(args.input_video)
    template_path = project_path(args.ds_config_template)

    validate_file_exists(input_video, "input video")
    validate_file_exists(template_path, "DeepStream config template")

    if not getattr(args, "out_base", ""):
        args.out_base = input_video.stem

    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")

    preflight_rendered_text, refs = render_app_config(
        template_path=template_path,
        input_uri=input_video.as_uri(),
        output_file=Path("/tmp/ds_baseline_preflight_nvdcf.mp4"),
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

    output_video = run.out_dir / f"{args.out_base}_nvdcf.mp4"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    rendered_text, _ = render_app_config(
        template_path=template_path,
        input_uri=input_video.as_uri(),
        output_file=output_video,
    )

    save_rendered_config(rendered_config_path, rendered_text)

    cmd = [deepstream_app or "deepstream-app", "-c", str(rendered_config_path)]
    cmd_str = shell_join(cmd)

    if run.outputs_enabled:
        dump_run_meta(
            run.out_dir,
            {
                "stage": STAGE,
                "input_video": str(input_video),
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

    logger.info("input_video=%s", input_video)
    logger.info("ds_config_template=%s", template_path)
    logger.info("infer_config_source_of_truth=%s", infer_config_local)
    logger.info("tracker_config_source_of_truth=%s", tracker_config_local)
    logger.info("rendered_config_path=%s", rendered_config_path)
    logger.info("output_video=%s", output_video)
    logger.info("deepstream_cmd=%s", cmd_str)
    logger.info("run_out_dir=%s", run.out_dir)
    logger.info("run_log_path=%s", run.log_path)
    logger.info("run_cmd_path=%s", run.cmd_path)

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
