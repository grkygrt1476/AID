#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
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
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "deepstream" / "ds_yolo11_tracker_nvdcf_poseanchorassist.txt"
DEFAULT_PLUGIN_LIB = PROJECT_ROOT / "scripts" / "03_ds_single_stream" / "gst-dsposeanchorassist" / "libnvdsgst_dsposeanchorassist.so"
REPO_ALIAS = Path("/workspace/AID")
DEFAULT_POSE_MIN_KEYPOINT_CONF = 0.20


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


def parse_args():
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
    parser.add_argument(
        "--hard_max_proxy_age_frames",
        type=int,
        default=18,
        help="Hard safety cap for proxy age; not the normal survival rule.",
    )
    parser.add_argument("--min_good_points", type=int, default=4, help="Minimum LK points required to keep the proxy alive.")
    parser.add_argument("--feature_max_corners", type=int, default=64, help="Max corners seeded inside the pose patch.")
    parser.add_argument("--lk_win_size", type=int, default=21, help="LK window size in pixels.")
    parser.add_argument("--patch_width_ratio", type=float, default=0.55, help="Base patch width as a ratio of the real bbox width.")
    parser.add_argument("--patch_height_ratio", type=float, default=0.48, help="Base patch height as a ratio of the real bbox height.")
    parser.add_argument(
        "--patch_y_offset_ratio",
        type=float,
        default=0.08,
        help="Vertical patch-center offset applied below the pose anchor.",
    )
    parser.add_argument(
        "--max_center_shift_px",
        type=float,
        default=32.0,
        help="Max allowed per-frame proxy center jump in original-frame pixels.",
    )
    parser.set_defaults(freeze_on_patch_fail=True)
    parser.add_argument(
        "--freeze_on_patch_fail",
        dest="freeze_on_patch_fail",
        action="store_true",
        help="Allow a bounded frozen-hold state after technical patch failure.",
    )
    parser.add_argument(
        "--no_freeze_on_patch_fail",
        dest="freeze_on_patch_fail",
        action="store_false",
        help="Disable frozen-hold after patch failure.",
    )
    parser.add_argument(
        "--hold_after_fail_frames",
        type=int,
        default=3,
        help="Number of frozen-hold frames after technical patch failure.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Render config and print command without running deepstream-app")
    return parser.parse_args()


def cli_option_present(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def normalize_output_args(args) -> None:
    out_dir_explicit = cli_option_present("--out_dir")
    out_root_explicit = cli_option_present("--out_root")

    if out_dir_explicit and out_root_explicit:
        raise SystemExit("Use only one of --out_dir or --out_root.")

    if out_dir_explicit:
        args.out_root = args.out_dir

    args.out_root = str(project_path(args.out_root))
    args.log_root = str(project_path(args.log_root))


def render_app_config(template_path: Path, input_uri: str, output_file: Path):
    current_section = ""
    source_uri_replaced = 0
    sink_output_replaced = 0
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

            if current_section == "secondary-gie0" and key == "config-file":
                refs["pose_sgie_config"] = raw_value

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
    if "pose_sgie_config" not in refs:
        raise BaselineConfigError(f"Missing [secondary-gie0].config-file in template: {template_path}")
    if not has_ds_example:
        raise BaselineConfigError(f"Missing [ds-example] section in assist template: {template_path}")

    return "".join(rendered_lines), refs


def parse_property_section(path: Path) -> dict[str, str]:
    current_section = ""
    props: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1].strip().lower()
                continue
            if current_section != "property" or "=" not in line:
                continue
            prefix, value = line.split("=", 1)
            props[prefix.strip()] = value.strip()
    return props


def extract_pose_runtime_requirements(pose_config_path: Path) -> dict[str, object]:
    props = parse_property_section(pose_config_path)

    required_keys = ["onnx-file", "custom-lib-path", "gie-unique-id", "output-tensor-meta", "process-mode"]
    missing = [key for key in required_keys if key not in props]
    if missing:
        raise SystemExit(f"Pose SGIE config is missing required keys {missing}: {pose_config_path}")

    try:
        pose_sgie_uid = int(props["gie-unique-id"])
    except ValueError as exc:
        raise SystemExit(f"Invalid pose SGIE gie-unique-id in {pose_config_path}: {props['gie-unique-id']}") from exc

    if str(props["output-tensor-meta"]).strip() != "1":
        raise SystemExit(
            f"Pose SGIE config must enable output-tensor-meta=1 for tensor-driven pose anchor parsing: {pose_config_path}"
        )
    if str(props["process-mode"]).strip() != "2":
        raise SystemExit(f"Pose SGIE config must use process-mode=2: {pose_config_path}")

    return {
        "pose_sgie_uid": pose_sgie_uid,
        "onnx_path": alias_mapped_path(props["onnx-file"]),
        "custom_lib_path": alias_mapped_path(props["custom-lib-path"]),
        "engine_path": alias_mapped_path(props["model-engine-file"]) if "model-engine-file" in props else None,
        "raw_props": props,
    }


def validate_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")


def validate_repo_alias_runtime(*texts: str, dry_run: bool) -> None:
    if dry_run or REPO_ALIAS.exists():
        return

    if any(str(REPO_ALIAS) in text for text in texts):
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


def build_runtime_env(base_env: dict[str, str], plugin_dir: Path, sidecar_path: Path, args, pose_sgie_uid: int) -> dict[str, str]:
    env = dict(base_env)
    env["GST_PLUGIN_PATH"] = prepend_path_env(env.get("GST_PLUGIN_PATH"), plugin_dir)
    env["AID_DSPOSEANCHORASSIST_HARD_MAX_PROXY_AGE_FRAMES"] = str(args.hard_max_proxy_age_frames)
    env["AID_DSPOSEANCHORASSIST_MIN_GOOD_POINTS"] = str(args.min_good_points)
    env["AID_DSPOSEANCHORASSIST_FEATURE_MAX_CORNERS"] = str(args.feature_max_corners)
    env["AID_DSPOSEANCHORASSIST_LK_WIN_SIZE"] = str(args.lk_win_size)
    env["AID_DSPOSEANCHORASSIST_PATCH_WIDTH_RATIO"] = str(args.patch_width_ratio)
    env["AID_DSPOSEANCHORASSIST_PATCH_HEIGHT_RATIO"] = str(args.patch_height_ratio)
    env["AID_DSPOSEANCHORASSIST_PATCH_Y_OFFSET_RATIO"] = str(args.patch_y_offset_ratio)
    env["AID_DSPOSEANCHORASSIST_MAX_CENTER_SHIFT_PX"] = str(args.max_center_shift_px)
    env["AID_DSPOSEANCHORASSIST_FREEZE_ON_PATCH_FAIL"] = "1" if args.freeze_on_patch_fail else "0"
    env["AID_DSPOSEANCHORASSIST_HOLD_AFTER_FAIL_FRAMES"] = str(args.hold_after_fail_frames)
    env["AID_DSPOSEANCHORASSIST_POSE_SGIE_UID"] = str(pose_sgie_uid)
    env["AID_DSPOSEANCHORASSIST_POSE_MIN_KEYPOINT_CONF"] = str(DEFAULT_POSE_MIN_KEYPOINT_CONF)
    env["AID_DSPOSEANCHORASSIST_SIDECAR_PATH"] = str(sidecar_path)
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
        raise SystemExit("--no_outputs is not supported for this pose-anchor assist wrapper.")

    input_video = project_path(args.input_video)
    template_path = project_path(args.ds_config_template)
    plugin_lib = project_path(DEFAULT_PLUGIN_LIB)

    validate_file_exists(input_video, "input video")
    validate_file_exists(template_path, "DeepStream pose-anchor assist config template")

    if not getattr(args, "out_base", ""):
        args.out_base = input_video.stem

    deepstream_app = shutil.which("deepstream-app")
    if not args.dry_run and not deepstream_app:
        raise SystemExit("Missing required binary: deepstream-app")
    if not args.dry_run:
        validate_file_exists(plugin_lib, "pose-anchor assist plugin library")

    preflight_rendered_text, refs = render_app_config(
        template_path=template_path,
        input_uri=input_video.as_uri(),
        output_file=Path("/tmp/ds_poseanchorassist_preflight.mp4"),
    )

    infer_config_local = alias_mapped_path(refs["infer_config"])
    tracker_config_local = alias_mapped_path(refs["tracker_config"])
    pose_sgie_config_local = alias_mapped_path(refs["pose_sgie_config"])

    validate_file_exists(infer_config_local, "infer config")
    validate_file_exists(tracker_config_local, "tracker config")
    validate_file_exists(pose_sgie_config_local, "pose SGIE config")

    pose_requirements = extract_pose_runtime_requirements(pose_sgie_config_local)
    validate_file_exists(pose_requirements["onnx_path"], "pose ONNX model")

    if not args.dry_run:
        validate_file_exists(pose_requirements["custom_lib_path"], "pose SGIE custom lib")

    infer_text = infer_config_local.read_text(encoding="utf-8")
    pose_sgie_text = pose_sgie_config_local.read_text(encoding="utf-8")
    validate_repo_alias_runtime(
        preflight_rendered_text,
        infer_text,
        pose_sgie_text,
        dry_run=args.dry_run,
    )

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)
    if args.dry_run and not deepstream_app:
        logger.warning("deepstream-app not found; continuing because --dry_run was requested.")
    if args.dry_run and not plugin_lib.exists():
        logger.warning("pose-anchor assist plugin library not found; continuing because --dry_run was requested: %s", plugin_lib)
    if args.dry_run and not pose_requirements["custom_lib_path"].exists():
        logger.warning(
            "pose SGIE custom lib not found; continuing because --dry_run was requested: %s",
            pose_requirements["custom_lib_path"],
        )

    output_video = run.out_dir / f"{args.out_base}_nvdcf_poseanchorassist.mp4"
    sidecar_path = run.out_dir / "pose_anchor_assist_sidecar.csv"
    rendered_config_path = run.out_dir / "ds_app_runtime.txt"
    rendered_text, _ = render_app_config(
        template_path=template_path,
        input_uri=input_video.as_uri(),
        output_file=output_video,
    )
    save_rendered_config(rendered_config_path, rendered_text)

    runtime_env = build_runtime_env(
        base_env=os.environ,
        plugin_dir=plugin_lib.parent,
        sidecar_path=sidecar_path,
        args=args,
        pose_sgie_uid=int(pose_requirements["pose_sgie_uid"]),
    )

    cmd = [deepstream_app or "deepstream-app", "-c", str(rendered_config_path)]
    cmd_str = shell_join(cmd)

    if run.outputs_enabled:
        dump_run_meta(
            run.out_dir,
            {
                "stage": STAGE,
                "input_video": str(input_video),
                "output_video": str(output_video),
                "sidecar_path": str(sidecar_path),
                "ds_config_template": str(template_path),
                "rendered_config_path": str(rendered_config_path),
                "infer_config_source_of_truth": str(infer_config_local),
                "tracker_config_source_of_truth": str(tracker_config_local),
                "pose_sgie_config": str(pose_sgie_config_local),
                "pose_onnx_model": str(pose_requirements["onnx_path"]),
                "pose_custom_lib": str(pose_requirements["custom_lib_path"]),
                "pose_engine_path": str(pose_requirements["engine_path"] or ""),
                "assist_plugin_lib": str(plugin_lib),
                "assist_plugin_dir": str(plugin_lib.parent),
                "pose_metadata_available_in_baseline": False,
                "pose_anchor_mode": "secondary_pose_tensor_or_fallback",
                "patch_source_mode": "true_pose_if_available_else_fallback",
                "survival_mode": "points_alive_proxy_alive",
                "assist_env": {
                    "AID_DSPOSEANCHORASSIST_HARD_MAX_PROXY_AGE_FRAMES": args.hard_max_proxy_age_frames,
                    "AID_DSPOSEANCHORASSIST_MIN_GOOD_POINTS": args.min_good_points,
                    "AID_DSPOSEANCHORASSIST_FEATURE_MAX_CORNERS": args.feature_max_corners,
                    "AID_DSPOSEANCHORASSIST_LK_WIN_SIZE": args.lk_win_size,
                    "AID_DSPOSEANCHORASSIST_PATCH_WIDTH_RATIO": args.patch_width_ratio,
                    "AID_DSPOSEANCHORASSIST_PATCH_HEIGHT_RATIO": args.patch_height_ratio,
                    "AID_DSPOSEANCHORASSIST_PATCH_Y_OFFSET_RATIO": args.patch_y_offset_ratio,
                    "AID_DSPOSEANCHORASSIST_MAX_CENTER_SHIFT_PX": args.max_center_shift_px,
                    "AID_DSPOSEANCHORASSIST_FREEZE_ON_PATCH_FAIL": args.freeze_on_patch_fail,
                    "AID_DSPOSEANCHORASSIST_HOLD_AFTER_FAIL_FRAMES": args.hold_after_fail_frames,
                    "AID_DSPOSEANCHORASSIST_POSE_SGIE_UID": pose_requirements["pose_sgie_uid"],
                    "AID_DSPOSEANCHORASSIST_POSE_MIN_KEYPOINT_CONF": DEFAULT_POSE_MIN_KEYPOINT_CONF,
                    "AID_DSPOSEANCHORASSIST_SIDECAR_PATH": str(sidecar_path),
                    "GST_PLUGIN_PATH": runtime_env["GST_PLUGIN_PATH"],
                },
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
    logger.info("pose_sgie_config=%s", pose_sgie_config_local)
    logger.info("pose_onnx_model=%s", pose_requirements["onnx_path"])
    logger.info("pose_custom_lib=%s", pose_requirements["custom_lib_path"])
    logger.info("assist_plugin_lib=%s", plugin_lib)
    logger.info("pose_metadata_available_in_baseline=%s", False)
    logger.info("pose_anchor_mode=%s", "secondary_pose_tensor_or_fallback")
    logger.info("patch_source_mode=%s", "true_pose_if_available_else_fallback")
    logger.info("survival_mode=%s", "points_alive_proxy_alive")
    logger.info("rendered_config_path=%s", rendered_config_path)
    logger.info("output_video=%s", output_video)
    logger.info("sidecar_path=%s", sidecar_path)
    logger.info("GST_PLUGIN_PATH=%s", runtime_env["GST_PLUGIN_PATH"])
    logger.info(
        "assist_env=hard_max_proxy_age_frames=%s min_good_points=%s feature_max_corners=%s "
        "lk_win_size=%s patch_width_ratio=%s patch_height_ratio=%s patch_y_offset_ratio=%s "
        "max_center_shift_px=%s freeze_on_patch_fail=%s hold_after_fail_frames=%s "
        "pose_sgie_uid=%s pose_min_keypoint_conf=%s",
        args.hard_max_proxy_age_frames,
        args.min_good_points,
        args.feature_max_corners,
        args.lk_win_size,
        args.patch_width_ratio,
        args.patch_height_ratio,
        args.patch_y_offset_ratio,
        args.max_center_shift_px,
        args.freeze_on_patch_fail,
        args.hold_after_fail_frames,
        pose_requirements["pose_sgie_uid"],
        DEFAULT_POSE_MIN_KEYPOINT_CONF,
    )
    logger.info("deepstream_cmd=%s", cmd_str)
    logger.info("run_out_dir=%s", run.out_dir)
    logger.info("run_log_path=%s", run.log_path)
    logger.info("run_cmd_path=%s", run.cmd_path)

    print(f"rendered config saved: {rendered_config_path}")
    print(f"output video path: {output_video}")
    print(f"proxy sidecar path: {sidecar_path}")
    print(f"assist plugin lib: {plugin_lib}")
    print(f"pose SGIE config: {pose_sgie_config_local}")
    print(f"pose ONNX model: {pose_requirements['onnx_path']}")
    print(f"pose SGIE custom lib: {pose_requirements['custom_lib_path']}")
    print("pose metadata in current DeepStream baseline source of truth: unavailable")
    print("pose anchor mode in this v4 branch: secondary pose tensor if available, otherwise explicit fallback")
    print("patch source mode: true_pose_if_available_else_fallback")
    print("survival mode: points_alive_proxy_alive")
    print(f"GST_PLUGIN_PATH: {runtime_env['GST_PLUGIN_PATH']}")
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
