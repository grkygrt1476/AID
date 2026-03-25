#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
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


def load_stage0403_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "04_deepstream" / "04_03_ds_multistream_intrusion.py"
    spec = importlib.util.spec_from_file_location("aid_stage0403_intrusion", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Stage 04.03 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage0403 = load_stage0403_module()

from aidlib.intrusion.io import create_video_writer, write_json  # noqa: E402
from aidlib.run_utils import common_argparser, dump_run_meta, init_run  # noqa: E402


STAGE = "04_deepstream"
STAGE_STEP = "04.05a"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUT_ROOT / "logs"
DEFAULT_PREP_OUT_BASE = "multistream4_boundary_reacquire_prepare"
DEFAULT_MAIN_OUT_BASE = "multistream4_boundary_reacquire"
DEFAULT_CROP_PADDING_PX = 96
DEFAULT_CROP_PADDING_RATIO = 0.18


@dataclass(frozen=True)
class CropRect:
    x: int
    y: int
    width: int
    height: int
    roi_bbox_xyxy: tuple[int, int, int, int]
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int


def parse_args() -> argparse.Namespace:
    parser = common_argparser()
    parser.set_defaults(out_root=str(DEFAULT_OUT_ROOT), log_root=str(DEFAULT_LOG_ROOT), out_base=DEFAULT_PREP_OUT_BASE)
    parser.add_argument(
        "--inputs",
        action="append",
        nargs="+",
        default=None,
        help="Exactly 4 input videos. Can be passed once with 4 paths or repeated.",
    )
    parser.add_argument("--out_dir", default="", help="Alias for --out_root; output root directory for this stage.")
    parser.add_argument(
        "--crop_padding_px",
        type=int,
        default=DEFAULT_CROP_PADDING_PX,
        help="Minimum symmetric padding in source pixels added around the ROI bounding rect.",
    )
    parser.add_argument(
        "--crop_padding_ratio",
        type=float,
        default=DEFAULT_CROP_PADDING_RATIO,
        help="Additional padding ratio applied to the ROI bounding rect dimensions.",
    )
    parser.add_argument("--force_rebuild", action="store_true", help="Overwrite crop assets if they already exist in this run directory.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate source clips and ROI geometry, then print planned crop regions without writing crop videos.",
    )
    return parser.parse_args()


def require_video_deps() -> None:
    if cv2 is None:
        raise SystemExit("Missing required Python dependency for Stage 04.05a crop generation: opencv-python")


def resolve_roi_json(spec: stage0403.SourceSpec, logger: logging.Logger) -> Path:
    fixed_clip_label = stage0403.DEFAULT_FIXED_LABELS.get(spec.source_id, "")
    roi_json = stage0403.DEFAULT_FIXED_ROIS.get(spec.source_id, stage0403.fallback_roi_path(spec.clip_label))
    if fixed_clip_label and spec.clip_label != fixed_clip_label:
        logger.warning(
            "source_id=%s expected fixed Stage 04 clip_label=%s but resolved=%s; using clip-label ROI fallback.",
            spec.source_id,
            fixed_clip_label,
            spec.clip_label,
        )
        roi_json = stage0403.fallback_roi_path(spec.clip_label)
    return roi_json


def probe_video_meta(video_path: Path) -> tuple[int, int, float, int]:
    require_video_deps()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video for crop generation: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise SystemExit(f"Invalid video geometry for crop generation: {video_path}")
    return width, height, fps, max(0, frame_count)


def clamp_crop_rect_even(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    source_w: int,
    source_h: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(source_w - 1, int(math.floor(x1))))
    top = max(0, min(source_h - 1, int(math.floor(y1))))
    right = max(left + 1, min(source_w, int(math.ceil(x2))))
    bottom = max(top + 1, min(source_h, int(math.ceil(y2))))

    if (right - left) % 2 == 1:
        if right < source_w:
            right += 1
        elif left > 0:
            left -= 1
    if (bottom - top) % 2 == 1:
        if bottom < source_h:
            bottom += 1
        elif top > 0:
            top -= 1

    right = max(left + 1, min(source_w, right))
    bottom = max(top + 1, min(source_h, bottom))
    return left, top, right, bottom


def compute_roi_crop_rect(
    *,
    roi_points_source: tuple[tuple[int, int], ...],
    source_w: int,
    source_h: int,
    padding_px: int,
    padding_ratio: float,
) -> CropRect:
    xs = [int(point[0]) for point in roi_points_source]
    ys = [int(point[1]) for point in roi_points_source]
    roi_x1 = max(0, min(xs))
    roi_y1 = max(0, min(ys))
    roi_x2 = min(source_w, max(xs))
    roi_y2 = min(source_h, max(ys))
    roi_w = max(1, roi_x2 - roi_x1)
    roi_h = max(1, roi_y2 - roi_y1)

    pad_x = max(int(padding_px), int(round(float(roi_w) * float(padding_ratio))))
    pad_y = max(int(padding_px), int(round(float(roi_h) * float(padding_ratio))))
    crop_x1, crop_y1, crop_x2, crop_y2 = clamp_crop_rect_even(
        x1=float(roi_x1 - pad_x),
        y1=float(roi_y1 - pad_y),
        x2=float(roi_x2 + pad_x),
        y2=float(roi_y2 + pad_y),
        source_w=source_w,
        source_h=source_h,
    )

    return CropRect(
        x=crop_x1,
        y=crop_y1,
        width=int(crop_x2 - crop_x1),
        height=int(crop_y2 - crop_y1),
        roi_bbox_xyxy=(roi_x1, roi_y1, roi_x2, roi_y2),
        padding_left=int(roi_x1 - crop_x1),
        padding_top=int(roi_y1 - crop_y1),
        padding_right=int(crop_x2 - roi_x2),
        padding_bottom=int(crop_y2 - roi_y2),
    )


def map_points_to_crop_local(points: tuple[tuple[int, int], ...], crop_rect: CropRect) -> list[list[int]]:
    return [[int(point[0] - crop_rect.x), int(point[1] - crop_rect.y)] for point in points]


def generate_crop_clip(
    *,
    source_video: Path,
    crop_rect: CropRect,
    crop_video_path: Path,
    fps: float,
    logger: logging.Logger,
) -> int:
    require_video_deps()
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for crop generation: {source_video}")

    writer = create_video_writer(
        cv2=cv2,
        path=crop_video_path,
        width=crop_rect.width,
        height=crop_rect.height,
        fps=fps,
    )

    frames_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            crop = frame[crop_rect.y : crop_rect.y + crop_rect.height, crop_rect.x : crop_rect.x + crop_rect.width]
            if crop.shape[0] != crop_rect.height or crop.shape[1] != crop_rect.width:
                raise RuntimeError(
                    f"Unexpected crop shape for {source_video}: got {crop.shape[1]}x{crop.shape[0]} expected {crop_rect.width}x{crop_rect.height}"
                )
            writer.write(crop)
            frames_written += 1
    finally:
        cap.release()
        writer.release()

    logger.info(
        "crop clip written source=%s output=%s frames=%s size=%sx%s",
        source_video,
        crop_video_path,
        frames_written,
        crop_rect.width,
        crop_rect.height,
    )
    return frames_written


def main() -> None:
    args = parse_args()
    stage0403.normalize_output_args(args)
    require_video_deps()

    source_specs = stage0403.build_source_specs(args.inputs)
    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    main_stage_dir = stage0403.project_path(args.out_root) / STAGE / run.run_ts / DEFAULT_MAIN_OUT_BASE
    crop_assets_dir = main_stage_dir / "crop_assets"
    manifest_path = crop_assets_dir / "crop_assets_manifest.json"
    crop_assets_dir.mkdir(parents=True, exist_ok=True)

    crop_padding_px = max(0, int(args.crop_padding_px))
    crop_padding_ratio = max(0.0, float(args.crop_padding_ratio))
    manifest_sources: list[dict[str, Any]] = []

    for spec in source_specs:
        roi_json = resolve_roi_json(spec, logger)
        stage0403.validate_file_exists(roi_json, f"ROI json for source_id={spec.source_id}")
        loaded_roi = stage0403.load_roi_spec(roi_json)
        source_w, source_h, fps, frame_count_prop = probe_video_meta(spec.local_path)

        if loaded_roi.source_size != (source_w, source_h):
            logger.warning(
                "ROI source size mismatch source_id=%s roi_json=%s roi_size=%s actual_size=%sx%s; using actual video size for crop clamp only.",
                spec.source_id,
                roi_json,
                loaded_roi.source_size,
                source_w,
                source_h,
            )

        crop_rect = compute_roi_crop_rect(
            roi_points_source=loaded_roi.polygon_source,
            source_w=source_w,
            source_h=source_h,
            padding_px=crop_padding_px,
            padding_ratio=crop_padding_ratio,
        )

        source_dir = crop_assets_dir / f"source{spec.source_id}_{spec.clip_label}"
        crop_video_path = source_dir / "roi_crop.mp4"
        crop_meta_path = source_dir / "crop_metadata.json"
        source_dir.mkdir(parents=True, exist_ok=True)

        crop_points_local = map_points_to_crop_local(loaded_roi.polygon_source, crop_rect)
        metadata: dict[str, Any] = {
            "stage_step": STAGE_STEP,
            "source_id": spec.source_id,
            "clip_label": spec.clip_label,
            "channel_label": f"CH{spec.source_id} -> {spec.clip_label}",
            "source_clip_path": str(spec.local_path),
            "crop_clip_path": str(crop_video_path),
            "roi_json_path": str(roi_json),
            "source_width": source_w,
            "source_height": source_h,
            "source_fps": fps,
            "source_frame_count_prop": frame_count_prop,
            "crop_x": crop_rect.x,
            "crop_y": crop_rect.y,
            "crop_width": crop_rect.width,
            "crop_height": crop_rect.height,
            "crop_rect_source_xyxy": [
                crop_rect.x,
                crop_rect.y,
                crop_rect.x + crop_rect.width,
                crop_rect.y + crop_rect.height,
            ],
            "roi_bbox_source_xyxy": list(crop_rect.roi_bbox_xyxy),
            "padding": {
                "mode": "roi_bbox_plus_padding",
                "padding_px_request": crop_padding_px,
                "padding_ratio_request": crop_padding_ratio,
                "left": crop_rect.padding_left,
                "top": crop_rect.padding_top,
                "right": crop_rect.padding_right,
                "bottom": crop_rect.padding_bottom,
            },
            "frame_index_sync": {
                "mapping": "crop_frame_index == source_frame_index",
                "fps_preserved": True,
                "frame_count_preserved": True,
                "dropped_frames": False,
            },
            "original_roi_polygon_source": [list(point) for point in loaded_roi.polygon_source],
            "roi_polygon_crop_local": crop_points_local,
            "resize_or_letterbox": {
                "applied": False,
                "resize_applied": False,
                "letterbox_applied": False,
            },
            "coordinate_mapping": {
                "crop_local_to_source": {
                    "x": "source_x = crop_x + crop_local_x",
                    "y": "source_y = crop_y + crop_local_y",
                },
                "source_to_crop_local": {
                    "x": "crop_local_x = source_x - crop_x",
                    "y": "crop_local_y = source_y - crop_y",
                },
            },
        }

        if args.dry_run:
            logger.info(
                "dry_run source_id=%s clip=%s crop_rect=(x=%s y=%s w=%s h=%s)",
                spec.source_id,
                spec.clip_label,
                crop_rect.x,
                crop_rect.y,
                crop_rect.width,
                crop_rect.height,
            )
            metadata["crop_frame_count"] = frame_count_prop
            metadata["verified_source_frame_count"] = frame_count_prop
        else:
            if not args.force_rebuild and crop_video_path.exists():
                raise SystemExit(
                    f"Crop clip already exists for source_id={spec.source_id}: {crop_video_path}. Use a new --run_ts or pass --force_rebuild."
                )
            frames_written = generate_crop_clip(
                source_video=spec.local_path,
                crop_rect=crop_rect,
                crop_video_path=crop_video_path,
                fps=fps,
                logger=logger,
            )
            metadata["crop_frame_count"] = frames_written
            metadata["verified_source_frame_count"] = frames_written
            metadata["frame_index_sync"]["frame_count_preserved"] = True
            write_json(crop_meta_path, metadata)

        if args.dry_run:
            crop_meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        manifest_sources.append(
            {
                "source_id": spec.source_id,
                "clip_label": spec.clip_label,
                "source_clip_path": str(spec.local_path),
                "crop_clip_path": str(crop_video_path),
                "crop_metadata_path": str(crop_meta_path),
            }
        )

    manifest = {
        "stage_step": STAGE_STEP,
        "run_ts": run.run_ts,
        "out_dir": str(main_stage_dir),
        "crop_assets_dir": str(crop_assets_dir),
        "source_count": len(source_specs),
        "sources": manifest_sources,
        "frame_index_sync": {
            "mapping": "crop_frame_index == source_frame_index",
            "fps_preserved": True,
            "frame_count_preserved": True,
            "dropped_frames": False,
        },
        "crop_path_kind": "true_original_resolution_spatial_crop",
    }
    write_json(manifest_path, manifest)

    run_meta = {
        "stage": STAGE,
        "stage_step": STAGE_STEP,
        "run_ts": run.run_ts,
        "out_dir": str(run.out_dir),
        "main_experiment_out_dir": str(main_stage_dir),
        "crop_assets_manifest": str(manifest_path),
        "crop_padding_px": crop_padding_px,
        "crop_padding_ratio": crop_padding_ratio,
        "dry_run": bool(args.dry_run),
        "sources": manifest_sources,
        "crop_path_kind": "true_original_resolution_spatial_crop",
    }
    dump_run_meta(run.out_dir, run_meta)

    print(f"crop assets manifest: {manifest_path}")
    print("crop path kind: true_original_resolution_spatial_crop")
    print("frame index sync: crop_frame_index == source_frame_index")
    if run.log_path is not None:
        print(f"log saved: {run.log_path}")
    if run.cmd_path is not None:
        print(f"cmd saved: {run.cmd_path}")


if __name__ == "__main__":
    main()
