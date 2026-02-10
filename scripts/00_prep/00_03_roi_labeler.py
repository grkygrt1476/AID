#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aidlib.run_utils import common_argparser, init_run, safe_mkdir

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - runtime availability
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - runtime availability
    np = None


STAGE = "00_prep"
WINDOW_NAME = "AID ROI Labeler"


@dataclass
class PreviewItem:
    video_id: str
    image_path: Path
    labeled_on: str  # "disp" or "orig"
    disp_scale: float
    image_size: Tuple[int, int]  # original (w, h)
    orig_snapshot_path: Optional[Path]


@dataclass
class LabelState:
    vertices: List[Tuple[int, int]] = field(default_factory=list)
    status_text: str = ""
    status_deadline: float = 0.0

    def set_status(self, message: str, duration_s: float = 1.5) -> None:
        self.status_text = message
        self.status_deadline = time.time() + duration_s

    def visible_status(self) -> str:
        if time.time() <= self.status_deadline:
            return self.status_text
        return ""


def ensure_project_dirs() -> None:
    for rel_path in (
        "data",
        "data/videos",
        "data/labels",
        "data/previews",
        "data/sets",
        "configs/cameras",
        "configs/rois",
        "outputs/00_prep",
        "outputs/logs/00_prep",
    ):
        safe_mkdir(PROJECT_ROOT / rel_path)


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_image_size(meta: Dict[str, object]) -> Optional[Tuple[int, int]]:
    image_size = meta.get("image_size")
    if isinstance(image_size, list) and len(image_size) >= 2:
        try:
            return int(image_size[0]), int(image_size[1])
        except (TypeError, ValueError):
            return None
    if isinstance(image_size, dict):
        try:
            return int(image_size.get("width")), int(image_size.get("height"))
        except (TypeError, ValueError):
            return None
    return None


def load_meta(meta_path: Path, logger: logging.Logger) -> Optional[Dict[str, object]]:
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse meta json %s: %s", meta_path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Meta json is not an object: %s", meta_path)
        return None
    return payload


def discover_preview_items(
    previews_dir: Path,
    meta_suffix: str,
    use_disp: bool,
    logger: logging.Logger,
) -> List[PreviewItem]:
    jpgs = sorted([p for p in previews_dir.glob("*.jpg") if p.is_file()])
    disp_by_id: Dict[str, Path] = {}
    orig_by_id: Dict[str, Path] = {}

    for jpg in jpgs:
        stem = jpg.stem
        if stem.endswith("_disp"):
            video_id = stem[: -len("_disp")]
            if video_id:
                disp_by_id[video_id] = jpg
        else:
            orig_by_id[stem] = jpg

    all_ids = sorted(set(disp_by_id.keys()) | set(orig_by_id.keys()))
    items: List[PreviewItem] = []

    for video_id in all_ids:
        disp_path = disp_by_id.get(video_id)
        orig_path = orig_by_id.get(video_id)
        meta_path = previews_dir / f"{video_id}{meta_suffix}"
        meta = load_meta(meta_path, logger) if meta_path.exists() else None

        meta_image_size = parse_image_size(meta) if meta else None
        meta_scale = None
        if meta is not None:
            try:
                meta_scale = float(meta.get("snapshot_disp_scale", 1.0))
            except (TypeError, ValueError):
                meta_scale = None

        if use_disp and disp_path is not None:
            if meta_scale is not None and meta_scale > 0:
                image_size = meta_image_size
                if image_size is None and orig_path is not None and cv2 is not None:
                    img = cv2.imread(str(orig_path))
                    if img is not None:
                        image_size = (int(img.shape[1]), int(img.shape[0]))
                if image_size is None and cv2 is not None:
                    img = cv2.imread(str(disp_path))
                    if img is not None:
                        image_size = (
                            int(round(img.shape[1] / meta_scale)),
                            int(round(img.shape[0] / meta_scale)),
                        )
                if image_size is None:
                    logger.warning("Could not determine original image_size for %s; skipping", video_id)
                    continue
                items.append(
                    PreviewItem(
                        video_id=video_id,
                        image_path=disp_path,
                        labeled_on="disp",
                        disp_scale=float(meta_scale),
                        image_size=image_size,
                        orig_snapshot_path=orig_path,
                    )
                )
                continue
            logger.warning(
                "Disp preview exists but valid meta scale is missing for %s; falling back to original image",
                video_id,
            )

        if orig_path is not None:
            if meta_image_size is not None:
                image_size = meta_image_size
            elif cv2 is not None:
                img = cv2.imread(str(orig_path))
                if img is None:
                    logger.warning("Failed to read original preview image: %s", orig_path)
                    continue
                image_size = (int(img.shape[1]), int(img.shape[0]))
            else:
                logger.warning("cv2 unavailable while probing image size for %s", orig_path)
                continue

            items.append(
                PreviewItem(
                    video_id=video_id,
                    image_path=orig_path,
                    labeled_on="orig",
                    disp_scale=1.0,
                    image_size=image_size,
                    orig_snapshot_path=orig_path,
                )
            )

    return items


def draw_roi_overlay(image, vertices: List[Tuple[int, int]], alpha: float = 0.28):
    canvas = image.copy()
    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2)) if vertices else None
    if len(vertices) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color=(40, 210, 80))
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)

    if len(vertices) >= 2:
        cv2.polylines(canvas, [pts], isClosed=len(vertices) >= 3, color=(0, 255, 0), thickness=2)

    for x, y in vertices:
        cv2.circle(canvas, (int(x), int(y)), 4, (0, 180, 255), thickness=-1)
    return canvas


def clamp_point(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    xx = max(0, min(int(x), max(0, width - 1)))
    yy = max(0, min(int(y), max(0, height - 1)))
    return xx, yy


def to_orig_vertices(
    disp_vertices: List[Tuple[int, int]],
    scale: float,
    orig_size: Tuple[int, int],
) -> List[List[int]]:
    if scale <= 0:
        scale = 1.0
    out: List[List[int]] = []
    width, height = orig_size
    for x, y in disp_vertices:
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        ox, oy = clamp_point(ox, oy, width, height)
        out.append([ox, oy])
    return out


def save_overlay_images(
    item: PreviewItem,
    roi_dir: Path,
    roi_prefix: str,
    vertices_disp: List[Tuple[int, int]],
    vertices_orig: List[List[int]],
    logger: logging.Logger,
) -> None:
    overlay_orig_path = roi_dir / f"{roi_prefix}_overlay.jpg"
    overlay_disp_path = roi_dir / f"{roi_prefix}_overlay_disp.jpg"

    original_img = None
    if item.orig_snapshot_path is not None and item.orig_snapshot_path.exists():
        original_img = cv2.imread(str(item.orig_snapshot_path))

    if original_img is None:
        disp_img = cv2.imread(str(item.image_path))
        if disp_img is None:
            logger.warning("Cannot create overlay image for %s; failed to read %s", item.video_id, item.image_path)
            return
        width, height = item.image_size
        original_img = cv2.resize(disp_img, (width, height), interpolation=cv2.INTER_LINEAR)
        logger.warning(
            "Original snapshot missing for %s; overlay generated from display image resized to original size",
            item.video_id,
        )

    orig_vertices_tuple = [(int(x), int(y)) for x, y in vertices_orig]
    overlay_orig = draw_roi_overlay(original_img, orig_vertices_tuple)
    cv2.imwrite(str(overlay_orig_path), overlay_orig)

    if item.labeled_on == "disp":
        disp_img = cv2.imread(str(item.image_path))
        if disp_img is not None:
            overlay_disp = draw_roi_overlay(disp_img, vertices_disp)
            cv2.imwrite(str(overlay_disp_path), overlay_disp)


def print_run_guidance() -> None:
    print("\nCommands to run:")
    print("cd /home/serdic/project/AID")
    print("python scripts/00_prep/00_03_roi_labeler.py --previews_dir data/previews --out_dir configs/rois")
    print("\nOutputs are saved to:")
    print("- ROI JSON: /home/serdic/project/AID/configs/rois/{video_id}/roi_{roi_id}_v{roi_version}.json")
    print("- Overlay:  /home/serdic/project/AID/configs/rois/{video_id}/roi_{roi_id}_v{roi_version}_overlay.jpg")
    print("- Logs:     /home/serdic/project/AID/outputs/logs/00_prep/00_03_roi_labeler_{run_ts}.cmd.txt/.log")
    print("\nResume from a given video_id:")
    print("python scripts/00_prep/00_03_roi_labeler.py --start_from E01_009")
    print("\nExample run command:")
    print("python scripts/00_prep/00_03_roi_labeler.py --roi_id area01 --roi_version 1 --use_disp true")


def print_cv2_install_help() -> None:
    print("ERROR: OpenCV (cv2) is not installed in the current environment.", file=sys.stderr)
    print("Install it inside the current WSL venv with:", file=sys.stderr)
    print("cd /home/serdic/project/AID", file=sys.stderr)
    print("source .venv/bin/activate", file=sys.stderr)
    print("python -m pip install --upgrade pip", file=sys.stderr)
    print("python -m pip install opencv-python", file=sys.stderr)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    ensure_project_dirs()

    parser = common_argparser()
    parser.add_argument("--previews_dir", default="data/previews")
    parser.add_argument("--meta_suffix", default=".meta.json")
    parser.add_argument("--roi_id", default="area01")
    parser.add_argument("--roi_version", type=int, default=1)
    parser.add_argument("--out_dir", default="configs/rois")
    parser.add_argument("--max_vertices", type=int, default=None)
    parser.add_argument("--use_disp", type=parse_bool, default=True)
    parser.add_argument("--start_from", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run = init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    if cv2 is None:
        print_cv2_install_help()
        logger.error("cv2 import failed; install opencv-python in current venv.")
        return 1
    if np is None:
        print_cv2_install_help()
        logger.error("numpy import failed; reinstall opencv-python in current venv.")
        return 1

    previews_dir = Path(args.previews_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    safe_mkdir(previews_dir)
    safe_mkdir(out_dir)

    items = discover_preview_items(
        previews_dir=previews_dir,
        meta_suffix=args.meta_suffix,
        use_disp=bool(args.use_disp),
        logger=logger,
    )
    if not items:
        logger.warning("No preview items found in %s", previews_dir)
        print_run_guidance()
        return 0

    if args.start_from:
        video_ids = [item.video_id for item in items]
        if args.start_from not in video_ids:
            logger.error("--start_from '%s' not found. Available first IDs: %s", args.start_from, video_ids[:10])
            print_run_guidance()
            return 1
        start_idx = video_ids.index(args.start_from)
        items = items[start_idx:]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    labeled_count = 0
    skipped_existing = 0

    try:
        for item in items:
            roi_dir = out_dir / item.video_id
            safe_mkdir(roi_dir)
            roi_prefix = f"roi_{args.roi_id}_v{args.roi_version}"
            roi_json_path = roi_dir / f"{roi_prefix}.json"

            if roi_json_path.exists() and not args.overwrite:
                logger.info("Skip existing ROI (use --overwrite to replace): %s", roi_json_path)
                skipped_existing += 1
                continue

            display_img = cv2.imread(str(item.image_path))
            if display_img is None:
                logger.warning("Failed to read preview image: %s", item.image_path)
                continue

            state = LabelState()
            state.set_status("Left click: add point | u:undo r:reset n:save-next q/ESC:quit", duration_s=5.0)

            def on_mouse(event, x, y, _flags, _userdata):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                if args.max_vertices is not None and len(state.vertices) >= int(args.max_vertices):
                    state.set_status(f"Max vertices reached: {args.max_vertices}")
                    return
                height, width = display_img.shape[:2]
                xx, yy = clamp_point(x, y, width, height)
                state.vertices.append((xx, yy))
                state.set_status(f"Vertex added ({len(state.vertices)})")

            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

            should_quit = False
            should_next = False

            while True:
                frame = draw_roi_overlay(display_img, state.vertices)
                h, _w = frame.shape[:2]

                cv2.putText(
                    frame,
                    f"video_id: {item.video_id} | mode: {item.labeled_on}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"vertices: {len(state.vertices)}",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Left click:add  u:undo  r:reset  n:save-next  q/ESC:quit",
                    (10, max(80, h - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

                status = state.visible_status()
                if status:
                    cv2.putText(
                        frame,
                        status,
                        (10, max(104, h - 42)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(20) & 0xFF

                if key in (27, ord("q")):
                    should_quit = True
                    break
                if key == ord("u"):
                    if state.vertices:
                        state.vertices.pop()
                        state.set_status(f"Undo -> vertices: {len(state.vertices)}")
                    else:
                        state.set_status("No vertices to undo")
                elif key == ord("r"):
                    state.vertices.clear()
                    state.set_status("Vertices reset")
                elif key == ord("n"):
                    if len(state.vertices) < 3:
                        state.set_status("Need at least 3 vertices to save")
                        continue
                    should_next = True
                    break

            if should_quit:
                logger.info("User requested quit. Stopping without saving current unsaved ROI.")
                break

            if not should_next:
                continue

            vertices_orig = to_orig_vertices(
                disp_vertices=state.vertices,
                scale=float(item.disp_scale),
                orig_size=item.image_size,
            )

            payload = {
                "video_id": item.video_id,
                "roi_id": args.roi_id,
                "roi_version": int(args.roi_version),
                "roi_type": "AREA",
                "vertices_px": vertices_orig,
                "image_size": {"width": int(item.image_size[0]), "height": int(item.image_size[1])},
                "labeled_on": item.labeled_on,
                "disp_scale_used": float(item.disp_scale),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            with roi_json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.write("\n")

            save_overlay_images(
                item=item,
                roi_dir=roi_dir,
                roi_prefix=roi_prefix,
                vertices_disp=state.vertices,
                vertices_orig=vertices_orig,
                logger=logger,
            )

            labeled_count += 1
            logger.info("Saved ROI for %s -> %s", item.video_id, roi_json_path)

    finally:
        cv2.destroyAllWindows()

    logger.info("Finished ROI labeling. labeled=%d skipped_existing=%d", labeled_count, skipped_existing)
    logger.info("Logs saved at %s and %s", run.cmd_path, run.log_path)
    print_run_guidance()
    return 0


if __name__ == "__main__":
    exit(main())
