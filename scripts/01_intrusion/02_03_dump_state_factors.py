#!/usr/bin/env python3
# 이 스크립트는 "비디오당 하나의 이벤트" 원칙을 적용합니다. manifest에 이벤트가 여러 개 있어도
# event_policy(first/longest)로 대표 이벤트 1개만 선택해 학습용 행을 생성합니다.
# 또한 "프레임당 대표 인원 1명" 원칙을 적용합니다. 같은 프레임에 사람이 여러 명이면
# rep_policy(기본 max_f_ov)로 대표 bbox 1개만 선택해 f_dist/f_ov/p_gap를 기록합니다.
# y_state 라벨은 event_frame과 cand_window_sec으로 결정합니다: IN(이벤트 내부),
# CAND(이벤트 경계 전후 cand window), OUT(그 외)로 매핑해 OUT=0/CAND=1/IN=2를 사용합니다.
# 다중 이벤트 비디오의 라벨 오염 방지를 위해 drop_other_events 옵션이 켜져 있으면
# 선택된 이벤트 외 다른 이벤트 구간에 속한 프레임은 CSV에서 제거합니다.
# 출력 CSV(state_factors.csv)는 비디오/이벤트/시간/원본프레임/y_state와 대표 bbox,
# 그리고 3개 요인(f_dist, f_ov, p_gap) 및 실행 정책(hold_det, event_policy, rep_policy 등)을 포함합니다.
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aidlib import run_utils

STAGE = "01_intrusion"


@dataclass
class ManifestEvent:
    video_id: str
    event_idx: Optional[int]
    src_video: str
    label_json: str
    ev_start_frame: Optional[int]
    ev_end_frame: Optional[int]
    fps: Optional[float]
    clip_start_sec: Optional[float]
    clip_path: str
    raw: dict[str, str]


@dataclass
class RepCandidate:
    bbox_clamped: list[float]
    conf: float
    f_dist: float
    f_ov: float
    p_gap: float
    key: float


def build_parser() -> Any:
    p = run_utils.common_argparser()
    p.set_defaults(out_root="outputs", log_root="outputs/logs", out_base="")

    p.add_argument("--manifest", default="data/clips/manifest.csv")
    p.add_argument("--labels_dir", default="data/videos/labels")
    p.add_argument("--rois_dir", default="data/videos/rois")
    p.add_argument("--cfg", default="configs/intrusion/mvp_v1.yaml")

    p.add_argument("--yolo_model", default="yolo11s.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=960)

    p.add_argument("--det_fps", type=float, default=10.0)
    p.set_defaults(hold_det=True)
    p.add_argument("--hold_det", dest="hold_det", action="store_true")
    p.add_argument("--no_hold_det", dest="hold_det", action="store_false")

    p.add_argument("--cand_window_sec", type=float, default=3.0)
    p.add_argument("--event_policy", choices=["first", "longest"], default="first")
    p.add_argument("--rep_policy", choices=["max_f_ov", "max_linear_score"], default="max_f_ov")
    p.add_argument("--w1", type=float, default=None)
    p.add_argument("--w2", type=float, default=None)
    p.add_argument("--w3", type=float, default=None)

    p.set_defaults(drop_other_events=True)
    p.add_argument("--drop_other_events", dest="drop_other_events", action="store_true")
    p.add_argument("--no_drop_other_events", dest="drop_other_events", action="store_false")

    p.add_argument("--choice", nargs="+", default=None)
    p.add_argument("--out_csv", default="")
    return p


def _norm_video_id(raw: str) -> str:
    token = str(raw).strip()
    if not token:
        return ""
    if token.lower().endswith(".mp4"):
        return Path(token).stem
    return Path(token).stem if "." in Path(token).name else token


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _load_manifest(path: Path) -> list[ManifestEvent]:
    rows: list[ManifestEvent] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for raw in rdr:
            row = {str(k): str(v) for (k, v) in raw.items()}
            rows.append(
                ManifestEvent(
                    video_id=str(row.get("video_id", "")).strip(),
                    event_idx=_to_int(row.get("event_idx", "")),
                    src_video=str(row.get("src_video", "")).strip(),
                    label_json=str(row.get("label_json", "")).strip(),
                    ev_start_frame=_to_int(row.get("ev_start_frame", "")),
                    ev_end_frame=_to_int(row.get("ev_end_frame", "")),
                    fps=_to_float(row.get("fps", "")),
                    clip_start_sec=_to_float(row.get("clip_start_sec", "")),
                    clip_path=str(row.get("clip_path", "")).strip(),
                    raw=row,
                )
            )
    return rows


def _select_one_event_per_video(
    *,
    rows: list[ManifestEvent],
    event_policy: str,
    choice_set: set[str],
) -> dict[str, ManifestEvent]:
    grouped: dict[str, list[ManifestEvent]] = {}
    for r in rows:
        vid = _norm_video_id(r.video_id)
        if not vid:
            continue
        if choice_set and vid not in choice_set:
            continue
        if r.ev_start_frame is None or r.ev_end_frame is None:
            continue
        if not r.clip_path:
            continue
        grouped.setdefault(vid, []).append(r)

    picked: dict[str, ManifestEvent] = {}
    for vid, items in grouped.items():
        if event_policy == "first":
            items_sorted = sorted(items, key=lambda x: ((x.event_idx if x.event_idx is not None else 10**9), x.ev_start_frame or 10**9))
            picked[vid] = items_sorted[0]
        else:
            items_sorted = sorted(
                items,
                key=lambda x: (
                    -((x.ev_end_frame or 0) - (x.ev_start_frame or 0)),
                    (x.event_idx if x.event_idx is not None else 10**9),
                ),
            )
            picked[vid] = items_sorted[0]
    return picked


def _load_events_from_label(label_path: Path) -> list[tuple[int, int]]:
    try:
        obj = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []

    ann = obj.get("annotations", {})
    raw = ann.get("event_frame", None) if isinstance(ann, dict) else None
    if raw is None:
        raw = obj.get("event_frame", None)

    out: list[tuple[int, int]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            s = int(item[0])
            e = int(item[1])
        except Exception:
            continue
        if e < s:
            s, e = e, s
        out.append((s, e))
    return out


def _pick_label_path(row: ManifestEvent, labels_dir: Path) -> Path:
    if row.label_json:
        p = Path(row.label_json)
        if p.exists():
            return p
    return labels_dir / f"{row.video_id}.json"


def _pick_roi_path(video_id: str, rois_dir: Path) -> Optional[Path]:
    p1 = rois_dir / video_id / "roi_area01_v1.json"
    if p1.exists():
        return p1
    p2 = rois_dir / video_id / "roi_area01_v1_fix.json"
    if p2.exists():
        return p2
    return None


def _inside_other_event(
    *,
    frame: int,
    all_events: list[tuple[int, int]],
    selected_idx: Optional[int],
    selected_pair: tuple[int, int],
) -> bool:
    selected_consumed = False
    for i, (s2, e2) in enumerate(all_events):
        if selected_idx is not None and 0 <= selected_idx < len(all_events):
            if i == selected_idx:
                continue
        else:
            if not selected_consumed and s2 == selected_pair[0] and e2 == selected_pair[1]:
                selected_consumed = True
                continue
        if s2 <= frame <= e2:
            return True
    return False


def _calc_y_state(orig_frame: int, s: int, e: int, cand_window_frames: int) -> int:
    if s <= orig_frame <= e:
        return 2
    if cand_window_frames > 0:
        if (s - cand_window_frames) <= orig_frame <= (s - 1):
            return 1
        if (e + 1) <= orig_frame <= (e + cand_window_frames):
            return 1
    return 0


def _load_feature_cfg(
    cfg_path: Path,
    logger: logging.Logger,
    FeatureConfig_cls,
    load_yaml_config_fn,
) -> Any:
    if cfg_path.exists():
        try:
            cfg = load_yaml_config_fn(cfg_path)
            score_cfg = cfg.get("score", {}) if isinstance(cfg, dict) else {}
            if isinstance(score_cfg, dict):
                return FeatureConfig_cls.from_score_cfg(score_cfg)
        except Exception as exc:
            logger.warning("Failed to parse cfg for FeatureConfig (%s): %s", cfg_path, exc)
    logger.warning("Using default FeatureConfig values")
    return FeatureConfig_cls()


def _choose_rep(
    *,
    bboxes: list[list[float]],
    roi_cache,
    feature_cfg: Any,
    image_w: int,
    image_h: int,
    rep_policy: str,
    w1: Optional[float],
    w2: Optional[float],
    w3: Optional[float],
    compute_bbox_factors_fn,
) -> Optional[RepCandidate]:
    best: Optional[RepCandidate] = None
    use_linear = rep_policy == "max_linear_score" and (w1 is not None and w2 is not None and w3 is not None)

    for bbox in bboxes:
        try:
            factors = compute_bbox_factors_fn(
                bbox=bbox,
                roi_cache=roi_cache,
                cfg_norms=feature_cfg,
                image_w=image_w,
                image_h=image_h,
            )
        except Exception:
            continue

        f_dist = float(factors.get("f_dist", 0.0))
        f_ov = float(factors.get("f_ov", 0.0))
        p_gap = float(factors.get("p_gap", 1.0))
        bb = list(factors.get("bbox_clamped", bbox))
        conf = float(bb[4]) if len(bb) >= 5 else 1.0

        key = (float(w1) * f_dist + float(w2) * f_ov - float(w3) * p_gap) if use_linear else float(f_ov)
        cand = RepCandidate(
            bbox_clamped=bb,
            conf=conf,
            f_dist=f_dist,
            f_ov=f_ov,
            p_gap=p_gap,
            key=float(key),
        )
        if best is None or cand.key > best.key:
            best = cand
    return best


def _parse_yolo_bboxes(result, image_w: int, image_h: int) -> list[list[float]]:
    out: list[list[float]] = []
    if result is None:
        return out
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) <= 0:
        return out

    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [1.0] * len(xyxy)

    for xy, c in zip(xyxy, confs):
        if len(xy) < 4:
            continue
        x1 = max(0.0, min(float(image_w - 1), float(xy[0])))
        y1 = max(0.0, min(float(image_h - 1), float(xy[1])))
        x2 = max(0.0, min(float(image_w - 1), float(xy[2])))
        y2 = max(0.0, min(float(image_h - 1), float(xy[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1, y1, x2, y2, float(c)])
    return out


def main() -> int:
    args = build_parser().parse_args()

    if float(args.det_fps) <= 0:
        print("[ERR] --det_fps must be > 0", file=sys.stderr)
        return 2
    if float(args.cand_window_sec) < 0:
        print("[ERR] --cand_window_sec must be >= 0", file=sys.stderr)
        return 2

    run = run_utils.init_run(stage=STAGE, script_file=__file__, args=args)
    logger = logging.getLogger(__name__)

    try:
        import cv2  # type: ignore
    except Exception as exc:
        logger.error("Missing dependency: cv2 (%s)", exc)
        return 2

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        logger.error("Missing dependency: ultralytics (%s)", exc)
        return 2

    try:
        from aidlib.intrusion import (
            FeatureConfig,
            build_roi_cache,
            compute_bbox_factors,
            load_roi_polygon,
            load_yaml_config,
        )
    except Exception as exc:
        logger.error("Failed to import intrusion factor utilities (%s)", exc)
        return 2

    manifest_path = Path(args.manifest)
    labels_dir = Path(args.labels_dir)
    rois_dir = Path(args.rois_dir)

    rows = _load_manifest(manifest_path)
    if not rows:
        logger.error("No rows loaded from manifest: %s", manifest_path)
        return 2

    choice_set = set(_norm_video_id(v) for v in (args.choice or []) if _norm_video_id(v))
    selected = _select_one_event_per_video(rows=rows, event_policy=str(args.event_policy), choice_set=choice_set)
    if not selected:
        logger.error("No videos selected after filtering/policy.")
        return 2

    feature_cfg = _load_feature_cfg(
        Path(args.cfg),
        logger=logger,
        FeatureConfig_cls=FeatureConfig,
        load_yaml_config_fn=load_yaml_config,
    )
    model = YOLO(str(args.yolo_model))

    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else (run.out_dir / "state_factors.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "video_id",
        "event_idx",
        "clip_path",
        "fps",
        "clip_start_sec",
        "t_sec",
        "orig_frame",
        "y_state",
        "f_dist",
        "f_ov",
        "p_gap",
        "rep_bbox_x1",
        "rep_bbox_y1",
        "rep_bbox_x2",
        "rep_bbox_y2",
        "rep_conf",
        "rep_policy",
        "event_policy",
        "cand_window_sec",
        "hold_det",
        "drop_other_events",
    ]

    videos_processed = 0
    rows_written = 0
    rows_dropped_other_events = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for video_id in sorted(selected.keys()):
            row = selected[video_id]

            clip_path = Path(row.clip_path)
            if not clip_path.exists():
                logger.warning("Skip %s: clip not found (%s)", video_id, clip_path)
                continue

            roi_path = _pick_roi_path(video_id, rois_dir=rois_dir)
            if roi_path is None:
                logger.warning("Skip %s: ROI json not found under %s", video_id, rois_dir / video_id)
                continue

            label_path = _pick_label_path(row, labels_dir=labels_dir)
            all_events = _load_events_from_label(label_path)
            if not all_events:
                logger.warning("Skip %s: no events in label (%s)", video_id, label_path)
                continue

            if row.ev_start_frame is None or row.ev_end_frame is None:
                logger.warning("Skip %s: missing event frame in manifest", video_id)
                continue

            s = int(row.ev_start_frame)
            e = int(row.ev_end_frame)
            if e < s:
                s, e = e, s
            selected_pair = (s, e)

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                logger.warning("Skip %s: failed to open clip (%s)", video_id, clip_path)
                continue

            fps_clip_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps_clip = fps_clip_raw if fps_clip_raw > 0 else (float(row.fps) if row.fps and row.fps > 0 else 30.0)
            fps_src = float(row.fps) if row.fps and row.fps > 0 else fps_clip
            clip_start_sec = float(row.clip_start_sec) if row.clip_start_sec is not None else 0.0
            det_stride = max(1, int(round(float(fps_clip) / float(args.det_fps))))
            cand_window_frames = max(0, int(round(float(args.cand_window_sec) * float(fps_src))))

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            roi_cache = None
            if w > 0 and h > 0:
                try:
                    poly = load_roi_polygon(roi_path)
                    roi_cache = build_roi_cache(roi_id=str(video_id), poly=poly, width=w, height=h)
                except Exception as exc:
                    cap.release()
                    logger.warning("Skip %s: failed to load/build ROI cache (%s)", video_id, exc)
                    continue

            frame_idx = -1
            last_bboxes: list[list[float]] = []
            videos_processed += 1

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    frame_idx += 1

                    if roi_cache is None:
                        h = int(frame.shape[0])
                        w = int(frame.shape[1])
                        try:
                            poly = load_roi_polygon(roi_path)
                            roi_cache = build_roi_cache(roi_id=str(video_id), poly=poly, width=w, height=h)
                        except Exception as exc:
                            logger.warning("Skip %s: failed ROI cache build after first frame (%s)", video_id, exc)
                            break

                    is_update = (frame_idx % det_stride) == 0
                    if is_update:
                        try:
                            results = model.predict(
                                source=frame,
                                classes=[0],
                                conf=float(args.conf),
                                imgsz=int(args.imgsz),
                                device=str(args.device),
                                verbose=False,
                            )
                        except Exception as exc:
                            logger.warning("YOLO failed %s frame=%d (%s)", video_id, frame_idx, exc)
                            continue
                        first = results[0] if results else None
                        last_bboxes = _parse_yolo_bboxes(first, image_w=w, image_h=h)
                        curr_bboxes = last_bboxes
                    else:
                        curr_bboxes = last_bboxes if bool(args.hold_det) else []

                    # det_fps 샘플링 기준 행 생성: update frame만 출력
                    if not is_update:
                        continue

                    if not curr_bboxes:
                        continue

                    rep = _choose_rep(
                        bboxes=curr_bboxes,
                        roi_cache=roi_cache,
                        feature_cfg=feature_cfg,
                        image_w=w,
                        image_h=h,
                        rep_policy=str(args.rep_policy),
                        w1=args.w1,
                        w2=args.w2,
                        w3=args.w3,
                        compute_bbox_factors_fn=compute_bbox_factors,
                    )
                    if rep is None:
                        continue

                    t_sec = float(frame_idx) / float(fps_clip) if fps_clip > 0 else 0.0
                    orig_frame = int(round((float(t_sec) + float(clip_start_sec)) * float(fps_src)))

                    if bool(args.drop_other_events) and _inside_other_event(
                        frame=orig_frame,
                        all_events=all_events,
                        selected_idx=row.event_idx,
                        selected_pair=selected_pair,
                    ):
                        rows_dropped_other_events += 1
                        continue

                    y_state = _calc_y_state(orig_frame=orig_frame, s=s, e=e, cand_window_frames=cand_window_frames)
                    class_counts[y_state] = class_counts.get(y_state, 0) + 1

                    bb = rep.bbox_clamped
                    writer.writerow(
                        {
                            "video_id": video_id,
                            "event_idx": "" if row.event_idx is None else int(row.event_idx),
                            "clip_path": str(clip_path),
                            "fps": f"{fps_src:.6f}",
                            "clip_start_sec": f"{clip_start_sec:.6f}",
                            "t_sec": f"{t_sec:.6f}",
                            "orig_frame": int(orig_frame),
                            "y_state": int(y_state),
                            "f_dist": f"{rep.f_dist:.6f}",
                            "f_ov": f"{rep.f_ov:.6f}",
                            "p_gap": f"{rep.p_gap:.6f}",
                            "rep_bbox_x1": f"{float(bb[0]):.3f}",
                            "rep_bbox_y1": f"{float(bb[1]):.3f}",
                            "rep_bbox_x2": f"{float(bb[2]):.3f}",
                            "rep_bbox_y2": f"{float(bb[3]):.3f}",
                            "rep_conf": f"{float(rep.conf):.6f}",
                            "rep_policy": str(args.rep_policy),
                            "event_policy": str(args.event_policy),
                            "cand_window_sec": f"{float(args.cand_window_sec):.3f}",
                            "hold_det": bool(args.hold_det),
                            "drop_other_events": bool(args.drop_other_events),
                        }
                    )
                    rows_written += 1
            finally:
                cap.release()

    logger.info("state_factors csv saved: %s", out_csv)
    logger.info(
        "summary | videos_processed=%d rows_written=%d rows_dropped_other_events=%d OUT=%d CAND=%d IN=%d",
        videos_processed,
        rows_written,
        rows_dropped_other_events,
        class_counts.get(0, 0),
        class_counts.get(1, 0),
        class_counts.get(2, 0),
    )

    print(
        "videos_processed={} rows_written={} rows_dropped_other_events={} OUT={} CAND={} IN={} csv={}".format(
            videos_processed,
            rows_written,
            rows_dropped_other_events,
            class_counts.get(0, 0),
            class_counts.get(1, 0),
            class_counts.get(2, 0),
            out_csv,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
