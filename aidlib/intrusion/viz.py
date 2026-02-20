from __future__ import annotations

from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .fsm import STATE_CAND, STATE_IN, STATE_OUT
from .roi import RoiCache


def _draw_text_block(
    frame,
    lines: list[str],
    x0: int,
    y0: int,
    *,
    text_color: tuple[int, int, int] = (235, 235, 235),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    if not lines:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    th = 1
    line_h = 18

    text_w = 0
    for line in lines:
        tw = cv2.getTextSize(line, font, scale, th)[0][0]
        text_w = max(text_w, tw)
    box_w = text_w + 12
    box_h = line_h * len(lines) + 8

    x0 = int(max(0, min(w - box_w - 1, x0)))
    y0 = int(max(0, min(h - box_h - 1, y0)))
    x1 = int(min(w - 1, x0 + box_w))
    y1 = int(min(h - 1, y0 + box_h))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        y = y0 + 18 + i * line_h
        cv2.putText(frame, line, (x0 + 6, y), font, scale, text_color, th, cv2.LINE_AA)


def _state_bbox_color(state: str) -> tuple[int, int, int]:
    if state == STATE_CAND:
        return (0, 165, 255)  # orange
    if state == STATE_IN:
        return (0, 0, 255)  # red
    return (130, 130, 130)  # optional gray for OUT


def draw_roi_view(
    frame,
    roi_cache: RoiCache,
    *,
    state: str,
    score_t: float,
    best_bbox: Optional[Sequence[float]],
    best_factors: Optional[dict[str, Any]],
    roi_index: int = 0,
    draw_out_bbox: bool = False,
) -> None:
    poly = roi_cache.poly.astype(np.int32, copy=False).reshape((-1, 1, 2))
    cv2.polylines(frame, [poly], isClosed=True, color=(80, 220, 80), thickness=2, lineType=cv2.LINE_AA)

    if best_bbox is not None and len(best_bbox) >= 4:
        draw_bbox = (state in (STATE_CAND, STATE_IN)) or draw_out_bbox
        if draw_bbox:
            x1 = int(round(float(best_bbox[0])))
            y1 = int(round(float(best_bbox[1])))
            x2 = int(round(float(best_bbox[2])))
            y2 = int(round(float(best_bbox[3])))
            color = _state_bbox_color(state)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    poly2 = roi_cache.poly
    cx = int(round(float(np.mean(poly2[:, 0]))))
    cy = int(round(float(np.mean(poly2[:, 1]))))
    label_x = cx - 80
    label_y = cy - 46 + roi_index * 16

    f_dist = float(best_factors.get("f_dist", 0.0)) if best_factors else 0.0
    f_ov = float(best_factors.get("f_ov", 0.0)) if best_factors else 0.0
    p_gap = float(best_factors.get("p_gap", 1.0)) if best_factors else 1.0
    lines = [
        f"{roi_cache.roi_id} {state} S={float(score_t):.2f}",
        f"d={f_dist:.2f} ov={f_ov:.2f} gap={p_gap:.2f}",
    ]
    _draw_text_block(frame, lines, label_x, label_y)

