from __future__ import annotations

from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .fsm import STATE_CAND, STATE_IN, STATE_OUT
from .roi import RoiCache

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TABLE_SCALE = 0.60
_TABLE_TH = 2
_ROW_H = 24
_MARGIN = 12
_TABLE_COL_X = [14, 190, 285, 390, 490, 600, 700]


def _state_bbox_color(state: str) -> tuple[int, int, int]:
    if state == STATE_CAND:
        return (0, 165, 255)  # orange
    if state == STATE_IN:
        return (0, 0, 255)  # red
    return (0, 165, 255)


def _state_bbox_thickness(state: str) -> int:
    if state == STATE_IN:
        return 3
    if state == STATE_CAND:
        return 2
    return 2


def _draw_text_with_bg(
    frame,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int],
    scale: float = 0.50,
    th: int = 2,
) -> None:
    tw, th_text = cv2.getTextSize(text, _FONT, scale, th)[0]
    x0 = int(max(0, x - 2))
    y0 = int(max(0, y - th_text - 2))
    x1 = int(min(frame.shape[1] - 1, x + tw + 2))
    y1 = int(min(frame.shape[0] - 1, y + 2))
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), _FONT, scale, color, th, cv2.LINE_AA)


def _draw_table(
    frame,
    *,
    header: list[str],
    rows: list[list[str]],
    x0: int,
    y0: int,
    table_w: int,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 10 + _ROW_H * (1 + len(rows)) + 10
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = min(w - 1, x0 + table_w)
    y1 = min(h - 1, y0 + bar_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), 1, cv2.LINE_AA)

    y = y0 + 24
    for i, title in enumerate(header):
        cv2.putText(frame, title, (x0 + _TABLE_COL_X[i], y), _FONT, _TABLE_SCALE, (210, 210, 210), _TABLE_TH, cv2.LINE_AA)
    for row in rows:
        y += _ROW_H
        color = (235, 235, 235)
        if len(row) >= 3 and row[2] == STATE_OUT:
            color = (180, 180, 180)
        for i, text in enumerate(row):
            if i >= len(_TABLE_COL_X):
                break
            cv2.putText(frame, text, (x0 + _TABLE_COL_X[i], y), _FONT, _TABLE_SCALE, color, _TABLE_TH, cv2.LINE_AA)


def _draw_top_table(frame, rows: Optional[list[dict[str, Any]]]) -> None:
    if not rows:
        return
    h, w = frame.shape[:2]
    head = ["id", "conf", "state", "score", "dist", "ov", "gap"]
    non_out = [r for r in rows if str(r.get("state", "")) != STATE_OUT]
    out_rows = [r for r in rows if str(r.get("state", "")) == STATE_OUT]
    draw_rows = non_out + out_rows
    if not draw_rows:
        return

    table_w = _TABLE_COL_X[-1] + 120
    x0 = max(0, w - table_w - _MARGIN)
    y0 = _MARGIN
    text_rows: list[list[str]] = []
    for row in draw_rows:
        text_rows.append(
            [
            str(row.get("roi_id", "")),
            f"{float(row.get('conf', 0.0)):.2f}",
            str(row.get("state", "")),
            f"{float(row.get('score', 0.0)):.2f}",
            f"{float(row.get('dist', 0.0)):.3f}",
            f"{float(row.get('ov', 0.0)):.3f}",
            f"{float(row.get('gap', 0.0)):+.3f}",
            ]
        )
    _draw_table(frame, header=head, rows=text_rows, x0=x0, y0=y0, table_w=table_w)


def _draw_person_table(frame, bbox_draw: Optional[list[dict[str, Any]]]) -> None:
    h, w = frame.shape[:2]
    head = ["pid", "conf", "state", "score", "dist", "ov", "gap"]
    shown = []
    for row in bbox_draw or []:
        if float(row.get("score", 0.0)) >= float(row.get("cand_thr", 0.35)):
            shown.append(row)
    shown.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    shown = shown[:6]

    text_rows: list[list[str]] = []
    if not shown:
        text_rows.append(["no candidates", "", "", "", "", "", ""])
    else:
        for i, row in enumerate(shown):
            text_rows.append(
                [
                    f"P{i}",
                    f"{float(row.get('conf', 0.0)):.2f}",
                    str(row.get("state", "")),
                    f"{float(row.get('score', 0.0)):.2f}",
                    f"{float(row.get('wd', 0.0)) * float(row.get('f_dist', 0.0)):.3f}",
                    f"{float(row.get('wo', 0.0)) * float(row.get('f_ov', 0.0)):.3f}",
                    f"{-float(row.get('wg', 0.0)) * float(row.get('p_gap', 1.0)):+.3f}",
                ]
            )

    table_w = _TABLE_COL_X[-1] + 120
    bar_h = 10 + _ROW_H * (1 + len(text_rows)) + 10
    x0 = max(0, w - table_w - _MARGIN)
    y0 = max(0, h - bar_h - _MARGIN)
    _draw_table(frame, header=head, rows=text_rows, x0=x0, y0=y0, table_w=table_w)


def _draw_bboxes(frame, bbox_draw: Optional[list[dict[str, Any]]]) -> None:
    if not bbox_draw:
        return
    h, w = frame.shape[:2]
    scale = 0.50
    th = 2
    pad = 4
    for row in bbox_draw:
        x1 = int(round(float(row.get("x1", 0.0))))
        y1 = int(round(float(row.get("y1", 0.0))))
        x2 = int(round(float(row.get("x2", 0.0))))
        y2 = int(round(float(row.get("y2", 0.0))))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        state = str(row.get("state", STATE_OUT))
        if state not in (STATE_CAND, STATE_IN):
            continue
        conf = float(row.get("conf", 0.0))
        score = float(row.get("score", 0.0))

        color = _state_bbox_color(state)
        thick = _state_bbox_thickness(state)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        c_text = f"c={conf:.2f}"
        s_text = f"s={score:.2f}"
        c_sz = cv2.getTextSize(c_text, _FONT, scale, th)[0]
        s_sz = cv2.getTextSize(s_text, _FONT, scale, th)[0]

        cx = max(0, min(w - c_sz[0] - 1, x2 - c_sz[0] - pad))
        cy = max(12, y1 + c_sz[1] + 2)
        _draw_text_with_bg(frame, c_text, cx, cy, color=color, scale=scale, th=th)

        sx = max(0, min(w - s_sz[0] - 1, x2 - s_sz[0] - pad))
        sy = min(h - 6, y2 - 4)
        _draw_text_with_bg(frame, s_text, sx, sy, color=color, scale=scale, th=th)


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
    draw_global: bool = False,
    top_table_rows: Optional[list[dict[str, Any]]] = None,
    bbox_draw: Optional[list[dict[str, Any]]] = None,
) -> None:
    poly = roi_cache.poly.astype(np.int32, copy=False).reshape((-1, 1, 2))
    cv2.polylines(frame, [poly], isClosed=True, color=(80, 220, 80), thickness=2, lineType=cv2.LINE_AA)
    if draw_global:
        _draw_bboxes(frame, bbox_draw=bbox_draw)
        _draw_top_table(frame, rows=top_table_rows)
        _draw_person_table(frame, bbox_draw=bbox_draw)
