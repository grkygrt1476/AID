from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


STATE_OUT = "OUT"
STATE_CAND = "CAND"
STATE_IN = "IN"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass(frozen=True)
class FsmParams:
    cand_thr: float = 0.35
    in_thr: float = 0.60
    out_thr: float = 0.25
    enter_n: int = 2
    in_n: int = 3
    exit_n: int = 5
    grace_sec: float = 2.0
    fps: float = 30.0
    dwell_sec: float = 0.0

    @classmethod
    def from_cfg(cls, fsm_cfg: Mapping[str, Any], fps: float) -> "FsmParams":
        return cls(
            cand_thr=float(fsm_cfg.get("cand_thr", 0.35)),
            in_thr=float(fsm_cfg.get("in_thr", 0.60)),
            out_thr=float(fsm_cfg.get("out_thr", 0.25)),
            enter_n=max(1, int(fsm_cfg.get("enter_n", 2))),
            in_n=max(1, int(fsm_cfg.get("in_n", 3))),
            exit_n=max(1, int(fsm_cfg.get("exit_n", 5))),
            grace_sec=max(0.0, float(fsm_cfg.get("grace_sec", 2.0))),
            fps=max(1e-6, float(fps)),
            dwell_sec=max(0.0, float(fsm_cfg.get("dwell_sec", 0.0))),
        )

    def grace_frames(self) -> int:
        return max(0, int(round(self.grace_sec * self.fps)))

    def dwell_frames(self) -> int:
        return max(0, int(round(self.dwell_sec * self.fps)))


@dataclass(frozen=True)
class FsmSnapshot:
    state: str
    score_t: float
    cand_count: int
    in_count: int
    exit_count: int
    grace_left: int
    transition: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "score_t": float(self.score_t),
            "cand_count": int(self.cand_count),
            "in_count": int(self.in_count),
            "exit_count": int(self.exit_count),
            "grace_left": int(self.grace_left),
            "transition": self.transition,
        }


@dataclass
class RoiFsm:
    params: FsmParams
    state: str = STATE_OUT
    cand_count: int = 0
    in_count: int = 0
    exit_count: int = 0
    grace_left: int = 0
    cand_start_frame: Optional[int] = None
    cand_start_ts: Optional[float] = None

    def _reset_to_out(self) -> None:
        self.state = STATE_OUT
        self.cand_count = 0
        self.in_count = 0
        self.exit_count = 0
        self.grace_left = 0
        self.cand_start_frame = None
        self.cand_start_ts = None

    def update(self, score_t: float, frame_idx: int, ts_sec: float) -> FsmSnapshot:
        s = _clamp01(score_t)
        transition: Optional[str] = None

        if self.state == STATE_OUT:
            self.cand_count = (self.cand_count + 1) if (s >= self.params.cand_thr) else 0
            self.in_count = 0
            self.exit_count = 0
            if self.cand_count >= self.params.enter_n:
                self.state = STATE_CAND
                self.cand_start_frame = int(frame_idx)
                self.cand_start_ts = float(ts_sec)
                self.in_count = 0
                self.exit_count = 0
                transition = f"{STATE_OUT}->{STATE_CAND}"

        elif self.state == STATE_CAND:
            self.cand_count = (self.cand_count + 1) if (s >= self.params.cand_thr) else 0
            self.in_count = (self.in_count + 1) if (s >= self.params.in_thr) else 0
            self.exit_count = (self.exit_count + 1) if (s <= self.params.out_thr) else 0

            dwell_ok = True
            need_dwell_frames = self.params.dwell_frames()
            if need_dwell_frames > 0 and self.cand_start_frame is not None:
                dwell_ok = (int(frame_idx) - int(self.cand_start_frame) + 1) >= need_dwell_frames

            if self.in_count >= self.params.in_n and dwell_ok:
                self.state = STATE_IN
                self.grace_left = self.params.grace_frames()
                self.exit_count = 0
                transition = f"{STATE_CAND}->{STATE_IN}"
            elif self.exit_count >= self.params.exit_n:
                transition = f"{STATE_CAND}->{STATE_OUT}"
                self._reset_to_out()

        elif self.state == STATE_IN:
            if s > self.params.out_thr:
                self.exit_count = 0
                self.grace_left = self.params.grace_frames()
            else:
                if self.grace_left > 0:
                    self.grace_left -= 1
                else:
                    self.exit_count += 1

            if self.exit_count >= self.params.exit_n:
                transition = f"{STATE_IN}->{STATE_OUT}"
                self._reset_to_out()

        return FsmSnapshot(
            state=self.state,
            score_t=s,
            cand_count=self.cand_count,
            in_count=self.in_count,
            exit_count=self.exit_count,
            grace_left=self.grace_left,
            transition=transition,
        )

