from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass(frozen=True)
class ScoreWeights:
    wd: float = 0.60
    wo: float = 0.40
    wg: float = 0.35

    @classmethod
    def from_score_cfg(cls, score_cfg: Mapping[str, Any]) -> "ScoreWeights":
        weights = score_cfg.get("weights", {}) if isinstance(score_cfg, Mapping) else {}
        return cls(
            wd=float(weights.get("wd", 0.60)),
            wo=float(weights.get("wo", 0.40)),
            wg=float(weights.get("wg", 0.35)),
        )


def compute_score(
    factors: Mapping[str, float],
    weights: ScoreWeights | Mapping[str, float],
) -> float:
    if isinstance(weights, ScoreWeights):
        wd = float(weights.wd)
        wo = float(weights.wo)
        wg = float(weights.wg)
    else:
        wd = float(weights.get("wd", 0.60))
        wo = float(weights.get("wo", 0.40))
        wg = float(weights.get("wg", 0.35))

    f_dist = float(factors.get("f_dist", 0.0))
    f_ov = float(factors.get("f_ov", 0.0))
    p_gap = float(factors.get("p_gap", 1.0))

    return clamp01(wd * f_dist + wo * f_ov - wg * p_gap)

