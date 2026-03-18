from .features import FeatureConfig, compute_bbox_factors
from .fsm import FsmParams, FsmSnapshot, RoiFsm, STATE_CAND, STATE_IN, STATE_OUT
from .io import IOContext, append_jsonl, create_video_writer, init_io, load_yaml_config, save_yaml, write_json
from .roi import RoiCache, build_integral, build_roi_bottom_y, build_roi_cache, build_roi_mask, build_signed_distance, load_roi_polygon
from .score import ScoreWeights, clamp01, compute_score
try:
    from .viz import draw_roi_view
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    draw_roi_view = None  # type: ignore[assignment]

__all__ = [
    "IOContext",
    "RoiCache",
    "FeatureConfig",
    "ScoreWeights",
    "FsmParams",
    "FsmSnapshot",
    "RoiFsm",
    "STATE_OUT",
    "STATE_CAND",
    "STATE_IN",
    "load_roi_polygon",
    "build_roi_mask",
    "build_signed_distance",
    "build_integral",
    "build_roi_bottom_y",
    "build_roi_cache",
    "compute_bbox_factors",
    "clamp01",
    "compute_score",
    "draw_roi_view",
    "init_io",
    "load_yaml_config",
    "save_yaml",
    "write_json",
    "append_jsonl",
    "create_video_writer",
]
