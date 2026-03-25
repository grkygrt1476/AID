# gst-dsposepatchpersistent

Persistent v3 pose-patch continuity debug fork for the DeepStream AID baseline.

## What this branch is

- a separate fork from `03_02b`
- still post-tracker / pre-OSD
- still `deepstream-app` driven
- still display/proxy continuity only
- intentionally dirtier than v2 so continuity can be observed longer

## Important limitation

The current DeepStream baseline still does **not** provide true pose metadata in-pipeline.

This plugin therefore uses an explicit bbox-derived upper-body fallback patch:

- horizontally centered in the person bbox
- vertically anchored in the head/shoulder band
- smaller than the full bbox

The hook for future real pose-anchor integration is kept explicit in
`try_extract_pose_anchor_from_meta()`.

## Survival philosophy

This branch is different from v2:

- if tracked points are still alive, keep the proxy alive
- tiny motion is not a kill condition
- hard max age is only a safety cap
- technical patch failure can enter a bounded frozen-hold state
- real NvDCF reacquisition still wins immediately

## Build

```bash
cd scripts/03_ds_single_stream/gst-dsposepatchpersistent
CUDA_VER=12.8 make
```

The `scripts/03_ds_single_stream/03_02c_ds_pose_patch_persistent.py` wrapper sets
`GST_PLUGIN_PATH` automatically.

## Wrapper-controlled env

- `AID_DSPOSEPATCHPERSISTENT_HARD_MAX_PROXY_AGE_FRAMES`
- `AID_DSPOSEPATCHPERSISTENT_MIN_GOOD_POINTS`
- `AID_DSPOSEPATCHPERSISTENT_FEATURE_MAX_CORNERS`
- `AID_DSPOSEPATCHPERSISTENT_LK_WIN_SIZE`
- `AID_DSPOSEPATCHPERSISTENT_PATCH_WIDTH_RATIO`
- `AID_DSPOSEPATCHPERSISTENT_PATCH_HEIGHT_RATIO`
- `AID_DSPOSEPATCHPERSISTENT_PATCH_Y_OFFSET_RATIO`
- `AID_DSPOSEPATCHPERSISTENT_MAX_CENTER_SHIFT_PX`
- `AID_DSPOSEPATCHPERSISTENT_FREEZE_ON_PATCH_FAIL`
- `AID_DSPOSEPATCHPERSISTENT_HOLD_AFTER_FAIL_FRAMES`
- `AID_DSPOSEPATCHPERSISTENT_SIDECAR_PATH`

## Sidecar

The sidecar CSV records:

- `mode=real|proxy|frozen_hold`
- proxy age / event
- stop and handoff reason
- bbox and patch rectangles
- patch source
- tracked point count
- compact flow summary
