# gst-dsposepatchassist

Bounded v2 pose-patch-style assist for the DeepStream AID baseline.

## What this branch is

- A clean fork of `03_02` v1.
- Still post-tracker / pre-OSD.
- Still `deepstream-app` driven.
- Still proxy/display continuity only.
- Not a tracker replacement and not intrusion logic.

## Important limitation

The current DeepStream baseline does **not** provide true pose metadata in-pipeline.

That means this v2 plugin currently uses an explicit bbox-derived fallback patch:

- centered horizontally on the person bbox
- vertically anchored near the upper-body / head-shoulder band
- smaller than the full bbox
- isolated in `try_extract_pose_anchor_from_meta()` and `build_patch_geometry()`

When real pose-anchor metadata becomes available later, that hook can be filled in without rewriting the rest of the plugin.

## Runtime behavior

- real NvDCF tracks always win
- proxy starts only for recently lost person tracks
- LK points are seeded inside the upper-body patch, not the full box
- proxy bbox translation follows patch motion
- proxy stops on TTL expiry, weak patch quality, implausible motion, or immediate handoff to a nearby real track
- one small bounded frozen-survival step is allowed for transient patch failure

## Build

```bash
cd scripts/03_ds_single_stream/gst-dsposepatchassist
CUDA_VER=12.8 make
```

The `scripts/03_ds_single_stream/03_02b_ds_pose_patch_assist.py` wrapper sets `GST_PLUGIN_PATH` automatically.

## Wrapper-controlled env

- `AID_DSPOSEPATCHASSIST_PROXY_TTL_FRAMES`
- `AID_DSPOSEPATCHASSIST_MAX_CENTER_SHIFT_PX`
- `AID_DSPOSEPATCHASSIST_MIN_GOOD_POINTS`
- `AID_DSPOSEPATCHASSIST_FEATURE_MAX_CORNERS`
- `AID_DSPOSEPATCHASSIST_LK_WIN_SIZE`
- `AID_DSPOSEPATCHASSIST_PATCH_WIDTH_RATIO`
- `AID_DSPOSEPATCHASSIST_PATCH_HEIGHT_RATIO`
- `AID_DSPOSEPATCHASSIST_PATCH_Y_OFFSET_RATIO`
- `AID_DSPOSEPATCHASSIST_SIDECAR_PATH`

## Sidecar

The sidecar CSV records:

- frame/source/track
- `mode=real|proxy`
- proxy age / event
- stop or handoff reason
- current bbox
- current patch bbox
- patch source
