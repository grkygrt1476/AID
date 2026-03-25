# gst-dsposeanchorassist

Bounded v4 pose-anchor comparison fork for the DeepStream AID baseline.

## What this branch is

- a separate fork from `03_02c`
- still post-tracker / pre-OSD
- still `deepstream-app` driven
- still display/proxy continuity only
- intended for honest comparison against bbox-derived fallback branches

## Important baseline fact

The current DeepStream baseline does **not** already contain pose metadata.

This v4 branch adds a secondary body-pose SGIE and then reads its `pose25d`
tensor metadata directly inside the assist plugin.

That means this branch is:

- `true_pose` when the SGIE is present and the `pose25d` tensor is available
- explicit fallback when pose tensor metadata is missing or weak

It does **not** relabel bbox heuristics as pose.

## Runtime behavior

- real NvDCF tracks always win
- while the real track is visible, the plugin tries to refresh a compact patch from real pose tensor metadata
- when the real track disappears, KLT carries continuity from that last pose-anchored patch
- real reacquisition still wins immediately
- a bounded frozen-hold can absorb technical patch failure
- hard max age remains a safety cap

## Runtime requirements

Besides the baseline detector/tracker configs, this branch also needs:

- `configs/deepstream/config_infer_secondary_bodypose3d_poseanchor.txt`
- `deepstream_tao_apps/models/bodypose3dnet/bodypose3dnet_accuracy.onnx`
- `deepstream_tao_apps/apps/tao_others/deepstream-pose-classification/nvdsinfer_custom_impl_BodyPose3DNet/libnvdsinfer_custom_impl_BodyPose3DNet.so`

The body-pose engine may be built by DeepStream from the ONNX if the engine file
is missing, but the custom infer library still needs to exist.

## Build

```bash
cd scripts/03_ds_single_stream/gst-dsposeanchorassist
CUDA_VER=12.8 make
```

The `scripts/03_ds_single_stream/03_02d_ds_pose_anchor_assist.py` wrapper sets
`GST_PLUGIN_PATH` automatically.

## Wrapper-controlled env

- `AID_DSPOSEANCHORASSIST_HARD_MAX_PROXY_AGE_FRAMES`
- `AID_DSPOSEANCHORASSIST_MIN_GOOD_POINTS`
- `AID_DSPOSEANCHORASSIST_FEATURE_MAX_CORNERS`
- `AID_DSPOSEANCHORASSIST_LK_WIN_SIZE`
- `AID_DSPOSEANCHORASSIST_PATCH_WIDTH_RATIO`
- `AID_DSPOSEANCHORASSIST_PATCH_HEIGHT_RATIO`
- `AID_DSPOSEANCHORASSIST_PATCH_Y_OFFSET_RATIO`
- `AID_DSPOSEANCHORASSIST_MAX_CENTER_SHIFT_PX`
- `AID_DSPOSEANCHORASSIST_FREEZE_ON_PATCH_FAIL`
- `AID_DSPOSEANCHORASSIST_HOLD_AFTER_FAIL_FRAMES`
- `AID_DSPOSEANCHORASSIST_POSE_SGIE_UID`
- `AID_DSPOSEANCHORASSIST_POSE_MIN_KEYPOINT_CONF`
- `AID_DSPOSEANCHORASSIST_SIDECAR_PATH`

## Sidecar

The sidecar CSV records:

- `mode=real|proxy|frozen_hold`
- proxy age / event
- stop and handoff reason
- bbox and patch rectangles
- `patch_source=true_pose|pose_unavailable_fallback`
- `pose_anchor_source`
- tracked point count
- compact flow summary
