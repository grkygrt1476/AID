# gst-dskltassist

Minimal DeepStream v1 KLT assist plugin for the AID repository.

## Scope

- Runs post-tracker / pre-OSD through the existing `deepstream-app` pipeline hook.
- Keeps NvDCF as the real tracker.
- Draws short-lived proxy boxes only for recently lost person tracks.
- Stops proxy output immediately when a plausible real track reappears nearby.
- Does not overwrite DeepStream object identity and does not implement intrusion logic.

## Runtime shape

- DeepStream app config still uses the `[ds-example]` section for insertion.
- For compatibility with that hook, the plugin registers the `dsexample` element name.
- The shared object name is `libnvdsgst_dskltassist.so`.

## Build

```bash
cd scripts/03_ds_single_stream/gst-dskltassist
CUDA_VER=12.8 make
```

To make `deepstream-app` see the plugin without installing it system-wide:

```bash
export GST_PLUGIN_PATH=/home/kihun/AID/scripts/03_ds_single_stream/gst-dskltassist:${GST_PLUGIN_PATH}
```

The `scripts/03_ds_single_stream/03_02_ds_klt_assist.py` wrapper sets `GST_PLUGIN_PATH` automatically.

## Wrapper-controlled tuning

The wrapper sets these environment variables before launching `deepstream-app`:

- `AID_DSKLTASSIST_PROXY_TTL_FRAMES`
- `AID_DSKLTASSIST_MAX_CENTER_SHIFT_PX`
- `AID_DSKLTASSIST_MIN_GOOD_POINTS`
- `AID_DSKLTASSIST_FEATURE_MAX_CORNERS`
- `AID_DSKLTASSIST_LK_WIN_SIZE`
- `AID_DSKLTASSIST_SIDECAR_PATH`

## Sidecar

If `AID_DSKLTASSIST_SIDECAR_PATH` is set, the plugin writes a minimal CSV with:

- `frame_num`
- `source_id`
- `track_id`
- `proxy_active`
- `proxy_age`
- `event`
- proxy bbox in original-frame pixels

## Current limits

- v1 assumes the current single-source validation path.
- person class only
- short TTL only
- translation-only proxy update
- fail-fast when feature quality is weak
