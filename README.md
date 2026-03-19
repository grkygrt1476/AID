# AID

This repository contains an intrusion-detection proof-of-concept pipeline with multiple iterations, archived experiments, and a DeepStream-based Stage 04 stack for hard boundary / occlusion cases.

The current working focus is `scripts/04_deepstream/`, especially the Stage 04 hard-case pipeline:

- detector + tracker baseline
- monitoring / operator OSD
- intrusion FSM
- continuity assist
- ROI crop preparation
- boundary reacquire with KLT-backed continuity

## Project Overview

Stage 04 is the DeepStream pipeline that handles difficult intrusion cases near ROI boundaries, partial occlusion, detector dropout, and short-lived reacquisition noise.

The intended confirmed-intrusion semantics are deliberately narrow:

- confirmed intrusion = `ankle_in_roi` OR `lower_body_overlap_confirm`
- upper/head KLT is continuity and geometry carry-over only
- head-only confirmation is intentionally disabled
- KLT-backed rows are allowed only to preserve or recover lower-body semantics, not to create new head semantics

In practice, the hard problem in this repo is not basic ROI scoring. It is maintaining enough continuity through dropouts and reacquires so that existing ankle / lower-body semantics still have stable geometry to operate on.

## Stage 04 Breakdown

| Stage | Role | Key files | Handoff |
| --- | --- | --- | --- |
| `04_01` baseline | Runs the raw DeepStream detector + tracker multistream baseline and exports the initial tracking artifacts. | `scripts/04_deepstream/04_01_ds_multistream_baseline.py`, `configs/deepstream/04_ds_yolo11_tracker_nvdcf_multistream4.txt`, `configs/deepstream/config_infer_primary_yolo11_clean.txt` | Produces the baseline multistream run used as the reference tracking/export pass. |
| `04_02` monitoring OSD | Adds monitoring overlays with the `gst-dsmonitorosd` plugin for operator-facing inspection. | `scripts/04_deepstream/04_02_ds_multistream_osd.py`, `scripts/04_deepstream/gst-dsmonitorosd/` | Keeps the same source layout and makes tracking / ROI state easier to inspect visually. |
| `04_03` intrusion | Runs the main intrusion decision pass and render helper using the shared FSM. | `scripts/04_deepstream/04_03_ds_multistream_intrusion.py`, `aidlib/intrusion/decision_fsm.py`, `configs/intrusion/mvp_v1.yaml` | Converts tracking sidecar rows into `OUT` / `CANDIDATE` / `IN_CONFIRMED` events plus summaries and overlays. |
| `04_04` continuity assist | Extends Stage 04.03 with upper-anchor continuity support to survive short tracking losses. | `scripts/04_deepstream/04_04_ds_multistream_continuity_assist.py`, `aidlib/intrusion/decision_fsm.py` | Keeps candidate / confirm context alive long enough for hard boundary cases to recover. |
| `04_05a` ROI crop clips | Generates per-source ROI crop videos and metadata with frame-index lockstep. | `scripts/04_deepstream/04_05a_make_roi_crop_clips.py` | Writes `crop_assets_manifest.json` consumed by `04_05`. |
| `04_05` boundary reacquire | Final hard-case stage. Augments sidecar rows with KLT-backed proxy / hold rows, gates weak real reacquires, then reruns the same intrusion FSM semantics. | `scripts/04_deepstream/04_05_ds_multistream_boundary_reacquire.py`, `aidlib/intrusion/decision_fsm.py`, `scripts/04_deepstream/04_03_ds_multistream_intrusion.py` | Produces the final Stage 04 intrusion events, summaries, sidecars, and tiled videos. |

## Current Intrusion Logic

### Confirm semantics

- `ankle_in_roi` remains the direct pose-confirm path.
- `lower_body_overlap_confirm` remains the lower-body confirm path.
- Stage 04 operator-facing basis stays `ankle` or `lower-body`.
- Head-only confirm paths remain disabled in the FSM and in the render/display mapping.

`scripts/04_deepstream/04_03_ds_multistream_intrusion.py` still maps:

- `ankle_in_roi` -> `ankle`
- `lower_body_overlap_confirm`
- `klt_ankle_proxy_confirm`
- `klt_bottom_center_proxy_confirm`
- `klt_current_lowerband_confirm`
- `klt_projected_bottom_center_confirm`

to the same operator-facing `lower-body` basis.

### What row types exist downstream from `04_05`

There are two different concepts and they should not be confused:

- `mode` is what the downstream FSM actually consumes.
- `row_source` is a Stage 04.05 debug / ownership field.

Downstream `mode` values currently loaded by `SidecarRow` are:

- `real`
- `proxy`
- `frozen_hold`

Stage 04.05 also writes `row_source` debug values such as:

- `real`
- `proxy`
- `frozen_hold`
- `real_support_only`

Important: `real_support_only` is not a new downstream FSM mode. It is a debug owner/status marker written by `04_05` while the actual row handed to the FSM remains a preserved KLT-backed row such as `frozen_hold`.

### What KLT / proxy / hold labels mean

- `real`: a detector-produced row currently owning geometry.
- `proxy`: a KLT-backed extrapolated row carrying geometry through missing detections.
- `frozen_hold`: a preserved KLT/hold row used when continuity is still trusted but geometry should not advance aggressively.
- `real_support_only`: a Stage 04.05 debug label meaning a real reacquire was observed, but it was intentionally treated as provisional support instead of immediate geometry ownership.

### Candidate and confirm behavior

Stage 04 currently does the following:

- candidate entry can come from real rows, proxy/frozen rows, or KLT continuity support
- lower-band overlap is computed from the current bbox for any valid row geometry
- ankle-proxy / bottom-center proxy / projected-bottom-center evidence can be computed on KLT-backed continuity rows
- confirm basis remains ankle or lower-body only

The practical hard-case bottlenecks are:

- weak one-frame real reacquires can still corrupt continuity if adoption is too aggressive
- KLT continuity quality is sensitive to reliable upper-anchor carry-over
- source numbering in outputs is 0-based (`source0`..`source3`), not human ordinal numbering
- `row_source` debug values can suggest a special row class, but the downstream FSM still keys primarily on `mode`

## Audit Summary

The current audit of Stage 04 found:

- KLT-backed rows do participate in candidate logic.
- Lower-body overlap is computed for KLT-backed rows.
- Ankle-proxy and other ankle-like lower-body proxy evidence is computed for KLT-backed rows.
- KLT-backed rows can already confirm through existing lower-body semantics.
- There was one real mismatch in the active lower-band confirm path: candidate support accepted broader upper-anchor continuity, but `klt_current_lowerband_confirm` still required a head-like anchor.

That mismatch has now been narrowed so the active current-lower-band confirm path uses the same upper-anchor continuity gate already used by candidate support and projected-bottom confirm. This preserves semantics because the confirm reason is still `lower_body_overlap_confirm`; it does not reintroduce any head-only confirm path.

## Inputs And Outputs

### Expected inputs

- 4 source clips, typically under `data/clips/<EVENT>/...`
- matching ROI JSON files under `data/videos/rois/...`
- DeepStream primary detector config under `configs/deepstream/`
- intrusion thresholds from `configs/intrusion/mvp_v1.yaml`
- for `04_05`, a crop manifest from `04_05a`

### Crop manifest usage

`04_05a` writes a manifest at:

- `outputs/04_deepstream/<RUN_TS>/multistream4_boundary_reacquire/crop_assets/crop_assets_manifest.json`

The manifest records:

- `source_id`
- `clip_label`
- `source_clip_path`
- `crop_clip_path`
- `crop_metadata_path`
- frame-index synchronization guarantees

The current manifest format explicitly states:

- `crop_frame_index == source_frame_index`
- FPS preserved
- frame count preserved
- no dropped frames

### Per-run output structure

Representative Stage 04.05 output tree:

```text
outputs/04_deepstream/<RUN_TS>/multistream4_boundary_reacquire/
├── boundary_reacquire_run_summary.json
├── ds_app_runtime.txt
├── multistream4_boundary_reacquire_tiled_boundary_reacquire.mp4
├── multistream4_boundary_reacquire_tiled_tracking_export.mp4
├── run_meta.json
├── tracking_sidecar_combined.csv
└── per_source/
    ├── source0_E01_001/
    ├── source1_E01_004/
    ├── source2_E01_008/
    └── source3_E01_011/
```

Each per-source directory typically contains:

- `tracking_sidecar.csv`
- `intrusion_events.jsonl`
- `intrusion_summary.json`

### Key output files

- `tracking_sidecar.csv`: tracking/export rows after Stage 04.05 sidecar augmentation. This is where `mode`, `proxy_age`, `row_source`, `real_reacquire_class`, KLT flow fields, and geometry ownership transitions are visible.
- `intrusion_events.jsonl`: one record per emitted decision/event frame. This contains `candidate_metrics`, `confirm`, `reasons`, and counters, and is the best file for auditing why a track entered candidate or confirmed intrusion.
- `intrusion_summary.json`: per-source run summary with sidecar stats, decision params, confirmed event count, row modes present, and high-level run metadata.
- `boundary_reacquire_run_summary.json`: top-level Stage 04.05 summary including tiled outputs, per-source split sidecar paths, crop clip paths, and KLT augmentation stats.
- `logs`: `outputs/logs/04_deepstream/*.log` captures the full DeepStream run, sidecar split summary, KLT augmentation counts, and final render completion.
- tiled videos: `*_tiled_tracking_export.mp4` and `*_tiled_boundary_reacquire.mp4` are useful for frame-by-frame operator review.
- crop assets: `04_05a` writes `crop_assets_manifest.json`, per-source `roi_crop.mp4`, and `crop_metadata.json` for the ROI-crop lockstep pass.

## Useful Debugging Fields

High-value fields currently present in Stage 04.05 outputs include:

- `mode`: the row type actually consumed by the downstream FSM (`real`, `proxy`, `frozen_hold`)
- `row_source`: Stage 04.05 geometry-ownership/debug label; can be `real_support_only` while `mode` remains `frozen_hold`
- `stop_reason`: why a row stopped, froze, or stayed provisional; useful values include `real_first_frame_provisional`, `real_temporal_consistency_insufficient`, and `real_geometry_reject`
- `handoff_reason`: compact ownership transition reason such as `real_support_only` or rejection-preserve cases
- `real_reacquire_class`: `REAL_ADOPT`, `REAL_SUPPORT_ONLY`, or `REAL_REJECT`
- `predicted_anchor_x`, `predicted_anchor_y`: conservative KLT-predicted anchor used to score a returning real row
- `real_anchor_innovation_px`: distance between predicted anchor and observed returning-real anchor
- `real_height_ratio`: returning real bbox height relative to the recent valid geometry
- `real_aspect_ratio_ratio`: returning real aspect ratio relative to the recent valid geometry
- `recent_real_adoptable_count`: short-window count of recent returning-real observations that looked adoptable
- `klt_reliable_chain_length`: how long the current reliable KLT continuity chain has been alive
- `klt_candidate_support_count`: how many recent frames satisfied the active KLT candidate-support rule
- `klt_current_lowerband_confirm`: whether the active current lower-band confirm branch fired on this frame
- `klt_ankle_proxy_confirm`: whether projected ankle proxy evidence confirmed lower-body intrusion
- `klt_projected_bottom_center_confirm`: whether projected bottom-center lower-body evidence confirmed intrusion
- `klt_current_reliable_continuity`: compact indicator of whether the current row is considered reliable KLT continuity

Practical debugging note:

- If `row_source=real_support_only` but `mode=frozen_hold`, the FSM is intentionally preserving the KLT/hold geometry owner while recording that a weak real reacquire was seen.

## Commands

Host syntax checks:

```bash
cd /home/kihun/AID
python3 -m py_compile scripts/04_deepstream/04_05_ds_multistream_boundary_reacquire.py
python3 -m py_compile aidlib/intrusion/decision_fsm.py
python3 -m py_compile scripts/04_deepstream/04_03_ds_multistream_intrusion.py
```

Container syntax checks:

```bash
docker exec ds8 bash -lc 'python3 -m py_compile /workspace/AID/scripts/04_deepstream/04_05_ds_multistream_boundary_reacquire.py'
docker exec ds8 bash -lc 'python3 -m py_compile /workspace/AID/aidlib/intrusion/decision_fsm.py'
docker exec ds8 bash -lc 'python3 -m py_compile /workspace/AID/scripts/04_deepstream/04_03_ds_multistream_intrusion.py'
```

Representative full Stage 04.05 run with crop manifest:

```bash
docker exec ds8 bash -lc 'set -e; cd /workspace/AID; /usr/bin/python3 scripts/04_deepstream/04_05_ds_multistream_boundary_reacquire.py --run_ts 20260319_140500 --crop_manifest /workspace/AID/outputs/04_deepstream/20260318_183500/multistream4_boundary_reacquire/crop_assets/crop_assets_manifest.json'
```

Representative crop-asset preparation flow:

```bash
python3 scripts/04_deepstream/04_05a_make_roi_crop_clips.py --run_ts <RUN_TS>
python3 scripts/04_deepstream/04_05_ds_multistream_boundary_reacquire.py --run_ts <RUN_TS> --crop_manifest outputs/04_deepstream/<PREP_TS>/multistream4_boundary_reacquire/crop_assets/crop_assets_manifest.json
```

## Known Issues And Current Status

- Weak real reacquire is still the main Stage 04.05 failure mode. The current mitigation is to classify returning real rows as `REAL_ADOPT`, `REAL_SUPPORT_ONLY`, or `REAL_REJECT` and preserve KLT-backed geometry when the first real row is too weak.
- KLT-backed confirm eligibility now stays within existing semantics: ankle proxy and lower-body overlap can confirm, but head-only confirmation remains disabled.
- The active lower-band KLT confirm path now accepts broader upper-anchor continuity, not just head-like anchors, which aligns it with candidate support and projected-bottom confirm behavior.
- `row_source` and `mode` are intentionally not the same thing. Most downstream logic reads `mode`, so debugging should always inspect both fields together.
- Output source folders are 0-based. `source3_E01_011` is the fourth stream.
- The ROI crop assets used by `04_05` may live under an earlier prep run timestamp rather than inside the final `04_05` output tree.
- This repo contains older archived experiments and vendor snapshots alongside the active Stage 04 path. Their presence is intentional for reference, not an indicator that the runtime uses all of them.

## Folder Usage Notes

These are inspection findings only. No cleanup or reorganization has been done.

- `deepstream_tao_apps`: actively used for the BodyPose3D secondary infer path. Current repo references point to `configs/deepstream/config_infer_secondary_bodypose3d_poseanchor.txt` and `scripts/03_deepstream/gst-dsposeanchorassist/README.md`.
- `DeepStream-Yolo-clean`: actively used by current DeepStream primary infer configs. `configs/deepstream/config_infer_primary_yolo11_clean.txt` and `configs/deepstream/04_config_infer_primary_yolo11_clean_b4.txt` both point to the compiled custom parser library under this folder.
- `DeepStream-Yolo`: appears to be a vendor/reference snapshot rather than an active Stage 04 runtime dependency. The current repo-level configs point at `DeepStream-Yolo-clean`, not this folder.
- `analysis`: appears to be passive analysis material only. The top level currently contains `analysis/eda_state_factors.ipynb`, and no active repo-level script/config references were found.
- `archive`: legacy reference material. The top level currently contains archived scripts and `archive/scripts/README.md`, but no active Stage 04 script/config references were found outside the archive itself.
- `runs`: apparently unused at the top level right now. The folder exists but no files were found under it during this audit and no active repo-level references were found.
- `weights`: actively used and not suspicious. DeepStream infer configs and older tracking scripts still reference model assets under `weights/`.

## Fast Navigation

- Stage 04 baseline: `scripts/04_deepstream/04_01_ds_multistream_baseline.py`
- Stage 04 intrusion render/helper: `scripts/04_deepstream/04_03_ds_multistream_intrusion.py`
- Shared intrusion FSM: `aidlib/intrusion/decision_fsm.py`
- Continuity assist: `scripts/04_deepstream/04_04_ds_multistream_continuity_assist.py`
- ROI crop prep: `scripts/04_deepstream/04_05a_make_roi_crop_clips.py`
- Final hard-case boundary reacquire: `scripts/04_deepstream/04_05_ds_multistream_boundary_reacquire.py`
