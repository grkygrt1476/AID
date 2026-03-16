```markdown
# Stage 03 DeepStream Intrusion Pipeline

## What this document is

This README documents the Stage 03 single-stream intrusion pipeline under `scripts/03_deepstream` as it currently exists in this repo.

It is grounded in:

- the Stage 03 Python entrypoints and wrappers
- the `gst-dsposepatchpersistent` continuity plugin source
- the intrusion FSM and ROI code under `aidlib/intrusion`
- the DeepStream config templates under `configs/deepstream`
- representative Stage 03 outputs and logs under `outputs/03_deepstream` and `outputs/logs/03_deepstream`

Where a statement is directly supported by code, it is presented as code-backed. Where it is supported by run artifacts, the relevant run or output path is named. Where something is inferred or still ambiguous, that is stated explicitly.

---

## 1. Stage 03 goal

Stage 03 is the DeepStream **single-stream** intrusion stage for one fixed CCTV camera and one manually defined ROI polygon.

The practical goal of this stage was to make one stream technically honest and inspectable before moving to multi-stream orchestration.

Concretely, Stage 03 tries to do the following:

- use DeepStream for person detection and NvDCF tracking
- preserve short-term continuity near the ROI boundary when the detector or tracker drops a person
- convert raw tracking geometry into a track-level intrusion decision sequence
- keep the final truth definition honest: **final confirmation requires ankle evidence entering the ROI**
- make the user workflow simple enough that one command produces inspectable artifacts

DeepStream was used here because:

- the repo already had a working DeepStream detector + tracker baseline
- single-stream file-input execution was enough to debug continuity, ROI logic, and confirmation semantics
- the main need at this stage was not deployment polish, but reproducible videos, logs, and event artifacts

“Done” for Stage 03 does **not** mean “perfect intrusion detection.”  
It means:

- there is a working one-command single-stream entrypoint
- continuity and final truth are separated cleanly enough to defend
- the runtime leaves behind inspectable outputs
- known limitations are documented instead of hidden

---

## 2. Final pipeline overview

The final Stage 03 pipeline has **two separate layers** that are easy to confuse.

### 2.1 Tracking-time continuity

Tracking-time continuity lives in:

- `scripts/03_deepstream/03_02c_ds_pose_patch_persistent.py`
- `scripts/03_deepstream/gst-dsposepatchpersistent/gstdsposepatchpersistent.cpp`

Its role is to preserve short-lived continuity when the real tracked person disappears near the ROI.

What it does:

- runs the standard DeepStream detector + NvDCF tracker
- derives a small upper-body mini patch from the bbox
- uses KLT optical flow on that patch
- keeps a proxy box alive while tracked patch points remain usable
- writes proxy geometry to a sidecar CSV
- renders a tracking video with continuity overlay

What it does **not** do:

- it does **not** consume true pose metadata from the baseline
- it does **not** directly define final intrusion truth
- it does **not** directly confirm ankle entry

Important clarification:

In practice, `03_02c` uses a **bbox-top-derived upper-body mini patch**, often around the head/shoulder neighborhood, but this is **not** a true head-keypoint patch and not a pose-metadata-driven anchor. The anchor is derived from bbox geometry, not from real pose metadata.

In current code, the baseline is explicitly marked as:

- `pose_metadata_available_in_baseline: false`
- `patch_source_mode: bbox_fallback_only`

That is the correct interpretation of the Stage 03 continuity branch.

### 2.2 Decision-time ankle confirmation

Decision-time ankle confirmation lives in:

- `scripts/03_deepstream/03_03_ds_intrusion_fsm.py`
- `aidlib/intrusion/decision_fsm.py`

Its role is to convert continuity-aware tracking evidence into an event-level intrusion decision.

What it does:

- reads the original input video
- reads the ROI polygon
- reads the `03_02c` sidecar CSV
- builds candidate evidence from bbox / ROI geometry
- runs an FSM with `OUT`, `CANDIDATE`, and `IN_CONFIRMED`
- when a track is in a confirm-worthy state, runs a pose model on the current bbox crop and checks whether at least one ankle is inside the ROI

What matters semantically:

- proxy continuity can keep a track in `CANDIDATE`
- proxy continuity can contribute candidate evidence
- proxy continuity **cannot** directly create final truth
- final truth remains **ankle-based confirmation**, when the pose branch is actually available at runtime

### 2.3 Final Stage 03 entrypoint

The user-facing entrypoint for Stage 03 is:

- `scripts/03_deepstream/03_03_ds_intrusion_fsm.py`

This script supports two modes:

- **reuse-sidecar mode**: reuse an existing `03_02c` sidecar CSV
- **auto-run mode**: internally launch `03_02c`, then continue into the intrusion decision pass

In normal one-command use, Stage 03 produces:

- the `03_02c` tracking MP4
- the `03_02c` sidecar CSV
- `intrusion_events.jsonl`
- `intrusion_summary.json`
- the final intrusion overlay MP4
- `run_meta.json`

---

## 3. Important files

### Main scripts

#### `scripts/03_deepstream/03_01_ds_baseline.py`

Thin DeepStream wrapper around the plain detector + NvDCF baseline.

It:

- renders a runtime config from `configs/deepstream/ds_yolo11_tracker_nvdcf.txt`
- validates runtime-dependent paths
- writes the baseline MP4 and `run_meta.json`

This is a baseline wrapper, not the final Stage 03 entrypoint.

#### `scripts/03_deepstream/03_02_ds_klt_assist.py`

Earlier KLT continuity experiment.

It:

- uses the `gst-dskltassist` plugin
- writes `klt_proxy_sidecar.csv`
- demonstrates that LK-based continuity can be attached to DeepStream output

This was an intermediate experiment, not the final continuity baseline.

#### `scripts/03_deepstream/03_02c_ds_pose_patch_persistent.py`

Final Stage 03 continuity wrapper.

It:

- launches DeepStream tracking with the `gst-dsposepatchpersistent` plugin
- writes `*_nvdcf_posepatchpersistent.mp4`
- writes `pose_patch_persistent_sidecar.csv`
- records explicit metadata showing that the baseline has no true pose metadata

This is the continuity layer used by the final one-command Stage 03 path.

#### `scripts/03_deepstream/03_03_ds_intrusion_fsm.py`

Final Stage 03 entrypoint.

It owns:

- sidecar reuse vs internal tracking auto-run
- intrusion decision pass
- final overlay rendering
- consolidated artifact output

This is the top-level Stage 03 script.

### Plugin / C++ continuity code

#### `scripts/03_deepstream/gst-dsposepatchpersistent/gstdsposepatchpersistent.cpp`

Core continuity plugin.

It:

- builds the upper-body mini patch from the bbox
- seeds Shi-Tomasi corners with `cv::goodFeaturesToTrack`
- advances them with `cv::calcOpticalFlowPyrLK`
- emits `real`, `proxy`, and `frozen_hold` rows to the sidecar
- renders proxy / patch overlay into the tracking video

#### `scripts/03_deepstream/gst-dsposepatchpersistent/gstdsposepatchpersistent.h`

Declares `PersistentTrackState`, including state such as:

- `proxy_age_frames`
- `last_tracked_points`
- `patch_points`
- `frozen_hold_remaining_frames`

#### `scripts/03_deepstream/gst-dsposepatchpersistent/README.md`

Documents the plugin intent and build flow.

It is useful because it also states the important truth that the baseline still does not provide true pose metadata.

### Intrusion logic

#### `aidlib/intrusion/decision_fsm.py`

Defines the offline intrusion decision logic.

It:

- loads the sidecar CSV and ROI cache
- reconstructs candidate evidence
- runs the FSM over frame/track records
- contains `PoseAnkleProbe`
- emits `intrusion_events.jsonl`
- builds the summary consumed by the final overlay

#### `aidlib/intrusion/roi.py`

Loads the ROI polygon from JSON and normalizes it into pixel-space caches that can be used during offline decision logic.

### DeepStream configs

#### `configs/deepstream/ds_yolo11_tracker_nvdcf.txt`

Baseline detector + tracker template.

#### `configs/deepstream/ds_yolo11_tracker_nvdcf_intrusionfsm.txt`

Tracking template used by the Stage 03 intrusion path.

#### `configs/deepstream/config_infer_primary_yolo11_clean.txt`

Primary detector config, referencing:

- `weights/person/yolo11s.onnx`
- `weights/person/yolo11s_ds.engine`

#### `configs/deepstream/config_tracker_NvDCF_viz.yml`

NvDCF tracker config used in this Stage 03 branch.

### Sample ROI / input

#### `configs/rois/E01_001/roi_area01_v1_fix.json`

Area ROI used in the Stage 03 sample clip.

#### `data/clips/E01_001/ev00_f1826-2854_50s.mp4`

Main Stage 03 sample clip used across representative run artifacts.

---

## 4. Execution flow

### Canonical Stage 03 command

~~~bash
cd /workspace/AID

/usr/bin/python3 scripts/03_deepstream/03_03_ds_intrusion_fsm.py \
  --input_video data/clips/E01_001/ev00_f1826-2854_50s.mp4 \
  --roi_json configs/rois/E01_001/roi_area01_v1_fix.json \
  --out_root outputs \
  --out_base ev00_ds_intrusionfsm
~~~

### What happens when `03_03` runs

When `03_03_ds_intrusion_fsm.py` runs, it:

1. validates the input video and ROI JSON
2. loads decision defaults
3. decides whether to reuse a sidecar or auto-run tracking
4. in auto-run mode, launches `03_02c_ds_pose_patch_persistent.py`
5. verifies that the tracking MP4 and sidecar were actually produced
6. runs `run_intrusion_decision_pass(...)`
7. writes `intrusion_events.jsonl`
8. writes `intrusion_summary.json`
9. renders the final intrusion overlay MP4 unless disabled
10. writes `run_meta.json`

### Reuse-sidecar mode

If `--tracking_sidecar_csv` is provided:

- `03_03` skips internal tracking
- directly consumes the supplied continuity CSV
- can optionally gather it into the run folder for inspection

### Auto-run tracking mode

If `--tracking_sidecar_csv` is omitted:

- `03_03` internally launches `03_02c`
- the tracking run uses the same `run_ts` and output root
- the resulting MP4 and sidecar land in the same run folder
- the continuity artifacts become input to the decision pass

### Output layout

Stage 03 run folders are created under:

~~~text
outputs/03_deepstream/<run_ts>/<out_base>/
~~~

Typical outputs include:

- `ev00_ds_intrusionfsm_nvdcf_posepatchpersistent.mp4`
- `pose_patch_persistent_sidecar.csv`
- `intrusion_events.jsonl`
- `intrusion_summary.json`
- `ev00_ds_intrusionfsm_intrusion_overlay.mp4`
- `run_meta.json`
- `ds_app_runtime.txt`

Logs and saved commands live under:

~~~text
outputs/logs/03_deepstream/
~~~

---

## 5. Major implementation steps and evolution

### Step 1: plain DeepStream baseline

`03_01_ds_baseline.py` established the basic detector + tracker wrapper around `deepstream-app`.

This stage focused on:

- runtime config rendering
- config validation
- stable output/log folder conventions

### Step 2: early KLT continuity experiments

`03_02_ds_klt_assist.py` introduced a short-miss LK continuity branch.

This proved that:

- DeepStream output could be augmented with a sidecar
- LK-based continuity could bridge some short gaps

But it was not yet the final continuity design.

### Step 3: finalize the continuity baseline as `03_02c`

The final Stage 03 continuity baseline became:

- `03_02c_ds_pose_patch_persistent.py`

This changed the continuity design in important ways:

- the tracked patch became a smaller upper-body mini patch rather than the whole bbox
- the proxy stayed alive while tracked points remained usable
- a bounded `frozen_hold` state was added for technical failures
- sidecar output became more useful for offline decision logic

This is where the Stage 03 semantic split became important:

- `03_02c` is continuity-only
- `03_02c` is not the final truth layer

### Step 4: make `03_03` the one-command entrypoint

`03_03_ds_intrusion_fsm.py` became the final Stage 03 entrypoint.

This mattered because it removed the need to manually coordinate:

- continuity generation
- sidecar reuse
- decision pass
- overlay rendering

Instead, one command now owns the full Stage 03 flow.

### Step 5: add candidate-triggered pose confirmation

The final truth branch lives in:

- `aidlib/intrusion/decision_fsm.py`

This pose branch is:

- model-backed
- candidate-triggered
- ankle-focused
- intentionally separate from the tracking continuity layer

Representative run artifacts show both degraded and successful cases:

- `outputs/03_deepstream/20260316_160236/ev00_ds_intrusionfsm/intrusion_summary.json`
  - `pose_probe_status = ultralytics_import_failed:ModuleNotFoundError`
  - `confirmed_events = 0`

- `outputs/03_deepstream/20260316_215933/ev00_ds_intrusionfsm/intrusion_summary.json`
  - `pose_probe_status = ready`
  - `confirmed_events = 5`

This is important because the repo contains not only the intended design, but also actual success and failure artifacts.

### Step 6: late debugging and runtime fixes

Several important fixes landed after the one-command path already existed:

- the `03_03 -> 03_02c` auto-run path was updated to force `--hard_max_proxy_age_frames 120`
- the FSM pose gate was adjusted so `CANDIDATE` grace frames can still attempt probing if a usable bbox exists
- the continuity overlay code was updated to render KLT points and an anchor marker in the tracking video

These fixes matter because they changed actual runtime behavior, not just comments or naming.

---

## 6. Key technical decisions

### Why `03_03` became the one-command entrypoint

Because Stage 03 needed to be inspectable as **one run**, not as a fragile chain of manually coordinated scripts.

That design makes it easier to defend:

- one input clip
- one ROI
- one continuity source
- one decision pass
- one artifact bundle

### Why `03_02c` is continuity-only

Because LK-tracked proxy geometry is useful for temporal continuity, but not honest enough to define final truth by itself.

A proxy box can:

- preserve candidate continuity
- help bridge reacquisition
- reduce event flicker

But it cannot honestly answer the product question:

> Did an ankle enter the ROI?

### Why proxy can maintain `CANDIDATE` but cannot confirm

This is the core semantic split of Stage 03.

Candidate evidence can come from:

- a real tracked bbox
- a proxy bbox
- a short `frozen_hold` continuation

Final confirmation requires ankle evidence. That is why continuity and truth are intentionally not the same thing.

### Why pose confirm is candidate-triggered instead of always-on

Because always-on pose would be expensive and unnecessary.

The implemented strategy is:

- use cheaper bbox / ROI geometry first
- only probe with the pose model when the track is already in a candidate-worthy context

This preserves the semantic design and keeps runtime cost bounded.

### Why Stage 03 is single-stream first

Because the hard part here was not scaling out to many streams. The hard part was making one stream **technically honest**:

- one ROI
- one continuity layer
- one decision model
- one artifact structure that can actually be debugged

Multi-stream orchestration belongs after this is stable.

---

## 7. Problems encountered

This section is intentionally blunt. These were real Stage 03 problems.

### Problem 1: pose-like naming implied more than the code actually did

The word `pose` was the biggest source of confusion.

Examples:

- `03_02c_ds_pose_patch_persistent.py`
- `gst-dsposepatchpersistent`
- `pose_patch_persistent_sidecar.csv`

Those names suggest real pose support, but the actual continuity path is bbox-fallback upper-body mini-patch KLT.

Code-backed evidence:

- `pose_metadata_available_in_baseline: false`
- `patch_source_mode: bbox_fallback_only`
- `try_extract_pose_anchor_from_meta()` currently returns `false`

### Problem 2: tracking-time continuity and decision-time ankle confirm were easy to conflate

There are two different systems:

1. tracking-time mini-patch KLT in `03_02c`
2. decision-time ankle confirmation in `03_03` / `decision_fsm.py`

They serve different purposes and run in different places. Much of the confusion in Stage 03 came from treating them as if they were one “pose feature.”

### Problem 3: pose model configured did not mean pose confirm was active at runtime

`03_03` can record a pose configuration status in `run_meta.json`, but that is only a preflight/config-level status.

Representative mismatch:

- `outputs/03_deepstream/20260316_160236/ev00_ds_intrusionfsm/run_meta.json`
  - `pose_probe_status = pose_model_configured`

- `outputs/03_deepstream/20260316_160236/ev00_ds_intrusionfsm/intrusion_summary.json`
  - `pose_probe_status = ultralytics_import_failed:ModuleNotFoundError`

That means a run can look configured while actually degrading to candidate-only behavior.

### Problem 4: early auto-run still used the old 18-frame cap

Before the later handoff patch, `03_03` did not explicitly pass the longer proxy age into its internal `03_02c` command.

Artifact-backed evidence:

- `outputs/logs/03_deepstream/03_03_ds_intrusion_fsm_20260316_160236.log`
  - internal command had no explicit `--hard_max_proxy_age_frames`

- `outputs/logs/03_deepstream/03_02c_ds_pose_patch_persistent_20260316_160236.log`
  - `assist_env=hard_max_proxy_age_frames=18 ...`

That meant the intended persistent behavior existed conceptually, but not yet in the normal one-command path.

### Problem 5: KLT points were not visible in the tracking video

The plugin had internal point-tracking state, but the video did not clearly show it.

That made it hard to tell whether the proxy was:

- truly following local texture
- freezing
- drifting
- surviving for valid reasons or fallback reasons

### Problem 6: continuity improved, but floating proxy geometry remained

Even with improved survival, continuity did not solve the geometric realism problem.

The continuity layer can preserve an upper-body patch and a proxy bbox, but that does not guarantee correct contact semantics at the ankle / floor boundary.

In other words:

- continuity can improve temporal stability
- continuity cannot fundamentally solve the “floating proxy” problem by itself

### Problem 7: host vs container path confusion

The DeepStream configs assume `/workspace/AID/...` as the canonical mount path.

But parts of the repo may also be viewed or launched from host paths such as `~/AID`.

This created real friction when validating configs and artifacts.

### Problem 8: timezone / run timestamp confusion

Run folders and log stamps were not always in the same timezone context.

Example:

- run folder like `outputs/03_deepstream/20260317_051150/...`
- log lines stamped `2026-03-16 19:54:50`

That is a 9-hour offset consistent with host/container timezone mismatch.

---

## 8. How each problem was resolved

### Resolution 1: make the repo honest about `03_02c`

This was resolved mostly through explicit metadata and logging.

What changed:

- `03_02c` writes `pose_metadata_available_in_baseline: false`
- `03_02c` writes `patch_source_mode: bbox_fallback_only`
- the plugin README explicitly states the limitation

What remains unsolved:

- the filenames still carry historical `pose` wording
- the runtime is more honest than the naming

### Resolution 2: separate continuity from final truth in the architecture

This was the most important correction.

What changed:

- `03_02c` was kept continuity-only
- `03_03` became the final truth entrypoint
- the FSM allows proxy continuity to sustain candidate evidence, but not final truth

What remains unsolved:

- the distinction still has to be read carefully; it is not obvious from filenames alone

### Resolution 3: make pose-confirm runtime-visible

This was only partially resolved.

What changed:

- `decision_fsm.py` reports the effective pose branch status in `intrusion_summary.json`
- the repo contains both degraded and successful runs

What remains unsolved:

- `run_meta.json` still contains a preflight `pose_probe_status` that can be misread as runtime success

### Resolution 4: force the 120-frame cap in the normal auto-run path

This was resolved by updating `03_03_ds_intrusion_fsm.py` so the internal `03_02c` command explicitly receives:

~~~text
--hard_max_proxy_age_frames 120
~~~

Artifact-backed evidence after the patch:

- `outputs/logs/03_deepstream/03_03_ds_intrusion_fsm_20260317_052546.log`
  - `tracking_stage_cmd=... --hard_max_proxy_age_frames 120`

- `outputs/logs/03_deepstream/03_02c_ds_pose_patch_persistent_20260317_052546.log`
  - `assist_env=hard_max_proxy_age_frames=120 ...`

What remains unsolved:

- this is still a cap, not a guarantee of perfect continuity

### Resolution 5: add visible KLT markers to the tracking overlay

This was resolved in the plugin source by updating the overlay path to draw:

- a proxy / patch anchor marker
- sampled KLT points from `patch_points`
- a `pts=<n>` count in the proxy label

What remains unsolved:

- the C++ plugin must be rebuilt before this becomes visible in runtime videos
- some older output videos predate the rebuild and therefore do not prove the new overlay behavior

### Resolution 6: allow pose probing during `CANDIDATE` grace if a bbox still exists

The earlier confirm gate was stricter than intended and could skip probing during candidate grace.

What changed in `decision_fsm.py`:

- if `candidate_geom` exists, probe that bbox
- otherwise, if the FSM is already in `STATE_CANDIDATE` and the current row still has a usable bbox, probe that bbox
- only skip if there is no usable bbox at all

What remains unsolved:

- this only helps if the pose runtime is actually available

---

## 9. Known limitations and remaining issues

These limitations should not be hidden.

- `03_02c` is still bbox-fallback upper-body mini-patch KLT, not true pose tracking
- `03_02c` can improve continuity without fundamentally solving ankle / floor contact realism
- the floating proxy problem is still a real limitation near the boundary
- `03_03` is still a single-stream design
- decision-time pose confirm depends on runtime availability of `ultralytics`, the pose model file, and compatible Python dependencies
- a run can be configured for pose confirm but still degrade at runtime
- the tracking overlay is debug-oriented, not polished UI
- DeepStream config paths still assume `/workspace/AID` as the canonical mount point

---

## 10. How to run

### Rebuild the continuity plugin

Do this whenever `gstdsposepatchpersistent.cpp` changes:

~~~bash
cd /workspace/AID/scripts/03_deepstream/gst-dsposepatchpersistent
CUDA_VER=12.8 make
~~~

### Run the one-command Stage 03 path

~~~bash
cd /workspace/AID

/usr/bin/python3 scripts/03_deepstream/03_03_ds_intrusion_fsm.py \
  --input_video data/clips/E01_001/ev00_f1826-2854_50s.mp4 \
  --roi_json configs/rois/E01_001/roi_area01_v1_fix.json \
  --out_root outputs \
  --out_base ev00_ds_intrusionfsm
~~~

### Reuse an existing tracking sidecar

~~~bash
cd /workspace/AID

/usr/bin/python3 scripts/03_deepstream/03_03_ds_intrusion_fsm.py \
  --input_video data/clips/E01_001/ev00_f1826-2854_50s.mp4 \
  --roi_json configs/rois/E01_001/roi_area01_v1_fix.json \
  --tracking_sidecar_csv outputs/03_deepstream/<run_ts>/<out_base>/pose_patch_persistent_sidecar.csv \
  --out_root outputs \
  --out_base ev00_ds_intrusionfsm_reuse
~~~

### Check the latest logs

~~~bash
cd /workspace/AID
find outputs/logs/03_deepstream -maxdepth 1 -type f | sort | tail -n 20
~~~

### Confirm the 120-frame handoff actually applied

~~~bash
cd /workspace/AID

LATEST_03_03_LOG=$(find outputs/logs/03_deepstream -type f -name '03_03_ds_intrusion_fsm_*.log' | sort | tail -n 1)
LATEST_03_02C_LOG=$(find outputs/logs/03_deepstream -type f -name '03_02c_ds_pose_patch_persistent_*.log' | sort | tail -n 1)

grep -n "tracking_stage_cmd=.*hard_max_proxy_age_frames 120" "$LATEST_03_03_LOG"
grep -n "assist_env=.*hard_max_proxy_age_frames=120" "$LATEST_03_02C_LOG"
~~~

### Confirm whether pose confirm was actually active

Do **not** rely only on `run_meta.json`.

Check the final summary:

~~~bash
cd /workspace/AID

LATEST_SUMMARY=$(find outputs/03_deepstream -type f -name 'intrusion_summary.json' | sort | tail -n 1)
cat "$LATEST_SUMMARY"
~~~

Interpretation:

- `pose_probe_status = ready` means the pose branch actually loaded
- `pose_probe_status = ultralytics_import_failed:...` means the run degraded to candidate-only behavior
- `confirmed_events > 0` shows that the final ankle-confirm path succeeded at least once in that run

### Find the latest tracking and intrusion videos

~~~bash
cd /workspace/AID

find outputs/03_deepstream -type f -name '*_nvdcf_posepatchpersistent.mp4' | sort | tail -n 3
find outputs/03_deepstream -type f -name '*_intrusion_overlay.mp4' | sort | tail -n 3
~~~

### What to look for in the tracking video

After rebuilding the plugin, the tracking MP4 should show debug continuity markers during proxy-active periods:

- proxy bbox
- mini-patch rectangle
- small anchor marker
- sampled KLT points
- proxy label text with `pts=<n>`

### What to look for in the intrusion overlay

The intrusion overlay is intended to show:

- the ROI polygon
- track boxes reconstructed from the sidecar
- track-level FSM state
- transition highlights
- compact run-state HUD elements

Exact visual details may vary slightly depending on the current overlay revision.

---

## 11. Output and artifact guide

### Tracking outputs from `03_02c`

#### `*_nvdcf_posepatchpersistent.mp4`

Tracking inspection video from the continuity stage.

#### `pose_patch_persistent_sidecar.csv`

Per-frame tracking continuity sidecar.

Representative fields include:

- `frame_num`
- `track_id`
- `mode`
- `proxy_active`
- `proxy_age`
- `event`
- bbox geometry
- patch geometry
- `patch_source`
- `tracked_points`
- optical-flow summary fields

Representative rows from `outputs/03_deepstream/20260317_052546/ev00_ds_intrusionfsm/pose_patch_persistent_sidecar.csv` show:

- `mode=real` for detector / tracker output
- `mode=proxy` for continuity output
- `patch_source=bbox_fallback`

### Decision outputs from `03_03`

#### `intrusion_events.jsonl`

Per-record event stream across frames and active tracks.

Contains state, transition, candidate metrics, confirm status, and ankle information when pose probing succeeds.

Representative confirm event from:

- `outputs/03_deepstream/20260316_215933/ev00_ds_intrusionfsm/intrusion_events.jsonl`

shows:

- `event_type = in_confirmed`
- `mode = proxy`
- `confirm.status = ankle_in_roi`
- `evidence.ankle_confirm = true`

This matters because it proves that proxy continuity can sustain the event while final truth still comes from ankle evidence.

#### `intrusion_summary.json`

Compact run summary including:

- `pose_probe_status`
- `confirmed_events`
- sidecar summary
- effective decision parameters

#### `*_intrusion_overlay.mp4`

Offline inspection video generated by `03_03`.

#### `run_meta.json`

Run-level metadata describing the planned flow, paths, and wrapper-level settings.

### Logs

`outputs/logs/03_deepstream/<script>_<run_ts>.log` is the authoritative source for:

- which wrapper command actually ran
- whether `03_03` auto-ran `03_02c`
- whether the 120-frame handoff was applied
- whether the continuity baseline still reported `bbox_fallback_only`

---

## 12. Final status of Stage 03

What is effectively closed for Stage 03:

- the one-command single-stream workflow exists
- `03_03` is the entrypoint
- `03_02c` is the continuity baseline used by that entrypoint
- continuity and final truth are separated clearly enough to defend
- the repo contains both degraded and successful pose-confirm runs
- the auto-run path now explicitly forces the longer 120-frame proxy cap

What is good enough but still imperfect:

- `03_02c` materially improves continuity, but it is still bbox-derived upper-body patch logic
- the overlays are useful for debugging, not polished presentation
- the runtime still depends heavily on environment correctness

A known visual/geometry limitation remains:

- proxy continuity can improve temporal stability without fully resolving the **floating proxy** or physical-contact realism problem near the boundary

What is intentionally left for later stages:

- multi-stream orchestration
- stronger ankle / floor contact reasoning
- deeper redesign of the continuity model
- true in-pipeline pose-anchor continuity instead of bbox-fallback patch logic

---

## Portfolio / interview framing

The strongest honest Stage 03 story is:

- start from a standard DeepStream detector + tracker baseline
- identify that boundary-near misses break event continuity
- add a bounded continuity layer that preserves candidate evidence without pretending it is final truth
- keep final truth separate by requiring ankle confirmation
- turn the whole stage into a one-command artifact-producing pipeline
- document the messy parts, including dependency failures, naming confusion, and runtime-vs-config ambiguity

That is a stronger and more believable story than claiming Stage 03 solved intrusion detection completely.
```

