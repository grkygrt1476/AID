# Stage 04 -- Multistream DeepStream Intrusion Pipeline

## 1. What Stage 04 is

Stage 04 is the multistream extension of the AID intrusion detection pipeline. It takes the single-stream event-level intrusion logic developed in Stage 03 and runs it across multiple camera sources simultaneously on a shared GPU, using NVIDIA DeepStream for batched inference and NvDCF tracking.

The core design goal of Stage 04 is not simply "run more cameras." It is to demonstrate that the boundary-aware confirmation semantics -- the three-state FSM, ankle-level evidence, KLT continuity bridging, and gated reacquisition -- work correctly and at acceptable throughput when multiple sources compete for the same compute resources.

Stage 03 proved the intrusion semantics in a controlled single-stream setting. Stage 04 validates that those semantics survive under realistic multi-camera pressure.

## 2. Why multistream was needed

### Event-level intrusion vs. per-frame alerts

Most CCTV intrusion systems trigger on per-frame bounding-box overlap with an ROI. This is fast but produces many false positives near boundaries -- a person walking along a fence, a partially occluded climber, or a short detector dropout can all cause spurious alerts.

AID instead uses a state machine that requires sustained, multi-frame evidence before confirming an intrusion event. This means the system must track a person across frames, maintain continuity through brief detector gaps, and confirm entry using lower-body (ankle) evidence. The tradeoff is that this approach is more expensive per track than a simple overlap check.

### Why multistream changes the engineering problem

In single-stream mode, the expensive confirmation path (pose probing, KLT continuity, gated reacquisition) runs only on the tracks that are near the ROI boundary. Most frames have few such tracks.

In multistream mode with N sources, the probability that at least one source has an active boundary-crossing event at any given frame increases roughly proportionally. Under worst-case conditions (all sources running intrusion clips simultaneously), the pipeline must handle the expensive path on most frames across most sources. This changes the cost profile from "mostly idle with occasional spikes" to "sustained heavy load."

This is why multistream is not just a deployment convenience -- it is a system-level validation of whether the confirmation semantics are computationally viable at scale.

### Why boundary hard cases matter more under shared compute

Under shared compute, every model invocation on one source's boundary track is time stolen from another source. Tail latency spikes in the render or decision pass affect all sources simultaneously. The design challenge is to maintain confirmation quality while bounding worst-case per-frame cost.

## 3. Stage 04 architecture

### Pipeline phases

Stage 04 processes each run through five sequential phases:

| Phase | Name | What it does |
|-------|------|--------------|
| 1 | DeepStream tracking export | Runs YOLO11s detection + NvDCF tracking on all sources via DeepStream. Exports per-frame, per-track sidecar CSVs with bounding box data, tracking mode, and metadata. |
| 2 | Sidecar split | Splits the combined multi-source sidecar into per-source files for independent downstream processing. |
| 3 | KLT continuity augmentation | Per-source: runs KLT optical flow to bridge short detector gaps. Produces augmented sidecar rows with proxy geometry, patch coordinates, tracked feature points, and flow vectors. |
| 4 | Decision pass | Per-source: runs the intrusion FSM over the augmented sidecar. Classifies each track-frame pair, runs pose probes for ankle detection when needed, and produces confirmed intrusion events. |
| 5 | Render with boundary reacquire | Reads all source videos and renders a tiled visualization with keypoint overlays, boundary indicators, and gated reacquisition for missing tracks. Also produces per-frame render timing data. |

Phases 3-5 contain the boundary-aware logic. Phase 1 is standard DeepStream inference. Phase 2 is a fast data reshaping step.

### Runner vs. core separation

Stage 04 uses a two-layer execution model:

- **Runner** (`04_06_ds_multistream_16ch_runner.py`): orchestrates input preparation, configuration templating, experiment definitions, and execution status tracking. Manages prepared-input caching so that DeepStream inference results can be reused across experiment variants without re-running Phase 1.
- **Core** (`04_05_ds_multistream_boundary_reacquire.py` + `aidlib/intrusion/decision_fsm.py`): contains the actual pipeline logic -- KLT augmentation, the decision FSM, and render-side recovery.

This separation allows the same prepared inputs to be reused across multiple experiments with different optimization flags, making A/B comparisons reproducible without redundant GPU inference.

### Config-template-driven execution

DeepStream pipeline configuration (source URIs, inference model paths, tracker parameters, tiler layout) is generated from templates in `configs/deepstream/16ch/`. The runner resolves source paths, ROI mappings, and experiment-specific flags into a concrete configuration before launching the core pipeline.

### Key code locations

| Component | File | Role |
|-----------|------|------|
| Decision FSM | `aidlib/intrusion/decision_fsm.py` | Three-state FSM (OUT / CANDIDATE / IN_CONFIRMED), pose probing, evidence evaluation, event emission |
| KLT augmentation | `scripts/04_ds_multi_stream/04_05_ds_multistream_boundary_reacquire.py` | Optical flow continuity bridging, lazy grayscale gating, proxy row injection |
| Render + reacquire | Same file as above | Keypoint refresh (H6), missing-track reacquire (H7/H8), per-frame crop gating, tiled output |
| 16ch runner | `scripts/04_ds_multi_stream/04_06_ds_multistream_16ch_runner.py` | Experiment orchestration, prepared-input caching, status artifact production |

### Output artifacts

Each run produces:
- `execution_status.json` -- wall-clock timing, experiment configuration, per-phase metrics, guardrail warnings
- `boundary_reacquire_run_summary.json` -- detailed render-phase statistics: H6/H7 call counts, policy reason counters, crop gating stats, per-frame latency distribution
- Per-source `intrusion_summary.json` -- decision-pass statistics: path classification counts, pose probe timing, confirmed event details
- Per-source `intrusion_events.jsonl` -- one record per state transition
- Tiled overlay video -- visual rendering of all sources with boundary indicators and reacquisition overlays

## 4. Four-source deliverable vs. 16-channel benchmark

### The core deliverable: 4-source multistream

The main Stage 04 deliverable is a four-source simultaneous intrusion pipeline. This is the system described in the main README: four RTSP or file sources processed through detection, tracking, continuity, and event-level decision logic, producing tiled overlay video and per-source intrusion events.

The four-source configuration is the intended operating point for the system. It represents a realistic CCTV monitoring scenario and is where the design tradeoffs (FSM grace periods, cadence intervals, continuity parameters) were tuned.

### The benchmark extension: 16-channel synchronized burst

After the four-source pipeline was working, the system was scaled to 16 simultaneous sources as a stress test and scalability benchmark. This is explicitly a validation exercise, not the primary project identity.

The 16-channel benchmark uses a different operating regime than the four-source deliverable:
- 4x the source count, meaning 4x the tracking state, 4x the potential boundary events, and shared GPU inference at batch-size 16
- All sources contain active intrusion clips, forcing the expensive confirmation path on most frames -- an adversarial workload by design
- `input_read_fps=10` is deliberately chosen to stress the pipeline; a production system would likely adapt read rate to available compute

The benchmark answers a specific engineering question: under worst-case simultaneous boundary load, how does the full confirmation pipeline perform?

## 5. What the 16-channel burst benchmark actually tested

### Benchmark configuration

- **Sources**: 16 synchronized 50-second clips (E01_001 through E01_016), each containing at least one intrusion event near the ROI boundary
- **Input read rate**: 10 fps per source (160 fps aggregate)
- **Detector**: YOLO11s, batch-size 16, FP16, 640x640 input
- **Tracker**: NvDCF
- **Total frames**: 8,015 (501 frames per source on average)

### Why this is a worst-case burst benchmark

In normal CCTV monitoring, most sources most of the time show empty scenes or people far from the ROI. The expensive confirmation path (pose probing, KLT continuity, reacquisition) activates only for tracks near the boundary. Under typical load, the pipeline is mostly idle.

This benchmark deliberately eliminates the idle case. Every source contains an active intrusion within 50 seconds, and many sources have tracks near the boundary simultaneously. This forces the pipeline into sustained heavy-path operation across all channels -- a condition that would be unusual in production but reveals the system's worst-case behavior.

### What it stresses

- **Decision pass pose probing**: with many tracks classified as `rich_pose` across multiple sources, the per-track YOLO pose inference cost accumulates. The decision pass was the single largest model-inference consumer in the baseline.
- **Render-side model calls**: H6 (keypoint refresh for tracked actors) and H7 (reacquisition for missing tracks) run gated model inference per boundary-relevant track. With 16 sources, the per-frame model call count can spike during simultaneous boundary events.
- **KLT continuity**: optical flow computation on every source's frames for tracks near the boundary. The heavy predicate gates this to ~32% of frames per source, but with 16 sources the aggregate CPU cost is significant.
- **Tail latency**: a single expensive frame (many simultaneous model calls) blocks all sources. The benchmark measures both mean and p95/max per-frame render latency to characterize spike behavior.

## 6. Bottleneck breakdown and tuning interpretation

### Phase timing breakdown (baseline, 16 channels)

| Phase | Wall-clock | % of total |
|-------|-----------|------------|
| Decision pass | 33.2s | 32.4% |
| Render + reacquire | 32.4s | 31.5% |
| KLT augmentation | 21.2s | 20.6% |
| DeepStream inference | ~15.9s | 15.5% |
| **Total** | **102.7s** | |

### Where the model inference time goes

| Caller | Calls | Wall-sec | What it does |
|--------|-------|----------|--------------|
| Decision pose probe | 1,496 | 21.9s | YOLO pose inference for ankle detection on candidate/confirmed tracks |
| H6 keypoint refresh | 380 | 5.3s | YOLO pose inference on rendered tracks to refresh keypoint overlay |
| H7 reacquisition | 247 | 3.3s | YOLO pose inference to recover missing tracks after gap |
| **Total** | **2,123** | **~30.5s** | |

The decision-pass pose probe dominates model inference cost: ~1,500 calls at ~14.6ms each. This is because the decision pass probes every `rich_pose` track on every frame with no per-track cadence -- it needs fresh ankle evidence to make correct state transitions.

H6 and H7 are render-side only and have per-track cadence systems (4-7 frame intervals, reuse TTLs, cooldowns). Their call counts are already well-gated.

### What kind of tuning was done

The baseline configuration already includes "Pack A" optimizations that were safe to enable unconditionally:

- **H6 geometry_jump cadence demotion**: previously, a significant bounding-box shift between frames triggered immediate H6 re-probing (bypassing cadence). This was demoted to a halved-cadence trigger, reducing H6 calls substantially without affecting decision semantics (H6 is render-only).
- **Per-frame latency instrumentation**: max, p95, and mean render-frame timing tracked across all frames.

These changes reduced H6 calls from ~950 to ~380 in the baseline and are reflected in all experiment runs.

Beyond Pack A, the benchmark tested three "Pack B" experimental flags, each individually togglable:

- **B1 -- Pose probe reuse**: cache per-track pose results for 1 frame when the track is stable, with mandatory fresh probes on state transitions and near-boundary frames.
- **B2 -- Decision lazy decode**: skip full video frame decode on decision-pass frames where no track needs a pose probe, using `cv2.VideoCapture.grab()` instead of `read()`.
- **B3 -- Render model budget**: hard cap on H6+H7 model calls per video frame to bound tail latency.

### How to interpret the tuning

The tuning was designed around a separation of concerns:

- **Render-side changes** (H6 cadence, render budget) cannot affect confirmed intrusion events because the render phase runs after the decision pass. These are safe to tune aggressively.
- **Decision-side changes** (probe reuse, lazy decode, fast-reject margin) can potentially change event outcomes. These require guardrail validation: same event counts, zero miss counters, and no confirmed-state starvation.

Each experiment was run with the same prepared inputs (cached DeepStream Phase 1 output), same ROI definitions, and same FSM parameters. Only the optimization flags differ.

## 7. Final benchmark interpretation

### Safe headline result

The baseline configuration (Pack A optimizations, all experimental flags off) processed 16 simultaneous channels at approximately **4.9 fps per stream** under worst-case burst conditions, producing 17 confirmed intrusion events with zero guardrail warnings.

This represents the reliable, always-on throughput of the full boundary-aware pipeline at 16-channel scale. No decision-side shortcuts were taken. Every candidate track was fully probed on every eligible frame.

### Best guardrail-clean experimental result

With decision-pass lazy decode enabled (B2), throughput increased to approximately **5.4 fps per stream** -- an 11% improvement over the baseline. The system produced the same 17 confirmed intrusion events as the baseline, with:

- `lazy_decode_miss_count = 0` (no frame was skipped that turned out to need a probe)
- `lazy_decode_skip_count = 6,644` (out of ~8,000 total decision-pass frames, 83% were skipped)
- Zero guardrail warnings

This result is evidence-backed and semantically equivalent to the baseline. The optimization avoids full video frame decode on decision-pass frames where all tracks are classified as `light_out` (far from boundary) or `medium_no_pose` (no pose probe needed). The pre-classification uses the same logic as the main decision path, and the miss counter confirms that the pre-classification never disagreed with the actual probe requirements.

### Result excluded from headline use

The render-side model budget cap (B3, set to 2 calls per frame) achieved similar throughput (~5.3 fps/stream) and notably reduced p95 frame render latency (0.076s vs baseline 0.097s). However, it was excluded from the headline because the budget was too aggressive: 176 frames exceeded the cap, including 83 in `IN_CONFIRMED` state. This means confirmed-state tracks were denied keypoint refresh on those frames -- a guardrail violation (`frame_budget_exceeded_by_state.IN_CONFIRMED > 0`).

While this did not change event counts in this particular run (18 confirmed events, same as the probe-reuse variant), the starvation of confirmed-state tracks is not acceptable as a default production setting. The render budget idea itself is sound, but requires a higher budget threshold to avoid confirmed-state starvation.

### Probe reuse (B1) -- secondary result

Pose probe reuse (TTL=1, with mandatory fresh probes on state transitions and near-boundary frames) achieved ~5.3 fps/stream. It produced 18 confirmed events vs. the baseline's 17 -- the additional event is attributable to a marginal timing shift where slightly faster confirmation enabled a borderline event. This is not a regression, but the event-count difference means it cannot be claimed as semantically identical to the baseline.

## 8. What this means for service design

### Why the benchmark matters

The benchmark demonstrates that the full confirmation pipeline -- FSM state tracking, ankle-level pose evidence, KLT continuity, gated reacquisition -- can run at approximately 5 fps per stream across 16 simultaneous channels under worst-case boundary load on a single GPU. This is not real-time at 10 fps, but it is operationally useful.

### Why "10 fps everywhere" is the wrong mental model

A production CCTV intrusion service does not need 10 fps confirmation on every source at every moment. Most sources most of the time show empty scenes or people far from the ROI boundary. The expensive confirmation path activates only when a track enters the boundary band.

The benchmark deliberately forces worst-case conditions: all 16 sources simultaneously running intrusion clips. In practice, simultaneous boundary events across all channels are rare. A service-oriented deployment would:

- Run detection and tracking at full frame rate on all sources (DeepStream handles this efficiently at batch-16)
- Activate the heavy confirmation path only for tracks that enter the boundary band
- Use the event-level FSM's grace period (30 frames) to absorb brief confirmation delays without losing events

Under this model, the benchmark's ~5 fps per stream under worst-case burst is a conservative lower bound on real-world throughput.

### Light path vs. heavy verifier

The system is designed around a two-tier cost model:

- **Light path**: detection, tracking, fast-reject for tracks far from boundary. This is cheap and runs on every frame for every source.
- **Heavy verifier**: pose probing, KLT continuity, reacquisition. This is expensive and activates only when a track is near the boundary and may be entering the ROI.

The benchmark stresses the heavy verifier by making it active on all sources simultaneously. The tuning work (lazy decode, probe reuse, cadence gating) is about making the heavy verifier more efficient without weakening its confirmation guarantees. The guardrail framework ensures that efficiency gains do not come at the cost of silently skipping meaningful work.

## 9. See also

- `README.md` -- project overview, design rationale, pipeline stage summary
- `docs/stage03/README.md` -- Stage 03 single-stream semantics: the continuity-vs-truth problem and how the FSM confirmation logic was developed
- `outputs/analysis/exp_compare_latest.md` -- compact comparison table across the 16-channel benchmark experiment variants

Representative run artifacts:
- `outputs/04_deepstream/20260324_baseline_232545/multistream16_runner/execution_status.json` -- baseline (Pack A) benchmark run
- `outputs/04_deepstream/20260324_b2_232932/multistream16_runner/execution_status.json` -- best guardrail-clean result (lazy decode)
- `outputs/04_deepstream/20260324_b3_light_233300/multistream16_runner/execution_status.json` -- render budget variant with guardrail warning
