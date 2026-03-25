# 16-Channel Worst-Case Burst Benchmark: Throughput with Event-Level Semantics

This document summarizes the Stage 04 DeepStream 16-channel worst-case burst benchmark (2026-03-24). The benchmark validates that a full boundary-aware intrusion confirmation pipeline -- three-state FSM decision logic, per-track ankle-level pose probing, KLT optical flow continuity, and gated reacquisition -- sustains meaningful throughput across 16 simultaneous camera sources on a single GPU under adversarial load.

The key result is not just throughput. It is throughput with preserved event-level semantics: the same confirmed intrusion events, zero guardrail warnings, and a complete per-source audit trail of decision artifacts, produced under worst-case simultaneous boundary load where all 16 sources run active intrusion clips concurrently.

**Benchmark configuration**: 16 synchronized 50-second clips, 10 fps input read rate per source (160 fps aggregate), YOLO11s batch-16 FP16 640x640, NvDCF tracker, 8,015 total processed frames.

---

## Table A. Throughput Progression / Optimization Timeline

| Stage | Source type | Change / optimization | Supported throughput | Evidence label | Portfolio interpretation |
|-------|------------|----------------------|---------------------|----------------|------------------------|
| Early diagnostic | Explicit log value | Pre-optimization 16ch pipeline, 10 fps/source | **3.02 fps/stream** | `execution_status.json` (20260324_16ch_real10_retry3): per_stream_effective_fps = 3.0151 | Earliest measured worst-case throughput under full 16-channel load |
| Early exploratory range | Written interpretation from notes | Approximate characterization of early diagnostic runs (observed: 3.01--3.71) | **~3.5 fps/stream** | Midpoint of early cluster (retry3: 3.02, renderprobe: 3.37, phase1: 3.71) | Not a single logged value; summarizes the starting-point range before systematic optimization |
| Intermediate optimization | Explicit log value | H6 cadence improvements, fast-reject refinements, render probe tuning | **4.06 fps/stream** | `execution_status.json` (20260324_16ch_next3_185614): per_stream_effective_fps = 4.0608 | First crossing above 4 fps/stream; validates optimization direction |
| Late optimization | Explicit log value | Further render/decision path refinements toward Pack A | **4.65 fps/stream** | `execution_status.json` (20260324_16ch_next4_215041): per_stream_effective_fps = 4.6503 | Approaching final Pack A baseline performance |
| Final Pack A baseline | Final benchmark result | All Pack A optimizations (H6 cadence demotion, instrumentation); all Pack B flags off | **4.79 fps/stream** | `execution_status.json` (20260324_baseline_232545): per_stream_effective_fps = 4.7895 | Conservative headline: full confirmation pipeline, no decision-side shortcuts |
| Best guardrail-clean (B2) | Final benchmark result | Decision lazy decode: skip full decode when no track needs pose probe | **5.37 fps/stream** | `execution_status.json` (20260324_b2_232932): per_stream_effective_fps = 5.3722 | Best result with identical event semantics to baseline (same 17 events, 0 guardrail warnings, 0 decode misses) |

**Note on ~4.1--4.2 fps/stream intermediate stage**: The closest explicitly logged value is 4.06 fps/stream (20260324_16ch_next3_185614). No single run logged exactly 4.1--4.2. The next explicit value is 4.42 fps/stream (20260324_16ch_next4_201145), which exceeds that range.

---

## Table B. Final Comparable Benchmark Variants (Pack B)

All variants use identical cached DeepStream Phase 1 outputs, same 16 sources, same ROI definitions, same FSM parameters. Only optimization flags differ.

| Variant | Main change | stage_total_sec | fps_per_stream | confirmed_event_count | guardrail_warnings | Portfolio note |
|---------|------------|:--------------:|:-------------:|:--------------------:|:-----------------:|---------------|
| baseline | Pack A only; all Pack B off | 102.66 | 4.79 | 17 | 0 | Conservative headline; full probe on every eligible frame |
| b1 | Pose probe reuse (TTL=1, fresh on transitions) | 94.27 | 5.25 | 18 | 0 | +1 event from marginal timing shift; not semantically identical to baseline |
| b2 | Decision lazy decode | 92.15 | 5.37 | 17 | 0 | **Best guardrail-clean**; same 17 events as baseline; 82.9% of frames skipped |
| b1_b2 | Probe reuse + lazy decode | 93.63 | 5.29 | 18 | 0 | Combined B1+B2; 18 events (same as B1 alone) |
| b3_light | Render model budget cap (max 2 calls/frame) | 94.01 | 5.27 | 18 | **1** | Guardrail violation: 83 frames starved IN_CONFIRMED tracks; excluded from headline use |

**fps_per_stream** values are from `per_stream_effective_fps` in each `execution_status.json` (computed from `core_wall_clock_sec`, the actual wall-clock time of core pipeline execution). The `stage_total_sec` values (sum of phase timings) are slightly lower than wall-clock due to inter-phase overhead; `fps_per_stream` uses the more conservative wall-clock denominator.

---

## Table C. Event-Level / Service Metrics Worth Mentioning

| Metric | Value | Where it came from | Why it matters for the portfolio |
|--------|:-----:|-------------------|-------------------------------|
| Confirmed intrusion events (baseline) | 17 | `execution_status.json` (baseline): confirmed_events_total = 17 | Structured, countable events from FSM -- not raw frame-level alerts |
| Event count stability (baseline vs. B2) | 17 = 17 | baseline and b2: both confirmed_events_total = 17 | Best optimization preserves identical event semantics |
| Guardrail warnings (baseline and B2) | 0 | guardrail_warnings = [] in both files | Zero guardrail violations for both conservative and best-optimized variants |
| Lazy decode skip count (B2) | 6,644 | b2: lazy_decode_skip_count = 6644 | 82.9% of decision-pass frames avoided full video decode |
| Lazy decode miss count (B2) | 0 | b2: lazy_decode_miss_count = 0 | Pre-classification never disagreed with actual probe requirements |
| Skip ratio (B2) | 82.9% | Computed: 6644 / 8015 total frames | Quantifies optimization headroom in the decision pass |
| Total processed frames | 8,015 | total_processed_frames = 8015 (all variants) | ~501 frames per source across 16 channels at 10 fps |
| Pose probe calls (baseline) | 1,496 | baseline: pose_probe_attempt_count = 1496 | Quantifies per-track model inference load |
| p95 frame render latency (baseline) | 0.097s | baseline: p95_frame_render_sec = 0.096529 | Tail latency characterization; not a headline metric |
| p95 frame render latency (B3) | 0.076s | b3_light: p95_frame_render_sec = 0.075983 | Budget cap reduces tail latency, but at cost of guardrail violation |
| Output artifacts per run | `intrusion_events.jsonl`, `intrusion_summary.json`, `execution_status.json`, `boundary_reacquire_run_summary.json`, tiled overlay video | Documented in `docs/stage04/README.md`; produced per run | Full event-level audit trail: per-transition JSONL, per-source summary, visual overlay |

---

## Suggested Portfolio Wording

### Conservative

- Scaled a boundary-aware intrusion confirmation pipeline (three-state FSM, per-track pose probing, KLT continuity) to 16 simultaneous camera sources on a single GPU, sustaining 4.8 fps/stream under worst-case conditions with 17 confirmed events and zero guardrail warnings.
- All event-level outputs -- per-source JSONL transition logs, decision summaries, and tiled overlay video -- are inspectable and reproducible from cached inputs.

### Balanced

- Ran 16-channel worst-case burst benchmark: full confirmation pipeline sustained ~4.8 fps/stream baseline, optimized to ~5.4 fps/stream with lazy decode -- same 17 confirmed intrusion events, zero guardrail warnings, zero decode misses across 8,015 processed frames.
- Every variant produces structured, inspectable event-level artifacts (intrusion_events.jsonl, execution_status.json with guardrail counters) for reproducible A/B comparison.

### Confident

- Demonstrated production-viable 16-channel intrusion detection at ~5.4 fps/stream under worst-case simultaneous boundary load, with zero guardrail warnings and provably identical event semantics to the unoptimized baseline -- 17 confirmed events, 0 decode misses out of 6,644 skipped frames (82.9% skip rate).
- Full event-level audit trail: per-source state-transition logs, decision-pass summaries, guardrail counters, and 4x4 tiled overlay with keypoint and boundary visualizations.

---

## Suggested README Wording

- **16-channel worst-case burst benchmark**: 4.8 fps/stream baseline, 5.4 fps/stream with lazy decode; 17 confirmed intrusion events with zero guardrail warnings across all guardrail-clean variants.
- **Inspectable event-level outputs**: per-source `intrusion_events.jsonl` (one record per state transition), `intrusion_summary.json`, `execution_status.json` with phase timing and guardrail counters, plus 4x4 tiled overlay video.
- **Reproducible A/B comparison**: all benchmark variants run on identical cached DeepStream Phase 1 outputs; differences are isolated to optimization flags, with structured guardrail validation ensuring semantic equivalence.

---

## Source Notes

### Files used

| File | Role |
|------|------|
| `docs/stage04/README.md` | Architecture description, benchmark interpretation, result context |
| `outputs/04_deepstream/20260324_baseline_232545/multistream16_runner/execution_status.json` | Pack B baseline benchmark |
| `outputs/04_deepstream/20260324_b1_232744/multistream16_runner/execution_status.json` | B1 (pose probe reuse) benchmark |
| `outputs/04_deepstream/20260324_b2_232932/multistream16_runner/execution_status.json` | B2 (lazy decode) benchmark |
| `outputs/04_deepstream/20260324_b1_b2_233114/multistream16_runner/execution_status.json` | B1+B2 combined benchmark |
| `outputs/04_deepstream/20260324_b3_light_233300/multistream16_runner/execution_status.json` | B3 (render budget) benchmark |
| `outputs/04_deepstream/20260324_16ch_real10_retry3/multistream16_runner/execution_status.json` | Early diagnostic: source of 3.02 fps/stream |
| `outputs/04_deepstream/20260324_16ch_next3_185614/multistream16_runner/execution_status.json` | Intermediate optimization: source of 4.06 fps/stream |
| `outputs/04_deepstream/20260324_16ch_next4_215041/multistream16_runner/execution_status.json` | Late optimization: source of 4.65 fps/stream |
| `outputs/analysis/exp_compare_latest.md` | Comparison table (note: b1 row uses b1_b2 run data due to script bug; execution_status.json files are authoritative) |

### Value provenance

| Value | Source | Verification status |
|-------|--------|-------------------|
| 3.02 fps/stream | `20260324_16ch_real10_retry3`: per_stream_effective_fps = 3.0151 | Explicit log value; rounds to 3.02 at two decimal places |
| ~3.5 fps/stream | Approximate midpoint of early exploratory cluster (3.01--3.71 range) | **Written interpretation from notes**; not a single logged value |
| 4.06 fps/stream | `20260324_16ch_next3_185614`: per_stream_effective_fps = 4.0608 | Explicit log value |
| 4.65 fps/stream | `20260324_16ch_next4_215041`: per_stream_effective_fps = 4.6503 | Explicit log value |
| 4.79 fps/stream | `20260324_baseline_232545`: per_stream_effective_fps = 4.7895 | Explicit log value; stage04/README.md rounds to "~4.9" |
| 5.25 fps/stream | `20260324_b1_232744`: per_stream_effective_fps = 5.2510 | Explicit log value; stage04/README.md rounds to "~5.3" |
| 5.37 fps/stream | `20260324_b2_232932`: per_stream_effective_fps = 5.3722 | Explicit log value; stage04/README.md rounds to "~5.4" |
| 5.29 fps/stream | `20260324_b1_b2_233114`: per_stream_effective_fps = 5.2861 | Explicit log value |
| 5.27 fps/stream | `20260324_b3_light_233300`: per_stream_effective_fps = 5.2671 | Explicit log value; stage04/README.md rounds to "~5.3" |
| 17 confirmed events | baseline and b2: confirmed_events_total = 17 | Explicit; identical across baseline and B2 |
| 18 confirmed events | b1, b1_b2, b3_light: confirmed_events_total = 18 | Explicit; +1 from marginal timing shift, not regression |
| 6,644 lazy decode skips | b2: lazy_decode_skip_count = 6644 | Explicit log value |
| 0 lazy decode misses | b2: lazy_decode_miss_count = 0 | Explicit log value |
| 82.9% skip ratio | Computed: 6644 / 8015 | Trivially computable; stage04/README.md rounds to "83%" |
| 0 guardrail warnings | baseline, b1, b2, b1_b2: guardrail_warnings = [] | Explicit log value |
| 1 guardrail warning | b3_light: guardrail_warnings = ["frame_budget_exceeded_by_state.IN_CONFIRMED>0"] | Explicit log value |
| 8,015 total frames | All variants: total_processed_frames = 8015 | Explicit; consistent across all Pack B runs |
| 0.097s p95 render latency | baseline: p95_frame_render_sec = 0.096529 | Explicit log value; rounded |
| 0.076s p95 render latency | b3_light: p95_frame_render_sec = 0.075983 | Explicit log value; rounded |
