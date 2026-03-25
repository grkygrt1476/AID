# Stage 03 -- Single-Stream DeepStream Proving Stage

Stage 03 is the single-stream DeepStream proving stage for AID.
It sits between the earlier tracking / continuity experiments and the later Stage 04 multistream pipeline.

This stage was not the final scaling target.
Its job was to make the intrusion semantics technically honest on one stream first, then carry that clarified design into multistream validation.

The main project overview in [`README.md`](../../README.md) stays intentionally service-oriented.
This document is narrower and more technical: it explains what Stage 03 had to prove, what became fixed here, and why Stage 04 came next.

## 1. What Stage 03 is

Stage 03 is where the project moved from "tracked people near an ROI" to "event-level intrusion decisions with explicit semantics."

In practical terms, Stage 03 combined:

- a DeepStream single-stream detector + tracker path
- continuity-assist experiments carried forward from Stage 02
- an intrusion FSM that emits candidate / confirmed event records instead of only drawing boxes
- overlay and summary artifacts that make the decision process inspectable

The important point is not just that it runs on one stream.
It is that one stream was used as a controlled environment to settle the meaning of a confirmed intrusion before scaling to shared-compute multistream execution.

Relevant entrypoints in this directory reflect that progression:

- `03_01_ds_baseline.py`
- `03_02_ds_klt_assist.py`
- `03_02b_ds_pose_patch_assist.py`
- `03_02c_ds_pose_patch_persistent.py`
- `03_02d_ds_pose_anchor_assist.py`
- `03_03_ds_intrusion_fsm.py`

## 2. Why single-stream proving was necessary

Single-stream proving was necessary because event-level CCTV intrusion detection is not the same problem as per-frame person detection.

A naive system can trigger on any person box that overlaps the ROI.
That is simple, but it is not a strong definition of actual entry.
Near a wall or fence, a box can overlap the ROI while the person has not clearly entered.
At the same time, the stronger cue for real entry -- lower-body or ankle evidence -- is exactly the cue that tends to disappear during hard boundary crossings.

That created a design problem:

- naive ROI overlap was too weak
- ankle-centered confirmation was more meaningful
- but ankle evidence could disappear during partial visibility, occlusion, or short detector dropout

Before moving to multistream, the project needed to answer a more basic question:

what should count as useful continuity support, and what should count as actual confirmed intrusion truth?

Stage 03 exists to answer that question in a controlled setting before the extra complexity of multistream scheduling, shared GPU cost, and 16-channel benchmarking is introduced.

## 3. Continuity vs. truth

This is the key idea of Stage 03.

Continuity aids are useful.
They help the system remain stable when a tracked actor becomes briefly ambiguous near the boundary.
Examples in this repository include KLT bridging, pose-patch persistence, anchor-assisted proxy geometry, and later display / reacquire support.

But continuity is not the same thing as truth.

A continuity layer can say:

- this missing actor is probably the same person as a moment ago
- the geometry probably continued in this direction
- the visual story should stay stable long enough to inspect what happened

It cannot, by itself, redefine what a confirmed intrusion means.

Stage 03 is where the project makes that separation explicit:

- continuity support helps survive dropout and preserve context
- confirmation semantics remain grounded in the chosen event definition

That distinction is visible in both the code and the output artifacts.
The decision FSM keeps explicit states (`OUT`, `CANDIDATE`, `IN_CONFIRMED`) in [`aidlib/intrusion/decision_fsm.py`](../../aidlib/intrusion/decision_fsm.py).
The Stage 03 summaries also record the semantic definitions directly.
For example, [`outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/intrusion_summary.json`](../../outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/intrusion_summary.json) states:

- candidate definition: `overlap_or_(near_roi_boundary_and_moving_toward_roi)_or_klt_upper_head_boundary_continuity`
- confirmed intrusion definition: `ankle_keypoint_confirmed_or_klt_hard_case_confirm`

That is the Stage 03 outcome in one line:

- broad evidence can open or sustain a candidate
- final confirmation remains stricter and more physically meaningful

## 4. Intrusion semantics fixed in Stage 03

Several semantics became stable here and then carried forward into Stage 04.

### Event-level view instead of per-frame alerting

The system does not treat every ROI-overlapping frame as a separate alert.
It tracks per-person state over time and emits event-level transitions.

This is implemented in [`aidlib/intrusion/decision_fsm.py`](../../aidlib/intrusion/decision_fsm.py) and visible in Stage 03 outputs such as:

- [`outputs/03_deepstream/20260317_051150/ev00_ds_intrusionfsm/intrusion_summary.json`](../../outputs/03_deepstream/20260317_051150/ev00_ds_intrusionfsm/intrusion_summary.json)
- [`outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/intrusion_summary.json`](../../outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/intrusion_summary.json)

### FSM-based state transitions

The core FSM states are fixed here:

- `OUT`
- `CANDIDATE`
- `IN_CONFIRMED`

This makes the system easier to reason about than a framewise threshold rule.
The state machine supports stabilization through entry / exit counters and grace behavior rather than relying on one-frame decisions.

### Lower-body / ankle-centered confirmation

Stage 03 fixes the idea that lower-body evidence is the preferred basis for final confirmation.
In [`aidlib/intrusion/decision_fsm.py`](../../aidlib/intrusion/decision_fsm.py), `confirm_requires_ankle` is enabled in the decision parameters, and the pose-probe path explicitly evaluates ankle keypoints relative to the ROI.

This does not mean every hard case is solved only by fresh ankle visibility.
It means the system's preferred truth definition remains ankle-centered, and any continuity-backed hard-case confirmation has to justify itself against that stronger baseline rather than replace it casually.

### Candidate / confirm stabilization

Stage 03 also fixes the idea that candidate opening and confirmed entry must be stabilized over time.
That includes:

- candidate accumulation instead of instant alerting
- explicit confirmation paths
- bounded grace / persistence behavior so brief ambiguity does not immediately destroy event context

These details become even more important later in Stage 04, but the semantic framing is established here first.

## 5. Stage 03 technical approach

The technical role of Stage 03 was to make the semantics observable and testable on one stream.

### DeepStream single-stream pipeline

Stage 03 uses DeepStream as the single-stream execution base so that the later Stage 04 multistream pipeline does not need to reinvent the decision logic.
The single-stream proving stage already exercises:

- detector + tracker integration
- custom GStreamer plugin paths
- output sidecars and overlay generation
- event summary production

### Continuity support layers

The Stage 02 to Stage 03 progression explores several ways of preserving continuity through short ambiguous intervals:

- KLT-based assist
- pose-patch assist
- persistent pose-patch support
- anchor-assisted proxy behavior

The point of these experiments was not to declare a single magic tracker.
It was to learn which forms of continuity support were technically useful and which ones risked drifting away from the actual intrusion truth definition.

### Decision-pass semantics

The FSM layer in [`aidlib/intrusion/decision_fsm.py`](../../aidlib/intrusion/decision_fsm.py) is the center of the Stage 03 story.
It combines:

- per-track state
- ROI-relative geometry
- pose-probe evidence
- continuity-backed context
- explicit event emission

That is where the project stopped thinking only in terms of tracking quality and started thinking in terms of service semantics.

### Explainability-oriented outputs

Stage 03 produces outputs that make the decision process inspectable rather than hidden:

- per-run `intrusion_summary.json`
- per-event `intrusion_events.jsonl`
- overlay videos

Representative artifacts include:

- [`outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/intrusion_summary.json`](../../outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/intrusion_summary.json)
- [`outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/ev00_f1826-2854_50s_intrusion_overlay.mp4`](../../outputs/03_deepstream/20260323_211259/ev00_f1826-2854_50s/ev00_f1826-2854_50s_intrusion_overlay.mp4)

That matters because this stage was about proving what the system meant, not just whether it ran.

## 6. What Stage 03 proved and what it did not prove

### What Stage 03 proved

Stage 03 successfully established:

- the project should be framed as event-level intrusion detection, not per-frame box alerting
- continuity support is valuable, especially near the ROI boundary
- continuity support and confirmation truth must remain separated
- an FSM with explicit candidate and confirmed states is a workable way to express the service semantics
- lower-body / ankle-centered confirmation is a meaningful anchor for final intrusion judgment
- DeepStream integration can support these semantics while still producing inspectable outputs

### What Stage 03 did not prove

Stage 03 did not answer the multistream systems questions.
It did not prove:

- shared-compute behavior across multiple sources
- multistream scheduling tradeoffs
- worst-case simultaneous boundary load
- 4-source or 16-source throughput
- how render-side recovery and decision-side confirmation interact under multistream budget pressure

Those became the next problem precisely because the single-stream semantics were now clear enough to carry forward.

## 7. Why Stage 04 came next

Once Stage 03 settled the semantic framing, Stage 04 became the logical next step.

The question changed from:

"What should count as a confirmed intrusion, and how should continuity help without redefining truth?"

to:

"Can that same semantics survive in a multistream DeepStream service where sources share compute and hard boundary cases can happen at the same time?"

That is why Stage 04 focuses on:

- multistream execution
- boundary-aware continuity and reacquire under shared compute
- runner/core separation
- prepared-input reuse for controlled experiments
- benchmark interpretation rather than only single-clip correctness

If Stage 03 is where the semantics were clarified, Stage 04 is where those semantics were validated as a service-oriented multistream system.

For that next step, see [`docs/stage04/README.md`](../stage04/README.md).

## 8. See also

- [`README.md`](../../README.md) -- project-level overview and service-oriented framing
- [`docs/stage04/README.md`](../stage04/README.md) -- Stage 04 multistream architecture and benchmark interpretation
- [`aidlib/intrusion/decision_fsm.py`](../../aidlib/intrusion/decision_fsm.py) -- decision FSM and confirmation logic
