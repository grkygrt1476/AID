# AID Baseline — Intrusion (Ankle-only + Zone FSM)

## Purpose
Security PoC baseline: decide intrusion using **ankle keypoints only** and a **single Zone-level FSM**.
No ghost logic, no track-based decision dependency.

## Definition (Baseline scope)
- Evidence: `L/R ankle` keypoint
- Signal:
  - `ankle_valid_any`: any person has ankle detected
  - `inside_any`: any ankle point is inside ROI polygon
  - `evidence_type = "ankle"`
- FSM: `OUT -> CAND -> IN -> OUT`
  - defaults: `dwell_s=1.0`, `grace_s=2.0`, `enter_n=3`, `exit_n=5`
- Single Zone incident (not per-actor / not per-track)

## Entrypoint
`scripts/run_aid_baseline_intrusion.py`

## Run (example)
```bash
.venv/bin/python scripts/run_aid_baseline_intrusion.py \
  --video_id E01_007 \
  --roi_json configs/rois/E01_007/roi_area01_v1.json \
  --pose_model weights/person/<POSE_WEIGHTS>.pt \
  --device cuda:0 \
  --dwell_s 1.0 --grace_s 2.0 --enter_n 3 --exit_n 5

## Outputs (aidlib/run_utils convention)

- **Command / Logs**
  - `outputs/logs/02_intrusion/`
    - `*_cmd.txt`
    - `*_log.txt` (or `*.log`)

- **Run Artifacts (per run)**
  - `outputs/02_intrusion/<RUN_TS>/<video_id>/`
    - `overlay.mp4`
    - `events.json`
    - `run_meta.json`

- **Notes**
  - `<RUN_TS>` is generated per execution.
  - `run_meta.json` should include at least: `git_commit`, `input`, `roi_version`, `fps_eff`, `params`.
