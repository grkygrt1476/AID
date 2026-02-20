# 01_intrusion MVP 스코어 파이프라인

## 개요
이 MVP는 **추적(Tracking) 없이** 프레임 단위 탐지 박스(`bbox`)만으로 ROI 침입 점수를 계산합니다.  
ROI별로 `bbox -> factor(3개) -> score(0~1)`를 계산하고, 프레임마다 최고 점수 박스(argmax) 1개만 대표로 선택한 뒤 ROI 단위 FSM(`OUT/CAND/IN`)을 갱신합니다.

## 파일 구성
- `aidlib/intrusion/roi.py`
  - 역할: ROI 폴리곤 로드 및 캐시 생성
  - 핵심 함수/클래스: `load_roi_polygon`, `build_roi_mask`, `build_signed_distance`, `build_integral`, `build_roi_bottom_y`, `RoiCache`
- `aidlib/intrusion/features.py`
  - 역할: bbox와 ROI 캐시로 팩터 계산
  - 핵심 함수/클래스: `FeatureConfig`, `compute_bbox_factors`
- `aidlib/intrusion/score.py`
  - 역할: 팩터 가중합으로 최종 스코어 계산
  - 핵심 함수/클래스: `ScoreWeights`, `compute_score`
- `aidlib/intrusion/fsm.py`
  - 역할: ROI 단위 상태머신(OUT/CAND/IN)
  - 핵심 함수/클래스: `FsmParams`, `RoiFsm`, `FsmSnapshot`
- `aidlib/intrusion/viz.py`
  - 역할: ROI 폴리곤/대표 bbox/디버그 텍스트 오버레이
  - 핵심 함수/클래스: `draw_roi_view`
- `aidlib/intrusion/io.py`
  - 역할: 출력 폴더/로그/메타/JSONL/비디오 writer 관리
  - 핵심 함수/클래스: `IOContext`, `init_io`, `load_yaml_config`, `save_yaml`, `write_json`, `append_jsonl`
- `scripts/01_intrusion/01_01_score_mvp.py`
  - 역할: CLI 엔트리, dry-run/real mode 실행
- `configs/intrusion/mvp_v1.yaml`
  - 역할: 스코어/FSM 하이퍼파라미터 설정

## 실행 방법
### 1) Dry-run (권장 시작점)
```bash
python scripts/01_intrusion/01_01_score_mvp.py \
  --dry_run \
  --cfg configs/intrusion/mvp_v1.yaml \
  --out_root outputs \
  --log_root outputs/logs \
  --out_base DRY_MVP
```

### 2) Dry-run 커스텀(해상도/프레임/ROI 수)
```bash
python scripts/01_intrusion/01_01_score_mvp.py \
  --dry_run \
  --dry_wh 640x360 \
  --dry_frames 60 \
  --dry_fps 10 \
  --dry_rois 2 \
  --cfg configs/intrusion/mvp_v1.yaml \
  --out_base DRY_MVP_MULTI
```

### 3) Real mode (video + roi + det_jsonl)
```bash
python scripts/01_intrusion/01_01_score_mvp.py \
  --video data/videos/E01_007.mp4 \
  --roi_json configs/rois/E01_007/roi_area01_v1.json \
  --det_jsonl outputs/some_detector.jsonl \
  --cfg configs/intrusion/mvp_v1.yaml \
  --out_base E01_007_MVP
```

## Real mode 입력 형식 (det_jsonl)
프레임별 JSON line 형식입니다.

공통 bbox 사용 예:
```json
{"frame_idx": 0, "bboxes": [[10,20,80,160,0.9], [200,30,260,180,0.8]]}
```

ROI별 bbox 분리 예:
```json
{"frame_idx": 0, "rois": {"area01": [[10,20,80,160,0.9]], "area02": [[200,30,260,180,0.8]]}}
```

## 산출물
실행 시 아래가 생성됩니다.

```text
outputs/01_intrusion/<RUN_TS>/<out_base>/
  <out_base>_mvp.mp4
  scores.jsonl
  meta.json
  params_used.yaml

outputs/logs/01_intrusion/
  <out_base>_<RUN_TS>.cmd.txt
  <out_base>_<RUN_TS>.log
```

- `scores.jsonl`: 프레임별 ROI 상태/점수/대표 bbox/FSM 카운터
- `meta.json`: 실행 메타(W,H,fps,nframes,cmd,cfg_path,roi_paths,git_commit 등)
- `params_used.yaml`: 로드된 설정 파일 덤프

## 튜닝 포인트
- 스코어 가중치: `configs/intrusion/mvp_v1.yaml > score.weights`
- 팩터 정규화/밴드: `score.norms`, `score.band`
- 상태 전이 임계/카운터/유예: `fsm.cand_thr`, `fsm.in_thr`, `fsm.out_thr`, `fsm.enter_n`, `fsm.in_n`, `fsm.exit_n`, `fsm.grace_sec`
