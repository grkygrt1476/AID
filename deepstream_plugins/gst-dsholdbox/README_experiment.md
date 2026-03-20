# README_0310_ds_plugin

## What this is

DeepStream 기반 AID tracker pipeline에서, detector/tracker의 짧은 miss 구간 때문에 bbox가 깜빡이는 문제를 줄이기 위해
post-tracker / pre-OSD 위치에 경량 custom plugin(`gst-dsholdbox`)을 추가한 실험 기록이다.

핵심 목적은 추론 성능 향상이 아니라, operator-facing 화면에서 bbox flicker를 줄이는 것이다.

---

## Why this was added

NvDCF와 inactive/buffered output 설정만으로는
사람 ID는 유지되더라도 bbox가 순간적으로 꺼졌다 다시 나타나는 구간이 남았다.

특히 담을 넘는 동작처럼
- 자세 변화가 크고
- 일부가 가려지고
- detector bbox가 약해지는 구간에서

시각적으로 화면이 지저분하게 느껴졌다.

그래서 detector/tracker의 짧은 miss를 완전히 없애기보다,
마지막 bbox를 짧게 유지해서 화면 flicker를 줄이는 방향으로 접근했다.

---

## What was implemented

DeepStream sample plugin(`gst-dsexample`)을 복사해 `gst-dsholdbox` 기반으로 수정했다.

현재 plugin은 tracker 뒤, OSD 앞에서 object metadata를 읽고,
각 track의 마지막 bbox와 마지막 관측 시각을 캐시에 저장한다.

현재 프레임에 실제 bbox가 없는 경우에도,
같은 track이 최근 0.3초 이내에 관측되었다면
마지막 bbox를 display meta로 다시 그리도록 구현했다.

중요하게, 이 hold box는
- intrusion 판단용이 아니라
- 화면 표시용 보조 레이어로만 사용한다.

즉 event logic이나 intrusion entry 기준은 바꾸지 않고,
OSD 품질만 보정하는 구조다.

---

## Current behavior

- 현재 프레임에 실제 track이 있으면 hold box는 그리지 않는다.
- 현재 프레임에 실제 track이 없고, 마지막 관측 시각이 0.3초 이내면 주황색 bbox를 추가로 그린다.
- 0.3초가 지나면 hold box를 제거한다.

이 규칙으로 reacquire 시 중복 bbox가 뜨는 문제를 피하려고 했다.

---

## Files

- `deepstream_plugins/gst-dsholdbox/gstdsholdbox.h`
- `deepstream_plugins/gst-dsholdbox/gstdsholdbox.cpp`
- `deepstream_plugins/gst-dsholdbox/Makefile`

관련 DeepStream tracker config 실험 파일:
- `configs/deepstream/config_tracker_NvDCF_viz.yml`
- `configs/deepstream/ds_yolo11_tracker_nvdcf.txt`

---

## What worked

- DeepStream custom plugin을 별도 `.so`로 빌드하고 pipeline에 삽입하는 것까지 확인했다.
- `transform_ip()`에서 tracker 결과 metadata를 읽는 것까지 확인했다.
- hold-box 방식으로 짧은 miss 구간의 bbox flicker가 일부 줄어드는 것을 확인했다.

---

## Remaining limitation

이 방식은 짧은 miss 구간의 화면 품질 개선에는 도움이 되지만,
사람이 담에 많이 가려지거나 target size가 작아져 detector 자체가 약해지는 경우까지 해결하진 못한다.

즉 현재 한계는
- 표시 문제 일부 개선
- detector small/occlusion 한계는 여전히 남음

으로 정리할 수 있다.

---

## Scope boundary

이번 plugin은 아래 범위로 제한했다.

- short-gap bbox hold
- display meta 기반 OSD 보조
- intrusion logic unchanged

아직 하지 않은 것:
- point tracking / KLT 기반 bbox propagation
- bbox size/shape prediction
- object meta 재주입
- intrusion evidence 자체 변경

---

## Why this matters

이번 작업은 "새 모델을 추가했다"가 아니라,
DeepStream 기반 운영형 비전 파이프라인에서
tracker 한계와 operator-facing UI 품질 문제를 분리해서 다룬 사례에 가깝다.

정리하면:

- detection/tracking = 본체
- hold box plugin = 짧은 miss에 대한 표시 보조기

라는 역할 분리를 실제로 구현해본 셈이다.

---

## Next candidates

다음 후보는 두 가지다.

1. 현재 hold-box plugin을 유지한 채, detector/tracker 쪽 파라미터와 함께 실제 개선 폭을 더 비교
2. 필요하면 2~3프레임 정도의 point tracking 보조를 추가해, 고정 hold보다 조금 더 자연스럽게 bbox를 유지

다만 현재 단계에서는
point tracking 없이도 "DeepStream plugin을 추가해 bbox flicker를 줄여봤다"는 수준으로 정리하는 것이 적절하다.