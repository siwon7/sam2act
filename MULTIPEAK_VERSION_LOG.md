# Multi-Peak Implementation Version Log

## TGM-VLA 원본 방식 (참조)

### Intra-Mixup (rate=0.4, cross 아닌 경우에만)
1. SE(3) aug 전에 `ori_pc`, `ori_action_trans_con` 저장
2. SE(3) aug 적용 → `pc`(augmented), `action_trans_con`(augmented)
3. `ori_pc`도 별도로 `place_pc_in_cube` → `ori_wpt_local`
4. **Point cloud concat**: `pc[i] = cat([pc[i], ori_pc[i]])` — **모델 입력이 2배**
5. **Label 합산**: `mixup_hm = hm(augmented_target) + hm(original_target)` — **정규화 안 함**
6. 모델이 2개의 point cloud를 동시에 보면서 2-peak label로 학습

### Cross-Mixup (rate=0.5)
1. 다른 task의 sample에서 point cloud을 가져옴
2. **Point cloud concat**: `pc[i] = cat([pc[i], pc[random_task]])` — **다른 task의 scene 합침**
3. Label은 원래 것 유지 (다른 task의 target은 무시)
4. 목적: scene이 복잡해져도 자기 target을 찾도록 robustness 학습

### 핵심 메커니즘
- **Point cloud concat이 필수**: 모델이 2개의 scene을 동시에 보기 때문에 2-peak이 의미 있음
- 정규화 안 함 → multi-peak label의 gradient가 single-peak과 동등

---

## v1: 좌표 변환 버그 (2026-04-28)
- **exp_name**: `stage1_multipeak_putblock`
- **문제**: `alt_target_positions`에 SE(3) aug + `place_pc_in_cube` 미적용
  - Alt target이 raw world 좌표 → augmented view에서 엉뚱한 2D 위치
  - Multi-peak label이 noise → 모델 무시 → collapse
- **결과**: Epoch 17에서 peak1/peak2 ratio > 100만, 완전한 single-peak
- **커밋**: `b7b8e79`, `c001276`

## v2: 좌표 수정 (2026-04-29)
- **exp_name**: `stage1_multipeak_putblock_v2_coordfix`
- **수정**: alt target에 동일한 SE(3) translation shift + `place_pc_in_cube` 적용
- **검증**: label 생성 시 p1/p2=1.05 (정상 2-peak) 확인
- **문제**: Multi-peak 비율 25% (구조적 ambiguity만) → 75% single-peak이 dominant → collapse
- **결과**: Epoch 14에서도 peak1/peak2 ratio > 28만
- **원인**: 비율 부족 + point cloud concat 없음

## v3: SE(3) intra-mixup 추가 (2026-04-30)
- **exp_name**: `stage1_multipeak_putblock_v3_se3mixup`
- **수정**: non-ambiguous sample에 40% 확률로 SE(3) intra-mixup 적용
  - ori_wpt_local(aug 전 target)을 2번째 peak으로 추가
  - 비율 25% → ~55%로 증가
- **문제**: **point cloud concat 없이** label만 2-peak으로 줌
  - 모델은 augmented point cloud 하나만 봄 → 2번째 peak에 대한 시각적 근거 없음
  - Epoch 2에서 ratio 2000~7000으로 개선되었으나, epoch 8에서 다시 collapse
- **결과**: Epoch 8에서 peak2 prob=0.000000
- **원인**: TGM-VLA의 핵심인 **point cloud concatenation**이 빠져 있었음
- **커밋**: `ec9e686`

## v4: Point Cloud Concat 추가 (2026-04-30, 현재 학습 중)
- **exp_name**: `stage1_multipeak_putblock_v4_pcconcat`
- **수정**: TGM-VLA처럼 intra-mixup 시 point cloud + img_feat concat
  - `pc[i] = cat([pc[i], ori_pc_cube[i]])` — 모델이 2개의 scene을 동시에 봄
  - `img_feat[i] = cat([img_feat[i], ori_img_feat_moved[i]])` — point feature도 함께
  - `se3_mixup_flags`를 update()에서 한 번만 결정, get_action_trans()에 전달 (동기화)
- **비율**: 구조적 25% + SE(3) mixup 30% = ~55% multi-peak
- **커밋**: `7254b92`
- **학습**: GPU 0-3, 90 epochs

## v5: Partial-View Multi-Peak + Rate 조정 (2026-04-30, 현재 학습 중)
- **exp_name**: `stage1_multipeak_putblock_v5_partialview`
- **수정**:
  1. Multi-peak label을 5개 view 중 3개(top, front, back)에만 적용
     - 나머지 2개(left, right)는 single-peak anchor로 유지
     - TGM-VLA의 `action_trans[i][:, :3]`과 동일한 전략
  2. SE(3) mixup rate: 0.4 → 0.1
     - 총 multi-peak 비율: structural(25%) + SE(3)(~7%) ≈ 32%
     - TGM-VLA effective rate ~20%에 근접
- **v4 실패 원인**: 5개 view 모두 multi-peak → 모든 방향에서 ambiguous → collapse
- **v5 해결**: 2개 view가 항상 정답만 가리킴 → 학습 안정성 + multi-peak 유지
- **커밋**: `87f83e3`
- **학습**: GPU 0-3, 90 epochs

### 버전별 비율 요약

| 버전 | Multi-peak views | SE(3) rate | Total MP ratio | 결과 |
|------|:---:|:---:|:---:|:---:|
| v1 | 5/5 (버그) | 0% | 25% | collapse (좌표 버그) |
| v2 | 5/5 | 0% | 25% | collapse (비율 부족) |
| v3 | 5/5 | 40% | ~55% | collapse (pc concat 없음) |
| v4 | 5/5 | 40% | ~55% | collapse (anchor 없음) |
| v5 | 3/3(=all) | 10% | ~32% | collapse (anchor 0개, nc=3) |
| **v5.1** | **2/3** | **10%** | **~32%** | **학습 중** |
| v5.2 | 3/6 (stage1 only) | 10% | ~32% | ep6: KF0 p1/p2=14! → ep13: collapse |
| v6 (OOM) | 3/6 | 10% | ~32% | 3-task bs=24, OOM |
| **v6.1** | **3/6** | **10%** | **~32%** | **3-task bs=16, KF4 p1/p2=2.0! 학습 중** |
| TGM-VLA | 3/5 | 20% | ~20% | 작동 확인 |

---

### TGM-VLA 대비 차이점 정리

| 항목 | TGM-VLA | v3 (현재) | v4 (TODO) |
|------|---------|-----------|-----------|
| Point cloud concat (intra) | O (`cat([pc, ori_pc])`) | **X** | O |
| Point cloud concat (cross) | O (`cat([pc, pc_other_task])`) | X | 선택적 |
| Label 합산 (정규화 안 함) | O | O | O |
| Intra-mixup rate | 0.4 | 0.4 | 0.4 |
| Cross-mixup rate | 0.5 | 0 | 선택적 |
| 구조적 multi-peak (revisit) | X | O | O |
| 구조적 multi-peak (variation) | X | O | O |
| ori_pc 별도 place_pc_in_cube | O | O | O |
| ori_wpt_local 계산 | O | O | O |

### 우리만의 contribution vs TGM-VLA 차용

| 항목 | 출처 | 설명 |
|------|------|------|
| **Revisit ambiguity detection** | 우리 | 에피소드 내 재방문 분기 자동 탐지 |
| **Variation ambiguity detection** | 우리 | 에피소드 간 variation target 차이 탐지 |
| **Scene-aware clustering** | 우리 | EE+grip+object 기반 truly ambiguous 판별 |
| SE(3) intra-mixup | TGM-VLA 차용 | augmented + original point cloud concat |
| Cross-task mixup | TGM-VLA 차용 | 다른 task point cloud concat |
| Label 합산 (정규화 안 함) | TGM-VLA 차용 | hm + ori_hm 그대로 |
