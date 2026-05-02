# Ambiguity Analysis Revised V2 -- 3D Keyframe 시각화 기반 재분석

> 작성일: 2026-05-01
> 기반: episodewise JSON 10개 에피소드 데이터 + 3D keyframe 시각화 결과

---

## 핵심 발견: 현재 전처리의 구조적 문제

현재 `compute_scene_aware_clusters`는 EE 위치 + gripper 상태 + object 위치 3가지를 모두 일치시켜야 같은 클러스터로 분류한다. 그러나 **모델이 보는 것은 point cloud (3D 장면)**이며, gripper 상태나 object 위치는 point cloud에서 시각적으로 구분 가능하다. 따라서:

- **Intra ambiguity의 정의**: 같은 EE 위치에서 다른 next-target을 가지는 모든 경우 (grip/object 무관)
- **모델이 해결할 수 있는 분기**: grip 상태가 다르거나, object 위치가 다른 경우 -> point cloud에서 구별 가능
- **핵심 통찰**: spatial-only clustering으로 intra 후보를 넓게 잡되, 모델이 학습 중 ratio를 스스로 조정하도록 함

---

## 1. put_block_back (12 KF/episode)

### 1.1 Spatial Clusters (EE distance < 0.04m)

모든 에피소드에서 동일한 클러스터 구조:

| 클러스터 | KF 목록 | Gripper 상태 | 평균 EE 위치 |
|---------|---------|-------------|-------------|
| A | (2, 4, 7, 9) | (0, 1, 1, 0) | (0.2744, 0.176, 0.945) |
| B | (0, 11) | (0, 1) | (0.2746, 0.000, 0.796) |
| C | (1, 10) | (0, 0) | (0.2746, 0.001, 0.946) |
| D | (3, 8) | (1, 0) | (0.2747, 0.177, 0.797) |
| - | (5) | (0) | 에피소드별 상이 |
| - | (6) | (0) | 에피소드별 상이 |

### 1.2 Intra Ambiguity 분석

#### 클러스터 A: (2, 4, 7, 9) -- 가장 복잡한 클러스터

4개 KF 모두 동일한 spatial position에 있으나, **4개의 서로 다른 next-target 방향**을 가짐:

| KF | Grip | Next Target (Ep0) | 의미 |
|----|------|-------------------|------|
| KF2 | 0 (open) | (0.275, 0.177, 0.797) = 내려가서 잡기 | 블록 grasp 위치로 하강 |
| KF4 | 1 (closed) | (0.045, -0.214, 0.914) = 먼 곳으로 이동 | 블록 들고 새 위치로 |
| KF7 | 1 (closed) | (0.274, 0.177, 0.796) = 내려가서 놓기 | 블록 들고 원래 자리로 |
| KF9 | 0 (open) | (0.275, 0.001, 0.947) = y축 이동 | 원래 블록 위치로 복귀 |

**Pairwise 분기 거리:**
- KF2<->KF4: 0.468m (strongly divergent)
- KF2<->KF9: 0.231m (divergent)
- KF4<->KF7: 0.468m (strongly divergent)
- KF4<->KF9: 0.316m (divergent)
- KF7<->KF9: 0.231m (divergent)
- KF2<->KF7: 0.001m (convergent - 거의 같은 방향)

**현재 코드가 잡는 것:** KF4-KF7만 (같은 grip=1, 같은 object position)
**누락:** KF2, KF9 (grip 또는 object position이 달라서 필터링됨)

#### 클러스터 B: (0, 11)
- KF0 (grip=0) -> KF1: 위로 올라감
- KF11은 마지막 KF (next target 없음)
- **분기 없음** (KF11에 next가 없으므로)

#### 클러스터 C: (1, 10)
- KF1 (grip=0) -> KF2: (0.274, 0.175, 0.945) = 블록 위치로 이동
- KF10 (grip=0) -> KF11: (0.275, 0.000, 0.797) = home으로 복귀
- **Next-target 분기: 0.230m** (divergent)
- **현재 코드:** 잡히지 않음 (scene-aware에서 grip=같지만 obj_dist > 0.05)

#### 클러스터 D: (3, 8)
- KF3 (grip=1) -> KF4: (0.274, 0.176, 0.945) = 올라감
- KF8 (grip=0) -> KF9: (0.275, 0.177, 0.946) = 올라감
- **Next-target 분기: ~0.001m** (convergent -- 같은 방향)
- **Intra가 아님** (같은 방향으로 감)

### 1.3 Cross-Episode Ambiguity

| KF | Max Spread | Cross 여부 | 설명 |
|----|-----------|-----------|------|
| KF0 | 0.267m | YES | 에피소드별 블록 시작 위치 상이 (4가지 패치) |
| KF1 | 0.267m | YES | KF0와 동일 패턴 |
| KF5 | 0.202m | YES | 새 블록 위치 (에피소드별 상이) |
| KF6 | 0.202m | YES | KF5와 동일 |
| KF10 | 0.266m | YES | 원래 블록 위치로 복귀 (에피소드별 상이) |
| KF11 | 0.267m | YES | KF10과 동일 |
| KF2-KF9 | <0.002m | NO | 고정 위치 |

**현재 코드가 잡는 것:** KF9만 (cross에서 3개 alt target)
**누락:** KF0, KF1, KF5, KF6, KF10, KF11도 cross 후보

### 1.4 현재 전처리 vs 실제 필요

| 구분 | 현재 Intra | 필요한 Intra | 현재 Cross | 필요한 Cross |
|------|-----------|-------------|-----------|-------------|
| KF | 4,7 | 2,4,7,9 + 1,10 | 9 | 0,1,5,6,10,11 |
| 설명 | grip+obj 필터로 축소 | spatial-only로 확장 | 불완전 | 모든 varying KF |

---

## 2. rearrange_block (11 KF/episode)

### 2.1 Spatial Clusters

모든 에피소드에서 동일:

| 클러스터 | KF 목록 | Gripper 상태 | 평균 EE 위치 |
|---------|---------|-------------|-------------|
| A | (0, 10) | (0, 1) | (0.275, 0.175, 0.796) |
| B | (1, 9) | (0, 0) | (0.275, 0.175, 1.011) |
| - | (2) | (0) | 에피소드별 y값 상이 |
| - | (3) | (1) | 에피소드별 y값 상이 |
| - | (4) | (0) | 에피소드별 상이 |
| - | (5) | (0) | 에피소드별 상이 |
| - | (6) | (1) | 에피소드별 y값 상이 |
| - | (7) | (0) | 에피소드별 y값 상이 |
| - | (8) | (0) | 에피소드별 y값 상이 |

### 2.2 Intra Ambiguity 분석

#### 클러스터 A: (0, 10)
- KF0 (grip=0): 마지막에 KF10이 grip=1이므로 grip 다름
- KF10은 마지막 KF (11번째) -> next target 없음
- **분기 없음** (next 없음)

#### 클러스터 B: (1, 9) -- 핵심 intra 후보
- KF1 (grip=0) -> KF2: 첫 번째 블록 위치로 이동
  - Ep0: (0.281, 0.002, 1.013) = 패치 y~0 (unmoved block #1)
  - Ep1: (0.275, 0.349, 1.013) = 패치 y~0.35 (unmoved block #2)
- KF9 (grip=0) -> KF10: home으로 복귀
  - 모든 에피소드: (0.275, 0.175, 0.797) = home position

**Next-target 분기: ~0.276m** (strongly divergent)
- KF1은 "위에서 블록 찾으러 감", KF9는 "위에서 home으로 내려감"
- **현재 코드:** Intra에서 잡히지 않음 (scene-aware에서 object position이 다름)

### 2.3 Cross-Episode Ambiguity

| KF | Max Spread | Cross 여부 | 설명 |
|----|-----------|-----------|------|
| KF0 | 0.000m | NO | 고정 HOME |
| KF1 | 0.001m | NO | 고정 위치 (리프트업) |
| KF2 | 0.174m | YES | 첫 블록 위치 (2가지 패치) |
| KF3 | 0.175m | YES | 첫 블록 place 위치 |
| KF4 | 0.202m | YES | 에피소드별 상이 |
| KF5 | 0.202m | YES | KF4와 동일 |
| KF6 | 0.175m | YES | 두 번째 블록 위치 |
| KF7 | 0.175m | YES | 두 번째 블록 place |
| KF8 | 0.175m | YES | 두 번째 블록 lift |
| KF9 | 0.002m | NO | 고정 위치 (리프트업) |
| KF10 | 0.000m | NO | 고정 HOME |

**현재 코드가 잡는 것:** Cross에서 KF1 (모든 에피소드), KF6 (일부 에피소드)
**이유:** KF1의 next-target인 KF2가 에피소드별 다르므로 cross 감지
- KF6의 next-target인 KF7도 에피소드별 다르므로 일부에서 감지

**핵심 cross 대상:** KF2, KF3, KF6, KF7 (어떤 블록을 먼저 옮기느냐에 따라 위치 변동)

### 2.4 현재 전처리 vs 실제 필요

| 구분 | 현재 Intra | 필요한 Intra | 현재 Cross | 필요한 Cross |
|------|-----------|-------------|-----------|-------------|
| KF | 없음 | 1,9 | 1,6 (일부) | 2,3,4,5,6,7,8 |

---

## 3. reopen_drawer (10 KF/episode)

### 3.1 Spatial Clusters

모든 에피소드에서 동일:

| 클러스터 | KF 목록 | Gripper 상태 | 평균 EE 위치 |
|---------|---------|-------------|-------------|
| A | (0, 4) | (1, 1) | (0.278, -0.008, 1.472) = HOME |
| B | (1, 2, 9) | (1, 0, 1) | (0.257, -0.114~-0.148, ~1.0) |
| C | (3, 8) | (1, 0) | (0.257, 0.096, ~1.0) |
| - | (5) | (0) | 에피소드별 상이 |
| - | (6) | (0) | 에피소드별 상이 |
| - | (7) | (1) | 고정 위치 |

**주의:** 클러스터 B에서 KF1과 KF2/KF9의 y좌표 차이:
- KF1: y ~ -0.148 (drawer 앞쪽)
- KF2: y ~ -0.114 (drawer 잡은 후)
- KF9: y ~ -0.114 (drawer 위치)
- KF1과 KF2/KF9 사이 거리: ~0.034m < 0.04m 이므로 같은 클러스터

### 3.2 Intra Ambiguity 분석

#### 클러스터 A: (0, 4) -- HOME position
- KF0 (grip=1) -> KF1: 첫 번째 drawer로 이동
- KF4 (grip=1) -> KF5: 두 번째 drawer로 이동 (에피소드별 다른 위치)
- **Next-target 분기: 0.183~0.258m** (strongly divergent)
- **현재 코드:** Intra에서 잡힘 (KF0, KF4 모두 grip=1, HOME position)

#### 클러스터 B: (1, 2, 9)
- KF1 (grip=1) -> KF2: drawer 잡기 (약간 앞으로) -> 거리 ~0.034m
- KF2 (grip=0) -> KF3: drawer 열기 (y: -0.114 -> 0.096) -> 거리 ~0.210m
- KF9 (grip=1): 마지막 KF -> next 없음
- **KF1<->KF2 분기: 0.210m** (divergent)
- **현재 코드:** 잡히지 않음 (KF1 grip=1, KF2 grip=0 -> grip 불일치)

#### 클러스터 C: (3, 8)
- KF3 (grip=1) -> KF4: HOME으로 복귀 (0.278, -0.008, 1.472) -> 거리 ~0.452m
- KF8 (grip=0) -> KF9: drawer 위치로 이동 (0.257, -0.114, ~1.0) -> 거리 ~0.210m
- **Next-target 분기: 0.452~0.534m** (strongly divergent)
- **현재 코드:** 잡히지 않음 (grip 불일치)

### 3.3 Cross-Episode Ambiguity

| KF | Max Spread | Cross 여부 | 설명 |
|----|-----------|-----------|------|
| KF0 | 0.000m | NO | 고정 HOME |
| KF1 | 0.051m | MARGINAL | z좌표 변동 (drawer 높이별) |
| KF2 | 0.050m | MARGINAL | z좌표 변동 |
| KF3 | 0.050m | MARGINAL | z좌표 변동 |
| KF4 | 0.001m | NO | 고정 HOME |
| KF5 | 0.138m | YES | 에피소드별 다른 drawer 위치 |
| KF6 | 0.138m | YES | KF5와 동일 |
| KF7 | 0.050m | MARGINAL | z좌표 변동 |
| KF8 | 0.051m | MARGINAL | z좌표 변동 |
| KF9 | 0.050m | MARGINAL | z좌표 변동 |

**z좌표 변동의 의미:** reopen_drawer에서 drawer는 3단 높이가 있음
- 1단 (z~0.949), 2단 (z~1.033), 3단 (z~1.013 가정) 등
- KF1-3, 7-9의 z좌표가 에피소드별로 ~0.05m 변동 -> drawer 높이 차이

**현재 코드가 잡는 것:** 
- Intra: KF0, KF4 (맞음)
- Cross: KF0 (2개 alt), KF6 (1~2개 alt)

**문제:** Cross에서 KF0에 alt를 주는 것은 의미상 맞지만, KF5/6의 위치 변동이 핵심

### 3.4 현재 전처리 vs 실제 필요

| 구분 | 현재 Intra | 필요한 Intra | 현재 Cross | 필요한 Cross |
|------|-----------|-------------|-----------|-------------|
| KF | 0,4 | 0,4 + (1,2) + (3,8) | 0,6 | 5,6 + (1~3,7~9 marginal) |

---

## 4. 전처리 변경 방안

### 4.1 현재 코드의 문제

`compute_scene_aware_clusters`는 3가지 조건을 AND로 결합:
```
ee_dist < 0.04 AND grip_same AND obj_dist < 0.05
```

이로 인해:
1. **put_block_back**: (2,4,7,9) 중 (4,7)만 잡힘 -> KF2, KF9 누락
2. **rearrange_block**: (1,9) 전혀 잡히지 않음
3. **reopen_drawer**: (1,2,9)와 (3,8) 잡히지 않음

### 4.2 제안: Spatial-Only Intra Clustering

**변경 내용:** `compute_scene_aware_clusters`에 `spatial_only` 모드 추가

```python
def compute_scene_aware_clusters(
    gripper_poses, gripper_opens,
    ee_radius=0.04, obj_radius=0.05,
    spatial_only=False,  # NEW: EE position만으로 클러스터링
):
    if spatial_only:
        # EE position만 사용
        # grip, object position 무시
    else:
        # 기존 로직 (하위 호환)
```

**근거:**
- 같은 EE 위치에서 grip이 다르더라도, 모델은 point cloud에서 grip 상태를 볼 수 있음
- 같은 EE 위치에서 object position이 다르더라도, 모델은 point cloud에서 물체를 볼 수 있음
- 따라서 spatial-only cluster의 모든 멤버에 multi-peak label을 주면, 모델은 시각 정보로 분기를 해소할 수 있는 경우 자연스럽게 하나의 peak에 집중하게 됨
- 해소할 수 없는 경우 (진짜 ambiguity) multi-peak가 도움

### 4.3 예상 효과: Multi-peak Label 수 변화

| Task | 현재 Intra KF/ep | 변경 후 Intra KF/ep | 증가 |
|------|-----------------|-------------------|------|
| put_block_back | 2 (KF4,7) | 6 (KF2,4,7,9 + KF1,10) | +4 |
| rearrange_block | 0 | 2 (KF1,9) | +2 |
| reopen_drawer | 2 (KF0,4) | 6 (KF0,4 + KF1,2 + KF3,8) | +4 |

### 4.4 주의사항

1. **Over-labeling 리스크:** spatial-only로 확장하면, 실제로 모델이 구분 가능한 경우에도 multi-peak label이 붙음. 하지만 이 경우 모델은 학습 과정에서 조건부 정보(grip/object)를 활용하여 하나의 peak에만 집중하게 되므로, 정보 손실은 없음.

2. **Convergent clusters 제외 필요:** 클러스터 D (3,8)에서 put_block_back의 경우 next-target이 수렴함 (거리 ~0.001m). 이런 경우는 multi-peak가 아님. -> `target_diverge_threshold` 체크는 유지해야 함.

3. **Cross-episode와의 중복:** KF9 (put_block_back)는 intra + cross 모두에 해당. `_both.json`에서 통합 처리 필요.

### 4.5 구현 우선순위

1. `multipeak_utils.py`의 `compute_scene_aware_clusters`에 `spatial_only=True` 옵션 추가
2. `collect_alt_targets`에서 `spatial_only=True`로 호출하여 intra 후보 확장
3. 전처리 스크립트 재실행하여 새 JSON 생성
4. 학습 실험으로 효과 검증

---

## 5. 요약 테이블

### 최종 Intra Ambiguity (Spatial-only clustering)

| Task | 클러스터 | KFs | Grip 차이 | Next-target 분기 | 모델 해소 가능? |
|------|---------|-----|----------|-----------------|--------------|
| put_block_back | (2,4,7,9) | 4개 | O | 0.001~0.468m | grip+obj로 구분 가능 |
| put_block_back | (1,10) | 2개 | X (둘다 0) | 0.230m | obj position으로 구분 가능 |
| put_block_back | (3,8) | 2개 | O | 0.001m | **수렴 -> multi-peak 아님** |
| put_block_back | (0,11) | 2개 | O | N/A (KF11 last) | N/A |
| rearrange_block | (1,9) | 2개 | X (둘다 0) | 0.276m | obj position으로 구분 가능 |
| rearrange_block | (0,10) | 2개 | O | N/A (KF10 last) | N/A |
| reopen_drawer | (0,4) | 2개 | X (둘다 1) | 0.183~0.258m | 기억으로만 구분 (어떤 drawer를 닫았는지) |
| reopen_drawer | (1,2,9) | 3개 | 혼합 | 0.210m | grip 차이로 구분 가능 |
| reopen_drawer | (3,8) | 2개 | O | 0.452~0.534m | grip 차이로 구분 가능 |

### 최종 Cross-Episode Ambiguity

| Task | KF | Max Spread | 원인 |
|------|-----|-----------|------|
| put_block_back | 0,1,10,11 | 0.267m | 블록 시작 위치 4가지 패치 |
| put_block_back | 5,6 | 0.202m | 새 블록 위치 (에피소드별) |
| rearrange_block | 2,3,6,7,8 | 0.174m | unmoved block 위치 (2가지 패치) |
| rearrange_block | 4,5 | 0.202m | 에피소드별 상이 |
| reopen_drawer | 5,6 | 0.138m | 닫을 drawer 위치 (에피소드별) |
| reopen_drawer | 1~3,7~9 | 0.050m | drawer 높이 (marginal) |
