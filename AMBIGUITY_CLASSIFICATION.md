# MemoryBench Ambiguity Classification

## 원칙

Stage1은 memory가 없다. 따라서:
- **현재 observation만으로 정답을 결정 가능** → single-peak (모델이 visual cue 학습)
- **현재 observation만으로 정답을 결정 불가 (과거 정보 필요)** → multi-peak (Stage2 memory가 선택)

---

## 1. put_block_back (12 KFs, 4 variations)

### Task 설명
```
Scene: 4개 color patch(target1~4) + center 영역 + button(랜덤 spawn)
       block이 처음에 하나의 color patch 위에 놓여있음
Instruction: "block을 center로 옮기고, button 누르고, block을 원래 자리로 되돌림"
Variation: 4개 (block 시작 patch가 다름)
Button: 매 에피소드 SpawnBoundary로 랜덤 위치 (z≈0.914)
Goal conditions (순서대로):
  1. block이 원래 patch에서 들려짐
  2. block이 center에 놓임
  3. button 눌림 (JointTriggerCondition 0.003m)
  4. block이 원래 patch로 복귀
  5. gripper 해제
```

### Keyframe 흐름
```
Phase 1: Block을 원래 자리에서 center로 옮김
  KF0:  PATCH_HOME (variation마다 다름), grip=0       → 위로 올림
  KF1:  PATCH_HIGH (variation마다 다름), grip=0       → CENTER_HIGH로 수평 이동
  KF2:  CENTER_HIGH (0.275,0.176,0.946), grip=0       → 아래로 (grasp 준비)
  KF3:  CENTER_LOW (0.275,0.177,0.797), grip=1 [GRASP] → 위로 올림

Phase 2: Block 들고 button 누르기
  KF4:  CENTER_HIGH (0.275,0.177,0.945), grip=1       → BUTTON으로 이동  ★
  KF5:  BUTTON (매번 다름), grip=0 [RELEASE]           → 아래로 (눌러서)
  KF6:  BUTTON_LOW (매번 다름), grip=0                 → CENTER_HIGH로 복귀

Phase 3: Block 다시 잡아서 원래 자리로 복귀
  KF7:  CENTER_HIGH (0.274,0.177,0.946), grip=1 [GRASP] → 아래로  ★
  KF8:  CENTER_LOW (0.274,0.177,0.796), grip=0 [RELEASE] → 위로 올림
  KF9:  CENTER_HIGH (0.275,0.176,0.945), grip=0       → PATCH_HOME으로 이동  ★
  KF10: PATCH_HIGH (variation마다 다름), grip=0         → 아래로 (놓기)
  KF11: PATCH_HOME (variation마다 다름), grip=1 [GRASP]  → END
```

### Ambiguity 분석

#### KF4 vs KF7: Revisit Ambiguity ★★★

| | KF4 | KF7 |
|---|---|---|
| 위치 | CENTER_HIGH | CENTER_HIGH (동일) |
| Gripper | 1 (block 들고) | 1 (block 들고) (동일) |
| 다음 행동 | button으로 이동 (0.2~0.5m 멀리) | 아래로 내림 (0.149m) |
| 차이 | button 안 눌림 | button 눌림 (3mm) |

**Stage1에서 구분 가능?**: **불가**
- button joint 변위 3mm = 224×224 이미지에서 sub-pixel
- 모든 에피소드에서 동일한 패턴
- **memory 필요**: "button을 이미 눌렀는가?" = task phase

**→ MULTI-PEAK (revisit)**

#### KF4 에피소드 간: Button 위치 Variation

| 에피소드 | KF4 위치 | → button 위치 |
|:---:|:---:|:---:|
| 0 | CENTER_HIGH | (0.045,-0.214) |
| 1 | CENTER_HIGH | (0.186,-0.258) |
| 2 | CENTER_HIGH | (0.196,-0.332) |
| ... | CENTER_HIGH | 매번 다른 8개 위치 |

**Stage1에서 구분 가능?**: **가능**
- button은 scene에 visible object로 존재
- 모델이 RGB에서 button 위치를 찾으면 그리로 가면 됨
- z≈0.914 필터로 이미 cross alt target에서 제외됨

**→ SINGLE-PEAK (visually resolvable)**

#### KF9 에피소드 간: Block 원래 위치 Variation ★★★

| 에피소드 | KF9 위치 | → 원래 patch |
|:---:|:---:|:---:|
| 0,2,3,7,8 | CENTER_HIGH | (0.275,0.001) = 앞쪽 patch |
| 1,4,5,9 | CENTER_HIGH | (0.455,0.176) = 오른쪽 patch |
| 6 | CENTER_HIGH | (0.076,0.175) = 왼쪽 patch |

**KF9에서 scene 상태**:
- block은 CENTER_LOW에 놓여있음 (KF8에서 release)
- 4개 color patch 모두 비어있음 (block은 center에 있으므로)
- robot은 CENTER_HIGH에서 어떤 patch로 갈지 결정해야 함

**Stage1에서 구분 가능?**: **불가**
- 4개 빈 patch가 다 보이지만, 어떤 것이 "원래 자리"인지는 현재 관측만으로 알 수 없음
- KF0에서 block이 어디 있었는지 = **초기 상태 정보** = memory 필요
- instruction "move the block back to where it was placed" → 어디인지 안 알려줌

**→ MULTI-PEAK (memory-dependent variation)**

#### KF0, KF10 에피소드 간: Block 시작/복귀 위치

| 에피소드 | KF0/KF10 위치 | → 다음 |
|:---:|:---:|:---:|
| variation 마다 | 서로 다른 patch | 위/아래 |

**Stage1에서 구분 가능?**: **가능**
- KF0: block이 해당 patch 위에 있음 → block 위치가 보임
- KF10: 해당 patch 위에 있음 → 현재 위치에서 아래로 내리면 됨
- EE 자체가 이미 variation별로 다른 위치

**→ SINGLE-PEAK (EE 위치가 이미 다르므로 자연스럽게 구분)**

### put_block_back 요약

| KF | Ambiguity 유형 | Stage1 해결? | Multi-peak? |
|:---:|:---:|:---:|:---:|
| KF0~3 | 없음 (EE가 variation별 다름) | 가능 | X |
| **KF4** | **Revisit** (KF7과 동일 위치, phase만 다름) | **불가** (3mm) | **O** |
| KF5~6 | 없음 (EE가 button별 다름) | 가능 | X |
| **KF7** | **Revisit** (KF4와 동일) | **불가** | **O** |
| KF8 | 없음 | 가능 | X |
| **KF9** | **Memory-dep variation** (원래 patch 모름) | **불가** | **O** |
| KF10~11 | 없음 (EE가 variation별 다름) | 가능 | X |

**Multi-peak: KF4, KF7, KF9 = 3/12 = 25%**

---

## 2. rearrange_block (11 KFs, 2 variations)

### Task 설명
```
Scene: 2개 color patch(target1, target2) + center 영역 + button(랜덤)
       patch_block(block1)이 하나의 patch 위에, center_block(block2)이 center에 놓여있음
Instruction: "center block을 빈 patch로, button 누르고, patch block을 center로"
Variation: 2개 (block1이 target1 위 vs target2 위)
  - variation 0: block1=target1, 빈 patch=target2
  - variation 1: block1=target2, 빈 patch=target1
Goal conditions (순서대로):
  1. center_block이 center에서 들려짐
  2. center_block이 dual_patch에 놓임
  3. button 눌림
  4. patch_block이 patch에서 들려짐
  5. patch_block이 center에 놓임
  6. gripper 해제
```

### Keyframe 흐름
```
Phase 1: Center block을 빈 patch로 옮김
  KF0:  CENTER_LOW (0.275,0.175,0.796), grip=0        → 위로 올림
  KF1:  CENTER_HIGH (0.274,0.175,1.011), grip=0       → DUAL_PATCH 쪽 이동  ★
  KF2:  DUAL_PATCH (variation마다 다름), grip=0         → 아래로
  KF3:  DUAL_PATCH_LOW (variation마다 다름), grip=1 [GRASP] → BUTTON으로  ★

Phase 2: Button 누르기
  KF4:  BUTTON (매번 다름), grip=0 [RELEASE]            → 아래로
  KF5:  BUTTON_LOW (매번 다름), grip=0                  → PATCH_BLOCK 쪽 이동

Phase 3: Patch block을 center로 옮김
  KF6:  PATCH_BLOCK (variation마다 다름), grip=1 [GRASP]  → 아래로  ★
  KF7:  PATCH_BLOCK_LOW (variation마다 다름), grip=0 [RELEASE] → 위로
  KF8:  PATCH_BLOCK_HIGH (variation마다 다름), grip=0    → CENTER_HIGH로 이동
  KF9:  CENTER_HIGH (0.275,0.175,1.012), grip=0        → 아래로
  KF10: CENTER_LOW (0.275,0.175,0.797), grip=1 [GRASP]  → END
```

### Ambiguity 분석

#### KF1 에피소드 간: Dual Patch 위치 ★★★

| 에피소드 | KF1 위치 | → dual patch |
|:---:|:---:|:---:|
| var 0 (5ep) | CENTER_HIGH | (0.281,0.002) = target2 (앞) |
| var 1 (5ep) | CENTER_HIGH | (0.275,0.349) = target1 (뒤) |

**KF1에서 scene 상태**:
- center_block은 위로 들린 상태 (center 위)
- patch 하나에 block1 있고, 다른 patch 비어있음
- 비어있는 patch로 가야 함

**Stage1에서 구분 가능?**: **가능 — 조건부**
- 비어있는 patch를 시각적으로 찾을 수 있음 (block이 없는 patch)
- **BUT**: center_block을 이미 들고 있으므로 center가 비어있고, patch_block은 patch에 있고, dual patch가 비어있음
- 2개 patch 중 block이 없는 쪽을 찾으면 됨 → visual cue 존재

**→ SINGLE-PEAK (빈 patch가 보임)** ... 하지만 실제로 모델이 이걸 학습하는지?

실제 데이터를 보면: KF1의 EE는 모든 에피소드에서 CENTER_HIGH(0.274,0.175,1.011)로 동일. 하지만 **block1 위치가 다름** (target1 vs target2에 있음). block1 위치가 보이므로, "block1이 없는 patch"로 가면 됨.

**→ SINGLE-PEAK (visually resolvable)**

#### KF3 에피소드 간: Button 위치 (z≈0.914)

8개 다른 button 위치. z≈0.914 필터로 이미 제외됨.

**→ SINGLE-PEAK**

#### KF5 에피소드 간: Patch Block으로 이동

| 에피소드 | KF5 위치 | → patch_block |
|:---:|:---:|:---:|
| var 0 | BUTTON_LOW | (0.275,0.350) = target1 |
| var 1 | BUTTON_LOW | (0.281,0.001) = target2 |

**KF5에서 scene 상태**:
- button 이미 누름, center_block은 dual_patch에 놓여있음
- patch_block은 원래 patch에 그대로 있음
- patch_block의 위치가 보임

**Stage1에서 구분 가능?**: **가능** (block이 보임)

**→ SINGLE-PEAK**

#### KF6 에피소드 간: Variation에 따른 위치

KF6의 EE 자체가 variation마다 다름 (patch_block 위치). EE spread=0.175.

**→ SINGLE-PEAK (EE 자체가 다름)**

### rearrange_block 요약

| KF | Ambiguity 유형 | Stage1 해결? | Multi-peak? |
|:---:|:---:|:---:|:---:|
| KF0 | 없음 | 가능 | X |
| KF1 | Variation (dual patch 방향) | **가능** (빈 patch 보임) | X |
| KF2~5 | EE가 variation별 다름 / button visible | 가능 | X |
| KF6~8 | EE가 variation별 다름 | 가능 | X |
| KF9~10 | 없음 | 가능 | X |

**Multi-peak: 없음 = 0/11 = 0%**

**핵심**: rearrange_block은 **모든 variation에서 visual cue가 존재**한다.
- 2개 patch에 각각 다른 block이 있거나 비어있음 → 보면 알 수 있음
- Revisit ambiguity도 없음 (같은 위치를 같은 상태로 재방문하지 않음)
- REVISED 분석과 일치: rearrange_block truly ambiguous = 0%

---

## 3. reopen_drawer (10 KFs, 3 variations)

### Task 설명
```
Scene: 3단 서랍(bottom/middle/top) + button(랜덤 spawn)
       처음에 하나의 서랍이 열려있음 (joint position = 0.21m)
Instruction: "Close the opened drawer, push the button, and then open the previous drawer again"
Variation: 3개 (bottom=0, middle=1, top=2 중 어느 서랍이 열려있는지)
  - 서랍 높이: bottom z≈0.949, middle z≈1.033 (차이 0.084m = 8.4cm)
Button: 매 에피소드 SpawnBoundary 랜덤 위치
Goal conditions:
  1. 서랍 닫힘 (joint 변위 > 0.19m)
  2. button 눌림 (joint 변위 > 0.003m)
  3. 서랍 다시 열림 (detector로 서랍 파트 감지)
```

### Keyframe 흐름
```
Phase 1: 열린 서랍 닫기
  KF0:  HOME (0.279,-0.008,1.471), grip=1               → 서랍 앞으로 이동  ★
  KF1:  DRAWER_FRONT (0.257,-0.148,z), grip=1            → 서랍 핸들 잡기
  KF2:  DRAWER_OPEN_EDGE (0.257,-0.114,z), grip=0 [REL]  → 서랍 밀어 닫기
  KF3:  DRAWER_CLOSED (0.257,0.096,z), grip=1 [GRASP]    → HOME 복귀

Phase 2: Button 누르기
  KF4:  HOME (0.278,-0.008,1.472), grip=1                → BUTTON으로  ★
  KF5:  BUTTON (매번 다름), grip=0 [RELEASE]               → 아래로 (누름)
  KF6:  BUTTON_LOW (매번 다름), grip=0                     → 서랍 안쪽으로  ★

Phase 3: 서랍 다시 열기
  KF7:  DRAWER_INSIDE (0.257,0.054,z), grip=1 [GRASP]    → 서랍 핸들로
  KF8:  DRAWER_HANDLE (0.257,0.096,z), grip=0 [RELEASE]   → 서랍 당기기
  KF9:  DRAWER_FRONT (0.257,-0.114,z), grip=1 [GRASP]     → END

※ KF1~3, KF7~9의 z는 variation(어떤 서랍)에 따라 0.949 또는 1.033
※ KF0, KF4는 HOME으로 z=1.471~1.472 (서랍 높이와 무관)
```

### Ambiguity 분석

#### KF0 vs KF4: Revisit Ambiguity ★★★

| | KF0 | KF4 |
|---|---|---|
| 위치 | HOME (0.279,-0.008,1.471) | HOME (0.278,-0.008,1.472) — **동일** |
| Gripper | 1 | 1 — **동일** |
| 다음 행동 | 서랍 앞으로 (z=0.949 or 1.033) | button으로 (z=0.914, 랜덤 xy) |
| Scene 차이 | 서랍이 **열려있음** (0.21m 돌출) | 서랍이 **닫혀있음** |

**시각적 차이**: 서랍 0.21m 열림/닫힘은 이론적으로 보일 수 있음
**실제 관찰**: Stage1 모델이 KF0에서 5/5 MISS (collapse 확인됨)
- HOME(z=1.471)에서 서랍(z≈1.0)을 내려다보는 각도에서 서랍 돌출이 잘 안 보일 수 있음
- 5개 카메라 중 서랍 앞쪽을 보는 카메라가 제한적

**→ MULTI-PEAK** (실증적으로 collapse 확인됨, 서랍 상태 구분 어려움)

#### KF0 에피소드 간: 서랍 높이 Variation ★★★

| Variation | KF0 → 서랍 앞 위치 |
|:---:|:---:|
| bottom (4ep) | → z=0.949 |
| middle (6ep) | → z=1.033 |

**KF0 시점에서**: 서랍이 열려있으므로 **어떤 서랍이 열려있는지는 보임**
(열린 서랍이 돌출되므로 높이를 알 수 있음)

**BUT**: HOME에서 아래를 내려다볼 때, bottom(z=0.949)과 middle(z=1.033)의 차이 8.4cm가 충분히 구분 가능한가?
- 3단 서랍이 층층이 있고, 열린 서랍만 돌출
- 서랍 자체는 구조가 유사하므로 **높이만으로 구분해야 함**
- 8.4cm는 224×224 이미지에서 수 pixel 차이

**→ 판단이 어려움: 서랍 높이 차이가 보이면 single-peak, 안 보이면 multi-peak**
**보수적 판단: MULTI-PEAK** (실제 모델 collapse 확인됨)

#### KF4 에피소드 간: Button 위치

5개 다른 button 위치 (z≈0.914). Button이 scene에 보임.

**→ SINGLE-PEAK** (button visible, z≈0.914 필터로 이미 제외)

#### KF6 에피소드 간: 어떤 서랍을 다시 열 것인가 ★★★

| Variation | KF6 위치 (BUTTON_LOW) | → 서랍 안쪽 |
|:---:|:---:|:---:|
| bottom | 랜덤 xy, z=0.772 | → (0.257,0.054,**0.949**) |
| middle | 랜덤 xy, z=0.772 | → (0.257,0.054,**1.033**) |

**KF6 시점에서**:
- Phase 1에서 서랍을 닫았고, Phase 2에서 button을 눌렀음
- **모든 서랍이 닫혀있음** — 외관상 동일
- 어떤 서랍을 다시 열어야 하는지? = **아까 닫았던 서랍**
- Stage1은 "아까 어떤 서랍을 닫았는지" 기억 못함 (no memory)
- KF6의 EE 위치는 button 위치(랜덤)에 따라 다르지만, button 위치와 서랍 높이는 무관

**→ MULTI-PEAK** (memory-dependent: "Phase 1에서 어떤 서랍을 닫았는가?")

### reopen_drawer 요약

| KF | Ambiguity 유형 | 원인 | Stage1 해결? | Multi-peak? |
|:---:|:---:|------|:---:|:---:|
| **KF0** | **Revisit** | KF4와 동일 HOME, 서랍 상태만 다름 | **불가** (collapse 확인) | **O** |
| KF1~3 | Variation (z) | 서랍 높이 차이 8.4cm | EE 자체가 다름 | X |
| **KF4** | **Revisit** | KF0과 동일 HOME, phase 다름 | **불가** | **O** |
| KF5 | Variation | button 위치 | button 보임 | X |
| **KF6** | **Mem-dep Variation** | 서랍 다 닫힘, 어떤 건지 모름 | **불가** (memory 필요) | **O** |
| KF7~9 | Variation (z) | 서랍 높이 | EE 자체가 다름 | X |

**Multi-peak: KF0, KF4, KF6 = 3/10 = 30%**

---

## 전체 요약

### Multi-peak이 필요한 경우 (Memory-Dependent)

| Task | KF | 유형 | 왜 구분 불가 |
|------|:---:|:---:|------|
| put_block_back | **KF4, KF7** | Revisit | 버튼 3mm 변위 → 안 보임 |
| put_block_back | **KF9** | Mem-dep Variation | 원래 patch 모름 (4개 빈 patch) |
| reopen_drawer | **KF0, KF4** | Revisit | 서랍 상태가 HOME에서 안 보일 수 있음 |
| reopen_drawer | **(KF6)** | Mem-dep Variation | 서랍 다 닫혀서 어떤 건지 모름 |

### Multi-peak이 불필요한 경우 (Visually Resolvable)

| 구분 가능한 이유 | 예시 |
|------|------|
| **EE 위치 자체가 다름** | KF0/10/11(put_block), KF2~8(rearrange), KF1~3/7~9(reopen) |
| **Button이 보임** | KF4 cross(put_block), KF3(rearrange), KF4(reopen) |
| **Block/Object 위치가 보임** | KF1(rearrange: 빈 patch 보임), KF0 cross(reopen: 서랍 높이 보임) |

### 비율

| Task | Multi-peak KFs | 비율 |
|------|:---:|:---:|
| put_block_back | KF4, KF7, KF9 | 3/12 = 25% |
| rearrange_block | 없음 | 0/11 = 0% |
| reopen_drawer | KF0, KF4, KF6 | 3/10 = 30% |

### 현재 precomputed JSON과의 비교

| Task | `intra` 모드 | `cross` 모드 | 이 분석 결과 (memory_dependent) |
|------|:---:|:---:|:---:|
| put_block_back | KF4,7 ✓ | KF9 ✓ | **KF4,7 (revisit) + KF9 (mem-dep variation)** |
| rearrange_block | 없음 ✓ | KF1,6 ✗ | **없음** (KF1: 빈 patch 보임, KF6: 버튼 필터) |
| reopen_drawer | KF0,4 ✓ | KF0,6 | **KF0,4 (revisit) + KF6 (mem-dep variation)** |

### 현재 모드별 문제점

- **`intra` 모드**: put_block_back KF9, reopen_drawer KF6 누락 (memory-dependent variation)
- **`cross` 모드**: rearrange_block KF1,6 과잉 (visually resolvable)
- **`both` 모드**: 위 문제 합산

### 제안: `memory_dependent` 모드

기준: **Stage1(memoryless)이 현재 observation만으로 결정 불가능한 경우만** multi-peak

포함:
1. **Revisit** (에피소드 내, 같은 위치 재방문, 시각적 구분 불가)
   - put_block_back: KF4, KF7 (버튼 3mm)
   - reopen_drawer: KF0, KF4 (서랍 열림/닫힘이 HOME에서 안 보임)
2. **Memory-dependent variation** (에피소드 간, 과거 정보 없이 결정 불가)
   - put_block_back: KF9 (4개 빈 patch 중 원래 자리 = memory 필요)
   - reopen_drawer: KF6 (서랍 다 닫힘, 어떤 걸 열었는지 = memory 필요)

제외:
- **Visually resolvable variation** (현재 observation에 답이 있음)
  - 버튼 위치 (scene에 보임) — z≈0.914 필터
  - block/patch 위치 (block이 보임) — rearrange_block KF1
  - 서랍 높이 (EE 자체가 다른 높이) — reopen_drawer KF1~3,7~9

### 최종 비율

| Task | memory_dependent KFs | 비율 |
|------|:---:|:---:|
| put_block_back | KF4, KF7, KF9 | 3/12 = **25%** |
| rearrange_block | 없음 | 0/11 = **0%** |
| reopen_drawer | KF0, KF4, KF6 | 3/10 = **30%** |
| **가중 평균** | | **~18%** |

이 비율은 TGM-VLA의 effective multi-peak rate (~20%)와 유사하므로,
별도의 SE(3) mixup 비율 조정 없이도 적절한 수준.
