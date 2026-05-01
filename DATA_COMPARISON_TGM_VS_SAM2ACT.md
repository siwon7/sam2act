# TGM-VLA vs SAM2Act: 데이터/학습 차이 분석

## 왜 TGM-VLA에서 multi-peak이 살아남고, SAM2Act에서는 collapse하는가?

---

## 1. 학습 데이터 규모 및 다양성

| 항목 | TGM-VLA | SAM2Act (ours) |
|------|:---:|:---:|
| Task 수 | **18** | **1** |
| Demo/task | 100 | 100 |
| 총 demo | **1,800** | **100** |
| 고유 scene 수 | **1,800+** | ~100 |
| Batch size | 24 | 8 |
| GPUs | 8 | 4 |
| Epochs | 12 | 90 |
| Total steps | ~7,500 (8GPU) | ~450,000 (4GPU) |
| Total samples | ~180K | ~3.6M |

**핵심**: SAM2Act는 총 sample 수는 많지만 (3.6M vs 180K), **같은 100개 demo를 90번 반복**함.
TGM-VLA는 1,800개 demo를 12번만 반복 — **diversity가 30배 이상**.

---

## 2. Multi-Target Task의 존재 (TGM-VLA만)

TGM-VLA의 18 tasks 중 다수가 **본질적으로 multi-target 구조**를 가짐:

| Task | Multi-target 구조 | 설명 |
|------|------|------|
| **push_buttons** | 3개 버튼을 순서대로 누름 | 어떤 버튼을 먼저 누를지 = multi-target |
| **stack_blocks** | 여러 block을 쌓음 | 어떤 block을 먼저 잡을지 |
| **stack_cups** | 여러 cup을 쌓음 | 어떤 cup을 먼저 |
| **place_cups** | 여러 cup 배치 | 어떤 위치에 먼저 |
| **put_groceries_in_cupboard** | 여러 물건을 넣음 | 어떤 것을 먼저 |

이 task들은 **multi-peak label이 없어도** variation에 의해 같은 observation에서 다른 target을 보게 됨.
→ 모델이 자연스럽게 "여러 위치가 동시에 유효할 수 있다"는 것을 학습.
→ **Intra-mixup은 이 자연적 multi-modality를 강화하는 역할**.

**SAM2Act (put_block_back)**:
- 12 KF 중 3개만 multi-target (KF4,7,9)
- 나머지 9개는 명확한 single-target
- Multi-peak signal이 약해서 single-peak gradient에 밀림

---

## 3. Cross-Mixup의 역할 (TGM-VLA만)

```python
# TGM-VLA: 50% 확률로 다른 task의 point cloud를 concat
if cross_mixup_flag[i]:
    pc[i] = cat([pc[i], pc[random_different_task]])
    # label은 single-peak 유지!
```

Cross-mixup은 multi-peak label을 만들지 않지만:
1. **Distractor robustness**: 다른 scene의 point가 섞여도 자기 target을 찾는 법을 배움
2. **Feature space 풍부화**: 다양한 point cloud 패턴을 보면서 feature가 더 일반화됨
3. **Implicit regularization**: 모델이 point cloud의 특정 패턴에 overfit하지 않게 됨

**이것이 multi-peak 유지에 간접적으로 기여**: feature space가 풍부해지면 multi-peak representation이 더 쉬움.

SAM2Act에는 cross-task mixup이 없음 (single task 학습이라 불가능).

---

## 4. Anchor View 비율 버그 (수정 완료)

| | TGM-VLA | SAM2Act v5.1 | SAM2Act v5.2 (수정) |
|---|:---:|:---:|:---:|
| Total views | 6 (3+3) | 6 (3+3) | 6 (3+3) |
| Multi-peak views | 3 (stage1) | **5** (버그) | **3** (stage1) |
| Anchor views | 3 (stage2) | **1** | **3** (stage2) |
| Anchor ratio | **50%** | 17% | **50%** |

v5.2에서 수정 완료. `nc // 2`로 stage1 views만 multi-peak.

---

## 5. LoRA vs Full Fine-tuning

| | TGM-VLA | SAM2Act |
|---|:---:|:---:|
| SAM2 backbone | **Full fine-tune** | **LoRA (r=16)** |
| Trainable params | ~73M | ~73M (비슷) |
| Backbone adaptation | 전체 weight 변경 가능 | Low-rank 제약 |

LoRA는 low-rank 제약으로 인해 **multi-modal distribution을 표현하기 어려울 수 있음**.
Full fine-tuning은 backbone의 전체 capacity를 사용 → multi-peak feature를 더 잘 유지.

---

## 6. 종합: 왜 TGM-VLA에서만 multi-peak이 살아남는가

**주요 원인 (중요도 순):**

### 1위: Multi-target task들의 존재 (데이터 구조)
push_buttons, stack_blocks 등에서 **본질적으로 multi-target gradient**가 발생.
이 gradient가 모델의 feature space를 multi-modal 방향으로 유도.
SAM2Act는 단일 task(put_block_back)로 이 효과가 없음.

### 2위: Cross-task mixup (50%)
다른 task의 point cloud를 섞으면서 feature space가 풍부해짐.
Distractor에 강해지면서도 multi-peak 표현 능력이 간접적으로 향상.

### 3위: Anchor view ratio (50% vs 17%)
v5.2에서 수정 완료. 이제 TGM-VLA와 동일한 50% anchor.

### 4위: Scene diversity (1800 vs 100 demos)
더 다양한 scene → 모델이 특정 패턴에 overfit하기 어려움 → multi-peak 유지에 유리.

### 5위: Full fine-tuning vs LoRA
Backbone capacity 차이. 부차적이지만 기여 가능.

---

## 7. SAM2Act에서 할 수 있는 것

### 이미 적용 (v5.2)
- [x] SE(3) intra-mixup (pc concat 포함)
- [x] Anchor view ratio 50% (stage1 views만 multi-peak)
- [x] 좌표 변환 수정 (alt target에 SE(3) aug 적용)

### 적용 불가 (구조적 제약)
- [ ] Multi-task 학습 — MemoryBench가 목표이므로 단일 task 학습이 자연스러움
- [ ] Cross-task mixup — 단일 task이므로 불가능
- [ ] Full fine-tuning — SAM2Act 논문 설정 (LoRA) 유지 필요

### 추가 고려 가능
- [ ] MemoryBench 3개 task(put_block_back + rearrange_block + reopen_drawer) 동시 학습
  → cross-task diversity 확보 가능
  → 하지만 각 task 성능이 떨어질 수 있음
- [ ] Mixup rate 조정 (현재 SE(3) 10% → 20~40% 등)
- [ ] Batch size 증가 (8 → 24) — GPU 메모리 허용 시

---

## 8. 결론

**TGM-VLA의 multi-peak 유지는 주로 데이터 구조 (multi-task, multi-target tasks)에 의한 것이며,
mixup 기법은 이 자연적 multi-modality를 강화하는 보조 역할.**

**SAM2Act에서 단일 task 학습으로 multi-peak을 유지하려면,
데이터 구조적 한계를 극복하기 위한 추가적인 메커니즘이 필요할 수 있음.**

가능한 방향:
1. 3개 MemoryBench task 동시 학습 (mini multi-task)
2. Memory-dependent ambiguity에 대한 별도 loss term (entropy regularization 등)
3. Multi-peak을 Stage1이 아닌 Stage2에서 처리하는 방향으로 전환
