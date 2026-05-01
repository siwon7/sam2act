# Stage2 Survey & Design: Memory-Based Manipulation의 현황과 Graph-Guided Peak Selection 설계

---

## Part A: Memory-Based Manipulation Methods Survey

### A.1 KC-VLA: Keyframe Chaining VLA (코드 분석)

**논문**: "Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining" (Chen et al., 2026)

#### A.1.1 전체 아키텍처

KC-VLA는 두 단계로 구성된다:

1. **Keyframe Selection Module (KSM)**: 현재 observation 스트림에서 "event-driven keyframe"을 검출
2. **Keyframe-Chaining VLA**: 검출된 keyframe들을 sparse semantic history로 VLA에 전달하여 action 생성

```
[Online Stream] → KSM → keyframe 검출 → [keyframe_0, keyframe_1, ...] → VLA → action
                   ↑                                                        ↑
              Phase + Task 조건부                              HistoryQueryModule로 keyframe 집합 활용
```

#### A.1.2 Task-Modulated Keyframe Selection Module (KSM) 상세

KSM은 2-stage 학습 파이프라인이다.

**Stage 1: Contrastive Backbone 학습**

파일: `keyframe_selection_module/model/stage1_network.py`, `train_stage1.py`

```python
class ResNetContrastive(nn.Module):
    # ResNet18 (ImageNet pretrained) + projector → 128-dim L2-normalized embedding
    # Triplet Margin Loss (margin=1.0)로 학습
```

- **Anchor**: keyframe 이미지
- **Positive**: 같은 task, 같은 phase의 다른 에피소드 keyframe
- **Negative**: 3가지 전략으로 hard negative 구성
  - 33%: 시간 근접 non-keyframe (offset +-3~10 frame)
  - 33%: 같은 task, 다른 phase의 keyframe (intra-task phase confusion)
  - 33%: 다른 task의 keyframe (inter-task confusion)

이 contrastive 학습의 결과: "같은 task-phase의 keyframe은 embedding이 가깝고, 다른 것은 먼" backbone이 형성된다.

**Stage 2: Task-Modulated Binary Classifier**

파일: `keyframe_selection_module/model/network.py`, `train_stage2.py`

```python
class TransformerKeyframeSelector(nn.Module):
    # Frozen contrastive backbone → vis_projector → 128-dim
    # Task embedding + Phase embedding + FiLM generator
    # Self-attention (temporal) + Cross-attention (task-modulated query vs visual)
    # Binary classifier: "이 시점이 현재 phase의 keyframe인가?"
```

핵심 구조:

```
Input: curr_window_imgs [B, T=3, C, H, W]  (3-frame sliding window)
       phase_id [B]                           (현재 찾고 있는 phase)
       task_id [B]                            (어떤 task인지)

1. Frozen backbone → vis_feat [B, T, 512] → vis_projector → [B, T, 128]
2. Self-attention: 3-frame window 내 temporal context 형성
3. FiLM Query 생성:
     task_emb = task_embedding(task_id)    # [B, 128]
     phase_emb = phase_embedding(phase_id) # [B, 128]
     gamma, beta = FiLM_generator(task_emb) → 각각 [B, 128]
     logic_query = gamma * phase_emb + beta  # [B, 128]
4. Cross-attention: query=logic_query, key/value=vis_contextual
5. Binary classification: sigmoid(classifier(attended_output)) → keyframe 여부
```

#### A.1.3 FiLM-based Query Generation의 작동 원리

FiLM (Feature-wise Linear Modulation)은 원래 visual reasoning에서 language conditioning에 사용된 기법이다.
KC-VLA에서의 적용:

```
task_emb → FiLM generator → (gamma, beta)
logic_query = gamma * phase_emb + beta
```

- `phase_emb`는 "지금 몇 번째 milestone을 찾고 있는가"의 정보
- `task_emb`가 FiLM 파라미터 (gamma, beta)를 생성하여 phase embedding을 task-specific하게 변환
- 즉, **같은 phase_id=2라도 task가 다르면 완전히 다른 query**가 생성됨
- 이 query가 visual feature에 cross-attention하여 "이 시각 패턴이 현재 task-phase의 keyframe인가"를 판단

**의의**: Task와 Phase의 interaction을 단순 concat이 아니라 multiplicative modulation으로 포착.
같은 "phase 3"이라도 PickPlace에서는 "파란 큐브 집기"이고 Swap에서는 "중간 큐브 이동"임을 구분.

#### A.1.4 Non-Markovian Ambiguity 해결 메커니즘

KC-VLA의 non-Markovian 해결은 **Phase Counter + KSM의 조합**으로 이루어진다:

1. **Phase counter**: KSM이 keyframe을 검출할 때마다 `current_phase += 1`
2. **Phase-conditioned detection**: 다음 phase의 keyframe만 찾도록 phase_id를 입력
3. **Early stopping**: `max_keyframes`에 도달하면 detection 중단
4. **Threshold + Clustering**: 연속 high-confidence 구간을 cluster로 묶고, cluster 내 최고 확신도 frame을 keyframe으로 확정

```python
# 추론 시 (eval_for_maniskill.py 간소화):
for step in range(max_steps):
    prob = KSM(current_window, current_phase, task_id)
    if prob > threshold and not in_cluster:
        # 새 cluster 시작
    if cluster_ended:
        confirm_keyframe(best_frame_in_cluster)
        current_phase += 1
```

**한계**: Phase counter가 monotonic → "phase 2를 다시 방문"하는 것을 모델링하지 못함.
우리 MemoryBench의 revisit ambiguity (KF4 vs KF7: 같은 위치, 다른 phase)에 대해서는
KC-VLA도 phase counter로 구분 가능하지만, **phase가 하드코딩되어야 한다** (task별 max_keyframes 지정).

#### A.1.5 HistoryQueryModule: VLA에서의 Keyframe 활용

파일: `gr00t/model/modules/history_query.py`

KSM이 검출한 keyframe들은 VLA의 HistoryQueryModule에 전달된다:

```python
class HistoryQueryModule(nn.Module):
    # Learnable query tokens [1, 32, D] + Phase embedding
    # Cross-attention: query=phase-conditioned tokens, key/value=history keyframe features
    # 시간 위치 인코딩 + 뷰 위치 인코딩 (multi-view 지원)
    # Padding mask 처리 (가변 길이 history)
```

이 모듈의 역할:
- 검출된 keyframe의 visual feature를 시간 순서대로 encoding
- Phase embedding으로 "현재 어떤 단계에서 어떤 history가 중요한지" query
- Cross-attention 출력이 VLA backbone feature에 추가되어 action head로 전달

**핵심 통찰**: KC-VLA는 **전체 trajectory를 저장하지 않고 "event-driven sparse keyframe"만 저장**.
이것이 "Keyframe Chaining"의 의미 — keyframe들의 chain이 task progress를 compact하게 인코딩.

#### A.1.6 학습 파이프라인 요약

```
Phase 1: Contrastive Backbone (30 epochs)
  - TripletMarginLoss로 keyframe embedding 학습
  - 출력: ResNet18 backbone weights

Phase 2: KSM Classifier (50 epochs)  
  - Frozen contrastive backbone 위에 TransformerKeyframeSelector 학습
  - BCEWithLogitsLoss (pos_weight=5, class imbalance 보정)
  - 출력: binary keyframe detector

Phase 3: VLA Fine-tuning
  - NVIDIA GR00T N1.5 base model
  - Eagle2 vision backbone + Flow Matching action head
  - HistoryQueryModule로 keyframe chain 활용
  - 출력: 최종 policy
```

#### A.1.7 KC-VLA vs SAM2Act 비교

| 관점 | KC-VLA | SAM2Act |
|------|--------|---------|
| **Memory 형태** | Sparse keyframe chain (event-driven) | Dense memory bank (SAM2 attention) |
| **Memory 선택** | KSM이 keyframe 검출 → 저장 | 매 keyframe마다 자동 저장 |
| **Non-Markovian** | Phase counter + FiLM query | Memory attention (implicit) |
| **Base Model** | GR00T N1.5 (Eagle2 + Flow Matching) | PerAct-style (voxel + heatmap) |
| **Ambiguity 해결** | Phase-conditioned keyframe detection | Stage1 single-peak → collapse |
| **Task conditioning** | FiLM (task_emb × phase_emb) | Language embedding |
| **학습 비용** | 3-stage (contrastive + classifier + VLA) | 2-stage (Stage1 + Stage2) |

---

### A.2 MemoryVLA: Perceptual-Cognitive Dual Memory (논문 기반)

**접근**: Working memory를 perceptual(지각적)과 cognitive(인지적) dual stream으로 분리.

- **Perceptual Memory**: 최근 N프레임의 visual feature 저장 (sliding window)
  - 단기 시각 변화 추적에 유리
  - 빠른 반응 (grasp timing 등)
- **Cognitive Memory**: Task progress를 language/symbolic level로 요약
  - "Phase 1 완료: block을 center로 옮김" 같은 추상적 상태 저장
  - VLM으로 summarization → compact representation
- **Memory Fusion**: 두 memory stream을 cross-attention으로 결합하여 action 생성

**강점**: 인지 memory가 explicit → interpretable, 디버깅 용이
**약점**: VLM 기반 summarization의 latency, phase boundary가 명확하지 않은 task에서 한계

**우리와의 차이**: MemoryVLA는 "무엇을 기억할지"를 VLM이 결정하지만,
SAM2Act는 spatial attention mask가 자동으로 중요 영역을 기억. MemoryVLA는 "왜 기억하는지"에 집중하고,
SAM2Act는 "어디를 기억하는지"에 집중한다.

---

### A.3 VQ-Memory: Vector-Quantized Temporal Representation (논문 기반)

**접근**: 긴 trajectory를 vector quantization으로 압축된 discrete token sequence로 변환.

- **VQ Encoder**: 연속 observation을 codebook의 nearest code로 mapping
- **Temporal Transformer**: VQ token sequence를 처리하여 temporal pattern 학습
- **Advantage**: 고정 크기 codebook → memory 사용량 일정, 긴 horizon에도 scalable

**강점**: 압축률이 높고, discrete token이라 LLM-style reasoning과 결합 가능
**약점**: Codebook 크기 선택이 task-dependent, fine-grained spatial 정보 손실 가능

**우리와의 차이**: VQ-Memory는 "temporal 패턴"을 discrete하게 학습하지만,
spatial 해상도가 제한됨. SAM2Act의 memory bank는 pixel-level spatial feature를 유지.

---

### A.4 RoboMAP: Adaptive Affordance Heatmaps (논문 기반)

**접근**: 과거 interaction history로부터 adaptive affordance heatmap 생성.

- **Memory Module**: 과거 grasp/place 위치를 spatial memory map에 축적
- **Affordance Prediction**: 현재 observation + spatial memory → 어디를 잡을지 heatmap 출력
- **Adaptation**: 실패한 grasp 위치를 memory에서 suppress → 재시도 시 다른 위치 선택

**강점**: 실패 기반 학습(trial-and-error)에 자연스럽게 적용
**약점**: Single-step affordance에 특화, multi-step sequential manipulation에서의 효과 미검증

**우리와의 관련성**: RoboMAP의 "heatmap + memory" 구조는 SAM2Act와 유사.
차이점은 RoboMAP이 grasp affordance에 집중하는 반면, SAM2Act는 sequential waypoint prediction.

---

### A.5 SAM2Act+: SAM2 Memory Attention (우리 baseline)

**접근**: SAM2의 video object segmentation memory mechanism을 robot manipulation에 적용.

- **Memory Bank**: 과거 keyframe의 (feature, positional encoding, masking info) 저장
- **Memory Attention**: 현재 frame의 feature를 memory bank와 cross-attention
- **Stage1**: Memory 없이 학습 → single-peak heatmap (ambiguity collapse 발생)
- **Stage2**: Memory attention 추가 → memory-enhanced feature → refined heatmap

**현재 한계 (코드 분석 기반)**:
1. Stage1의 multi-peak 출력을 Stage2가 **명시적으로 선택하지 못함**
   - Memory attention은 feature 전체를 부드럽게 수정할 뿐
   - "어떤 peak을 선택"하는 직접적 mechanism이 없음
2. Memory에 **structural/phase 정보가 없음**
   - 같은 위치에서 다른 phase → 같은 feature → memory attention이 구분 못함
3. Revisit ambiguity (put_block_back KF4 vs KF7, reopen_drawer KF0 vs KF4)에서
   현재 observation만으로 시각적 차이가 거의 없어 Stage1이 collapse

---

### A.6 Methods 비교 요약

| 방법 | Memory 형태 | Ambiguity 해결 | Spatial 해상도 | 확장성 |
|------|------------|---------------|--------------|--------|
| **KC-VLA** | Sparse keyframe chain | Phase counter + FiLM | Low (ResNet global) | Task별 phase 수 지정 필요 |
| **MemoryVLA** | Dual (perceptual + cognitive) | VLM summarization | Medium | VLM latency |
| **VQ-Memory** | Discrete VQ tokens | Temporal transformer | Low (quantized) | Codebook 크기 의존 |
| **RoboMAP** | Spatial affordance map | Failure suppression | High (pixel-level) | Single-step 특화 |
| **SAM2Act** | Dense feature bank | Implicit attention | High (pixel-level) | 구조적 ambiguity 미해결 |
| **Ours (제안)** | Dense + Graph nodes | Explicit peak selection | High (pixel-level) | Data-driven graph |

---

## Part B: Benchmark Survey

### B.1 MemoryBench (SAM2Act)

**플랫폼**: RLBench (CoppeliaSim)
**Task 수**: 3개
**Test Episodes**: 25 per task (총 75)

| Task | KFs | Memory-Dependent KFs | 비율 | Memory 유형 |
|------|:---:|:---:|:---:|------|
| put_block_back | 12 | KF4, KF7, KF9 | 25% | Revisit + Mem-dep Variation |
| rearrange_block | 11 | 0 | 0% | (없음 - visual cue 존재) |
| reopen_drawer | 10 | KF0, KF4, KF6 | 30% | Revisit + Mem-dep Variation |

**Memory 유형 분류**:
- **Revisit Ambiguity**: 같은 위치 재방문, 시각적 구분 불가 (button 3mm, 서랍 상태)
- **Memory-dependent Variation**: 과거 정보 없이 현재 target 결정 불가 (원래 patch, 원래 서랍)

**한계 (감사 결과)**:
- Test split 불균형 (majority baseline 48~56%)
- Residual visual cue 존재 (단일 프레임 RGB로도 variation 분류 가능한 구간)
- 3개 task 중 rearrange_block은 memory 불필요 → 실질적 memory benchmark = 2개 task

---

### B.2 RMBench: Memory-Dependent Robotic Manipulation Benchmark (코드 분석)

**논문**: "RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design" (Chen et al., 2026)
**플랫폼**: SAPIEN (RoboTwin 2.0 기반), Dual-arm ALOHA

**Task 수**: 코드 기준 10개 task 파일 + eval config 13개 (일부 미공개)

코드에서 확인된 task (환경 파일 기준):

| # | Task | 설명 | Memory 요구 |
|---|------|------|------------|
| 1 | **cover_blocks** | 3색 block 위에 lid 덮었다가, RGB 순서로 다시 벗김 | 순서 기억 (어떤 lid가 어떤 색 위에) |
| 2 | **put_back_block** | Block을 center로 → button → 원래 위치로 복귀 | 원래 위치 기억 (4개 mat 중 어디) |
| 3 | **rearrange_blocks** | Block 재배치 → button → 다른 block 이동 | 빈 mat 위치 기억 |
| 4 | **swap_blocks** | 3 tray에서 2 block swap (빈 tray 활용) + button | Swap 순서/경유지 기억 |
| 5 | **press_button** | 숫자 카드 관찰 → 해당 횟수만큼 버튼 누르기 | 숫자 기억 + 카운팅 |
| 6 | **battery_try** | 배터리 2개를 올바른 방향으로 슬롯에 삽입 (gauge 관찰) | 시행착오 결과 기억 |
| 7 | **blocks_ranking_try** | 3색 block 정렬 → button → 맞을 때까지 반복 | 이전 시도 결과 기억 |
| 8 | **observe_and_pickup** | 선반의 target 관찰 → 화면 가림 → 테이블에서 동일 물체 집기 | Target 외형 기억 |
| 9 | **swap_T** | T자 블록 2개의 위치+방향 교환 | 원래 pose 기억 |
| 10 | **place_block_mat** | Block들을 mat 간 이동 후 원위치 | 원래 위치 기억 |

**Memory 복잡도 분류 (Mem-0 README 기반)**:

- **M(1)-type**: 단일 memory step (observation → remember → act)
  - observe_and_pickup (target 한 번 기억)
- **M(n)-type**: 다중 memory step (반복적 관찰-기억-행동)
  - cover_blocks, blocks_ranking_try, battery_try (시행착오)
  - put_back_block, swap_blocks (multi-phase)

**Mem-0 Policy 구조**:
- **Execution Module**: Qwen3-VL-2B backbone, single-task fine-tuning
- **Planning Module**: Qwen3-VL-8B-Instruct + LoRA, vLLM 서빙
- M(n) task에서는 Planning Module이 key memory를 reasoning하여 Execution Module에 전달

**SAM2Act 평가 가능성**:
- **Simulator 차이**: RMBench는 SAPIEN (RoboTwin), SAM2Act는 CoppeliaSim (RLBench)
- **Embodiment 차이**: RMBench는 dual-arm ALOHA, SAM2Act는 single-arm Franka
- **직접 평가 불가**: 환경 + embodiment 변환 필요
- **Task 개념 유사**: put_back_block (RMBench) ≈ put_block_back (MemoryBench), 같은 memory 패턴
- **가능한 접근**: RMBench의 task 설계를 참고하여 RLBench에 memory-intensive task 추가

---

### B.3 RoboMME: 16-Task Memory Taxonomy

**Task 수**: 16개
**Memory Taxonomy**:
- **Spatial Memory**: 위치 기억 (물체 원래 위치, 목표 위치)
- **Temporal Memory**: 순서 기억 (action sequence, 시행착오 결과)
- **Semantic Memory**: 속성 기억 (색, 모양, 크기)
- **Episodic Memory**: 에피소드 전체 맥락 기억

**특징**: Taxonomy 중심의 체계적 분류를 제공하지만, 실제 벤치마크 task가 다소 인위적.
우리 MemoryBench와의 차이: RoboMME는 "어떤 종류의 memory인가"를 분류하는 데 집중,
MemoryBench는 "memory 없이 해결 가능한가"라는 실증적 기준.

---

### B.4 RuleSafe: Non-Markovian Articulated Tasks

**초점**: Articulated object (서랍, 문, 레버)에서의 non-Markovian constraint.
**예**: "이 서랍을 열기 전에 저 레버를 먼저 당겨야 한다" — 순서 규칙 기억.

**우리와의 관련성**: reopen_drawer가 이 카테고리에 해당.
서랍의 열림/닫힘 상태가 시각적으로 모호한 경우 memory가 필요.

---

### B.5 MIKASA-Robo: Memory-Intensive Tabletop

**초점**: Tabletop 환경에서 memory 집약적 task 모음.
**특징**: Long-horizon + 다수 물체 + 순서 의존 → memory가 핵심.

---

### B.6 Benchmark 비교 요약

| Benchmark | Task 수 | 환경 | Embodiment | Memory 분류 | 직접 비교 가능? |
|-----------|:---:|------|-----------|------------|:---:|
| **MemoryBench** | 3 | RLBench/CoppeliaSim | Franka single-arm | Revisit + Mem-dep | **현재 baseline** |
| **RMBench** | ~10 | SAPIEN/RoboTwin | ALOHA dual-arm | M(1)/M(n) | 환경 변환 필요 |
| **RoboMME** | 16 | 다양 | 다양 | Spatial/Temporal/Semantic/Episodic | 환경 변환 필요 |
| **RuleSafe** | - | SAPIEN | - | Non-Markovian constraint | 개념적 참고 |
| **MIKASA-Robo** | - | Tabletop | - | Memory-intensive | 개념적 참고 |

---

## Part C: Stage2 Design — Graph-Guided Peak Selection

### C.1 문제 정의

#### 현재 구조의 한계

```
Stage1 (memory 없음):
  observation → encoder → transformer → decoder → single-peak heatmap
  
  문제: ambiguous KF에서 collapse (항상 한쪽만 예측)
  - put_block_back KF4/KF7: 같은 center_high, 다른 target → 한쪽만 학습
  - put_block_back KF9: 4개 patch 중 하나만 학습
  - reopen_drawer KF0/KF4: 같은 HOME, 다른 target → 한쪽만 학습
  - reopen_drawer KF6: 서랍 다 닫힘, 어떤 건지 모름

Stage2 (memory attention):
  memory-enhanced feature → decoder → refined heatmap
  
  문제: memory attention이 feature를 수정하지만, peak 선택이 간접적
  → "어떤 peak을 선택"이 아니라 "feature를 어떻게 바꾸느냐"에 의존
```

#### Stage1 Multi-Peak 도입 후의 문제

Stage1을 multi-peak으로 학습하면 (v6.1):
- Ambiguous KF에서 **복수 후보 위치**를 출력
- 하지만 Stage2가 이 중 **어떤 peak을 선택할지 명시적으로 결정하는 mechanism이 없음**

#### 필요한 것

```
Stage1: "갈 수 있는 곳들" 제안 (multi-peak, k개 후보)
Stage2: "이 중 어디로 가야 하는가" 결정 (graph-guided selection)

결정의 근거:
  1. 직전에 어디에 있었는가 (prev_node)
  2. 지금 어디에 있는가 (curr_node, memory-enhanced)
  3. 각 후보의 의미가 무엇인가 (peak_node_embed)
  → 이 세 정보로 transition score 계산 → peak 선택
```

---

### C.2 핵심 아이디어: Structural Ambiguity-Aware Memory

#### 기존 memory 방법과의 차별점

| 방법 | Memory의 역할 | Ambiguity 해결 |
|------|-------------|---------------|
| KC-VLA | "어떤 phase인지" 카운팅 | Phase counter (하드코딩) |
| MemoryVLA | "무슨 일이 있었는지" 요약 | VLM summarization |
| VQ-Memory | "temporal 패턴" 압축 | Discrete token matching |
| **SAM2Act (기존)** | "spatial feature" 저장 | Implicit attention (불충분) |
| **Ours (제안)** | "structural position + transition" | Explicit graph-guided peak selection |

**핵심 novelty**: "Ambiguity가 발생하는 구조적 위치"에 대해서만 memory-guided selection을 적용.
모든 keyframe에 generic memory를 쓰는 것이 아니라,
**multi-peak이 발생한 ambiguous keyframe에서만 graph가 활성화**된다.

이것은 KC-VLA의 "전체 trajectory를 기억"이나 MemoryVLA의 "모든 것을 요약"과 다르다.
**구조적 ambiguity 위치를 data-driven으로 발견하고, 그 위치에서만 transition을 학습**한다.

---

### C.3 제안 아키텍처: Graph-Guided Peak Selection Network (GPSN)

#### 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage1 (Coarse Branch) — Multi-Peak Heatmap                      │
│                                                                   │
│ observation → SAM2Act encoder → transformer → decoder             │
│   → trans_coarse: multi-peak heatmap [bs, num_img, 1, H, W]      │
│                                                                   │
│ Peak 추출:                                                         │
│   peaks = extract_topk_peaks(trans_coarse, k=3, nms_distance=5)   │
│   → [{pos_2d, score, view_idx}, ...]                              │
│                                                                   │
│ Ambiguity 판단:                                                    │
│   if num_significant_peaks >= 2:                                  │
│     → Graph-guided selection 활성화                                │
│   else:                                                           │
│     → Stage1 heatmap 그대로 사용 (fast path)                       │
│                                                                   │
│ 각 peak에서 node embedding 추출:                                    │
│   peak_embeds = PeakNodeEmbed(encoder_feat, peak_positions)       │
│                                                                   │
└──────────────┬────────────────────────────────────────────────────┘
               │ peaks + peak_embeds + ambiguity_flag
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage2 (Memory Branch) — Graph-Guided Peak Selection              │
│                                                                   │
│ A. SAM2 Memory Attention (기존, 변경 없음):                         │
│    memory_feat = SAM2_cross_attention(curr_feat, memory_bank)     │
│                                                                   │
│ B. Node Embedding 추출:                                            │
│    curr_node = NodeEmbedHead(memory_feat)       # [bs, D_node]    │
│    prev_node = memory_bank[-1].node_embed       # 직전 저장        │
│                                                                   │
│ C. Language-Conditioned Transition Scoring:                        │
│    lang_emb = lang_encoder(instruction)         # [bs, D_lang]    │
│    context = ContextEncoder(prev_node, curr_node, lang_emb)       │
│    for each peak_i:                                               │
│      score_i = dot(context_proto, peak_embed_i) # [bs]            │
│    peak_weights = softmax([score_0, ..., score_k])                │
│                                                                   │
│ D. Heatmap Reweighting:                                            │
│    trans_refined = reweight(trans_coarse, peaks, peak_weights)    │
│                                                                   │
│ E. Memory 업데이트:                                                 │
│    memory_bank.add(features, pos_enc, curr_node_embed)            │
│                                                                   │
└──────────────┬────────────────────────────────────────────────────┘
               │ trans_refined
               ▼
           decoder → final heatmap → argmax → 3D waypoint
```

#### C.3.1 PeakNodeEmbedding Module

```python
class PeakNodeEmbedding(nn.Module):
    """Stage1 encoder feature에서 각 peak 위치의 node embedding 추출"""
    def __init__(self, feat_dim=128, node_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
    
    def forward(self, feature_map, peak_positions):
        """
        feature_map: [bs, C, H, W] — Stage1 encoder의 intermediate feature
        peak_positions: [bs, k, 2] — 각 peak의 (row, col) 좌표
        Returns: [bs, k, node_dim] — L2-normalized node embeddings
        """
        peak_feats = bilinear_sample(feature_map, peak_positions)  # [bs, k, C]
        return F.normalize(self.proj(peak_feats), dim=-1)
```

#### C.3.2 Language-Conditioned Transition Scorer

KC-VLA와의 핵심 차이: KC-VLA는 task_id로 FiLM conditioning하지만,
우리는 **language instruction embedding**으로 conditioning한다.
이것은 3개 task의 language instruction이 다르기 때문에,
같은 graph 구조가 instruction에 따라 다른 transition을 학습할 수 있게 한다.

```python
class LanguageConditionedTransitionScorer(nn.Module):
    """
    prev_node + curr_node + language → 각 candidate peak의 score
    Language가 task와 phase 정보를 동시에 전달
    """
    def __init__(self, node_dim=64, lang_dim=128):
        super().__init__()
        # Language → FiLM parameters (KC-VLA에서 영감)
        self.lang_film = nn.Sequential(
            nn.Linear(lang_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim * 2),  # gamma, beta
        )
        # Context: prev + curr (language-modulated)
        self.context_proj = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
        )
        self.score_proj = nn.Linear(node_dim, node_dim)
    
    def forward(self, prev_node, curr_node, peak_embeds, lang_emb):
        """
        prev_node: [bs, node_dim]
        curr_node: [bs, node_dim]
        peak_embeds: [bs, k, node_dim]
        lang_emb: [bs, lang_dim]
        
        Returns: [bs, k] — peak selection logits
        """
        # Language-modulated node representations
        gamma_beta = self.lang_film(lang_emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        prev_mod = gamma * prev_node + beta
        curr_mod = gamma * curr_node + beta
        
        context = self.context_proj(torch.cat([prev_mod, curr_mod], dim=-1))
        next_proto = self.score_proj(context)
        
        # Dot product scoring
        scores = torch.bmm(peak_embeds, next_proto.unsqueeze(-1)).squeeze(-1)
        return scores
```

#### C.3.3 Adaptive Ambiguity Gate

모든 keyframe에서 graph를 쓰면 불필요한 overhead. Multi-peak이 의미 있는 경우에만 활성화:

```python
class AmbiguityGate(nn.Module):
    """Peak 분포의 entropy로 ambiguity 판단"""
    def __init__(self, threshold=0.3):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, heatmap_logits):
        """
        heatmap_logits: [bs, num_img, 1, H, W]
        Returns: [bs] boolean — True = ambiguous → graph 활성화
        """
        hm = F.softmax(heatmap_logits.flatten(-2), dim=-1)  # [bs, num_img, H*W]
        
        # View별 entropy 계산
        entropy = -(hm * (hm + 1e-8).log()).sum(dim=-1)  # [bs, num_img]
        max_entropy = entropy.max(dim=-1).values           # [bs]
        
        # Entropy가 높으면 ambiguous (multi-peak)
        return max_entropy > self.threshold
```

---

### C.4 학습 전략

#### C.4.1 Loss 구성

```python
total_loss = (
    L_heatmap                                    # 기존 heatmap CE loss
    + lambda_peak * L_peak_select                # Graph peak selection
    + lambda_contrastive * L_node_contrastive    # Node embedding 형성
    + lambda_transition * L_transition           # Transition 학습
)
```

| Loss | 목적 | Supervision 출처 |
|------|------|----------------|
| `L_heatmap` | 최종 heatmap 정확도 | GT keyframe position (기존) |
| `L_peak_select` | 올바른 peak 선택 학습 | GT position과 가장 가까운 peak (자동) |
| `L_node_contrastive` | 같은 위치 → 같은 node embed | 3D position 거리 < threshold (자동) |
| `L_transition` | prev+curr → next 전이 학습 | Sequence 순서 (자동) |

**모든 supervision이 기존 데이터에서 자동 추출 가능** — 추가 annotation 불필요.

#### C.4.2 학습 순서

```
Phase 1: Stage1 Multi-Peak 학습 (기존 v6.1)
  - SE(3) mixup으로 ambiguous KF에 multi-peak 유도
  - 출력: multi-peak capable Stage1 weights
  
Phase 2: Stage2 Graph + Memory 공동 학습 (신규)
  - Stage1 weights freeze
  - SAM2 memory attention + Graph peak selection 동시 학습
  - 학습 데이터: 기존 MemoryBench 3-task training set
  
  구체적으로:
  for each episode:
    prev_node = None
    for each keyframe observation:
      # Stage1 forward (frozen)
      trans_coarse, peaks, peak_embeds = stage1_forward(obs)
      
      # Stage2 forward (trainable)
      memory_feat = sam2_memory_attention(obs, memory_bank)
      curr_node = node_embed_head(memory_feat)
      
      if ambiguity_gate(trans_coarse):
        peak_scores = transition_scorer(prev_node, curr_node, peak_embeds, lang_emb)
        peak_weights = softmax(peak_scores)
        trans_refined = reweight(trans_coarse, peaks, peak_weights)
        
        L_peak += peak_selection_loss(peak_scores, peaks, gt_position)
      else:
        trans_refined = trans_coarse
      
      L_heatmap += heatmap_loss(trans_refined, gt_heatmap)
      L_contrastive += node_contrastive(all_nodes, all_positions)
      if prev_node is not None:
        L_transition += transition_prediction(prev_node, curr_node, next_node_gt)
      
      memory_bank.add(memory_feat, pos_enc, curr_node)
      prev_node = curr_node.detach()
```

---

### C.5 구체 예시: 3 Task에서의 동작

#### C.5.1 put_block_back — KF4 vs KF7 (Revisit Ambiguity)

```
KF4 (button으로 가야 함):
  Stage1 multi-peak: peak_A = button 방향, peak_B = 아래로
  prev_node = KF3 (center_low, grasp) → "방금 block을 잡았음"
  curr_node = center_high, memory에서 "button 안 눌림" 정보
  
  TransitionScorer:
    context = [grasp_low → center_high] + lang("press the button")
    → score_A(button) = high, score_B(아래) = low
    → peak_A 선택

KF7 (아래로 내려야 함):
  Stage1 multi-peak: 동일한 peak_A, peak_B
  prev_node = KF6 (button_low, release) → "방금 button을 눌렀음"
  curr_node = center_high, memory에서 "button 눌림" 정보
  
  TransitionScorer:
    context = [button_low → center_high] + lang("put back")
    → score_A(button) = low (이미 함), score_B(아래) = high
    → peak_B 선택
```

**핵심**: prev_node가 다름 (KF3 vs KF6) → context가 달라짐 → 같은 peaks에 대해 다른 score.

#### C.5.2 put_block_back — KF9 (Memory-Dependent Variation)

```
KF9 (원래 patch로 가야 함):
  Stage1 multi-peak: 4개 patch 방향의 peaks (peak_A~D)
  prev_node = KF8 (center_low, release) → "block을 center에 놓았음"
  curr_node = center_high
  Memory bank: KF0의 memory 포함 → "처음에 block이 target1에 있었음"
  
  Memory attention이 KF0 feature를 강하게 attend →
  curr_node에 "target1 방향" 정보 내재
  
  TransitionScorer:
    context = [center_low → center_high(target1-informed)]
    → score_A(target1 방향) = high
    → peak_A 선택
```

**핵심**: Memory attention이 curr_node에 episode-specific 정보를 주입하고,
graph가 이를 활용하여 올바른 peak 선택.

#### C.5.3 reopen_drawer — KF0 vs KF4 (Revisit) + KF6 (Mem-dep)

```
KF0 (서랍 앞으로):
  Stage1 multi-peak: 서랍 앞(z=0.949) + 서랍 앞(z=1.033) + button
  prev_node = None (첫 step)
  curr_node = HOME, memory에서 "서랍이 열려있음" (초기 상태)
  TransitionScorer + lang("close the opened drawer"):
    → 열린 서랍 방향 peak 선택

KF4 (button으로):
  Stage1 multi-peak: 서랍 앞 + button
  prev_node = KF3 (drawer_closed) → "서랍을 닫았음"
  curr_node = HOME, memory에서 "서랍 닫힘"
  TransitionScorer + lang("push the button"):
    → button peak 선택

KF6 (서랍 안쪽으로):
  Stage1 multi-peak: bottom(z=0.949) + middle(z=1.033) 서랍
  prev_node = KF5 (button_low) → "button 눌렀음"
  curr_node = button 위치, memory에서 "아까 bottom/middle을 닫았음"
  TransitionScorer + lang("open the previous drawer"):
    → Memory가 KF0~3의 서랍 정보를 curr_node에 주입
    → 해당 서랍 방향 peak 선택
```

---

### C.6 기존 방법 대비 Novelty 정리

#### Novelty 1: Structural Ambiguity-Aware Memory

| 기존 | 제안 |
|------|------|
| KC-VLA: 모든 keyframe에 phase counter | Multi-peak 발생 시에만 graph 활성화 |
| MemoryVLA: 모든 step에 VLM summarization | Ambiguity gate로 선택적 활성화 |
| SAM2Act: 모든 step에 generic memory attention | Graph가 ambiguous step에서만 peak selection |

**Ambiguity가 없는 step (75~80%)에서는 Stage1 heatmap을 그대로 사용** → overhead 최소화.

#### Novelty 2: Graph-Based Transition Learning (Peak = Node)

| 기존 | 제안 |
|------|------|
| KC-VLA: phase embedding은 학습되지만 transition은 implicit | TransitionScorer가 prev+curr → peak 선택을 명시적으로 학습 |
| Generic graph memory: node가 하드코딩 | Peak 위치에서 자동 추출된 node embedding |
| Phase counter: monotonic | Graph edge: 비-monotonic transition 가능 (revisit) |

**Stage1의 multi-peak이 graph의 candidate node를 자연스럽게 정의.**
별도의 graph construction이 필요 없이, peak 자체가 node.

#### Novelty 3: Language-Conditioned FiLM Transition (KC-VLA 영감)

| 기존 | 제안 |
|------|------|
| KC-VLA: task_id embedding (discrete, 학습 시 고정) | Language embedding (continuous, 새 instruction에 일반화 가능) |
| SAM2Act: language가 encoder에만 사용 | Language가 transition scoring에도 직접 참여 |

**"press the button" vs "put back" vs "close the drawer" 같은 instruction이
transition score에 직접 영향 → 같은 observation에서도 instruction에 따라 다른 peak 선택.**

---

### C.7 구현 계획

#### 수정 파일

| 파일 | 수정 내용 | 난이도 |
|------|-----------|:---:|
| `mvt_sam2_single.py` | GPSN 모듈 통합, peak 추출, graph forward | 중 |
| `sam2act_agent.py` | L_peak, L_contrastive, L_transition 추가 | 중 |
| `sam2act/utils/graph_peak_selector.py` | **신규**: GPSN 모듈 전체 | 중 |

#### 신규 모듈 파라미터

| Module | 파라미터 수 |
|--------|:---:|
| PeakNodeEmbedding | ~12K |
| LanguageConditionedTransitionScorer | ~20K |
| NodeEmbedHead | ~12K |
| AmbiguityGate | 0 (threshold 기반) |
| **총** | **~44K (negligible)** |

SAM2Act 전체 파라미터 대비 0.01% 미만 → 추가 학습 비용 무시 가능.

#### 평가 계획

| 실험 | 목적 | 지표 |
|------|------|------|
| Stage2 Graph vs Stage2 Memory-only | Graph의 효과 | 3-task average success rate |
| Ambiguous KF only 분석 | Revisit/Mem-dep에서의 개선 | KF4,7,9 (put_block), KF0,4,6 (reopen) 성공률 |
| Ablation: w/o language conditioning | Language의 기여 | 3-task multi-task 성능 |
| Ablation: w/o ambiguity gate | Gate의 효과 | 비-ambiguous KF에서의 성능 유지 확인 |
| Ablation: w/o contrastive loss | Node embedding 질 | Graph 내 node 분리도 시각화 |

---

### C.8 요약

```
문제:
  Stage1 multi-peak이 복수 후보를 제안하지만,
  Stage2의 memory attention은 이 중 어떤 peak을 선택할지 직접 결정 못함.

해결:
  Peak = Graph Node로 대응시키고,
  TransitionScorer(prev_node, curr_node, peak_embeds, lang)가
  각 peak의 선택 확률을 명시적으로 계산.

Novelty:
  1. Ambiguity-aware selective activation (비-ambiguous step은 bypass)
  2. Data-driven graph (peak에서 자동 node 추출, 하드코딩 없음)
  3. Language-conditioned transition (FiLM으로 instruction이 peak 선택에 참여)

결과 기대:
  - Ambiguous KF에서의 정확한 peak 선택 → 성공률 향상
  - Non-ambiguous KF에서의 성능 유지 (gate로 bypass)
  - 3-task multi-task에서 language conditioning의 자연스러운 task 구분
```
