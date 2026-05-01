# 관련 논문 분석 및 Novelty 평가

> 작성 기준: 실제 코드와 논문을 읽고, 솔직하고 비판적으로 분석.  
> 각 아이디어에 대해 "차용된 것"과 "진정으로 새로운 것"을 명확히 구분.

---

## Section 1: 논문별 상세 분석

### 1.1 KC-VLA (arxiv 2603.01465)

**문제**: VLA 모델이 long-horizon task에서 과거 상태에 의존하는 non-Markovian 의사결정을 못 함.

**방법 (코드 기반 상세 분석)**:

KC-VLA는 3-stage 파이프라인:

1. **Stage 1 — Contrastive Backbone 학습** (`stage1_network.py`, `train_stage1.py`):
   - ResNet18 (ImageNet pretrained) + projector → 128-dim L2-normalized embedding
   - TripletMarginLoss (margin=1.0)로 학습
   - Negative 전략: 시간 근접 non-keyframe (33%), 같은 task 다른 phase (33%), 다른 task (33%)
   - 결과: "같은 task-phase의 keyframe embedding은 가깝고, 다른 것은 먼" backbone

2. **Stage 2 — Task-Modulated Keyframe Selector** (`network.py`, `train_stage2.py`):
   - Frozen contrastive backbone → vis_projector → 128-dim
   - **FiLM Query 생성** (핵심 기법):
     ```
     task_emb = task_embedding(task_id)       # discrete, [B, 128]
     phase_emb = phase_embedding(phase_id)    # discrete, [B, 128]
     gamma, beta = FiLM_generator(task_emb)   # MLP: 128→256→256, split
     logic_query = gamma * phase_emb + beta   # [B, 128]
     ```
   - Self-attention (3-frame window temporal context) → Cross-attention (logic_query vs vis_contextual) → Binary classifier
   - BCEWithLogitsLoss (pos_weight=5)
   - **Phase counter**: keyframe 검출 시 monotonic increment

3. **Stage 3 — VLA (GR00T N1.5) + HistoryQueryModule**:
   - `HistoryQueryModule`: 32개 learnable query tokens + phase embedding
   - Cross-attention: phase-conditioned queries가 history keyframe features에서 정보 추출
   - 시간 + 뷰 위치 인코딩, padding mask 처리, LayerNorm + dropout

**한계**:
- Phase counter가 **monotonic** → revisit (phase 역행) 불가능
- Task별 `max_keyframes` **하드코딩** 필요
- **Discrete task_id** → 새로운 task에 대한 zero-shot 일반화 불가
- Keyframe detection은 별도 모듈 → 오검출 시 연쇄 오류 (cascading error)
- Global ResNet feature만 사용 → spatial resolution 낮음

**우리와의 관계**:
- FiLM conditioning 기법을 차용 (Idea 1, 2, 3 모두 FiLM 사용)
- 단, 우리는 discrete task_id 대신 **continuous language embedding** 사용
- KC-VLA의 phase counter → 우리는 **data-driven graph/phase prediction**으로 대체
- Contrastive learning 전략 일부 차용 (Idea 4)

---

### 1.2 SAM2Act / SAM2Act+ (arxiv 2501.18564)

**문제**: 로봇 조작 시 multitask 능력, 미지 환경 일반화, 공간 기억(spatial memory) 동시 달성.

**방법 (코드 기반 분석)**:

1. **SAM2Act (Base)**:
   - Multi-view transformer (MVT) 기반, 5-view 렌더링
   - SAM2 vision encoder를 feature extractor로 활용 (LoRA fine-tune)
   - Heatmap-based waypoint prediction (single-peak)
   - Language conditioning via CLIP embedding

2. **SAM2Act+ (Memory)**:
   - SAM2의 video segmentation memory mechanism을 조작에 적용
   - **MemoryAttention** (`memory_attention.py`): Self-attention + Cross-attention + FFN
     - `_forward_sa`: 현재 frame self-attention
     - `_forward_ca`: 현재 frame ↔ memory bank cross-attention (RoPE attention)
     - `memory_attn_mask`: additive bias 지원 (우리 Idea 2가 여기에 개입)
   - **Memory Bank**: 과거 keyframe의 (feature, pos_enc) 저장, multi-view별 독립
   - `num_maskmem`: 저장할 최대 memory entry 수

3. **MemoryBench**:
   - 3개 task: put_block_back (12 KF), rearrange_block (11 KF), reopen_drawer (10 KF)
   - Memory-dependent keyframe 비율: 18~30% (task에 따라)

**한계 (코드 분석 기반)**:
- Stage1이 memory 없이 학습 → **ambiguous KF에서 collapse** (한쪽만 예측)
- Memory attention은 feature를 **부드럽게 수정**할 뿐, **peak 선택 메커니즘이 없음**
- Memory에 **phase/structural 정보가 명시적으로 없음**
- Revisit ambiguity (KF4 vs KF7)에서 시각적 차이가 sub-pixel → Stage1 실패

**우리와의 관계**: 우리의 **직접적 baseline**. 우리는 SAM2Act+의 한계를 해결하려 함.

---

### 1.3 MemoryVLA (arxiv 2508.19236)

**문제**: VLA 모델이 temporal context를 무시 → non-Markovian task 실패.

**방법**:
- 인지과학 영감: **working memory** (즉각적 제어) + **hippocampal-like 장기 기억**
- Pretrained VLM으로 observation → perceptual + cognitive token 인코딩
- **Memory bank**: low-level detail + high-level semantics 이중 저장
- Working memory가 결정에 관련된 과거 정보를 retrieve, adaptive fusion
- Diffusion-based action module로 시간적 맥락 반영한 action 생성

**성능**: SimplerEnv-Bridge 71.9%, Fractal 72.7%, LIBERO-5 96.5%, real 84.0% (+26 long-horizon)

**한계**:
- **VLM 기반 summarization** → 연산 비용 높음 (7B+ 파라미터)
- Phase boundary가 명확하지 않은 task에서 cognitive memory 품질 저하
- Memory bank이 **unstructured bag of tokens** → 구조적 transition 정보 없음

**우리와의 차이**:
- MemoryVLA: "무엇을 기억할지"를 VLM이 결정 (비쌈, implicit)
- 우리: spatial attention + explicit graph structure (저렴, ~45K params)
- MemoryVLA의 dual memory 구조 → Idea 3 (Dual-Level Memory)에 영감

---

### 1.4 VQ-Memory (arxiv 2603.09513)

**문제**: 기존 벤치마크가 simple pick-and-place에 집중, non-Markovian + articulated object 미반영.

**방법**:
- **RuleSafe Benchmark** 제안: 금고 조작 (키 잠금, 비밀번호, 논리 잠금) — multi-stage reasoning
- **VQ-VAE** 기반 temporal encoding:
  - 과거 proprioceptive states → discrete latent tokens
  - Codebook으로 양자화 → 고정 크기 memory, long-horizon에도 scalable
  - "low-level noise 필터링 + high-level task-phase context 보존"
- VLA/Diffusion policy에 plug-in 가능

**한계**:
- **Proprioceptive only** — visual information 미활용
- Codebook 크기가 task-dependent
- Fine-grained spatial 정보 손실
- Articulated object에 특화 — tabletop manipulation과 직접 비교 어려움

**우리와의 차이**:
- VQ-Memory: temporal pattern을 discrete하게 압축 (proprioception 기반)
- 우리: pixel-level spatial feature를 memory bank에 유지 (SAM2 memory)
- VQ-Memory의 "과거 정보 압축" 아이디어는 우리 node embedding과 개념적 유사

---

### 1.5 RoboMAP (arxiv 2510.10912)

**문제**: 언어 기반 로봇 시스템이 spatial reasoning을 discrete point로 collapse → 노이즈/모호성에 취약.

**방법**:
- **Continuous, adaptive affordance heatmap**으로 spatial target 표현
- Uncertainty를 dense probability distribution으로 인코딩
- 50x 속도 향상, real-world 82% 성공률, zero-shot navigation 일반화

**한계**:
- **Single-step affordance**에 특화 — multi-step sequential manipulation 미검증
- Memory 메커니즘 없음 — non-Markovian task 해결 불가
- Heatmap은 "어디를 잡을지"이지 "어떤 순서로"가 아님

**우리와의 관련성**:
- Heatmap 기반 표현의 장점을 공유 (continuous, uncertainty-aware)
- RoboMAP은 single-step, 우리는 multi-step sequential → 상호 보완적
- **우리의 multi-peak heatmap은 RoboMAP의 adaptive heatmap과 다른 맥락**: 
  RoboMAP은 single target의 uncertainty, 우리는 multiple valid targets의 존재

---

### 1.6 RMBench (arxiv 2603.01229)

**문제**: Memory-aware policy 평가를 위한 체계적 벤치마크 부재.

**방법**:
- **9개 manipulation task**, 다양한 memory 복잡도
- **Mem-0 Policy**: modular design with explicit memory components
  - Execution Module: Qwen3-VL-2B, single-task fine-tuning
  - Planning Module: Qwen3-VL-8B-Instruct + LoRA
  - M(n) task에서 Planning Module이 key memory를 reasoning

**Task 유형** (코드 분석):
- `put_back_block`: 원래 위치 기억 (우리 put_block_back과 거의 동일)
- `cover_blocks`: RGB 순서 기억
- `swap_blocks`: swap 순서/경유지 기억  
- `press_button`: 숫자 카운팅
- `battery_try`: 시행착오 결과 기억
- `observe_and_pickup`: target 외형 기억

**한계**:
- SAPIEN + dual-arm ALOHA → 우리와 환경/embodiment 다름
- VLM 기반 Planning Module은 latency 높음
- 우리 MemoryBench (3 task)보다 task 수 많지만, 시뮬레이터 변환 필요

**우리와의 관계**: 
- Memory taxonomy가 우리 ambiguity classification과 유사 (위치 기억 vs 순서 기억)
- RMBench의 task 설계 참고 가능 (MemoryBench 확장 시)
- Mem-0의 explicit memory와 우리 graph memory는 개념적으로 유사하나 구현이 다름

---

## Section 2: 이미 존재하는 것들 (Prior Art)

### 2.1 Multi-Peak / Multi-Modal Action Prediction

| 기존 연구 | 방법 | 한계 |
|-----------|------|------|
| **RoboMAP** (2510.10912) | Continuous affordance heatmap | Single-step only, memory 없음 |
| **PerAct** + heatmap variants | Voxel-space probability | Single-peak, collapse on ambiguity |
| **Diffusion Policy** | Multi-modal action distribution | Implicit, peak 선택 어려움 |
| **BESO/BeT** | Discrete action set + scoring | Action space discretization 필요 |

**결론**: Multi-peak heatmap을 **명시적으로 생성하고 선택하는** 메커니즘은 기존에 없음.
대부분 암묵적 multi-modality (diffusion) 또는 single-peak (heatmap)에 그침.

### 2.2 Memory-Based Manipulation

| 기존 연구 | Memory 형태 | 한계 |
|-----------|------------|------|
| **SAM2Act+** | Dense pixel-level memory bank | Implicit attention, peak 선택 불가 |
| **KC-VLA** | Sparse keyframe chain | Monotonic phase counter, discrete task_id |
| **MemoryVLA** | Dual perceptual+cognitive | VLM 의존 (7B+), 구조적 전이 없음 |
| **VQ-Memory** | Discrete VQ tokens (proprioception) | Spatial info 손실 |
| **Mem-0** (RMBench) | VLM planning + explicit memory | Latency, 환경 제한 |

**결론**: Memory를 **어디에/어떻게** 사용할지는 다양한 접근이 있으나,
"memory-dependent ambiguity가 있는 keyframe에서만 선택적으로 개입"하는 접근은 없음.

### 2.3 Graph-Based Task Understanding

| 기존 연구 | Graph 용도 |
|-----------|-----------|
| **Task Graph (TAMP 계열)** | 사전 정의된 task decomposition |
| **Scene Graph** | 물체 간 관계 인코딩 |
| **KC-VLA HistoryQueryModule** | Keyframe chain (sequential, no explicit graph) |

**결론**: Task graph를 **data-driven으로 heatmap peak에서 자동 추출**하는 것은 기존에 없음.
기존은 대부분 사전 정의 또는 scene-level.

### 2.4 Contrastive Learning for Phase/State Discrimination

| 기존 연구 | 대상 | 방법 |
|-----------|------|------|
| **KC-VLA Stage1** | Raw image → keyframe vs non-keyframe | TripletMarginLoss |
| **TCN (Time-Contrastive Networks)** | Video frame → temporal alignment | Multi-view contrastive |
| **R3M, VIP** | Video representation | Time-contrastive pretraining |

**결론**: Contrastive learning은 representation에 널리 사용.
단, **memory-enhanced feature에 대한 contrastive + hard-negative mining (same position, different phase)**은 
기존에 명시적으로 사용된 적 없음. KC-VLA는 raw image 수준에서만.

### 2.5 FiLM Conditioning in Robotics

| 기존 연구 | FiLM 용도 |
|-----------|----------|
| **KC-VLA** | task_id → FiLM → phase_emb modulation → keyframe detection |
| **FiLM (original, Perez et al. 2018)** | Language → visual reasoning modulation |
| **CLIPort** | Language → transport network modulation |

**결론**: FiLM은 conditioning 기법으로 널리 사용. KC-VLA가 로봇 keyframe detection에 적용.
우리가 **continuous language embedding으로 FiLM을 transition scoring에 사용**하는 것은
KC-VLA의 직접적 확장이지만 (1) continuous vs discrete, (2) transition vs detection이라는 차이.

---

## Section 3: 우리 4개 Stage2 아이디어의 Novelty 분석

### 공통 기반 (Stage1 Multi-Peak v6.1)

**차용**: 
- Heatmap-based action prediction (PerAct, SAM2Act에서)
- SE(3) data augmentation (기존 robotics에서 널리 사용)

**새로운 것**:
- **Scene-aware ambiguity classification**: EE 위치 + gripper 상태 + 추적된 object 위치를 
  함께 고려하여, "진정으로 ambiguous한 keyframe"만 multi-peak 레이블 부여
- 기존 multi-peak 접근은 무조건 multi-modal이거나 무조건 single-peak. 
  **조건부 multi-peak** (ambiguity가 있을 때만)은 새로움

---

### Idea 1: Graph-Guided Peak Selection Network (GPSN)

**차용된 것**:
- FiLM conditioning: KC-VLA에서 직접 차용 (FiLM_generator → gamma, beta)
- Node embedding via bilinear sampling: 표준 기법
- Contrastive loss for embedding: KC-VLA, R3M 등에서 널리 사용
- Peak extraction via NMS: Object detection의 표준 기법

**진정으로 새로운 것**:
- **Peak = Node 대응**: Multi-peak heatmap의 각 peak을 graph의 node로 자동 매핑.
  별도의 graph construction 없이 peak 위치에서 node embedding을 추출.
  → 기존에 이 대응 관계를 만든 논문 없음
- **Transition Scoring**: prev_node + curr_node + lang → 각 peak의 선택 확률 계산.
  KC-VLA는 phase counter (monotonic, hardcoded), 우리는 학습된 scorer (non-monotonic 가능)
- **Ambiguity Gate**: 비-ambiguous keyframe (75~80%)은 graph를 bypass.
  기존 memory 방법은 모든 step에 uniform하게 적용.

**Novel Combination**:
- SAM2 memory attention (spatial) + graph-based transition (structural) 결합
- Language-conditioned FiLM (KC-VLA 확장) + peak selection (새로움) 결합

---

### Idea 2: FiLM-Conditioned Keyframe Memory

**차용된 것**:
- FiLM bias generation: KC-VLA의 FiLM generator 직접 차용
- Phase prediction from history: Transformer-based sequence encoding (표준)
- Memory cross-attention: SAM2의 기존 MemoryAttention 그대로 사용

**진정으로 새로운 것**:
- **FiLM attention bias injection**: SAM2 MemoryAttentionLayer의 `_forward_ca()`에
  additive attn_bias를 주입. SAM2의 기존 코드를 수정하지 않고 bias만 추가.
  → 기존에 SAM2 memory attention에 task-conditioned bias를 넣은 연구 없음
- **Continuous phase prediction** (PhasePredictor): KC-VLA는 discrete phase counter,
  우리는 node history → Transformer → learnable query → continuous phase embedding.
  Non-monotonic transition 가능.
- **Memory-token-level selectivity**: 각 memory entry에 node embedding을 저장하고,
  FiLM-conditioned phase와의 compatibility로 개별 memory 가중.

**Novel Combination**:
- SAM2 memory architecture (기존) + FiLM bias (KC-VLA 확장) + continuous phase (새로움)

---

### Idea 3: Dual-Level Memory with Episodic Graph

**차용된 것**:
- Dual memory 구조: MemoryVLA에서 영감 (perceptual + cognitive 이중 memory)
- TransitionScorer: Idea 1과 동일한 FiLM-based scoring
- Node contrastive loss: 표준 contrastive learning
- Episodic graph (sequential edges): KC-VLA의 keyframe chain과 구조적으로 유사

**진정으로 새로운 것**:
- **경량 dual memory**: MemoryVLA는 VLM (7B+), 우리는 학습된 graph (~70K params).
  1/100,000 파라미터로 유사한 기능 구현
- **Per-view peak extraction + scoring**: 각 뷰별로 독립적으로 peak 추출 후 scoring.
  Multi-view 환경에 특화된 설계

**솔직한 평가**: Idea 3은 Idea 1의 변형. 핵심 차이는 per-view 처리와 
EpisodicGraphMemory 클래스의 명시적 분리 정도. **Novelty 한계가 있음**.

---

### Idea 4: Contrastive Phase-Aware Memory Selection

**차용된 것**:
- Contrastive learning: KC-VLA Stage1에서 직접 영감
- Phase embedding: 표준 기법
- Peak extraction + selection: Idea 1, 3과 공유
- Heatmap reweighting: Idea 1, 3과 공유

**진정으로 새로운 것**:
- **Automatic phase discovery**: 3D position + gripper state clustering으로 phase를 
  자동 발견. KC-VLA는 task별 phase를 하드코딩, 우리는 data-driven.
- **Hard-negative mining (same position, different phase)**: 
  "같은 EE 위치 + 같은 gripper 상태이지만 다른 next target" 쌍을 hard negative로 사용.
  이것은 revisit ambiguity (KF4 vs KF7)에 정확히 대응.
  → KC-VLA의 contrastive는 이런 세밀한 pair mining 없음
- **Proprioception 직접 활용**: PhaseEncoder가 pooled visual feature + proprio (x,y,z,gripper)를 
  함께 인코딩. Memory-enhanced feature에서의 phase encoding은 새로움.
- **Phase consistency loss**: 연속 step 간 phase embedding 변화를 smooth하게 유지.

**솔직한 평가**: FiLM을 사용하지 않아 아키텍처가 가장 단순. 
Contrastive learning 자체는 새롭지 않지만, **hard-negative mining 전략이 우리 문제에 매우 적합**.

---

## Section 4: Novelty 강도 랭킹

### Strong Novelty (기존에 없는 것)

1. **Scene-aware conditional multi-peak heatmap (Stage1 v6.1)**
   - "EE + gripper + object position이 동일하고 next target만 다를 때만 multi-peak"
   - 무조건적 multi-modal (diffusion)도, 무조건적 single-peak도 아닌 **조건부** 접근
   - 이를 위한 object position tracking + scene-aware clustering은 새로운 기법
   - **Prior art에 직접 대응하는 연구 없음**

2. **Peak-as-Node graph with ambiguity gating**
   - Multi-peak heatmap의 peak을 graph node로 자동 매핑하고,
     ambiguity가 있을 때만 graph-based selection 활성화
   - "언제 memory를 쓸지" 자체를 학습/판단하는 메커니즘
   - **Prior art에 직접 대응하는 연구 없음**

### Medium Novelty (기존 기법의 새로운 조합/적용)

3. **Language-conditioned FiLM transition scoring**
   - KC-VLA의 FiLM (discrete task_id → keyframe detection)을
   - continuous language embedding → transition scoring으로 확장
   - 기법 자체는 FiLM의 변형이지만, 적용 맥락이 새로움
   - **KC-VLA의 직접적 확장으로 인정해야 함**

4. **Hard-negative mining for revisit ambiguity (Idea 4)**
   - Contrastive learning 자체는 표준이지만,
   - "same position + same gripper + different next target" pair mining은
   - 우리 문제(revisit ambiguity)에 대한 새로운 적용
   - **KC-VLA contrastive의 문제 특화 확장**

5. **FiLM attention bias injection into SAM2 memory (Idea 2)**
   - SAM2의 기존 memory attention에 최소 침습적으로 개입
   - 구현적으로 우아하지만, 개념적으로는 attention bias의 표준적 사용

### Weak Novelty (점진적 개선)

6. **Continuous phase prediction (vs monotonic counter)**
   - KC-VLA의 phase counter → Transformer-based phase prediction
   - 개념적으로 자연스러운 확장. 구현이 새롭지만 아이디어는 예측 가능.

7. **Dual-level memory (Idea 3)**
   - MemoryVLA의 dual memory를 경량화한 것
   - "경량화"는 엔지니어링이지 novelty가 아님
   - Idea 1과의 차이가 불분명

8. **Episodic graph memory container**
   - 사실상 list of node embeddings. "Graph"라 부르지만 edge가 sequential뿐.
   - KC-VLA의 keyframe chain과 구조적으로 거의 동일.

---

## Section 5: 제안하는 Paper Story

### 가장 강력한 기여 조합

4개 아이디어 중 **개별적으로 가장 강한 것**은 없다. 
**Stage1 multi-peak + Stage2 peak selection의 2-stage 프레임워크 자체**가 가장 강한 기여.

### One-Sentence Contribution

> "Memory-dependent manipulation에서 발생하는 structural ambiguity를 
> **조건부 multi-peak heatmap (Stage1)** 과 **graph-guided peak selection (Stage2)** 의 
> 2-stage 파이프라인으로 해결하며, ambiguity가 없는 75~80%의 keyframe에서는 
> overhead 없이 bypass한다."

### 추천하는 Paper Framing

**Option A: "Ambiguity-Aware 2-Stage Memory Manipulation" (가장 추천)**

핵심 주장:
1. **관찰**: Memory-dependent task에서 memoryless policy는 ambiguous keyframe에서 collapse.
   그러나 전체 keyframe의 75~80%는 memory 없이도 해결 가능.
2. **Stage1**: Scene-aware analysis로 진정한 ambiguity만 multi-peak 허용. 나머지는 single-peak.
3. **Stage2**: Ambiguity gate가 multi-peak을 감지하면, memory-guided transition scorer가 올바른 peak 선택.
4. **결과**: Ambiguous KF에서의 정확도 대폭 향상, non-ambiguous KF에서의 성능 유지.

이 framing의 장점:
- **문제 정의가 명확**: "전부 memory를 쓰는 것은 과도하고, 전부 안 쓰는 것은 부족하다"
- **실증적 뒷받침**: AMBIGUITY_CLASSIFICATION.md의 상세 분석이 이 주장을 지지
- **기존 연구와의 차별점이 명확**: KC-VLA (전부 memory), SAM2Act (전부 memory), 우리 (선택적)

**Option B: "Conditional Multi-Peak Heatmap for Non-Markovian Manipulation"**

Stage1의 multi-peak 기법에 초점. 
장점: 기여가 깔끔. 단점: Stage2 없이는 실용적 가치 불분명.

**Option C: "Graph-Guided Transition Learning for Peak Selection in Memory-Dependent Tasks"**

Stage2에 초점. 
장점: 기술적 깊이. 단점: Stage1 multi-peak에 의존적이라 standalone 기여가 약함.

### Venue 추천

| Venue | 적합도 | 이유 |
|-------|:---:|------|
| **CoRL 2026** | ★★★★★ | 로봇 학습 + memory manipulation 정확히 부합 |
| **ICRA 2027** | ★★★★☆ | 넓은 acceptance rate, 실험 결과 강조 가능 |
| **RSS 2026** | ★★★★☆ | 문제 분석의 깊이 어필 가능 |
| **NeurIPS 2026** | ★★★☆☆ | Novelty가 ML 관점에서는 incremental. 로봇 workshop 추천 |

### 정직한 자기 평가

**강점**:
- 문제 분석이 매우 꼼꼼 (AMBIGUITY_CLASSIFICATION.md의 수준이 높음)
- "선택적 memory 활성화"라는 practical insight
- 기존 SAM2Act 위에 최소 파라미터 추가 (<0.1%)

**약점**:
- 4개 아이디어 중 어떤 것이 실제로 작동하는지 **아직 실험 결과가 없음**
- MemoryBench 자체가 3개 task로 규모가 작음
- Peak-as-Node graph는 개념적으로 매력적이나, "graph"라고 부르기에는 
  사실상 sequential list에 불과 (edge가 prev→curr뿐)
- FiLM transition scoring은 KC-VLA의 직접적 확장이라고 인정해야 함

### 제안: 4개 아이디어 중 최종 선택

1. **Idea 1 (GPSN)** 또는 **Idea 4 (Contrastive Phase)**를 메인으로 선택
   - Idea 1: 아키텍처가 풍부하고 story가 명확 (graph-guided)
   - Idea 4: 더 단순하고 hard-negative mining이 문제에 직결
2. **Idea 2 (FiLM Memory)**는 Idea 1의 보조 컴포넌트로 활용 가능
3. **Idea 3 (Dual Memory)**는 Idea 1과 중복이 많아 독립 기여로 내세우기 어려움

**최종 추천**: Idea 1 (GPSN) + Idea 4의 hard-negative mining을 결합.
Stage1 multi-peak + GPSN peak selection + contrastive phase loss로 구성하되,
**"선택적 ambiguity-aware"**를 핵심 narrative로 삼을 것.
