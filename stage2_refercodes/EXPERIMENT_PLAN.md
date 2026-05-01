# Stage2 실험 계획

## 4가지 Stage2 아이디어 요약

| # | 이름 | 핵심 메커니즘 | 파라미터 | 복잡도 | 영감 |
|:-:|------|-------------|:-------:|:-----:|------|
| 1 | **GPSN** | Peak=Node, TransitionScorer, AmbiguityGate | ~90K | 중 | 우리 설계 |
| 2 | **FiLM Memory** | FiLM으로 memory attention bias 조절 | ~213K | 중 | KC-VLA |
| 3 | **Dual Memory** | Spatial + Episodic Graph 분리 | ~70K | 중 | MemoryVLA |
| 4 | **Contrastive Phase** | Phase embedding + peak selection | ~38K | **낮** | KC-VLA contrastive |

## 각 아이디어 상세

### Idea 1: GPSN (Graph-Guided Peak Selection Network)
- **파일**: `idea1_gpsn.py`, `idea1_config.yaml`
- **구조**: PeakExtractor → PeakNodeEmbedding → LanguageConditionedTransitionScorer → AmbiguityGate → Heatmap Reweight
- **Novelty**: Peak이 곧 graph node. Ambiguity가 없으면 bypass (overhead 최소)
- **Loss**: peak_selection + node_contrastive + transition_prediction
- **장점**: 명시적 graph 구조, language conditioning
- **단점**: Peak 추출 + scoring 파이프라인이 길어서 학습 불안정 가능

### Idea 2: FiLM-Conditioned Keyframe Memory
- **파일**: `idea2_film_memory.py`, `idea2_config.yaml`
- **구조**: PhasePredictor(Transformer) → FiLMBiasGenerator(lang → gamma,beta → bias) → memory attention에 additive bias
- **Novelty**: KC-VLA의 FiLM을 binary detection이 아닌 continuous memory attention에 적용. 기존 memory_attention.py 수정 불필요
- **Loss**: node_contrastive + bias L2 regularization
- **장점**: 기존 SAM2 memory attention 구조 유지, 변경 최소
- **단점**: Phase를 Transformer로 예측해야 해서 학습 초기 불안정 가능

### Idea 3: Dual-Level Memory (Spatial + Episodic Graph)
- **파일**: `idea3_dual_memory.py`, `idea3_config.yaml`
- **구조**: SAM2 spatial memory(기존) + EpisodicGraphMemory(node list + transition scoring)
- **Novelty**: MemoryVLA의 dual memory를 VLM 없이 graph embedding으로 구현 (cheap)
- **Loss**: peak_selection + node_contrastive + transition_prediction
- **장점**: 두 레벨의 memory가 상호보완
- **단점**: Idea 1과 구조적으로 유사, 차별화 포인트가 약할 수 있음

### Idea 4: Contrastive Phase-Aware Memory Selection
- **파일**: `idea4_contrastive_phase.py`, `idea4_config.yaml`
- **구조**: PhaseEncoder(MLP) → PeakSelector(dot-product) + Contrastive Loss with Hard Negatives
- **Novelty**: Phase discovery가 완전 자동 (position clustering, 하드코딩 없음). Hard negative mining이 정확히 ambiguous pair(KF4 vs KF7)를 타겟
- **Loss**: contrastive_phase + peak_selection + phase_consistency
- **장점**: 가장 단순, 가장 적은 파라미터, hard negative이 문제를 직접 공격
- **단점**: Graph 구조 없이 MLP만으로 phase를 구분할 수 있는지 불확실

## 추천 실험 순서

### Phase 1: Baseline + 가장 단순한 것 (1주)

```
실험 0: SAM2Act+ baseline (memory only, no graph)
  → Stage1 v6.1 (3-task) weights freeze → Stage2 memory attention만 학습
  → 기존 SAM2Act+ 재현 실험
  → 이것이 모든 비교의 baseline

실험 1: Idea 4 (Contrastive Phase) ★ 먼저
  → 가장 단순 (38K params, MLP only)
  → Hard negative mining이 ambiguity를 직접 타겟
  → 빠르게 "phase 구분이 peak selection에 효과 있는지" 검증
  → 실패해도 빠르게 원인 파악 가능
```

### Phase 2: 핵심 아이디어 검증 (1주)

```
실험 2: Idea 1 (GPSN) 
  → Peak=Node graph + TransitionScorer
  → AmbiguityGate 효과 검증
  → Language conditioning 효과 검증

실험 3: Idea 2 (FiLM Memory)
  → 기존 memory attention에 최소 개입
  → FiLM bias가 memory selection을 개선하는지
```

### Phase 3: 조합 + Ablation (1주)

```
실험 4: Idea 3 (Dual Memory) 또는 Best of 1+4 조합
  → Phase 2 결과 기반으로 가장 효과적인 요소 조합
  → 예: Idea 4의 contrastive phase + Idea 1의 AmbiguityGate

실험 5: Ablation studies
  → w/o language conditioning
  → w/o contrastive loss
  → w/o ambiguity gate
  → w/o multi-peak Stage1 (single-peak baseline)
```

## 실험 명령어 (예시)

```bash
# Baseline: Stage2 memory only
torchrun --nproc_per_node=4 train_plus.py \
  --exp_cfg_opts "tasks put_block_back,rearrange_block,reopen_drawer \
    exp_name stage2_baseline wandb False" \
  --mvt_cfg_opts "use_graph_peak_select False use_film_memory False \
    use_contrastive_phase False"

# Idea 1: GPSN
torchrun --nproc_per_node=4 train_plus.py \
  --exp_cfg_opts "tasks put_block_back,rearrange_block,reopen_drawer \
    exp_name stage2_idea1_gpsn wandb False" \
  --mvt_cfg_opts "use_graph_peak_select True"

# Idea 2: FiLM Memory
torchrun --nproc_per_node=4 train_plus.py \
  --exp_cfg_opts "tasks put_block_back,rearrange_block,reopen_drawer \
    exp_name stage2_idea2_film wandb False" \
  --mvt_cfg_opts "use_film_memory True"

# Idea 4: Contrastive Phase (먼저 실행 추천)
torchrun --nproc_per_node=4 train_plus.py \
  --exp_cfg_opts "tasks put_block_back,rearrange_block,reopen_drawer \
    exp_name stage2_idea4_contrastive wandb False" \
  --mvt_cfg_opts "use_contrastive_phase True"
```

## 전제 조건

1. **Stage1 v6.1 학습 완료** (현재 진행 중, ~90 epoch)
2. **Stage1 checkpoint**에서 multi-peak이 살아있는지 최종 확인
3. **Stage2 replay buffer** 생성 (temporal sequence로, num_maskmem+1 keyframes)
4. **Evaluation pipeline** 준비 (CoppeliaSim headless eval)

## 평가 지표

| 지표 | 설명 |
|------|------|
| **Task success rate** | 3-task average (per-task도 보고) |
| **Ambiguous KF accuracy** | KF4,7,9 (put_block), KF0,4,6 (reopen) 정확도 |
| **Peak selection accuracy** | Multi-peak 중 올바른 peak 선택 비율 |
| **p1/p2 ratio** | Stage2 output의 multi-peak 유지 정도 |
