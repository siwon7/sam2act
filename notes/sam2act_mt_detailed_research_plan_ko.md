# SAM2Act-MT++ 상세 연구 계획서

이 문서는 현재 구현되어 있는 `SAM2Act` 멀티태스크 메모리 확장 코드를 기준으로 정리한 상세 연구 계획서다.

기준 구현 파일:

- [sam2act/mvt/config.py](/home/cv25/siwon/sam2act/sam2act/mvt/config.py)
- [sam2act/mvt/mvt_sam2.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2.py)
- [sam2act/mvt/mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py)
- [scripts/run_memorybench_mt_ablation.sh](/home/cv25/siwon/sam2act/scripts/run_memorybench_mt_ablation.sh)

---

## 1. 문제 정의

### 1.1 현재 실무 문제

`SAM2Act+`는 원래 `MemoryBench`에서 single-task 기준으로 보고되었지만, 현재 재현은 다음 문제를 동시에 안고 있다.

1. 과거 crash는 simulator/plugin 이슈였다.
2. crash를 분리해도 eval 성능은 낮거나 불안정하다.
3. MemoryBench 체크포인트는 RLBench보다 훨씬 민감하다.
4. 최근 routing 코드가 들어오면서 old checkpoint와의 호환성 문제가 있었다.
5. stage1/stage2 품질, replay setup, eval variance가 모두 결과를 흔든다.

### 1.2 연구 문제

single-task 재현이 안정화되더라도, `MemoryBench`는 태스크 수가 너무 적고 멀티태스크 메모리 연구용 benchmark로는 약하다.

우리가 풀고 싶은 핵심 질문은 다음이다.

`공유된 policy가 멀티태스크 long-horizon manipulation을 하면서도 task-specific episodic recall을 유지할 수 있는가?`

이 질문은 단순히

- joint training이 되는가
- task conditioning이 되는가

를 넘어서, 실제 메모리 메커니즘을 묻는다.

- 무엇을 쓸 것인가
- 무엇을 오래 남길 것인가
- 무엇을 읽을 것인가

---

## 2. 핵심 주장

### 2.1 메인 thesis

`MemoryBench의 멀티태스크 실패는 단순한 data sharing 문제가 아니라 memory lifecycle 문제다.`

태스크마다 다르다.

- 중요한 과거 상태가 다르고
- 필요한 기억 길이가 다르고
- 최근 local context가 더 중요한 경우와 sparse event recall이 더 중요한 경우가 다르다

### 2.2 방법 가설

`Task-conditioned dual-level memory와 adaptive event retention을 쓰면 shared FIFO memory와 routing-only memory보다 더 잘 작동할 것이다.`

---

## 3. 현재 메서드

### 3.1 메서드 이름

작업용 이름:

`SAM2Act-MT++: Task-Conditioned Dual-Level Memory with Adaptive Event Consolidation`

### 3.2 메서드 개요

현재 코드에는 stage2 기준으로 네 가지 variant가 이미 들어가 있다.

1. `shared_fifo`
2. `routing`
3. `dual_memory`
4. `full`

### 3.3 메모리 구조

메모리는 두 단계로 나뉜다.

- `local memory`
  - 최근 step들의 dense memory
  - short-term spatial context용

- `event memory`
  - 중요한 순간만 남기는 sparse memory
  - long-term episodic recall용

### 3.4 라우팅

task identity는 다음에 쓰인다.

- memory channel gating
- local/event memory fusion
- event write score 계산

### 3.5 유지 전략

event memory는 두 방식 중 하나로 관리된다.

- adaptive prune off: FIFO와 유사한 oldest drop
- adaptive prune on: importance 낮은 slot부터 제거

### 3.6 heatmap-aware event feature

full method에서는 predicted heatmap에서 아래 summary를 뽑는다.

- peak confidence
- entropy
- mean
- standard deviation

이 값들은 event write 판단의 spatial signal로 사용된다.

---

## 4. 도식

```text
입력 observation sequence
        |
        v
  SAM2 image encoder
        |
        v
  SAM2Act coarse branch
        |
        +------------------------------+
        |                              |
        v                              v
   local memory bank             event memory bank
   최근 dense slot                중요한 과거 slot
        |                              |
        +--------------+---------------+
                       |
                       v
         task-conditioned read routing
         - channel gate
         - local/event fusion
                       |
                       v
            memory-aware coarse feature
                       |
                       v
              translation heatmap output
                       |
                       +----------------------+
                       |                      |
                       v                      v
               heatmap summary         event write score
                       |                      |
                       +----------+-----------+
                                  |
                                  v
                     event memory write / prune
```

---

## 5. Variant 설명

### 5.1 shared_fifo

파일:

- [sam2act_plus_mt_shared_fifo.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_shared_fifo.yaml)

의미:

- 원래 shared memory baseline
- task routing 없음
- event memory 없음

역할:

- 가장 기본적인 멀티태스크 baseline

### 5.2 routing

파일:

- [sam2act_plus_mt_routing.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_routing.yaml)

의미:

- shared memory 유지
- task-conditioned routing만 켬

역할:

- retrieval selection만으로 gain이 나는지 확인

### 5.3 dual_memory

파일:

- [sam2act_plus_mt_dual_memory.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_dual_memory.yaml)

의미:

- local + event memory 사용
- local/event fusion 사용
- event write는 dense
- prune은 FIFO-like

역할:

- dual-level 구조만으로도 도움이 되는지 확인

### 5.4 full

파일:

- [sam2act_plus_mt_full.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_full.yaml)

의미:

- routing
- dual memory
- event write
- adaptive prune
- heatmap-aware feature

역할:

- 최종 제안 방법

---

## 6. Contribution

### 6.1 Contribution 1: Task-conditioned memory routing

새로운 점:

- task가 passive metadata가 아니라 실제 memory read를 바꾼다

중요한 이유:

- 멀티태스크 interference는 task마다 다르게 나타난다

관련 구현:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L258)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L604)

### 6.2 Contribution 2: Dual-level memory

새로운 점:

- recent local context와 sparse episodic event를 분리했다

중요한 이유:

- short-term manipulation context와 delayed recall은 같은 메모리 정책으로 다루기 어렵다

관련 구현:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L251)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L699)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L717)

### 6.3 Contribution 3: Adaptive event consolidation

새로운 점:

- event memory를 fixed FIFO로만 유지하지 않고 importance 기준으로 pruning할 수 있다

중요한 이유:

- SAM2Act 원논문에서 fixed window가 limitation으로 직접 언급된다

관련 구현:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L665)

### 6.4 Contribution 4: Heatmap-aware event writing

새로운 점:

- event write 판단에 spatial signal을 직접 쓴다

중요한 이유:

- 정말 중요한 과거 상태는 종종 명확한 spatial peak와 연결된다

관련 구현:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L652)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L811)

---

## 7. 왜 이 방법이 그럴듯한가

### 7.1 shared FIFO의 한계

멀티태스크 환경에서는

- 어떤 task는 최근 상태가 중요하고
- 어떤 task는 오래된 scene state가 중요하다

shared FIFO는 모든 task에 같은 retention policy를 강요한다.

### 7.2 routing-only의 한계

routing-only는 읽는 방식을 바꾸지만,

- 무엇이 메모리에 들어가는지
- 얼마나 오래 남는지

는 그대로 둔다.

### 7.3 dual memory의 장점

dual memory는 동시에 유지할 수 있다.

- 최근 정밀 local context
- 오래 남는 sparse event context

### 7.4 adaptive prune의 필요성

event memory가 sparse여도 FIFO라면 여전히 중요한 event를 덮어쓸 수 있다.

---

## 8. 참고 논문과 반영 포인트

### 8.1 SAM2Act

- 기본 구조
- MemoryBench 문제 정의
- fixed memory window limitation

### 8.2 MemoryVLA

- retrieval / fusion / consolidation 관점
- non-FIFO memory 관리 동기

### 8.3 ReMem-VLA

- dual-level memory 아이디어
- short-term / long-term 분리

### 8.4 RMBench

- memory benchmark framing
- architecture choice가 왜 중요한지에 대한 스토리

### 8.5 BridgeVLA

- spatially aligned event signal
- heatmap-aware memory idea

### 8.6 MemoAct

- short-term vs long-term memory framing 보강

### 8.7 CORAL

- task specialization baseline 아이디어

### 8.8 HSC-VLA

- 미래 확장으로 scene-cleared write 방향

---

## 9. 성능 비교 계획

### 9.1 성능 비교 원칙

성능 비교는 평균 점수 하나만으로 끝내면 안 된다.

반드시 네 가지를 봐야 한다.

1. `single-task 대비 multitask가 얼마나 손해/이득인지`
2. `joint training만으로 되는지, memory design이 필요한지`
3. `hard memory task가 실제로 좋아지는지`
4. `variance가 줄어드는지`

### 9.2 주 비교 축

#### Axis A. single-task vs multitask

```text
single-task SAM2Act+
vs
multitask shared_fifo
vs
multitask full
```

의미:

- shared_fifo가 많이 떨어지면 interference가 real
- full이 gap을 줄이면 제안 방법이 메모리 문제를 푼 것

#### Axis B. shared_fifo vs routing

```text
shared_fifo
vs
routing
```

의미:

- memory selection만 바꿔도 gain이 있는지

#### Axis C. routing vs dual_memory

```text
routing
vs
dual_memory
```

의미:

- 구조적 local/event 분리가 실제로 가치가 있는지

#### Axis D. dual_memory vs full

```text
dual_memory
vs
full
```

의미:

- selective event write와 adaptive prune이 추가 gain인지

### 9.3 메인 표

```text
Table 1. MemoryBench Multitask Main Results

1. Single-task SAM2Act+ per task
2. Multitask Shared FIFO
3. Multitask + Task Routing
4. Multitask + Dual Memory
5. Multitask + Full Method
```

컬럼:

- put_block_back
- rearrange_block
- reopen_drawer 가능하면 추가
- average
- std

### 9.4 Gap 표

```text
Table 2. Multitask Gap

- task
- single-task score
- shared_fifo score
- full score
- gap(shared_fifo - single-task)
- gap(full - single-task)
```

의미:

- 제안 방법이 single-task와의 격차를 얼마나 줄였는지 정량화

### 9.5 Stability 표

```text
Table 3. Stability

- method
- eval repeat 1
- eval repeat 2
- eval repeat 3
- mean
- std
```

의미:

- MemoryBench가 원래 variance가 커서 반드시 필요

### 9.6 Memory behavior 표

```text
Table 4. Memory Behavior

- method
- avg event writes / episode
- retained event slots
- local/event fusion ratio
- success episode에서 event usage
- failure episode에서 event usage
```

의미:

- 제안한 메모리 구조를 실제로 쓰는지 보여줌

### 9.7 어떤 결과가 의미 있는 승리인가

추천 기준:

1. `full > shared_fifo`가 분명해야 함
   - 평균 기준 `+5 ~ +10 absolute`면 강함

2. `full > routing`
   - dual memory와 event lifecycle이 진짜 기여인지 확인

3. hard task가 좋아져야 함
   - 쉬운 task만 좋아지면 약함

4. 가능하면 variance도 줄어야 함

### 9.8 실패 해석 프레임

실패는 네 가지로 나눠 본다.

1. `write failure`
   - 중요한 event가 memory에 안 들어감

2. `retention failure`
   - event가 들어갔지만 나중에 사라짐

3. `retrieval failure`
   - memory에는 남았지만 읽히지 않음

4. `execution failure`
   - memory retrieval은 맞았는데 manipulation이 실패

---

## 10. 실험 단계

### 10.1 Phase 0: single-task 재현 안정화

필수:

- put_block_back fresh stage1/stage2
- rearrange_block fresh stage1/stage2
- 5ep / 25ep eval

### 10.2 Phase 1: multitask baseline

먼저 돌릴 것:

- `shared_fifo`

### 10.3 Phase 2: routing

- `routing`

### 10.4 Phase 3: dual memory

- `dual_memory`

### 10.5 Phase 4: full

- `full`

---

## 11. MemoryBench 밖 확장

MemoryBench만으로는 약할 수 있으므로, 최종 논문은 아래 조합이 좋다.

1. `MemoryBench`
2. `RLBench 같은 플랫폼 확장 실험`
3. 가능하면 `RMBench` 또는 `RoboMME`

같은 플랫폼에서 바로 memory benchmark를 하나 더 붙이기는 어렵다. 그래서 현실적으로는:

- 메인 결과는 MemoryBench
- 같은 플랫폼 일반성은 RLBench 계열
- memory benchmark 확장은 RMBench/RoboMME

조합이 가장 설득력 있다.

---

## 12. 실행 방법

### 12.1 자동 ablation 실행

```bash
bash scripts/run_memorybench_mt_ablation.sh shared_fifo put_block_back,rearrange_block mt_fifo_pb_rb
bash scripts/run_memorybench_mt_ablation.sh routing put_block_back,rearrange_block mt_routing_pb_rb
bash scripts/run_memorybench_mt_ablation.sh dual_memory put_block_back,rearrange_block mt_dualmem_pb_rb
bash scripts/run_memorybench_mt_ablation.sh full put_block_back,rearrange_block mt_full_pb_rb
```

### 12.2 stage2만 수동 실행

```bash
cd /home/cv25/siwon/sam2act/sam2act
torchrun --nproc_per_node=4 --nnodes=1 train_plus.py \
  --device 0,1,2,3 \
  --fresh-start \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus_mt_full.yaml \
  --exp_cfg_opts "tasks put_block_back,rearrange_block exp_name mt_full_pb_rb wandb False"
```

### 12.3 eval

```bash
cd /home/cv25/siwon/sam2act/sam2act
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --tasks put_block_back \
  --model-folder runs/sam2act_mt_full_pb_rb \
  --model-name model_plus_last.pth \
  --eval-datafolder ./data_memory/test \
  --eval-episodes 5 \
  --episode-length 25 \
  --device 0 \
  --log-name eval_put_block_back_5ep \
  --headless
```

---

## 13. 리스크

### 13.1 baseline instability

single-task fresh reproduce가 여전히 불안정하면 멀티태스크 해석이 약해진다.

### 13.2 conservative init

초기에는 full method가 routing-only처럼 보일 수 있다.

### 13.3 weak supervision for event write

현재는 explicit event supervision이 없다.

### 13.4 benchmark weakness

MemoryBench alone은 약하다.

---

## 14. 앞으로 확장 가능성

1. merge-based event consolidation
2. scene-cleared write
3. event sparsity auxiliary loss
4. task-specific memory head baseline

---

## 15. 내일 읽을 논문 순서

Top priority:

1. `MemoryVLA`
2. `ReMem-VLA`
3. `RMBench`

Second priority:

4. `RoboMME`
5. `BridgeVLA`
6. `MemoAct`

Third priority:

7. `CORAL`
8. `HSC-VLA`
9. `MindExplore`

---

## 16. 바로 다음 액션

1. single-task fresh reproduce 마무리
2. `shared_fifo`와 `full` smoke 실행
3. event usage logging 확인
4. full ablation 실행
5. 같은 플랫폼 확장 실험 추가

---

## 17. 요약 도식

```text
문제:
  MemoryBench multitask는 약하고 불안정하다.

가설:
  shared FIFO memory는 부족하다.

방법:
  task-conditioned dual-level memory
    -> local recent memory
    -> sparse event memory
    -> heatmap-aware event writing
    -> adaptive event pruning

실험:
  shared_fifo -> routing -> dual_memory -> full

확장:
  MemoryBench + RLBench 확장 + 가능하면 RMBench/RoboMME
```
