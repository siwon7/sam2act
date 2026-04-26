cv11 Handoff: MemoryBench Stage2 Experiments

Summary
- Base problem: `put_block_back` requires both:
  - correct next-action prediction for new positions
  - correct recall of previously visited positions for return/revisit behavior
- Multiple graph-style manual-role experiments were implemented and executed locally.
- They consistently ran end-to-end, but rollout success remained `0.0%` in smoke/full evaluations.

Completed experiment families and observed outcomes

1. Graph node auxiliary loss
- Idea:
  - classify current semantic node/keyframe role during stage2
- Result:
  - node accuracy became very high
  - rollout success stayed `0.0%`
- Interpretation:
  - easy auxiliary shortcut; did not improve action policy

2. Graph-only loss
- Idea:
  - remove action-driving effect and optimize graph classification objective
- Result:
  - graph classification solved
  - trans/action loss stayed high
  - rollout success `0.0%`
- Interpretation:
  - classification branch detached from policy learning

3. Manual role graph + persistent anchor + role retrieval bias
- Idea:
  - grouped manual role nodes such as initial-slot / center / button
  - bias retrieval toward expected role groups
- Result:
  - auxiliary metrics improved
  - rollout success `0.0%`
- Interpretation:
  - manual role supervision over-constrained retrieval without fixing policy

4. Transition-pointer variants
- Idea:
  - split behavior into `new` vs `revisit`
  - use grouped role retrieval in revisit phase
- Result:
  - multiple balanced/soft/late-gated variants all returned `0.0%` smoke
- Interpretation:
  - explicit revisit logic alone was not enough; still too hand-crafted

New direction implemented
- Text-conditioned latent prototype memory
- No manual role labels or manual graph edges
- Keep original SAM2Act view-wise memory architecture
- Learn a soft `retrieve vs explore` gate instead of manually splitting phases

New design
- Each memory write stores a learned latent prototype distribution over `K` prototypes
- Current observation + text builds:
  - current query prototype distribution
  - retrieve probability
- Retrieval bias increases when:
  - retrieve probability is high
  - memory entry prototype matches current query prototype
- If retrieve probability is low, the model is expected to behave like the original next-action predictor and rely more on the current observation.

Training supervision
- no manual role labels
- retrieve targets are computed online from the GT action trajectory in the same sequence:
  - if current GT action target is close to an earlier GT target, it is treated as retrieve/revisit
- current query prototype is aligned to the average write-prototype distribution of those matching previous steps

Losses
- `trans_loss`
- `latent_retrieve_loss`
- `latent_proto_align_loss`
- `latent_proto_usage_loss`

Expected benefit
- The model can learn repeated spatial/event structure without hand-crafted nodes
- Text conditions retrieval selection across tasks
- Episode-specific memory still determines which exact location to revisit

Implementation notes
- The actual run folder name is `exp_id + "_" + exp_name`.
  - With default `exp_id=sam2act`, a smoke run named `foo` is saved under `sam2act_foo`.
- Local `libs/YARR` must take precedence over any editable install in the environment.
  - Entry points and smoke scripts were updated to prepend the repo-local `libs/YARR`.
- Smoke replay creation is safest when:
  - `num_workers=0`
  - replay is rebuilt locally at least once
  - eval uses `device 0`

Latest validated smoke
- Experiment family:
  - text-conditioned latent prototype memory
  - soft `retrieve vs explore`
  - no manual node labels
- Training recipe:
  - dirty `put_block_back` stage1 checkpoint as initialization
  - `num_maskmem=11`
  - `demo=20`
  - `train_iter=12000` which yields `500` train updates at `bs=12` and `world_size=2`
- Training outcome:
  - trans loss improved from roughly `10.82` to `4.59` over the short smoke
  - retrieve loss remained unstable late in training
- Eval outcome:
  - `put_block_back` `5ep smoke = 0.0%`
  - all five episodes timed out at length `25`
- Interpretation:
  - the path is now technically correct end-to-end
  - the current latent prototype objective is not yet improving rollout success

Key references behind this direction
- SAM2Act
- ReMem-VLA
- MemoryVLA
- Pointer Networks
