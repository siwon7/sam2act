# Stage2 Recent Attempts Summary

This note is meant as a compact handoff for `cv11`.

## What this repo is for

`sam2act_spatial_graph_memory` is a stage2-only experimental branch for SAM2Act.

Intent:
- keep SAM2Act's original per-view heatmap-guided memory backbone
- add graph-style memory structure without modifying the working `sam2act` tree

## Recent attempts worth keeping

### 1. Spatial graph memory, first smoke

Idea:
- merge memory entries by 3D proximity
- add strong spatial attention mask
- add spatial contrastive loss

Result:
- implementation worked end-to-end
- `5ep smoke = 0.0%`

Main failure mode:
- hard retrieval constraints were too brittle

### 2. Dual-bank gripper-aware spatial graph

Idea:
- preserve a raw temporal bank
- add a summary spatial bank
- merge only in summary bank using:
  - 3D closeness
  - gripper-open similarity
  - optional feature similarity
- remove hard mask and replace with soft additive retrieval bias

Bias terms:
- summary-node transition counts
- gripper-state similarity
- optional spatial prior

Result:
- numerically more stable than hard-mask version
- still `5ep smoke = 0.0%`
- suggests retrieval geometry is still too weak to help the actual policy

### 3. Teacher-forced node inspection

Reason:
- before trusting rollout-based graph behavior, inspect whether GT keyframes
  naturally produce sensible summary nodes and transitions

Script:
- `scripts/inspect_spatial_graph_teacher_forced_nodes.py`

What it checks:
- which GT keyframes collapse into the same summary node
- how many summary nodes remain
- transition edge counts between summary nodes
- whether gripper-aware merging improves purity

This is the most important diagnostic to run before further architectural changes.

## What not to prioritize right now

- stronger hard masks
- more exact hand-crafted node supervision
- action heads that fully replace the base `trans` path

These directions repeatedly improved auxiliary losses without improving rollout.

## Recommended next experiments

1. teacher-forced node inspection
2. dual-bank retrieval bias without contrastive loss
3. text-conditioned retrieval bias on top of the dual-bank structure

## Relation to other local repos

- `sam2act_baseline`
  - dirty practical baseline + analysis utilities
- `sam2act_multitask_txt_memory`
  - earlier multitask/text-memory scaffold
- `sam2act_text_latent_prototype`
  - non-manual latent prototype attempt

Current recommendation:
- keep `sam2act_spatial_graph_memory` as the main repo for graph-memory stage2 iteration
- use the other repos only as references
