## Dual-Bank Soft-Bias Spatial Graph Smoke

Date: 2026-04-26

Branch:
- `exp/spatial-graph-memory`

Design change from the first spatial-graph smoke:
- keep a `raw temporal bank` that preserves phase-specific memories
- add a `summary spatial bank` that merges revisits only when
  - 3D position is close
  - gripper-open state is similar
  - pooled feature similarity is above threshold
- remove the hard spatial mask from retrieval
- use a soft additive bias only on summary-bank memories
- bias terms:
  - transition-count bonus between summary nodes
  - gripper-state similarity bonus
  - optional query-position bonus (disabled in this smoke)

Intent:
- allow long-range/far memory recall for "return-to-origin" behavior
- avoid destroying phase-specific raw memories
- avoid brittle retrieval that depends on a single predicted query location

Smoke command:
- `scripts/run_put_block_back_spatial_graph_smoke.sh sam2act_pbb_spatial_graph_dualbank_smoke_20260426_235048`

Run:
- `sam2act/runs/sam2act_sam2act_pbb_spatial_graph_dualbank_smoke_20260426_235048`

Training summary:
- 1 epoch / 500 updates
- around step 100:
  - `total_loss ~= 5.07`
  - `trans_loss ~= 5.06`
- around step 400:
  - `total_loss ~= 4.76`
  - `trans_loss ~= 4.76`
- final log:
  - `spatial_graph_aux_loss = 0.7284`
  - `spatial_graph_aux_top3_acc = 0.0762`
  - `total_loss = 4.8856`
  - `trans_loss = 4.8784`

Evaluation:
- `5ep smoke = 0.0%`
- all 5 episodes timed out at length 25

Interpretation:
- the dual-bank + soft-bias version is numerically better than the first hard-mask smoke
- it no longer collapses retrieval as aggressively
- but the learned retrieval geometry is still weak
- bottleneck remains:
  - summary-bank bias is not yet informative enough to improve policy rollout
  - the auxiliary signal still does not separate useful revisit structure well enough

Immediate next candidates:
1. disable summary merge feature-sim threshold and use gripper+transition only
2. remove contrastive loss entirely and test retrieval bias only
3. add text-conditioned retrieval query on top of the dual-bank design
