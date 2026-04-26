# cv11 Handoff: Spatial Graph Memory

This repo is independent from the working `sam2act` tree on cv11.

Recommended remote target:
- `sam2act_spatial_graph_memory`

Do not overwrite:
- `sam2act`

Purpose:
- stage2-only SAM2Act experiment
- replace plain queue-style memory behavior with:
  - 3D spatial node coarsening
  - sparse spatial masking in memory attention
  - spatial contrastive auxiliary loss

Latest validated local result:
- `put_block_back` 5ep smoke = `0.0%`
- implementation path works end-to-end

Start from:
- dirty stage1 `model_38.pth`

Main script:
- `scripts/run_put_block_back_spatial_graph_smoke.sh`
