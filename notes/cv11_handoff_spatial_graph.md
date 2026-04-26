# cv11 Handoff: Spatial Graph Memory

This repo is independent from the working `sam2act` tree on cv11.

Recommended remote target:
- `sam2act_spatial_graph_memory`

Do not overwrite:
- `sam2act`

Purpose:
- stage2-only SAM2Act experiment
- replace plain queue-style memory behavior with graph-structured retrieval ideas
- current best-maintained line in this repo is:
  - raw temporal bank + summary spatial bank
  - gripper-aware summary-node merge
  - transition/gripper soft retrieval bias
  - spatial contrastive auxiliary loss

Latest validated local result:
- `put_block_back` 5ep smoke = `0.0%`
- implementation path works end-to-end

Start from:
- dirty stage1 `model_38.pth`

Main script:
- `scripts/run_put_block_back_spatial_graph_smoke.sh`

Useful diagnostics:
- `scripts/inspect_spatial_graph_teacher_forced_nodes.py`
  - GT keyframe based node/edge inspection
- `scripts/analyze_memorybench_attention_teacher_forced.py`
  - teacher-forced attention summary

Useful notes:
- `notes/stage2_recent_attempts_summary.md`
- `notes/dualbank_softbias_20260426.md`
