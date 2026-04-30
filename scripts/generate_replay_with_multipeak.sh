#!/bin/bash
# Generate replay data with multi-peak alt targets for MemoryBench tasks.
#
# Prerequisites:
#   1. Raw demo data downloaded from HuggingFace:
#      https://huggingface.co/datasets/hqfang/memorybench
#      Place under: sam2act/data_memory/{train,test}/
#
#   2. Delete existing replay cache if any:
#      rm -rf sam2act/replay_temporal_memory/
#
# The standard train_plus.py will auto-generate replay data on first run.
# This script just documents the flow — no separate generation step needed.
#
# Usage:
#   # Step 1: Ensure raw data exists
#   ls sam2act/data_memory/train/put_block_back/
#
#   # Step 2: Delete old replay cache (IMPORTANT for multi-peak)
#   rm -rf sam2act/replay_temporal_memory/
#
#   # Step 3: Train with multi-peak enabled
#   cd sam2act
#   WANDB_MODE="offline" \
#   torchrun --nproc_per_node=1 --nnodes=1 \
#     train_plus.py \
#     --exp_cfg_opts "tasks put_block_back" \
#     --exp_cfg_path configs/sam2act_plus.yaml \
#     --mvt_cfg_path mvt/configs/sam2act_plus.yaml
#
# The replay buffer will be auto-generated with alt_target_positions
# and alt_target_mask fields included.
#
# To verify multi-peak labels before training:
#   python scripts/verify_multipeak_labels.py

echo "=== Multi-Peak Replay Data Generation ==="
echo ""
echo "This is handled automatically by train_plus.py on first run."
echo "Make sure to:"
echo "  1. Place raw data in sam2act/data_memory/train/"
echo "  2. Delete old replay cache: rm -rf sam2act/replay_temporal_memory/"
echo "  3. Set use_multipeak: True in mvt/configs/sam2act_plus.yaml"
echo ""
echo "To verify labels: python scripts/verify_multipeak_labels.py"
