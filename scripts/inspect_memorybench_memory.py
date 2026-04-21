from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def find_line(lines: list[str], pattern: str) -> int | None:
    for idx, line in enumerate(lines, start=1):
        if pattern in line:
            return idx
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/cv25/siwon/sam2act/sam2act",
        help="Code root containing mvt/ and runs/",
    )
    parser.add_argument(
        "--run-dir",
        default="/home/cv25/siwon/sam2act/sam2act/runs/sam2act_memorybench_put_block_back_stage2revert_20260410",
        help="Run dir whose mvt/exp config will be inspected",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    run_dir = Path(args.run_dir)

    mvt_cfg_path = run_dir / "mvt_cfg_plus.yaml"
    exp_cfg_path = run_dir / "exp_cfg_plus.yaml"
    source_path = repo_root / "mvt" / "mvt_sam2_single.py"

    mvt_cfg = yaml.safe_load(mvt_cfg_path.read_text())
    exp_cfg = yaml.safe_load(exp_cfg_path.read_text())
    lines = source_path.read_text().splitlines()

    num_maskmem = int(mvt_cfg["num_maskmem"])
    seq_len = num_maskmem + 1

    reset_line = find_line(lines, "def reset_memory_bank")
    init_bank_line = find_line(lines, "self.memory_bank_multiview =")
    read_line = find_line(lines, "memory_bank_list = self.memory_bank_multiview[view_idx]")
    num_mem_line = find_line(lines, "num_mem = min(len(memory_bank_list), net.num_maskmem)")
    tpos_line = find_line(lines, "maskmem_enc + net.maskmem_tpos_enc[t_pos - 1]")
    attn_line = find_line(lines, "pix_feat_with_mem = net.memory_attention(")
    encode_line = find_line(lines, "maskmem_features, maskmem_pos_enc = net._encode_new_memory(")
    write_line = find_line(lines, "memory_bank_list[self.curr_obs_idx] = [memory, memory_pos]")
    seq_reset_line = find_line(lines, "self.reset_memory_bank()")
    seq_inc_line = find_line(lines, "self.curr_obs_idx += 1")

    print(f"run_dir: {run_dir}")
    print(f"task: {exp_cfg['tasks']}")
    print(f"batch_size: {exp_cfg['bs']}")
    print(f"lr: {exp_cfg['peract']['lr']}")
    print(f"same_trans_aug_per_seq: {exp_cfg['peract']['same_trans_aug_per_seq']}")
    print(f"transform_augmentation: {exp_cfg['peract']['transform_augmentation']}")
    print(f"use_memory: {mvt_cfg['use_memory']}")
    print(f"num_maskmem: {num_maskmem}")
    print(f"sequence_length_used_for_memory: {seq_len}")
    print("--- memory flow ---")
    print(f"init_bank_line: {init_bank_line}")
    print(f"reset_bank_line: {reset_line}")
    print(f"read_bank_line: {read_line}")
    print(f"read_num_mem_line: {num_mem_line}")
    print(f"temporal_pos_enc_line: {tpos_line}")
    print(f"memory_attention_line: {attn_line}")
    print(f"encode_new_memory_line: {encode_line}")
    print(f"write_bank_line: {write_line}")
    print(f"sequence_reset_line: {seq_reset_line}")
    print(f"sequence_curr_obs_idx_increment_line: {seq_inc_line}")


if __name__ == "__main__":
    main()
