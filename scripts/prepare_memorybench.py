#!/usr/bin/env python3

import os
import shutil
import zipfile


REPO_ROOT = "/home/cv25/siwon/sam2act"
CODE_ROOT = os.path.join(REPO_ROOT, "sam2act")
DATA_ROOT = os.path.join(CODE_ROOT, "data_memory")
TASKS_ROOT = os.path.join(CODE_ROOT, "libs", "RLBench", "rlbench", "tasks")
TTM_ROOT = os.path.join(CODE_ROOT, "libs", "RLBench", "rlbench", "task_ttms")

TASKS = ("put_block_back", "rearrange_block", "reopen_drawer")


def ensure_extracted(split: str, task: str) -> None:
    zip_path = os.path.join(DATA_ROOT, split, f"{task}.zip")
    out_dir = os.path.join(DATA_ROOT, split)
    task_dir = os.path.join(out_dir, task)
    if os.path.isdir(task_dir):
        print(f"already extracted: {task_dir}")
        return
    print(f"extracting: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def install_task_files(task: str) -> None:
    py_src = os.path.join(DATA_ROOT, "files", f"{task}.py")
    ttm_src = os.path.join(DATA_ROOT, "files", f"{task}.ttm")
    py_dst = os.path.join(TASKS_ROOT, f"{task}.py")
    ttm_dst = os.path.join(TTM_ROOT, f"{task}.ttm")
    shutil.copy2(py_src, py_dst)
    shutil.copy2(ttm_src, ttm_dst)
    print(f"installed task files for {task}")


def main() -> None:
    for task in TASKS:
        ensure_extracted("train", task)
        ensure_extracted("test", task)
        install_task_files(task)
    print("memorybench ready")


if __name__ == "__main__":
    main()
