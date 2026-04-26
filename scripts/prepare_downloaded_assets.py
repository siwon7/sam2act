#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "sam2act"
DATA_ROOT = ROOT / "data"
REPLAY_ROOT = ROOT / "replay_temporal" / "replay_train"
PRETRAINED_SRC = ROOT / "sam2act_rlbench"
PRETRAINED_DST = ROOT / "runs" / "sam2act_rlbench"


def extract_zip(path: Path) -> None:
    target_dir = path.parent
    task_dir = target_dir / path.stem
    if task_dir.exists():
        return
    print(f"Extracting {path} -> {target_dir}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(target_dir)


def extract_tar_xz(path: Path) -> None:
    target_dir = path.parent
    task_dir = target_dir / path.stem.replace(".tar", "")
    if task_dir.exists():
        return
    print(f"Extracting {path} -> {target_dir}")
    with tarfile.open(path, "r:xz") as tf:
        tf.extractall(target_dir)


def move_pretrained() -> None:
    if not PRETRAINED_SRC.exists():
        return
    PRETRAINED_DST.parent.mkdir(parents=True, exist_ok=True)
    if PRETRAINED_DST.exists():
        return
    print(f"Moving pretrained files {PRETRAINED_SRC} -> {PRETRAINED_DST}")
    shutil.move(str(PRETRAINED_SRC), str(PRETRAINED_DST))


def main() -> None:
    for archive in sorted(DATA_ROOT.rglob("*.zip")):
        extract_zip(archive)
    for archive in sorted(REPLAY_ROOT.glob("*.tar.xz")):
        extract_tar_xz(archive)
    move_pretrained()
    print("Asset preparation complete.")


if __name__ == "__main__":
    main()
