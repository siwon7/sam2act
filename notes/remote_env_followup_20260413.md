# Remote Env Follow-up

Date: 2026-04-13

Target host:
- `cvlab-dgx`
- `/home/cvlab-dgx/siwon`

## Goal

Record which conda environments were migrated cleanly, which failed, and what to do next.

## Export bundle location

Remote:
- `~/siwon/migration_bundle_20260413/env_exports`

Driver script:
- `~/siwon/migration_bundle_20260413/create_missing_envs.sh`

## Confirmed remote environments after migration work

Already present before this migration:
- `sam2act`
- `rvt`
- many unrelated existing research envs

Created successfully during this migration:
- `elsa-robotics-challenge`
- `kdd2026`

## Environments that need follow-up

### `ibrl`

Observed failure:
- `cuda-nvtx=12.1.105` not found from the currently used channels

Meaning:
- the exported spec is too specific for the current remote channel setup

Recommended fix:
1. create a relaxed env yaml with:
   - `python`
   - `mesalib`
   - `glew`
   - `patchelf`
2. install any CUDA-side extras manually afterward only if actually required

### `sam2act118`

Observed failure:
- missing or unavailable packages on the remote channels:
  - `pytorch3d=0.7.8=py310_cu118_pyt231`
  - `torchaudio=2.3.1`
  - `pytorch-cuda=11.8`
  - `cuda-nvcc=11.8`
  - `cuda-libraries-dev=11.8`
  - `cuda-cudart-dev=11.8`
  - `cuda-cccl=11.8.89`
  - `iopath`
  - `fvcore`

Meaning:
- this env needs explicit channels, not just `defaults`

Recommended fix:
1. create from a channel-aware yaml using:
   - `pytorch`
   - `nvidia`
   - `conda-forge`
2. install:
   - `pytorch`
   - `torchvision`
   - `torchaudio`
   - `pytorch-cuda=11.8`
   - `fvcore`
   - `iopath`
3. install `pytorch3d` separately if the exact build is unavailable

### `sam2act5090`

Status:
- creation was started from the exported yaml
- not yet confirmed complete in this handoff

Meaning:
- do not assume it is ready until checked with `conda env list`

Recommended verification:
```bash
~/anaconda3/bin/conda env list | rg sam2act5090
```

If missing, create again from a relaxed, channel-aware yaml rather than the minimal export.

### `sam2act5090w`

Status:
- not confirmed complete

Recommended approach:
- only create it if there is a real need for this env on the remote host

### `sigir26`

Status:
- not confirmed complete

Recommended approach:
- this one is likely easy because the export is minimal
- create only if actually needed

### `tgm-vla`

Status:
- yaml was regenerated with `--no-builds`
- not yet confirmed created remotely

Recommended approach:
- create only when `TGM-VLA` work resumes
- the yaml is much larger than the others and may require manual channel cleanup

## Practical recommendation

Do not try to recreate every env immediately.

Priority order:
1. `sam2act`
2. `elsa-robotics-challenge`
3. `kdd2026`
4. only then create project-specific envs on demand

Reason:
- most env creation failures are channel/package-availability issues
- solving them proactively has low value unless the corresponding repo is going to be used soon

## Recommended next commands on remote

Check current migrated envs:
```bash
~/anaconda3/bin/conda env list
```

Re-run the missing-env script if needed:
```bash
bash ~/siwon/migration_bundle_20260413/create_missing_envs.sh
```

For manual relaxed creation, start from:
```bash
cd ~/siwon/migration_bundle_20260413/env_exports
```

