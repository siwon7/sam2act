cv11 Repo Overview

This workspace contains multiple SAM2Act-related repositories with different roles.

1. `sam2act_upstream_main`
- pristine upstream-style baseline
- use when reproducing the clean default recipe

2. `sam2act_upstream_mbdebug`
- clean-derived debug workspace
- use for minimal clean-side fixes or ablations without touching the pristine baseline

3. `sam2act_baseline`
- preserved dirty baseline
- use as the practical local reference line

4. `sam2act_memory_aux_loss`
- manual graph node auxiliary loss experiments
- result trend so far: auxiliary metrics improved, rollout stayed `0%`

5. `sam2act_multitask_txt_memory`
- multitask/text-memory scaffold
- contains earlier anchor, role graph, and graph-retrieval probe work
- result trend so far: technically valid, no rollout improvement

6. `sam2act_text_latent_prototype`
- current non-manual direction
- text-conditioned latent prototype memory
- soft `retrieve vs explore` gate
- no manual node labels
- latest validated smoke: `0.0%` on `put_block_back`

Recommended cv11 starting order
1. clean default reproduction in `sam2act_upstream_main`
2. practical dirty baseline checks in `sam2act_baseline`
3. new non-manual retrieval work in `sam2act_text_latent_prototype`
