# Transition Pointer Memory Design

## Branch

- repo: `/home/cv25/siwon/sam2act_transition_pointer_memory`
- branch: `exp/slot-mode-pointer`

## Problem Framing

For MemoryBench, the next action is not always "go to a new place".
It is a mixture of:

1. **move to a new spatial prototype**
2. **revisit a previously seen prototype**

The key local observation from `put_block_back` is that the 12 extracted
keyframes collapse into a small set of repeated spatial/event prototypes.

## Put-Block-Back Prototype Roles

Grouped roles used in this branch:

1. `initial_slot_low`  -> keyframes `{0, 11}`
2. `initial_slot_high` -> keyframes `{1, 10}`
3. `center_high`       -> keyframes `{2, 4, 7, 9}`
4. `center_low`        -> keyframes `{3, 8}`
5. `button_high`       -> keyframes `{5}`
6. `button_low`        -> keyframes `{6}`

This matches the key spatial pattern visible in the grouped keyframe plot:

- early steps visit new prototypes
- later steps repeatedly revisit already seen prototypes
- the final return requires recalling the initial slot prototypes rather than
  just the most recent memory

## New-vs-Revisit Labels

This branch adds an explicit binary target:

- `visit_mode_label = 0`: first visit to a prototype
- `visit_mode_label = 1`: revisit of an already seen prototype

For `put_block_back`, the current sequence is:

`[0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]`

Interpretation:

- `0,1,2,3` are first visits
- `4` revisits the center-high prototype
- `5,6` are first visits to the button region
- `7,8,9,10,11` are revisit-heavy late steps

## Role Reference Supervision

Late revisit steps also get grouped retrieval targets:

- `7,8` -> `{center_high, center_low}`
- `9`   -> `{initial_slot_high, center_high}`
- `10`  -> `{initial_slot_low, initial_slot_high}`
- `11`  -> `{initial_slot_low, initial_slot_high}`

This is intentionally soft and grouped by prototype rather than exact
keyframe index.

## Anchor Policy

Persistent anchors are no longer treated as "first k timesteps".

Instead, this branch stores the **first occurrence of each prototype role** in
the persistent anchor bank.

That means the anchor bank becomes prototype-indexed:

- first `initial_slot_low`
- first `initial_slot_high`
- first `center_high`
- first `center_low`
- first `button_high`
- first `button_low`

Current `anchor_use_label` is deliberately narrower than `visit_mode_label`.
It only activates for the late return steps:

- `9`
- `10`
- `11`

This avoids overusing anchors on every revisit.

## Model Changes

### 1. Visit-Mode Head

Added a binary `visit_mode_head` alongside the existing grouped role heads in
`sam2act/mvt/mvt_sam2_single.py`.

It predicts whether the current step should be treated as:

- a new target prototype
- or a revisit step

### 2. Revisit-Gated Memory Bias

The existing grouped-role additive memory bias is now multiplied by the
predicted revisit probability:

- on new steps, the graph bias should stay weak
- on revisit steps, the graph bias should matter more

This avoids forcing retrieval logic onto the very first exploratory steps.

### 3. Prototype-First Anchor Write

Persistent anchors are written once per role, not once per early timestep.

This makes the anchor bank prototype-centric rather than strictly temporal.

## Losses

This branch now supports:

- `L_trans`
- `L_phase`
- `L_role`
- `L_visit_mode`
- `L_role_ref`
- optional `L_anchor_use`
- optional `L_role_contrastive`

The current intended usage is:

- `trans` always on
- `phase/role` as weak structural regularizers
- `visit_mode` as the main new-vs-revisit auxiliary
- `role_ref` only for late revisit supervision
- `anchor_use` optional and usually weaker than `visit_mode`

## Why This Is Different From The Earlier Branches

Earlier branches focused on:

- exact or grouped role prediction
- retrieval probes
- anchor-use supervision

This branch changes the control question from:

> "what role am I in?"

to:

> "should I generate a new target or revisit an old prototype?"

That is closer to the actual MemoryBench structure.

## Related Papers

This design is informed by:

1. **SAM2Act**
   - heatmap-guided spatial memory with view-wise memory banks
   - arXiv: `2501.18564`

2. **ReMem-VLA**
   - recurrent memory and auxiliary memory objectives
   - arXiv: `2603.12942`

3. **MemoryVLA**
   - separation between working memory and a memory bank
   - arXiv: `2508.19236`

4. **RMBench / Mem-0**
   - anchor memory and memory-critical evaluation
   - arXiv: `2603.01229`

5. **Pointer Networks**
   - conceptually relevant for selecting a previous state instead of always
     generating a fresh target
   - arXiv: `1506.03134`

## Current Scope

This branch currently implements the **grouped role + new-vs-revisit + soft
retrieval bias** version.

It does **not** yet implement:

- a hard per-entry pointer over previous memories
- hard graph-masked attention
- a separate slot-id head

Those are reserved for follow-up branches if this softer transition-pointer
variant shows promise.
