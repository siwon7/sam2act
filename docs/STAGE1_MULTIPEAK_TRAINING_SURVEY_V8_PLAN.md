# Stage1 Multi-Peak Training Survey and V8 Plan

Date: 2026-05-04

## 1. Problem Definition

The current failure is not simply "heatmap collapse". Collapse is acceptable
when the highest peak is the ground-truth or task-equivalent direction. The
actual failure is:

> Stage1 compresses a one-to-many next-target distribution into a single top-1
> crop, and the selected crop is sometimes the wrong target, an off-target
> point, or a GT-direction-but-too-far point.

This creates a pipeline ceiling:

```text
observation -> Stage1 heatmap -> top1 crop center -> Stage2 local action
```

If Stage1 top1 is wrong, Stage2 receives the wrong crop. Oracle Stage2 training
can bypass this by forcing the GT crop, but normal rollout cannot.

The right metric is therefore not only `trans_loss` or `p1/p2` collapse ratio.
The diagnostic metric must include:

- top1 GT hit
- top1 valid/equivalent hit
- top1 alt/off/far classification
- GT mass and alt mass
- p1/p2 ratio
- crop distance to GT
- per-keyframe and per-episode stability

## 2. Local Evidence

Current code already supports multi-peak labels, but only with positions and
masks. It does not store or supervise per-peak probability weights.

Relevant local implementation:

- `models/sam2act_agent.py`: soft heatmap CE for `trans_loss`
- `models/sam2act_agent.py`: `get_action_trans()` builds primary + alt
  heatmaps
- `utils/dataset.py`: replay stores `alt_target_positions` and
  `alt_target_mask`, but no `alt_target_weights`
- `mvt/multipeak_utils.py`: target collection is cluster-based; v7 uses
  spatial-only clustering

Empirical pattern from GPU7:

- Dirty checkpoints and active `model_34` both show wrong-target rows.
- `put_block_back` KF4/KF9 persistently prefer alt targets.
- `put_block_back` KF6 remains GT-direction-but-far.
- `rearrange_block` KF0 improved in `model_34`, but KF5/KF8 still show
  far-crop or mixed behavior.
- `reopen_drawer` KF0 persists, and KF2/KF3/KF4/KF8 show vision/target
  semantics issues.

This means the issue is not only one dirty checkpoint. The current objective
does not directly force correct top1/valid peak selection under ambiguity.

## 3. Survey Takeaways

### 3.1 One-to-many BC is a distribution modeling problem

Bishop's Mixture Density Networks note states the classical failure: for
multi-valued inverse problems, average predictions are not necessarily valid
outputs. The right target is the conditional distribution, not a single
conditional mean.

Relevance to us:

- Stage1 should model `p(next target | observation, task, history state)`.
- A single argmax heatmap is too lossy when several next targets are valid or
  visually/history ambiguous.
- Soft multi-peak labels are reasonable, but only if their weights match the
  intended conditional distribution.

Source: https://www.microsoft.com/en-us/research/publication/mixture-density-networks/

### 3.2 Divergence choice changes mode behavior

Ke et al. frame imitation learning as f-divergence minimization and emphasize
that behavior cloning can behave poorly with multi-modal demonstrations by
interpolating or selecting modes depending on the loss/model class.

Relevance to us:

- Vanilla CE to a heatmap is not enough unless the label distribution is
  semantically correct.
- For Stage1, "cover all modes" is useful for proposal generation, but final
  execution needs a selector.
- We should separate Stage1 proposal quality from Stage2 peak selection.

Source: https://arxiv.org/abs/1905.12888

### 3.3 Discretization plus correction is a practical multimodal policy pattern

Behavior Transformers handle multimodal behavior by discretizing actions into
mode-like bins and predicting an offset correction. This is close in spirit to
Stage1 heatmap peaks plus Stage2 local action refinement.

Relevance to us:

- Stage1 heatmap peaks are discrete candidate bins.
- Stage2 crop/action is the correction/refinement around a selected candidate.
- Therefore Stage1 should expose top-K candidates, not only top1.

Source: https://arxiv.org/abs/2206.11251

### 3.4 Implicit and diffusion policies support multimodal actions, but are heavier

Implicit Behavioral Cloning models action choice with an energy function and is
designed for complex, discontinuous, multi-valued mappings. Diffusion Policy and
related diffusion imitation work model full action distributions and are strong
for multimodal visuomotor policies.

Relevance to us:

- These are evidence that the problem is real and distributional.
- Full diffusion/EBM Stage1 would be a larger architecture change.
- A heatmap energy/proposal view is the lightest compatible variant for the
  current SAM2Act/RVT-style code path.

Sources:

- https://arxiv.org/abs/2109.00137
- https://arxiv.org/abs/2303.04137
- https://arxiv.org/abs/2301.10677

### 3.5 Latent mode or memory variables matter under partial observability

InfoGAIL learns latent structure in demonstrations when behavior varies because
of unobserved factors. In our tasks, some ambiguity is visible in the current
frame, but some is historical: the same current pose can require different next
targets depending on what happened earlier.

Relevance to us:

- If the relevant variable is visible, Stage1 should collapse to the visible
  correct target.
- If the relevant variable is not visible but recoverable from history, Stage1
  should keep multiple candidates and Stage2/memory should select.
- If the label mixes visually distinguishable cases, Stage1 can learn bad
  priors instead of visual semantics.

Source: https://arxiv.org/abs/1703.08840

### 3.6 PerAct/RVT validate discretized next-action detection, not ambiguity handling

PerAct and RVT show that discretized next-action detection in 3D/multi-view
space is a strong formulation for RLBench manipulation. They do not by
themselves solve memory-dependent one-to-many ambiguity.

Relevance to us:

- Keeping the heatmap/discretized target formulation is compatible with strong
  manipulation baselines.
- Our missing piece is conditional multi-peak target construction and peak
  selection, not replacing the whole perception/action architecture first.

Sources:

- https://arxiv.org/abs/2209.05451
- https://arxiv.org/abs/2306.14896

## 4. Training Principles for the Next Version

### Principle A: classify ambiguity before assigning labels

Do not treat every spatial revisit as the same kind of ambiguity.

Use three classes:

| class | condition | Stage1 target |
|---|---|---|
| visible single | current visual/proprio state determines one target | single peak |
| visible multi-valid | multiple targets are valid under the same current state | weighted multi-peak |
| history-dependent | current state is insufficient; history/memory needed | weighted multi-peak + Stage2 selector |

Bad case to avoid:

```text
visual state says target B, but label says A/B by dataset prior
```

That teaches Stage1 to ignore the visual cue.

### Principle B: weight multi-peak labels by current GT plus data prior

Pure equal-weight multi-peak is too blunt. Pure data prior can overpower the
current sample. Use a mixture:

```text
prior_i = (count_i + alpha)^tau / sum_j (count_j + alpha)^tau
w_i = (1 - lambda) * one_hot_current_gt_i + lambda * prior_i
```

Initial values:

```text
alpha = 1.0
tau = 0.75
lambda = 0.4
min_weight = 0.03
```

Example for a cluster with A:B = 8:2 and current GT = B:

```text
prior(A)=0.8, prior(B)=0.2
w(A)=0.32
w(B)=0.68
```

So the current observation still learns B, while the dataset-level average
retains the observed 8:2 ratio.

### Principle C: supervise peak ratios directly, not only pixel CE

Pixel CE can decrease while top1 remains wrong. Add a small auxiliary loss on
mass around each candidate peak:

```text
pred_mass_i = sum softmax(q_trans)[pixels within radius around peak_i]
ratio_loss = KL(label_peak_weights || normalize(pred_mass))
```

Initial value:

```text
ratio_kl_weight = 0.05
peak_radius_px = 3 * gt_hm_sigma
```

This should be logged separately and should not dominate the base heatmap CE.

### Principle D: Stage1 proposes top-K; Stage2 selects and refines

Normal rollout should not collapse the full pipeline to Stage1 top1 when a
multi-peak candidate set exists. Stage1 should expose top-K peaks and Stage2
should choose one by local evidence and memory.

Initial rollout variants:

| variant | crop policy | purpose |
|---|---|---|
| top1 baseline | current behavior | compatibility |
| oracle topK | include GT peak if present | ceiling |
| predicted topK rerank | Stage2 scores K crops | real target |
| noisy/wrong crop training | train Stage2 on topK-like crops | robustness |

### Principle E: measure direction correctness, not only collapse

Every checkpoint should run the collapse probe with:

- `wrong-collapse`
- `ok-collapse`
- `alive-wrong`
- `ok-alive`
- `far-crop`
- `off-target`
- `topK contains GT`
- `topK selected GT`

TopK contains GT is especially important. If Stage1 top1 is wrong but topK
contains GT, Stage2/memory can still recover.

## 5. V8 Proposal

Name:

```text
stage1_v8_weighted_multipeak_topk
```

Main changes:

1. Add `alt_target_weights` to replay.
2. Precompute or dynamically compute candidate target counts per ambiguity
   subgroup.
3. Build weighted heatmap labels:

   ```text
   hm = sum_i w_i * gaussian(candidate_i)
   hm = hm / hm.sum()
   ```

4. Add weak peak-ratio KL loss.
5. Add Stage1 topK audit and later topK Stage2 handoff.
6. Preserve single-peak labels where visual/proprio state should disambiguate.

## 6. Implementation Plan

### Step 1: data schema

Add replay field:

```python
ReplayElement("alt_target_weights", (multipeak_max_peaks,), np.float32)
```

Also define the implicit primary weight at train time, because primary is not
stored as an alt target.

### Step 2: target computation

Extend `collect_alt_targets()` or add `collect_weighted_alt_targets()`:

```text
return alt_positions, alt_mask, alt_weights
```

The function should:

- form ambiguity groups
- deduplicate next targets
- count target frequency
- locate current GT in the candidate set
- compute alt weights using the `one_hot_current_gt + prior` formula

### Step 3: label generation

Modify `get_action_trans()`:

- read `alt_target_weights`
- compute `primary_weight = 1 - sum(valid_alt_weights)` or explicitly compute
  all candidate weights and split primary/alt
- generate weighted heatmaps
- normalize per view

Important: for stage-two training views, keep the anchor behavior unless the
experiment explicitly changes it.

### Step 4: loss

Keep current CE:

```text
trans_loss = CE(q_trans, weighted_action_trans)
```

Add optional:

```text
total_loss += ratio_kl_weight * peak_ratio_kl
```

Log:

- `peak_ratio_kl`
- `pred_peak_entropy`
- `label_peak_entropy`
- `top1_gt_hit`
- `topk_gt_hit`

### Step 5: probe before long training

Before full training, run a label-only sanity script:

- For each task/KF, show candidate positions and weights.
- Verify label p1/p2 matches desired weight ratio.
- Verify visually distinguishable single-target KFs remain single.

Then run short training:

```text
put_block_back only, 2-3 epochs, GPU7 or free GPU
```

Success condition:

- KF4/KF9 no longer have persistent wrong alt top1 unless GT is present in
  topK and the planned Stage2 selector can recover.
- topK GT hit improves even if top1 still follows the dominant prior.

## 7. Experiment Matrix

| exp | Stage1 label | ratio KL | rollout handoff | question |
|---|---|---:|---|---|
| v7 baseline | equal/position-only multi-peak | no | top1 | current behavior |
| v8a | weighted multi-peak | no | top1 | does weighting reduce wrong collapse? |
| v8b | weighted multi-peak | 0.05 | top1 | does ratio supervision preserve modes? |
| v8c | weighted multi-peak | 0.05 | topK oracle | is Stage1 topK enough? |
| v8d | weighted multi-peak | 0.05 | topK learned rerank | can Stage2 recover normal rollout? |

## 8. Expected Outcome

Weighted multi-peak alone may not make top1 always correct. If the data prior
is 8:2, top1 may still prefer the 8 side. That is acceptable only if:

- the current observation does not determine the 2 side, and
- topK contains the 2-side GT, and
- Stage2/memory can select it.

So the real target for V8 is:

```text
Stage1: high topK recall, calibrated candidate mass
Stage2: correct candidate selection and local action refinement
```

The next version should avoid treating Stage1 as a deterministic final
waypoint predictor. Stage1 should become a calibrated proposal generator.

## 9. Decision

The recommended next version is:

```text
V8 = weighted conditional multi-peak Stage1 + peak-ratio KL + topK audit
```

After V8 confirms that GT is present in topK, move to:

```text
V9 = topK Stage2 reranker / memory-guided peak selector
```

Do not start with full diffusion, EBM, or MoG architecture changes. They are
valid research directions, but they are larger than needed for the immediate
SAM2Act failure.
