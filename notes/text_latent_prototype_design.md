Text-Conditioned Latent Prototype Memory

Goal
- Avoid manual role/node preprocessing.
- Keep SAM2Act's per-view heatmap-guided memory write/read structure.
- Learn memory selection from text-conditioned latent prototypes plus online retrieve-vs-explore supervision.

Core Idea
- Each memory entry stores:
  - view-wise memory feature
  - memory position encoding
  - learned latent prototype distribution over `K` prototypes
- Current observation builds:
  - latent prototype query logits
  - retrieve-vs-explore logit
  - the query is conditioned on current visual feature and language embedding
- Memory retrieval bias is:
  - stronger when retrieve probability is high
  - stronger for memory entries whose latent prototype distribution matches the current query distribution

Retrieve vs Explore framing
- `explore/new`:
  - the next action should go to a new location based mainly on the current observation
- `retrieve/revisit`:
  - the next action should reuse an earlier location encoded in memory
- The model is not given manual phase labels or manual graph edges.
- Instead, it learns a soft gate that increases memory reliance only when the current GT action is close to a previously visited GT action in the same sequence.

Why this is less hand-crafted
- No manual grouped role labels.
- No manual keyframe graph or edge table.
- Retrieve supervision is computed online from the GT action trajectory inside the batch:
  - if the current GT target is close to a previous GT target in the same sequence, treat it as a revisit
  - use the previous matching steps' write-prototype distributions as the soft target for the current query-prototype distribution
- Text is used only as a routing prior for memory selection, not as a manual task graph.

Losses
- `L_trans`: original action heatmap loss
- `L_retrieve`: BCE on retrieve vs explore, built online from GT action coordinates
- `L_proto_align`: KL between current query prototype distribution and the average prototype distribution of matching previous steps
- `L_proto_usage`: small uniform-usage regularizer to reduce prototype collapse

What stayed unchanged
- SAM2Act per-view memory banks
- heatmap-guided memory encoder
- memory attention fusion path
- stage1 -> stage2 handoff

References
- SAM2Act: heatmap-guided view-wise memory write/read
- ReMem-VLA: recurrent memory retrieval and past observation supervision
- MemoryVLA: working memory vs long-term memory bank separation
- Pointer Networks: explicit revisit-like selection over past structured states
