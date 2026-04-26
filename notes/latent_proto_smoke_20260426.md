Latent Prototype Smoke 2026-04-26

Run
- family: text-conditioned latent prototype memory
- branch: `exp/text-latent-prototype-memory`
- initialization: dirty `put_block_back` stage1 checkpoint (`model_38`)
- configuration:
  - `num_maskmem=11`
  - `demo=20`
  - `epochs=1`
  - `train_iter=12000`
  - `bs=12`
  - `world_size=2`
  - effective train loop length: `500` updates

What was validated
- local replay rebuild and reuse
- local `YARR` import precedence
- stage2 handoff into the correct run directory
- train completion
- eval completion on device `0`

Training summary
- around step `100`:
  - `total_loss ~= 5.60`
  - `trans_loss ~= 5.14`
- around step `300`:
  - `total_loss ~= 5.43`
  - `trans_loss ~= 5.00`
- around step `500`:
  - `total_loss ~= 5.04`
  - `trans_loss ~= 4.59`
- final logged metrics:
  - `latent_revisit_acc ~= 0.366`
  - `latent_revisit_loss ~= 2.651`
  - `latent_proto_align_loss ~= 6.291`
  - `latent_proto_usage_loss ~= 0.714`

Eval summary
- task: `put_block_back`
- evaluation: `5ep smoke`
- success rate: `0.0%`
- average length: `25.0`
- total transitions: `125`

Readout
- This experiment is the first clean validation of the non-manual latent prototype route.
- The code path is correct end-to-end.
- The current objective still does not produce useful rollout behavior.
- The most likely next fixes are:
  - stabilize retrieve-vs-explore learning
  - weaken or redesign prototype alignment targets
  - keep text-conditioned retrieval but reduce direct auxiliary pressure on policy

