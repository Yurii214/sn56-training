# Justice SN56 text recipe (v2)

Fork of the open-source text baseline with three live changes on the instruct/chat path.
Everything else (LR search, batch policy, WSD, row-weighted loss, soups, second trajectory,
dev pass, restore-scaling, DPO/GRPO) is the baseline's.

## 1. Trajectory EMA candidate — `scripts/ema_track.py`, wired in `scripts/train_instruct.py`

New mechanism. During the main (a0) run an fp32 exponential moving average of the trainable
weights is maintained (on the GPU when it fits, else in host RAM), updated every 4 optimizer
steps with a time constant of 15 % of the planned step count. The callback is inserted ahead
of the soup callback, so at `on_train_end` it:

1. snapshots the raw final weights,
2. loads the EMA iterate, scores it on the dev split (`_eval_synced`),
3. offers it to the snapshot pool (`_consider_for_pool`, tag `ema`) so the greedy soup can
   include it, and adopts it as the best single state when its dev loss beats the best raw
   snapshot,
4. restores the raw weights and frees the EMA buffers.

Rationale: the a0 horizon is planned with a 1.25× overshoot and a WSD floor of 0.25, so the
run is normally clock-cut mid-decay; the EMA iterate is an annealed point that costs one dev
evaluation. The dev split decides between raw best, EMA, and soups as before.

Env (defaults): `SN56_EMA=1`, `SN56_EMA_EVERY=4`, `SN56_EMA_HORIZON=0.15`, `SN56_EMA_DEVICE=auto`.
Skipped automatically for sharded (DeepSpeed/FSDP) or multi-process runs.

## 2. KL reference = the handed-out weights — `scripts/text_trainer.py`, `scripts/train_instruct.py`

Bug fix. The validator scores KL tasks as `CE + KL_COEF · KL(model ‖ handed-out base)`.
The baseline runs restore-scaling first and then loads the KL reference from the possibly
rescaled copy (`train_request["model_path"]`), so on a rescaled task the regulariser pulled
toward the wrong model. `kl_reference_path` now records the original model directory before
restore-scaling, and the frozen KL base is loaded from it. Mask and direction are unchanged
(unshifted `labels != -100`, `KL(finetuned ‖ base)`), which is what the validator computes.

## 3. Dev split follows the grader's length cut — `scripts/validator_eval.py`, `scripts/tokenize_instruct.py`

Fidelity fix. The validator evaluates with `sequence_len = 4096`, or `ceil(max_position_embeddings / 2)`
when the model's context is below 8192, and drops longer rows. The baseline dropped dev rows
above its adaptive training `max_length`, which can exceed the grader's cut (so dev then
contained rows the grader never scores). Dev rows are now cut at
`min(train max_length, validator cut)`.

## Not entered

Image tournaments: `TRAINING_REPO_IMAGE` is left empty in the miner until a separate image
recipe exists.
