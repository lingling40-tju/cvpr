# Matched positive-trajectory scale: training evidence

The frozen expansion has six fresh runs: control and candidate at seeds
11, 22, and 33, each with 512 fit rows, 128 optimizer steps, two data
epochs, and four rollouts per group. This directory records completed
training evidence; it contains no scale navigation result yet.

At 2026-10-07 18:18 UTC, the seed-11 control has completed all 128 steps.
Its remote completion timestamp is `2026-10-07T18:15:48Z`. The copied
`control_seed11_train_audit.json` reports 128 nonzero actor-gradient steps,
finite KL metrics, and no missing optimizer steps. Console metrics have
three-decimal precision. The same-seed candidate has started from the
frozen SFT initialization; it is not initialized from this control.

`control_seed11_tensorboard.json` additionally preserves 16 scalar tags at
all 128 optimizer steps at their stored TensorBoard precision, usually
float32. The [CPU exporter](../export_positive_tensorboard.py) checks the
completed run's configuration and original audit/log hashes, exact step
coverage, finite values, and 1,920 comparable values against the rounded
console. All 128 gradients and KL losses are nonzero in this export;
early KL values rounded to `0.000` in the console. The raw event-file
hash is retained. Console agreement allows three-decimal rounding and
float32 conversion. Rollout return and the goal-reached reason frequency
are training diagnostics, not independently evaluated navigation SR or
semantic accuracy. This adds evidence without changing any frozen gate
or inference schedule.

`control_seed11_checkpoint_metadata.json` records a separate CPU check of
the saved Hugging Face export: all four indexed safetensors shards exist,
their tensor names match the index, declared payload ranges are contiguous,
and each file's extent matches its header. The export contains 825 tensor
entries in float32. Header hashes refer to canonicalized header JSON, not
the complete weight payload. This checks file structure and completeness;
it does not validate tensor values, perform inference, or establish a
navigation benefit.

The remaining five training runs, reserved-screen evaluation, full
val-unseen evaluation, and shared SFT comparison are still pending. The
existing frozen relay and selection rules are unchanged. Full protocol:
[PILOT_PROTOCOL.md](../PILOT_PROTOCOL.md).
