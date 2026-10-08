# Matched positive-trajectory scale: training evidence

The frozen expansion has six fresh runs: control and candidate at seeds
11, 22, and 33, each with 512 fit rows, 128 optimizer steps, two data
epochs, and four rollouts per group. This directory records completed
training evidence; it contains no scale navigation result yet.

At 2026-10-08 04:02 UTC, three arms have completed all 128 steps.
The control completed at `2026-10-07T18:15:48Z`; the candidate completed
at `2026-10-07T23:10:21Z`, both at seed 11. Seed-22 control completed
at `2026-10-08T03:58:09Z`. Their three copied `*_train_audit.json` files
each report 128 nonzero actor-gradient steps, finite KL metrics, and no
missing optimizer steps. The candidate has positive advantage at every
step, with logged minimum zero and maximum 1.5. Console metrics have
three-decimal precision. All three arms start from the frozen SFT initialization.
The seed-22 candidate has started and its actual TaskRunner is alive; its
optimizer steps have not yet been reported at this snapshot.

The three `*_tensorboard.json` files additionally preserve 16 scalar
tags at all 128 optimizer steps at their stored TensorBoard precision, usually
float32. The [CPU exporter](../export_positive_tensorboard.py) checks the
completed run's configuration and original audit/log hashes, exact step
coverage, finite values, and 1,920 comparable values against the rounded
console per arm. All 128 gradients and KL losses are nonzero in each export;
early KL values rounded to `0.000` in the console. The raw event-file
hash is retained. Console agreement allows three-decimal rounding and
float32 conversion. Rollout return and the goal-reached reason frequency
are training diagnostics, not independently evaluated navigation SR or
semantic accuracy. This adds evidence without changing any frozen gate
or inference schedule.

The three `*_checkpoint_metadata.json` files record separate CPU checks
of the saved Hugging Face exports: all four indexed safetensors shards exist,
their tensor names match the index, declared payload ranges are contiguous,
and each file's extent matches its header. Each export contains 825 tensor
entries in float32. Header hashes refer to canonicalized header JSON, not
the complete weight payload. This checks file structure and completeness;
it does not validate tensor values, perform inference, or establish a
navigation benefit.

The remaining three training runs, reserved-screen evaluation, full
val-unseen evaluation, and shared SFT comparison are still pending. The
existing frozen relay and selection rules are unchanged. Full protocol:
[PILOT_PROTOCOL.md](../PILOT_PROTOCOL.md).
