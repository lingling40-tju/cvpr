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
