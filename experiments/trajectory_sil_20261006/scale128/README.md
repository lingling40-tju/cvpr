# Matched positive-trajectory scale: training evidence

The frozen expansion has six fresh runs: control and candidate at configured seeds
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

The three `*_rollout_budget.json` exports recount all original
`rollout.jsonl` records with the [raw budget auditor](../audit_positive_rollout_budget.py).
Each completed arm has exactly 128 step records, eight distinct fit
episode IDs per step, and four recorded trajectories per episode: 1,024
episode exposures and 4,096 trajectories. Both 64-step epochs cover every
one of the frozen 512 fit IDs exactly once as a group. Their logged data
epochs agree with these blocks. All three completed arms have identical
per-step episode membership, including the matched seed-11 pair.

The [independent compact tally](../verify_positive_rollout_budget_compact.py)
reconstructs each epoch and total from the copied step membership and links
the manifest, original audit, configuration and scalar-export hashes.
Its [three-arm recount](rollout_budget_local_recount_three_completed.json)
passes. This checks recorded training membership and budget; navigation
performance and physical-scene correspondence require separate evidence.
Run the local tally on subsequently copied evidence with a fresh output:

```sh
python3 experiments/trajectory_sil_20261006/verify_positive_rollout_budget_compact.py \
  --root experiments/trajectory_sil_20261006 \
  --output /tmp/positive_rollout_budget_recount.json
```

The [sampling-seed provenance audit](sampling_seed_provenance_three_completed.json)
finds that all 32 first-batch action/reward signatures match across these
three runs, with four distinct signatures within each episode group. The
separate [executed-action export](first_batch_executed_actions_three_completed.json)
also contains four distinct flattened and turn-bounded action sequences
in every first-batch group, excluding generated text. Local record recount
agrees and ties all three raw hashes to the existing budget exports.
These checks cover the first batch only. The configured seeds reach
data/engine configuration, but the agent clears
request seeds and the manager initializes GPU generation state at
`1000 + DP rank`. Report three configured-seed replications with a common
initial generation-state rule; independence of every rollout RNG stream
is unestablished. This first-batch check does not show whole-training
identity or explain later variance. The [scope and future correction](../SAMPLING_SEED_SCOPE.md)
document an unapplied patch and CPU-only constructor check. The running
source and the planned six-model evaluation are unchanged.

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

The reusable [CPU metadata exporter](../export_positive_checkpoint_metadata.py)
performs these header and file-extent checks for subsequent completed arms.
It also records the original audit, training log, training configuration,
and exporter hashes. On the completed seed-22 control, every previously
recorded structure field matched `control_seed22_checkpoint_metadata.json`
exactly; its added provenance hashes matched the audit and TensorBoard
export. It refuses incomplete training and an existing output path. It
reads no tensor payloads and writes only the requested report.

For a newly completed arm, run this from the remote experiment root with
the appropriate arm and seed, using a fresh report path:

```sh
CUDA_VISIBLE_DEVICES="" python3 tools/export_positive_checkpoint_metadata.py \
  --root "$PWD" --arm candidate --seed 22 \
  --output runlogs/positive_scale/candidate_seed22_checkpoint_metadata.json
```

The remaining three training runs, reserved-screen evaluation, full
val-unseen evaluation, and shared SFT comparison are still pending. The
existing frozen relay and selection rules are unchanged. Full protocol:
[PILOT_PROTOCOL.md](../PILOT_PROTOCOL.md).

## Automatic CPU archive for remaining runs

The [CPU collector](../collect_positive_remaining_evidence.py) waits for
candidate-22 and both seed-33 runs to complete their 128-step training and
original audit. It then invokes the three hash-guarded exporters above with
GPU visibility disabled. It verifies each report's run, original log,
training audit, configuration and exporter identities before marking that
reporting job archived. Reports are created under remote
`runlogs/positive_remaining_evidence/`; repeat collection reuses them.
Operational status is written atomically in that directory's `status.json`.
The collector owns its separate lock and does not write training or
navigation-evaluation completion markers. Pending jobs require the actual
primary scale owner to remain alive.

A real completed control-22 preflight exercised all three exporters, then
reused their reports on a second pass with unchanged report hashes. The
[local comparison](evidence_collector_preflight_recount.json) agrees with
every previously archived non-UTC field; metadata additionally records its
audit/log/configuration/exporter hashes. The scalar export is byte-identical
to the original control-22 export. Reference reports are retained remotely
under `runlogs/positive_remaining_evidence/preflight_control22/`.

The waiting CPU collector was launched at 2026-10-08 07:25:20 UTC and its
actual process was verified alive. Its three requested reporting jobs were
still pending; neither the preflight nor this waiting state supplies a new
navigation result. After each future job is archived, copy its three reports
and original `positive_scale/*_train_audit.json` to this directory and run
the independent local budget tally before paper updates. Archive completion
is separate from six-model navigation evaluation and the shared SFT comparison.

## Candidate seed-11 recorded-credit reconstruction

For one completed candidate training run, `candidate_seed11_weight_groups.jsonl`
reconstructs the frozen per-trajectory credit rule from its actual 128-step,
4,096-trajectory rollout log. The reconstruction matches the stored per-step
score, advantage extrema, and success-frequency scalars exactly. An independent
Python arithmetic recount verifies all 1,024 four-trajectory groups and agrees
with the reported counts and same-record rule comparisons; its report is
`candidate_seed11_weight_mechanism_independent_recount.json`.

In this run, 1,270 trajectories receive positive reconstructed credit (1,021
successful and 249 failed), while 133 successful trajectories receive zero.
The rule caps 777 positive credits; 165 all-failure groups still contain at
least one credited trajectory. Removing success priority, the failure-score
floor, or the cap changes credit assignment for 519, 462, and 777 trajectories,
respectively. These are descriptions of recorded training data, not policy
ablations, saved token-level advantages, or navigation outcomes. They cover
one configured seed only and do not change the frozen three-seed evaluation.
