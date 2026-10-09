# Three SFT-anchored n=4 directions: frozen pilot

Remote root: `/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_direction_20261009`.
All three real two-update multimodal smoke runs completed with finite,
nonzero actor gradients. Those updates count toward the fixed total of
64 updates; each resumes for 62 more. GRPO and turn-level RLOO have each
completed all 64 updates with 64 nonzero actor gradients, 512 fit exposures
and 2,048 recorded trajectories; raw-source audits and independent local
compact recounts passed. RLOO completed at 2026-10-09 07:59:18 UTC, and
the original sequence automatically launched SRGPO-style continuation.
**No new candidate navigation result is available.**

| Direction | Update |
| --- | --- |
| `grpo_anchor` | Terminal GRPO with reference KL coefficient 0.1 |
| `turn_rloo` | Active-peer turn-level return-to-go/leave-one-out, normalized geodesic progress weight 0.5, gamma 1 |
| `srgpo` | Trajectory group credit plus shuffled cross-state process groups of 16, process weight 0.5 |

The SRGPO-style implementation also uses population rather than native
GRPO sample-standard-deviation normalization. It is a composite; gains
alone cannot isolate process grouping. Process feedback uses privileged
simulator geometry, with generated action spans and observation masks
checked at runtime. It is not an observation-only semantic verifier.

## Completed RLOO training

The [complete RLOO audit](runlogs/completed_training_archive20261009/turn_rloo_completed64_audit.json)
has SHA `6a95fdcf76c5df4a7c7a7b98bae048d97bf249411f6bf5696345b48b57ab848a`.
The unchanged V3 raw auditor checked all 64 stored optimizer updates,
nine finite scalar tags, reference KL 0.1, learning rate 1e-6, reward
signals at all 64 updates, exact n=4 fit membership and original
per-episode budgets. There are 1,028 unexecuted turn-overflow bookkeeping
responses among its 2,048 recorded trajectories; these are not thirteenth
executed turns.

The [separate local compact recount](runlogs/completed_training_archive20261009/turn_rloo_independent_compact_recount.json)
passed without importing the raw auditor or TensorBoard/Parquet reader.
Its [source](tools/verify_three_completed_training_compact.py) independently
recounts exported scalar rows, episode membership, response histograms
and frozen source identities. It does not independently decode the remote
raw records, replay physical trajectories, validate weight tensor values
or measure navigation. This additive archive does not occupy the waiting
evaluation suite's own `turn_rloo_train_audit.json` path or change its
frozen source identity, comparison logic or advancement gates.

## Matched budget and resource plan

All arms start from the same navigation SFT model, use the frozen 512 fit
IDs, batch 8, n=4, seed11 configuration, constant learning rate 1e-6,
success15+nDTW5 reward without a semantic floor, 12 turns and the same
token/action limits. A one-GPU initialization had no available KV cache
blocks at utilization 0.25 and 0.4 before any rollout/update; the recorded
resource revision uses two trainer GPUs at utilization 0.6. With Habitat
on GPU2 and the unrelated GPU3 service, three trainers cannot use that
tested topology concurrently. Training therefore runs sequentially;
evaluation runs two models on GPU0/1 with four GPU2 Habitat shards each.

`runlogs/smoke_evidence/smoke_tensorboard_and_scheduler.json` preserves
the actual smoke scalar precision and constant optimizer scheduler.
The original GRPO smoke console was overwritten by its continuation;
the copied original TensorBoard events supplied the smoke export.
The training gate requires exact optimizer steps 1..64, finite scalars,
nonzero actor/reward signals, all 512 fit exposures and 2,048 trajectories.
Uniform-return groups may have zero advantage/gradient; these are reported.

## Fixed evaluation and advancement

The separate evaluation descriptor SHA is
`0ad45c7e518166fd8ffce502b7bd6354b201cce12729bd9e127e33ff9a319a99`.
It binds the unchanged parent training identity, 104 source/evidence files (the original 96 plus eight audit-revision/source entries) and
265 files of a completed, same-manifest FP16 SFT reference. CPU preflight
verified all 13 SFT files including both weight payload hashes, exact
four-shard coverage, actual FP16 startup and a same-file raw-to-independent
compact recount. It made zero model calls. The reused SFT has 109/256
successes, SR 42.58%, SPL 41.26%; this saves repeated baseline inference.

The evaluation relay (launch PID 1686932) waits for all three audited
64-step completions under the training lock, closes only its own 5086
Habitat service, and starts fixed development256 inference after GPU
availability checks. Each arm must cover the exact 256 IDs/8 scenes
with zero final inference errors and actual FP16 engine proof. Five
raw paired comparisons are independently recounted before suite completion.

Advancement requires SR **and** SPL each >= +2 pp versus SFT.
RLOO and SRGPO-style arms additionally require each >= +2 pp versus the
new matched GRPO anchor. Only a passing frozen method proceeds to a
fresh, matched 512-row/128-step three-configured-seed expansion. If none
passes, use a different mechanism. No reserved or val-unseen screen is
opened for pilot tuning. Development has been reused adaptively, and
configured seeds do not establish independent rollout sampling streams.
No further human annotation is requested.

## Pre-inference audit correction

A CPU inspection found an incorrect assertion in the original audit: every
record was required to have `step_budget=36`. The frozen trainer actually
uses `min(2 * len(gt_actions), 36)` from each original fit row. Of 512 rows,
501 have budget 36 and 11 have a smaller budget (24, 26, 30, 32 or 34);
all have 12 turns. The [revision evidence](runlogs/evaluation_audit_revision20261009/)
retains the original descriptor, waiter launch/log/status files and an
explicit intentional replacement record. GPU training was untouched.

The new [auditor](tools/audit_three_training_per_episode_budget.py) validates
exact per-episode budgets against the hash-frozen fit Parquet, trainer,
YAML and runner. All original optimizer, finite scalar, learning-rate/KL,
fit membership, n=4, 64-update and checkpoint checks remain. A real 64-record
two-update smoke preflight passed, and corrupted command budgets, turn
budgets and IDs were rejected. The new [suite](tools/run_three_dev_suite_budget_v2.sh)
has exactly one source-line change: it calls this distinct audit. Original
inference, metrics, comparisons and +2/+2 advancement gates are unchanged.
The [independent local recount](runlogs/evaluation_audit_revision20261009/independent_local_revision_recount.json)
checks source diffs and all 96 original/265 reused SFT identity entries;
it is not a second Parquet decode or a navigation result.

This amendment was frozen before any candidate development inference.
Old waiters PID 1374861/1508260 were intentionally stopped after exact
command/cwd checks, their states archived, and new owners 1659575/1659631
verified live. The original source files remain present and hash unchanged.
This repairs a false audit rejection, not the separate training/evaluation
timeout-reward convention documented in the old post-hoc budget audit.

## Generated-response trace correction and GRPO completion

Full GRPO auditing exposed a second incorrect assertion inherited from
the original audit: `gen_traj` was treated as at most 12 executed turns.
The frozen environment appends **generated responses**, including the
unexecuted thirteenth response that reports turn-budget exhaustion.
There are 739 such records among the completed GRPO's 2,048 trajectories.
This does not mean 13 turns were physically executed.

The [V3 auditor](tools/audit_three_training_recorded_trace_v3.py) checks
exactly `turn_budget+1` responses only for the turn-limit ending, with an
empty terminal executed-action list, zero final reward and recorded task
failure; every other ending remains within the 12-response budget. It
also hashes the original environment and wrapper source. All prior scalar,
optimizer, row-specific budget and fit-membership checks are retained.
The [complete real training audit](runlogs/evaluation_trace_audit_revision20261009/grpo_anchor_completion_audit.json)
and [independent compact recount](runlogs/evaluation_trace_audit_revision20261009/independent_compact_recount.json)
confirm steps 1..64, all 64 nonzero gradients, 512 fit exposures and 2,048
recorded trajectories. Three forged overflow traces were rejected.

The [transparent second revision](runlogs/evaluation_trace_audit_revision20261009/)
archives the V2 identity and waiters. Current evaluation identity is V3,
with all 99 V2 source identities preserved and five new entries. The
[V3 suite](tools/run_three_dev_suite_trace_v3.sh) only changes the audit
invocation; inference, metric logic, reused SFT, model selection and +2/+2
advancement remain unchanged. Waiters 1659575/1659631 were intentionally
replaced with verified live owners 1686932/1687038 while development was
still unopened; training was untouched. Retained V1/V2 programs are
historical, not the active orchestration. No navigation gain is claimed.

## Additional final raw-ID check

The original validator checks exact filenames and metrics; the evaluator
records internal episode identity under `id`. The [V3 verifier](tools/verify_three_development_raw_ids_trace_v3.py)
and [V3 supplement identity](runlogs/freeze/raw_id_supplement_identity_trace_v3.json)
retain the original raw-ID and paired-metric logic, rebound to the
transparent audit revisions. After suite completion, require all 1,024
internal IDs, five recomputed raw paired metrics and advancement
eligibility to agree before scale or paper claims.

The [automatic CPU handoff](tools/watch_three_final_raw_ids_trace_v3.py)
is running as PID 1687038; its [launch identity](runlogs/final_raw_id_verification/launch.json)
binds the current descriptor and owner PID 1686932. Its
[real-owner preflight](runlogs/final_raw_id_verification/preflight.json)
passed before development opened. It waits for `suite.completed`, rechecks
the full freeze and runs the separate raw-ID verifier with
`CUDA_VISIBLE_DEVICES` empty. Avoid a duplicate manual run.

Monitor remote `runlogs/final_raw_id_verification/status.json`,
`watcher.launcher.pid`, `failure.json` and `watcher.completed`; confirm
actual PID/cmd/cwd. Only `verification_completed` plus the PASS final raw
report establishes the check. Waiting is not a navigation result. Diagnose
handoff failure separately from healthy GPU jobs.
