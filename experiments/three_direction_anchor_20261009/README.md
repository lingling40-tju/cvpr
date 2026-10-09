# Three SFT-anchored n=4 directions: frozen pilot

Remote root: `/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_direction_20261009`.
All three real two-update multimodal smoke runs completed with finite,
nonzero actor gradients. Those updates count toward the fixed total of
64 updates; each resumes for 62 more. GRPO is currently training, and the
frozen sequential relay will run RLOO and SRGPO-style continuations.
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
`807a815a3de2143215f8e8075a086c9d02a89b5ca14a7c978191ee308cd2d93e`.
It binds the parent training identity, 96 evaluation/data sources and
265 files of a completed, same-manifest FP16 SFT reference. CPU preflight
verified all 13 SFT files including both weight payload hashes, exact
four-shard coverage, actual FP16 startup and a same-file raw-to-independent
compact recount. It made zero model calls. The reused SFT has 109/256
successes, SR 42.58%, SPL 41.26%; this saves repeated baseline inference.

The evaluation relay (launch PID 1374861) waits for all three audited
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

## Additional final raw-ID check

The original frozen validator checks exact filenames and metrics; the evaluator records internal episode identity under `id`. Before any candidate development output, a separate [raw-ID/metric verifier](tools/verify_three_development_raw_ids_20261009.py) and [supplement identity](runlogs/freeze/raw_id_supplement_identity.json) were added. They change none of the original training/evaluation sources or thresholds. After original suite completion, require all 1,024 internal IDs, recomputed five raw paired metrics, and advancement eligibility to agree before using results for scale or updating paper claims.

Run on the remote root after `development_suite/suite.completed`, with a fresh output:

```sh
CUDA_VISIBLE_DEVICES="" python tools/verify_three_development_raw_ids_20261009.py \
  --root "$PWD" --output runlogs/development_suite/final_raw_id_independent_recount.json
```

The old completed [post-hoc termination diagnostic](../trajectory_sil_20261006/completed20261009/posthoc_terminal_proposals/) is available to inform later mechanism design. It does not alter these three current pilots or establish a cause of the old performance loss.

### Automatic CPU final verification

The [supplemental handoff](tools/watch_three_final_raw_ids_20261009.py) is running as PID 1508260; [launch identity](runlogs/final_raw_id_verification/launch.json) binds its source SHA, the original evaluation owner PID 1374861 and both identities. Its [real-owner preflight](runlogs/final_raw_id_verification/preflight.json) passed before any candidate development output. It waits for original `suite.completed`, rechecks the full original freeze and runs the separate raw-ID verifier with `CUDA_VISIBLE_DEVICES` empty. It changes no frozen training/evaluation source, metric, gate or original marker.

Monitor remote `runlogs/final_raw_id_verification/status.json`, `watcher.launcher.pid`, `failure.json` and `watcher.completed`; confirm the actual PID rather than a state file alone. Only `verification_completed` plus the PASS final raw report establishes this supplemental check. Waiting is not a navigation result. A handoff failure must be diagnosed independently of healthy GPU jobs; do not launch a duplicate.
