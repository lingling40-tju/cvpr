# Three VLN alternatives: pilot protocol

This checkout isolates three candidate methods from the completed EventTrace
study. The old Qwen3.8 256-step suite was stopped after a valid seed-11
control step-64 checkpoint; its remaining arms were never run.

## Candidates

- `branch`: 256 train episodes with 4–9 executed policy actions replayed to a
  later decision point. Two independent continuations start from the same
  simulator state and receive standard destination outcome rewards.
- `recovery`: 256 train episodes with nine executed actions replayed from an
  unsuccessful control trajectory. Two independent continuations learn to
  recover, with the same outcome reward.
- `counterfactual`: 128 natural instruction pairs from the R2R train split.
  Each pair shares scene and exact starting pose but has different goals and
  opposite expert initial turn directions. The two instructions share a
  four-rollout GRPO advantage group. A small training-only initial-turn
  bonus uses expert actions; deployment uses only images and instructions.

The 128 pair rows draw from 84 disjoint natural pairs, with repeats to fill
the pilot. The first two methods replay a saved action history in separate
simulator instances; the two continuations therefore reach the same physical
pose and trajectory history, but do not use a simulator snapshot/fork API.
The ordinary GRPO control also samples two continuations per episode. The
branch intervention is specifically the policy-executed late decision state
used for those continuations, versus the control's usual expert-prefix state;
sample multiplicity alone is not an ablation of this idea.

`tools/prepare_three_directions.py` generates the Parquet data and records
counts in `runlogs/three_direction_data_diagnostics.json`. Existing completed
control rollouts supply branch and recovery prefixes. All rows are train
split only. The alternate trainer implementation is in
`verl/trainer/ppo/{ray_trainer.py,alternative_curriculum.py}`.

## Resources and pilot gates

Two independent eight-simulator services are needed for two training jobs.
Port 5002 runs on GPU 3; port 5007 runs on GPU 0. A single shared eight-instance
pool deadlocked when two four-episode, two-sample jobs each reserved half of
it. The first concurrent smoke was terminated after HTTP timeouts; no
checkpoint or metric was accepted from that attempt. The services were
restarted separately before retrying.

The 256-episode val-unseen manifest in `runlogs/three_direction_val256/` is
scene-balanced across 11 unseen scenes. `existing_baselines.json` recomputes
SFT and three earlier step-64 controls on exactly those episode IDs from
their complete 1,839-episode outputs. The seed-11 control has 75/256 SR and
0.2842 SPL. A promising pilot requires a validated held-out gain in SR with
non-decreasing SPL on this frozen manifest. A short pilot alone is a screen:
any promising method then needs matched-data controls, three seeds, all
1,839 val-unseen episodes, and honest paired uncertainty analysis.

The 2-step smoke checks code and reward flow only. Its rollout success is not
a held-out navigation metric. GPU-time and interaction budgets must be
reported beside the navigation scores, particularly for branching.

`run_matched_control_followup.sh` watches the held-out screening result. For
any arm whose 256-episode SR exceeds the old seed-11 control's 75/256 and
whose SPL does not fall, it trains a 64-step control on the **same training
episode rows**, from the same SFT initializer and with the same rollout
count, while disabling that arm's intervention. It then evaluates those
controls on the exact same 256 held-out episodes and writes
`matched_analysis.json` with paired differences and scene-cluster bootstrap
intervals. This screening rule launches a comparison; it is not a statistical
claim of improvement. The training runner accepts an optional fourth seed
argument for later multi-seed work.

`prepare_scaled_directions.py` prepares optional 512-row, non-repeated
curricula. The branch and recovery rows draw from 608 and 517 eligible unique
episodes in prior train rollouts. Counterfactual pairs come from the complete
10,819-episode R2R train JSON and train ground-truth actions; it validates
the numeric-to-text action encoding against all 4,000 existing Parquet rows.
The runner accepts `VLN_TRAIN_DATASET=data/<name>.parquet` to select these
curricula for a later larger experiment. These data are preparation only;
they do not imply that a method passed the held-out pilot gate.
R2R episode numbers can repeat across splits, so the generator checks the
scene identities: the 61 train scenes and 11 val-unseen scenes are disjoint.

The evaluator accepts `VLN_EVAL_RESULT_ROOT`, `VLN_EVAL_MANIFEST`, and
`VLN_EVAL_COUNT` for a later complete evaluation. An exact copy of the
previously validated 1,839-episode manifest is staged under
`runlogs/three_direction_full_val_unseen/manifest.json`; no new full
evaluation has been run yet.

The shared evaluator now archives malformed or inference-error episode files
and retries those episodes up to twice. The final label validator still
requires exact episode coverage and zero inference errors; archived failures
remain available for audit.

The branch and recovery pilot wrappers hit a post-save shell parse error when
their script was overwritten in place. The saved checkpoints were independently
validated from original rollout and TensorBoard records; see
`INCIDENT_64STEP_WRAPPER.md`. Their held-out evaluations started in parallel
with the remaining counterfactual training.

The counterfactual pilot subsequently completed all 64 steps with its own
validated checkpoint. The training audit found 256 diverse instruction-pair
groups and nonzero actor gradients at every step; the exact validation record
is `counterfactual_64step_validation.json`. After that run ended, the recovery
finalizer preserved the two original wrapper failure markers, accepted their
independently checked checkpoints, and restarted the held-out evaluation and
matched-control follow-up.

`counterfactual_val_pairs/` freezes 72 disjoint natural val-unseen pairs
(144 episodes, 9 scenes) that share a start pose but have different goals and
opposite expert initial turns. Cached full-evaluation traces give SFT 62.5%
and the earlier seed-11 destination-only control 57.6% episode-level expert
turn agreement on this selected set. These are path-expert agreement
diagnostics, not navigation SR or a unique correct-action ground truth. The
counterfactual pilot will be evaluated on the same pairs after the main
navigation and matched-control checks finish.

The first complete 256-episode navigation screen is `branch64`:
82/256 success (32.03% SR, 31.38% SPL), versus the earlier seed-11
destination-only checkpoint's 75/256 (29.30% SR, 28.42% SPL) on the exact
same episodes. The paired changes are +2.73 SR and +2.95 SPL percentage
points; scene-cluster 95% bootstrap intervals cross zero. The old checkpoint
was trained on different episode rows. Thus this is a screening signal and
does not establish an effect of branching; the same-data control remains
pending. See `val256/branch_interim_analysis.json` and its validation record.

The completed recovery screen has 65/256 success (25.39% SR, 24.75% SPL),
below the same earlier seed-11 control by 3.91 SR and 3.67 SPL percentage
points. It therefore does not pass the predeclared screening rule. Both
256-episode evaluations had zero inference errors; see
`val256/two_arm_interim_analysis.json`. The branch matched-data control was
started on GPUs 2/3 and service port 5007 while counterfactual training
continued on GPUs 0/1 and service port 5002. The later evaluation suite uses
GPU 0 for Habitat so it can coexist with the matched control if needed.

All three 256-episode screens have now finished with exact coverage and zero
inference errors. Counterfactual has 72/256 success (28.13% SR, 27.38% SPL),
below the earlier seed-11 control by 1.17 SR and 1.04 SPL percentage points.
Only branch passed the predeclared screen against that older checkpoint;
recovery and counterfactual did not. The complete three-arm comparison is
`val256/three_arm_analysis.json`, with the counterfactual coverage audit in
`val256/counterfactual64.validated.json`. The branch same-data control is still
training, so no method-specific gain is established yet.

`analyze_pilot_failure_modes.py` recomputes termination counts on those same
256 IDs. The old seed-11 control reaches its turn cap on 80 episodes; the
branch, recovery, and counterfactual pilots do so on 55, 86, and 113.
Counterfactual therefore has a conspicuous turn-cap failure mode. The
evaluator's trajectory-level oracle-success metric exceeds task success by
3, 3, 9, and 6 episodes, respectively. These descriptive counts are in
`val256/failure_modes.json`; they do not identify the cause of any score
difference or replace the matched-control comparison.

`run_scaled_pair_suite.sh` is queued for branch only and remains idle until
the matched 64-step comparison and targeted diagnostic finish. It starts
only if branch beats its **same-data** control in both SR and SPL on the
fixed 256 episodes. If that gate passes, it trains branch and control on
the same 512 unique train rows for 128 steps at each of seeds 11, 22, and
33, with the two arms concurrent per seed. `run_scaled_val256_suite.sh`
then evaluates all six checkpoints on the frozen 256 episodes and
`analyze_scaled_val256.py` computes per-seed paired changes and scene
bootstrap intervals. These jobs have no result yet; complete 1,839-episode
evaluation is a later replication gate.

`run_scaled_full_val_suite.sh` is also queued without using GPU time. It
opens the full 1,839-episode evaluation only if the scaled 256-episode
analysis has positive mean paired SR, nonnegative mean paired SPL, and
positive SR differences for at least two of three seeds. Candidate and
matched-control checkpoints are then evaluated on separate GPU lanes
(model GPUs 1 and 3, eight Habitat shards in total on GPU 2), and the same
analyzer checks exact full coverage and computes three-seed paired results.
Passing these gates would justify a larger empirical claim; the gates do
not themselves establish one.

The three smoke validations are included here for debugging. The 64-step
pilot suite and automated val-unseen evaluation were launched on 2026-10-02;
their initial metrics above have exact episode coverage and zero inference
errors. The trainer patch applies to the same
ActiveVLN base used by `experiments/implementation/` after its independent
group sampling patch.
