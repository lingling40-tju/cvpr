# Conditional fallback: success-triggered same-episode resampling

This is an ablation of the existing ActiveVLN dynamic-sampling
option, not a claim to have invented it. After the KL pilot failed and
released GPUs 0/1, a two-step wiring check was started in that lane while
the four-sample pilot advanced to three-seed training on GPUs 2/3. This
parallel pilot tests a different mechanism without waiting for the larger
four-sample result.

## Motivation

The destination-only two-sample 128-step seed-11 control had equal returns in
348/512 episode groups. With GRPO's within-group comparison, these ties can
remove the policy-gradient signal even when the trajectories differ. The
four-sample pilot increases the group size for every episode. This fallback
instead keeps two final samples per episode and resamples a group only when
neither original trajectory reaches the goal. The implementation permits at
most two extra two-sample attempts and keeps the last sampled group. It does
not ensure a successful group; all attempts and simulator cost must be
counted. It might bias training toward lucky successful rollouts or worsen
held-out performance.

## Paired protocol

1. Use a dedicated eight-slot original-reward service on port 5015/GPU 1
   after the KL service on that GPU has stopped. Train on GPUs 0/1 while
   the four-sample scale occupies GPUs 2/3. The older conditional watcher
   for a both-negative outcome stays idle and will not duplicate this run.
2. Use the exact `branch_pilot_train.parquet` (SHA-256
   `a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3`),
   seed 11, the SFT initializer, the destination-only reward, two final
   trajectories per episode, and 64 steps. The sole intervention against
   `branch_control64` is `enable_dynamic_sampling=true` with
   `max_sample_attempts=2`. Use the dedicated eight-slot original-reward
   simulator service.
3. Run two steps first. Audit exact per-step episode pairing, no prefix
   replay, finite rewards and gradients, the code's
   `from_dynamic_sampling` marker, and the number of extra simulator
   trajectories parsed from the logged resampling attempts. Then run the
   64-step pilot with the same audit.
4. Evaluate on the frozen 256 val-unseen manifest, with exact episode
   coverage and zero inference errors. Compare paired SR/SPL against
   `branch_control64`. A positive pilot can trigger three-seed 128-step
   expansion and complete 1,839-episode evaluation; the pilot alone is a
   screening result, not a paper claim.

`run_dynamic_parallel_watcher.sh` waits for the two-step check, then runs
the 64-step pilot and frozen 256-episode evaluation in the free GPU lane.
It stops the dedicated simulator before loading the evaluation model on
GPU 1. The two-step check completed: all eight episode groups matched the
control's train rows, five had different final returns, and eight final
trajectories carried the dynamic-sampling marker. The two-step log recorded
at least eight extra simulator trajectories; Ray's log deduplication in
that smoke prevents an exact total. The 64-step pilot disables log
deduplication so its attempt count can be audited exactly. The smoke
wrapper was replaced while finishing and reported a shell parse error
after training, but the two-step checkpoint and separate audit passed; the
recovery is recorded in `dynamic_smoke/recovery_note.txt`, and no step was
retrained. The 64-step follow-up completed. The older
`run_dynamic_resampling_conditional.sh` watcher will ultimately mark
`not_eligible` because four-sample training passed its screen. No
additional dynamic run should be started by that watcher.

`run_dynamic_scale_conditional.sh` was launched as a no-GPU waiter.
Only if the 64-step pilot has exact fixed-256 coverage, zero inference
errors, paired SR > 0, and paired SPL >= 0 will it train seeds 11/22/33
for 128 steps on the fixed 512-row dataset (SHA-256
`2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea`).
`audit_dynamic_resampling_scaled.py` requires exact per-step train-row
pairing with the completed destination controls, 512 unique episodes per
seed in the exact row order of the Parquet file with the checked SHA-256,
finite rewards and gradients, no prefix replay, and an exact count of
extra simulator rollouts from undeduplicated logs. A passing scale trains
on GPUs 0/1 while the four-sample experiment uses GPUs 2/3. Its full
evaluation uses GPU 1 for inference and GPU 0 for Habitat, separately from
the four-sample evaluation lane. It evaluates both 256 and complete 1,839
val-unseen episodes against the corresponding same-seed controls. The
For an eligible expanded run, `run_dynamic_publication_watcher.sh` would
export compact paired records and checksums after completion. A positive
64-step screen still is not a confirmed navigation gain. This pilot was
negative, so the scale watcher recorded `no_pilot_gain` and did not start
three-seed training.

## Completed seed-11 pilot

The 64-step run matched all 64 step-level episode sets of
`branch_control64`, covering 256 distinct train episodes. It retained 512
final trajectories, the same count as the control, but logged 656 extra
simulator trajectories across 223 resampling attempts: 1,168 total, or
2.28 times the control's 512. In 127/256 candidate groups the two final
returns differed. All four zero-gradient steps were exactly the steps with
fully tied episode groups. These are training and compute observations, not
held-out navigation evidence. The full audit is in
`dynamic64/paired_train_audit.json`.

The frozen 256-episode val-unseen evaluation had exact unique-ID coverage
and zero inference errors. Dynamic resampling succeeded on 74 episodes
(SR 28.91%, SPL 28.55%) versus 75 (SR 29.30%, SPL 29.01%) for the same-data
destination-only control. Exact-ID paired changes are **-0.39 SR and -0.46
SPL percentage points**; the 11-scene exploratory bootstrap intervals are
[-5.00, 4.56] and [-5.10, 4.56] points. The candidate-only and control-only
success counts are 17 and 18. This single-seed screen misses the
predeclared scale gate and cannot establish a general negative effect.
`dynamic64/` contains the paired episode records, manifest-linked analysis,
coverage validation, and package hashes; `verify_val256_pair_package.py`
independently recomputes the paired metrics from the compact records.
