# Conditional alternative: four trajectories per GRPO episode

This is a prospective optimization experiment. It changes the number of
on-policy trajectories sampled per training episode from two to four while
keeping the destination-only reward, training episode rows, SFT initializer,
action budget, optimizer settings, and evaluation manifest fixed. It does
not use a semantic verifier, distance-progress reward, or reference-route
reward.

## Motivation and limits

In the completed seed-11 128-step destination-only control, 348 of 512
two-trajectory episode groups had equal returns, although 297 of those tied
groups had final-distance differences greater than 0.5 m. Equal-return
groups supply no within-group preference to GRPO. Four trajectories may
increase the chance that a group contains both a successful and a failed
route. This is a hypothesis about training-signal frequency, not evidence
that a larger group improves held-out navigation. The four-sample run costs
about twice as many simulator trajectories per training episode and may
also change the number of optimizer minibatches per step.

The current local verl configuration disables both reward KL and actor KL
(`kl_coef=0`, `use_kl_loss=false`). Its documented GRPO interface permits
an actor KL penalty, but this test leaves those settings unchanged so that
the group size is the intervention. See the
[verl GRPO documentation](https://verl.readthedocs.io/en/latest/algo/grpo.html).

## Test protocol

1. Use the exact `branch_pilot_train.parquet` (SHA-256
   `a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3`),
   seed 11, 64 optimizer steps, four episodes per step, and the same
   destination-only reward and starting checkpoint as `branch_control64`.
   `run_group4_pilot.sh` uses a separate Ray directory and output name.
2. Run a two-step wiring check first. Require four distinct rollouts per
   episode, the same episode IDs at each step as the two-sample control,
   finite returns, actual actor updates, and no replayed policy prefix.
   Then run the 64-step pilot and repeat the full audit.
3. Evaluate its checkpoint on the frozen 256 val-unseen episodes, with exact
   coverage and zero inference errors. Compare paired SR and SPL to
   `branch_control64`. The gate for expansion is SR strictly higher and SPL
   nondecreasing. Report simulator rollout count and GPU time alongside any
   gain because sampling cost differs.
4. If the pilot passes, train matched 128-step seeds 11, 22, and 33 on the
   fixed 512-row dataset and evaluate all 1,839 val-unseen episodes, including
   the 1,583 episodes outside the screen. A positive single-seed screen is
   insufficient for a navigation claim.

This experiment can run concurrently with the isolated route-fidelity
pilot only if GPU memory and simulator health remain stable. It uses the
original destination-only simulator and never points at the route or
progress reward services.

The first two-step wiring attempt pointed at an eight-simulator service and
timed out in `batch/reset` before recording any rollout. The service's
`acquire_many` call waits for all 16 environments at once, so an eight-slot
pool cannot satisfy it. That attempt is an infrastructure failure, not a
training or navigation result. `start_group4_service.sh` starts a dedicated
16-slot service on port 5013 before the check is retried.

The retried two-step check completed on that service. Both steps used the
same four episode IDs as the two-sample control, with four trajectories per
episode and no replayed prefix. All eight groups had distinct trajectories;
four had nonzero return variance. The actor gradient norms were 3.557 and
1.134, and the step-2 checkpoint was saved. See
`group4_smoke/paired_train_audit.json`. The 64-step pilot has started on
GPUs 2 and 3, concurrent with the route-fidelity pilot on GPUs 0 and 1;
there is no held-out navigation result yet.
The full-run auditor permits a zero gradient only on a step where all four
episode groups have equal returns; it still requires finite gradients and
effective updates on other steps. This avoids labeling a mathematically
uninformative sparse-reward batch as an infrastructure failure.

The 64-step seed-11 run completed and passed this audit. Its 256 train
episode sets matched the two-sample control exactly at every step. All 256
four-trajectory groups contained distinct trajectories; 94 had nonzero
return variance. Eight optimizer steps had zero actor gradient, and these
were exactly the eight steps where all four episode groups had tied returns.
The run sampled 1,024 trajectories versus 512 for the control. See
`group4_64/paired_train_audit.json`.

The fixed 256-episode val-unseen evaluation completed with 80 successes
for the four-sample checkpoint versus 75 for the same-data two-sample
control. Paired SR is +1.953125 percentage points and SPL is +1.287711
points, with all 256 episode IDs covered and zero inference errors. The
exploratory 11-scene bootstrap 95% intervals are [-3.20, 6.44] SR points
and [-3.78, 5.58] SPL points; both include zero. The 29 candidate-only and
24 control-only successes show a small net difference amid substantial
per-episode discordance. The compact episode package and independent
recomputation are in `group4_64/`. This satisfies the predeclared pilot
gate for three-seed 128-step expansion, which has started. It is not a
confirmed navigation improvement.

## Compute-matched mechanism check if the full result is positive

The current two-sample control uses half as many trajectories and fewer
actor examples per step. Thus even a three-seed full-set gain would first
establish a quality-versus-compute tradeoff, not prove that four-way GRPO
normalization caused it. The trainer assigns one `uid` per episode before
repeating each row for four rollouts; GRPO normalizes returns by this
`uid`. A direct compute-matched ablation would draw the same four
trajectories per row, retain the same 16 actor examples and optimizer
settings, but assign two `uid`s to adjacent pairs of trajectories for
two independent two-way advantage groups. It must use the same train rows,
seeds, model initializer, reward, simulator budgets, and complete 1,839
episode evaluation. If that ablation matches the four-way policy, the
extra rollout budget explains the apparent benefit more plausibly than
four-way grouping. This ablation is specified here before seeing any
three-seed full-val result and has not been run.

The implementation preflight is saved as `group4_pairwise_uid.patch` and
`group4_pairwise_uid.py`. With `VLN_GROUP4_PAIRWISE_ABLATION=1`, the patch
retains four simulator trajectories and 16 actor examples per four-episode
step, then assigns two GRPO `uid`s per episode: rollouts 0--1 and 2--3.
The normal four-way run leaves this flag unset. The helper rejects incomplete,
non-interleaved, or reused episode quartets. Its isolated local test and a
dry-run patch against trainer source SHA-256
`1d1334ac7c4267b32e6354bdc27a4e313b27dc025b6d6d1f535d5729b8420289`
passed. The active remote trainer source was not changed. If the three-seed
full-val result passes, first stage the patch in a separate source tree,
verify the training log reports eight pairwise UID groups per four-episode
step, then use the same 512 train rows, three seeds, and complete 1,839
episode evaluation. A successful wiring test alone is not a navigation
result.

The isolated source tree was staged at
`/Knowin/foundation/haozhiwang/whz/ActiveVLN_group4_pairwise_20261002` by
`stage_group4_pairwise_ablation.sh`. A first NumPy-backed preflight caught
an ambiguous array truth-value check in the helper; it was corrected before
any ablation training. The corrected helper was tested against the actual
GRPO advantage function: rewards [15, 0, 0, 0] give four-way advantages
[1.5, -0.5, -0.5, -0.5] but paired advantages approximately
[0.707, -0.707, 0, 0]. The patched trainer and source hashes are checked
when staging. The isolated tree has no simulator or trainer running yet.
