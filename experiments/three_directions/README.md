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
used for those continuations, versus the control's task-start state;
`prob_from_scrath=1` makes its saved history empty. Sample multiplicity alone
is not an ablation of this idea. This comparison tests the full branch
curriculum, including its later start and remaining action budget. An
equal-length expert-prefix control would be needed to isolate the effect of
using a policy-generated prefix specifically.
The trainer counts each grouped action command once: for a replay prefix of
length $L$, it configures branch's generated-command budget as $36-L$,
while the from-scratch control receives a configured budget of 36. The
environment checks this command count after executing each action, so its
strict `>` termination condition can permit one command beyond the configured
budget; a command such as `move forward 75cm` can still
execute multiple lower-level simulator steps. Turn budgets likewise deduct
the turns occupied by replayed history.
`prepare_expert_prefix_ablation.py` now prepares that optional control for
both the 256-row pilot and 512-row scale curricula by replacing each saved
policy prefix with the same number of grouped reference actions on the same
train episode. It has not been trained or evaluated. Equal action count does
not imply equal physical displacement or the same simulator pose; the
ablation would test the curriculum choice more closely without fully
isolating state geometry. Dataset hashes and prefix-length counts are in
`branch_expertprefix_*_diagnostics.json`.
`run_expert_prefix_ablation.sh` is an isolated runner for those datasets. It
uses the branch replay mechanism, a separate checkpoint namespace, and the
same 64-step or 128-step optimizer settings as the corresponding branch run.
The script checks each dataset hash before training. It is staged only; no
expert-prefix model has run. If the three-seed full val-unseen branch result
supports a gain, first compare the 64-step expert-prefix model on the fixed
256 episodes, then decide whether a three-seed complete evaluation is worth
the extra budget. Neither an equal-count pilot nor a positive branch screen
alone identifies a policy-prefix effect.

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
Each training job reserves two of the four available A800 GPUs, so this
server supports two concurrent training arms under the validated setup.
The three pilot directions were scheduled across those two lanes; three
simultaneous training jobs were not feasible at the current per-job memory
and simulator allocation.

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
`VLN_EVAL_COUNT` for complete evaluation. An exact copy of the previously
validated 1,839-episode manifest is under
`runlogs/three_direction_full_val_unseen/manifest.json`; all six scaled
branch/control checkpoints completed this manifest.
The staged file is byte-identical to the earlier EventTrace full-evaluation
manifest (SHA-256 `262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e`):
1,839 unique episode IDs across 11 val-unseen scenes.

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
counterfactual pilot was evaluated on the same pairs with exact 144-episode
coverage and zero inference errors: its expert initial-turn agreement is
63.89%, compared with SFT 62.50% and the older control 57.64%. The paired
change versus SFT is +1.39 percentage points with a nine-scene bootstrap
95% interval of [-4.00, 8.77], so it does not establish a robust gain.
Navigation SR on the separate 256-episode screen was lower than the old
control. The evaluator wrote each shard's episode IDs in a different order
than the manifest; the first diagnostic analyzer incorrectly required list
order and exited after all inference had completed. Its exact-ID and
zero-error checks now accept any unique ordering. The repaired analysis
reused the completed inference files and archived the original failure
marker. `counterfactual_val_pairs/analysis.json` contains all 72 pair-level
observations, and `counterfactual_val_pairs/counterfactual64_pairs.validated.json`
records coverage.

The first complete 256-episode navigation screen is `branch64`:
82/256 success (32.03% SR, 31.38% SPL), versus the earlier seed-11
destination-only checkpoint's 75/256 (29.30% SR, 28.42% SPL) on the exact
same episodes. The paired changes are +2.73 SR and +2.95 SPL percentage
points; scene-cluster 95% bootstrap intervals cross zero. The old checkpoint
was trained on different episode rows. Thus this is a screening signal and
does not establish an effect of branching; the same-data control was pending
at this interim stage. See `val256/branch_interim_analysis.json` and its
validation record.

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
`val256/counterfactual64.validated.json`.

The branch same-data control has now completed 64 steps and its full 256
held-out episodes with zero inference errors. The branch arm succeeds on
82/256 episodes (32.03% SR, 31.38% SPL); the matched control succeeds on
75/256 (29.30% SR, 29.01% SPL). The paired changes are +2.73 SR and +2.37
SPL percentage points, with 16 candidate-only and 9 control-only successes.
Scene-cluster bootstrap 95% intervals cross zero: SR [-1.53, 7.53] and SPL
[-1.79, 7.05] percentage points. This passes the predeclared scale-up gate
but remains a pilot signal. `val256/matched_analysis.json`,
`val256/branch_control64.validated.json`, and
`val256/branch_matched_episodes.jsonl` preserve the summary, coverage audit,
and all 256 paired episode metrics; `export_matched_episodes.py` reproduces
the compact export from raw evaluator output.
`analyze_branch_scene_breakdown.py` summarizes those paired records by the
11 unseen scenes. The three scenes with the largest post-hoc net success
change contain nine branch-only successes and zero control-only successes;
the other eight scenes together have seven branch-only and nine
control-only successes. The pilot's net +7 successes are therefore uneven
across scenes. `val256/branch_scene_breakdown.json` preserves every scene
count. This exploratory decomposition does not define a validated subgroup
effect; the scene-bootstrap interval for the overall pilot already crosses
zero.
Both arms read identical Parquet rows, including the saved policy prefix
metadata. The trainer applies that prefix only when the process-level
`VLN_ALTERNATIVE_MODE` is `branch`; the matched control sets it to empty and
uses the ordinary from-scratch logic because `prob_from_scrath=1`.
Candidate scaled logs show 4–9 prefix actions and `[ALTERNATIVE_PREFIX]`
records; control logs show zero prefix actions and no such records.

`analyze_pilot_failure_modes.py` recomputes termination counts on those same
256 IDs. The old seed-11 control reaches its turn cap on 80 episodes; the
branch, recovery, and counterfactual pilots do so on 55, 86, and 113.
Counterfactual therefore has a conspicuous turn-cap failure mode. The
evaluator's trajectory-level oracle-success metric exceeds task success by
3, 3, 9, and 6 episodes, respectively. These descriptive counts are in
`val256/failure_modes.json`; they do not identify the cause of any score
difference or replace the matched-control comparison.
The recomputed file also includes the **same-data branch control**, which
reaches its turn cap on 54 episodes versus branch's 55. Their paired max-turn
counts are 30 for both, 25 for branch only, 24 for control only, and 177 for
neither. Four of branch's 16 candidate-only successes occur when control
hits the turn cap; three of the nine control-only successes occur when branch
does. Thus the old-control 80-to-55 comparison cannot explain the matched
branch result as fewer turn-cap failures. This is an exploratory failure
analysis, not a causal test of the curriculum.

`run_scaled_pair_suite.sh` completed branch scale-up after the targeted
counterfactual diagnostic and positive matched 64-step screen. Branch and
control used the same 512 unique train rows for 128 steps at each of seeds
11, 22, and 33, with two concurrent training arms per seed. The waiting
serial `run_scaled_val256_suite.sh` watcher was stopped before inference.
`run_scaled_val256_parallel_suite.sh` evaluated all six checkpoints on the frozen 256 episodes in two
model lanes (GPUs 1 and 3, ports 8011 and 8012) with four Habitat shards per
lane on GPU 2. No completed labels or checkpoint data were changed by the
scheduling handoff. All six labels have exact 256-episode coverage and zero
inference errors. `analyze_scaled_val256.py` computes per-seed paired changes
and scene-bootstrap intervals. It also computes an exploratory interval that
resamples training seeds and held-out scenes; three seeds remain a small
sample, so the individual seed results and mean/standard deviation remain
essential. Branch versus matched control succeeds on 56 versus 60 episodes
for seed 11, 70 versus 79 for seed 22, and 73 versus 73 for seed 33. Paired
SR changes are -1.56, -3.52, and 0.00 percentage points; SPL changes are
-1.39, -3.02, and +0.99 points. The three-seed mean paired changes are
-1.69 SR and -1.14 SPL points. Exploratory seed-and-scene intervals are
[-7.63, 3.61] SR and [-7.15, 4.09] SPL points. This screen does not support
a gain; its checked analysis is in `val256/scale_branch_128_analysis.json`.
Complete 1,839-episode evaluation finished for every trained checkpoint,
regardless of this interim screen. Its checked results are under
`scale_full1839/`; the three-seed mean is -1.14 SR and -0.55 SPL percentage
points, with seed SR changes of -7.18, +0.49, and +3.26 points.
`audit_scaled_training_pair.py` checks each seed's 128 contiguous steps,
two rollouts for each of the same four episodes per step in both arms,
512 unique train episodes, and branch-prefix versus from-scratch behavior.
It passed a partial 34-step check on seed 11; all three final 128-step audits
have now passed after their paired checkpoints completed.
`run_scaled_pair_audit_watcher.sh` runs independently of training and
evaluators. After each seed's two arms have completed, it runs that full
128-step audit and writes `seed{seed}_pair_audit.json` in the scaled suite
directory; its own status is in
`runlogs/three_direction_scale_branch_128step_pair_audit/`. It never alters
training or evaluation checkpoints.
The first full audit, for seed 11, has passed: 128 paired steps use the same
512 unique train episodes in both arms, with two rollouts per episode and
4–9 replayed commands only in branch. Branch generated 25,311 commands and
replayed 6,088; control generated 32,367 and replayed none. Total grouped
commands were 31,399 and 32,367. Wrapper elapsed time from config write
through checkpoint validation was 1.81 and 1.98 hours, respectively; this
is not a precise GPU-hour measurement. Both arms had 512 diverse trajectory
groups; branch and control had 130 and 164 groups with nonzero return
variance, and 84 and 97 training steps with nonzero actor gradient norm.
These are training diagnostics, not held-out navigation scores. The exact
audit and two validation records are under `scale_budget/seed11_*`.
The first 32 steps of seed 22 also pass a **partial** pairing audit:
128 unique episodes appear in both arms at the same optimizer steps, with
two rollouts each, branch prefixes of 4–9 commands, and zero replayed
commands in control. The corresponding interaction counts are in
`scale_budget/seed22_partial32_audit.json`; they are not a substitute for
the completed 128-step audit or held-out evaluation.
The 64-step partial audit extends this check to 256 matched train episodes
and 512 rollouts per arm. Branch generated 13,190 grouped commands and
replayed 3,012; control generated 16,148 and replayed none. Totals were
16,202 and 16,148. In the first 64 logged optimizer steps, 46 branch and
49 control steps had nonzero actor gradient norms. These are training
diagnostics only; see `scale_budget/seed22_partial64_audit.json` and
`scale_budget/seed22_partial64_gradients.json`.
The completed seed-22 audit now confirms all 128 paired steps, the same
512 unique training episodes and 1,024 rollouts per arm. Branch generated
26,056 grouped commands and replayed 6,088, versus 31,791 generated
commands and no replay in control; total grouped commands were 32,144
and 31,791. Wrapper elapsed time was 1.84 and 1.96 hours. All 512 groups
per arm had diverse trajectories; branch and control had 123 and 167 groups
with nonzero return variance, and 89 and 99 steps with nonzero actor
gradient norm. These are training diagnostics only. The exact audit and
two validation records are in `scale_budget/seed22_*`.
The first 64 steps of seed 33 pass a partial pairing audit: 256 unique
train episodes and 512 rollouts per arm match at every step. Branch
generated 12,864 grouped commands and replayed 3,012; control generated
16,240 and replayed none. This verifies the partial training budget only;
the exact record is `scale_budget/seed33_partial64_audit.json`.
A later 96-step partial audit also passes: both arms contain the same 384
unique training episodes and 768 rollouts, with 23,968 and 24,177 total
grouped commands for branch and control. Its exact record is
`scale_budget/seed33_partial96_audit.json`. The final 128-step seed-33 audit
also passes: 512 identical train episodes and 1,024 rollouts per arm, with
25,767 generated plus 6,088 replayed commands in branch versus 32,143
generated in control. Total grouped commands are 31,855 versus 32,143;
wrapper elapsed times are 1.86 and 2.00 hours. All 512 groups per arm have
diverse trajectories. Nonzero return variance occurs in 132 versus 165
groups, and nonzero actor gradient norm in 93 versus 97 steps. The audit and
two validations are in `scale_budget/seed33_*`.
The audit also checks the environment's action-command counters against each
dataset prefix and reports generated, replayed, and total grouped commands.
On the first 48 matched steps of seed 11 (192 unique episodes, 384 rollouts
per arm), branch generated 9,962 and replayed 2,242 commands, while control
generated 12,234 and replayed zero. Thus total commands were 12,204 and
12,234, respectively, in this **partial** prefix of training. A forward
command may comprise multiple lower-level simulator steps, and replaying a
saved command costs differently from sampling a new model action. These
counts are not GPU compute or a final interaction-budget comparison. See
`scale_budget/seed11_partial48_budget_audit.json`. Across all three completed
seeds, `summarize_training_budget.py` checks the final audits and validation
records and writes `scale_budget/three_seed_training_budget.json`. Mean total
grouped commands are 31,799 for branch and 32,100 for control; mean wrapper
elapsed time is 1.84 versus 1.98 hours. These are descriptive training costs,
not GPU-hours or held-out navigation results.

Before any scaled validation result was available, the full-evaluation plan
was changed to evaluate **all six** 128-step checkpoints on all 1,839
val-unseen episodes. The single-seed 256-episode pilot was uneven across
scenes, so its three-seed 256-episode mean is a useful interim screen but
not a reason to omit the larger evaluation. The original conditional
`run_scaled_full_val_suite.sh` watcher was stopped while it was waiting;
`run_scaled_full_val_suite_all.sh` now waits for the six-model 256-episode
analysis and then runs the full set regardless of its sign. Candidate and
matched-control checkpoints use separate model GPU lanes (GPUs 1 and 3),
with eight Habitat shards in total on GPU 2. The analyzer requires exact
coverage, zero inference errors, and paired three-seed SR/SPL results.
During this full evaluation, `run_extra_full_val_gpu0.sh` adds a third model
lane on GPU 0 and port 8013 for the two seed-33 checkpoints, sequentially,
with four more Habitat shards on GPU 2. The existing per-label evaluator lock
prevents simultaneous inference for the same checkpoint, and its completed
marker makes the main two-lane suite skip a label finished by the extra lane.
All three lanes use the same exact 1,839-episode manifest and unchanged
decoding settings. The extra lane began writing seed-33 control episodes
while the main lanes were evaluating seed 11; the full six-model analysis
remains the completion gate.
Neither the interim gate nor completion of inference alone establishes a
positive method effect.
The upstream ActiveVLN evaluator samples each turn at temperature 0.2 and
top-$p$ 0.8, with no per-request seed. The fixed manifest pairs episodes,
but one decode per episode and checkpoint leaves inference randomness in
the observed SR/SPL; the scene/seed bootstrap is conditional on those
realized decodes. If the complete three-seed comparison has a positive
mean paired SR with nondecreasing SPL, a second six-model pass on the same
1,839 episodes in a separate output directory will check decode stability
before a strong manuscript claim. It will be reported as a separate
replicate rather than folded into the training-seed standard deviation.
The waiting `run_scaled_full_val_replication_suite.sh` implements this gate
without using GPU before the first complete comparison. If eligible, it
reuses the exact 1,839-episode manifest in a separate result root, sets
vLLM server seed 20261003, and evaluates all six models in two lanes.
`run_direction_eval.sh` accepts this optional `VLN_VLLM_SEED`; the primary
evaluations leave it unset and retain their prior server invocation. If the
gate is not met, the watcher records `not_eligible` without inference.
For the subsequent decision, a positive mean paired SR and nonnegative mean
paired SPL in **both** complete passes is a candidate benefit, not proof of
statistical reliability: we will still report all three seed differences and
the exploratory scene/seed interval. If the first pass misses that sign
criterion, or a triggered second pass reverses it, we will not claim a
branching improvement. The first pass missed the criterion, and the isolated
progress-reward pilot below has now run.
If both passes retain the sign, we will inspect the prepared expert-prefix
ablation before attributing the effect specifically to policy-visited states.
`render_scaled_table.py` rendered the two-split CVPR result table only
after both six-model analysis JSON files existed. It rechecks the expected
labels, episode counts, zero inference errors, paired differences, and
three-seed mean/standard deviation before writing LaTeX. The table is in
`scale_full1839/scaled_results_table.tex`.
`export_matched_episodes.py` now accepts `--expected-count 1839` and the
scaled model labels, so each seed's complete candidate/control outcomes can
be exported as a compact paired JSONL. Its unchanged 256-episode behavior
was checked against the original pilot export byte for byte (SHA-256
`83c4ad2cba9e908ce9b8f8418b46563e7ec7534fd38b153b34f08e37a04120bf`).
Full-scale paired JSONLs were created only after their raw evaluations
passed exact-coverage and zero-error validation.
`run_publication_package_watcher.sh` monitors the paired training suite,
three final training audits, and both six-model evaluation suites. Once
all four stages complete, it runs the gated `package_scaled_results.sh` to
export six compact paired JSONLs, both aggregate analyses, a checked LaTeX
table, and SHA-256 checksums into one publication-input directory. It also
uses `analyze_disjoint_holdout.py` to compute a secondary comparison on the
1,583 complete val-unseen episodes outside the fixed 256-episode development
screen. This remainder spans ten scenes: the screen contains all 18 episodes
of one small scene. The standard 1,839-episode benchmark remains the primary
reported result, and this disjoint analysis checks how much the development
subset affects it; it does not erase earlier exposure to other full-set
results. `analyze_decode_overlap.py` also compares the six models' two
separate stochastic decodes on the same 256 episode IDs: the original screen
and their appearances in the full-set run. It records per-model outcome
discordance and the change in paired SR/SPL, so a noisy small screen is not
mistaken for a training effect. The disjoint analysis was specified before
any of the six scaled full-set model labels completed; the decode-overlap
analysis was added after seed 11 completed and exposed substantial
same-checkpoint outcome changes. The package completed and its SHA-256
checksums were verified after copying to `scale_full1839/`. The 1,583-episode
subset mean differences are -1.22 SR and -0.55 SPL points; its exploratory
interval includes zero. On the overlapping 256 episodes, between 20 and 41
success outcomes change per checkpoint between the two stochastic decodes.
`python3 verify_scaled_package.py` independently recomputes every model's
SR, SPL, and mean goal distance, each paired seed difference, and the
three-seed mean/standard deviation for the 256, 1,839, and screen-disjoint
1,583 episode sets from the archived JSONLs. It also checks package hashes,
episode identities, and scene identities; all checks passed.
The full 1,839-episode result is the primary comparison: three-seed mean
changes of -1.14 SR and -0.55 SPL points, with exploratory scene-and-seed
95% intervals [-6.51, 4.02] and [-6.16, 4.85], respectively. No consistent
navigation benefit was found.
`analyze_matched_pair.py` provides the same exact-episode, zero-error gate
and scene-cluster paired analysis for later fallback or ablation labels on
either the 256- or 1,839-episode manifest. On the completed branch64 versus
same-data control64 pilot, it reproduced the existing SR/SPL changes,
discordant-success counts, and bootstrap interval exactly.

The three smoke validations are included here for debugging. The 64-step
pilot suite and automated val-unseen evaluation were launched on 2026-10-02;
their initial metrics above have exact episode coverage and zero inference
errors. The trainer patch applies to the same
ActiveVLN base used by `experiments/implementation/` after its independent
group sampling patch.

`RELATED_WORK_POSITIONING.md` records the closest methods and the claim
boundary for any later branch-curriculum paper. It does not promote the pilot
screen to a confirmed navigation result.
`BRANCH_METHOD_DRAFT.md` records manuscript-ready method and protocol text
checked against the implementation. It includes the completed scaled
256-episode screen and complete 1,839-episode comparison.
`PROGRESS_REWARD_FALLBACK.md` defines and reports a route-independent,
bounded distance-progress reward test. Its motivation came from the three
seeds' **training** rollouts, but the held-out pilot was negative. The
64-step seed-11 checkpoint trained on the same rows as the destination-only
control; all 64 step-level episode sets matched, and 510/512 rollouts had a
nonzero progress component. On the frozen 256 val-unseen episodes it reached
50 successes versus 75 for the control: paired SR -9.77 and SPL -9.63
percentage points, with exact coverage and zero inference errors. The
independently checked paired records and training audits are in
`progress64/`. `run_progress_scale_conditional.sh` recorded `no_pilot_gain`,
so no 128-step expansion or full-val decoder replication was run.
`ROUTE_FIDELITY_FALLBACK.md` and `terminal_ndtw_all_reasons.patch` define a
separate, conventional reference-route reward experiment. The patch extends
terminal generated nDTW to well-formed budget-exhausted rollouts. Its
64-step seed-11 run matched all train-row sets, with 507/512 nonzero nDTW
components. The fixed 256 val-unseen pilot succeeded on 74 episodes versus
75 for its same-data destination-only control: paired SR -0.39 and SPL
-0.40 percentage points, with exact coverage and zero inference errors.
The conditional scale watcher recorded `no_pilot_gain`, so this arm does
not advance to three-seed training. `route64/` contains the audits and
independently checked paired episode package. It does not show a navigation
improvement.

`GROUP4_SPARSE_REWARD.md` defines a separate destination-only GRPO
optimization test that samples four trajectories per training episode rather
than two. Its two-step wiring check passed exact train-row pairing and
nonzero actor-update checks after a dedicated 16-simulator service replaced
an insufficient eight-slot service. The matched 64-step seed-11 pilot
completed and passed training audit: all 256 train episode sets matched the
control, 94/256 four-sample groups had return variance, and all eight
zero-gradient steps had fully tied episode groups. The run used 1,024
trajectories versus 512 for its control. On the frozen 256 val-unseen
episodes, it succeeded 80 times versus 75 for the control: paired SR
+1.95 and SPL +1.29 percentage points, exact coverage, and zero inference
errors. The exploratory scene intervals include zero. This passed the
predeclared gate and triggered three-seed 128-step training, but is not a
confirmed navigation improvement. The training audit and independently
recomputed paired episode package are archived in `group4_64/`.
The prospective compute-matched mechanism check draws four trajectories
per episode but normalizes them as two independent pairs. Its isolated
two-step wiring run passed train-row, actual UID pairing, reward-isolation,
gradient, and checkpoint checks; compact evidence is in
`group4_pairwise_smoke/`. Three-seed training and full-val evaluation of
this ablation remain conditional on the four-way arm's full result.
The 64-step seed-11 compute-matched pilot is running concurrently on the
released GPU 0/1 lane. It will compare paired outcomes with both the
four-way 64-step checkpoint and the two-sample control on the same frozen
256-episode screen; its result is not yet available.
`run_group4_early_eval_after_pairwise.sh` waits for that pilot to finish
and release GPU 0/1, then evaluates the already trained four-way seed-11
128-step checkpoint on the fixed 256 and complete 1,839 val-unseen sets.
The regular scaled evaluator uses GPU 2/3 after all three training seeds
finish and skips any completed seed-11 labels. This overlap saves wall
time without using the development screen as the final result.
`run_group4_pairwise_scale_conditional.sh` waits for the 2+2 pilot's
completed paired analysis against the two-sample control. The same
predeclared gate, strictly positive SR and nonnegative SPL on the fixed
256, permits three 128-step seeds on the hashed 512-row training set.
An ineligible pilot records `no_pilot_gain`. An eligible pilot waits for
the early four-way evaluation to release GPU 0/1, then trains there with
four trajectories per row and two actual GRPO UIDs per episode. It waits
for each matching four-way seed's completed run before the corresponding
2+2 audit, and stops its dedicated simulator when training completes.
This gate is only a reason to spend the larger training budget; complete
1,839-episode evaluation remains necessary for a navigation claim.
`run_group4_pairwise_scale_eval_conditional.sh` waits for that training
suite. If eligible, it evaluates all three 128-step 2+2 checkpoints on
both fixed 256 and full 1,839 manifests using GPU 0/1 after the training
service stops. It checks each same-seed comparison with the two-sample
control, then waits for four-way evaluation to finish and compares with
the compute-matched four-way checkpoint. `analyze_group4_pairwise_scaled.py`
records both sets of paired differences and descriptive seed mean/SD.
`run_group4_pairwise_publication_watcher.sh` then uses
`package_group4_pairwise_scaled.sh` to archive both comparisons at both
evaluation sizes, three train audits, and hashes. The separate
`verify_group4_pairwise_scaled_package.py` recomputes each seed's metrics,
both paired differences, mean/SD, and the 1,583 episodes outside the
development screen from the compact records. Only after that check should
`analyze_group4_pairwise_uncertainty.py` calculate exploratory scene and
seed intervals for the two comparisons.
`package_val256_pair.py` exports completed fixed-screen comparisons as
compact paired records after checking raw episode files, shard summaries,
coverage, and the saved analysis. `verify_val256_pair_package.py`
independently recomputes the exported metrics and hashes; the pair was
smoke-tested on the already completed progress pilot.

`KL_ANCHORED_GRPO.md` defines a separate optimizer intervention: actor KL
coefficient 0.001 relative to the SFT reference, with the destination-only
reward, two samples per episode, and train rows unchanged. Its two-step
wiring check matched the same-data control's episode sets, logged KL loss
and nonzero actor gradients, and saved a checkpoint. The 64-step seed-11
pilot completed with exact training-row pairing and nonzero full-precision
KL scalars at all 64 steps. On the frozen 256 val-unseen episodes it had
72 successes versus 75 for the control: paired SR -1.17 and SPL -1.43
percentage points, exact coverage, and zero inference errors. This misses
the scale gate, so no three-seed KL expansion runs. `kl_anchor64/` holds
the training audit and independently recomputed paired episode package;
the dedicated simulator was stopped after evaluation.

`run_optimizer_scale_conditional.sh` waits for each fixed-256 pilot's
completed, zero-error paired analysis. Only a strictly positive paired SR
and nonnegative paired SPL against `branch_control64` permit its matching
128-step three-seed training suite. An ineligible pilot records
`no_pilot_gain` and consumes no scaled training GPU time. The scaled
`run_optimizer_scale.sh` uses the same 512 train rows and seed-specific
episode order as the completed destination-only controls, with either four
rollouts per row or the 0.001 actor KL loss. `audit_optimizer_scaled_pair.py`
checks all 128 per-step episode sets, all 512 distinct training episodes,
the exact order of those episode sets in the hashed Parquet dataset,
reward isolation, gradient finiteness, and completed checkpoints for each
seed. A completed scale training marker is only a training result; held-out
navigation claims require subsequent fixed-256 and complete-1839 paired
evaluation. The four-sample pilot passed this gate; seed 11 of its scaled
training finished the 128-step audit, while seeds 22 and 33 are being run.
The KL pilot missed the gate and has no scaled run. No scaled held-out
navigation result is available yet.
`run_optimizer_scale_eval_conditional.sh` waits for both gate decisions and
any eligible three-seed training suites. It then stops the two dedicated
training simulators and evaluates eligible modes on separate GPU pairs. For
each seed it compares the candidate with the already completed same-seed
destination-only control on both fixed 256 and complete 1,839 val-unseen
episodes, checking exact manifest hashes, coverage, and zero inference
errors. `analyze_optimizer_scaled.py` records each paired seed and the
descriptive three-seed mean and sample standard deviation. The full 1,839
episodes remain the primary navigation comparison even if the expanded
256-episode result loses the pilot's apparent gain.
After eligible full evaluations complete, `run_optimizer_publication_watcher.sh`
uses `package_optimizer_scaled.sh` to export six compact same-seed paired
episode JSONLs (three for 256 and three for 1,839), both manifests, both
aggregate analyses, three train audits, and SHA-256 checksums. The separate
`verify_optimizer_scaled_package.py` recomputes every seed's SR, SPL, mean
goal distance, paired differences, three-seed mean/SD, and the 1,583
episodes outside the development screen from those archived rows. It does
not treat a package marker as proof until this independent check passes.
After that check, `analyze_optimizer_uncertainty.py` can derive exploratory
paired scene-cluster intervals per seed and scene-plus-seed intervals for
the 256 development episodes, all 1,839 episodes, and the 1,583 episodes
outside the screen. It reads the compact package, resamples the 11 scenes
and three seeds, and keeps the full 1,839-episode result primary. Run it
from this directory once the package is complete:

```bash
python analyze_optimizer_uncertainty.py PACKAGE --mode group4 --output uncertainty.json
```

`DYNAMIC_RESAMPLING_FALLBACK.md` reports the completed conditional experiment
using the free GPU 0/1 lane after the KL pilot ended. It uses the existing
ActiveVLN same-episode success-triggered resampling option, up to two extra
two-trajectory attempts per unsuccessful group. Its 64-step seed-11 run
passed exact training-pair and rollout-cost audit: 512 final plus 656
additional simulator trajectories. On the fixed 256 val-unseen episodes,
it succeeded 74 times versus 75 for the same-data control: paired SR -0.39
and SPL -0.46 percentage points, with exact coverage and zero inference
errors. It missed the predeclared scale gate, so no three-seed dynamic
expansion runs. The compact paired episode package and audit are in
`dynamic64/`, and the smoke recovery note is in `dynamic_smoke/`.
The remote has four 80-GB GPUs, and each current trainer reserves two.
The four-sample scale uses GPUs 2/3 while dynamic resampling uses 0/1;
their Habitat services share GPUs 0 and 1 with the latter trainer. A
2026-10-02 09:31 UTC snapshot reported about 52, 48, 40, and 44 GB in use
on GPUs 0--3, respectively. Two disjoint training lanes are therefore in
use. A third simultaneous two-GPU trainer would have to share devices and
has not been tested; the experiments instead overlap different pairs of
directions over time.
`run_dynamic_scale_conditional.sh` independently checked the fixed-256
paired result and recorded `no_pilot_gain`, without starting 128-step
dynamic training or a complete-1,839 evaluation.

## Representation-derived reward pivot (2026-10-02)

The next method fixes group size at four for its main comparison and uses a
small group-eight replication. `REPRESENTATION_REWARD_PROTOCOL.md` records
the train-only ordinal visual-progress representation, same-start
counterfactual instruction negatives, confidence-gated reward/advantage,
offline calibration gates, paired navigation tests, and compute accounting.
This is a protocol, not a result. The completed four-sample outcome-only
pilot and the running three-seed extension remain baselines. The 64-step
four-rollout 2+2 pairing pilot completed exact 256-episode evaluation with
71 successes (27.73% SR, 27.20% SPL), compared with the ordinary four-way
pilot's 80 (31.25% SR, 30.29% SPL) and the older two-way control's 75
(29.30% SR, 29.01% SPL). Its paired SR/SPL differences are -3.52/-3.09
percentage points versus four-way and -1.56/-1.80 versus the older control.
The three-seed 2+2 watchers were temporarily paused before the pilot result;
after the negative pilot they were resumed solely to record the predeclared
`no_pilot_gain` decision. No scaled 2+2 training was launched.

`prepare_ordinal_progress_manifest.py` selects 512 representation-fit and
128 calibration episodes from disjoint R2R train scenes, including 128 and
32 same-start/different-goal pairs. The remote manifest has SHA-256
`4522b6de453c079d6865d7484d38cdc196ba61103f45f2e79b4e9fcb925f6b65`;
all ten calibration scenes contain a selected counterfactual pair, and
neither subset uses any val-unseen scene. `collect_ordinal_progress_frames.py`
is a resumable renderer for sparse train-only expert frames. An initial
two-episode smoke exposed Habitat's reset ordering, which was repaired and
checked by actual episode ID. A full calibration collection then exposed an
inconsistent nDTW split; after repair, 43 atomic records were collected
before the run was intentionally replaced by a balanced 32-episode pilot.
`ordinal_progress_selection.py` selects 16 fit and eight calibration natural
instruction pairs among the 64/32 pilot episodes. Both subsets were
collected with exact episode coverage and no errors. The cached CLIP-B/32 and
SigLIP-B/16 features and small progress heads were evaluated on 480 ordered
frame pairs and 32 counterfactual comparisons from those 32 calibration
episodes. Raw CLIP scored 56.9%/53.1% (ordinal/counterfactual); its
current-only head scored 47.9%/56.3%, and start-relative head 53.5%/59.4%.
Raw SigLIP scored 51.7%/53.1%; its current-only head scored 47.3%/56.3%,
and start-relative head 58.3%/68.8%. Checkpoint selection used this same
small calibration set, so these are optimistic development diagnostics,
not independent validation or navigation results.
`run_ordinal_full_collection.sh` then completed the 512/128 train-scene
collection, and `run_ordinal_full_representation.sh` cached SigLIP features
and fitted three head seeds plus the current-only ablation. The full-data
script selected checkpoints on five calibration scenes and reported the
other five as a locked audit. Neither script launched RL.
`REPRESENTATION_REWARD_PROTOCOL.md` fixed the offline quality gate before
the full result was read. Compact pilot reports and the
train-only selection manifest are in `ordinal_progress/`; rendered images
and model weights remain on the experiment host.

The full 512/128 train-only frame collection has now completed. A Habitat
top-down-map boundary error interrupted the first attempt immediately after
calibration; the collector was repaired to request only its needed
distance-to-goal audit metric and resumed from atomic episode records.
The initial exception log and recovery note remain on the host. Five scenes
selected checkpoints, and the other five formed the full-run locked audit.
`verify_ordinal_full_representation.py` reloads all three checkpoints,
recomputes the scene-disjoint audit, and reports scene bootstrap intervals.
It confirmed mean ordinal accuracy 74.30% but same-start different-goal
accuracy only 47.22%, versus the frozen SigLIP baseline's 54.22%/53.33%.
The prespecified counterfactual gate failed. The full results and compact
collection summaries are in `ordinal_progress/full512x128/`. Some audit
episodes were already part of the earlier small architecture screen, so
this is not a wholly untouched model-development test. No RL was launched
with this reward.

`fit_temporal_counterfactual.py` is an exploratory next representation:
a causal visual-history matcher trained with same-start instruction swaps
and a repeated-frame penalty. It reuses the frozen image features and
reserves separate fit scenes for checkpoint selection and audit. Its seed-11
audit obtained 98.97% ordinal order accuracy but only 50.0% instruction
counterfactual accuracy; repeated-image positive score increase averaged
0.029. The order score is vulnerable to a sequence-position shortcut and
the candidate failed its offline gate. Its compact report is archived with
the other full-data diagnostics. A next reward model needs demonstrated
instruction grounding on failed *policy* trajectories before group-four
RL is justified.

The first endpoint-image/instruction contrastive adapter on the same 512
expert episodes also failed: its held-out train-scene counterfactual score
was 53.33%, equal to frozen SigLIP, and only 51.67% of expert endpoints
scored above their starts. `fit_goal_contrastive.py` and its report document
the attempt. Directly rendered four-view train goal panoramas from those
same 640 episodes showed 53.33% correct-versus-swapped instruction accuracy
on the five-scene audit. The images and embeddings stay on the experiment
host; `collect_goal_views.py`, `cache_goal_view_features.py`, and
`analyze_goal_views.py` reproduce collection and analysis.

`prepare_goal_view_manifest.py` then froze a larger train-only selection:
2,755 fit episodes with 4,654 natural different-goal pairs and 400
calibration episodes with 382 pairs. The resumable
`run_goal_view_expanded.sh` collects four goal-pose views once, caches
SigLIP embeddings, and audits frozen retrieval. It groups simulator
episodes by scene to reduce asset reloads and runs on GPU 0 while the
group-four baseline trainer occupies GPUs 2/3. `fit_panoramic_goal.py`
sets the subsequent goal-view and ordinary expert-view transfer gates;
it trains no navigation policy. The intended online score sees only an
image and instruction, never a goal coordinate. A positive goal-view score
alone cannot trigger RL without transfer to ordinary agent views.

The expanded 2,755/400 train-only goal-view collection and frozen feature
pass completed with exact coverage and zero collection errors. Frozen
SigLIP's five-scene image-to-instruction accuracy was 57.18%; the
scene-balanced pairwise adapter reached 64.87%, while its reciprocal
instruction-to-image score was 64.36%. On ordinary expert-view transfer,
the adapter reached 60.0% correct-instruction preference and 80.0%
endpoint-above-start rate. The predeclared 75% matching requirements
failed, so seeds 22/33 and the four-sample RL pilot were not launched for
this adapter. `goal_views_expanded/raw_audit.json` and
`goal_views_expanded/panoramic_goal_seed11_report.json` retain the
train-scene diagnostics. The next candidate should use *policy* success
and failure trajectories, then pass scene-disjoint instruction and process
reward audits before online training.

The next screen reuses the completed group-four seed-11 and seed-22 train
rollouts. `prepare_policy_preference_manifest.py` selected 400
same-episode success/failure pairs from the two 512-episode runs, excluding
failures within 3.5 m of the goal. The fixed manifest has 280/66/54 pairs
across disjoint 41/8/8 train-scene fit/development/audit splits. Two pairs
were replayed from executed actions and matched all four original terminal
distances exactly. `run_policy_preference_collection.sh` then replayed all
800 trajectories with exact coverage, zero errors, and terminal-distance
validation; the sparse images and frozen features stay on the experiment
host. `fit_policy_goal_joint.py`
predeclares a visual preference and hard instruction-negative audit before
any further group-four RL pilot. The independent blind semantic audit still
has only one AI labeler and no human ground truth.

The 800-trajectory visual cache and seed-11 joint preference audit have
finished. On the eight-scene, 54-pair train audit, frozen SigLIP ranked
successful over failed policy trajectories correctly in 42.59% of pairs;
the jointly trained adapter scored 44.44%. It discriminated the correct
instruction on 55.95% of 538 held-out natural goal pairs. All predeclared
offline gate criteria failed, so no RL branch or additional head seeds were
launched. `policy_preference/raw_audit.json` and
`policy_preference/joint_seed11_report.json` contain the results. The next
offline representation check should use the navigation SFT model's own
multimodal hidden state before considering another reward implementation.

`cache_navigation_sft_state.py` is the next frozen offline probe. It reads
the same 800 train-only replay records, uses the original Qwen2.5-VL-3B
navigation SFT checkpoint and action prompt, and caches four fused hidden
states and STOP-versus-MOVE/TURN logits per trajectory. A two-trajectory
smoke produced eight finite state vectors. `analyze_navigation_sft_state.py`
predeclares scene-disjoint pair ranking and a small regularized linear
probe; `run_navigation_sft_probe.sh` caches the complete features on GPU
0 without another policy rollout. A positive result must still survive
swapped instructions, group-four RL, and complete val-unseen evaluation.
The first full frozen SFT pass used a simplified evaluation-style prompt;
its 54-pair train-scene audit yielded 59.26% final-frame STOP ranking and
55.56% regularized hidden-state ranking. `run_navigation_sft_server_prompt.sh`
now reruns the feature cache with the VLNCE training server's exact
initial/post-action observation templates, using a prompt-version marker
to prevent reusing the earlier cache. This remains an exploratory
train-scene screen because the scenes have already been inspected.
`run_group4_seed22_early_eval.sh` schedules the already trained seed-22
group-four checkpoint's fixed-256 and complete-1,839 evaluations on GPUs
1/0 after the short SFT feature pass. This overlaps inference with seed-33
training on GPUs 2/3 and uses the same manifest, evaluator, labels, and
paired analyzer as the main suite; the main suite skips completed labels.

The corrected single-observation SFT pass completed all 3,200 train-only
frames. Its 54-pair audit obtained 57.41% final STOP-readiness ranking
and 64.81% for a development-selected linear hidden-state probe; the
probe's eight-scene interval was 56.82--70.37%. Success-vs-failure
endpoint progress differed by 9.26 points. This misses the 75% ranking
and 10-point progress-gap gates, so no reward rollout was launched.
`policy_preference/navigation_sft_server_prompt_audit.json` records the
corrected result. The prompt is per-observation, not a full replay of the
policy's multi-turn history.

Seed-22 group-four validation ran on GPUs 1/0 with
`run_group4_seed22_early_eval.sh`, overlapping seed-33 training on
GPUs 2/3. On the fixed 256-episode screen, group four succeeded on 74/256
versus 79/256 for its same-data group-two control, a paired SR difference
of -1.95 points. On the complete 1,839-episode val-unseen evaluation,
it succeeded on 520 versus 495 for control: paired SR +1.36 points and
SPL +1.44 points, with zero inference errors. The eleven-scene bootstrap
interval for SR is -2.13 to +4.66 points. The contrast between the
fixed-256 screen and full result is a reminder that small screens are
only filters. Seed-33 fixed-256 evaluation has 86/256 successes versus
73/256 for its control (paired SR +5.08 points); its complete evaluation
and the three-seed aggregate are pending. Exact manifest coverage, zero
inference errors, and paired multi-seed comparison are required before
using any result in the paper.

The single-observation SFT probe also differed from the actual trainer:
the trainer strips its system block and includes every user observation and
assistant action response in a multi-turn transcript. A small, fixed
`full_history_probe_manifest.json` therefore selected 27 development and
30 untouched audit pairs from 16 train scenes formerly assigned to the
previous probe's fit partition. `collect_full_history_probe.py` replayed
all 54 development trajectories at turn boundaries with zero distance
errors. `score_full_history_probe.py` used the trainer's text template and
image size limits, excluded terminal STOP answers, and scored the frozen
SFT checkpoint without a policy update. Successful endpoints outranked
failed endpoints in 18/27 development pairs (66.67%), versus 16/27
(59.26%) for the earlier single-observation prompt on those pairs.
That reference changes both history and system-message handling, so the
7.41-point difference is not a controlled history ablation. The predeclared
70% development gate failed; `run_full_history_probe.sh` skipped the
30-pair audit and reward training. The frozen selection and development
report are under `ordinal_progress/policy_preference/`.

The next low-cost screen reuses the 3,200 existing SFT feature vectors.
`label_policy_geodesic_progress.py` replayed the same 800 group-four
policy trajectories with exact coverage and zero errors, attaching
train-set geodesic distance to the four cached RGB checkpoints.
`fit_geodesic_potential.py` fits a linear
potential to the frozen image-instruction state difference from the start;
the supervision is the fraction of geodesic distance reduced. The fit,
development, and audit scenes stay disjoint. Development selects ridge
regularization, and audit labels are read only if development reaches
70% paired endpoint ranking and 60% temporal ordering. The audit screen
requires at least 75% paired ranking, a 5-point gain over the prior raw
STOP score, and 60% temporal ordering before any online reward test.
No simulator distance is available to the learned reward at inference.
The fit-scene endpoint ranking reached 89.29%, but the disjoint eight-scene
development ranking was only 56.06% across 66 pairs. Development temporal
ordering was 56.94% over 764 comparable frame pairs. Both predeclared
development criteria failed; the audit labels were not read for scoring,
and no online reward test was started. The frozen report is
`policy_preference/geodesic_potential_screen.json`. This large train-to-dev
gap makes a linear readout of the frozen representation a poor candidate;
the next method needs to train an instruction-grounded temporal
representation rather than only adjust its readout. A candidate passing
train-scene gates would proceed to a matched group-size-four RL pilot;
group-size-eight replication is reserved for a promising pilot. This
cascade overlapped low-memory replay and frozen feature analysis on GPU 0
with seed-22 evaluation on GPU 1 and seed-33 training on GPUs 2/3, while
avoiding redundant multimodal encoding.

`train_temporal_progress_encoder.py` then tested an actual temporal
representation change on the same cached four-frame group-four policy
trajectories. A one-layer causal transformer received only frozen
image-instruction SFT states; train-scene geodesic progress, same-episode
success/failure contrast, and temporal order supervised it. A fixed seed-11,
50-epoch run selected epoch 8 on development scenes. Development ranked
52/66 endpoint pairs correctly (78.79%) and ordered 78.01% of comparable
frame pairs. It passed the predeclared development gate, so the eight-scene
54-pair audit was read once: 42/54 (77.78%) endpoint ranking and 77.21%
temporal ordering, compared with 31/54 (57.41%) for the frozen raw STOP
readout. The audit scene bootstrap interval for ranking is 69.44--84.38%,
so this is a promising train-scene signal, not a demonstrated navigation
gain. `policy_preference/temporal_progress_screen.json` preserves the
epoch history and all gates. The cached SFT states still use the older
single-observation system prompt; a deployable online reward must be
matched to the policy input and must prove instruction grounding.

`prepare_temporal_instruction_swaps.py` froze 54 same-scene alternative
instructions whose goal positions are at least 4 m from the original
train goal. `run_temporal_swap_audit.sh` re-encoded the same success-path
RGB frames under those wrong instructions. The potential preferred the
correct instruction in 39/54 pairs (72.22%), with an eight-scene bootstrap
interval of 58.70--79.71%. Its average correct-minus-swapped progress was
positive, but the predeclared 75% grounding gate failed. Therefore no
online group-four RL was launched for this version. The result is in
`policy_preference/temporal_instruction_swap_audit.json`. A follow-up
representation should train with same-scene different-goal instruction
negatives, then use a fresh scene-held-out gate before online RL. The
online reward must also match the policy's image and prompt pipeline.

The next adapter adds same-scene different-goal instruction negatives to
the temporal progress loss. `prepare_temporal_contrastive_v2.py` fixed a
fresh 300/52/48 train-pair fit/development/audit split: its new held-out
scenes were taken from old fit scenes outside the full-history probe;
previously inspected scenes are fit-only. All 400 wrong instructions were
selected by a deterministic hash and a train-goal separation of at least
4 m. `cache_temporal_contrastive_swaps.py` reused existing policy RGB and
computed only fit/development wrong-instruction SFT features before model
selection. With seed 11 and the fixed 50-epoch budget, epoch 4 met the
new development gates: 37/52 (71.15%) successful-over-failed endpoint
ranking, 72.13% temporal ordering, and 39/52 (75.0%) correct-instruction
preference. `policy_preference/temporal_contrastive_v2_development.json`
records the complete selection history. The frozen 48-pair audit was
read once after those gates passed. It reached 38/48 (79.17%) endpoint
ranking and 78.73% temporal ordering, but only 31/48 (64.58%) correct
instruction preference. The predeclared 75% grounding gate failed on
this fresh audit, so online group-four RL was not launched. The complete
report is `policy_preference/temporal_contrastive_v2_audit.json`. This
version is better at detecting route progress than identifying which
destination the instruction specifies; another reward update based on
it would risk reinforcing the wrong route.

To avoid repeatedly judging new readouts on the same 400 pairs,
`prepare_seed33_novel_pairs.py` froze 38 additional success/failure pairs
from the completed seed-33 group-four train rollout whose episode IDs do
not overlap those 400 pairs. They cover 32 train scenes; scene overlap
with earlier fitting remains possible, so this is an independent-episode
diagnostic rather than val-unseen. `run_seed33_novel_collection.sh` replayed
all 76 trajectories, cached 304 SFT image-instruction states, and attached
four-point simulator distances with zero collection errors. Same-scene
wrong-goal instructions were frozen before either encoder was tested.
On these new pairs, the original temporal encoder ranked success over
failure in 23/38 (60.53%) and preferred the correct instruction in 30/38
(78.95%); the instruction-contrastive encoder scored 26/38 (68.42%) and
29/38 (76.32%). Both missed the 75% endpoint-ranking screen despite
roughly 77--79% temporal ordering. The complete episode-disjoint report is
`policy_preference/seed33_novel_representation_check.json`; no reward RL
was launched.

`probe_explicit_goal_alignment.py` also tested whether cached frozen
SigLIP features could provide cheap explicit goal conditioning on the
v2 development pairs. The terminal policy image preferred its correct
instruction over a same-scene different-goal instruction in only 14/52
(26.92%) pairs; endpoint-minus-start similarity improved to 24/52
(46.15%). The report is `policy_preference/explicit_goal_alignment_preflight.json`.
This rules out a simple frozen similarity add-on. The next representation
experiment needs broader instruction-goal supervision and backbone
adaptation, with the seed-33 episode-disjoint set held out from fitting.

`train_siglip_lora_goal_policy.py` then adapted both SigLIP towers with
589,824 LoRA parameters using the train-only goal panoramas and the same
group-four success/failure pairs. Its three losses compare same-scene
different-goal images, successful versus failed endpoints, and correct
versus wrong instructions; a fourth term encourages progress from the
start to the successful endpoint. The v2 scene split is reused. The 38
seed-33 episode-disjoint pairs and all v2 audit scenes are excluded from
fitting and selection. Four fixed goal pairs per development scene give
32 pairs (64 image-to-instruction comparisons), alongside 52 policy
pairs. Training used idle GPU1 while the seed-33 complete navigation
evaluation continued on GPU3/GPU2.

The first implementation used dynamic text padding. This is invalid for
deployment: Transformers 4.51.3 SigLIP pools the last text position, so
the same instruction changed representation with the other members of
its batch. Its four JSON reports are retained with `_batchpadding_invalid`
in their filenames and must not be used as evidence. The corrected
training, offline evaluations, and online service pad every instruction
to 64 tokens. The corrected backbone starts at 65.63% goal matching,
61.54% successful endpoint ranking, and 46.15% correct-instruction
preference on the development subset. At 256 steps these are 76.56%,
67.31%, and 71.15%; at the selected 768-step checkpoint they are
78.13%, 69.23%, and 75.0%. The adapter alone misses the 70% policy
ranking gate. Its full history is
`policy_preference/siglip_lora_goal_policy_fixed64_768_development.json`.

`probe_fused_reward.py` tested a representation-level alternative: the
causal temporal potential from the frozen navigation SFT states and the
adapted visual/text similarity each produce endpoint and instruction
margins. Each margin is divided by its mean absolute fit-scene margin,
then the two normalized terms receive a fixed 1:1 weight. Neither the
weight nor the scales use development labels. The final probe encodes
one image and one fixed-length instruction per forward call, matching
the intended online service. On the 52 development pairs, the fused
reward ranks 43/52 successful endpoints (82.69%) and prefers the
correct instruction in 39/52 (75.0%). The temporal term alone scores
37/52 and 39/52; the visual term alone scores 36/52 and 39/52. Exact
fit/development measurements and calibration scales are in
`policy_preference/equal_fused_reward_online_parity_development.json`.

After freezing this rule, `check_fused_reward_heldout.py` evaluated the
previously used v2 scene audit and the 38 seed-33 episode-disjoint pairs.
The equal fusion scores 41/48 endpoint and 37/48 instruction comparisons
on the reused v2 audit, and 30/38 and 34/38 on the seed-33 pairs. The
corresponding temporal-only counts were 38/48 and 31/48, and 26/38 and
29/38. All four fused proportions reached the predeclared 75% screen;
scene-bootstrap intervals and every pair's normalized margins are in
`policy_preference/equal_fused_reward_online_parity_heldout.json`. The v2 audit had
already been inspected for the temporal component, and seed-33 shares
train scenes with fitting despite episode disjointness. These are
promising offline reward diagnostics, not val-unseen navigation gains.
The next test must use group size four, a matched outcome-only control,
and a small, fixed training budget before any scale-up.

The separate outcome-only optimizer scale study has now completed three
matched seeds at 128 steps, with group size four against a two-sample
same-data control. `optimizer_scale_full/` contains the six compact
paired val-unseen records, 256- and 1,839-episode manifests, train audits,
and checksums. `verify_optimizer_scaled_package.py` independently
confirmed 1,839 unique episodes per model and zero inference errors.
Full-val paired SR gains are +0.65, +1.36, and +4.30 percentage points;
the three-seed mean is +2.10 (sample SD 1.93). SPL gains are +0.96,
+1.44, and +5.12 points, with mean +2.51 (sample SD 2.27). On the
1,583 episodes outside the original fixed screen, mean changes are
+1.94 SR and +2.40 SPL points. This is evidence about optimizer group
size at doubled simulator-rollout cost, not evidence that the fused
representation improves navigation. Future online reward arms should
both use group size four.

## Group-four reward pilots and turn-wise credit (2026-10-04)

The original Qwen3-VL-8B group-rank reward pilot completed 64 training
steps with `rollout.n=4`. On its matched 256-episode val-unseen screen,
the candidate succeeded on 63/256 (24.61% SR, 24.42% SPL), versus
68/256 (26.56% SR, 26.25% SPL) for the same-data outcome-only control.
The paired changes were -1.95 SR and -1.83 SPL percentage points, so
this candidate did not meet the positive pilot gate. Exact paired
records are in `qwen_group4/`. The confidence-gated pair variant has
completed a separate 64-step, group-four seed-11 training run and
independent audit of all 256 episode groups, 828 teacher requests, and
64 nonzero-gradient steps. Its matched fifth 256-episode evaluation
completed with 80/256 successes (31.25% SR, 30.78% SPL), compared with
88/256 (34.38% SR, 33.92% SPL) for the same-data control. Paired
changes were -3.13 SR and -3.14 SPL percentage points; both arms had
exact coverage and zero inference errors. This variant also failed
its positive pilot gate; no three-seed scale was launched. The
original Qwen full-val sensitivity recheck completed using existing
checkpoints. Both arms covered the same 1,839 episodes with zero
inference errors: the candidate reached 471 successes versus 562 for
its control, paired SR -4.95 and SPL -4.72 points. The 1,583 episodes
outside the reused screen also favored control (-4.93 SR and -4.69
SPL points). This post hoc one-seed development recheck supports the
negative finding; it is not an independent test or scale-up gate.

The next mechanism diagnostic asks whether action-aligned process
credit can help when the process label is accurate. It uses simulator
geodesic distance **only during training** as a privileged upper bound;
the navigation policy never receives that distance as an observation.
The isolated implementation and source checks are in
`prepare_turnwise_oracle_tree.sh`, `oracle_turnwise_v1.patch`, and
`oracle_turnwise_group4_advantage.py`. A synthetic adapter smoke checks
that equal episode totals can assign different second-turn advantages,
observation tokens receive zero, STOP gets no auxiliary credit in
all-failure groups, and inconsistent reward/token alignment raises an
error. The STOP mask follows the parsed generated action, including a
STOP produced after the per-turn execution budget is exhausted. This
case has `extracted_actions=["stop"]` but no executed STOP; the adapter
and independent audit now require zero process credit and zero STOP
token advantage there too. The separate
`audit_oracle_turnwise_train.py` recomputes each turn's distance delta
from the recorded simulator trace and checks exact four-rollout
train-row pairing and gradients. The isolated remote CPU checks passed.
A real two-step online wiring test passed after the full recheck
released its GPUs. Its independent audit checked eight matched
four-rollout groups, 370 turns, 303 nonzero progress turns, 14 generated
STOP turns, and nonzero actor gradients at both steps
(`ordinal_progress/policy_preference/oracle_turnwise_2step_audit.json`).
The conditional 64-step group-four pilot finished training and passed
its independent audit: 256 exact four-rollout groups, 11,450 turns,
9,364 nonzero progress turns, all 141 all-failure groups with active
turnwise contrast, and nonzero actor gradients on all 64 steps
(`ordinal_progress/policy_preference/oracle_turnwise_64step_audit.json`).
Its matched fourth-manifest 256-episode val-unseen candidate/control
evaluation finished with exact coverage and zero inference errors:
81/256 versus 72/256 successes, paired SR +3.52 and SPL +3.76
percentage points. The nine-scene exploratory bootstrap intervals
are [-2.73, 9.13] SR and [-2.58, 9.28] SPL points, both including zero.
The frozen positive pilot gate passed, but this is one seed and an
unavailable-at-deployment simulator-distance upper bound, not a
validated semantic reward. The episode-level pair and analysis are
`ordinal_progress/policy_preference/paired_oracle_episodes.jsonl` and
`paired_oracle_vs_control.json`.
Because that gate passed, `run_oracle_full_recheck_after_screen.sh`
started a one-seed, same-checkpoint sensitivity recheck on all 1,839
val-unseen episodes. It overlaps the oracle candidate's GPU-3/GPU-2
evaluation with the observation-only LoRA fit on GPU 1, then runs the
same-data outcome control on GPU 1/GPU 0 after that fit releases its
GPU. `analyze_oracle_full_recheck.py` will verify exact paired coverage
and report the 1,583 episodes outside the reused 256-item screen
separately. This post-screen full recheck remains development evidence;
it is not three-seed confirmation.
`NEXT_PROCESS_REWARD_PROTOCOL.md` gives
the gates, resource schedule, and limitations. The single-screen gain
requires larger-sample and multi-seed verification.
Two conditional watchers sequence the diagnostic: the first runs and
audits the two-step wiring smoke after GPU release;
only a passing smoke lets the second run the 64-step group-four oracle
upper-bound pilot and paired evaluation on the frozen fourth 256-item
screen. Neither watcher changes a live reward-training process.

An additional STOP-specific mechanism has been prepared while the
oracle pilot uses GPUs. Its train-only n=4 preflight finds 105 eligible
failed-STOP/closer-continuation pairs in 51 all-failure outcome-control
groups. The rule credits only continuations at least 1 m closer while
still outside the success radius; successful groups retain outcome
reward. Source, test, isolated-tree preparation, and both control/Qwen
preflight reports are in `stop_pair_group4_reward.py`,
`test_stop_pair_group4_reward.py`, `prepare_stop_pair_tree.sh`, and
`ordinal_progress/policy_preference/`. Its conditional watcher skipped
the pilot because the oracle n=4 screen passed its positive gate. No
STOP-pair policy training or navigation result exists. Pair endpoints are
correlational, and the rule uses privileged simulator distance during
training only.

The proposed observation-only STOP representation needs successful
policy histories paired with a natural wrong instruction at the exact
same start. A CPU-only coverage audit of the three existing n=4
training rollouts checked the frozen scene split, source hashes, and
all 6,144 rollout records. Requiring the alternative goal to be at
least 6.5 m from the original goal guarantees that a successful STOP
within 3 m remains at least 3.5 m from the alternative goal. The
selected policy-history cache has only 29 such fit records from 21
unique episodes and 14 development records from eight episodes. Even
all eligible source rollouts yield only 179 fit records from 46 unique
episodes and 43 development records from 13 episodes; repeated
rollouts must not be treated as independent examples. At a looser
3.5 m goal gap, all-source coverage rises to 76 fit and 20
development episodes, but correctness would require endpoint replay.
These counts are too sparse for a costly new STOP LoRA pilot with a
credible scene-disjoint screen, so no model fit was launched. The
script and checked report are `preflight_policy_stop_swaps.py` and
`ordinal_progress/policy_preference/policy_stop_swap_coverage.json`.

A different observation-only representation screen targets the failure
seen in the earlier STOP-only LoRA audit: far-goal policy histories had
20.11% false STOP predictions and wrong expert instructions had 26.83%.
`train_policy_stop_hardneg_lora.py` keeps expert correct/wrong
instruction contrast, explicitly balances near-goal and far-goal
policy terminal histories by scene and episode, and feeds the same
multi-turn prompt shape as the online policy (without its stripped
system block). The original `history_grounding_lora.py` prompt remains
the default for reproducing its earlier result. This new method uses
only fit-scene gradients; the predeclared development gate requires
STOP AUROC at least .80, instruction-swap accuracy at least .80,
pooled FPR at most .05 with recall at least .55, wrong-instruction FPR
at most .12, far-policy FPR at most .08, and near-policy recall at
least .50. An isolated GPU-1 four-microstep wiring smoke completed one
optimizer update with finite loss and 1,843,200 trainable LoRA
parameters, using 627 expert and 768 policy fit records. Its source
and log are `run_policy_stop_hardneg_lora.sh` and
`ordinal_progress/policy_preference/policy_stop_hardneg_smoke.log`.
An explicit processor parity audit first caught an automatically
inserted system block. After matching the trainer's block removal and
PNG image path, a fit policy record at 0, 1, and 3 history turns had
identical token IDs, attention masks, image grids, and pixel values;
the corrected four-step smoke then passed. The check is
`audit_policy_stop_prompt_parity.py`, with results in
`ordinal_progress/policy_preference/policy_stop_prompt_parity.json`.
The no-GPU watcher (`run_policy_stop_hardneg_after_oracle.sh`) completed
the 512-step STOP representation fit on GPU 1 after the oracle's paired
evaluation and prompt-parity check. The selected step-512 checkpoint
reached development STOP AUC .921 and instruction-swap accuracy .919.
At the predeclared 5% pooled-FPR target, actual pooled FPR is 4.33%,
wrong-instruction FPR 7.45%, and far-policy FPR 5.29%. Pooled recall is
50.95% (required 55%) and near-policy recall 41.58% (required 50%),
so the full development gate failed. The report and fit log are
`ordinal_progress/policy_preference/policy_stop_hardneg_development.json`
and `policy_stop_hardneg_train.log`. The staged
`audit_policy_stop_hardneg_lora.py` must not read its audit split, and
this checkpoint will not enter group-four online RL. The old
STOP-only audit scenes have already been inspected in method
development, so any reuse of them is exploratory, not an independent
accuracy claim.
