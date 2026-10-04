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
completed a one-seed, same-checkpoint sensitivity recheck on all 1,839
val-unseen episodes. Both arms have exact unique-ID coverage and zero
inference errors. The privileged oracle succeeds on **602/1,839**
(32.74% SR, 32.26% SPL), compared with **545/1,839** (29.64% SR,
29.21% SPL) for its same-data n=4 outcome-only control: paired SR
**+3.10** and SPL **+3.05** percentage points. On the 1,583 episodes
outside the reused 256-item screen, the respective success counts
are 521 and 481, paired SR **+2.53** and SPL **+2.50** points. The
compact episode-level export was independently recomputed locally
against both analysis reports and the frozen screen IDs. Source files
are `paired_oracle_full.json`, `paired_oracle_full_episodes.jsonl`, and
`oracle_full_recheck_analysis.json` under
`ordinal_progress/policy_preference/`. This is still post-screen,
one-seed development evidence using training-only simulator distance;
it is not three-seed confirmation or a deployable learned reward.
An exploratory paired failure-mode recount found 220 episodes rescued
by the oracle checkpoint and 163 lost. It hit the 12-turn cap on
635 episodes versus 520 for the control, including 304 candidate-only
versus 189 control-only turn-cap cases. On the 1,583 episodes outside
the reused screen, the corresponding counts are 186 rescued, 146
lost, and 551 versus 449 turn-cap cases. This suggests a possible
termination-behavior difference; it does not show that delayed STOP
caused the navigation gain. The reproducible recount is
`analyze_oracle_full_failure_modes.py` and
`ordinal_progress/policy_preference/oracle_full_failure_modes.json`.
If the three-seed scale confirms a gain, a matched n=4 STOP-only
diagnostic should test whether dense turn-wise credit adds value
beyond changed termination behavior.
Both full and screen-complement SR/SPL gates passed, so the staged
`run_oracle_exact512_scale_after_full.sh` launched matched n=4,
128-step, three-seed oracle/control training on 512 unique train rows
after the independent representation fit released the GPUs. All three
outcome-only controls completed training; seed 33's full validation is
in progress while the oracle seed-11 candidate trains. The
control full-val labels for seeds 11 and 22 passed exact 1,839-ID
coverage with zero inference errors, recording 450 and 545 successes
respectively. These are control-only results, not paired gains. There
are no scaled paired navigation results yet. Its sources and audits
are `run_oracle_exact512_train.sh`,
`audit_oracle_exact512_scale.py`, and
`analyze_oracle_exact512_scale.py`. `NEXT_PROCESS_REWARD_PROTOCOL.md`
records the resource schedule and limits on interpretation.
`run_oracle_control_eval_overlap.sh` waits until the GPU-1 representation
fit releases memory, then evaluates each already-trained n=4 control on
the frozen 1,839-episode manifest while the remaining training uses
GPUs 2/3. The ordinary paired suite skips a validated completed label.
`run_direction_eval.sh` takes a shared per-inference-GPU lock across the
overlap watcher and paired suite, preventing two model servers from
starting on one GPU. Later labels disable vLLM's verbose per-request
prompt logging to reduce disk writes; per-episode simulator records,
worker errors, and the exact-coverage validator remain in place. The
overlap cannot select a candidate from
control-only metrics; training and full evaluation retain their frozen
seeds, manifest, and exact-coverage checks. GPU-0 simulator and CPU
contention will be monitored before allowing sustained overlap.
`check_exact512_eval_progress.py` counts only the per-episode
`stats_*_0.json` files, checks their manifest IDs and shard placement,
and accepts a completed label only with all 1,839 unique episodes and
a zero-error validator. The evaluator also writes other JSON files,
so counting every `*.json` overstates progress. The progress script
confirmed seed-11/22 control coverage and validator successes above;
an in-progress seed-33 count remains provisional.
`run_oracle_candidate_eval_overlap.sh` waits for the control lane, then
evaluates each audited candidate checkpoint as soon as its seed finishes,
again using GPU 1 while the next n=4 seed trains on GPUs 2/3. It shares
the label locks and result root with the final paired suite, so an
already validated label is skipped there and every reported pair is
still analyzed only after both arms finish.
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
On the newer exact512 n=4 outcome-control seed-11 rollout, a separate
CPU-only preflight checked all 128 steps, 512 unique training episode
groups, and 2,048 trajectories against the frozen source and reward
helper hashes. The same STOP-pair rule activates in 118 of 290
all-failure groups, with 235 eligible pairs and 346 trajectories
receiving a nonzero pair vote. This establishes train-signal support
for a possible matched diagnostic; it does not measure benefit from
delaying STOP or justify a semantic-reward claim. The reproducible
script and compact report are `preflight_stop_pair_exact512.py` and
`ordinal_progress/policy_preference/stop_pair_exact512_preflight.json`.

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

The next observation-only process-reward preflight reads only the
64-step oracle *training* rollouts. It recomputes raw geodesic change
in meters from before/after distances rather than treating the
normalized oracle reward as a meter value. The corrected report finds
7,454 executed action turns from 180 unique fit episodes in 38 scenes:
3,545 turns advanced at least 0.25 m and 1,351 regressed at least
0.25 m. The scene-disjoint development and previously used audit partitions
contain 39 and 37 unique episodes. Reward and raw-distance signs
agreed on all 10,429 action turns across the three partitions. These
are correlated supervision candidates, not learned-reward or
navigation results. Source and report are
`preflight_oracle_turn_labels.py` and
`ordinal_progress/policy_preference/oracle_turn_label_preflight.json`.
For the next reward model, `freeze_process_reward_scene_audit.py` froze
an ID-only prospective check containing all 123 episodes in the seven
R2R-train scenes absent from that 54-scene split; no labels or model
scores were read to select it. Its manifest is
`ordinal_progress/policy_preference/process_reward_prospective_scene_audit.json`.
These remain train scenes and cannot replace val-unseen navigation tests.
The already verified `policy_process_turns` cache contains more reusable
training data than the oracle pilot: 768 fit trajectories from 314
episodes and 7,981 motion turns, plus 320 development trajectories and
3,310 turns. The next representation will reuse these RGB histories;
only its prospective seven-scene check needs new collection. The older
frozen-SFT pairwise head did not meet its direction-balance development
gate, so reusing the frames does not imply reusing that failed head.
`NEXT_PROCESS_REWARD_PROTOCOL.md` fixes n=4 for the primary paired
candidate/control comparison and limits n=8 to a later, matched,
small diagnostic after an algorithmic n=4 gain.

`train_policy_progress_lora.py` is the first new offline representation
screen. It fits the navigation SFT's small Qwen2.5-VL LoRA with the
policy's verified system-stripped prompt and PNG processor path. Fit
examples are scene/episode-balanced across true >=1 m progress, >=1 m
regression, and <0.1 m stationary turns; natural same-start wrong-goal
expert instructions provide a separate grounding loss. Simulator
distance is used only for train labels and development metrics, never
as a model input. Its development gate checks balanced and per-class
local direction, instruction-conditioned progress gain, and forward
recall at a stationary false-positive threshold. No old audit split or
val-unseen outcomes enter this script. CPU-only source validation found
934/307/1,500 fit forward/regression/stationary turn examples and
397/161/654 on development, with 596 safe fit instruction swaps.
The source and run protocol are `train_policy_progress_lora.py`,
`run_policy_progress_lora.sh`, and
`run_policy_progress_after_oracle_candidate.sh`. The oracle candidate
full evaluation completed exact 1,839-episode coverage with zero
inference errors and released GPU 3. The four-microstep wiring smoke
then passed one nonzero-gradient update with 1,843,200 trainable LoRA
parameters; the bounded 256-microstep representation fit completed on
GPU 3 while the oracle control evaluation continued on GPUs 1/0.
The selected step-256 checkpoint **failed** its fixed small-development
gate: forward local-direction accuracy 66.39%, regression accuracy
31.37%, balanced accuracy 48.88%, correct-versus-wrong instruction
gain preference 66.67%, and forward recall 15.97% at a threshold with
9.95% stationary false positives. The complete development set and
prospective audit were not scored; no online n=4 RL will use this
checkpoint. The report and fit log are
`ordinal_progress/policy_preference/policy_progress_lora_development.json`
and `policy_progress_lora_train.log`. The failure suggests that an
absolute history potential still poorly recognizes genuine regression;
the next representation should jointly compare before/after views
and test order reversal rather than infer local change by subtracting
two independent scalar predictions. That remains a hypothesis, not an
experiment result.

An ID-only audit-reuse preflight checked the three earlier n=4 training
rollouts against the frozen seven-scene/123-episode reward-model check:
only eight unique audit episodes occur there, despite 96 repeated
trajectories. Thus 115 episode IDs would have required fresh policy
rollout had this checkpoint passed development. No such audit rollout
was started. `preflight_process_reward_audit_reuse.py` and
`ordinal_progress/policy_preference/process_reward_audit_source_reuse.json`
record the exact IDs and source hashes; no audit labels or model scores
were read.

The second representation changed the model input rather than tuning the
failed scalar potential. `train_joint_pair_progress_lora.py` jointly
encodes the instruction with before/after RGB views and trains an
antisymmetric signed comparison: reversing image order should reverse
the progress score. It uses the same true-forward, true-regression,
stationary, and safe wrong-goal fit examples, while keeping the old
audit and val-unseen inaccessible. A CPU prompt check verified exactly
two images, finite pixel tensors, identical text tokens under reversal,
and changed image order; a four-microstep GPU-1 smoke passed one
nonzero-gradient update. The bounded 256-microstep fit completed on
otherwise idle GPU 1 concurrently with the n=4 scale training on GPU
0/2/3. The step-128 checkpoint was selected by the fixed small-development
rule, but **failed five of six gates**: 141 forward and 42 regression
pairs had 47.52%/59.52% sign accuracy (53.52% balanced), the correct
instruction beat a safe wrong-goal instruction on 25/48 routes
(52.08%), and forward recall was 0% at a threshold yielding 9.88%
stationary false positives. Step 256 also failed (49.70% balanced,
62.50% instruction preference, 7.09% forward recall). The full
development set, prospective audit, and online RL were not scored or
run. The source and launcher are `train_joint_pair_progress_lora.py`
and `run_joint_pair_progress_lora.sh`; the immutable report and log are
`ordinal_progress/policy_preference/joint_pair_progress_lora_development.json`
and `joint_pair_progress_lora_train.log`.

Both independent-state and joint two-view scorers fail even the cheap
development check. A previous frozen ordered-clause/region probe also
failed its calibration gate, while the earlier evidence-onset LoRA
reached only 37/94 all-four crossed-instruction comparisons on reused
development scenes against a 75% gate. Repeating a simple landmark
score would not address those failures.

`train_action_memory_progress_lora.py` instead conditions a signed
local-change score on the start, before, and after RGB views, the
executed action, and the natural instruction. Its fifth training
objective directly ranks a forward turn above a regression turn from
the same trajectory. The cached fit data have 60 such trajectories
from 49 episodes in 28 scenes; development has 20 trajectories from
17 episodes in seven scenes. Those examples are correlated, so this
is a bounded probe, not 60 independent route trials. Train-only
geodesic deltas define supervision, never an input. The fixed
development gates and prospective-audit policy remain unchanged.
The three-image CPU prompt check passed (369 text tokens, finite
image tensors), as did a five-microstep GPU-1 smoke with one nonzero
gradient update and 1,843,200 trainable LoRA parameters. Its fixed
1,000-microstep fit completed on GPU 1 concurrently with the n=4
privileged-oracle/control scale on GPU 0/2/3. The selected step-1,000
checkpoint **failed** its small-development gate: forward accuracy
88.79%, regression 22.22%, balanced 55.51%, correct-versus-wrong
instruction preference 66.67%, and forward recall 62.07% at 9.09%
stationary false positives. The 250/500/750 checkpoints also failed;
the highest balanced score among them was 57.81% at step 750.
The full development set, prospective audit, online n=4 RL, and
val-unseen were not scored. The signed action-memory head still
largely mistakes regression for progress. Source and runner are
`train_action_memory_progress_lora.py` and
`run_action_memory_progress_lora.sh`; the immutable report and fit
log are `ordinal_progress/policy_preference/action_memory_progress_lora_development.json`
and `action_memory_progress_lora_train.log`.

`probe_action_only_progress.py` fixes a cheap action-text-only control
on the same fit/development policy records. A class-balanced ridge
readout of executed motion counts reaches only 50.14% balanced
forward/regression accuracy on the action-memory model's fixed
96-trajectory small development subset (52.07% on all 320), and its
forward recall is 0.86% at 9.09% stationary false positives on that
small subset. This demonstrates that these geodesic labels are not
trivially recovered from the action string alone; it does not prove
that the three-view model uses the images. The fixed report is
`ordinal_progress/policy_preference/action_only_progress_baseline.json`.

The original fit inventory has 4,456 eligible trajectories but only
314 unique episode IDs among the 768 replayed records. A blind
2,048-trajectory round-robin expansion covered only 366 episodes.
`prepare_policy_process_manifest.py --fit-target 1024 --diversity-extra`
instead froze the same original 768 fit records plus 256 distinct
failed-route episode variants, including 58 previously unseen fit
episode IDs. The development/audit identities and three rollout
source hashes are unchanged; the new manifest is
`ordinal_progress/policy_preference/diversity_fit1024_manifest.json`
(SHA-256 `bd27cac517c7909f43bca210ad8eee62456f876c56af32968e285261c58fbc37`).
`run_diversity_policy_fit_replay.sh` reused the old 768 frame records
and replayed only the 256 new train-fit trajectories on GPU 1.
`audit_diversity_policy_fit.py` independently checked exact records,
images, terminal distances and per-turn geodesic labels. The extra
trajectories have 129 one-meter regression turns from 77 episode IDs,
247 forward turns, 582 stationary turns, and 25 trajectories with
both directions; the predeclared extra-data sample gate passed.
Combined fit has 1,024 trajectories, 372 episode IDs, 436 regression
turns, 1,181 forward turns and 10,709 motion turns. A separate
recomputation of the extra records reproduced the regression and
episode counts. The compact evidence is in
`ordinal_progress/policy_preference/diversity_fit1024_audit.json` and
`diversity_fit1024_collection_summary.json`. No development/audit
labels or val-unseen images were opened by this collection. This is
data coverage, not a learned reward or navigation result.

The in-progress n=4 exact512 outcome control can supply new fit-only
policy trajectories without additional model rollout. An ID-only
manifest, `ordinal_progress/policy_preference/control_exact512_fit_extension_ids.json`,
now freezes 256 episodes across all 38 fit scenes; none overlaps any
old fit, development, or audit episode. After the complete seed-11
control rollout is audited, a predeclared SHA/uniform plus
regression-enriched variant rule can reuse two of its four trajectories
per selected episode. The complete seed-11 control source yielded
1,021 replayable variants among the 1,024 generated: three
action-format failures were excluded before geodesic replay, and all
256 frozen episode IDs retained at least three valid variants. The
regression choice first requires a label-only Habitat replay of the
1,021 replayable variants; rendering
the 512 chosen trajectories is conditional on a fit-only gate of at
least 100 one-meter regressions from 50 episode IDs. Both replay
costs will be reported. No rollout labels or visual records from this
source have entered a representation fit yet; this is a collection
plan only.
`prepare_control_fit_all_variants.py`, `collect_control_fit_labels.py`,
`select_control_fit_render.py`, and `audit_control_fit_render.py`
implement the two-pass audit. The detached
`run_control_fit_extension_after_seed11.sh` watcher waits for the
complete seed-11 control marker, runs a four-trajectory smoke, and
only starts the full label/RGB replay when each prior check passes.
The source eligibility and Habitat reset-order bugs were repaired;
the four-trajectory no-RGB smoke passed before the full replay.

The fit-only extension subsequently completed and passed its frozen
sample gate. The label-only pass replayed 1,021 valid variants from
256 new R2R-train episode IDs in 383.96 seconds, finding 397 one-meter
regression turns across 139 IDs. The selected 512 trajectories include
316 such turns across the same 139 IDs. RGB replay took 355 seconds,
produced 5,826 images, and exactly matched all selected per-turn
geodesic distances (maximum absolute drift 0 m). No new policy
inference was required. The copied `control_fit_extension/` package
under `ordinal_progress/policy_preference/` contains the label-only,
selection, RGB, and audit summaries; the audit SHA-256 is
`047d0711ec4112f18eea6247538bbd2e914a248e5e2ecb8af4f23aa0c8385a45`.
These are fit-data coverage and simulator-cost results, not learned
reward or navigation results.

To test a state-potential representation without noisy always-wrong
instruction swaps, the train-only crossed-goal preflight selected
same-scene alternate goals at least 3 m away and checked start-state
navmesh reachability before reading motion labels. Five frozen fit IDs
had no distinct alternate and one more had no reachable alternate;
250 IDs retained two rendered variants each. The full label-only
replay audited 500/500 trajectories in 273.55 s after 181.10 s of
reachability checking. It yielded 929 turns from 172 IDs where one
goal improves by at least 0.5 m and the other regresses by at least
0.5 m (685/244 by direction), passing the prior 100-turn/50-ID gate.
Correct-goal distance drift against the independent RGB replay was
0 m. The compact package is `ordinal_progress/policy_preference/control_fit_extension/cross_goal/`.
These labels are fit-only supervision; the candidate representation,
prospective audit, n=4 policy comparison, and val-unseen gain remain
unverified.

The next fit, `train_cross_goal_potential_lora.py`, uses these
geometry-verified contrasts to train one observation-only potential
rather than another categorical local-change head. Its CPU source
preflight validated 1,536 fit policy trajectories, 500 crossed-goal
trajectories, 369/126 early contrast turns by direction, and 178
same-start ranking pairs. The six-microstep GPU-1 smoke completed one
nonzero-gradient update with 1,843,200 LoRA parameters. The full fit
completed 1,500 microsteps; all four fixed small-development checks
failed. The selected step 1,000 reaches 49.44% balanced forward/backward
accuracy, 45.83% correct-instruction preference, and 0.84% forward
recall at 9.95% stationary false positives. Its full development set,
prospective audit, online n=4 RL, and val-unseen remain closed. Logs,
development report, and
source hashes are in `ordinal_progress/policy_preference/cross_goal_potential/`.

A separately staged `train_unbounded_cross_goal_potential_lora.py`
keeps the same audited training data, six losses, checkpoint schedule,
and scene-disjoint development gates, changing only the state-potential
head from a final `tanh` to a raw scalar. The fit-label audit found
three of 512 newly rendered trajectories with a contiguous supervised
target range above 2, which cannot be matched exactly by a potential
restricted to [-1, 1]. If unlabeled intermediate motion is instead
assumed to leave potential unchanged, the count is 12; that assumption
is not a hard constraint, so three is the relevant conservative count.
This diagnostic identifies one representational limitation, not the
cause of the current model's navigation performance. Its CPU source
preflight and six-microstep GPU-0 gradient smoke passed with the same
1,536 fit trajectories and 500 crossed-goal trajectories. A separate
full fit is running on GPU 0 while n=4 RL uses GPUs 2/3. The Habitat
service remained healthy and the control training advanced during the
smoke. Its full 1,500-microstep fit completed, but all four fixed
small-development checks failed. The selected step 1,000 reaches
53.22% balanced direction accuracy, 41.67% correct-instruction
preference, and 2.52% forward recall at 9.95% stationary false
positives. Full development, prospective audit, online n=4 RL, and
val-unseen remain closed. The compact development report and source
hash are archived alongside the preflight evidence. It has no
navigation result.
Scripts and compact preflight evidence are in
`ordinal_progress/policy_preference/cross_goal_potential/unbounded/`.
`train_unbounded_expert_cross_goal_potential_lora.py` is a further
preflighted variant that adds one training task: correct-versus-wrong
instruction ranking on 596 safe expert routes in the existing fit
scenes. It keeps the same observation-only input, unbounded head,
development split, and gates. CPU source preflight and a six-microstep
GPU-0 gradient smoke passed. The full fit started on GPU 0 after the
unbounded predecessor failed its frozen gate and released its process;
all four fixed small-development checks failed. The selected
step-1,500 checkpoint reached 81.25% correct-versus-wrong instruction
preference, but only 52.80% balanced local direction accuracy and
25.21% forward recall at 9.95% stationary false positives. Full
development, prospective audit, online n=4 RL, and val-unseen remain
closed. The exact report, log, and source hashes are under
`ordinal_progress/policy_preference/cross_goal_potential/unbounded_expert/`.
`run_unbounded_expert_after_unbounded.sh` is a conditional GPU-0
watcher: it starts this full fit only if the simpler unbounded head
completes and fails its frozen development gate, after the predecessor
process releases the GPU. It records a skip if that gate passes. The
watcher completed on the remote host; it does not change n=4 training
or the full-val evaluation queue.
`run_unbounded_expert_fit_diagnostic.sh` reloaded the selected
checkpoint and scored a fixed small subset of the training scenes.
Fit/development balanced direction accuracy was 62.23%/52.80%, but
backward accuracy was only 38.10%/39.22%. The model therefore missed
regression even on fit examples; a fit/development gap alone does not
explain the failure. Fit/development instruction preference was
91.67%/81.25%. The report, log, and source hash are archived in the
same `unbounded_expert/` directory. This diagnostic does not select
a new checkpoint or reopen audit and RL gates.

A CPU-only preflight for a different representation target found
same-start route pairs with at least 1 m geodesic separation at turns
3 and 6: 677/1,011 fit pairs from 258/354 episode IDs, and 361/505
development pairs from 50/58 IDs. The fit and development scenes are
disjoint. A forward-motion-only baseline already reaches 68.28% and
67.51% episode-macro pairwise accuracy on development at those turns;
any learned visual-instruction critic must exceed it. The fit cache
also has 210/310 hard pairs at turns 3/6 where more commanded forward
motion actually ends farther from the goal, from 98/156 episode IDs.
This establishes supervision coverage, not a fitted model or reward
result. See
`preflight_same_start_pairwise.py`, the compact
`ordinal_progress/policy_preference/same_start_pairwise_preflight.json`,
and the frozen gates in `NEXT_PROCESS_REWARD_PROTOCOL.md`.
`train_same_start_relative_lora.py` implements the next fit-only
hypothesis: start from the selected expert-grounded checkpoint and
optimize same-start pairwise orderings, deliberately sampling the
action-length counterexamples. Its CPU preflight verified the exact
677/1,011 fit and 361/505 development pairs, including 210/310 and
126/150 hard pairs. Its six-microstep GPU-0 smoke passed one finite,
nonzero-gradient update. The fixed 1,500-step fit completed on GPU 0.
The selected step-1,500 checkpoint reached 69.88%/66.94%
episode-macro ordering accuracy at turns 3/6, below the prespecified
75% at both anchors and without the required five-point margin over
the action-only baseline. Its hard-pair checks passed, and instruction
preference was 91.67%; the complete gate nevertheless failed. The
independent checker verified all thresholds and counts. The exact
report, preflight, smoke log, and source hashes are in
`ordinal_progress/policy_preference/same_start_relative/`. Prospective
audit, online n=4 RL, and val-unseen evaluation were not run for this
representation.

The [Route2Step MIA](https://arxiv.org/abs/2608.03143) instruction-analysis
checkpoint is a distinct, published representation probe, not a model
developed in this project. A pinned public checkpoint produced correctly
tagged responses for four queries from two SHA-selected fit trajectories;
this is a format smoke, not semantic accuracy. Our cached observations
are one frame per multi-action turn, so their spacing differs from the
model's original trajectory input. A metadata-only, scene-balanced
offline screen is frozen before its development scores: two episode
groups per each of eight train-development scenes, exactly four route
variants per group, at turns 3 and 6 (128 queries). It waits for the
third control's full evaluation, then shares the GPU-1 inference lock
with candidate evaluation. This lets it use GPU 1 during candidate
training without overlapping model inference. Its answer-span
alignment, tie-aware same-start ordering, hard-pair performance, and
action-only comparison are fixed in `analyze_route2step_mia_screen.py`.
The inference loop can stop early only when 13 missing answer tags or
26 unaligned answers make the predeclared format gate impossible even
with perfect remaining responses; otherwise it runs all 128 queries.
The frozen manifest and format-only compact summary are under
`ordinal_progress/policy_preference/route2step_mia/`.

The frozen 128-query development screen completed. Every answer had
a parseable tag and mapped to an instruction span, but same-start
stage ordering was weak: episode-macro accuracy was 56.67%/65.89% at
turns 3/6 versus 73.13%/75.31% for the action-only baseline on the
same pairs. The anchor-3 non-tie rate was only 20.41%. Six
prespecified gate checks failed, so this direct stage-position reward
does not enter prospective audit or online n=4 RL. This is a reused
R2R-train development proxy; all 16 four-route sets mix records from
two or three policy seeds and are not online n=4 rollout groups. The
proxy does not provide independently human-validated semantic accuracy
or a val-unseen navigation result. The exact
label-free summary is
`ordinal_progress/policy_preference/route2step_mia/development.json`;
private text responses remain remote.

In parallel with the active n=4 scale, a CPU-only watcher waits for
the first audited exact512 oracle candidate rollout. It will recheck
whether all-failure groups supply enough future-return comparisons
to justify rendering more RGB or fitting a group-relative critic.
The minimum, fixed before the candidate rollout was available, is
150 fit episode groups and 35 development episode groups plus 120
qualifying development pairs at each of turns 3 and 6. A separate
same-terminal-mode check requires 100 fit groups, 25 development
groups, and 60 development pairs at each anchor, reducing the chance
that a future model can exploit STOP versus timeout. It uses no
evaluation GPU and cannot itself establish a learned reward gain.
The runner and checker are `run_future_advantage_exact512_preflight.sh`
and `preflight_group_future_advantage_exact512.py`.

All three completed exact512 outcome-only control rollouts now have a
CPU-only group-signal audit. In seeds 11/22/33, 290/306/292 of 512
four-rollout episode groups have no success and four zero terminal
rewards, despite text-diverse routes in every group. Mean all-failure
coverage is 57.81%. Within those groups, 269/290, 284/306, and 271/292
per seed retain at least one same-terminal-mode pair separated by 1 m in terminal
goal distance (mean conditional coverage 92.79%). The compact report is
`ordinal_progress/policy_preference/exact512_control_group_signal.json`;
`audit_exact512_control_group_signal.py` recomputes the counts from
all 128 steps per seed. This is a train-only opportunity for a process
reward, not evidence that any learned teacher helps navigation.

`train_balanced_change_lora.py` is the next bounded representation
test on this expanded fit set. It predicts forward, backward, or
stationary local visual change from the start/before/after images,
executed action, and instruction. Class-balanced updates, a
within-trajectory direction ranking, and a safe wrong-instruction
contrast target the previous model's poor regression recognition.
The five-microstep GPU-1 smoke passed one nonzero-gradient update;
the 1,500-microstep fit completed on GPU 1 while matched n=4
oracle/control scaling occupied GPUs 0/2/3. All four prespecified
small-development checkpoints failed at least three of six gates.
The selected step-1,000 model reached 58.81% balanced direction
accuracy (62.07% forward, 55.56% backward), 58.33% instruction
preference, and 58.62% forward recall at 9.63% stationary false
positives. Full development, prospective audit, online RL, and
val-unseen were not run. The runner is `run_balanced_change_lora.sh`;
the exact report and log are
`ordinal_progress/policy_preference/balanced_change_lora_development.json`
and `balanced_change_lora_train.log`. No navigation gain is claimed.

`train_route_history_change_lora.py` tests an input-level explanation
for this bias: the three-view model cannot see the intervening route.
It adds up to four preceding RGB boundary views and the executed
actions between them, while retaining the same audited fit records,
three-class objective, scene-disjoint development examples, and fixed
gates. The five-microstep smoke passed one nonzero-gradient update
with 1,843,200 LoRA parameters; the 1,500-microstep fit completed on
GPU 1 using spare memory and compute. The source wrapper checks the
frozen base trainer hash. All four fixed small-development checks
failed. The selected step-1,000 model reached 55.59% balanced
direction accuracy (74.14% forward, 37.04% backward), 37.50%
instruction preference, and 54.31% forward recall at 9.63%
stationary false positives. Full development, prospective audit,
online RL, and val-unseen remained closed. The immutable report and
log are `ordinal_progress/policy_preference/route_history_change_lora_development.json`
and `route_history_change_lora_train.log`.

A CPU-only future-advantage probe reused frozen policy-history features
from genuine same-seed n=4 rollout groups. The all-failure development
subset has only 18 distinct episode IDs. A fixed linear head ranked
future progress on same-terminal-mode pairs at 49.91%/52.22% for turns
3/6, versus 45.37%/36.11% for the action-only baseline. The gap
between fit and development performance does not support deploying
this head as a reward. The script and compact report are
`probe_cached_future_advantage.py` and
`ordinal_progress/policy_preference/cached_future_advantage_probe.json`.
The larger audited exact512 coverage check remains pending.

A separate CPU-only contextual motion head used the same cached
same-seed n=4 source to classify three-turn forward versus regression
changes. At fixed epoch 12, forward episode-macro accuracy across
three head seeds was 72.36%–74.50%, while regression was only
54.20%–58.37%. Every seed missed the frozen directional checks, and
stationary and wrong-instruction controls were not tested. The report
is `ordinal_progress/policy_preference/antisymmetric_motion_probe.json`;
no reward or navigation run was launched from it.

The spare GPU-1 lane completed a matched step-64 exact512 oracle versus
outcome-only control screen on the frozen 256 val-unseen episodes. With
all IDs covered and zero inference errors, candidate/control successes
were 71/78: paired SR -2.73 and SPL -2.83 points. The nine-scene
bootstrap intervals include zero. This reused, one-seed interim check
does not replace the ongoing 128-step three-seed full evaluation.
The compact report and paired episode metrics are under
`ordinal_progress/policy_preference/oracle_exact512_scale/`.

The first two completed 128-step, group-four exact512 candidates now
have matched full 1,839-episode audits, each with zero inference errors.
Seed 11 has 513 versus 450 successes (paired SR +3.43, SPL +3.45
percentage points); seed 22 has 525 versus 545 (SR -1.09, SPL -0.61
points). On the 1,583 episodes outside the reused 256-item screen,
seed 11 is +3.60/+3.60 SR/SPL points and seed 22 is -1.01/-0.62.
Seed 22's source hashes, unique IDs, and paired metrics were independently
recomputed from its compact episode export. The negative second seed
precludes a consistent-gain claim; seed 33 and the prespecified
three-seed analysis remain pending. These are privileged geodesic
training rewards and reused val-unseen development evaluations, not
evidence of an observation-only learned reward. The reports,
validators, frozen screen manifest, and paired episode exports are in
`ordinal_progress/policy_preference/oracle_exact512_scale/`.

The first real sparse RGB future-advantage replay used the idle GPU-1
lane for one selected turn-6 trajectory. It saved the expected three
views, reproduced terminal geodesic distance exactly, and took 10 s
with 692 MiB incremental peak memory. A conservative 1,500 MiB
per-shard floor selects four concurrent shards within a 70% memory
budget. The full 1,795-record replay then ran on idle GPU 1 under the
evaluator's shared GPU lock. It finished in 6 minutes 9 seconds;
its exact-coverage verifier accepted 4,228 fit and 859 development
frames, 37/8 disjoint scenes, the expected pairs, and zero terminal
distance drift. The fixed 1,024-microstep reward-model fit is now
running on that same locked lane while seed 33 trains. The later
evaluation will wait for the lock if needed. A missing
`vlnce_server` module path caused the first smoke attempt to fail
before image collection; the runner path was corrected and the retry
completed. Compact smoke and full-replay verification evidence is under
`ordinal_progress/policy_preference/future_advantage_pooled/`.
