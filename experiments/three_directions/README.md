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
an insufficient eight-slot service. The matched 64-step seed-11 pilot is
still running after overlapping the route-fidelity run. It has no held-out
navigation result yet and uses twice as many sampled trajectories per
episode as its control; compute costs will be reported with any outcome.
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
pilot is running concurrently with the four-sample pilot; it has no
held-out navigation result yet.

`run_optimizer_scale_conditional.sh` waits for each fixed-256 pilot's
completed, zero-error paired analysis. Only a strictly positive paired SR
and nonnegative paired SPL against `branch_control64` permit its matching
128-step three-seed training suite. An ineligible pilot records
`no_pilot_gain` and consumes no scaled training GPU time. The scaled
`run_optimizer_scale.sh` uses the same 512 train rows and seed-specific
episode order as the completed destination-only controls, with either four
rollouts per row or the 0.001 actor KL loss. `audit_optimizer_scaled_pair.py`
checks all 128 per-step episode sets, all 512 distinct training episodes,
reward isolation, gradient finiteness, and completed checkpoints for each
seed. A completed scale training marker is only a training result; held-out
navigation claims require subsequent fixed-256 and complete-1839 paired
evaluation. Both conditional watchers were launched while the 64-step
pilots were still running, so neither scaled arm has been deemed eligible.
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
