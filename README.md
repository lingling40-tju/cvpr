# EventTrace: CVPR paper draft

This repository contains a CVPR-style manuscript on ordered semantic event rewards for online vision-language navigation training. The PDF and its LaTeX source report a wiring pilot, corrected three-seed 64-step training, and evaluation of seven checkpoints on all 1,839 R2R val-unseen episodes. **The current results do not establish improved navigation performance.**

The VLN-CE simulator uses Matterport3D scans. The paper and selected audit images derive from that dataset and are subject to the [Matterport3D academic terms of use](https://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf). We cite Chang et al. (3DV 2017) in the manuscript; this repository includes only a small set of image pairs needed to inspect the audit.

## Paper

- `main.tex`, `main.bib`, `figures/`: editable paper source.
- `main.pdf`: combined review-style working draft. The main text summarizes
  the semantic reward, complete matched results, human agreement, and the
  positive-trajectory pilot with its completed replication and SFT check. Detailed credit,
  start-state, group-size, and representation experiments are preserved in
  the supplementary section of the same `main.tex`. The current build has
  main text extending onto page 7, references on page 7, and supplementary
  material on pages 8–12. This combined file must be separated into the
  main-paper and supplementary uploads before submission, as the author kit
  specifies; the experiments and final submission preparation are ongoing. The
  project-aware `tectonic main.tex --keep-intermediates` build succeeds
  using the checked-in `cvpr.sty`. The standalone editor compiler
  cannot resolve that companion style file.
- `cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex`: files from the [official CVPR author kit](https://github.com/cvpr-org/author-kit), commit `291758547e923160eb4d37079b7b9f0dfce82355` (downloaded 2026-10-01). As checked on 2026-10-02, the public kit still identifies CVPR 2026; the manuscript header is set to 2027 provisionally. The [official CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) lists November 10, 2026 for registration and November 16 for submission (Anywhere on Earth). Check for a new kit before submission.

Build with `tectonic main.tex --keep-intermediates` or a standard LaTeX/BibTeX workflow. The paper uses only public citations and the provided pilot data. The submission ID and author identity remain unset because no submission has been made.

## Evidence and experiments

- `experiments/pilot8/`: original 8-episode R2R training pilot outputs and curated event manifest.
- `experiments/implementation/`: the reward prototype and patch against the [Apache-2.0 ActiveVLN repository](https://github.com/arvillion/ActiveVLN) at base commit `3a0c63b00e4f42c828cc74c3554afce17641da60`.
- `experiments/eval_val_unseen_subset.py`, `experiments/run_val_unseen_subset.sh`: fixed episode subset evaluation for SFT, destination-only, and semantic-event checkpoints.
- `experiments/val_unseen16/`: episode manifest, per-episode simulator statistics, arm summaries, and paired analysis from the same 16 val-unseen episodes across 11 scenes. `python3 experiments/analyze_val_subset.py` recomputes the paired results.
- `experiments/verifier_audit/`: 49 blind three-way visual review labels from 12 separate val-unseen episodes, selected RGB evidence, frozen verifier responses, and a confusion matrix. The original labels were prepared by one AI assistant before the 8B verifier was queried. Two independent human reviews are now summarized separately; their 16 disagreements have not been adjudicated.
- `experiments/verifier_audit/blind_review_package.zip`: prediction-free and label-free package for independent human reviewers; `score_independent_labels.py` requires an adjudicated CSV before reporting a human-referenced score.
- [`experiments/turn_rloo_20261005/`](experiments/turn_rloo_20261005/): checksummed two-human agreement (33/49, Cohen's kappa 0.459), a completed group-four turn-level return-to-go/leave-one-out pilot, and a frozen 256-episode val-seen paired evaluation. All 64 training steps have nonzero actor gradients; both models cover 256 unique episodes with zero inference errors. The candidate succeeds on 68/256 versus 78/256 for the matched destination-only GRPO control (paired SR -3.91 and SPL -3.72 percentage points). Independent per-episode recount and descriptive scene-bootstrap intervals are included. The fixed advancement gate failed, so this coupled optimizer/reward mechanism was not expanded. Geodesic progress is privileged training information, not a semantic-verifier result.
- [`experiments/turn_factorial_20261006/`](experiments/turn_factorial_20261006/): completed group-four, four-arm, one-seed factorial on a frozen disjoint 256-episode val-seen screen. Control, combined turn/progress, terminal-only turn, and progress-augmented GRPO succeed on 97, 96, 98, and 106 episodes, respectively, with exact coverage and zero inference errors. Relative to the control, progress-augmented GRPO reaches +3.52 SR but only +1.51 SPL percentage points; all three candidates fail the frozen joint +2-point advancement gate. A token-scale-matched terminal-RLOO contingency then completed another disjoint 256-episode val-seen screen: 108 versus 103 successes, paired SR +1.95 and SPL +1.26 points, zero inference errors, and an independent raw recount. Its joint gate also failed; the conditional three-seed scale was skipped. Compact records, validators, training audits, and the documented factorial verifier schema repair are included. These one-seed development comparisons do not establish unseen-scene gain or a deployable semantic reward.
- [`experiments/turn_factorial_20261006/posthoc_protocol.json`](experiments/turn_factorial_20261006/posthoc_protocol.json): user-directed exploratory expansion after normalized terminal-RLOO missed its frozen +2/+2 pp gate (SR +1.953125, SPL +1.2573 pp). The original conditional scale was skipped. The isolated n=8 sensitivity has now completed 64 training steps: [63/64 steps had a logged nonzero actor gradient](experiments/turn_factorial_20261006/posthoc_n8_val_seen256/train_gradients.json); step 23 had an all-failure group and zero advantage. On the same 256-episode, 38-scene val-seen development screen, n=8 succeeds on 112 episodes versus 103 for the matched destination-only control: paired SR **+3.52** and SPL **+2.82** percentage points. Relative to the earlier n=4 candidate's 108 successes, n=8 adds +1.56 SR and +1.56 SPL points. [Raw-stat independent recount](experiments/turn_factorial_20261006/posthoc_n8_val_seen256/independent_recount.json) and [local compact recount](experiments/turn_factorial_20261006/posthoc_n8_val_seen256/verify_n8_compact.py) agree; each arm covers all 256 IDs with zero inference errors. Descriptive scene-cluster intervals cross zero for n=8 versus control (SR [-4.92, 12.13], SPL [-5.28, 11.10] points). These one-seed, post-result results are suggestive only: n=8 doubles simulator rollouts at the same prompts and steps, and val-seen has been reused adaptively. The n=4 three-seed 512-row/128-step expansion has now completed; results are summarized below. The initial n=8 Habitat port and insufficient-actor smoke failures occurred before optimizer steps and were retained remotely; the repeated [two-step smoke](experiments/turn_factorial_20261006/n8_smoke_gradients.json) passed. The [schema repair](experiments/turn_factorial_20261006/posthoc_verify_n8.py) only allows numeric `0.0/1.0` success values in raw verification. Both validation splits have already been used adaptively.
- [Post-result n=4 three-seed full val-unseen expansion](experiments/turn_factorial_20261006/posthoc_n4_full1839/independent_three_seed_recount.json): matched 512-row × 128-step n=4 candidates versus same-seed destination-only controls cover all 1,839 episodes in each arm (11 scenes, zero inference errors). Seed 11: 523/450 successes, paired SR +3.97 and SPL +3.30 points; seed 22: 485/545, -3.26/-3.07; seed 33: 568/492, +4.13/+3.00. Mean paired SR/SPL +1.61/+1.07 points with seed SD 4.22/3.60; exploratory seed-and-scene intervals [-3.03, 5.12]/[-2.86, 4.21] cross zero. The [independent local compact recount](experiments/turn_factorial_20261006/posthoc_n4_full1839/verify_compact_recount.py) confirms all three 1,839-episode exports and the remote report. This was a user-authorized post-result expansion despite the earlier frozen gate failure; inconsistent seed signs do not establish a stable gain. Val-unseen had already been used for development.
- [`experiments/turn_gae_20261006/`](experiments/turn_gae_20261006/): multimodal GAE/PPO fallback after the normalized pilot failed its joint gate. The fixed 64-step pilot had nonzero actor and critic gradients at all 64 steps and finite critic losses ([training audit](experiments/turn_gae_20261006/real64/train_64_gradient_audit.json)). On the complete paired 778-episode val-seen screen (53 scenes, zero inference errors), control succeeded on 272 episodes and GAE on 22: paired SR **-32.13** and SPL **-31.56** percentage points. Descriptive scene-cluster intervals are [-36.83, -27.43] for SR and [-36.19, -26.95] for SPL. Independent raw recounts agree, and the [compact evidence](experiments/turn_gae_20261006/real64/val_seen778/) can be checked with [`verify_compact_recount.py`](experiments/turn_gae_20261006/verify_compact_recount.py). The precompletion verifier initially failed after inference under Python 3.8 because of a type annotation; the [recovery record](experiments/turn_gae_20261006/real64/val_seen778/verification_recovery.json) documents a Python 3.10 verification of the existing raw results without rerunning inference. The frozen +2/+2 pp gate failed and the conditional three-seed GAE scale was skipped. Shorter candidate paths are descriptive, not a causal diagnosis.
- [`experiments/trajectory_sil_20261006/`](experiments/trajectory_sil_20261006/): success-trajectory on-policy follow-up on frozen 45 fit / 8 development / 8 reserved R2R-train scenes. The 64-step n=4 pilot passed its original +2/+2 pp gate against GRPO (51/17 development successes; paired SR/SPL +13.28/+13.28 pp). The six matched 512-row/128-step runs and reserved/full evaluations have now completed. Full1839 candidate/control mean paired SR/SPL are +1.11/+1.07 pp, but the candidate is **-17.18/-16.28 pp below unchanged, matched-FP16 SFT**. No further scale is planned for this method. [Completed compact evidence and local independent recounts](experiments/trajectory_sil_20261006/completed20261009/) preserve all results and limitations; native BF16 SFT diagnostics remain separate.
- The [positive-trajectory pilot protocol](experiments/trajectory_sil_20261006/PILOT_PROTOCOL.md) and [CPU-checked advantage](experiments/trajectory_sil_20261006/positive_trajectory_advantage.py) are frozen as a backup after the authorized n=8 and n=4 jobs completed. Both 256-ID screens resolve against the actual Habitat R2R-train dataset after its scene-path prefix is normalized. The [derived train-scene evaluator](experiments/trajectory_sil_20261006/eval_train_scene_subset.py) passed CPU-only coverage checks for both screens and rejected a role mismatch; the [paired analyzer](experiments/trajectory_sil_20261006/analyze_train_scene_pair.py) passed a synthetic raw-shard I/O check. A separate remote source tree now contains the hash-verified fit512 data and [patched trainer](experiments/trajectory_sil_20261006/prepare_positive_source.py); CPU integration passed for both candidate and matched GRPO branches. Conditional [service](experiments/trajectory_sil_20261006/start_positive_service.sh), [training](experiments/trajectory_sil_20261006/run_positive_train.sh), [development evaluation](experiments/trajectory_sil_20261006/run_positive_development_eval.sh), and [pilot orchestrator](experiments/trajectory_sil_20261006/run_positive_pilot_suite.sh) were staged before evaluation; the [training audit](experiments/trajectory_sil_20261006/audit_positive_train.py) checks reward advantages separately from KL gradients. The new Hydra configuration and synthetic validator-to-analysis pipeline passed CPU checks. The corrected matched two-arm [real smoke audits](experiments/trajectory_sil_20261006/real_smoke/) each cover two optimizer steps with nonzero actor gradients; the candidate has nonnegative positive advantages and both arms have maximum score 19.279. The first control-only smoke revealed an unintended +2 success floor and was preserved before [the configuration repair](experiments/trajectory_sil_20261006/smoke_reward_range_recovery.json). The [control](experiments/trajectory_sil_20261006/real_64step_control_audit.json) and [candidate](experiments/trajectory_sil_20261006/real_64step_candidate_audit.json) real 64-step audits now pass, with nonzero actor gradients on all steps and nonnegative candidate advantages. The first development evaluation failed because the evaluator selected train episodes but retained val-unseen nDTW references. The [recovery record](experiments/trajectory_sil_20261006/ndtw_reference_recovery.json) preserves eight failed shard-log hashes and four partial stat files; all partial outputs were archived. The evaluator now requires train references for every frozen episode; [four real environment resets](experiments/trajectory_sil_20261006/repaired_reset_preflight.json) passed without model calls or navigation actions. An [evaluation-only recovery](experiments/trajectory_sil_20261006/run_positive_eval_recovery.sh) completed the unchanged pair concurrently, followed by independent compact recount before completion. Training, policy checkpoints, manifest, metric formulas, and the +2/+2-point gate were not changed. This is positive-only on-policy learning inspired by self-imitation, not replay-buffer SIL. Its one-seed development gain is relative to a weak GRPO control; the completed matched-FP16 initialization check is 109/256 successes versus candidate 51/256, and reserved configured-seed replication is complete. It does not establish unseen-scene benefit or validate a semantic verifier.
- `experiments/run_multiseed_train.sh`, `experiments/run_multiseed_suite.sh`: matched three-seed, 64-step training commands used on `wanghaozhihuoshanyun`.
- `experiments/multiseed_train_analysis.json`: completed corrected 64-step training summary for seeds 11, 22, and 33. Each arm sampled 256 train-episode instances and 512 rollouts per seed with matched episode order. These are training-rollout diagnostics, not held-out navigation results.
- `experiments/INVALID_RUNS.md`: audit trail for an excluded zero-gradient training attempt caused by duplicate GRPO samples; the corrected sampling patch and preflight gate are under `experiments/`.
- `experiments/prepare_full_val_manifest.py`, `experiments/run_full_val_unseen.sh`, `experiments/run_full_suite.sh`, `experiments/analyze_full_val.py`: full 1,839-episode val-unseen evaluation and analysis pipeline.
- `experiments/run_parallel_full_suite.sh`, `experiments/validate_full_label.py`: two evaluation lanes (model servers on GPUs 1 and 3, ports 8004 and 8005) with four Habitat shards each on GPU 2. They use the same manifest and check exact 1,839-episode coverage and inference errors before marking a model complete. The training-only verifier and simulator services on GPU 3 were stopped after training finished to free capacity.
- `experiments/full_val_unseen/`: exact episode manifest, compact per-episode metrics, paired analysis, completion markers, validation records, and evaluator logs for all seven models. The raw vLLM request logs are omitted due to their size. `experiments/export_full_val_compact.py` creates the compact export.
- `experiments/qwen38_upgrade/`: fixed-label audit comparing the original 8B verifier with DashScope `qwen3.8-max-0902` at training-matched image resolution, plus a two-step integration smoke. The planned 256-step paired suite was superseded after the seed-11 control step-64 checkpoint; no navigation result is claimed for that suite.
- `experiments/three_directions/`: code, fixed manifests, and audited results for policy-prefix branching, failure recovery, natural instruction counterfactual training, group-four optimization, and representation-to-reward probes. The three 64-step pilots completed a fixed 256-episode val-unseen screen. Branching reached 82/256 successes versus 75/256 for its same-data from-scratch control ($+2.73$ SR and $+2.37$ SPL percentage points), but its single-seed scene interval crossed zero; recovery and counterfactual training did not pass the screen. A larger matched branch/control comparison trained 128 steps on 512 unique train episodes at seeds 11, 22, and 33. On all 1,839 val-unseen episodes, its mean paired changes are **$-1.14$ SR and $-0.55$ SPL percentage points**; seed SR changes are $-7.18$, $+0.49$, and $+3.26$ points. All six models have exact coverage and zero inference errors. The exploratory scene-and-seed interval includes zero, and a 1,583-episode subset outside the fixed screen has mean changes of $-1.22$ SR and $-0.55$ SPL points. Separate stochastic decodes on the overlapping 256 episodes change many individual outcomes. `experiments/three_directions/scale_full1839/` contains checksummed paired episode records, analyses, and a checked LaTeX table. The predeclared positive-mean gate failed, so no second full decode pass was launched. An isolated 64-step terminal-distance progress-reward pilot also failed its fixed 256-episode screen (50 successes versus 75 for its same-data control; paired SR -9.77 and SPL -9.63 percentage points). Its matched training audit, exact-coverage analysis, and paired episode records are in `experiments/three_directions/progress64/`. A matched reference-route nDTW pilot also missed its fixed 256-episode gate (74 successes versus 75; paired SR -0.39 and SPL -0.40 percentage points), with audits and per-episode records in `experiments/three_directions/route64/`.
- `experiments/three_directions/optimizer_scale_full/`: three-seed, 128-step group-four destination-only comparison with a same-data two-sample control. The verified package contains six compact 256-episode and six compact full-1,839-episode paired records, manifests, training audits, and SHA-256 checks. The complete val-unseen mean paired difference is **+2.10 SR and +2.51 SPL percentage points**; seed SR differences are +0.65, +1.36, and +4.30 points. The four-sample arm uses twice as many simulator rollouts, and these results do not validate a semantic reward. Recompute coverage, metrics, and hashes with `python3 experiments/three_directions/verify_optimizer_scaled_package.py experiments/three_directions/optimizer_scale_full --mode group4`.
- `experiments/three_directions/ordinal_progress/policy_preference/`: offline representation-to-reward probes. A fixed equal-weight fusion of a causal temporal encoder and a LoRA-adapted SigLIP image--instruction encoder improves development successful-route ranking to 43/52 while retaining 39/52 instruction grounding. It scores 30/38 and 34/38 on seed-33 episode-disjoint training-scene pairs. The final scores use fixed 64-token text padding and one-image inference; earlier batch-dependent reports are marked invalid in their filenames. These are offline diagnostics, not navigation results.
- `experiments/three_directions/fused_online/`: isolated group-four online reward implementation. The frozen temporal/SigLIP fusion passed its 64-step training audit but failed the matched fixed-256 val-unseen screen: 68/256 successes versus 80/256 for the destination-only group-four control, paired SR −4.69 and SPL −4.34 percentage points, with exact coverage and zero inference errors. Its predeclared gate stopped three-seed scaling. A failure-aware temporal encoder improves train-scene failed-route ranking, but direct fusion weakens reused instruction-grounding probes. The unsuccessful-rollout-only follow-up also failed its matched 256-episode screen: 45/256 versus 93/256 successes, paired SR −18.75 and SPL −17.43 points. Its predeclared gate likewise stopped three-seed scaling. The directory records both negative results and STOP-calibration diagnostics.

The training pilot uses 16 rollouts per arm and reaches 13/16 successes in each arm. In the complete val-unseen study, EventTrace minus destination-only SR changes by +0.65, +0.44, and -2.88 percentage points across seeds 11, 22, and 33. The mean paired difference is -0.60 points (sample standard deviation 1.98 points); SPL changes by -0.80 points on average. All seven models cover 1,839 unique episodes with zero logged inference errors. The earlier 16-episode probe yielded 6/16 successes for all three pilot checkpoints and is exploratory. In the selected verifier audit replayed with training-time 448-pixel JPEG encoding, 7 of 15 predicted completions agree with the original single-AI blind review. Two subsequent independent human reviewers agree on 33/49 selected cases; the 16 disputes still require blinded adjudication before any model-versus-human accuracy estimate. This selected set cannot estimate population accuracy. Parser validation remains outstanding.

The GAE result is specific to the frozen token-clock implementation. Its
[CPU-only source diagnostic](experiments/turn_gae_20261006/gae_token_clock_cpu_diagnostic.json)
confirms that discounting proceeds per generated action-text token while
observation spans are skipped. An idealized zero-value example shows
text-length-sensitive credit; it does not identify the cause of the measured
navigation decline or evaluate decision-clock GAE.

The later group-size-four representation/reward pilots are documented in
[`experiments/three_directions/fused_online/README.md`](experiments/three_directions/fused_online/README.md).
The mode-stratified ordinal reward completed a matched 64-step pilot on a
third fixed 256-episode val-unseen set: 59 successes versus 73 for its
outcome-only control, paired SR -5.47 and SPL -5.23 percentage points.
The prespecified scale gate failed; no three-seed extension was launched.

A training-only geodesic turn-wise credit upper bound passed a fourth
matched group-four 256-episode screen: 81 successes versus 72 for its
same-data outcome control, paired SR +3.52 and SPL +3.76 points, with
zero inference errors. Its nine-scene interval includes zero, and the
distance label is not a deployable semantic reward. The one-seed full
1,839-episode recheck found 602 oracle successes versus 545 for its
same-data n=4 control (paired SR +3.10, SPL +3.05 percentage points,
zero inference errors). The 1,583 episodes outside the reused screen
gave SR +2.53 and SPL +2.50 points. These are post-screen,
single-seed development results. A matched n=4, 512-row, 128-step
three-seed oracle/control scale has completed training and full evaluation.
Its first candidate seed completed 128 n=4 steps,
passed the matched-row training audit, and completed exact-1,839
val-unseen evaluation with zero inference errors. It reached 513
successes versus 450 for its matched control (paired SR +3.43,
SPL +3.45 points); outside the reused 256-item screen, paired SR and
SPL were both +3.60 points. Seed 22's matched full 1,839-episode
evaluation is negative: 525 versus 545 successes, paired SR -1.09 and
SPL -0.61 points; on the 1,583 episodes outside the reused screen it
is -1.01/-0.62 SR/SPL points. All three exact512 seed evaluations had zero
inference errors and were checked against frozen manifests and paired
episode records. Seed 33 succeeds on 585 versus 492 episodes:
paired SR +5.06 and SPL +4.70 points. The three-seed mean is +2.47 SR
and +2.51 SPL points (sample SD 3.18/2.78); on the 1,583 episodes
outside the reused screen it is +2.63/+2.64 points. The exploratory
scene-and-seed 95% intervals, [-1.49, 6.05] SR and [-1.20, 5.78] SPL,
include zero. The matched four-rollout mechanism is therefore promising
but inconsistent across seeds, and its geodesic training reward is
privileged. The frozen manifests, three-seed paired analysis, per-episode
exports, validation records, and independent recount are under
[`experiments/three_directions/ordinal_progress/policy_preference/oracle_exact512_scale/`](experiments/three_directions/ordinal_progress/policy_preference/oracle_exact512_scale/).
An adaptive stop-boundary variant passed its matched n=4 64-step training
audit, but failed the reused 256-item screen: 67 versus 72 successes,
paired SR -1.95 and SPL -1.62 points. Its positive SR-and-SPL scale
gate closed. The [pilot package](experiments/three_directions/ordinal_progress/policy_preference/stop_boundary_pilot/)
includes the paired episode export, source audit, and independent recount.
An observation-only arrival representation was then tested on disjoint
R2R-train fit/development scenes from the existing n=4 rollouts. The
single-frame occupancy head failed its frozen development gate: AUC
0.6053, inside-region recall 8.33%, and near-failure recall 5.04% with
pooled FPR below 5%. A new joint two-RGB, instruction-conditioned transition
head used real near-goal retreat and same-frame wrong-instruction
negatives. Its fixed 1,024-microstep fit reached development AUC
0.6430 on 758 pairs, versus 0.5177 for a text-only control and 0.5211
after same-scene image-pair shuffling. Despite that visual signal, its
crossing recall was only 15.63% and near-failure recall 13.45% at
4.89% FPR, below the frozen 55%/50% gates. The correct instruction
ranked above an identical-image wrong instruction for only 98/176
pairs (55.68%). A CPU-only source audit confirms identical states
for each pair but finds no independent semantic adjudication of the
geometric alternate-goal labels. The independent recount,
exact RGB replay audit, controls, and
[two-view protocol](experiments/three_directions/MULTIVIEW_EVENT_REWARD_PROTOCOL.md)
are included. Neither head opened its reserved train-scene audit nor
entered online RL; no navigation gain follows from these offline AUCs.
A fit-only CPU check found that installed Matterport3D scans have
human room/object annotations, while the R2R goals provide only a
position and radius. A literal terminal-room and approximate region-box
link agrees for just 71 records from 13 episode IDs; it is not an
independent semantic correctness label. The aggregate
[source preflight](experiments/three_directions/ordinal_progress/policy_preference/multiview_event_source/mp3d_semantic_link_fit_preflight.json)
records the coverage and its geometric limitations.
No learned semantic reward has shown a navigation gain. The frozen CPU-only
future-advantage coverage gate passed after seed 22 finished its n=4
training audit. Seeds 11 and 22 yield 1,349/1,081 fit and 258/211
scene-disjoint development pairs at turns 3/6, without cross-seed
rollout pairs. A 1,490-record fit and 305-record development RGB
replay manifest is frozen. The four-shard real RGB replay completed
and passed source/frame verification with zero terminal-distance
drift. The fixed 1,024-microstep reward-model fit ran on idle GPU 1
under the evaluator's shared lock, but failed both frozen development
gates. At turns 3/6 its all-pair episode-macro accuracies were
50.44%/61.48%, below the required 70%; no online RL or navigation
gain is claimed for that checkpoint. See the
[observation-only learned reward protocol](experiments/three_directions/FUTURE_ADVANTAGE_LEARNED_PROTOCOL.md)
and its compact pooled report. Coverage is not reward accuracy.
An exploratory current-anchor visual potential fit using the same
verified group-four RGB source passed its turn-3 development gate but
failed the required turn-6 gate; it has no online RL or navigation
result. The separately frozen, post-development turn-3-only audit
also failed: 67.26% all-pair episode-macro ranking across 340 pairs,
only 1.43 points above an action-distance baseline. No online RL
was launched. See the [anchor potential protocol](experiments/three_directions/ANCHOR_POTENTIAL_PROTOCOL.md)
and [turn-3 audit](experiments/three_directions/EARLY_ANCHOR_AUDIT_PROTOCOL.md).
The step-64 interim 256-item paired screen was negative (SR -2.73,
SPL -2.83 points), while the final step-128 reused screen for seed 11
is positive (SR +2.34, SPL +2.52 points). A separate observation-only,
policy-prompt STOP representation fit reached development AUC .921 but
missed its predeclared recall gates, so it did not enter RL; its report is in
[`experiments/three_directions/README.md`](experiments/three_directions/README.md).

The current representation-to-reward study keeps the standard rollout
group size at **four**. A history-grounded LoRA progress head and a
pairwise head on its frozen states both failed their train-scene
development regression gates (31.06% and 52.80% respectively); neither
entered online RL. A new complete-group comparison has frozen
160/40/40 four-rollout fit/development/audit groups from existing
outcome-only training logs. Four-GPU replay passed exact coverage and
source-distance checks, with 1,628/381/398 matched-turn candidate pairs
at least one meter apart. Feature caching and a scene-disjoint offline
head screen completed with 73.2% same-turn and 73.9% forward rank
accuracy, but only 59.1% regression rank accuracy against a frozen
60% gate. The candidate was rejected before online RL. The locked
model audit and val-unseen set remain unopened for this candidate.
Scripts, frozen manifest, gate,
and negative reports are in
[`experiments/three_directions/NEXT_PROCESS_REWARD_PROTOCOL.md`](experiments/three_directions/NEXT_PROCESS_REWARD_PROTOCOL.md).

Two later observation-only progress LoRAs also missed fixed small
development gates: independently predicted history potential reached
48.88% balanced forward/regression accuracy, and a joint two-view
antisymmetric scorer reached 53.52%. Neither reached online RL.
An action-conditioned start/before/after visual probe completed its
1,000-microstep fit on the otherwise idle GPU 1 and also failed its
fixed development gate (55.51% balanced direction accuracy, 22.22%
regression accuracy, 66.67% correct-instruction preference). Its
action-text-only control reaches
50.14% balanced direction accuracy on the matched 96-trajectory
small development subset. The frozen prospective train-scene audit,
online RL, and val-unseen remained closed for this probe.
An audited diversity-first fit-only extension reused 768 cached
trajectories and replayed 256 more, adding 129 one-meter regression
turns from 77 episode IDs. A three-class visual-change LoRA using
the expanded 1,024-trajectory fit set completed its bounded fit
but failed all four fixed small-development checks. The selected
checkpoint reached 58.81% balanced direction accuracy and 58.33%
correct-instruction preference, below its prespecified gates; it
did not enter prospective audit, online RL, or val-unseen. The primary
online comparison remains matched group size four; an eight-sample
diagnostic is reserved for a confirmed n=4 algorithmic gain.
A second fit adds preceding visual route history while holding the
training split, loss, and gates fixed. Its five-step smoke passed,
but all four fixed small-development checks failed. The selected
checkpoint reached 55.59% balanced direction accuracy and 37.50%
correct-instruction preference, so it did not enter prospective
audit or online RL. Neither observation-only probe has a navigation
result.
An additional fit-only collection reuses completed n=4 control
rollouts from 256 previously unused R2R-train episode IDs. A
label-only pass over 1,021 valid variants selected 512 trajectories
with 316 one-meter regressions from 139 episode IDs. All 512 RGB
replays passed the per-turn geodesic consistency audit (zero measured
distance drift), producing 5,826 images. This supplies training data
for a future representation and is not a navigation result. The four
compact reports and resource-aware validation plan are under
`experiments/three_directions/ordinal_progress/policy_preference/control_fit_extension/`
and `experiments/three_directions/NEXT_PROCESS_REWARD_PROTOCOL.md`.
The next train-only wrong-goal preflight found a reachable alternate
goal for 250 of those IDs. Label-only replay of their 500 selected
trajectories found 929 turns where the correct and alternate goals
changed geodesic distance in opposite directions by at least 0.5 m,
across 172 IDs. The correct-goal trace matched the independent RGB
replay exactly. This passes the frozen data-coverage gate, but no
new potential model or navigation evaluation has yet passed a gate.
The bounded crossed-goal potential and its unbounded-head ablation
both failed all four fixed small-development checks. A third variant
adding 596 safe expert instruction swaps also failed all four checks:
instruction preference improved, but local direction remained weak.
A warm-started same-start pairwise representation is now fitting on
GPU 0 while the n=4 policy comparison runs on GPUs 2/3. None of these
representation probes has a held-out navigation result.

### Group-four representation screens (2026-10-03)

All new policy comparisons keep `rollout.n=4` as the standard. A group
larger than four would get only a small, compute-matched check with its
own control after a group-four method succeeds. The current tests
reuse existing four-rollout R2R-train histories and initial RGB; only
36 previously uncached initial views were rendered. Frozen SFT response
scoring was split over four A800s for fit and two for development.

- A first-action success-versus-failure preference LoRA did not pass its
  development gate: its best checkpoint ranked 52/86 pairs correctly,
  versus 51/86 for frozen SFT, below the required five-point gain. It
  received no model audit or navigation run.
- A linear value readout at preterminal turns 3 and 6 ranked 75/103
  development and 98/131 model-held-out audit success/failure pairs
  correctly, passing its fixed 70% offline ranking gates. Four A800s
  extracted the audit histories in independent shards.
- The required wrong-goal instruction test then reduced ranking only
  from 98/131 to 89/131, a 6.87-point drop below the required 10-point
  gate. None of the effective matched pairs had a near-identical-start
  natural alternative instruction. This value readout was **not**
  connected to RL, and no held-out navigation improvement is claimed.
- A follow-up linear readout jointly trained on group-four outcomes and
  same-start expert correct-versus-wrong instructions. Its frozen
  model audit ranked 96/131 policy outcome pairs and 105/123 expert
  instruction pairs correctly. On preterminal policy histories, a
  natural wrong-goal swap reduced ranking only to 90/131, a 4.58-point
  drop below the same predeclared 10-point gate. It was also **not**
  connected to RL. The next screen will move the instruction contrast
  to the intermediate history states where a process reward acts.
- That fixed prefix readout used 1,107 fit and 303 development
  same-start instruction contrasts, cached across four A800s. It kept
  outcome ranking at 74/103 but reached only 203/303 (67.0%)
  development instruction contrasts, below the frozen 75% gate. It
  stopped before policy swap scoring or RL. A new development swap
  manifest now contains natural near-identical-start alternatives for
  six mixed-success four-rollout groups; this can screen a future
  encoder-level representation method more directly.
- A fixed 512-microstep encoder-level LoRA pilot improved the frozen
  readout's development four-rollout outcome ranking from 74/103 to
  77/103 and same-start prefix instruction ranking from 196/303 to
  214/303. The latter reached only 70.6%, below the predeclared 75%
  floor (scene macro 68.8% versus a 70% floor), so the pilot stopped
  before policy swaps, online RL, or val-unseen. Its checkpoint,
  cache audit, and exact metrics are included for reproduction.
- A compute-matched bidirectional crossed-trajectory LoRA trained on
  exact-same-start, visually diverged expert routes. It improved the
  selected 2-by-2 development matching diagnostic from 67.4% to 79.0%
  but reached only 215/303 (71.0%) on the broader instruction-prefix
  check, with 78/103 four-rollout outcomes. Both instruction floors
  remained unmet, so the candidate also stopped before online RL and
  val-unseen. Its frozen manifest, checkpoint, per-pair diagnostic,
  and cache audit are included.
- A compute-matched instruction-conditioned temporal interaction LoRA
  used 503 fit and 143 development expert intervals with label-only
  geodesic selection. It reached 79/103 four-rollout outcomes but only
  208/303 prefix instruction contrasts and 102/143 temporal
  interactions. The latter two miss their fixed 75% floors, so this
  candidate stopped before policy swaps, model audit, online RL, and
  val-unseen. Training, cache audit, and fixed final checkpoint are
  retained with the development report.
- A source-image audit found identical three-turn RGB histories in 83
  of 156 checkable development pairs whose expert instructions have
  different goals. An evidence-onset LoRA then trained on 295
  source-selected branch pairs, keeping group size four. It improved
  the 94-pair development branch comparison from 251/376 to 274/376
  individual decisions and the 44-pair onset signal from 49/88 to
  61/88 route directions. It reached 76/103 outcome comparisons but
  remained at 208/303 on the broad instruction test. Only 37/94
  branch pairs got all four decisions right, below the frozen 75%
  gate. It received no policy-swap, model-audit, online RL, or
  val-unseen run. The source audit, per-pair scores, training report,
  checkpoint, and cache audit are included.
- A train-only clause-alignment preflight split 640 expert instructions
  into ordered short clauses and encoded them with fixed 64-token
  SigLIP padding. On 64 directional comparisons from 32 calibration
  route pairs, the complete instruction scored 48 correct; final-clause,
  monotone alignment, and alignment-minus-start rules scored 42, 46,
  and 47. No short-clause rule improved the fixed-padding baseline,
  so it received no reward or navigation training budget. The hashed
  manifest and paired per-route analysis are retained.
- A frozen SigLIP spatial feature pass cached 7x7 pooled patches for
  640 train-scene expert routes and independently audited all 3,840
  frames. A fixed rank-eight clause-to-patch grounder fit 256/256
  training directions but fell to **41/64** on 32 different-scene
  calibration pairs, below the untrained full-instruction baseline's
  **48/64**; strict both-direction pairs were 10/32 versus 17/32.
  The preregistered offline gate failed, so this head was not used
  for policy reward or val-unseen evaluation. Source hashes, per-pair
  margins, the final weight hash, and the cache audit are retained.
- A second frozen SigLIP pass encoded overlapping local crops through
  its pretrained image pooler, avoiding the small learned patch/text
  map. The prespecified ordered regional clause score reached 43/64
  different-scene calibration directions versus 48/64 for complete
  instructions; its offline gate failed. A local last-clause diagnostic
  reached 50/64 but did not improve the fit split and was not selected
  after calibration. The 19,200 region-vector cache audit and paired
  results are included; no online reward or val-unseen claim follows.
- A separate frozen local Qwen3-VL-8B route matcher compared six ordered
  expert views against two natural instructions with the same start and
  different goals. Its A/B order-averaged score reached **57/64** on
  32 calibration pairs versus **48/64** for frozen SigLIP, and
  **226/256** on 128 fit pairs versus **177/256**. Both pair, scene,
  and option-order gates passed. These are exploratory R2R-train
  representation results. Transfer to actual policy histories failed:
  at preterminal turn six it ranked successful group-four trajectories
  above same-group failures in only **5/15** comparisons, and only
  4/8 successful histories increased their goal-evidence margin from
  turn three. This teacher is not used as a dense reward. A separate
  terminal route-fidelity screen got 13/19 same-group success/failure
  rankings right but only 5/11 successful trajectories increased
  their score above their own initial view, failing its frozen gate.
  Neither absolute score entered online RL. A distinct group-relative
  terminal preference passed its fixed offline fit screen: 53/71
  same-group success/failure rankings and 74.2% group-macro across
  55 groups. In the 33 all-failure groups, the same teacher ranked
  the closer route above a farther route in 60/80 comparable
  same-terminal-mode pairs (69.6% group-macro), passing its frozen
  fit gate. A same-scene but different-start counterfactual failed its
  frozen transfer screen (42/71 outcome pairs; 57.9% group-macro), so
  it will not be used to cover the original 256-row training set.
  The next group-four pilot uses a new scene-balanced 256-row subset
  with exact-start natural counterfactuals and retrains its matched
  outcome-only control. These offline ranks are not a navigation gain.
  The prompt, model-shard hashes, paired
  margins, and source manifests are retained under `qwen3_route_match/`.
  The matched online group-four protocol, resource schedule, and
  current run status are in
  [`experiments/three_directions/qwen_group4/README.md`](experiments/three_directions/qwen_group4/README.md).

Source hashes, correlated-group intervals, scripts, negative results,
and the next candidate requirements are in
[`experiments/three_directions/NEXT_PROCESS_REWARD_PROTOCOL.md`](experiments/three_directions/NEXT_PROCESS_REWARD_PROTOCOL.md).

## Completed positive-trajectory evaluation

All six fresh 512-row/128-step, n=4 runs are complete. The
[six-arm training-budget recount](experiments/trajectory_sil_20261006/scale128/rollout_budget_local_recount_six_completed.json)
verifies 1,024 episode exposures and 4,096 recorded trajectories per arm,
two full passes over 512 fit IDs and identical paired episode membership.
The final three scalar/header exports and their collector SHA chains were
also independently checked. These are training evidence, not navigation SR.

The [completed results](experiments/trajectory_sil_20261006/completed20261009/)
include each fixed screen, native-SFT diagnostics, matched-FP16 SFT
comparison, validators, actual engine precision proofs and independent
local recounts. All final model episode sets are exact and inference errors
are zero. The same-precision comparison is:

| Screen | Shared FP16 SFT successes | Positive / GRPO successes | Paired positive minus GRPO SR / SPL, pp |
| --- | ---: | --- | --- |
| Development256, step64 | 109 | 51 / 17 | +13.28 / +13.28 |
| Reserved256, step128 | 91 | 35/24; 23/23; 30/27 | mean +1.82 / +1.76 |
| Full1839, step128 | 555 | 243/223; 220/216; 254/217 | mean +1.11 / +1.07 |

On full1839, unchanged FP16 SFT has SR **30.18%**, SPL **29.23%**.
The positive candidate mean is **13.00%/12.95%**, or **-17.18/-16.28 pp**
versus SFT. The earlier large pilot gain over the weak updated control
does not establish improvement over initialization; this method is not
expanded further. No causal diagnosis follows from these scores alone.

The [sampling scope](experiments/trajectory_sil_20261006/SAMPLING_SEED_SCOPE.md)
still applies: these are three configured-seed replications with a common
initial GPU generation rule; independence of all sampling streams is not
established. There is one shared SFT decode and zero independent SFT training
seeds. Extra/SFT evaluation was designed after pilot inspection and frozen
before scale outputs. Screens were reused adaptively and are exploratory.
The composite positive-credit rule has no component ablations. Human-review
limitations remain unchanged: 33/49 agreements, 16 unresolved disagreements;
no model semantic accuracy is calculated and no extra annotation is requested.

## Current three-direction pilot

The [new isolated n=4 protocol, smoke evidence and automatic evaluation](experiments/three_direction_anchor_20261009/)
compare GRPO with stronger reference KL (0.1), turn-level active-peer RLOO,
and an SRGPO-style process-group composite. All three real two-update smoke
runs passed. Each counts those updates in a fixed total of 64; GRPO is
running and the frozen training relay resumes the other two in sequence.
The tested two-GPU topology prevents three simultaneous trainers on this
host; paired evaluation will run two models concurrently with eight total
GPU2 Habitat shards. Unrelated GPU3 services remain available to their owner.

The fixed development256 FP16 SFT baseline is reused after complete file,
raw coverage and actual-engine checks (109 successes, SR42.58%, SPL41.26%).
The evaluation relay waits for all three exact 64-step training audits before
opening new candidate inference. Five paired comparisons get independent
recounts. Scale eligibility requires both SR and SPL >=+2 pp versus SFT;
RLOO and SRGPO-style candidates also require both >=+2 pp versus the new
GRPO anchor. **No candidate navigation gain has been measured yet.**
No reserved or val-unseen episode is opened for pilot tuning.

## Training and evaluation budget audit

A new [post-hoc source/record audit](experiments/trajectory_sil_20261006/completed20261009/posthoc_budget_alignment/) documents a concrete objective convention: training budget exhaustion is scored as failure with zero outcome reward, while evaluation forces STOP at the turn limit. In two completed seed-11 fit runs, 230/314 of 4,096 trajectories per arm finish strictly within 3 m but receive zero timeout reward. The shared full SFT evaluation records 238 successful forced turn-limit stops among its 555 successes. Independent local arithmetic reconciles all 8,192 training records and the previous SFT export. These observations are separate training/evaluation diagnostics; they establish no recovered SR or causal explanation. Current frozen three-direction pilots continue unchanged.


The current training audit now checks the original **per-episode** command
budget rather than incorrectly requiring 36 for every row. Its
[transparent pre-inference amendment](experiments/three_direction_anchor_20261009/#pre-inference-audit-correction)
preserves training, inference, metric sources and advancement gates;
real smoke preflight and independent source/identity recount passed. This
is an orchestration repair, with no new measured navigation gain.
