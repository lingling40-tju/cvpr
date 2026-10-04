# EventTrace: CVPR paper draft

This repository contains a CVPR-style manuscript on ordered semantic event rewards for online vision-language navigation training. The PDF and its LaTeX source report a wiring pilot, corrected three-seed 64-step training, and evaluation of seven checkpoints on all 1,839 R2R val-unseen episodes. **The current results do not establish improved navigation performance.**

The VLN-CE simulator uses Matterport3D scans. The paper and selected audit images derive from that dataset and are subject to the [Matterport3D academic terms of use](https://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf). We cite Chang et al. (3DV 2017) in the manuscript; this repository includes only a small set of image pairs needed to inspect the audit.

## Paper

- `main.tex`, `main.bib`, `figures/`: editable paper source.
- `main.pdf`: compiled review-style draft.
- `cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex`: files from the [official CVPR author kit](https://github.com/cvpr-org/author-kit), commit `291758547e923160eb4d37079b7b9f0dfce82355` (downloaded 2026-10-01). As checked on 2026-10-02, the public kit still identifies CVPR 2026; the manuscript header is set to 2027 provisionally. The [official CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) lists November 10, 2026 for registration and November 16 for submission (Anywhere on Earth). Check for a new kit before submission.

Build with `tectonic main.tex --keep-intermediates` or a standard LaTeX/BibTeX workflow. The paper uses only public citations and the provided pilot data. The submission ID and author identity remain unset because no submission has been made.

## Evidence and experiments

- `experiments/pilot8/`: original 8-episode R2R training pilot outputs and curated event manifest.
- `experiments/implementation/`: the reward prototype and patch against the [Apache-2.0 ActiveVLN repository](https://github.com/arvillion/ActiveVLN) at base commit `3a0c63b00e4f42c828cc74c3554afce17641da60`.
- `experiments/eval_val_unseen_subset.py`, `experiments/run_val_unseen_subset.sh`: fixed episode subset evaluation for SFT, destination-only, and semantic-event checkpoints.
- `experiments/val_unseen16/`: episode manifest, per-episode simulator statistics, arm summaries, and paired analysis from the same 16 val-unseen episodes across 11 scenes. `python3 experiments/analyze_val_subset.py` recomputes the paired results.
- `experiments/verifier_audit/`: 49 blind three-way visual review labels from 12 separate val-unseen episodes, selected RGB evidence, frozen verifier responses, and a confusion matrix. The labels were prepared by one AI assistant before the 8B verifier was queried; there is no independent human adjudication.
- `experiments/verifier_audit/blind_review_package.zip`: prediction-free and label-free package for two independent human reviewers; `score_independent_labels.py` requires their labels and an adjudicated CSV before reporting a human-referenced score.
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

The training pilot uses 16 rollouts per arm and reaches 13/16 successes in each arm. In the complete val-unseen study, EventTrace minus destination-only SR changes by +0.65, +0.44, and -2.88 percentage points across seeds 11, 22, and 33. The mean paired difference is -0.60 points (sample standard deviation 1.98 points); SPL changes by -0.80 points on average. All seven models cover 1,839 unique episodes with zero logged inference errors. The earlier 16-episode probe yielded 6/16 successes for all three pilot checkpoints and is exploratory. In the selected verifier audit replayed with training-time 448-pixel JPEG encoding, 7 of 15 predicted completions agree with blind review labels. The reviewer was an AI assistant and the examples were enriched for event transitions, so this is a failure analysis rather than a population accuracy estimate. Independent human labels and parser validation are still needed.

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
three-seed oracle/control scale is running; it has no scaled navigation
result yet. A separate observation-only,
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
