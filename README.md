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
- `experiments/three_directions/fused_online/`: isolated group-four online reward implementation. The frozen temporal/SigLIP fusion passed its 64-step training audit but failed the matched fixed-256 val-unseen screen: 68/256 successes versus 80/256 for the destination-only group-four control, paired SR −4.69 and SPL −4.34 percentage points, with exact coverage and zero inference errors. Its predeclared gate stopped three-seed scaling. A failure-aware temporal encoder improves train-scene failed-route ranking, but direct fusion weakens reused instruction-grounding probes. A follow-up that applies this bounded reward only to unsuccessful rollouts passed its two-step wiring audit; the same-budget group-four 64-step/256-episode test is running. The directory records the checks, scripts, and limitations.

The training pilot uses 16 rollouts per arm and reaches 13/16 successes in each arm. In the complete val-unseen study, EventTrace minus destination-only SR changes by +0.65, +0.44, and -2.88 percentage points across seeds 11, 22, and 33. The mean paired difference is -0.60 points (sample standard deviation 1.98 points); SPL changes by -0.80 points on average. All seven models cover 1,839 unique episodes with zero logged inference errors. The earlier 16-episode probe yielded 6/16 successes for all three pilot checkpoints and is exploratory. In the selected verifier audit replayed with training-time 448-pixel JPEG encoding, 7 of 15 predicted completions agree with blind review labels. The reviewer was an AI assistant and the examples were enriched for event transitions, so this is a failure analysis rather than a population accuracy estimate. Independent human labels and parser validation are still needed.

The later group-size-four representation/reward pilots are documented in
[`experiments/three_directions/fused_online/README.md`](experiments/three_directions/fused_online/README.md).
The mode-stratified ordinal reward completed a matched 64-step pilot on a
third fixed 256-episode val-unseen set: 59 successes versus 73 for its
outcome-only control, paired SR -5.47 and SPL -5.23 percentage points.
The prespecified scale gate failed; no three-seed extension was launched.

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
