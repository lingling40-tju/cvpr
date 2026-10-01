# EventTrace: CVPR paper draft

This repository contains a CVPR-style manuscript on ordered semantic event rewards for online vision-language navigation training. The PDF and its LaTeX source describe a measured feasibility pilot and a fixed, exploratory R2R val-unseen evaluation. **The current results do not establish improved navigation performance.**

The VLN-CE simulator uses Matterport3D scans. The paper and selected audit images derive from that dataset and are subject to the [Matterport3D academic terms of use](https://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf). We cite Chang et al. (3DV 2017) in the manuscript; this repository includes only a small set of image pairs needed to inspect the audit.

## Paper

- `main.tex`, `main.bib`, `figures/`: editable paper source.
- `main.pdf`: compiled review-style draft.
- `cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex`: files from the [official CVPR author kit](https://github.com/cvpr-org/author-kit), commit `291758547e923160eb4d37079b7b9f0dfce82355` (downloaded 2026-10-01). Its latest release at drafting time was CVPR 2026. The manuscript header is set to 2027 provisionally; replace the style files if the 2027 kit changes.

Build with `tectonic main.tex --keep-intermediates` or a standard LaTeX/BibTeX workflow. The paper uses only public citations and the provided pilot data. The submission ID and author identity remain unset because no submission has been made.

## Evidence and experiments

- `experiments/pilot8/`: original 8-episode R2R training pilot outputs and curated event manifest.
- `experiments/implementation/`: the reward prototype and patch against the [Apache-2.0 ActiveVLN repository](https://github.com/arvillion/ActiveVLN) at base commit `3a0c63b00e4f42c828cc74c3554afce17641da60`.
- `experiments/eval_val_unseen_subset.py`, `experiments/run_val_unseen_subset.sh`: fixed episode subset evaluation for SFT, destination-only, and semantic-event checkpoints.
- `experiments/val_unseen16/`: episode manifest, per-episode simulator statistics, arm summaries, and paired analysis from the same 16 val-unseen episodes across 11 scenes. `python3 experiments/analyze_val_subset.py` recomputes the paired results.
- `experiments/verifier_audit/`: 49 blind three-way visual review labels from 12 separate val-unseen episodes, selected RGB evidence, frozen verifier responses, and a confusion matrix. The labels were prepared by one AI assistant before the 8B verifier was queried; there is no independent human adjudication.
- `experiments/verifier_audit/blind_review_package.zip`: prediction-free and label-free package for two independent human reviewers; `score_independent_labels.py` requires their labels and an adjudicated CSV before reporting a human-referenced score.
- `experiments/run_multiseed_train.sh`, `experiments/run_multiseed_suite.sh`: matched three-seed, 64-step training commands running on `wanghaozhihuoshanyun`.
- `experiments/multiseed_train_analysis.json`: completed corrected 64-step training summary for seeds 11, 22, and 33. Each arm sampled 256 train episodes and 512 rollouts per seed with matched episode order. These are training-rollout diagnostics, not held-out navigation results; the full val-unseen evaluation is running separately.
- `experiments/INVALID_RUNS.md`: audit trail for an excluded zero-gradient training attempt caused by duplicate GRPO samples; the corrected sampling patch and preflight gate are under `experiments/`.
- `experiments/prepare_full_val_manifest.py`, `experiments/run_full_val_unseen.sh`, `experiments/run_full_suite.sh`, `experiments/analyze_full_val.py`: full 1,839-episode val-unseen evaluation and analysis pipeline, staged to run after training.
- `experiments/run_parallel_full_suite.sh`, `experiments/validate_full_label.py`: two evaluation lanes (model servers on GPUs 1 and 3, ports 8004 and 8005) with four Habitat shards each on GPU 2. They use the same manifest and check exact 1,839-episode coverage and inference errors before marking a model complete. The training-only verifier and simulator services on GPU 3 were stopped after training finished to free capacity.

The training pilot uses 16 rollouts per arm and reaches 13/16 successes in each arm. On the exploratory val-unseen subset, the SFT, destination-only, and EventTrace checkpoints each succeed in 6/16 episodes. EventTrace has 0.3529 SPL and destination-only has 0.3521 SPL; the paired episode-bootstrap interval for their SPL difference is [-0.238, 0.226]. These results do not show an improvement. In the selected verifier audit, 7 of 13 predicted completions agree with blind review labels. The reviewer was an AI assistant and the examples were enriched for event transitions, so this is a failure analysis rather than a population accuracy estimate. Human-adjudicated verifier labels and the running multi-seed full-split experiment are needed before a performance claim.
