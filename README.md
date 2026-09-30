# EventTrace: CVPR paper draft

This repository contains a CVPR-style manuscript on ordered semantic event rewards for online vision-language navigation training. The PDF and its LaTeX source describe a measured feasibility pilot and a fixed, exploratory R2R val-unseen evaluation. **The current results do not establish improved navigation performance.**

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

The training pilot uses 16 rollouts per arm and reaches 13/16 successes in each arm. On the exploratory val-unseen subset, the SFT, destination-only, and EventTrace checkpoints each succeed in 6/16 episodes. EventTrace has 0.3529 SPL and destination-only has 0.3521 SPL; the paired episode-bootstrap interval for their SPL difference is [-0.238, 0.226]. These results do not show an improvement. A credible CVPR submission still needs a verifier audit with independently labeled examples, larger equal-budget training across several seeds, and full held-out SR/SPL plus event-completion results.
