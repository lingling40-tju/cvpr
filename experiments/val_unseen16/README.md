# Exploratory R2R val-unseen subset

This directory records a fixed 16-episode evaluation of the SFT initialization, the destination-only checkpoint after two optimizer steps, and the EventTrace checkpoint after two optimizer steps. The episodes cover 11 scenes. The list in `manifest.json` is shared by all three arms. Each arm used one generation per episode, temperature 0.2, top-p 0.8, a 12-turn limit, and at most 76,800 image pixels. The evaluation followed ActiveVLN's R2R val-unseen configuration; it did **not** run the complete split.

The outputs come from `../eval_val_unseen_subset.py` and `../run_val_unseen_subset.sh` on `wanghaozhihuoshanyun`. Each arm's `summary.json` aggregates the 16 `log/stats_*_0.json` simulator records. `paired_analysis.json` is produced by `python3 ../analyze_val_subset.py`; it uses 10,000 episode-bootstrap resamples with a fixed seed. The interval describes variation over this selected set and does not account for training-seed variation or the selection process.

| Checkpoint | Successes | SR | SPL | Mean goal distance (m) |
| --- | ---: | ---: | ---: | ---: |
| SFT initialization | 6/16 | 0.375 | 0.3441 | 7.784 |
| Destination only | 6/16 | 0.375 | 0.3521 | 7.030 |
| EventTrace | 6/16 | 0.375 | 0.3529 | 6.821 |

EventTrace minus destination-only SR is 0.000 (95% paired episode-bootstrap interval [-0.25, 0.25]); SPL is +0.0009 (interval [-0.238, 0.226]). Each model succeeds on two episodes where the other fails. No model inference errors were logged. These results do not demonstrate a navigation improvement or equivalence.
