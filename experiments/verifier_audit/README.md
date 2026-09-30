# Blind verifier audit on R2R val-unseen transitions

`collect_verifier_audit.py` collected shortest-path follower trajectories from 16 R2R val-unseen episodes excluded from the earlier 16-episode navigation probe. The parser returned at least one event for 12 episodes. Collection saved before/after RGB views and executed actions, without calling `/verify` or recording its predictions. The same Qwen3-VL-8B service provided candidate event phrases through `/parse`; the audit therefore tests event verification conditioned on those parsed phrases, not parser accuracy.

`annotations.csv` contains 49 single-reviewer, three-way visual labels made from the image pairs, actions, and motion fields before querying `/verify`. `Y` means completion during that turn, `N` means no completion, and `U` means the pair cannot establish the event. The reviewer was an AI assistant, not an independent human annotator. This is a preliminary blind audit rather than a human ground-truth benchmark. Examples were enriched for apparent event transitions and final stops, so class proportions are not a natural rollout distribution.

`audit_results.py` queries the frozen service on the fixed labels and writes `verdicts.json` plus `summary.json`. It keeps verifier abstentions separate from service failures, and reports a confusion matrix rather than claiming population-level accuracy. `examples/` contains only the selected RGB pairs, with hashes recorded in `bundle.json`. The full 289-turn raw replay and frames remain on `wanghaozhihuoshanyun` under `runlogs/verifier_audit_valunseen16/`.

`blind_review_package.zip` contains the 49 image pairs and a randomized `labels.csv` with empty labels. It contains neither the AI review labels nor verifier predictions. Two independent reviewers can each fill a separate copy using `Y` (newly completed), `N` (not newly completed), or `U` (insufficient evidence), without consulting this repository's result files or each other. Resolve their disagreements in a third adjudicated CSV. Then run `python3 score_independent_labels.py reviewer1.csv reviewer2.csv adjudicated.csv` to compute agreement and verifier confusion against human adjudication. The package can be regenerated with `python3 prepare_blind_package.py blind_review_package.zip`. No human annotations have been supplied yet.

The image pairs derive from Matterport3D scans and are subject to its [academic terms of use](https://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf). Matterport3D is credited in the paper.

| Review label | Predicted Y | Predicted N | Predicted U |
| --- | ---: | ---: | ---: |
| Y (8) | 7 | 0 | 1 |
| N (29) | 2 | 15 | 12 |
| U (12) | 4 | 1 | 7 |

The frozen verifier predicts Y on 13 cases; 7 match reviewed Y, 2 are reviewed N, and 4 are reviewed U. One reviewed N was a turn after the agent had already entered the target room, showing that current visual state can be mistaken for a newly completed event. No review label was changed after obtaining these predictions.
