# Post-hoc generated-turn and termination diagnostic

| Model | Successes | Median saved assistant turns | Last turn contains a parsed STOP proposal |
| --- | ---: | ---: | ---: |
| Unchanged FP16 SFT | 555 | 12 | 738 |
| control seed 11 | 223 | 5 | 1832 |
| candidate seed 11 | 243 | 5 | 1822 |
| control seed 22 | 216 | 5 | 1830 |
| candidate seed 22 | 220 | 5 | 1826 |
| control seed 33 | 217 | 5 | 1837 |
| candidate seed 33 | 254 | 5 | 1823 |

These are the same completed seven-model, 1,839-episode evaluations, not new navigation runs. The CPU exporter checks all 12,873 internal raw IDs and assistant-output files against the frozen matched-FP16 metric export, then applies the exact hash-checked original parser methods to saved generated text. This compact package publishes parsed action IDs and source hashes, not images or original generated text.

Both updated arms generate fewer saved assistant turns than the unchanged reference (median 5 versus 12). In candidate seeds 11/22/33, 371/399/377 reference successes become failures; within those subsets median saved turns are 5/4/5 for the candidate and 12/12/12 for SFT. These observations motivate checking premature termination as one hypothesis if the current pilots fail. They do not prove it causes the loss, establish a shorter-turn policy is worse in general, or validate changing the STOP reward.

`early_stop_reason=null` alone is not classified as STOP. The features describe parsed generated proposals, not a separately stored low-level executed-action trace. Physical execution, instruction completion and causality are not independently verified here. Same split, single shared SFT decode and configured-seed limitations apply. The original frozen training, evaluators, gates and navigation metrics remain unchanged.

`verify_positive_terminal_proposals_20261009.py` independently reconciles every copied source metric, all turn/proposal summary counts and paired lost-success subsets. `independent_local_recount.json` passed. This is a compact arithmetic recount, not a second raw-text parse.
