# Outcome-consistent RLOO pilot (2026-10-09)

This isolated, group-four n=4 follow-up changed two parts of the previous terminal-RLOO setup together: budget truncation was converted to evaluator-equivalent STOP/task-success resolution, and nDTW path credit was restricted to successful terminal outcomes. There is no component ablation, so the combined result cannot attribute effects to either change.

## Frozen training and audit

- 512 fit rows/exposures, 2,048 recorded trajectories, configured seed 11, 64 optimizer updates (2 genuine smoke updates plus 62 resumed updates), group size 4.
- Training audit: all 64 updates had nonzero actor gradients; 741 success and 1,307 failure trajectory outcomes; zero violations of success-conditioned nDTW. Generated response count is not executed-turn count.
- Evaluation used the frozen train-scene development manifest (256 unique episodes, 8 scenes; SHA-256 `8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3`) and the same FP16 SFT reference plus the prior group-four RLOO candidate. Each model had complete coverage and zero inference errors.
- The model result and paired records were checked by the remote raw-ID recount (768 internal IDs, PASS) and a separate local compact recount. The exact source hashes in `runlogs/freeze/identity.json` match the archived training, evaluation, analyzer, and verifier scripts.

## Navigation result

| Model | Successes | SR | SPL |
|---|---:|---:|---:|
| FP16 SFT reference | 109/256 | 42.58% | 41.26% |
| Previous terminal RLOO | 105/256 | 41.02% | 40.73% |
| Outcome-consistent RLOO | 97/256 | 37.89% | 36.86% |

Paired candidate-minus-control changes were `-4.69/-4.41` SR/SPL points versus FP16 SFT and `-3.13/-3.88` points versus prior RLOO. The frozen +2/+2-point gates against both references failed; this method is not eligible for scale. The interval estimates are descriptive on an adaptively reused development screen and do not support generalization claims.

## Recovery record

The first evaluation wrapper exit was an orchestration failure, not an inference failure: it omitted the candidate root `completed` marker after the independent 256-episode validator succeeded. After validating each shard's exact manifest IDs, raw stats, summaries and totals, the missing marker was written and the original nonzero exit record preserved. A later aggregate helper compared two mathematically equivalent SPL computations using exact float equality; their summation-order difference was below `1e-14`. The recovered report uses `1e-10` absolute tolerance, and `verify_outcome_raw_ids.py` independently recounted the raw records. The failure and recovery details are retained in `runlogs/development_suite/suite_recovery.json`, `suite.failed.recovered_after_raw_audit`, and `suite.completed`.

No reserved or val-unseen episodes were used for this decision. The development scenes have been reused adaptively, and the configured rollout seed does not establish independent rollout streams.
