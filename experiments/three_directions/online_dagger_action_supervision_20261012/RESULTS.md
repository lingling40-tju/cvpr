# Online shortest-path DAgger pilot: result

The frozen 64-update n=4 pilot and development256 evaluation are complete. **The method failed its predeclared joint gate and was not scaled.**

| Arm | Successes / 256 | SR | SPL |
|---|---:|---:|---:|
| Matched FP16 SFT | 109 | 42.58% | 41.26% |
| Turn-level RLOO control | 52 | 20.31% | 20.31% |
| RLOO + online shortest-path action CE | 0 | 0.00% | 0.00% |

Candidate minus SFT: **-42.58 SR / -41.26 SPL percentage points**. Candidate minus RLOO control: **-20.31 / -20.31 points**. The frozen threshold required at least +2 pp on both metrics versus SFT. All arms have 256 unique IDs and zero inference errors; the remote raw-stat analysis and the separate local compact recount agree.

Both training arms completed 64 optimizer updates and passed the unchanged training auditor with 2,048 teacher targets, nonempty masks, finite KL, and nonzero actor gradients at every update. The candidate correction loss and target metrics were positive and finite. These training signals are not navigation gains.

This is a one-configured-seed result on an adaptively reused eight-scene train-development split. It is not a clean generalization test, does not establish causality, and uses privileged simulator teacher labels rather than a deployable semantic signal. Teacher-query latency was not separately instrumented. Per the frozen protocol, reserved and val-unseen were not opened for tuning; this method was not scaled.

## Recovery notes

The original training wrapper checked for global_step_2 after a 64-step run and wrote false failure markers after both arms had completed step 64. The original markers are preserved as failed.postcheck_expected_step2; hashes of the frozen trainer and audit source remained unchanged, and existing checkpoints/logs passed the unchanged 64-step audit after reconciliation.

The evaluation worker wrote complete raw outputs to the suite root while the aggregator expected a results/ subdirectory. The original suite.failed marker is preserved as suite.failed.result_root_path_mismatch. Raw directories were moved without editing their contents, all models were revalidated, and independent CPU raw-stat and local compact recounts matched. No inference was repeated during recovery.

## Audit files

- evidence_bundle_20261013.tar.gz: remote evidence bundle, SHA-256 7c0a962295df60061a27bf625a63e56ddfa5bbe2f368db79b389a36393cc63da.
- runlogs/online_dagger_pilot64/: frozen prelaunch identity, source hashes, both training audits, preserved marker reconciliations, and train freeze.
- runlogs/online_dagger_development256/: evaluation identity, raw recount, three paired analyses, compact episode exports, validators, and preserved path-mismatch marker.
- verify_dagger_compact.py: local independent recount over the compact 256-row export.
- Reserved and val-unseen were not opened.
