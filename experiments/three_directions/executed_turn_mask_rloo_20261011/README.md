# Executed-turn-mask RLOO pilot (2026-10-11)

This group-four, 64-update n=4 RLOO pilot masks the final generated turn when
that turn was forced by the turn budget but had no executed action. It otherwise
uses the outcome-consistent turn-RLOO setup, including privileged simulator
geodesic progress. The change is training-only and is not a deployable semantic
reward.

On the fixed 256-episode, 8-scene development screen, the candidate achieved
103/256 successes (SR 40.23%, SPL 39.48%). The same-precision SFT reference
achieved 109/256 (42.58%, 41.26%); prior terminal RLOO achieved 105/256
(41.02%, 40.73%). Paired candidate deltas were -2.34/-1.78 SR/SPL points
versus SFT and -0.78/-1.25 points versus prior RLOO. Exact coverage and zero
inference errors were checked. Raw-ID, analysis, compact-verifier, and local
recounts agree. Both frozen +2/+2 advancement gates failed, so this method was
not scaled.

The original evaluator wrapper failed after inference because it analyzed the
candidate before creating its completion marker and retained a stale candidate
alias in later scripts. Existing inference was not rerun. The recovered CPU
postprocessing preserves the original failure markers/log hash and records
the alias mismatch in runlogs/development_suite/executed_turn_recovery_report.json.
The evidence does not support a group-size effect: group size remained four.
These adaptively reused train-scene development episodes are not a clean
generalization test.

Key evidence is in runlogs/development_suite/: frozen identity, raw recovery
audit, paired compact records and reports, independent recount, and recovered
suite report. The exact code/config sources are under source_snapshot/ and are
checked against the identity SHA table by verify_source_snapshot.py. Only the
fit parquet and its manifest are omitted from this archive. The actual inference
label differs from the freeze declared candidate alias; consult the recovery
report before interpreting source labels.
