# SFT-anchored RLOO, KL coefficient 1.0

This isolated n=4 pilot tested whether a stronger KL penalty to the navigation
SFT initialization would preserve performance after earlier policy-gradient
updates fell below that reference. It used Qwen2.5-VL-3B SFT initialization,
512 frozen R2R-train rows, batch size 8, configured seed 11, and 64 total
optimizer updates (two real smoke updates plus 62 resumed updates). The
geodesic-progress reward is privileged simulator feedback, not a semantic
verifier or deployable reward. Source and evaluation identities are frozen in
`runlogs/freeze/identity_v3.json`.

## Result

On the fixed, adaptively reused development256 split (eight train scenes), the
candidate achieved 101/256 successes versus 109/256 for matched FP16 SFT.
Paired candidate-minus-SFT changes were -3.125 SR and -2.984383 SPL percentage
points. All 256 unique episodes were covered, with zero inference errors; the
independent compact recount agrees. The frozen gate required at least +2 points
on both SR and SPL, so it failed and the method was not scaled. The scene
bootstrap intervals span zero. This is a single configured-seed development
result, not a clean generalization test or evidence that a larger KL coefficient
causes the decline.

The training audit combines all 64 optimizer steps, including the two-step
smoke run and 62 resumed steps; every step had a nonzero actor gradient and
finite KL. The development evaluator initially omitted its candidate
`.completed` marker after validating inference output. We preserved that
original wrapper failure, independently audited the raw coverage and per-episode
statistics, then resumed CPU analysis without rerunning inference. The final
`suite.completed` reflects successful analysis and independent recount, not the
first wrapper exit. See `runlogs/development_suite_v3/` for the report, source
freeze checks, raw-ID/coverage evidence and compact records.
