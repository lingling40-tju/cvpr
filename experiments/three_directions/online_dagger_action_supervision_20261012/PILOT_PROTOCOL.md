# Frozen pilot protocol: on-policy DAgger-style action correction

Protocol status: frozen pilot and development256 evaluation are complete. Both candidate-vs-SFT metrics missed the advancement gate. The method was stopped without scale-up; preserve this as a negative one-seed development result.

## Question and comparison

Test whether dense, privileged shortest-path action correction on states actually visited by the current policy improves navigation over the same matched SFT initialization. This is supervised simulator feedback, not a semantic verifier, reward model, or deployed-time component.

- Control: turn-level RLOO with the existing terminal weighted-success plus nDTW reward.
- Candidate: identical RLOO and rollout behavior, plus coefficient `0.05` times per-target-token-mean cross-entropy on one online teacher action per sampled trajectory per optimizer update.
- Reference: unchanged matched FP16 SFT initialization, decoded under the same evaluation settings.

The candidate-vs-SFT comparison is the fixed decision gate. Candidate-vs-control is attribution only and cannot substitute for the SFT gate.

## Fixed training budget

- Qwen2.5-VL-3B, same SFT initialization for both arms.
- `fit512`, batch 8, `n=4`, configured/data seed 11, learning rate `1e-6`, reference KL coefficient `0.01`.
- Exactly 64 fresh optimizer updates per arm. Disposable two-step smoke updates are excluded and discarded.
- 12-turn episode limit, 36-command maximum, no more than 3 primitive actions per response.
- Each update samples 8 prompts and 4 responses per prompt (32 on-policy trajectories).
- Uniformly sample one observed decision turn per trajectory. Query `ShortestPathFollower` at that exact visited simulator state with goal radius 2.99m. Map its primitive action to one canonical DSL action: `stop`, `move forward 25cm`, `turn left 15 degrees`, or `turn right 15 degrees`.
- Save the exact pre-response multimodal context and teacher target. The teacher label is independent of the student-generated action; never require agreement or intersect with sampled response tokens.
- Add per-target-token-mean CE with coefficient 0.05. Exclude prompt, image, role, and EOS tokens. Student-generated rollouts remain the data for on-policy RLOO.

Expected label budget is 32 teacher targets/update and 2,048 total across 64 updates if all labels are valid. Require at least 90% valid teacher labels among sampled turns and exactly 32 sampled targets/update; record dropped-label reasons and actual additional simulator time.

## Smoke gate

Before either 64-update arm, run a real two-update multimodal candidate/control smoke in a new isolated source tree, then discard its weights. Require:

1. all four teacher action categories parse to the expected primitive action;
2. every sampled target has a nonempty action-only token mask, with prompt/image/role/EOS excluded;
3. exactly 32 sampled targets/update, with no agreement filtering;
4. finite, nonzero candidate correction loss and correction gradient;
5. finite actor gradients and KL for both arms;
6. valid teacher-label coverage and measured incremental label-generation time.

Missing labels, empty masks, nonfinite values, or zero correction gradient fail closed. Do not begin the full pilot if smoke fails.

## Evaluation and frozen gate

Evaluate candidate, matched control, and unchanged FP16 SFT on the same development256 manifest (SHA-256 `8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3`). The split has been adaptively reused and is not a clean generalization test. Require for every arm exactly 256 unique episode IDs, zero inference errors, matched decoding settings, and an independent per-episode recount.

Proceed only if candidate vs SFT has paired SR **and** SPL each at least `+2.0` percentage points. If either misses, stop this method without opening reserved or val-unseen for tuning and do not scale. No causal or clean-generalization claim from this single configured-seed pilot.


## Observed result and decision

Both 64-update training audits passed with 2,048 teacher targets per arm, 32 targets/update, nonempty masks, finite KL, and nonzero actor gradients throughout. On development256, matched FP16 SFT/control/candidate achieved 109/52/0 successes and SPL 41.26%/20.31%/0%. All arms had exact 256-ID coverage and zero inference errors. Independent raw-stat and local compact recounts agree. Candidate-minus-SFT paired SR/SPL were -42.58/-41.26 pp; the frozen +2/+2 pp gate failed. Do not tune on reserved or val-unseen and do not expand this method.

This is one configured seed on an adaptively reused train-scene screen, not a clean generalization test or causal result. The teacher is privileged simulator supervision. The smoke did not separately measure teacher-query latency. Recovery markers for a stale step-2 checkpoint check and a result-root mismatch are retained; no inference was repeated.
