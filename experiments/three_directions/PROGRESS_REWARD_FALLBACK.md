# Conditional fallback: bounded endpoint-distance progress

This is a **prospective experiment protocol**, not a trained model or a
navigation result. Run it only if the current three-seed full val-unseen
branch comparison does not establish a useful gain. It needs an isolated
implementation and its own held-out evaluation before any paper claim.

## Training-only observation motivating the test

The seed-11 128-step destination-only rollouts contain 512 two-sample GRPO
groups per arm. The branch arm has 382 equal-return groups; in 297 of those,
the two final Habitat `distance_to_goal` values differ by more than 0.5 m.
The from-scratch matched control has 348 equal-return groups, with 297
showing that same distance gap. No distance is nonfinite in these saved
rollouts. Thus the current sparse reward leaves potentially useful training
distinctions unranked. This is **training-set diagnosis**, not evidence that
distance shaping improves held-out SR or SPL.
`analyze_reward_ties.py` recomputes these counts from the completed seed-11
rollouts; `scale_budget/seed11_reward_ties.json` holds the compact output.

## Reward to test

Let \(d_0\) be Habitat's distance to the target after environment reset and
\(d_T\) the final distance when a rollout stops or exhausts its budget. Add
one terminal term to the existing destination reward:

\[
R_{\mathrm{progress}} = \operatorname{clip}\!\left(
    \frac{d_0-d_T}{\max(d_0,3\,\mathrm{m})},-1,1\right).
\]

The proposed coefficient is 1.0. The existing success floor is 2.0, so a
failed rollout's maximum progress bonus remains below that floor. The term
is evaluated **once at termination**; moving back and forth cannot collect
repeated progress rewards. If either distance is nonfinite, assign zero
progress bonus and count the incident explicitly. The actor sees only its
ordinary observations and instruction. Habitat distance is available to
the training reward and is absent from the evaluation policy input. This
term uses no reference route and no semantic verifier.

This endpoint bonus changes the training objective; it should not be called
policy-invariant potential shaping without a separate proof. It may reward
ending near the goal without stopping, so SR and SPL—not training return—
must determine whether the mechanism is useful.

## Matched experiment

1. Apply `geodesic_progress.patch` to an **isolated copy** of the current
   simulator reward wrapper, environment config, and rollout config plumbing.
   Deploy `progress_reward.py` as
   `vlnce_server/semantic_reward/progress.py` through that patch. Set
   `+actor_rollout_ref.rollout.agent.reward.geodesic_progress_weight=1.0`
   only for the new pilot, and start fresh simulator services. Do not modify
   the running three-seed branch services or their shell scripts. The patch
   has been dry-run and applied successfully against a local copy of the
   current remote source; the pure reward helper passed checks for
   boundedness and nonfinite handling. It has **not** been deployed or
   tested inside a live Habitat rollout.
2. Train a 64-step, seed-11 from-scratch GRPO pilot on the exact
   `branch_pilot_train.parquet` rows used by `branch_control64`, with the
   same SFT initializer, action budgets, sampling count, and optimizer
   settings. The only intervention is the terminal progress term. Compare
   with the existing same-data destination-only control on the frozen 256
   val-unseen episodes, with exact coverage and zero inference errors.
3. If that predeclared pilot has higher paired SR and nondecreasing SPL,
   train 128 steps on the same 512 unique rows for seeds 11, 22, and 33.
   The current `branch_control128` checkpoints provide matched destination-
   only controls only after their training audits pass. Evaluate all six
   models on the fixed 256 and full 1,839 val-unseen manifests, and report
   paired seed changes, scene uncertainty, and compute/interaction costs.

The branch pilot's single-seed scene interval crosses zero. This fallback
therefore cannot inherit a positive claim from that pilot or from the
training-set distance gaps above.
