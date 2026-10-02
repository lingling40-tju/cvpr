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

An offline re-scoring check uses the train episode metadata's initial
geodesic distance as a proxy for the runtime reset metric. At coefficient
1.0, the proposed bonus separates 370/382 and 345/348 originally tied
branch/control groups in seed 11, and 385/389 and 341/345 in seed 22.
Across these four saved-rollout sets it reverses no previously non-tied
pair ordering; median new reward gaps range from 0.16 to 0.34. This shows
the term can supply a training ranking on **these fixed trajectories**. It
does not show that a newly trained policy would navigate better, and the
metadata distance is not an exact logged reset-time metric. The calculation
is reproducible with `analyze_progress_tie_breaks.py` and
`scale_budget/seed{11,22}_progress_tie_breaks.json`.
The saved rollouts contain final distance but no per-rollout nDTW or path
coordinates, so they cannot support an equivalent offline comparison of
route-fidelity rewards without replaying the simulator.

An exploratory check on the fixed 256-episode branch pilot finds only a
0.059 m lower mean final distance than its same-data control; a paired
scene bootstrap interval is [-0.400, 0.278] m. Among the 165 episodes
where both models fail, the branch mean final distance is 0.170 m *higher*.
That outcome-conditioned subset is descriptive, not a causal effect estimate.
The small pilot SR difference therefore does not demonstrate broadly better
goal approach. `analyze_branch_terminal_distance.py` reproduces the check
from `val256/branch_matched_episodes.jsonl`, with its compact output in
`val256/branch_terminal_distance.json`.

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

1. Once the six-model complete branch evaluation is finished, inspect its
   paired result. If the fallback condition holds, use
   `stage_progress_fallback.sh` to copy the current source into a separate
   project directory and apply `geodesic_progress.patch` there. The stage
   script verifies the three source hashes, applies the patch, and compiles
   the changed Python modules. `start_progress_service.sh` then starts a
   fresh simulator on port 5011 with a separate Ray directory. Check GPU
   availability before running it. Neither script touches the running
   branch services. The patch has already been applied successfully against
   a local copy of the remote source, and the pure reward helper passed
   boundedness and nonfinite checks. **No patched fallback source, service, or
   trainer has been deployed or tested in live Habitat yet.**
2. Train a 64-step, seed-11 from-scratch GRPO pilot on the exact
   `branch_pilot_train.parquet` rows used by `branch_control64`, with the
   same SFT initializer, action budgets, sampling count, and optimizer
   settings. `run_progress_fallback.sh 64 0,1 11` uses the same destination
   reward plus the once-only terminal progress coefficient 1.0, and asserts
   the matched dataset hash. The only intervention is that progress term.
   `run_progress_pilot_eval.sh` evaluates its checkpoint on the frozen 256
   val-unseen episodes and uses `analyze_matched_pair.py` for an exact-ID,
   zero-error paired comparison with the already evaluated same-data
   destination-only `branch_control64` checkpoint. The evaluator uses the
   unmodified policy environment; the progress bonus is training-only.
3. If that predeclared pilot has higher paired SR and nondecreasing SPL,
   train 128 steps on the same 512 unique rows for seeds 11, 22, and 33.
   The current `branch_control128` checkpoints provide matched destination-
   only controls only after their training audits pass. Evaluate all six
   models on the fixed 256 and full 1,839 val-unseen manifests, and report
   paired seed changes, scene uncertainty, and compute/interaction costs.

The branch pilot's single-seed scene interval crosses zero. This fallback
therefore cannot inherit a positive claim from that pilot or from the
training-set distance gaps above.
