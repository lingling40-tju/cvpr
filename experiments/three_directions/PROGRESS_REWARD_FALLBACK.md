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
The completed seed-33 rollouts provide the same diagnostic: 380/512 branch
and 347/512 control groups have equal original returns; 305 and 301 of
those pairs differ by more than 0.5 m in final distance. The proposed bonus
separates 374/380 and 342/347 ties. Across all three seeds, it separates
1,129/1,151 branch ties and 1,028/1,040 control ties on saved trajectories,
with zero previously non-tied preferences reversed and zero invalid distance
rollouts. Exact seed-33 records are `scale_budget/seed33_reward_ties.json`
and `scale_budget/seed33_progress_tie_breaks.json`; the seed-22 tie counts
are also archived. This remains an offline training-rollout check using a
proxy initial distance, not a trained-policy or held-out navigation result.
The saved rollouts contain final distance but no per-rollout nDTW or path
coordinates, so they cannot support an equivalent offline comparison of
route-fidelity rewards without replaying the simulator.

The proposed narrower variant of adding terminal distance only to originally
tied two-sample groups is not a distinct next pilot on these saved rollouts.
The offline check found zero reversed preferences in every non-tied group
across all three seeds and both arms. Two-sample GRPO normalizes the two
returns within each episode, so a non-tied pair keeps the same preference
direction while the distance term chiefly activates tied pairs. The
completed progress pilot already tested that training signal and missed its
held-out gate. This is an argument against repeating a nearly identical
pilot, not a claim that all distance-based objectives are ineffective.

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
   branch services. The patch was applied successfully against the isolated
   remote source, and the pure reward helper passed boundedness and
   nonfinite checks. After the full branch comparison missed its
   predeclared positive-mean gate, the simulator on port 5011 passed its
   health check and the 64-step pilot completed. Its audit and held-out
   result are reported below.
2. Train a 64-step, seed-11 from-scratch GRPO pilot on the exact
   `branch_pilot_train.parquet` rows used by `branch_control64`, with the
   same SFT initializer, action budgets, sampling count, and optimizer
   settings. `run_progress_fallback.sh 64 0,1 11` uses the same destination
   reward plus the once-only terminal progress coefficient 1.0, and asserts
   the matched dataset hash. Before marking the pilot complete, it audits all
   64 four-episode training steps against the same-data control, checks two
   rollouts per episode and no replayed prefix, and requires a bounded,
   nonzero progress reward component in the recorded rollouts. The only
   intervention is that progress term.
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

## Conditional execution

`run_progress_fallback_conditional.sh` waits for the complete six-model
1,839-episode analysis and checks its model coverage, zero-error status, and
paired mean SR/SPL. A positive first pass waits for the separate stochastic
decode replication. It starts the isolated progress pilot only if the first
pass misses the predeclared sign rule or the replication reverses it. If both
passes retain the sign, it records `branch_gain_retained` and does not start
the fallback. A failed or incomplete upstream suite stops this watcher with a
failure marker. The watcher never interprets partial evaluation files as a
result; after a qualifying negative comparison, it stages the isolated
source, starts its simulator, trains the matched 64-step pilot, and evaluates
the frozen 256 episodes. A positive pilot still requires a separately
audited three-seed expansion before any held-out claim.

`run_progress_scale_conditional.sh` waits for that pilot. It checks the
exact 256-episode manifest and the paired SR/SPL against the existing
same-data 64-step control. Only a positive SR change with nondecreasing SPL
starts three 128-step runs on the fixed 512-row train dataset. Each seed's
rollouts are audited against its destination-only control for all 128
four-episode steps; the audit also checks that the bounded progress component
actually entered the training records. The watcher then evaluates the three
progress checkpoints on both the frozen 256 episodes and all 1,839 val-unseen
episodes. `analyze_progress_scaled.py` computes the exact-ID paired three-seed
summaries and scene/seed intervals. A negative pilot records `no_pilot_gain`
and consumes no additional training or inference. This pilot did not pass;
the scale watcher recorded `no_pilot_gain` and did not train larger models.
Because the 256-episode pilot screen is a subset of the 1,839-episode
benchmark, the complete three-seed report should also show the paired
comparison on the 1,583 episodes outside that screen, using
`analyze_disjoint_holdout.py progress`. That secondary subset spans ten
scenes; the screen exhausts one small scene. It limits direct reuse of the
pilot's episode outcomes when interpreting a later scale-up but is not a
fully untouched benchmark across all experiments in this project.

If the three-seed complete evaluation has a positive mean paired SR and
nonnegative mean paired SPL, `run_progress_full_replication_suite.sh` makes
a separate six-model decode pass over the same 1,839 episodes with vLLM
seed 20261003. It leaves the first-pass files intact and does not add
independent training seeds. Both complete passes must retain the positive
sign rule before the downstream route-fidelity watcher records a retained
progress benefit; otherwise that watcher may stage the next alternative.
This replication only probes decoding variability and cannot create an
independent held-out benchmark.

## Completed seed-11 pilot

The 64 training steps used exactly the same episode set at each step as the
64-step destination-only control: 256 unique training episodes and two
sampled trajectories per episode (512 rollouts). The logged progress term
was present in all 512 rollouts, nonzero in 510, and had no invalid distance
values. The checkpoint validation found finite actor gradients at every step.
These checks establish that the intervention ran; they are not navigation
outcomes. See `progress64/paired_train_audit.json` and
`progress64/validation.json`.

On the frozen 256-episode val-unseen manifest, the progress model succeeded
on 50/256 episodes (SR 19.53%, SPL 19.38%), while its same-data control
succeeded on 75/256 (SR 29.30%, SPL 29.01%). The exact-ID paired changes
were **-9.77 SR and -9.63 SPL percentage points**. Both models had zero
inference errors; the progress evaluator covered all 256 unique episodes.
The exploratory 11-scene bootstrap intervals were [-16.67, -2.66] SR and
[-16.59, -2.39] SPL percentage points. This is one seed and one decode,
so it does not establish a general effect size, but it fails the
predeclared positive pilot gate. The 128-step scale-up and independent
decoder replication were therefore not eligible.

`progress64/paired_progress64_vs_branch_control64.json` stores the analysis,
and its compact `.jsonl` counterpart stores the 256 paired outcomes. The
local archive also contains the progress-model coverage validator output.
The paired SR/SPL and distance aggregates are recomputed by
`verify_progress_pilot.py` from the `.jsonl` records. The reference-route nDTW
fallback is the next conditional experiment.
The richer `progress64/full_pair/` package also includes oracle and early-stop
fields for each episode and passes `verify_val256_pair_package.py`.
