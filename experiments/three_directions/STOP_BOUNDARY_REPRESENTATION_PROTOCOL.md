# Stop-boundary representation and reward: next exploratory hypothesis

Status (2026-10-05): **training-data coverage and offline reward-signal
preflights completed; the n=4 pilot is queued after the full evaluation;
no new checkpoint or navigation result**. This idea follows the failed
observation-only potential audits and inspection of the first two
scaled privileged-oracle val-unseen seeds. It is therefore adaptive
method development, not a prespecified test of the current oracle.

## Why test a boundary signal

The current turn-wise oracle rewards geodesic distance reduction on
movement turns but gives no auxiliary gradient to STOP tokens. In the
seed-22 complete val-unseen paired export, its terminal distance is
0.83 m lower on average than the matched n=4 outcome control, yet it
has 525 rather than 545 successes and 394 rather than 254
`max_turns_reached` terminations. These are post hoc checkpoint
associations, not evidence that distance credit caused late stopping.
The seed-11 contrast has the opposite success direction, so the
three-seed evaluation remains necessary.

The completed training rollouts provide a cheaper coverage check.
Across the three 512-episode, n=4 oracle runs, respectively
330/302/298 failed trajectories visited within 3 m of the goal at
least once; 239/213/216 ended within 3 m without task success.
Each seed had 97--102 all-failure groups containing at least one
such near-goal failure. The corresponding near-goal failures came
from 215/200/196 unique training episode IDs. This establishes an
opportunity to train a STOP-boundary representation; it says nothing
about the accuracy of an image-only predictor or about improvement in
SR/SPL. The [recount](ordinal_progress/policy_preference/oracle_exact512_scale/stop_boundary_train_preflight.json)
checks every source rollout and per-turn distance continuity.

Recomputing the frozen 0.1-penalty reward on those same n=4 training
rollouts changes 2,467/2,350/2,420 of 23,041/23,201/22,754 turns
for seeds 11/22/33. In 102/98/97 all-failure groups, at least one turn
changes reward. The [signal recount](ordinal_progress/policy_preference/oracle_exact512_scale/stop_boundary_signal_preflight.json)
is an offline check of reward variation, not evidence of improved policy
behavior. The [reward code](stop_boundary_reward.py),
[environment patch](stop_boundary_env.patch), and fail-closed
[training audit](audit_stop_boundary_train.py) pin the mechanism. The
isolated source on the server is staged; the ongoing full val-unseen
evaluation must finish before its GPU-dependent pilot starts. The
[gated launcher](run_stop_boundary_after_full.sh) first checks a two-step
same-row n=4 smoke, then a 64-step pilot. For the fixed 256-episode
development screen, it revalidates and reuses the exact matched
seed-11 control's existing rollout on the identical manifest and
checkpoint, so only the new candidate needs inference.

## Proposed mechanism and order of tests

1. **Privileged mechanism pilot.** Test a goal-boundary version of the
   existing turn-wise n=4 advantage. Let
   $\Phi(d)= -\max(d-3,0)/\max(d_0,3)$ for simulator geodesic distance
   $d$ and episode start distance $d_0$. For a generated movement turn,
   use $\Phi(d_t)-\Phi(d_{t-1})$ and subtract a fixed 0.1 when the turn
   begins within 3 m and continues moving. STOP and observation tokens
   receive zero auxiliary credit; successful groups retain ordinary
   outcome GRPO. This removes an incentive to move deeper inside the
   success region and discourages continued movement there. Check
   shape, signs, STOP masking, all-failure group contrast, and nonzero
   gradients in a two-step smoke before a 64-step n=4 pilot against a
   same-row, same-initialization n=4 outcome control. Do not change
   the 0.1 penalty after viewing pilot navigation outcomes. Positive
   paired SR **and** paired SPL on the fixed 256-item screen is
   required before a three-seed 128-step scale; report the screen as
   reused development data.
2. **Observation-only representation.** Independently freeze a
   scene-disjoint R2R-train source of near-goal crossings and matched
   just-outside negatives before RGB rendering. Replay only the
   selected before/after states. Train a goal-region occupancy head
   from instruction, available RGB history, and executed actions,
   with within-route crossing order and correct/wrong-instruction
   contrasts. Geodesic labels may define supervision and audits but
   must never enter model input or inference. Require exact replay
   coverage, at least 50 independent positive episode IDs in a held
   development partition, pooled FPR at most 5%, recall at least 55%,
   wrong-instruction FPR at most 12%, and near-failure recall at least
   50% before opening a separately frozen scene audit. These preserve
   the previous STOP-model gates; an area-under-curve score alone is
   insufficient. If source coverage fails, collect more train-only
   histories before fitting rather than weakening a threshold.
3. **Learned turn-wise reward.** Only if both the privileged pilot and
   observation-only representation gates pass, replace the boundary
   indicator by a calibrated high-confidence occupancy estimate in
   the same n=4 turn-wise reward. Center four active all-failure
   continuations at each turn, clip the auxiliary magnitude, and keep
   STOP free of auxiliary credit. Run a matched 64-step pilot first;
   expand to three seeds and full 1,839-episode val-unseen only after
   positive paired SR and SPL. An n=8 run is a small matched-budget
   sensitivity check after an n=4 learned-reward gain.

Use the existing completed rollout JSONL for source selection and
labels. Reuse any verified RGB states before rendering new ones. The
current full-val evaluator owns GPUs 0/1, so no Habitat or policy job
should disturb that run; CPU source checks can proceed concurrently,
and a separate GPU 2/3 representation fit may start only after
checking memory and service isolation. No result from the reused
val-unseen screen can be called an independent final test.
