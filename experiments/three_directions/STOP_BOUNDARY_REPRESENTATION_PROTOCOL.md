# Stop-boundary representation and reward: next exploratory hypothesis

Status (2026-10-05): **training-data coverage, exact fit/development RGB
replay, fixed-budget observation-only representation fit, offline
reward-signal preflights, and the matched n=4 pilot completed. Both the
representation development gate and navigation pilot gate failed.** This idea follows the failed
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
isolated source on the server is staged; the full val-unseen evaluation
has finished and released its GPUs. The
[gated launcher](run_stop_boundary_after_full.sh) first checks a two-step
same-row n=4 smoke, then a 64-step pilot. For the fixed 256-episode
development screen, it revalidates and reuses the exact matched
seed-11 control's existing rollout on the identical manifest and
checkpoint, so only the new candidate needs inference.

The audited 64-step pilot used 256 identical group-four training rows,
with nonzero actor gradients at all 64 steps, 149 active all-failure
groups, and 902 inside-boundary continuation penalties. On the fixed,
reused 256-episode val-unseen development screen it reaches 67 successes
versus 72 for the unchanged control: paired SR $-1.95$ and SPL $-1.62$
points, zero inference errors. The prespecified positive SR-and-SPL
gate fails. No three-seed scale or learned-reward substitution is
launched from this mechanism. The [compact package](ordinal_progress/policy_preference/stop_boundary_pilot/)
contains the paired episode export, exact manifest, validators, both
training audits, and an independent recount. It does not identify why
the mechanism fails.

## Train-scene RGB source gate

An independent observation-only occupancy source has now passed a
**CPU-only coverage check**. Before inspecting RGB, we assigned the 54
R2R-train scenes represented in the three audited n=4 rollouts by
SHA-256 of `boundary-occupancy-source-v1:` plus scene ID: the first eight
are development, the next eight are reserved audit, and the other 38
are fit. Each selected trajectory has a state 3.5--4.5 m from its
goal followed within two generated turns by a state within 3 m. We
select only the first such pair per trajectory. The fit/development/audit
parts contain 1,184/288/271 trajectory pairs from 260/62/59 unique
episode IDs. Development and audit exceed the frozen 50-positive-ID
minimum; fit exceeds 150. Across the three parts, 155/41/45 unique
episode IDs also have a natural, exact-same-start wrong instruction
whose goal is more than 7 m Euclidean from the true goal. This distance
guarantees that the alternative goal is more than 4 m from an inside
state, without a second Habitat geodesic query. All fixed source
coverage checks pass.

The [source report](ordinal_progress/policy_preference/boundary_occupancy_source/preflight.json)
pins the split, SHA-256 source identities, counts, and record IDs.
The [RGB replay manifest](ordinal_progress/policy_preference/boundary_occupancy_source/replay_manifest.json)
contains instructions and selected turn indices but no privileged
distances; the [labels](ordinal_progress/policy_preference/boundary_occupancy_source/privileged_labels.json)
are separate. The model must receive only RGB, the true or wrong
instruction, and available action history. The previous sparse RGB
cache supplied both needed boundary frames for zero selected records.
The new replay has now captured and independently checked all 1,184 fit
and 288 development pairs, with 2,368 and 576 JPEGs, respectively.
The exact source-state and terminal geodesic drift maximum is 0.0 m;
there were no replay errors. The [compact full verification](ordinal_progress/policy_preference/boundary_occupancy_replay/full_verification.json)
and [one-record smoke](ordinal_progress/policy_preference/boundary_occupancy_replay/smoke_verification.json)
pin the manifest SHA. Audit-scene RGB remains unopened. No classifier,
reward, or navigation result follows from RGB source integrity alone.

The [collector](collect_boundary_occupancy_frames.py) replays the frozen
train-only trajectories, saving only the outside/inside RGB views,
instruction contrast, and motion history as model input. Its separate
audit records check replayed geodesic distance against the source at
both states and at termination. The [independent verifier](verify_boundary_occupancy_replay.py)
checks exact record coverage, the frozen label correspondence, JPEG
integrity, and absence of simulator labels from the model input. The
[GPU-locked runner](run_boundary_occupancy_replay.sh) starts with a
single-record smoke test, then uses four Habitat shards on idle GPU 1.
This is representation data preparation, not a navigation result.

The first fit-only [occupancy scorer](train_boundary_occupancy_lora.py)
starts from the local navigation SFT Qwen2.5-VL-3B, adds a scalar head
to its last hidden state, and updates LoRA on `q_proj`/`v_proj`. A
single-state prompt contains the current RGB, instruction, and at most
four recent motion turns, with no boundary role, absolute turn index,
geodesic distance, or STOP label. The fixed loss ranks an inside state
above its outside state and, for exact-same-start far-wrong instructions,
ranks that same inside image with the correct instruction above the
wrong one. Scene/episode-balanced fit sampling alternates the full
fit set with the exact-start instruction-contrast subset. The fixed
budget is 768 microsteps, accumulation four, with checkpoint selection
at steps 256/512/768 on a hashed eight-record-per-scene development
subset by pooled AUC, crossing rank, then earlier step. The selected
checkpoint receives one complete development evaluation and one
threshold chosen to maximize recall under 5% pooled negative FPR.
No audit data are loaded. A four-microstep smoke produced one finite,
nonzero-gradient update over the exact manifest; it loaded only fit
records. The full fit used the same GPU-1 lock. The
[launcher](run_boundary_occupancy_lora.sh) pins inputs and model path.

The full 768-microstep fit finished and selected step 512 from the three
fixed checkpoints. On all 288 development trajectories (62 episode IDs,
eight held scenes), its pooled AUC is 0.6053. At the single threshold
selected under the 5% pooled-negative FPR cap, recall is **8.33%** at
4.81% FPR; wrong-instruction FPR is 8.79% and near-failure recall is
**5.04%**. The prespecified 55% and 50% recall gates fail. Episode-macro
positive recall is 6.11%; crossing and instruction order accuracy are
78.82% and 56.41%. The [per-record scores and report](ordinal_progress/policy_preference/boundary_occupancy_lora/)
were independently recounted by
[`verify_boundary_occupancy_development.py`](verify_boundary_occupancy_development.py).
A fixed text-and-motion-only logistic shortcut, which never opened an
image, reached AUC 0.5884, 6.94% recall at its 5% FPR cap, 81.94%
crossing-order accuracy, and 56.04% instruction-order accuracy. The
selected visual model's AUC advantage over that shortcut is only 1.69
points. Permuting all 576 development images across states within the
same scenes lowered the visual model's AUC to 0.5897 and crossing-order
accuracy to 74.31%. These controls show a small image contribution but
do not establish reliable instruction-grounded goal occupancy. The
audit scenes stay unopened; this checkpoint is ineligible for online
reward or val-unseen navigation evaluation. The fixed gate is not
weakened or retried by changing a threshold on this development screen.

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
   from instruction, current RGB, and available executed-action history,
   with within-route crossing order and correct/wrong-instruction
   contrasts. Geodesic labels may define supervision and audits but
   must never enter model input or inference. Require exact replay
   coverage, at least 50 independent positive episode IDs in a held
   development partition, pooled FPR at most 5%, recall at least 55%,
   wrong-instruction FPR at most 12%, and near-failure recall at least
   50% before opening a separately frozen scene audit. These preserve
   the previous STOP-model gates; an area-under-curve score alone is
   insufficient. Fit one state scorer without an `outside`/`inside`
   role token; at scoring time it sees the current RGB, the chosen
   instruction, and action history available through that state. For
   the wrong-instruction negative, score the identical inside RGB and
   identical motion history with only the instruction changed. Pick the
   checkpoint and one conservative threshold using development scenes
   alone. Report per-episode and per-scene rates alongside the pooled
   gates, since multiple sampled routes can share an episode. Freeze
   the selected checkpoint and threshold before opening audit scenes.
   Evaluate a history-only scorer and a fixed image-shuffle control on
   the same development records to expose shortcuts; neither may be
   mistaken for evidence of visual grounding. If source coverage
   fails, collect more train-only histories before fitting rather than
   weakening a threshold.
3. **Learned turn-wise reward.** Only if both the privileged pilot and
   observation-only representation gates pass, replace the boundary
   indicator by a calibrated high-confidence occupancy estimate in
   the same n=4 turn-wise reward. Center four active all-failure
   continuations at each turn, clip the auxiliary magnitude, and keep
   STOP free of auxiliary credit. Run a matched 64-step pilot first;
   expand to three seeds and full 1,839-episode val-unseen only after
   positive paired SR and SPL. An n=8 run is a small matched-budget
   sensitivity check after an n=4 learned-reward gain.

The privileged pilot in item 1 has failed its fixed navigation gate,
so item 3 is closed for that reward formula. Item 2 remains an
independent offline representation study. Any later online use requires
a distinct, frozen reward mechanism with its own same-budget n=4 pilot;
it cannot inherit the failed penalty pilot's evidence.

Use the completed rollout JSONL for source selection and labels. Check
the shared GPU lock and current GPU use before replay or fitting. No
result from the reused val-unseen screen can be called an independent
final test.
