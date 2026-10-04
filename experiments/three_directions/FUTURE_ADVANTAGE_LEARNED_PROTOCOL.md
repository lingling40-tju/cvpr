# Observation-only future-advantage reward: frozen next test

Status (2026-10-05): **protocol and staged code only**. No real sparse
RGB replay, model fit, prospective audit, online reward, or navigation
evaluation has run for this candidate. The training-only privileged
turn-wise oracle gives a positive seed-11 full-1,839 mechanism result,
seed 22 has finished training and begun full evaluation, and seed 33 is
training. This test asks whether the useful signal can be predicted
from observations rather than simulator state.

The frozen pooled coverage gate **passed** on audited seeds 11 and 22,
and the sparse replay manifest and separate pair labels were generated.
The [seed-22 training audit](ordinal_progress/policy_preference/oracle_exact512_scale/candidate_seed22_train_audit.json)
confirms 128 n=4 updates on the same 512 train rows as its control,
with nonzero candidate actor gradients at 128/128 steps versus 118/128
for the outcome-only control. Seed 11 has 128/128 versus 115/128.
These are training-signal diagnostics, not navigation outcomes.
The fit split has 247/236 unique episode groups and 1,349/1,081 pairs
at anchors 3/6; the scene-disjoint development split has 52/50 groups
and 258/211 pairs. Same-terminal-mode development coverage is 47 groups
at each anchor, with 149/122 pairs. The manifest selects 1,490 fit and
305 development trajectory records; 37 fit scenes and eight development
scenes are disjoint. Source rollout/audit hashes and manifest/label links
were independently checked. The compact [pooled report](ordinal_progress/policy_preference/future_advantage_pooled/report_seed11_22.json)
records the thresholds and counts. These numbers show label coverage
only, not learned reward or navigation quality. GPU-1 full evaluation
currently prevents the real RGB replay and LoRA fit.

## Fixed source and model input

Use the staged exact512 n=4 future-advantage coverage gate. If the
audited seeds 11+22 pass, freeze them as the initial fit/development
source and reserve seed 33 as a separate policy-rollout stress check.
Otherwise add seed 33 only if the unchanged coverage thresholds pass.
Do not create cross-seed pairs or count an episode ID more than once
for a sample-size claim. A failed three-seed gate stops this method
before RGB replay.

`prepare_future_advantage_sparse_manifest.py` selects all-failure,
same-episode pairs with four sampled routes and an absolute future
oracle return gap of at least 0.25 at turns 3 or 6. The pair labels
are separate from the replay manifest. The RGB collector replays every
executed action to check geodesic drift but saves only the initial
view and requested turn-3/turn-6 views. A turn-6 label requires both
turn-3 and turn-6 frames even when that route has no turn-3 label;
the frozen label counts are not altered by this capture dependency.
The model receives **only**
the nested `input`: instruction, available views, and executed action
history through the anchor. It cannot receive simulator distance,
goal coordinates, future actions, episode IDs, scene IDs, or terminal
mode. The standalone `future_advantage_visual_input.py` fixes this
prompt and its image processing before the pooled source is available
(SHA-256 `9d2b44353750106901a4c1c88bd07a382a6e7bfa3a0e0b0c6df7f8c63834d70c`).
The same code accepts in-memory live Habitat views without image-file
round trips. A real Qwen processor comparison found identical input IDs,
pixel tensors, and image grids for offline versus live synthetic prefixes
at anchors 3 and 6; a future frame, future action, and privileged distance
field were rejected.
Before a real LoRA fit, `verify_future_advantage_sparse_replay.py` must
recompute the passed coverage gate and check every selected record,
JPEG, prefix history, paired label count, and separate terminal-distance
audit. The trainer calls this verifier before loading the model and
saves its report; a missing or corrupt replay blocks fitting. A small
synthetic fixture passed and rejected a privileged input field and a
missing intermediate image. It is a wiring test, not real replay data.
The real Qwen2.5-VL processor accepted CPU synthetic examples with
two images/118 tokens at turn 3 and three images/166 tokens at turn 6;
an injected future action was rejected. The staged scalar-head/rank-eight
q/v LoRA trainer subsequently completed six synthetic GPU-1 microsteps
with finite loss and gradients in 4.67 seconds after model load. These
checks establish input and gradient wiring only; no real replay or
learned-reward navigation result exists yet.
After the offline/live input refactor and mandatory replay verifier were
added, the same six-microstep GPU-1 synthetic check passed again in
4.85 seconds after model load, with one optimizer update. Its raw log
and source hashes are saved under
`ordinal_progress/policy_preference/future_advantage_smoke_v2/`.

## Representation and offline decision

Use the navigation SFT Qwen2.5-VL-3B as the initializer. Fit one
rank-eight q/v LoRA and a LayerNorm–128 hidden unit–scalar head to
score each prefix independently. At each update, sample an underlying
fit episode uniformly, then one of its seed-specific pairs, alternating anchors 3 and
6. Fit only **same-terminal-mode** pairs so a score cannot pass by
learning whether a rollout ends in STOP or timeout. Use a logistic
pairwise ranking loss on the sign of the future geodesic return gap,
plus a 0.001 score-magnitude penalty; each underlying episode gets
equal total weight. The target is a ranking, never the true distance
or raw future return at inference. Use 1,024 microsteps, accumulation
four, AdamW learning rates 5e-5 for LoRA and 1.5e-4 for the head,
weight decay 0.01, and a fixed final checkpoint. A six-microstep
gradient/finite-loss smoke precedes the fit. Do not select a checkpoint
or tune its threshold on the reused development scenes.

At the fixed checkpoint, evaluate **episode-macro** and scene-macro
pairwise accuracy separately at anchors 3 and 6, both on all
qualifying pairs and on equal-terminal-mode pairs. The first gate
requires at both anchors: at least 70% all-pair and 70% same-mode
episode-macro accuracy, at least 65% scene-macro accuracy, and at
least a five-point episode-macro advantage over the corresponding
commanded-forward-distance action baseline computed on the *same*
pairs. Report the individual direction and termination-mode cells,
sample counts, and seven-scene uncertainty. This is a go/no-go
development gate, not a claimed expected score. If it fails, no
prospective audit or online n=4 RL run follows this candidate.

If it passes, test instruction dependence with a **separate**
scene-disjoint, start-matched goal-swap audit: retain only examples
where label-only Habitat replay verifies that the correct and
alternative goals reverse the route ordering. Require at least 50
underlying episode IDs in each reversal direction and 75% macro
preference for the correct instruction. The frozen 123-episode,
seven-scene prospective train manifest remains unopened until this
point. The candidate must also pass the same pairwise 70% criterion
on that audit before policy training. If goal-swap coverage is too
small, stop; do not drop the instruction check to admit the model.
These are train-scene audits, not an independent val-unseen test.

A CPU-only [source preflight](ordinal_progress/policy_preference/future_advantage_pooled/goal_swap_source_preflight.json)
counted exact start-pose matches with goals at least one meter apart,
without reading route outcomes, images, or model predictions. In the
eight reserved train audit scenes, the full dataset has 90 eligible
start groups and 656 episode IDs with an alternative goal. The existing
512-row policy training subset covers only 14 such start groups and 38
IDs with a second selected goal; 68 selected route IDs have *some*
alternative instruction in the full dataset. Thus an audit confined to
the existing rollouts is likely too small. An [ID-only manifest](ordinal_progress/policy_preference/future_advantage_pooled/goal_swap_id_manifest.json)
has now been frozen **before model scoring**: it selects the farthest
distinct-goal episode pair at each eligible start, 90 pairs and 180
unique IDs across those eight audit scenes. If the learned model passes
its development gate, collect four SFT-policy routes per ID (720 planned
rollouts) and perform separate label-only geodesic reversal replay.
There is no collection or reversal result yet. The seven prospective
scenes remain reserved. Metadata coverage is a necessary upper bound; it neither
establishes geodesic route reversal nor relaxes the 50-ID-per-direction
and 75% gates.

## Online n=4 reward and resource sequence

For a passing model, query only active prefixes at turns 3 and 6.
Within each four-rollout episode group, center the four scalar scores
and clip the auxiliary value to [-0.25, 0.25]. Apply it only to that
turn's generated movement-action tokens; observation tokens and STOP
receive zero auxiliary credit. If fewer than four routes remain active
at an anchor, skip that anchor for the whole group; do not center an
unequal subset or propagate the score to earlier turns. The staged
`future_advantage_group4_reward.py` passes synthetic group-four,
partial-anchor, token-mask, and mixed-outcome checks. It is not yet
wired into live rollouts. A separate
`future_advantage_activevln_adapter.py` now also passes a synthetic
ActiveVLN-style action-span test: only movement tokens at turns 3/6
receive all-failure auxiliary credit, mixed-success groups keep the
ordinary outcome signal, and a score without a post-anchor view is
rejected. The adapter is staged only; no live scorer, trainer patch,
or two-step online audit has run. Groups containing any successful
rollout retain the ordinary destination outcome advantage. An
all-failure group may use the clipped process signal; the frozen
model is never updated from policy rollouts. Record teacher-query
count, score coverage, group balance, STOP masking, and gradients in
a two-step n=4 wiring audit before a 64-step pilot. No claim of
policy-invariant shaping is made.

The 64-step pilot uses seed 11, the first 256 exact512 train rows,
four rollouts per episode, and the same optimizer/base model as the
audited outcome control. Reuse that control's step-64 checkpoint only
after checking its source/configuration and ordered train rows;
otherwise train a matched control. Evaluate both on the already
frozen, previously unused 256-episode fifth val-unseen development
screen (`qwen_group4/next_val256_manifest.json`, SHA-256
`e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1`).
Require strictly positive paired SR **and** SPL with exact ID
coverage and zero inference errors. A positive screen gets one full
1,839-episode recheck with the 1,583 screen-excluded episodes reported
separately; only a positive full/complement result warrants a matched
three-seed 128-step n=4 scale. The full val-unseen set has been used
in prior development and is not an independent test.

While the privileged n=4 scale trains on GPUs 2/3 with Habitat on
GPU 0, use CPU for manifest and audit checks. Sparse Habitat replay
and LoRA fit use GPU 1 only when the overlapping full evaluator has
released it; an inference/evaluation lane may be run concurrently
only if GPU memory and simulator throughput are measured to remain
stable. Each gate records wall time, GPU hours, trajectories, saved
frames, and teacher queries. A group size above four is reserved for
a small, separately controlled sensitivity diagnostic after a learned
n=4 navigation gain is confirmed.

`run_future_advantage_sparse_replay.sh smoke` targets one fit trajectory
with a turn-6 label after the selected seeds' full evaluation has
released GPU 1; it checks three images plus geodesic replay drift.
Its `full` mode waits for the current full-evaluation watcher to
finish before using GPU 1, and can split each scene-part replay into
1--4 independent shards using `VLN_SPARSE_SHARDS`. Shard count is chosen
after timing the small replay; the full mode runs the exact-coverage
verifier before marking completion. The real one-record replay is still
waiting for full evaluation to release GPU 1. Its launcher now samples
GPU-1 memory during the smoke and records wall time, baseline, peak,
and capacity in `resource.json`.
`run_future_advantage_sparse_fit.sh` then requires that verified full
replay and the completed evaluation lane, pins its source hashes, and
runs the fixed 1,024-microstep GPU-1 fit. It marks a negative
development result as a completed, failed gate rather than a process
crash. It is also staged only; neither its real fit nor its development
score exists yet.
The waiting `run_future_advantage_pipeline_after_smoke.sh` uses the
measured incremental smoke memory to select four, two, or one replay
shards while reserving 30% of GPU-1 capacity. It runs full replay only
after the smoke audit passes, and runs the fixed LoRA fit only after
full replay verification. A failed fit development gate is recorded as
a negative scientific result and does not launch an online policy.

## Fixed n=4 validation and resource budget

Group size **4** is the primary algorithm comparison. Keep the same
episode IDs, seed, model initialization, train rows, optimizer schedule,
and exact evaluation manifest for each reward/control pair. Pair outcomes
by episode before aggregating SR and SPL. Never interpret a larger group
alone as an algorithmic gain. An n=8 run is only a later, small
sensitivity diagnostic if the learned n=4 reward first improves both
paired SR and SPL; hold generated trajectory budget and evaluation IDs
fixed for that diagnostic, and report its different update count and
training-episode coverage.

The observed 2026-10-04 evaluation wall times give a practical budget:
the 256-episode frozen screen took 11m38s for its control and 12m22s
for its candidate, while completed 1,839-episode control/candidate
evaluations took 57m56s and 62m24s. These are measured launcher
durations on this host, not speed guarantees. The staged gate therefore
does cheap CPU/source checks first, then one fixed 256-episode n=4
pilot. A nonpositive paired SR or SPL ends that candidate before any
1,839-episode evaluation. A positive pilot gets one full evaluation,
with the 1,583 screen-excluded episodes reported separately; a positive
full/complement result admits the three-seed n=4 scale. Do not reuse
the screen to tune a candidate after seeing its outcome.

Keep one training lane on GPUs 2/3 and Habitat on GPU 0. During that
training, CPU-only manifest construction, integrity checks, and paired
analysis can run concurrently; a GPU-1 model/Habitat evaluation can
overlap only while observed memory and throughput remain stable. Run
sparse RGB replay and the fixed LoRA fit on GPU 1 after the overlapping
evaluation releases it. Time one replay record before choosing one to
four independent replay shards. Send the four route-prefix scores in
one request at each active anchor and query only turns 3 and 6; the
current scorer forwards uncached items sequentially rather than doing
a GPU tensor batch. Its bounded 4,096-prefix LRU uses an exact
instruction/image/action hash and a synthetic duplicate-prefix test
confirmed one model call for two identical requests. Record actual
model-call count, cache hits, and score latency so the reward's
extra compute is visible. `future_advantage_score_server.py` provides a
staged local-only, observation-field-checked score endpoint; its
synthetic JPEG request checks passed, but no real checkpoint has been
loaded and no live reward service has run.
The CPU-only `future_advantage_live_prefix.py` applies the same
336-pixel thumbnail and quality-82 JPEG encoding as the actual sparse
Habitat collector. A synthetic three-shape test compared its bytes to
the collector and found exact equality; future inputs and STOP were
rejected. The final checkpoint pins this encoder's source hash, and the
score service checks it before loading model weights. This verifies
input construction, not online reward quality.
