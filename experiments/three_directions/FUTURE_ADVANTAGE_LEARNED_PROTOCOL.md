# Observation-only future-advantage reward: frozen next test

Status (2026-10-05): **protocol and staged code only**. No real sparse
RGB replay, model fit, prospective audit, online reward, or navigation
evaluation has run for this candidate. The training-only privileged
turn-wise oracle gives a positive seed-11 full-1,839 mechanism result,
but seeds 22/33 are still running. This test asks whether the useful
signal can be predicted from observations rather than simulator state.

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
wired into live rollouts. Groups containing any successful
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
verifier before marking completion. This launcher is staged but has not
run on real gated data.
