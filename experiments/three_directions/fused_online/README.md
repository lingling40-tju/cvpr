# Frozen fusion reward: online group-four pilot

This directory records an isolated online test of the frozen reward
representation. The working server tree is
`/Knowin/foundation/haozhiwang/whz/ActiveVLN_fused_reward_20261003` on
`wanghaozhihuoshanyun`; it was copied from the 2026-10-02 three-directions
tree before `apply_env_patch.py` modified three checksum-guarded files.
The source experiment and its completed control checkpoints were not
modified.

The reward server loads the frozen Qwen2.5-VL-3B navigation-SFT model,
the v2 causal temporal encoder, and the SigLIP LoRA adapter selected at
step 768 with fixed 64-token text padding. A trajectory's final score is

```text
u = 0.5 * (temporal_potential / fit_temporal_endpoint_scale
           + image_instruction_similarity / fit_visual_endpoint_scale)
bonus = sigmoid(u)
```

The two scales are frozen from fit-scene margins in
`../ordinal_progress/policy_preference/equal_fused_reward_online_parity_development.json`.
One reward request is made at episode termination. The Habitat wrapper
adds the bounded bonus to the existing destination reward and success
floor; it fails the run if the reward service returns an error. The actor
observation and optimizer remain unchanged. The temporal encoder sees
four evenly spaced **turn observations**. Offline trajectory probes used
four evenly spaced **action observations**, so online temporal inputs
have a known sampling difference. Both use the same 336-pixel JPEG
encoding, navigation-SFT prompt, fixed SigLIP text length, and
single-image inference shape. A replay-pair service check matched the
offline fused margin within `4e-5` after these fixes.

`start_services.sh` places Habitat on GPU0/port5021 and the frozen reward
model on GPU1/port8021. `run_group4_pilot.sh` trains the candidate on
GPU2/3 using group size four, seed 11, and the exact
`branch_pilot_train.parquet` used by the existing outcome-only
`three_directions_group4_64step_seed11` control. `audit_pilot.py`
checks training-row pairing, one reward request per rollout, bounded
rewards, rollout diversity, and nonzero actor gradients. The two-step
wiring pilot passed: 8 matched training episodes, 32 rollouts, 32 reward
requests, 8 diverse groups, bonus range 0.379--0.733, and actor gradient
norms 4.429 and 1.416 (`two_step_training_audit.json`). This is a
training-wiring check, not a held-out navigation result.
The 64-step training audit subsequently passed: 256 matched unique
training episodes, 1,024 group-four rollouts, exactly 1,024 reward
requests, and 256 diverse groups (`paired_train_audit64.json`).

`analyze_onpolicy_signal.py` reads completed training rollout rows and
separately counts how often each frozen representation ranks a successful
rollout above an unsuccessful one, or a failed rollout ending closer to
the goal above another failed rollout of the same episode. Simulator
distance is used only as this diagnostic label, never as reward input.
These training-set rankings diagnose whether reward information reaches
the policy; they cannot substitute for held-out navigation evaluation.
At 64 steps, 162/256 episode groups had no successful rollout. In those
groups, the current fusion bonus ranked the nearer failed rollout above
the farther one in 680/1114 comparable pairs (61.0%). This indicates
that improving all-failure group ranking could affect many group-four
updates, while also showing that the current signal is noisy
(`onpolicy_signal64.json`).

For a possible failure-aware representation follow-up,
`prepare_failure_rank_manifest.py` freezes 928 pairs from the completed
128-step, group-four outcome-only training rollouts (seeds 11, 22, 33).
Each pair contains two unsuccessful rollouts for the same instruction;
both finish at least 3.5 m from the goal and their simulator distances
differ by at least 1.5 m. The scene-disjoint split has 697 fit, 125
development, and 106 audit pairs across 56 train scenes. The frozen
manifest is on the experiment server at
`runlogs/failure_rank/manifest.json`; its compact repository record is
`failure_rank/manifest_summary.json` (manifest SHA-256
`139b74b4d10c825cef47f3e1451e253923e646bc4a9ddb1e2e4182bbb45f20d0`).
`collect_failure_rank_frames.py` replays both trajectories, verifies
their terminal simulator distances, and samples the same initial/turn
observations used by the online reward.
The 10-pair replay smoke completed all 20 trajectories without drift.
`run_failure_rank_replay.sh` completed all 1,856 trajectories on GPU1
while the main group-four training used GPU0/2/3; the independent
`failure_rank/replay_summary.json` records zero replay errors.
`cache_failure_rank_sft.py` uses the exact navigation-SFT prompt and
initial-frame flags to cache four hidden states per trajectory. Its
two-record/eight-frame smoke passed; `run_failure_rank_cache.sh` cached
all 7,424 frames on GPU1 (`failure_rank/cache_summary.json`). These
jobs did not change the ongoing 64-step policy or its frozen reward.
`train_failure_rank_encoder.py` starts from the frozen v2 temporal
encoder and uses only fit-scene failed pairs to improve near-over-far
terminal ranking, with a small preservation loss to limit drift. It
selects an epoch on development scenes, requiring at least 65% ranking
accuracy and at least 5 percentage points above the frozen v2 reference
before the separate audit is opened. `run_failure_rank_training.sh`
ran that fit/development stage without reading the audit partition.
The selected epoch was 4: development near-over-far ranking rose from
76/125 (60.8%) for frozen v2 to 90/125 (72.0%) for the new encoder
(`failure_rank/development.json`).
`run_failure_rank_audit.sh` opens the new 106-pair scene audit only if
both development gates pass. `audit_failure_rank_encoder.py` compares
the candidate with the frozen v2 encoder on those identical pairs, and
also measures retention on the reused 48-pair v2 success/grounding
audit. Its declared audit gate requires at least 65% failed-pair
ranking, at least 5 percentage points over v2, and no more than a
5-point drop on either reused task. Passing these representation gates
was met: new-scene failed-pair ranking rose from 69/106 (65.1%) to
79/106 (74.5%). On the reused older 48-pair audit, success-over-failure
ranking stayed 38/48, while instruction grounding went from 31/48 to
30/48 (`failure_rank/audit.json`). The paired failed-pair ranking gain
is 9.43 percentage points; a scene-cluster bootstrap across eight audit
scenes gives a 95% interval of +1.54 to +15.25 points. The new
106-pair audit is scene-disjoint from the new fit/development split.
The reused 48-pair success/grounding audit overlaps the new fit split
in five scenes, so it is only a retention diagnostic. These are
**train-scene representation** checks, not online RL or val-unseen
navigation gains.

`calibrate_failure_rank_reward.py` set the new temporal scale to 1.01536
using only fit-scene failed-pair and successful-pair margins. Directly
substituting the new encoder into the old terminal fusion was not robust:
on the 38 seed-33 novel train-scene probes, success-over-failure
ranking fell from 30/38 to 25/38 and instruction grounding from 36/38
to 33/38 (`failure_rank/fusion_probe.json`). These probes share train
scenes and cannot establish generalization, but the regression rules
out treating the failed-pair representation gain as a ready online
reward improvement. The old fusion pilot continues unchanged.

`screen_failure_residual_blend.py` therefore tested normalized mixtures
of old and new temporal scores on fit/development scenes. Its
development-only rule selected 25% new representation: failed-pair
ranking improved from 76/125 to 83/125, while the 52 successful-pair
development examples improved from 37/52 to 41/52
(`failure_rank/residual_blend_development.json`). Five of those eight
older success-development scenes overlap the new encoder's fit scenes,
so this retention screen is not independent. On the reused fusion
probes this blend restored success ranking to the old 41/48 and 30/38,
but instruction grounding remained worse (39/48 to 35/48 and 36/38 to
34/38). These reused probes were inspected after the pure model audit,
so the blend is exploratory. A stronger reward design must preserve
instruction grounding before it merits an online group-four trial.
`train_multitask_rank_encoder.py` then used the cached fit-scene
success, failure, and swapped-instruction features in a joint loss,
screening 20 epochs against a development gate that required a
five-point failed-pair gain with at most two-point losses on successful
pair and instruction ranking. No epoch passed all three gates:
grounding dropped whenever failed-pair ranking improved enough
(`failure_rank/multitask_development.json`). No multitask checkpoint was
selected. A possible next reward variant confines the failure-aware
bonus to unsuccessful rollouts, keeping successful rollout rewards
anchored by the existing outcome reward; this needs an online matched
test before any benefit is claimed.

The 64-step candidate was launched on 2026-10-03. Its watcher
(`run_followup_watcher.sh`) audits the completed training and then runs
the fixed 256-episode val-unseen screen using `run_eval256.sh`, paired
against the existing group-four control. `run_scale_conditional.sh`
predeclares the larger-budget gate: exactly 256 episodes, 11 scenes,
zero inference errors, and strictly positive paired SR and SPL. If the
gate passes, it trains group-four fusion-reward candidates for 128 steps
with seeds 11, 22, and 33 against the already completed outcome-only
controls. The training job saves at steps 64 and 128 for recovery.
Afterward, it releases the two online services and evaluates the three
candidates on the same complete 1839-episode val-unseen manifest using
two GPU pairs in parallel. It reuses all three cached control
evaluations, validates exact episode coverage, and computes paired
metrics with `analyze_scale.py`. If the screen fails the gate, the scale
job ends without spending that training budget. The complete fixed-256
screen **failed**: against the matched group-four outcome-only control,
the frozen fusion candidate had SR 68/256 versus 80/256 (paired
−4.6875 percentage points) and SPL 0.25957 versus 0.30295
(paired −4.3379 percentage points). Both covered 256 unique
episodes in 11 scenes with zero inference errors
(`paired_fused_vs_group4_eval256.json`). The conditional scale suite
recorded `no_pilot_gain` and did not launch the 128-step three-seed run.

After that decision, the isolated environment was changed with the
checksum-guarded `apply_failure_only_env_patch.py`; the source tree and
completed fusion experiment were left as recorded. In this follow-up,
successful trajectories receive their original outcome reward with no
extra reward request. Only unsuccessful trajectories call
`failure_only_reward_server.py`, which returns a bounded sigmoid of the
new failure-aware temporal score divided by its fit-scene scale. On four
replayed fit-scene trajectories, live service and cached-feature bonus
agreed within `1.1e-5` (`failure_only_service_parity.json`). This is a
service consistency check, not a navigation result.

`run_failure_only_pilot.sh` keeps group size four, seed 11, the identical
training dataset and 64-step budget, GPU0 Habitat, GPU1 reward service,
and GPU2/3 actors. The two-step wiring smoke runs first.
`run_failure_only_followup.sh` audits it and, only on success, trains the
64-step candidate and evaluates the same fixed 256 val-unseen episodes
against the cached group-four control. `audit_failure_only_pilot.py`
requires one reward request per unsuccessful rollout and zero failure
bonus on successful rollouts. No failure-only navigation gain can be
claimed until its matched val-unseen evaluation completes.
The two-step smoke passed: 32 matched rollouts across eight diverse
groups, nine successes with no extra bonus, and exactly 23 reward
requests for the 23 unsuccessful rollouts
(`failure_only_two_step_audit.json`). The 64-step run has started.
`run_failure_only_scale_conditional.sh` waits for its fixed-256 result
and will use the three-seed 128-step/full-1839 budget only if coverage
and zero-error checks pass and both paired SR and SPL are strictly
positive. It reuses the completed outcome-only controls.
