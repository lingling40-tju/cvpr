# Post-development turn-3-only potential: frozen train-scene audit

Status (2026-10-05): **ID/action-length source frozen; audit RGB,
geodesic pair labels, and model scores remain unopened**. The
two-anchor anchor-potential development gate
failed because turn 6 lacked a same-terminal-mode and action-baseline
margin. The same fixed checkpoint did pass turn 3 on a previously used
development split. Restricting reward to turn 3 is a *post hoc* new
hypothesis, not a passed result for the original two-anchor method.

Pin the already fitted checkpoint at remote path
`runlogs/anchor_distance_potential_lora/full/adapter_head.pt`, SHA-256
`0b955f0cde2d77a89f48f0d72e2346c65afaf1b46bd1578d716caf4ae5ef3729`.
Do not update its weights, prompt, scaling, or threshold after the
audit. Its model input at turn 3 is the instruction, initial and
turn-3 RGB views, and the first three executed action turns; no
simulator field enters the model.

The audit source is the existing seed-11/22 n=4 training rollouts in
the eight frozen audit scenes of
`ordinal_progress/policy_preference/stop_history_lora_scene_split.json`
(SHA-256 `c82e495e07f8eb0ed01ad0373662ac4cae82e10aaa3eedd9d467844844f4ba32`).
Before inspecting geodesic labels or model scores, select **every**
route variant with three executed movement turns in those scenes;
retain an episode-seed group only if at least two of its four
variants qualify. Freeze the exact identities, source rollouts, scene
split, and train dataset hashes. Render only initial and turn-3 views,
replay all executed actions to check terminal geodesic drift, and keep
the distance audit separate from model-input JSON.

The deterministic selection is now frozen at manifest SHA-256
`dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1`:
68 qualifying episode-seed groups per seed, 540 route records, and 68
unique episode IDs across the eight scenes. The compact
[source summary](ordinal_progress/policy_preference/future_advantage_pooled/early_anchor_audit_source_summary.json)
contains hashes and counts but no dataset instructions or distance
labels. These counts do not establish the required 1 m comparison
coverage or any model accuracy.

After exact-coverage verification, compare only same-seed,
same-episode route pairs whose **current turn-3** geodesic distances
differ by at least 1 m. Require at least 100 pairs, 40 underlying
episode IDs, and six audit scenes with eligible pairs, or report the
audit as underpowered. Without tuning the model, require all-pair
episode-macro accuracy at least 75%, same-terminal-mode episode-macro
at least 70%, all-pair scene-macro at least 70%, and at least a
five-point episode-macro margin over a commanded-forward-distance
baseline computed on identical pairs. Report side balance, termination
mode cells, scene bootstrap intervals, ID coverage, and inference
errors. This is a train-scene audit after model-development decisions;
even a pass is not independent navigation evidence.

Only if this anchor-ranking audit passes, perform a separately frozen
same-start, different-goal instruction-swap audit. Use label-only
simulator replay under both goals to retain routes whose progress
ordering truly reverses, then require at least 50 underlying IDs in
each reversal direction and 75% correct-instruction score preference.
If coverage or grounding fails, stop. The seven-scene prospective
train audit remains closed until both stages pass. Only thereafter
could a turn-3-only group-four 64-step matched policy pilot be run.
Center the four turn-3 scores within each active all-failure episode
group, clip to [-0.25, 0.25], and apply only to generated movement
tokens at turn 3; STOP and observation tokens receive zero. Mixed
groups keep ordinary outcome credit. The control, seed, training rows,
optimizer steps, and fixed 256 val-unseen IDs must match. Positive
paired SR **and** SPL would be needed before a complete 1,839-episode
recheck. An n=8 run is only a small matched sensitivity check after an
n=4 navigation gain, never the primary method.
