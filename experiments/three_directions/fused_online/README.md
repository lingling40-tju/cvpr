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
job ends without spending that training budget. At the time of this
record, there is no online fusion-reward val-unseen result. Do not claim
a navigation benefit from the offline pair tests or the two-step smoke.
