# Conditional follow-up: terminal route-fidelity reward

This is a **prospective, unrun experiment protocol**. It is considered only
if the isolated endpoint-distance progress-reward experiment fails its
predeclared navigation gate, or its matched three-seed expansion fails to
retain a positive complete-set mean in the first or separate decode
replication pass. It is a conventional reference-route reward baseline,
not a novel semantic-verification contribution.

## Mechanism and contrast

The destination-only GRPO return leaves many unsuccessful two-rollout groups
tied. For the completed 128-step same-data controls, 348, 345, and 347 of
512 episode groups have equal return in seeds 11, 22, and 33. Many ties
include at least one rollout that exhausted the turn budget; the existing
`weighted_success_ndtw` reward computes generated nDTW at termination but
awards it only for an explicit `STOP`. Consequently, merely setting its
coefficient above zero would leave a large fraction of budget-exhausted
rollouts unranked. These are training-set diagnostics, not held-out evidence
that the proposed reward improves navigation.

The follow-up adds one terminal, bounded route-fidelity term for every
well-formed terminal reason, including exhausted action or turn budgets:

\[
R_{\mathrm{route}}(\tau) = \operatorname{clip}(\operatorname{gen\_nDTW}(\tau),0,1).
\]

`gen_nDTW` is the existing ActiveVLN alignment of generated positions to
the remainder of the train episode's reference trajectory. Its coefficient
is fixed to 1.0; malformed-response termination receives zero. A
nonfinite value also receives zero and must be counted in rollout audits.
The destination reward and the additional success floor of 2 are otherwise
identical to the matched control, so a failed rollout's maximum added term
cannot outrank a successful rollout solely through route fidelity. This
reference route is training-only: the actor observes its normal images and
instruction, and held-out evaluation receives no route coordinates.

Unlike endpoint geodesic progress, this term prefers reference-path
similarity. It may penalize a valid alternative route; SR and SPL on complete
val-unseen, not training return or nDTW itself, determine whether it helps.
No claim about semantic event completion follows from this experiment.

## Matched test and gate

1. After the progress-reward decision is final, stage a separate source
   checkout, never modifying the running progress trainer or service. Verify
   the original `vlnce_server/env.py` SHA-256
   `39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a`
   before applying `terminal_ndtw_all_reasons.patch`. Compile the modified
   module and smoke-test both budget and explicit-stop terminal cases with
   the route term; confirm malformed outputs receive zero.
2. Train a 64-step seed-11 from-scratch GRPO pilot on the exact 256 unique
   rows used by `branch_control64` and the progress pilot, from the same SFT
   initializer. Keep four train episodes per step, two independent rollouts
   per episode, 12 turns, 36 commands, and all optimizer settings fixed.
   Set `ndtw_reward_base=1.0`; keep `semantic_reward_weight=0` and
   `semantic_success_floor=2`. Audit all 64 paired episode sets, 512
   rollouts, no replayed prefix, and the logged nDTW component on every
   well-formed terminal rollout before accepting its checkpoint.
3. Evaluate the exact frozen 256 val-unseen episodes against the existing
   same-data `branch_control64` checkpoint, with zero inference errors.
   A positive paired SR change and nondecreasing SPL permits a separately
   audited three-seed 128-step expansion on the same 512 unique train rows.
   Then evaluate all three new checkpoints and existing matched controls on
   every one of the 1,839 val-unseen episodes and report per-seed effects,
   scene/seed uncertainty, the 1,583 episodes outside the pilot screen, and
   compute costs. A negative pilot is recorded as such; no claim of gain is
   made from rollout reward or the fixed-screen success count alone.

The fixed 256 episodes and complete 1,839 episodes have already been used
for earlier adaptive decisions in this project, so any positive result would
need independent replication before being described as a generalizable
navigation improvement or a CVPR contribution.

`run_route_fidelity_conditional.sh` waits for the completed progress pilot
or scale decision before staging the separate source, starting service port
5012 on GPU 2, training the matched 64-step pilot on GPUs 0 and 1, and
evaluating it on the frozen 256 episodes. `stage_route_fidelity_fallback.sh`
checks the original source and patch hashes; `audit_route_training_pair.py`
gates acceptance of the checkpoint on exact train-row pairing and nonzero
route reward on budget-exhausted rollouts. If the pilot clears its SR/SPL
gate, `run_route_scale_conditional.sh` trains three 128-step seeds, audits
their rollouts, and evaluates the fixed 256 and full 1,839 manifests using
`analyze_route_scaled.py`. The scripts were deployed to the original
experiment's `tools/` directory after local/remote hash and syntax checks.
Both conditional watchers are active, but only waiting: the route source
tree, service, trainer, and held-out evaluation have not started. Their
presence must not be interpreted as an experimental result.
