# Branch-curriculum positioning (working note)

This note is a literature and claim audit for the ongoing branch experiment. It is
not a result section. The 64-step, 256-episode gain is a screening signal; the
three-seed 128-step and full 1,839-episode matched comparisons are pending.

## Closest published or posted methods

- [ActiveVLN (2025)](https://arxiv.org/abs/2509.12618) is the implementation
  base. It uses multi-turn environment interaction and GRPO on multiple
  rollouts. A claim that our method first applies GRPO to closed-loop VLN, or
  first samples multiple continuations, would be incorrect.
- [JOP-VLN (2026)](https://arxiv.org/abs/2607.13461) joins off-policy
  imitation with on-policy RL and prioritizes error correction. A broad claim
  that ours first trains VLN from policy-induced states or first studies
  recovery would be unsupported. The relevant comparison is whether a
  *matched* policy-prefix branch curriculum improves this specific ActiveVLN
  policy and evaluation setup beyond ordinary task-start GRPO.
- [Hindsight-Divergence Localization (2026)](https://arxiv.org/abs/2609.36864)
  branches RLVR continuations at selected positions after full root rollouts
  in math, code, and agent tasks. Thus generic branching or shared-prefix
  group sampling is not a standalone novelty claim. Our current method uses
  executed navigation actions and separately replayed simulator trajectories
  to put multiple VLN continuations at a common physical decision state; it
  does not use HDL's hindsight likelihood criterion or establish its reported
  rollout-efficiency gain.

## What the current experiment actually tests

For each of 512 unique training episodes, the branch arm replays a 4–9
grouped-action prefix produced by an earlier policy, then samples two
continuations from the reached state. Its matched control samples two
continuations from the task start. Both use the same SFT initializer, train
episode rows, seed schedule, optimizer settings, and destination-only reward.
The comparison tests the *combined curriculum change*: later physical state,
policy-generated prefix, remaining action budget, and the resulting state
distribution. It does not isolate which of these components causes a gain.

An equal-action-count expert-prefix dataset has been prepared but not trained.
It would narrow the prefix-source comparison, while different action lengths
and physical endpoints would still complicate causal attribution. A stronger
ablation would compare policy and expert prefixes ending at a matched physical
pose, if enough such cases can be constructed without leaking validation
information.

## Claim gate

Write a positive paper claim only after all six 128-step checkpoints have
exact 1,839-episode val-unseen coverage with no inference errors, and the
full three-seed paired SR/SPL analysis supports a coherent effect. Report
the frozen 256-episode screen in full as an interim result, including any
disagreement with the larger evaluation; its sign is not a prerequisite for
running the full set. Report compute and environment interactions alongside
SR/SPL. If the full result is inconsistent or null, retain the experiment as
a negative finding and test another mechanism rather than presenting the
pilot difference as a confirmed gain.

The existing EventTrace blind audit has a single AI annotator; it supplies
neither independent human truth nor a validated verifier accuracy estimate.
