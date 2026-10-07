# Branch-curriculum positioning (working note)

This note records the scope and interpretation of the completed branch
experiment. The 64-step, 256-episode gain was a screening signal. All six
three-seed, 128-step checkpoints subsequently completed the full 1,839-episode
matched comparison with zero inference errors. The mean paired changes are
negative: -1.14 SR and -0.55 SPL percentage points. This follow-up did not
establish a reliable branching benefit.

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

## Completed comparison and claim boundary

The completed [full comparison](scale_full1839/full1839_analysis.json) records
candidate/control success counts of 368/500, 504/495, and 584/524 for seeds
11, 22, and 33. Paired SR changes are -7.18, +0.49, and +3.26 percentage
points; the exploratory scene-and-seed interval for their mean is
[-6.51, +4.02]. The corresponding SPL interval is [-6.16, +4.85].
The effects disagree across seeds and both intervals include zero.

Retain the frozen 256-episode screen beside these larger results as a
development finding, with compute and environment interactions reported.
The screen does not override the negative full-split mean. All three seeds
are reported regardless of their individual effect signs. Val-unseen was used
adaptively during development; this comparison is not a clean confirmatory
test. Further experiments now test different update mechanisms rather than
extending this branch claim.

The original EventTrace blind labels came from a single AI annotator.
[Two subsequent independent human reviewers](../turn_rloo_20261005/human_review_agreement.json)
agreed on 33 of the 49 selected cases (Cohen's kappa 0.459); 16 disagreements
remain unadjudicated. Neither the AI labels nor this selected, unresolved
human review supplies a validated semantic-verifier accuracy estimate.
