# Outcome-consistent RLOO pilot (frozen before new evaluation)

This isolated, one-seed pilot tests two changes as a bundle on the already
reused R2R-train development screen: (1) a trajectory truncated by the hard
turn or action budget receives the evaluator-equivalent forced-STOP outcome
using the existing geodesic success radius; and (2) terminal nDTW contributes
only on successful trajectories. Failed STOP trajectories receive zero path
credit. The policy update otherwise uses the existing turn-level active-peer
RLOO progress signal. This experiment cannot attribute any result to one
component.

## Frozen budget

- Same navigation SFT initialization; 512 fit rows, batch 8, group size 4.
- Configured seed 11; 64 optimizer updates total, including a real two-update
  multimodal smoke and 62 resumed updates; learning rate `1e-6`, reference KL
  coefficient `0.1`.
- Progress weight `0.5`; success reward `15`; successful terminal nDTW weight
  `5`; maximum 12 turns and 36 executed actions.
- Candidate inference only on the fixed 256-episode, eight-scene train-scene
  development manifest. Reuse exact previously measured FP16 SFT and RLOO
  outputs; no reserved or val-unseen inference for tuning.
- Candidate vs SFT and candidate vs previous RLOO must each reach paired
  `+2.0` percentage points in both SR and SPL to qualify for a separate
  scaling protocol. Otherwise stop this mechanism.

## Interpretation limits

The screen has been adaptively reused, this is one configured seed, and
independent rollout-stream RNG has not been established. The RLOO signal uses
privileged simulator geodesic information. A positive pilot is exploratory,
not evidence of clean generalization or a deployable semantic reward. The
49-example human audit remains a separate two-reviewer consistency exercise;
16 examples are unresolved and it is not a model-accuracy estimate.


## Executed-turn mask pilot (2026-10-11)

Only the policy-credit mask changes from the matched outcome-consistent RLOO pilot. Omit a final generated response when the simulator records `forced_stop_reason=turn_budget`, `executed_actions=[]`, nonempty `extracted_actions`, and zero oracle progress. Rewards, optimization, n=4/64 updates, SFT reference, development256, and the +2pp SR/SPL gates remain fixed. Previously reused development scenes make this exploratory, not clean-test evidence.
