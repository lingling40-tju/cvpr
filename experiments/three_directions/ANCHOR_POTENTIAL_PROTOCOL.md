# Instruction-conditioned anchor potential: next group-four reward test

Status (2026-10-05): **source coverage and a frozen test design only**.
No potential model, instruction-swap result, online policy, or navigation
gain exists for this candidate. The earlier future-return ranker failed
its fixed train-scene development gate, especially at turn 3. This
candidate changes the supervised target from *future policy return* to
the agent's **current** goal proximity at fixed turns 3 and 6. Simulator
geodesic distance supplies training labels but is never model input.

## Frozen source and representation

Reuse the verified group-four train-only sparse RGB records, with source
manifest SHA-256
`4a0a2403fc4345308545d36f17102664856e24129eab189a32c0478a5bcf7967`.
The fit split has 1,490 selected trajectories from 37 scenes; development
has 305 from eight disjoint scenes. They were previously inspected for
another representation, so development is exploratory. The separate
eight-scene audit and seven-scene prospective partition stay unopened.

For each same-seed, same-episode group, compare two selected routes at
turn 3 or 6 only when their **current** geodesic distances differ by at
least 1 m. The closer route is preferred. The frozen CPU
[preflight](ordinal_progress/policy_preference/future_advantage_pooled/dense_potential_preflight.json)
finds 965/1,131 fit pairs at turns 3/6, of which 581/706 share a
terminal mode; development has 169/208 pairs, of which 103/126 share
a terminal mode. The model receives only the instruction, initial and
anchor RGB views, and actions executed through the anchor. It never
receives distance, scene/episode ID, goal coordinates, terminal mode,
future action, or final result. Its prompt asks for current proximity,
not expected future return.

Start from the same navigation-SFT Qwen2.5-VL-3B model and fit one
rank-eight q/v LoRA plus LayerNorm–128–scalar head. Use 1,024 fixed
microsteps, gradient accumulation four, seed 11, AdamW learning rates
5e-5 for LoRA and 1.5e-4 for the head, and weight decay 0.01.
Alternate turns 3 and 6; sample a fit episode uniformly, then one
seed-specific **same-terminal-mode** route pair. Optimize logistic
preference loss with a 0.001 score-magnitude penalty. Use the final
checkpoint, with no development-selected epoch or threshold.

## Decision gates and reward use

On the scene-disjoint development split, independently report all and
same-terminal-mode episode-macro and scene-macro accuracy at both turns,
the forward-action-only baseline on the *same pairs*, preferred-side
balance, sample counts, and scene bootstrap intervals. The fixed
screen requires at each turn at least 70% episode-macro accuracy in
both modes, at least 65% scene-macro accuracy in both modes, and at
least a five-point episode-macro margin over the action baseline in
both modes. A failure stops this checkpoint before any online RL.

A passing development result would still need a separate instruction
goal-swap audit on start-matched different-goal routes whose simulator
labels truly reverse under the swapped instruction. The source
preflight has 67 fit selected-ID goal pairs across 45 starts and only
12 development pairs across 11 starts; those are metadata opportunities,
not reversal labels or a grounding score. Freeze and verify the swap
routes before scoring the model, then require at least 50 underlying
episode IDs in each reversal direction and 75% correct-instruction
preference on the reserved audit scenes. The already frozen 123-episode
prospective train-scene audit remains closed until this gate passes.

Only a model passing those gates may supply an auxiliary group-four
reward. At active turn 3, center four potential scores within the
episode group; at active turn 6, center each route's change from its
own turn-3 potential. Clip the auxiliary value to [-0.25, 0.25] and
place it only on that turn's generated movement-action tokens. Skip
the anchor for all four routes if any route lacks an active prefix;
STOP and observation tokens get zero. Groups containing any success
retain the ordinary outcome advantage. This is a bounded process
credit heuristic, not a claim of policy-invariant potential shaping.

The first policy test remains a matched 64-step, group-four seed-11
pilot on the first 256 frozen exact512 train rows, compared with the
same-data outcome-only control on the separate fixed 256-episode
val-unseen screen. Both paired SR and SPL must improve with exact ID
coverage and zero inference errors before a complete 1,839-episode
recheck. A positive full and screen-excluded result would warrant the
three-seed 128-step n=4 scale. An n=8 run is only a small same-trajectory-
budget sensitivity check after learned n=4 navigation gain. Previously
reused val-unseen data are development evidence, not an independent test.
