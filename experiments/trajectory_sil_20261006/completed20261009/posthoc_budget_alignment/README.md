# Training and evaluation budget conventions

This post-hoc CPU audit inspects two completed seed-11 positive-trajectory training runs and the existing matched-FP16 SFT full evaluation. No model or simulator call is added. All 8,192 training records match the previously archived raw rollout SHA, fit membership, per-step n=4 counts, and original budget/config identities; independent local compact arithmetic passes.

| Logged training outcome | GRPO control | Positive candidate |
| --- | ---: | ---: |
| Recorded trajectories | 4,096 | 4,096 |
| Budget-exhausted terminal trajectories | 956 | 1,205 |
| Within 3 m at budget exhaustion, recorded failure and zero total reward | 230 | 314 |
| Unsuccessful model STOP with positive nDTW reward | 1,893 | 1,682 |

All 230/314 near-goal timeout cases are strictly inside 3 m; none is exactly at the boundary. The compact records preserve actual per-episode budgets (most are 12 turns/36 commands, some have fewer command slots), terminal distance, logged task and geometric status, and reward components. These are repeated fit rollouts, not evaluation SR or independent sampled tasks.

The source snapshots are byte-identical between the completed positive root and the pending three-direction root. In `vlnce_server/env.py`, turn-budget exhaustion (lines 408–412) and step-budget exhaustion (447–452) unconditionally set `is_goal_reached=False`; nDTW is zero for these termination reasons (388–400). The semantic wrapper adds no reward when both weights/floors are zero. In the evaluator, the turn limit instead emits an actual STOP (327–340), followed by environment stepping and metric collection (93–119). Full source hashes and excerpts are in `report.json`.

The existing full SFT evaluation has 1,083 forced turn-limit stops; 238 are recorded successes, out of 555 total SFT successes. This recount uses the previous hash-checked seven-model diagnostic, not new SFT inference. It establishes different objective conventions in the checked code and their incidence in these records. The training and SFT figures concern different trajectories/screens: do not add 230/314 to evaluation SR or infer recovered navigation results.

This is one mechanism hypothesis for the old degradation. It does not identify a causal effect, validate a semantic label, or alter old metrics. The three current 64-step pilots, source identities, inference and advancement gates remain frozen. Any time-limit-aware credit mechanism must be a separate isolated, matched n=4 experiment with explicit termination/truncation handling and an unchanged shared reference; code corrections cannot be presented as evidence for an unmeasured algorithmic gain.


## Reward order within the same recorded n=4 episode group

A [post-hoc grouped recount](within_group_terminal_reward_order.json) uses
only the already archived 8,192 terminal records. Each unit is one run,
optimizer step and episode ID with four recorded trajectories; each arm
has 1,024 groups over two passes through the 512 fit IDs.

| Descriptive event | GRPO control | Positive candidate |
| --- | ---: | ---: |
| Groups containing a <3 m timeout and a >3 m failed STOP with positive nDTW | 114 | 145 |
| Such within-group trajectory pairs where the farther failed STOP has higher reward | 215 | 270 |
| Near-goal timeout rewards below their four-member group mean | 190/230 | 242/314 |
| All-failure groups containing a positive-reward failed STOP | 404/453 | 389/463 |

The [independent set/pair/count recount](within_group_terminal_reward_order_independent.json)
checks all 2,048 groups and all 485 listed pairs against the original
compact source. This is a conflict between terminal reward order and
terminal **geodesic distance** order, not proof that the nearer trajectory
followed the language better. nDTW measures route similarity, and a mean
centered reward sign does not reconstruct the positive-credit candidate's
transformed advantage or actual gradient. These are repeated fit samples,
not semantic truth, held-out navigation SR or causal evidence.

If the current frozen three directions all fail their navigation gates,
this supports a separate test of outcome-consistent time-limit handling
and success-conditioned path credit. That would require a new isolated,
matched n=4 protocol with the same SFT reference and real navigation
assessment. No current reward, source, group size, evaluation or gate was
changed, and no new GPU inference or human labeling was performed.
