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
