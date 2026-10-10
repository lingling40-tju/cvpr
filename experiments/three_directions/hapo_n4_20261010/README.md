# HAPO-style n=4 VLN pilot

This is a new, isolated follow-up after the matched GRPO, turn-RLOO, SRGPO,
positive-trajectory, and ReMax pilots did not establish an improvement over
the matched SFT policy. It adapts the temporal-kernel baseline from
[LongNav-R1 / HAPO](https://arxiv.org/abs/2602.12351) and its
[official implementation](https://github.com/UMich-CURLY/LongNav-R1) to the
current multi-turn R2R trainer. LongNav-R1 reports object-goal navigation;
that result does not predict performance in this experiment.

## Frozen proposal

- Qwen2.5-VL-3B navigation SFT initialization; 512 existing R2R-train rows.
- Group size 4, configured seed 11, 64 optimizer updates total (2-step
  wiring smoke plus 62 resumed steps), same rollout count and FP16 evaluation
  as the previous n=4 development comparisons.
- Turn reward equals the existing training-only geodesic-progress signal
  (weight 0.5) plus the same terminal score divided by 15. This uses privileged
  simulator geometry and is not a semantic verifier or deployable reward.
- For each trajectory, compute discounted return-to-go with `gamma=0.95`.
  Estimate its baseline from the other three rollouts of the same instruction
  using a Gaussian kernel over absolute turn index with `sigma=2.0`.
  Globally normalize active turn advantages, then divide each turn's credit
  evenly over its action tokens and executed turns.
- Evaluate once on the already reused 256-episode, 8-scene development
  manifest against the same-precision SFT reference. Require exact episode
  coverage and zero inference errors. Advancement requires paired SR and SPL
  each at least +2 percentage points; otherwise do not scale this method.

The group-conditioned kernel is a deliberate adaptation: unrelated R2R
instructions should not share a value baseline. The frozen proposal differs
from turn-RLOO by estimating each turn baseline from neighboring turns as well
as the same turn. It does not alter policy architecture, group size, data, or
evaluation thresholds.

## CPU preflight

`preflight_hapo_kernel.py` evaluated leave-one-trajectory-out return prediction
on completed privileged n=4 rollout archives from seeds 11, 22, and 33. With
the fixed `gamma=0.95`, `sigma=2` reduced RMSE against the held-out trajectory
return by 3.75%, 4.69%, and 4.73% respectively versus a uniform-time kernel
(`sigma=infinity`). `hapo_kernel_preflight.json` records source hashes and
counts. This is a baseline-prediction diagnostic, not evidence of a navigation
gain and not an independent validation set.

The first smoke will verify token/turn alignment, exact group membership,
finite advantages, nonzero actor gradients, and unchanged reward-source and
checkpoint identities before any full run.

## Completed run and result

The frozen run completed 64/64 optimizer updates from 512 fit rows with group
size four. All 64 actor-gradient checks were nonzero, with no recorded OOM or
exception. The source snapshot actually deployed is preserved at
`runlogs/verl/trainer/ppo/hapo_group4_advantage.py` (SHA-256
`3492e3c542190b17d33fdaa610cf8316a2ec99db587111cd1c66f6533c6bf6dd`); the
local vectorized `hapo_group4_advantage.py` differs from that deployed file and
must not be used as a byte-for-byte reproduction of the run. The corresponding
remote trainer and runner hashes are recorded in the freeze/evidence bundle.

On the fixed development256 screen, the candidate had 104 successes versus
109 for matched FP16 SFT. Paired changes were -1.953125 SR and -1.563939 SPL
percentage points. All 256 unique episodes across eight scenes were covered,
with zero inference errors; the independent compact recount agrees. The
scene-bootstrap intervals include zero. The frozen +2/+2 gate failed, so
there was no scale-up. This is one configured seed on adaptively reused
development scenes, not a clean generalization test. Reward includes
privileged simulator geodesic progress and is not a deployable semantic
verifier. Full report, raw episode records, source freeze and recount are
under `runlogs/hapo_development_eval/` and `runlogs/freeze/`.
