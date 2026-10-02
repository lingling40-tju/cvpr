# Instruction-grounded ordinal progress: preregistered pilot protocol

Status: train-only offline models have been tested. The preregistered
representation gate failed; no representation reward RL has run, and no
navigation gain has been observed for this method.

## Motivation and distinction

The existing 2-sample reward pilots (semantic events, geodesic progress,
nDTW) did not improve held-out navigation. Four-sample outcome-only GRPO had
a positive 256-episode seed-11 screen, but its confidence interval spans zero.
Sampling more trajectories can reduce all-tied groups; it is a baseline and
compute budget, not the proposed algorithm.

SACA (Li et al., arXiv:2603.09740) already uses a zero-shot
CLIP/GroundingDINO/SAM landmark auditor, process scoring, and all-failure
rescue. The proposed test instead **learns an instruction-conditioned visual
progress representation from R2R train trajectories**, with same-start,
different-goal instructions as hard negatives. Its key claim, if supported,
would be that counterfactual grounding and uncertainty calibration improve
the *reliability* of a progress-derived training signal. We must compare with
SACA's mechanism in the eventual paper; we cannot claim generic step-aware
reward or all-failure rescue as new.

## Representation to reward

Freeze a modest visual/text backbone. Render train-only expert trajectories
at turn boundaries, cache its image and instruction embeddings once, and
train a small progress head `p_phi(instruction, start_RGB, current_RGB)` in
[0, 1]. The start-relative visual difference is available at rollout time
without simulator state or goal coordinates.
Use ordered pairs from the same expert trajectory, nearby temporal negatives,
and matched natural instructions sharing the exact start pose but having
different goals. A separate calibration head/ensemble estimates uncertainty.
The representation never receives val-unseen images or simulator goal distance.

At rollout turn `t`, use an uncertainty-gated *new-high-watermark* score:
`m_t = max(m_(t-1), p_t)` and
`r_t = c_t * max(0, m_t - m_(t-1))`, with total auxiliary return <= 1.
Here `c_t` is a calibrated reliability factor in [0, 1], set to zero for
out-of-distribution visual features or ambiguous counterfactual scores. This
prevents repeated reward for loops and caps its magnitude. It does **not**
mathematically guarantee no reward hacking, so full navigation evaluation is
essential. It is not claimed to be policy-invariant potential shaping: a
strict telescoping potential with a fixed start and zero terminal potential
would give the same return to every trajectory and would not resolve GRPO's
all-failure ties.

In mixed success/failure groups, use only the environment outcome advantage.
In all-failure groups, use relative progress advantage only when a calibrated
margin separates trajectories; multiply it by a bounded reliability weight
**after** within-group normalization. Merely scaling the raw auxiliary reward
before GRPO normalization would not control its gradient when all outcomes
tie. If the gate fails, set the auxiliary advantage to zero. Success reward
and its definition stay unchanged.

## Stage gates and resource budget

1. Generate a deterministic scene-disjoint split *within R2R train*: 51
   representation-fit scenes and 10 calibration scenes, with 512 and 128
   expert episodes respectively. Require at least 128/32 disjoint natural
   same-start instruction pairs in these subsets. No val-unseen scene may be
   used for representation fit or calibration.
2. Collect RGB at sparse expert turn boundaries; cache frozen-backbone
   embeddings once. First use 64 fit episodes plus 32 calibration episodes
   to test the pipeline. If that screen is viable, collect all 512/128
   episodes and refit before any RL. Compare ordinal pair accuracy,
   same-start instruction discrimination, calibration, and behavior on
   deliberately mismatched instructions on the 10 held-out train scenes.
   Use the first five calibration scenes only for checkpoint selection;
   lock the other five for a single audit. The offline go/no-go rule, fixed
   before inspecting the full 512/128 result, applies to that locked audit:
   mean ordinal accuracy >= 70% and same-start counterfactual accuracy >= 75%
   across three head seeds, each at least five percentage points above the
   frozen backbone's raw score. Report per-scene breakdowns and seed
   variation. Do not launch RL if the representation fails this rule. The
   64/32 screen used the same calibration scenes for head selection, so its
   numbers are exploratory and are not an independent estimate. Several of
   those 32 episodes lie in the later five-scene audit; the full audit is
   locked against *full-run checkpoint selection*, but not wholly untouched
   by earlier architectural exploration.
3. Run a 64-step pilot at **group size 4**, batch size 4 (16 rollouts per
   optimizer step), seed 11, using the same 256 train rows and frozen SFT
   initializer as the completed group-4 outcome-only control. Compare its
   256 unseen episodes on the existing fixed manifest, with exact coverage,
   paired SR/SPL, scene-bootstrap uncertainty, train rollout count and GPU
   time. Include ablations for no hard negatives and no confidence gate if
   the full method passes the first screen.
4. Replicate a short pilot at **group size 8** with batch size 2 and 16
   simultaneous rollouts, comparing method and outcome-only control *within
   the same group size* on identical episode order and budget. This fits the
   validated 16-simulator service; it is not directly compared with group 4
   as an equal-data test because per-step episode counts differ.
5. Only a gain in SR with non-decreasing SPL on the frozen 256-episode
   screen triggers 128-step, three-seed work and full 1,839-episode paired
   val-unseen evaluation. A positive 256-episode screen is exploratory.
   Report group size, episodes, rollouts, simulator time, model time and
   inference errors for every arm. Repeated method selection on one screen
   can bias estimates; the complete evaluation and scene-level intervals
   determine the claim.

The currently running three-seed group-4 outcome-only suite remains a
baseline. The 64-step 2+2 pairing pilot is a compute-control diagnostic:
71/256 held-out successes, versus 80/256 for the ordinary group-4 pilot and
75/256 for the older two-sample control; SPL is lower than both. Its
predeclared gate recorded `no_pilot_gain`, so its three-seed expansion was
not launched. These are pilot comparisons, not evidence about the proposed
representation method. No scientific gain follows from training reward
variance alone.

Reference: Haoyuan Li et al., *Let's Reward Step-by-Step: Step-Aware
Contrastive Alignment for Vision-Language Navigation in Continuous
Environments*, arXiv:2603.09740 (2026),
https://arxiv.org/abs/2603.09740.

## Offline outcome (2026-10-02)

All 512 fit and 128 calibration train episodes were rendered with exact
coverage. Three SigLIP start-relative heads achieved 72.2--75.4% ordinal
accuracy but only 43.3--55.0% same-start/different-goal accuracy on the
five-scene full-run audit. The independently recomputed three-seed means are
74.30% and 47.22%; frozen-backbone scores are 54.22% and 53.33%. The
predeclared gate failed on the counterfactual criteria. `ordinal_progress/
full512x128/locked_audit_gate.json` includes every seed and scene interval.

An exploratory causal visual-history matcher using the same cached expert
frames and a separate 41/5/5 split of fit scenes reached 98.97% ordinal
accuracy but 50.0% different-goal accuracy on its audit. It can infer frame
order without grounding the route to the instruction. This candidate also
fails its offline gate. These diagnostics rule out deploying either current
progress score as an RL reward; they do not rule out all representation-based
methods.

## Goal image grounding screen (2026-10-02)

A further light adapter trained on the same 512 frozen expert trajectories
using only natural same-start/different-goal pairs reached 53.33% different-
goal accuracy and 51.67% endpoint-above-start rate on the five-scene audit;
the frozen image/text score was also 53.33% on the former. This did not pass
the instruction-grounding screen, so no RL was launched for that adapter.

The next screen uses a fixed, larger train-only manifest derived before its
images were scored: 2,755 fit episodes and 4,654 natural different-goal
pairs from the same 51 fit scenes, plus 400 calibration episodes and 382
pairs from the same ten held-out train scenes. A goal coordinate is used only
to render four RGB directions for each *train* positive image. The candidate
selection manifest is frozen at SHA-256
`3bb3925a9840aea2dd825c3c201f8d4ceadd8e4949650f534ebf3cda17be76ce`.
The candidate reward model takes only an RGB embedding and instruction
embedding; it
cannot read goal coordinates, geodesic distance, simulator success, or scene
IDs at RL time. Records and frozen SigLIP features are cached once, with
resumable per-episode collection and scene-grouped scene loading. This costs
no RL rollout and can run on GPU 0 while the baseline group-4 trainer uses
GPUs 2/3. Goal-pose views are a different image distribution from the
agent's normal camera, so their retrieval accuracy alone is insufficient.

The first five calibration scenes select a checkpoint. The other five are
used for a single model audit. The offline gate fixed in
`fit_panoramic_goal.py` requires at least 75% in both image-to-instruction
and instruction-to-goal-image matching on those held-out goal views, at least
75% correct-instruction preference when transferred to previously cached
*ordinary expert trajectory* views, a five-point improvement over frozen
SigLIP on that expert-view test, and at least 65% of expert endpoints scoring
above their starts. Only if seed 11 passes do we repeat fitting with seeds
22 and 33 and design an online confidence-gated reward. As with the earlier
audit, these scenes were seen during exploratory architecture work; the
held-out result is therefore a screening diagnostic, not an untouched
independent test. Any RL pilot still uses group size 4 and the same rollout
budget as its paired control; the optional group-size-8 screen matches its
own eight-sample control at 16 simultaneous rollouts.
