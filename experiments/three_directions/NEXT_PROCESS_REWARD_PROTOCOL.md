# Next group-four representation-to-reward experiment

Status: representation screens and a new group-four offline screen are in
progress. No new policy training run or navigation gain is claimed here.
The mode-stratified ordinal pilot failed its matched 256-episode screen
(SR -5.47, SPL -5.23 percentage points) and did not enter the 128-step
scale suite.

## Oracle turn-wise credit mechanism diagnostic (2026-10-04)

Before fitting another visual reward model, test whether the proposed
turn-wise optimizer can exploit *accurate* progress labels at all. The
isolated `ActiveVLN_turnwise_oracle_20261004` source tree uses simulator
distance-to-goal changes during **training only** as a privileged upper
bound. These distances are excluded from policy observations and inference.
This is not a deployable representation reward and cannot establish a
semantic method's effectiveness by itself.

The frozen group-four training rows are the same as the completed
64-step outcome-only control. The helper `oracle_turnwise_group4_advantage.py`
centers return-to-go among active rollouts of the same episode for an
all-failure group, assigns the resulting signal only to that turn's
action tokens, and gives no auxiliary credit to a STOP response. The
STOP mask follows the generated `extracted_actions` even if the
environment's turn budget prevents that STOP from executing. The
environment writes an `oracle_stop_response` flag, and the trainer and
independent audit cross-check the flag, parsed actions, and zero
process reward. A group with any success keeps the ordinary outcome
GRPO signal. Observation
tokens always receive zero. The adapter asserts exactly four rollouts
per group, action-block/turn-count agreement, process-reward/token-span
agreement, finite values, and no reward on observation tokens. The
environment, rollout, and trainer edits are preserved as
`oracle_turnwise_v1.patch`; `prepare_turnwise_oracle_tree.sh` pins the
source and training-data hashes before isolation. The environment also
records before/after distance for each turn so
`audit_oracle_turnwise_train.py` can independently recompute every
privileged reward, check the four-rollout train rows against the frozen
outcome control, and require nonzero actor gradients before admitting a
longer run.

CPU-only synthetic checks of the estimator and adapter passed. The
train-only preflight on 800 already collected four-rollout trajectories
found 80/160 all-failure fit groups and 21/40 all-failure development
groups. Every such group has nonzero turn-wise contrast under true
distance labels; in 64 fit and 21 development groups at least one
action-level advantage sign differs from endpoint-only broadcasting.
These are reused, correlated train-scene trajectories. The real
two-step n=4 wiring test and independent audit then passed: eight
matched groups, 370 turns, 303 nonzero progress turns, 14 generated
STOP turns, and nonzero actor gradients at both steps. Its source hashes
and counts are in
`ordinal_progress/policy_preference/oracle_turnwise_2step_audit.json`.
The same-seed 64-step n=4 pilot
has started; no navigation evaluation has completed. The standard
main comparison remains group size four;
group size eight is at most a later small diagnostic with its own
matched control.
`run_oracle_smoke_after_recheck.sh` waited for the four-GPU full-validation
sensitivity recheck to finish, checked that its model ports and the
required GPUs were released, and launched the two-step smoke. It
stopped its Habitat service after the independent audit.
`run_oracle_pilot_after_smoke.sh` then used the passed audit to start
a same-data, same-seed, 64-step group-four
training run and independent train audit, followed by concurrent
candidate/control evaluation on the previously frozen fourth 256-item
val-unseen manifest (SHA-256
`bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c`).
It records exact paired metrics and a conditional gate marker. A
positive result is still a **privileged simulator-distance upper-bound
diagnostic**; it does not authorize a paper claim or three-seed semantic
reward scale-up without a learned, observation-only representation.

## Conditional premature-STOP mechanism candidate (2026-10-04)

While the oracle turn-wise pilot runs, a second **train-only mechanism**
candidate is staged without using its GPUs. It isolates a failure mode
seen in the Qwen pilots: unsuccessful model-selected STOP. In an
all-failure group of exactly four rollouts from the same episode,
compare a failed STOP at least 3.5 m from the goal with a turn-cap
rollout that still lies at least 3 m away. Give the continuation a
bounded +1/6 vote and the STOP a -1/6 vote only if the continuation
finishes at least 1 m closer. Sum votes within the group, with absolute
per-rollout bound 0.5 and zero group total. Groups containing success
retain the ordinary outcome reward. This avoids giving positive credit
to a turn-cap rollout already inside the success radius, where the
correct action would have been STOP.

The frozen 64-step n=4 outcome control has 256 unique train episode
groups and 1,024 rollouts. Its offline preflight found 152 all-failure
groups, 85 with eligible STOP and turn-cap trajectories, and 51 active
groups with 105 qualifying pairs. The previously trained Qwen
confidence candidate has 146 all-failure groups, 91 eligible mixed
groups, and 62 active groups with 136 qualifying pairs. These are
correlated training trajectories; endpoints do not prove that the
continuation would have succeeded from the STOP state. The data and
rule are recorded in `control_seed11_preflight.json` and
`confident_seed11_preflight.json` under
`ordinal_progress/policy_preference/`. The helper is
`stop_pair_group4_reward.py`, with behavioral checks in
`test_stop_pair_group4_reward.py`. An isolated source copy was prepared
at `ActiveVLN_stop_pair_group4_20261004`; no policy training or
navigation claim follows from this preflight. If the current oracle
pilot misses its n=4 navigation gate, this distinct STOP mechanism can
receive a two-step wiring audit and then a matched n=4 pilot. It also
uses simulator distance **only during training**; any gain would still
need a learned observation-only STOP representation before a semantic
method claim.

## Question and algorithm

Test whether a policy-grounded *history* representation can provide a
calibrated movement-progress signal without encouraging premature STOP.
Previous frozen terminal scores have plausible train-scene pair ranks but
all four tested reward variants reduced matched val-unseen navigation.
The new representation must therefore pass a separate STOP and
instruction-grounding gate before its score enters RL.

Use the navigation SFT model's original multi-turn observation and action
format, not a single-image surrogate prompt. Adapt its visual-language
states with a small trainable module (and, if the small module fails, one
predeclared LoRA variant), producing two distinct outputs:

- `progress(I, history_t)`: instruction-conditioned progress from the
  trajectory start, trained by ordinal and local-distance comparisons on
  *R2R train* trajectories. Simulator geodesic distances may construct
  train labels, but must not be model inputs or validation reward inputs.
- `stop_ready(I, history_t)`: probability that STOP at this observation
  satisfies the task success radius. Train positive/negative views from
  train-scene expert paths and policy rollouts, including hard negatives
  from the same scene and near-goal but wrong-instruction pairs. This head
  is an audit/gate at first; do not turn it into a STOP bonus merely
  because its train loss is low.

The representation uses a scene-disjoint fit/development/audit split
within R2R train. Freeze episode IDs, scene IDs, feature hashes, and the
labeling rule before fitting. In addition to local progress accuracy,
measure same-start/different-goal instruction preference and STOP
calibration on the untouched audit scenes. Record positives, negatives,
scene counts, confidence intervals, and threshold selection on
development scenes. A candidate may enter RL only if the audit has at
least 100 local progress pairs and STOP positives/negatives from at
least 100 distinct underlying trajectories in each class, and reaches
all of: >=75% progress-pair accuracy, >=75%
instruction-swap accuracy, STOP AUROC >=0.80, and <=10% false STOP rate
at a development-selected threshold with >=50% recall. These are
go/no-go screens, not expected performance claims. A failed screen ends
this candidate without another GPU-heavy policy run.

For a passing representation, the *initial* RL reward uses bounded
progress differences only on movement/turn actions. Use signed
`clip(progress_t - progress_(t-1), -c, c)` and a confidence gate;
do not reward STOP from `stop_ready` in the first pilot. Preserve the
original outcome advantage in groups containing a success. In a
four-rollout all-failure group, compute per-turn returns-to-go from the
process signal, center them against the other active rollouts from that
same episode, and map each turn's advantage only to its own action
tokens. Mask observation tokens and the final STOP action from the
auxiliary term. This differs from the current verl implementation, which
sums token rewards and broadcasts one scalar advantage to all action
tokens. First prove with a synthetic example that equal terminal totals
at different turns yield different action-token advantages; then run a
two-step smoke test checking four rollouts per group, reward/turn masks,
nonzero actor gradients, and no accidental STOP bonus. Do not describe
this as policy-invariant shaping.

Step-aware VLN reward and process alignment are already studied by
[SACA](https://arxiv.org/html/2603.09740v1), whose auditor combines
CLIP, GroundingDINO, and SAM3 evidence with divergence-point masks,
all-failure rescue, and repair resampling. [Progress-Think](https://arxiv.org/html/2511.17097v2)
already aligns visual history with instruction prefixes and jointly
fine-tunes progress and navigation. Thus a generic step reward, failure
rescue, or monotone semantic progress is not a novel paper claim here.
Any later claim needs measured instruction-counterfactual grounding,
separate STOP calibration, and verified action-token credit on the
matched group-four implementation, with direct comparisons to these
methods. The simulator-distance oracle experiment is only a mechanism
upper bound and supplies none of those learned-representation claims.

## Four-A800 schedule and validation budget

1. **Data/representation preflight.** Reuse cached train-only policy
   rollouts and expert frames. Collect only missing history or hard
   negative views. Cache each frozen visual state once; run scene split,
   labeling, calibration, and audit before RL. GPU 0 may render Habitat
   views, GPU 1 may fit the representation, and GPUs 2/3 can perform
   independent feature extraction or a frozen-control check. Report
   GPU-hours and cache hits. Do not fill all GPUs with duplicate work.
2. **Training wiring.** Keep `rollout.n=4`, batch size 4, 16 rollouts per
   optimizer step, seed 11, the same SFT initializer and 256 train rows
   as the cached group-four outcome-only control. During policy training,
   the validated layout is GPU 0 for 16 Habitat instances, GPU 1 for the
   reward service, GPUs 2/3 for policy rollout and updates. Train two
   steps first, then at most 64 if all audits pass. Sequential candidate
   training is necessary when the four-card layout is saturated.
3. **Paired unseen screen.** Freeze a fourth disjoint 256-episode
   val-unseen manifest before looking at its outcomes. Evaluate the
   candidate and same-seed, same-group-size, same-step control on the
   *same* episode IDs. Run two lanes concurrently: each has one A800
   inference server and one A800 with four Habitat shards. Assert 256
   unique episodes per arm, no inference errors, and compute paired SR,
   SPL, discordant successes, and scene-bootstrap intervals. Repeated
   pilot selection on val-unseen is exploratory; do not claim a robust
   gain from this screen alone.
4. **Scale gate.** Only paired SR > 0 and paired SPL > 0 with valid
   coverage trigger 128-step training at seeds 11/22/33, all group 4,
   followed by paired complete 1,839-episode val-unseen evaluation.
   Existing full-eval control results can be reused only for exactly
   matching seed, training data, group size, and checkpoint step.
   If the group-four method genuinely improves full validation, an
   optional small group-eight check may use batch size 2 (still 16
   rollouts per step) and its **own** group-eight outcome control. It
   cannot replace the standard group-four comparison because episode
   diversity per step changes.

The blind semantic-verifier audit remains a single-AI annotation audit;
it is not independent human truth and is not used as a success label for
this experiment.

## Train-only data preflight (2026-10-03)

`preflight_stop_supervision.py` checked all three completed 128-step
group-four outcome-only training rollouts: 512 episode groups and 2,048
rollouts per seed, 6,144 rollouts total across 58 train scenes. It found
1,324 successful STOP endpoints within 3 m, 1,836 failed voluntary STOPs
at least 3.5 m away, and 913 ambiguous near-goal failures excluded from
binary STOP labels. Across seeds, these labels cover only 327 distinct
positive and 462 distinct negative episode IDs; repeated rollouts are
not independent examples. End reasons were checked against each label.
The source hashes and per-scene counts are in
`ordinal_progress/policy_preference/stop_supervision_preflight.json`.

`freeze_stop_scene_split.py` fixed a deterministic 40/9/9 fit/development/
audit scene partition before any new representation fit. The nine audit
scenes contain only 45 positive and 78 negative distinct policy episode
IDs, below the predeclared 100-per-class requirement. They contain
1,700 R2R-train instruction episodes representing 566 distinct expert
trajectories, so additional *ordinary expert-path* camera views could
meet the sample gate without another policy rollout. These views and
their path actions still need to be collected and validated; the
existing goal-pose panorama cache does not substitute for the policy's
normal camera distribution. The frozen scene list and counts are in
`ordinal_progress/policy_preference/stop_history_scene_split.json`.
No new model has been fitted or accepted by the gate.

## Frozen expert-history data and unseen pilot manifest

`prepare_stop_history_manifest.py` selected one natural instruction per
underlying train trajectory, excluded any development/audit trajectory
already used by the three policy-rollout sources, and required a natural
same-start instruction with a different goal. Availability capped the fit
set at 896 trajectories (965 eligible); development and audit contain
160 each (166 and 186 eligible). The frozen manifest SHA-256 is
`3d8ab23313377729501a0dee4a9274231fb927f73033809a68a08fdcf22e1e43`.

Three Habitat collectors ran concurrently on A800 GPUs 0/1/2, grouped
episodes by scene, and replayed expert atomic actions as valid navigation
action text with at most three actions per turn. The first fit smoke
completed 2/2. Full collection then completed 896/896 fit, 160/160
development, and 160/160 audit records with all image paths present,
disjoint scene/trajectory keys, far starts, and endpoints within 3 m.
An audit-only whitespace mismatch in the collection wrapper initially
marked the run failed; the comparison was corrected and the existing
records re-audited without recollecting them. The verified coverage and
label counts are in `ordinal_progress/policy_preference/
stop_history_collection_audit.json`.

The first model screen restricts histories to at most 12 turns, matching
the policy training turn budget: 692 fit, 112 development, and 112 audit
trajectories. A separate label audit checks the GT terminal location
against the Habitat distance and rejects swapped instructions unless
their goal remains at least 3.5 m away in Euclidean distance. This leaves
661/111/108 safe natural swaps across the three parts. The audit has
more than 100 distinct underlying trajectories in each STOP class and
108 verified wrong-instruction pairs, narrowly satisfying the
predeclared data-size floor. Passing sample coverage is not a model
performance result. The labels are in `ordinal_progress/policy_preference/
stop_history_label_audit.json`; raw RGB remains on the experiment host.

`prepare_process_val_manifest.py` froze a fourth scene-balanced
256-episode val-unseen set before any new model fit or policy evaluation.
Its SHA-256 is `bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c`;
all 256 episode IDs are disjoint from the previous three screens. It is
only a future paired pilot screen. No inference on this set has run.

## Frozen-history readout outcome (2026-10-03)

`cache_stop_history_features.py` used the actual ActiveVLN server system
and observation templates, replayed only histories of at most 12 turns,
and cached Qwen2.5-VL-3B SFT assistant-prefix states and STOP margins.
Three extraction workers used GPUs 0/1/2 concurrently. Independent
coverage auditing found 692/112/112 finite 2,048-dimensional records
for fit/development/audit, 661/111/108 safe swapped-instruction states,
and a maximum prompt length of 3,828 tokens. Safe weights-only reads
were used for cached tensors. `stop_history_feature_audit.json` records
counts and extraction time; raw features remain on the experiment host.

A two-head MLP with a shared trainable readout, progress regression,
local ordinal loss, STOP classification, and natural wrong-instruction
contrast used only fit scenes for parameter updates. Seed 11 and epoch 5
were selected by the development composite loss; a STOP threshold was
selected on development negatives to keep their false-positive rate
below 10%. The locked nine-scene train audit then gave 223/223 local
progress ranks, 98/108 correct-vs-wrong instruction preferences (90.74%),
STOP AUROC 0.917, recall 79.53%, and **false-positive rate 12.62%** at
the frozen threshold. The predeclared 10% false-positive gate therefore
failed. `stop_history_two_head_locked_audit.json` and the hashed
checkpoint preserve the result. No group-four RL training was launched
from this head.

The post-hoc breakdown explains why the apparent progress rank is weak
evidence: every one of the 223 evaluated expert-path segments moved
*closer* in geodesic distance; none tested recovery from a regression.
Among audit negatives, 0/112 far starts and 6/97 far mid-route views
cross the STOP threshold, versus 34/108 safe wrong-instruction endpoints.
Thus the remaining problem is instruction grounding at plausible goal
views, not merely an overall score offset. These diagnostics are
exploratory and cannot justify retuning the threshold on the opened
audit. A further candidate must learn stronger cross-modal grounding
and be checked on policy trajectories with both positive and negative
progress steps using a newly frozen train-scene audit.

`turnwise_group4_advantage.py` implements a standalone candidate
estimator and passes a synthetic check where two rollouts have equal
terminal totals but different second-turn advantages; observation
tokens and all-failure STOP auxiliary tokens remain zero. It is **not**
yet wired into verl. No process-reward navigation result exists.

`preflight_turnwise_oracle.py` uses the already replayed complete
four-rollout groups to isolate the possible effect of assigning
movement credit at the responsible turn. It reads only the 160 fit and
40 development groups; the locked audit is not opened. The diagnostic
per-turn signal is the simulator distance change divided by the initial
distance (with a 3 m denominator floor). With discount one, these
changes telescope exactly to terminal distance progress, so action-level
credit changes without changing the summed trajectory score. Among the
80 fit and 21 development all-failure groups, every group had a
nonzero turn-wise contrast; 64/80 and 21/21 respectively had at least
one action-turn advantage whose sign differed from broadcasting its
trajectory's terminal-progress contrast. All 800 reused trajectories
and their terminal distances were checked against the frozen manifest
(`turnwise_oracle_preflight.json`). This is an **oracle-label train-scene
mechanism check**, not a learned semantic reward or a policy result.
It motivates a future two-step wiring test only if a credible process
representation is available; it does not waive the representation or
paired val-unseen gates.

The next cross-modal adaptation has a **different** frozen scene split,
created before its fitting by `freeze_lora_history_split.py`. Eight
previously fit scenes are reserved for new development, eight for new
audit, and the remaining selected scenes are fit. It yields 627/162/127
histories and 596/161/123 safe swaps. These new audit scenes were in
the previous small readout's fit data, so this is a fresh split for a
*newly initialized* adapter, not a fully untouched research audit.
The existing 800 policy-rollout geodesic records contribute only 80
trajectories and 38 one-meter regressions in that new audit scene set;
this is too few to establish recovery-sensitive progress. Replaying
additional existing group-four training rollouts at turn boundaries is
the next data step. Any adaptation must be selected on its new
development scenes and evaluated once on its new audit scenes before
the frozen 256-episode val-unseen pilot is considered.

## Frozen policy-turn replay for recovery-sensitive labels

`prepare_policy_process_manifest.py` now freezes 768/320/320 policy
trajectories in the new fit/development/audit scene partition, using a
scene round robin and hashed trajectory order before intermediate
geodesic distances are observed. The corrected manifest SHA-256 is
`aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681`.
Its sources are the three completed outcome-only, group-four, 128-step
rollouts. The first selector accidentally discarded timeout failures:
their logs contain a 13th response with **no executed actions** after
12 executed turns. The corrected selector retains these trajectories
and replays only executed actions. This raises the eligible inventory
from 2,330/465/460 to 4,456/852/729 trajectories across the three
parts. The selected audit set has 320 trajectories from 61 unique
episodes and eight scenes; correlations within an episode must be
respected in later uncertainty estimates.

`collect_policy_process_turns.py` replays the frozen identities in
Habitat, records an ordinary RGB image and action history at every
executed turn, stores geodesic distances only as labels, and rejects
terminal drift from the source rollout. Three independent collectors
use GPUs 0/1/2; the collection wrapper pins the manifest hash and
resumes complete records. `audit_policy_process_turns.py` checks every
image, identity, scene split, terminal distance, and turn delta. It
requires at least 100 one-meter regressions from at least 30 distinct
audit episodes before that split may be used for a recovery-sensitive
representation screen. No LoRA result or navigation benefit is claimed yet.

Full replay subsequently completed without error: 768/320/320
trajectories over 38/8/8 scenes, with 314/70/61 unique episode IDs.
The audit scenes contain 146 one-meter regression turns from 40
different episodes and 422 one-meter forward-progress turns. The
predeclared regression sample gate passed. The fit and development
scenes contain 307/161 regression turns, respectively. These are
label-coverage counts, not learned-model accuracy or navigation
results. `ordinal_progress/policy_preference/
policy_process_collection_audit.json` contains the verified counts.

## First history-grounded adaptation (launched after data gate)

`history_grounding_lora.py` keeps the original ActiveVLN multi-turn
prompt and adds rank-eight LoRA to the navigation SFT model's language
query/value projections. A two-output head predicts STOP readiness and
bounded progress. The unused vocabulary projection is replaced by an
identity **only in this reward encoder** to save compute; the final
assistant-prefix hidden state is unchanged. A four-microstep smoke on
GPU 3 produced finite loss and nonzero gradients, with 1,843,200 LoRA
parameters. No policy checkpoint is modified by this probe.

The fit objective alternates an instruction-grounded expert pair and a
policy turn pair. Expert pairs compare the same visual history under the
correct and a verified wrong natural instruction, or compare the goal
view with the start. Policy pairs are sampled with equal probability
from at least one-meter progress and regression turns. STOP labels are
used only outside the 3.0--3.5 m ambiguity interval. Scenes are sampled
uniformly within each source to reduce repeated-episode dominance.
Training uses seed 11, 512 microsteps, gradient accumulation four, and
development checkpoint checks every 128 steps. The checks use fixed
hash-selected 64 expert and 128 policy histories; the selected model's
full development set fixes a STOP threshold at at most 10% development
false positives. If the full development STOP AUC, instruction swap,
balanced progress, regression-only accuracy, or STOP recall misses its
respective gate, the candidate is rejected **without opening the locked
audit**. A development-passing model then tests the previously stated
STOP/grounding/progress gates and additionally requires at least 100
regression pairs and >=60% regression-only rank accuracy. No audit
gradient, checkpoint selection, or threshold tuning is allowed.

The training and one-time audit wrapper uses GPU 3 after the three-GPU
Habitat replay. Even a passing representation is only permission to run
the two-step group-four RL wiring smoke; navigation performance is
unknown until paired val-unseen evaluation.

If the scalar progress head fails on development regressions, the next
candidate is a **local pairwise change model**, not a larger rollout
group. Freeze the selected cross-modal encoder, cache its full-history
states once on GPUs 0/1/2, and fit an antisymmetric scorer of the two
consecutive states, so reversing the pair reverses its score. Balance
at least one-meter progress and regression pairs by scene and episode;
select its margin/confidence on development scenes. Keep the STOP head
and its development threshold separate. The current audit stays closed
until both heads pass development. The same held-out recovery and STOP
gates then apply once; a pairwise progress score is an action-level
reward estimate, not a potential-shaping guarantee.

The fixed 512-microstep LoRA's small development checks improved STOP
and instruction grounding but not recovery: at step 512, STOP AUC was
0.909, natural instruction-swap accuracy 90.6%, one-meter forward rank
83.2%, and one-meter regression rank **35.3%** over 68 regression pairs.
Its full-development pass completed, but the first launcher hit a
`NameError` while constructing the final checkpoint because `digest`
was not imported. No checkpoint or locked-audit result was written.
The import is fixed; the same deterministic run is being repeated, now
writing the selected adapter/head checkpoint after each small
development check so a later failure cannot discard all trained weights.
The incomplete first run is not a final model result.

As a cheap representation diagnostic, `cache_pairwise_policy_states.py`
cached frozen navigation-SFT states for fit/development real progress
and regression pairs on GPUs 0/1/2 while the LoRA run used GPU 3.
The cache audit verified all 576 fit and 241 development trajectories
with a qualifying pair and their feature files. A CPU-trained
antisymmetric pairwise head selected at epoch 10 obtained 75.1%
forward and **46.6% regression** accuracy on development, balanced
60.8%; it failed both recovery gates. The locked audit remained
unopened. `pairwise_base_head_development.json` and
`pairwise_base_cache_audit.json` retain these negative diagnostics.
Testing the same pairwise architecture on the LoRA-adapted states is
the next conditional step if the repaired scalar model fails its full
development screen.

## Repaired LoRA and pairwise development outcomes

The repaired 512-microstep LoRA run completed and selected step 512. Its
full development result was STOP AUC 0.9214, 94.41% natural instruction
swap accuracy, 79.58% forward rank accuracy, and **31.06% regression
rank accuracy** over 161 regressions. The development gate failed, so
the locked model audit and paired unseen navigation set were not opened.
The protected interim checkpoint and final checkpoint have identical
144 adapter tensors and six head tensors; their file hashes differ
because their metadata differs. The equivalence report and development
result are in `ordinal_progress/policy_preference/`.

Using frozen step-512 LoRA states, a CPU antisymmetric pairwise head
selected epoch 1 and reached 70.24% forward and **52.80% regression**
accuracy on development (balanced 61.52%). Its predeclared 75% balanced
and 60% regression gates both failed. Feature cache coverage was 576
fit and 241 development trajectories, 1,858 and 819 states. The locked
audit was again kept closed. The base and adapted pairwise results
suggest that a single trajectory's local score is insufficiently
reliable; they are negative diagnostics, not navigation evaluations.

## Complete group-four relative screen

The next algorithm uses the **same four-rollout group** generated for
each training episode. It compares candidate histories at equal action
turns under one instruction and start. The pairwise loss depends on the
difference in their history representations, so scene, instruction, and
turn index are shared nuisance factors. A separate temporal comparison
loss requires the score to increase when a candidate moves closer and
decrease on regression. At deployment, the score is computed on one
history; no other candidate or privileged simulator distance is an
input. Simulator geodesic distances are used only to make train-scene
labels and to audit held-out train-scene ranks. STOP remains a separate
head and receives no auxiliary bonus in the first RL pilot.

`prepare_group_relative_manifest.py` froze 160/40/40 **complete**
groups for fit/development/audit from the three completed seed-11/22/33
outcome-only 128-step group-four rollouts, before intermediate geodesic
labels were replayed. The manifest hash is
`a99a15020b3d7ffe061ea82ab830e4d617e5ad8f90e3fff8979ab80079a26356`.
The scene inventories have 1,108/213/180 eligible complete groups.
Four A800 Habitat jobs replayed the selected 960 trajectories: two fit
shards and one development and audit shard each. Existing verified
trajectory frames are hardlinked only after matching source-rollout,
dataset, record, and terminal-distance identities; 319/960 selected
records overlap the prior replay. The replay audit passed with zero
errors and exact coverage: 640/160/160 trajectories, 1,628/381/398
same-turn pairs separated by at least one meter, and 155/40/39 groups
containing such a pair. The audit split has only 27 distinct episode
IDs across its 40 groups, so later uncertainty estimates must cluster
by episode and scene. These are label-coverage results; no learned
group-relative accuracy is claimed.

The frozen step-512 adapter cached histories at preselected turns 3,
6, 9, and 12 on four A800s (three fit shards, one development shard):
2,084 fit and 525 development states, all verified. Candidate heads
reuse these states on CPU. The head trains with equal numbers of
same-turn and temporal comparisons;
temporal comparisons balance progress and regression. The single
predeclared development screen requires at least 100 same-turn pairs
from 20 groups, 100 forward and 50 regression temporal pairs, at least
70% same-turn and forward accuracy, and at least 60% regression
accuracy. The selected epoch-3 head reached 73.23% same-turn rank
(381 pairs), 73.94% forward rank (165 pairs), and **59.14% regression
rank** (93 pairs) on development. It missed the regression threshold
by one correct pair and was rejected. No policy RL or locked model
audit was run. The cached-feature and head reports are in
`ordinal_progress/policy_preference/`.

An exploratory diagnostic subtracted each group's mean score at two
anchors and tested the relative change only where all four candidates
were present. On the already opened development set, it correctly
classified 31/48 relative improvements and 33/58 relative regressions
across 32 groups. These post-hoc results do not rescue the failed gate
or establish a usable reward. A future candidate needs a genuine
representation/objective change and a fresh confirmatory screen.

A failing candidate does not access the locked model audit
or consume policy-RL budget. A passing candidate gets one locked audit
and a separate STOP/instruction-grounding check before a two-step
group-four RL smoke. This is an algorithmic change to the representation
and reward, not a group-size increase.

The standard policy experiment keeps `rollout.n=4`. A group-eight check,
if later useful, is only a small auxiliary experiment with its own
group-eight outcome control and a fixed rollout budget; it cannot
substitute for the group-four paired comparison. For efficient held-out
evaluation, use a fixed 256-episode paired screen before three-seed
training, reuse only exact-matched checkpoints, and run two inference +
Habitat lanes concurrently for the complete 1,839-episode val-unseen
suite if the screen passes.

## Signed four-rollout transition candidate (predeclared before fitting)

The scalar group score failed because it did not reliably detect a
trajectory's own regression. The next exploratory candidate represents
each **transition** directly with the frozen before/after Qwen history
states. A shared layer normalization, odd difference projection, and
context gate yield a score that changes sign exactly when the two states
are reversed. A within-trajectory loss balances at-least-one-meter
forward and regression intervals. A second loss compares transition
scores at the same turn interval among the four rollouts of one episode,
ranking the transition with larger geodesic progress above the other.
This is a different action-level reward variable, not another scalar
potential or a larger rollout group.

Use only cached turns 3, 6, 9, and 12, so this first screen needs no new
GPU extraction. Label coverage before fitting is 748 forward, 301
regression, and 1,065 within-group transition comparisons in fit;
development has 165, 93, and 261, respectively. Fit uses scene/group
balanced sampling, one seed 11, up to 30 CPU epochs, and a single fixed
architecture. On development require at least 100 relative comparisons
from 20 groups, 100 forward and 50 regression intervals, >=70% relative
transition rank and forward sign accuracy, and >=60% regression sign
accuracy. Epoch selection maximizes the minimum margin to these three
accuracy thresholds. The prior group-scalar candidate already exposed
the development scenes, so this is **exploratory model selection**;
passing it does not prove generalization. Only a passing candidate may
open the locked model audit once. A failing candidate uses no new RL or
val-unseen budget. Any future visual-pair encoder should use the already
replayed RGB files and cache each frame once before spending four A800s
on online policy training.

The signed transition candidate selected epoch 5. Its development
forward sign was 71.52% (165 intervals), regression sign **53.76%**
(93 intervals), and same-interval four-rollout relative rank **67.05%**
(261 pairs). It failed both the 60% regression and 70% relative gates.
The exact antisymmetry check passed, but this frozen history-state
representation still did not generalize well enough to justify opening
the model audit or running RL. The report is
`ordinal_progress/policy_preference/group_transition_head_development.json`.
The next representation test must expose actual before/after visual
content, rather than read only the autoregressive history end state.

## Patch-level visual transition candidate (predeclared before fitting)

The next candidate freezes the local SigLIP-B/16 vision and text towers
and caches their 196 spatial image tokens at turns 3, 6, 9, and 12,
plus a fixed-length 64-token instruction embedding. It uses the already
replayed RGB frames; no Habitat rollout is repeated. An instruction
query attends to patches in the before/after frames, and an odd
difference scorer predicts signed local progress. This keeps spatial
visual evidence that the autoregressive history end state may discard.
The same group-four relative-transition and balanced forward/regression
losses and the same development sample/accuracy gates as the signed
transition candidate apply. The vision/text towers remain frozen in
this first screen so all candidate-head updates reuse one feature cache.

Four A800s cache fit in three shards and development in one shard.
Source hashes, exact frame coverage, tensor finiteness, and scene
isolation must pass audit before fitting. The existing development
scenes have already been inspected for earlier candidate design, so
the outcome is exploratory. If this model passes, test whether
permuting instructions among different-goal episodes of the same
scene lowers its predictions, then open the locked model audit once.
Only an audit-passing, instruction-grounded model may enter two-step
group-four RL wiring. A failed offline screen does not use policy or
val-unseen compute.

The four parallel cache jobs completed without error in about ten
seconds each. `group_visual_cache_audit.json` verifies 640/160
fit/development trajectories, 2,084/525 spatial-token frames, and
160/40 complete groups across 38/8 disjoint scenes. The SigLIP
weights, config, and processor are pinned by SHA-256 in that report.
The patch-transition head may now be fitted; this coverage audit is
not a reward-quality result.

The patch-level head selected epoch 1. Development forward sign was
62.42% (165 intervals), regression sign 66.67% (93 intervals), and
same-interval relative rank 65.90% (261 pairs). It missed the 70%
forward and relative gates. The report is
`ordinal_progress/policy_preference/group_visual_transition_development.json`;
the locked audit and policy RL remain unopened.

## Agreement-gated process signal (predeclared diagnostic)

The two representations have different observed weaknesses. Test a
fixed, uncalibrated agreement rule before any further model fitting:
the frozen group-scalar score's difference across turns and the frozen
patch-visual transition score must **both** be positive to give a
positive process signal, or both negative to give a negative signal;
otherwise abstain. Include intervals with under-one-meter displacement
as false predictions when calculating precision. Development must show
at least 75% positive precision with >=20% forward recall, at least
70% negative precision with >=20% regression recall, and at least 20
decisions in each signed category. Zero is the only score threshold;
there is no development-tuned margin or weight. Report single-head
precision/recall, agreement, and counts by unique episode/group.
This is a post-hoc exploratory diagnostic on already inspected scenes.
Only if it passes may a fixed implementation open the locked audit
once. No online group-four RL or val-unseen evaluation follows a failed
diagnostic.

The fixed agreement rule failed. On 365 development intervals (165
forward, 93 regression, 107 neutral), it issued 140 positive decisions
with 60.71% precision and 98 negative decisions with 43.88% precision;
forward/regression recalls were 51.52%/46.24%. Both predeclared
precision gates failed. `agreement_diagnostic.json` includes the two
individual model baselines, complete decision counts, and episode/group
coverage. Do not tune new score thresholds on these exposed scenes or
run group-four RL from this process signal.

## Independent STOP-only representation audit (predeclared)

The navigation-SFT LoRA's progress head failed, but its separate STOP
head passed the **development-only** STOP and instruction checks:
AUROC 0.9214, natural wrong-instruction endpoint preference 94.41%,
and 74.14% recall at a threshold fixed for 9.79% development false
STOPs. Test this as a distinct termination-decision mechanism, without
using the rejected progress head. The selected step-512 checkpoint and
threshold -0.055309150367975235 are frozen; no audit-based threshold
selection, model update, or margin sweep is allowed.

The one-time model audit uses the 127 expert histories and 320 policy
trajectories from the eight held-out R2R-train scenes. Before scoring,
source records show 123 safe wrong-instruction expert endpoints and
130/179 unambiguous policy positive/negative endpoints. Require at
least 100 distinct expert trajectories, 100 safe swaps, 50 policy
positives, and 100 policy negatives. The fixed-threshold gate requires
pooled STOP AUROC >=0.80, natural swap accuracy >=0.75, pooled false
STOP rate <=10% with recall >=50%, wrong-instruction endpoint false
STOP rate <=15%, policy far-endpoint false STOP rate <=10%, and policy
goal-endpoint recall >=50%. Report every class count and error rate.
The same audit scenes were used as fit data by an earlier *different*
small-readout study, so this is not a wholly untouched research-wide
test. For this newly initialized LoRA, they were excluded from fit and
development. Opening this audit for STOP-only means it cannot later be
presented as a locked progress-model audit. Only a passing STOP-only
audit can justify a group-four policy smoke, with the original outcome
reward preserved and STOP intervention isolated from movement reward.

The fixed STOP-only LoRA failed its one-time audit. Across 127 expert
trajectories and 320 policy trajectories, AUROC was 0.8944 and pooled
recall was 75.10%, but pooled false STOP rate was **16.08%** against the
10% gate. Wrong-instruction endpoint false STOP rate was **26.83%**
against 15%, and policy far-endpoint false STOP rate was **20.11%**
against 10%. The frozen threshold must not be tuned on this audit.
`stop_only_lora_locked_audit.json` contains counts and the complete
gate. No STOP intervention or navigation improvement is claimed.

## Outcome-grounded first-action preference: group four

Because the local process signals failed their offline gates, test an
algorithmic change that uses only the task's terminal outcome. In each
completed four-rollout R2R-train group, select one success and one
navigation failure whose *first* executed responses differ. They have
the exact same episode, instruction, and initial RGB observation. A
fixed SHA rank selects one pair when several qualify. The preferred
label is task success, not a geodesic or semantic-verifier score. This
is a **correlational trajectory outcome label**: later decisions also
affect the outcome, so a successful rollout does not establish that its
first action caused success. Do not present the pair label as a causal
action advantage.

`preflight_group_success_preferences.py` found 1,573 exact-shared-prompt
success/failure pairs in fit, 284 in development, and 239 in audit.
`prepare_group_success_preference_manifest.py` froze one pair per group,
giving 474/86/73 fit/development/audit groups from 37/8/8 disjoint
train scenes and 235/45/37 unique episodes. The manifest SHA-256 is
`286bcf78b2564c77bde2afd18d1bf87a175038aa81430fa2269b3e197bd2e232`.
Repeated groups from the same episode across seeds are correlated;
analysis must cluster by episode or scene. The audit scene IDs were
exposed by prior candidate studies, so this is a model-held-out
exploratory audit, not a pristine research-wide test.

1. Verify and reuse one initial image per episode from the earlier
   policy-history replay. Its manifest, scene, instruction, and image
   hashes must agree across variants. Render only episodes missing from
   that cache. The preflight found 199/235 fit episodes reusable and
   all 45/45 development and 37/37 audit episodes reusable. Cache once,
   then share it across every model trial. Raw RGB stays on the
   experiment host.
2. Score the frozen navigation SFT's preferred and rejected first
   responses under its original first-turn visual prompt. Report both
   mean action-token log probability and total log probability, plus
   response lengths. This is a diagnostic baseline, not an offline
   reward quality claim. Use development only; keep audit unopened.
3. If the signal is viable, fit one instruction-and-image-conditioned
   LoRA preference model from the navigation SFT, using fit scenes
   only, one frozen pair per four-rollout group. Limit initial fitting
   to one GPU and a small fixed checkpoint grid. The first candidate
   uses rank-8 `q_proj`/`v_proj` LoRA, DPO beta 0.1, learning rate
   2e-5, gradient accumulation four, and two passes over 474 groups.
   Weight each group's loss by inverse episode frequency to avoid
   treating repeated seeds as independent. Score checkpoints after
   256, 512, and 948 pairs. Compare teacher-forced response log
   probabilities including the assistant end token; all 86 development
   pairs have equal preferred/rejected action-token counts. Select by
   development preference accuracy, requiring at least 60% and at least
   five percentage points over the frozen SFT. Check action validity
   and length effects. A candidate meeting this screen may open audit
   once. Require at least 60% audit accuracy and a nonnegative
   episode-clustered accuracy difference to the frozen SFT; otherwise
   stop before online policy training. These are go/no-go rules, not
   claims of navigation improvement.
4. A passing offline model enters a two-step **group-four** policy
   smoke with the original outcome reward and matched outcome-only
   control. Then use the already frozen 256-episode val-unseen manifest
   for paired 64-step evaluation. Require valid action formatting,
   exact episode coverage, positive paired SR and SPL, and no severe
   STOP regression before any three-seed 128-step expansion. Any
   val-unseen selection is exploratory. Group size four is the primary
   setting throughout. A later small group-eight check needs its own
   identical-budget group-eight control and cannot substitute for the
   group-four comparison.

Use the four A800s according to the bottleneck: one Habitat GPU for
only missing initial views, up to two GPUs for independent frozen-SFT
score shards, and one free card for diagnostics. When the offline model
trains, reserve only the needed card; do not start duplicate Habitat
replays. During online group-four training the validated four-card
layout is saturated; the paired 256-episode evaluation can instead run
two inference/Habitat lanes concurrently after training. Record GPU
time, cache coverage, and exact checkpoint hashes. No val-unseen
outcomes enter offline label construction or model selection.

The first image pass completed 235/45/37 fit/development/audit unique
episodes. It reused 199/45/37 verified initial frames and rendered only
36 missing fit frames. Four parallel frozen-SFT score shards covered
all 474 fit pairs; two shards covered all 86 development pairs. SFT
preferred the successful first response in 248/474 fit pairs (52.32%)
and 51/86 development pairs (59.30%). All responses in these pairs
have equal action-token counts, so total and mean action-token ranking
agree. The 95% episode-cluster bootstrap interval for development is
47.78%–70.89%; this is a weak offline signal, not evidence of
navigation gain. Exact coverage, cache hashes, and cluster intervals
are in `group4_first_action_sft_baseline_analysis.json`.

The fixed rank-8 DPO first-action model also failed its development
gate. Its 256/512/948-pair checkpoints chose the successful response
in 52/49/50 of 86 pairs, versus 51/86 for frozen SFT. The selected
256-pair checkpoint's +1.16-point difference is below the predeclared
+5-point requirement. The selected output shifted only one pair's
sign, and mean reference-relative response margin moved by only
0.008 nats. `group4_first_action_dpo_development_report.json` and
the per-checkpoint rows preserve the result. No model audit, policy
RL, or val-unseen test follows this failed candidate. This result
does not establish that larger adaptation or different credit
assignment could not work; the first-action outcome label is
correlational and the sampled fit data have only 235 distinct episodes.

## Next screen: group-four future-success value at matched turns

First-action terminal labels assign credit too early in the trajectory.
Test whether a frozen instruction/history representation contains
future-success information **after** the rollouts have diverged. Use
the already audited four-rollout cache at turns 3 and 6. Require a
later executed motion turn so each compared state is preterminal. At
each anchor, compare the visual-language hidden state of a successful
rollout to a navigation-failure state from the same episode and turn.
Fit a *linear*, antisymmetric outcome-value score on fit-scene state
differences with a fixed L2-regularized logistic objective. This
changes both the representation readout and the reward target: rank
future task success within a group at a matched decision time, without
using geodesic process labels. Only the cached visual-language state
is an input; terminal outcome constructs a fit label. The encoder was
adapted in an earlier study and the development scenes have been
inspected for other hypotheses, so this is exploratory reuse.

Before fitting, the source contains 640 fit trajectories in 160
complete four-rollout groups and 160 development trajectories in 40
groups. There are 240/201 fit success-versus-failure comparisons at
turns 3/6 from 73/67 mixed groups, and 58/45 development comparisons
from 18/17 mixed groups after the preterminal filter. Multiple
comparisons in a group are
correlated. Audit exact source/cache hashes and scene separation.
Use fit-only coordinate scaling of same-group state differences (a
common centering term cancels), unit-norm differences, group-balanced
logistic loss, L2 coefficient 0.01, seed 11, both anchors, and a
single linear probe; do not search many heads on the exposed
development split.
Require at least 100 matched development comparisons from at least
15 groups and >=70% comparison-weighted *and* group-macro success
ranking before caching untouched-for-this-head audit states. A
passing development score is still an offline proxy. Audit must have
at least 100 comparisons from 20 groups and reach >=70% in both
comparison-weighted and group-macro ranking. Only after that, test
natural different-goal instruction swaps and validate group-four
turn-wise credit assignment. Positive paired SR/SPL against the
same-budget group-four outcome control is still required to scale.
The audit scenes are not pristine research-wide because previous
studies exposed them.

The fixed linear probe passed the exploratory development gate. It
ranked 342/441 fit and 75/103 development comparisons correctly;
development group-macro accuracy was 72.87% across 18 mixed groups.
The development group-cluster 95% interval was 64.22%–81.05%, and
turns 3 and 6 separately had 38/58 and 37/45 correct comparisons.
The fitted weights SHA-256 is
`014586d732cd15a095425a67940930a38c9813e6fdb564b4d026aefe0199330f`.
The one-time audit is authorized by the frozen gate, with 131
preterminal comparison opportunities from 22 audit groups before
feature extraction. Cache the 160 audit trajectories in four
independent 40-trajectory A800 shards, then independently verify all
records and source hashes before scoring the frozen weights. This
offline ranking is not a navigation result.

The four cache shards completed, and the exact audit independently
verified all 160 trajectories from eight scenes. The frozen readout
ranked 98/131 comparisons correctly (74.81%) across 22 mixed groups;
group-macro accuracy was 75.08% and the group-cluster 95% interval was
64.89%–84.03%. Turns 3 and 6 gave 50/72 and 48/59, respectively.
It passed the two fixed 70% ranking gates. The source and weight hashes
are in `group4_future_success_locked_audit.json`. This is a
model-held-out train-scene result with research-wide scene reuse, not a
navigation improvement.

Before using this value as reward, freeze one natural wrong-goal
instruction per audit episode by SHA rank, from the same R2R-train
scene with a goal at least 4 m from the original and a different
instruction. Prefer an alternative starting within 0.5 m when one
exists, then SHA rank within that tier. Re-encode the **same**
preterminal histories with that instruction and the same frozen
encoder/readout. Report same-start
alternatives separately; only five of 27 audit episode IDs have a
different-goal instruction within 0.5 m of the original start, so a
same-start-only gate would be underpowered. For all 131 original
success/failure matched-turn comparisons, the wrong-instruction
ranking accuracy must fall by at least 10 percentage points and at
least 60% of the 22 mixed groups must have lower average preference
margin. This is an instruction-dependence check, not independent
semantic ground truth. A failure prevents online RL from this reward.

The four instruction-swap shards covered all 160 audit trajectories,
40 groups, and 131 matched comparisons. Recomputed original-instruction
scores exactly matched the locked 98/131 audit result. Wrong-goal
instruction ranking was 89/131 (67.94%), a **6.87-point** drop against
the fixed 10-point gate. Mean preference margin fell in 15/22 mixed
groups (68.18%), satisfying that separate gate. No mixed comparison
came from the five episodes with an available near-identical start;
the tested pairs therefore use same-scene but different-start natural
instructions. `group4_future_success_instruction_swap_analysis.json`
records the failure and a group-cluster interval for the accuracy drop.
Do not wire this value into online reward or describe it as verified
instruction grounding. The next candidate must train an explicit
instruction-versus-different-goal contrast or use a cleaner same-start
counterfactual dataset, then be evaluated under a newly declared
protocol that discloses reuse of these model-audit scenes.

An efficient next-data option already exists: under the **same scene
partition as the group-four comparison cache**, the verified expert
history collection contains 596/161/123 fit/development/audit
trajectories of at most 12 turns with safe natural same-start,
different-goal instructions (`stop_history_lora_scene_split.json` and
`stop_history_label_audit.json`). The earlier 661/111/108 counts refer
to a different expert partition and must not be substituted here. Their
RGB histories need no new Habitat replay. Re-encode only the required
correct/wrong instruction states with the frozen navigation encoder,
sharded across the A800s, and cache each state once. A new readout can
combine fit-scene four-rollout future-success comparisons with these
fit-scene instruction contrasts, while development tests the two
objectives separately. Freeze the new loss, sample identities, and
thresholds before fitting; the existing audit scenes have been
exposed and must be described as exploratory for a new candidate.

## Joint outcome and same-start instruction readout (frozen screen)

`prepare_group4_joint_value_manifest.py` independently matched the
scene partitions and audited source records, same start poses, and
different goals. It froze 596/161/123 safe natural instruction
contrasts across 38/8/8 fit/development/audit train scenes; manifest
SHA-256 is
`db1ca63465012e426089a76dc161198d033da3b0b821add22b79783b3ed489ef`.
The group-four outcome comparison cache remains unchanged at 441 fit,
103 development, and 131 already-opened audit comparisons. No new
simulator rollout or val-unseen result is used to construct labels.

Keep the same frozen Qwen2.5-VL-3B navigation-SFT LoRA encoder as the
group-four comparison cache. For each selected expert trajectory,
encode its final RGB/action history twice, with the correct and
the natural same-start different-goal instruction. Cache each state
once, with exact record and encoder hashes. Fit and development are
sharded over the four A800s; defer expert audit extraction until the
development gate passes. The existing group-four states need no
re-encoding.

Fit one linear readout on normalized hidden-state differences. Use
the *fit-only* coordinate scale from the previous group-four probe;
optimize the mean within-group successful-versus-failed logistic loss
plus the mean correct-versus-wrong expert logistic loss with **equal
task weight**, and L2 coefficient 0.01. Weight outcome examples by
inverse group comparison count and expert examples by inverse scene
frequency. Optimize this convex objective with at most 200 LBFGS
steps, seed 11. Do not tune the task weight, regularizer, or epoch
after looking at development. Report the previous outcome-only linear
readout on the same development expert pairs as a fixed baseline.

Development must have >=100 outcome comparisons from >=15 groups and
>=100 safe expert contrasts from >=8 scenes. Require group-four
success ranking >=70% both comparison-weighted and group-macro, and
same-start correct-instruction preference >=75% pair-weighted and
>=70% scene-macro. If any gate fails, stop before expert audit or
online RL. If it passes, run one model-held-out but research-wide
reused audit: the same outcome ranking floors, >=75% expert
correct-instruction preference, and >=70% expert scene-macro. Then
recompute the already frozen policy wrong-goal instruction check: the
131 preterminal matched-turn comparisons must lose >=10 percentage
points of ranking accuracy, and at least 60% of the 22 groups must
lose mean margin. Only if all of these pass can this reward enter
the two-step group-four turn-wise advantage wiring smoke. Navigation
claims still require same-budget paired SR/SPL improvement.

Four A800 extraction shards completed and the independent cache audit
verified 596 fit expert pairs (1,192 states) and 161 development pairs
(322 states), with matching scene and encoder hashes. The fixed joint
readout passed its development floors: group-four outcome ranking
74/103 (71.84%), group-macro 71.76%, and same-start expert instruction
preference 152/161 (94.41%), scene-macro 96.18%. The prior
outcome-only readout scored 75/103 (72.82%) and 150/161 (93.17%) on
these exact development examples. Thus the joint readout trades one
outcome comparison for two instruction comparisons; this is not a
substantial validated improvement. Its weights SHA-256 is
`d8be765b4e6837af0dfb467a2935d6238c08b798182ea24b7caef9578e00ce45`.
The fixed gate nevertheless permits one model audit. Extract only the
123 previously selected expert audit pairs in four GPU shards; reuse
the already audited group-four policy states. Do not train or select
another readout on the audit scenes.

The one-time model audit verified 160 group-four trajectories, 131
matched preterminal comparisons across 22 groups, and 123 same-start
expert contrasts across eight scenes. The frozen joint readout ranked
96/131 (73.28%) group outcomes and 105/123 (85.37%) correct expert
instructions, passing its stated offline floors. The expert cache audit
verified all 246 correct/wrong states and their encoder/record hashes.
These scenes were previously used across the broader research, so this
is a model-held-out check for this readout, not a pristine research-wide
test or a navigation result.

The final frozen policy-history instruction check extracted and cached
all 160 natural wrong-goal states over four A800 shards. Independent
cache analysis verified 300 preterminal states and reproduced the
locked 96/131 correct-instruction result. Wrong-goal ranking was
90/131 (68.70%): a **4.58-percentage-point** decrease, below the
predeclared ten-point requirement. Average margin fell in 14/22 mixed
groups (63.64%), passing that separate requirement. The group-cluster
95% interval for the accuracy decrease is [-3.36, 12.03] points.
None of the effective 131 matched pairs had a near-identical-start
alternative; these natural swaps were same-scene different-start goals.
An unchanged-cache rerun of the previous outcome-only readout exactly
reproduced its 98/131 and 89/131 counts. The joint readout therefore
**fails the instruction-dependence gate and must not be wired into RL**.
See `group4_joint_value_locked_audit.json` and
`group4_joint_policy_instruction_swap_analysis.json` for frozen counts,
source hashes, and gate outcomes.

The next representation candidate should train **instruction
discrimination at preterminal prefixes**, where the reward would be
used. The current expert contrast acts at the final expert history,
which may explain why it transfers weakly to partial policy histories.
Reuse the existing same-start expert RGB histories and their natural
wrong-goal instructions, but extract paired correct/wrong states at
multiple nonterminal action prefixes. A prefix contrast must be
grouped by episode and scene and cannot use the open audit pairs for
model selection. Keep `rollout.n=4` for any later paired RL test.

## Prefix instruction contrast screen (frozen before development scoring)

Use the **same 596 fit and 161 development expert episodes** from the
joint manifest and the unchanged Qwen2.5-VL-3B navigation-SFT LoRA
encoder. Extract correct and natural same-start different-goal states
at action-history counts 3 and 6 only when that count is strictly
before the final expert motion turn. Cache all states by episode,
instruction, prefix, source hash, and encoder hash. Shard the 596 fit
episodes three ways and the 161 development episodes on the fourth
A800; this balances roughly 2,820 forward passes without replaying
Habitat or touching val-unseen. Independently verify exact cache
coverage before fitting. The existing group-four policy features and
fit-only coordinate scale are reused without inference.

Fit exactly one linear head using normalized state differences:
group-four success versus failure at matched turn plus correct versus
wrong same-start expert instruction at nonterminal prefix. Use equal
task weight, L2=0.01, at most 200 LBFGS iterations, seed 11, inverse
group frequency for policy outcomes, and inverse scene and per-episode
prefix count for expert contrasts. No hyperparameter choice from the
development or previously opened audit set. Report the frozen previous
joint head on the exact same development examples.

Development must contain at least 100 policy outcome comparisons from
15 groups and at least 250 expert prefix contrasts from eight scenes.
The new readout must rank at least 70% of outcome pairs and groups,
at least 75% of expert prefix instruction contrasts and 70% scene
macro, and lose no more than two outcome comparisons versus the frozen
previous joint head. It must also pass a **new development policy
wrong-goal instruction check**, selected before scoring from natural
R2R-train alternative instructions: at least ten percentage points
lower outcome ranking under swapped instructions and lower mean
outcome margin in at least 60% of effective four-rollout groups. If
any gate fails, stop without online RL. If all pass, use a new
episode-held-out policy check selected before scoring; disclose any
research-wide scene reuse. Only a same-budget `rollout.n=4` paired
navigation result can establish a real gain.

All four extraction shards completed. An independent pass verified
596/161 expert episodes, 1,107/303 nonterminal instruction contrasts,
2,214/606 states, and disjoint 38/8 fit/development scenes, with exact
record and encoder hashes. The single fixed linear readout kept
group-four development outcome ranking at 74/103 (71.84%) and group
macro at 71.94%. Its expert-prefix correct-instruction ranking was
203/303 (67.00%), scene macro 65.71%. The frozen previous terminal
joint readout scored 196/303 (64.69%) on these same prefix examples.
The small 2.31-point prefix increase did not meet the predeclared
75% pair or 70% scene-macro requirements. The candidate stops before
policy instruction swapping, model audit, or online RL. Its weights
and report are retained to reproduce the negative result, not to
claim an improvement.

For a later encoder-level candidate, the development policy wrong-goal
instructions were frozen from train metadata **without scoring**:
33 distinct episodes, 12 with an available natural alternative start
within 0.5 m, including six of the 18 mixed-success four-rollout
groups. Manifest SHA-256 is
`ef5a98162de2958cb7ffd0d2fa5413abed3bcdf484b14dd322a1c0951a7ad42c`.
This gives a cleaner same-start transfer diagnostic than the already
opened audit's zero effective same-start groups. It is development
data and must not be presented as a blind audit.

## Encoder-level prefix contrast (fixed pilot, before training)

The frozen-linear screens suggest a representation bottleneck at
partial histories. In a **single-seed pilot** (11), initialize the
Qwen2.5-VL-3B reward encoder from the existing navigation-SFT LoRA
checkpoint and keep the previous joint readout vector and fit-only
coordinate scale **frozen**. Update only its 1.84M LoRA parameters.
Alternate one four-rollout successful-versus-failed pair at a matched
preterminal count (3 or 6) with one correct-versus-natural-wrong
instruction pair on the *same* expert RGB/action prefix. Both use
`softplus(-normalized_pair_margin)` with equal task frequency. Sample
uniformly by fit scene, then uniformly within that scene. Use 512
microsteps, gradient accumulation 4, AdamW learning rate `5e-5`,
weight decay `0.01`, gradient clip 1.0, and seed 11. Save only the
fixed final adapter; do not select a step or tune hyperparameters on
development. Source episode membership and old readout hashes must
match the audited 441 fit group comparisons and 1,107 fit expert
prefix contrasts. No geodesic distance or val-unseen information is
passed to the encoder.

First run an 8-microstep wiring smoke (discard its weights), then
the fixed pilot. Re-encode development prefixes with the final adapter
in independent GPU shards. Evaluate the **unchanged** frozen readout:
at least 70% success/failure accuracy and group macro on 103 policy
comparisons, no more than two fewer correct outcome pairs than the
previous joint readout (74/103), at least 75% correct instruction
preference on 303 expert prefix contrasts and 70% scene macro. If
these pass, evaluate the pre-frozen 33-episode development natural
wrong-goal policy swaps. Require at least ten points lower outcome
ranking and lower mean margin in at least 60% of effective groups;
report the six same-start mixed groups separately, without treating
them as an independent audit. Any failure stops this pilot before
online RL. Only a later same-budget group-four paired navigation
comparison can establish a benefit.

The 8-microstep wiring smoke passed, and the fixed 512-microstep pilot
completed without development checkpoint selection. Its adapter SHA-256
is `a51092ac2050cb8ffaf52b823b183fcd19afda7d2215fd233e4252a09a80ad45`.
Four A800 shards re-encoded 160 development group-four trajectories
(303 preterminal states) and 161 expert trajectories (303 same-start
prefix contrasts, 606 states); the independent cache audit matched
all source and encoder hashes. With the unchanged old readout, the
adapted encoder ranked 77/103 (74.76%) development successful-versus-
failed comparisons, group macro 74.44%, versus the old 74/103 and
71.76%. Same-start prefix instruction discrimination improved from
196/303 (64.69%) to 214/303 (70.63%), with scene macro 68.78%.
The latter misses the fixed 75% pair and 70% scene-macro gates, so no
development policy swap, model audit, online RL, or val-unseen claim
follows from this pilot. The positive changes are exploratory offline
signals only. `group4_prefix_contrast_lora_seed11_training.json`,
`group4_prefix_contrast_lora_seed11_cache_audit.json`,
`group4_prefix_contrast_lora_seed11_development.json`, and the frozen
adapter preserve the evidence.

A fixed descriptive diagnostic, SHA-selecting two fit expert prefixes
per scene, ranked 62/76 (81.58%) same-start instruction pairs with the
adapted encoder versus 53/76 (69.74%) with its initial encoder. The
development result is 214/303 (70.63%), an approximately 11-point
adapted fit/development gap, though these sample sets differ in size
and difficulty. Some fit examples may have been sampled during
training. This argues against simply increasing optimization steps
on the same objective; the next algorithm should target goal-conditioned
**change across time** and cross-scene transfer. This diagnostic is
not a held-out metric or a basis to waive the stopped gate.

## Bidirectional crossed-trajectory representation (frozen pilot)

The previous pilot improved both offline metrics but retained a
fit/development gap. A new **label-only** manifest pairs two expert
trajectories whose instructions specify different goals from the
same exact initial RGB image; it requires different sixth-turn RGB
images and at least seven motion turns on both routes. Its SHA-256 is
`9a17cfcdd23d27295e35e80dff677d2a96daa20eaf03e3ba5133fc9333fc13fe`.
There are 147 fit crosses across 35 scenes and 56 development crosses
across eight disjoint scenes. Four encoder states per cross cover both
trajectories with both instructions at count 6. Selection uses source
records and image hashes only, with no model score or val-unseen data.

Initialize again from the original navigation-SFT LoRA and keep the
old joint linear readout and fit-only coordinate scale frozen. A
crossed sample produces a 2-by-2 score matrix. Minimize four logistic
rank losses: each trajectory under its own versus the other
instruction (two rows), and each instruction on its own versus the
other trajectory (two columns). Each margin uses the same normalized
hidden difference and frozen readout as the preceding pilot. One
crossed-sample loss is half the sum of its four rank losses. Alternate
two group-four successful-versus-failed fit comparisons with one
crossed sample; sample uniformly by fit scene and then within scene.
Use 384 microsteps, gradient accumulation over each three-microstep
cycle, divide their combined loss by four pair equivalents, and make
128 AdamW updates with learning rate `5e-5`, weight decay `0.01`,
gradient clip 1.0, seed 11. This uses **1,024 model forwards**, 256
outcome pairs and 256 expert row contrasts, matching the prior pilot's
forward count and pair count. Save only the final adapter, no
development checkpoint selection. The additional column comparisons
are the algorithmic change, not a larger group or more model forwards.

After a nine-microstep smoke (three complete update cycles; discard
its weights), complete the fixed train and encode development policy
and expert prefix states in shards. The fixed readout must reach >=70% on the
103 development outcome comparisons and group macro, and at least
75% on the 303 same-start expert prefix contrasts and 70% scene
macro; it may lose no more than two outcome comparisons relative to
the earlier 74/103 frozen joint baseline. Report the previous
encoder-level pilot (77/103 and 214/303) alongside it. If any floor
fails, stop before wrong-goal policy scoring, model audit, or RL. If
they pass, use the already frozen development wrong-goal manifest and
the previous >=10-point ranking-drop and >=60%-group-margin gates.
Any eventual navigation check remains an equal-budget
`rollout.n=4` paired control. Research-wide scene reuse and repeated
development use must be disclosed; these screens alone cannot support
a CVPR performance claim.

The nine-microstep smoke and compute-matched 384-microstep train
completed. The final adapter SHA-256 is
`d7b420f6985328634189e2a66686c3d2900b8d76a1ca4ebcb27c8da4f8c5198d`.
Four A800 development shards and a separate audit verified 160 policy
trajectories (303 preterminal states) and 161 expert trajectories
(303 prefix contrasts, 606 states), with matching record and model
hashes. With the old readout still frozen, the crossed encoder ranked
78/103 (75.73%) four-rollout outcome pairs, group macro 74.95%, and
215/303 (70.96%) expert prefix instruction contrasts, scene macro
69.27%. This is only one additional correct outcome pair and one
additional correct prefix pair relative to the previous compute-matched
encoder pilot (77/103 and 214/303). It misses the fixed 75% prefix
and 70% scene-macro floors, so the policy swap, model audit, online
RL, and val-unseen remain unopened for this candidate.

The method-specific 2-by-2 development diagnostic improved all-four
comparison accuracy from 67.41% to 79.02% over 56 paired crosses;
row accuracy rose from 66.07% to 75.89% and column accuracy from
68.75% to 82.14%. Its scene-cluster interval for the paired all-four
change is [+4.86, +20.75] percentage points. These crosses were used
to design the method and come from repeatedly exposed train scenes.
They show the objective learned its selected contrast, but its gain
did not transfer to the broader prefix instruction gate. Preserve the
negative result and do not convert this diagnostic into a navigation
claim. The next representation test should target instruction-conditioned
**temporal progress** on a broader expert population, not merely add
training steps to this narrow crossed objective.

## Instruction-conditioned temporal potential (frozen pilot)

The new label-only manifest freezes 503 fit and 143 development
expert trajectories across 38 and eight disjoint train scenes. Every
trajectory has at least seven motion turns, a natural exact-same-start
different-goal instruction, and at least one meter of correct-goal
geodesic progress between history counts 3 and 6. Its SHA-256 is
`634e427d696c34fa5e30bc8da2efc90ab4dec4fbb40eb09dd3fb167821871095`.
The geodesic values select reliable intervals only and are never
serialized into an encoder input. The original frozen encoder and
readout rank 342/503 fit and 97/143 development temporal interactions
positive on these selected intervals; this diagnostic is already
exposed and cannot count as a blind result.

Initialize the same original navigation-SFT LoRA and freeze the
previous joint linear readout and coordinate scale. For each expert
trajectory, encode its history at counts 3 and 6 under both the
correct instruction and natural wrong-goal instruction. Define a
row preference at each count (correct score versus wrong score) and
an interaction margin comparing the correct-instruction value change
from count 3 to 6 with the wrong-instruction value change. Normalize
each hidden-state difference by the fit-only coordinate scale and
its L2 norm before the fixed vector projection. Minimize
`0.5*(softplus(-row3)+softplus(-row6)) + softplus(-interaction)`.
Alternate two matched-turn group-four success/failure comparisons
with one temporal interaction. Divide the three microstep losses by
four pair equivalents before backpropagation. Use 384 microsteps,
128 AdamW updates, learning rate `5e-5`, weight decay `0.01`, clip
1.0, seed 11, and uniform scene-then-example sampling. This uses
1,024 model forwards, the same as both preceding LoRA pilots. Save
only the final adapter after a nine-microstep wiring smoke; do not
select a checkpoint on development.

Re-encode development policy and expert prefixes in independent
A800 shards, audit complete state/record/model hashes, and apply the
unchanged fixed readout. Require >=70% on 103 group-four outcome
comparisons and group macro, at least 72/103 correct outcomes, >=75%
on 303 expert instruction-prefix contrasts and >=70% scene macro.
The new temporal interaction must also rank >=75% of 143 development
intervals positive and reach >=70% scene macro. Report both earlier
encoder pilots for comparison. If any gate fails, stop before the
pre-frozen development wrong-goal policy check or online RL. If all
pass, retain the same >=10-point wrong-instruction ranking drop and
>=60%-group margin drop before further testing. Any navigation claim
still needs a same-budget `rollout.n=4` paired control; this pilot
does not use group size as an improvement mechanism.

The nine-microstep smoke and fixed 384-microstep train finished in
444.9 seconds; the final adapter SHA-256 is
`6ce88f940f700ff8cdf6e09faa18c002a77b1376596896e764d6b0cb93643489`.
The separate cache audit verified 160 policy trajectories (303
preterminal states) and 161 expert trajectories (303 prefix contrasts,
606 states), each with matching record and checkpoint hashes. The
frozen readout ranked 79/103 (76.70%) group-four outcome pairs and
76.99% by group macro. It ranked 208/303 (68.65%) expert prefix
instruction pairs, 66.63% by scene macro, and 102/143 (71.33%)
temporal interactions, 71.77% by scene macro. The original encoder
ranked 97/143 temporal interactions; the earlier prefix and crossed
pilots ranked 99/143 and 104/143. Thus this new objective improves
outcome ordering but misses the prespecified 75% prefix and temporal
interaction floors and the 70% prefix scene-macro floor. The
pre-frozen policy-swap check, model audit, online RL, and val-unseen
remain unopened for this candidate. This is a negative development
screen on previously reused train scenes, not a navigation result.

## Evidence-onset branch representation (frozen before training)

A source-image audit found that 83 of 156 checkable development
correct/wrong expert pairs had identical first-three RGB histories,
although they had different final goals. Only 17 of 142 checkable
six-turn pairs remained identical. The original encoder's instruction
preference was correct on 94/159 early and 102/144 later prefixes;
the most recent temporal encoder scored 97/159 and 111/144. Among the
73 development prefixes already visually diverged by turn three, the
temporal encoder was correct on 47. These are post-hoc splits of an
already opened train-scene development set. Exact RGB identity is a
lower bound on route-prefix ambiguity, not proof that an instruction
cannot be grounded from other evidence. Preserve all earlier failed
gates; this diagnosis cannot retroactively qualify their checkpoints.
The source audit and model split are saved as
`policy_preference/group4_expert_path_ambiguity_audit.json` and
`policy_preference/group4_prefix_grounding_posthoc.json`.

Before any new model fitting, a label-only selection froze 295 fit
and 94 development unique same-start, different-goal expert pairs,
with exact RGB divergence at a preterminal turn. It chooses turn six
when available, otherwise turn three. Among these, 73 fit and 44
development pairs have identical three-turn histories and diverged
six-turn histories, furnishing an evidence-onset objective. The
manifest SHA-256 is
`41b465132819ae6060d06b3cf0a6ff964d9be83d3ace265ae39414aa1258f26f`.
The pair selection uses source records and image bytes only; no model
prediction or val-unseen result selected examples. Fit and development
scenes remain disjoint, but the development scenes have already been
inspected repeatedly and this remains exploratory.

The next pilot starts from the original navigation-SFT LoRA, freezes
the same joint value readout and fit-only coordinate scale, and changes
only the encoder LoRA. In each three-microstep cycle, two same-turn
group-four success/failure examples maintain outcome grounding. One
four-forward expert microstep alternates between (i) a two-route,
two-instruction crossed preference at the manifest's evidenced turn
and (ii) an evidence-onset contrast on one route from a pair whose
first three RGBs are identical. The onset loss increases the
correct-versus-wrong instruction margin from turn three to six,
constrains the indistinguishable early margin toward zero, and favors
the correct instruction at turn six. Fix seed 11, 384 microsteps,
128 optimizer updates, 1,024 model forwards, AdamW learning rate
`5e-5`, weight decay `0.01`, gradient clip 1.0, and scene-uniform
sampling. Run a nine-microstep wiring smoke and discard its weights;
save only the final full-train adapter. No development checkpoint
selection.

Use the unchanged group-four development comparisons and a separate
cache audit. Require at least 70% outcome pair and group-macro
accuracy and no more than two correct outcomes below the old 74/103
frozen joint baseline. On all 94 evidence-conditioned development
crosses, require at least 75% all-four matching and at least five
percentage points above the original SFT encoder on the same pairs.
On the 44 onset-eligible pairs, require at least 65% positive
correct-minus-wrong margin increase on both routes combined and at
least 60% scene macro. Report broad early and late instruction
preferences, including identical-history and diverged strata, but
do not make the ambiguous early aggregate an acceptance criterion.
If any gate fails, stop before policy wrong-goal swaps, model audit,
or online RL. If they pass, use the already frozen policy swap and
its >=10-point ranking drop and >=60%-group margin drop gates before
opening the model audit. Only a matched `rollout.n=4` pilot and paired
navigation evaluation can establish an actual gain. A group size
above four is reserved for a later small, equal-rollout-budget
diagnostic after a group-four gain.

The fixed nine-microstep smoke and 384-microstep training completed;
the final adapter SHA-256 is
`79edf05c62ed4314f923ef62eee42d532dff88d7defc6e214d3514bec5a25be4`.
GPU0 trained while GPUs1--3 scored the original encoder on all 94
development branches. After training, GPU0 cached 160 policy histories
while GPUs1--3 scored the final adapter, then cached 161 expert
histories. The independent cache audit verified 303 preterminal policy
states and 606 expert states with exact source and checkpoint hashes.

With the same frozen readout, the candidate ranks 76/103 group-four
outcome comparisons (73.79%, group macro 73.56%). On the 94 branch
pairs, it gets 274/376 individual row/column comparisons right,
versus 251/376 for the original encoder (+6.12 percentage points).
It gets all four right on 37/94 pairs, versus 27/94 originally
(39.36% versus 28.72%). The onset margin is positive on 61/88
route directions, versus 49/88 originally (69.32% versus 55.68%);
scene macro is 72.53%. The broad 303-prefix instruction contrast
remains 208/303 (68.65%, scene macro 67.33%), matching the prior
temporal pilot's count. No navigation checkpoint was evaluated.

The earlier crossed diagnostic called 79.02% `all_four_accuracy` but
counted the four individual margins as 224 separate comparisons;
it also separately reported 26/56 pairs with all four margins correct.
The present frozen gate explicitly evaluates the stricter complete
pair event. The candidate misses that 75% gate by a wide margin;
even under the older pooled-comparison convention, 72.87% is below
75%. Preserve both denominators and do not relax the gate after
inspection. This candidate stops before development policy swaps,
locked model audit, online RL, and val-unseen. The selected-branch
gain did not improve the broad instruction check, so more updates or
larger rollout groups are not justified by this screen. The next
method needs a better goal representation and validation on genuine
policy histories, with a metric definition frozen unambiguously
before training.

## Ordered instruction-clause alignment preflight

The next low-cost representation check used all 512 fit and 128
calibration expert episodes in the existing train-only ordinal
manifest. A deterministic sentence/comma/`then` split produced 1,536
fit and 388 calibration clauses; the hashed source manifest is
`5203e2669fe511c37771812c23fccb1008851d38ef9ed2603d6b80017aa07e2b`.
No instruction text or image is copied into this repository. GPUs 1
and 2 encoded the clauses and entire instructions using the same
frozen SigLIP backbone with **fixed 64-token padding**, avoiding the
previously identified batch-dependent text representation. The
existing frozen image embeddings were reused, with matching backbone
config hashes. Five zero-shot rules were fixed before scoring: whole
instruction, final clause, mean of the final two clauses, monotone
six-frame clause path, and that path minus a repeated-initial-frame
path. This is an exploratory probe on previously used train scenes,
not a new navigation validation.

On the 32 natural same-start, different-goal calibration pairs (64
directional comparisons), the fixed-padding whole-instruction
baseline scores 48/64. The final-clause, last-two-clause, monotone
path, and path-gain rules score 42/64, 42/64, 46/64, and 47/64.
The path-gain rule changes paired accuracy by -1/64 from whole text;
its ten-scene bootstrap difference interval is [-15.63, +12.12]
percentage points. On 128 fit pairs (256 comparisons), whole text
scores 177/256 and path gain 179/256, a negligible fit-only gain.
The source hashes, every pair's margins, and paired scene analysis
are in `policy_preference/clause_alignment_*`. Simple clause slicing
does not provide an instruction-grounding improvement. No learned
adapter, reward service, group-four policy run, model audit, or
val-unseen evaluation was launched for these rules. A further
attempt would need learned state-conditioned visual/phrase grounding
and direct tests on policy histories. Related semantic-prefix progress
work already exists in [Progress-Think](https://arxiv.org/abs/2511.17097),
so a future paper must distinguish its technical contribution.

## Group-four standard and efficient spatial-grounding screen

The standard online comparison remains **four sampled rollouts per
prompt**. All new policy candidates must use the same training examples,
seeds, update count, total generated trajectories, inference budget,
and evaluation episode IDs as a group-four outcome-only control. A
group-eight run is only a small diagnostic *after* a group-four method
gain; it requires its own group-eight outcome-only control and matched
total trajectory/token budget. Changing the group size alone is not
evidence for the representation or reward method.

The next representation hypothesis is spatial, goal-contrastive visual
evidence. Freeze the existing SigLIP-B/16 towers and the fixed-length
clause embeddings. Cache each of the six already replayed expert RGB
frames once, keeping a 7x7 average-pooled grid of the 14x14 frozen
vision tokens. `cache_clause_spatial_features.py` uses the existing
512 fit and 128 calibration records, verifies source/model hashes,
and writes 49 patch tokens per frame. This is **feature extraction**,
not a learned result. A subsequent small scorer can learn whether
ordered clauses acquire spatial evidence along the route, using
natural same-start, different-goal paths as crossed negatives. Its
online reward would be the change in correct-goal evidence relative
to those negatives, with zero reward while the visual evidence is
ambiguous. Terminal success remains a separate outcome term. This
tests an actual representation-to-reward change rather than reward
weight tuning.

Fit-scene features can be cached on three A800s while the fourth
caches calibration; the RGB replay, frozen text features, and existing
group-four control checkpoints are reused. After caching, use one
GPU for the small head and CPU for paired bootstrap analysis. Before
any four-GPU online training, require the new scorer to beat the
fixed-padding whole-instruction baseline's 48/64 natural calibration
directional comparisons by at least four correct comparisons, and
to improve both pair-weighted and scene-macro accuracy. The
calibration scenes have already been viewed in prior research, so
this remains exploratory. Freeze the head before testing preterminal
policy histories and natural wrong-goal swaps; demand a positive
instruction-sensitive gain there, not just expert-path separation.
If these screens fail, do not launch group-four RL or val-unseen.

For a candidate that passes, use successive gates: a two-step wiring
smoke; one 64-step seed-11 candidate/control pair; paired evaluation
on the frozen 256-episode val-unseen subset; then three matched
128-step seeds with all 1,839 val-unseen episodes. Audit unique episode
coverage, inference errors, SR, SPL, and paired differences before
claiming a gain. The small val-unseen subset has been repeatedly used
for research decisions, so its result is exploratory. Evaluation
should shard episodes across independent Habitat workers and keep one
model service per concurrently evaluated checkpoint; do not rerun
simulator replay or feature extraction when a verified cache exists.

The first spatial cache pass completed on four A800s. Independent
`audit_clause_spatial_cache.py` checked all 512 fit and 128
calibration episodes, 3,840 frames, 188,160 pooled spatial tokens,
source/model hashes, finite tensors, and scene disjointness (51/10
scenes). The audit is in `policy_preference/clause_spatial_cache_audit.json`.
No learned spatial scorer, reward improvement, or navigation gain has
yet been established by this cache.

The first spatial scorer is fixed before fitting: normalize each frozen
patch token, map each fixed clause vector into patch space with an
identity-plus-rank-eight residual map, take the best patch evidence at
each of six frames, and maximize a monotone clause path that starts at
the first clause and ends at the last. Subtract the same path on six
copies of the initial frame to isolate newly observed evidence. Add
this spatial evidence to the frozen whole-instruction score after
scaling each component by its **fit-only** crossed-margin standard
deviation. The spatial coefficient is a bounded learned scalar. Fit
only the 128 natural same-start/different-goal fit pairs with the four
crossed route/instruction margins, rank-eight residual, seed 11,
AdamW `1e-3`, weight decay `0.01`, batch 16 pairs, 64 epochs, and
gradient clip 1.0. Use the final epoch only; do not choose a checkpoint
or coefficient on calibration. Report the 32 calibration pairs both
as 64 directional comparisons and strict both-directions-correct
pairs, with scene-macro and paired scene-bootstrap differences against
the fixed whole-instruction baseline. The minimum 52/64 and positive
scene-macro gate above remains unchanged. This is still an exploratory
train-scene screen.

The fixed 64-epoch fit completed in roughly two minutes on one A800.
It perfectly fit all 256 training-direction comparisons, while the
frozen whole-instruction baseline got 177/256. On the 32-pair
calibration split, however, the learned spatial score got **41/64**
directional comparisons and **10/32** strict pairs, below the frozen
baseline's **48/64** and **17/32**. Scene-macro accuracy fell from
75.42% to 63.75%; the paired scene-bootstrap difference interval is
[-29.03, +6.45] percentage points. The fixed >=52/64 and positive
scene-macro gate failed. This strong fit/calibration gap is consistent
with memorization of the small same-start pair set. Do not open
policy-history checks, online group-four RL, or val-unseen for this
checkpoint. Its exact source hashes, final weight hash, fit-only
scales, losses, and per-pair margins are in
`policy_preference/clause_spatial_grounder_screen.json`. A next
attempt should use spatial visual features already aligned to language
by pretraining rather than fit a patch/text map from 128 pairs.

## Frozen region-language alignment screen (fixed before extraction)

The failed rank-eight head learned a map from only 128 natural pairs.
The next candidate uses the same frozen SigLIP image **pooling head**
on local crops, which produces image vectors in the pretrained text
embedding space without fitting patch-to-text parameters. For each of
the same six replayed frames, encode the full frame and four
overlapping two-thirds-width/two-thirds-height crops anchored at the
four corners. Cache five normalized 768-dimensional region vectors
per frame. The existing fixed-64-token full-instruction and clause
vectors, natural pairs, and 51/10 scene split remain unchanged.
Fit frames are split across GPUs 0--2, calibration frames use GPU 3;
the cache is verified by episode, frame, crop geometry, source hash,
model hash, finiteness, and scene isolation. No simulator replay or
learned head is needed for the first screen.

The **primary frozen score** takes the maximum region-to-clause cosine
within each frame, finds the best monotone six-frame path from the
first to final clause, and subtracts that path on six repeated initial
frames. The four crop locations and full-frame vector are all eligible
at each frame. The baseline is the previously fixed full-instruction
last-two-frame whole-image cosine, 48/64 directional comparisons on
32 calibration pairs. Report fit and calibration pair-weighted,
strict-pair, and scene-macro metrics, with paired scene-bootstrap
differences. To open policy-history checks, the primary score must
get at least 52/64 correct and exceed baseline scene-macro accuracy;
later reward/online gates remain unchanged. Full-frame-only ordered
path gain and local last-clause endpoint gain may be reported as
diagnostics, but they cannot select the method after calibration.
The calibration scenes and fixed val-unseen screen have been exposed
in previous research; this is an exploratory representation screen.

The four cache shards completed in 19--25 seconds each. Independent
audit verified 512/128 fit/calibration episodes, 3,840 frames,
19,200 normalized region vectors, exact source/model hashes, and
51/10 disjoint scenes. The frozen primary region path gained only
175/256 fit and **43/64** calibration directional comparisons,
versus whole-instruction **177/256** and **48/64**. Calibration strict
pair counts were 12/32 versus 17/32; scene-macro accuracies were
68.33% versus 75.42%. The paired scene-bootstrap difference interval
for the primary score is [-20.31, +3.33] percentage points. It fails
the prespecified 52/64 and scene-macro gates, so no policy-history,
reward, group-four RL, or val-unseen test follows this score.

The diagnostic full-frame path gained 47/64 and local final-clause
gain reached 50/64 on calibration, but the latter scored 174/256 on
fit versus the baseline's 177/256. The diagnostic was not eligible
for post-hoc promotion and does not show a stable gain. The full
audited cache provenance and per-pair margins are in
`policy_preference/clause_region_cache_audit.json` and
`policy_preference/clause_region_probe.json`.

## Route-level visual-language teacher screen (fixed before query)

Both SigLIP global and regional clause scores failed the spatial
grounding gate. A distinct next representation test uses the already
available local Qwen3-VL-8B-Instruct as a **frozen route matcher**,
without new labels or external image transfer. For each of the 32
calibration same-start/different-goal natural pairs, show six ordered
expert RGB frames from one route and both complete instructions.
Ask which instruction matches the observed route, including its later
destination evidence. Do not show dataset IDs, success labels, goal
coordinates, or model-predicted events. Use deterministic decoding
and first-token A/B logit margins. Query **both A/B orderings** for
each route and average their correct-instruction margins to cancel
letter-position bias. This costs 128 six-image forwards and gives
64 order-averaged directional decisions. A tie is incorrect.

Pin the prompt, six frame files, 448-pixel maximum image edge,
model revision/config/tokenizer hashes, 32 pair identities, decoding
settings, and output parsing before any call. Run four independent
16-route GPU shards, first one smoke route per shard, then the fixed
calibration set. The previously established frozen SigLIP baseline
is 48/64. Require at least 52/64 and higher scene-macro accuracy,
plus at least 75% agreement of the two orderings on the direction of
preference, before spending 256 fit-direction queries or testing
policy histories. Report the 32 strict-pair count and a paired
scene-bootstrap difference. These train scenes are already exposed,
so passing is exploratory; it cannot establish reward or navigation
gain. If it passes, test whether the same scorer separates correct
and natural wrong-goal instructions on **preterminal policy** views,
then whether the evidence margin changes at actual decision turns.
Only such policy-history evidence may justify an n=4 reward pilot.
The earlier 8B event-completion audit is a different task and its
single-AI blind labels are not human ground truth.

The frozen calibration pass completed with all four verified shards:
57/64 correct versus the fixed SigLIP 48/64, 25/32 strict pairs
versus 17/32, and 57/64 order agreement. Scene-macro accuracy was
89.58% versus 75.42%; the ten-scene paired bootstrap difference
interval was [+3.13, +25.81] percentage points. It passed all three
prespecified exploratory gates. This does not verify policy-state
reward quality. Before moving to policy histories, score the already
fixed 128 fit pairs (256 route decisions), with no prompt edits;
require at least 190/256 correct against the frozen fit baseline's
177/256, higher scene-macro accuracy, and at least 75% ordering
agreement. This is a consistency check on training scenes, not a
second independent holdout. Only if it passes should the teacher be
queried on preterminal policy histories with natural wrong-goal
instructions.

The fixed fit consistency pass also passed: 226/256 correct versus
177/256 SigLIP, 98/128 strict pairs versus 52/128, 232/256 A/B
order agreement, and 88.82% versus 70.48% scene-macro accuracy.
Its 31-scene paired bootstrap interval for the direction-level gain
is [+13.71, +24.58] percentage points. The frozen teacher therefore
separates expert routes from exact same-start wrong-goal instructions
across both train-scene partitions; neither partition is an online
reward or navigation result.

Before querying policy histories, select every complete group-four
rollout whose original R2R-train episode has a natural **exact same
start pose** wrong-goal instruction at least 4 m from the original
goal, using SHA rank to choose one wrong episode. Do not use model
scores or val-unseen. The label-only manifest `qwen3_policy_route_manifest_v1`
contains 55/13/5 fit/development/audit groups, 220/52/20 rollouts,
and 203/48/20 preterminal six-turn states. Development has only
8 successful six-turn states and 15 within-group success/failure
comparisons across six mixed groups; the audit split has no successes
and cannot test outcome ranking. These counts come from exact replay
records, not the earlier source-plan turn counts.

The policy screen uses the *same* frozen route-matching prompt and
local Qwen3-VL-8B weights, with original versus exact-same-start wrong
instruction in both A/B orders. Also score six repetitions of the
initial view as a no-motion diagnostic. At each preterminal turn 3 and 6,
sample six views from the observed initial-to-current history at
indices `floor(i*t/5+0.5)` for `i=0..5`; repeated early views are
disclosed. Never include future frames. The score is the mean
correct-instruction A/B logit margin across the two orderings. Score
all eligible development rollouts, then require at turn 6: at least
7/8 successful states prefer the original instruction, at least
10/15 same-group success/failure comparisons rank the successful
history higher, and at least 5/8 successful histories increase their
margin from turn 3 to 6. Also report pooled, scene-macro, failure,
order-agreement, STOP/timeout, and group-cluster uncertainty metrics.
Report six-turn minus no-motion margins as a check for image-independent
instruction priors. If any gate fails, do not wire this teacher into group-four reward;
the small sample is exploratory even if it passes. A passing screen
must still face a two-step n=4 wiring audit and matched 64-step
navigation control before any claim of improvement.

The preterminal development screen **failed** all three frozen gates.
The teacher preferred the original instruction on 5/8 successful
turn-six histories, increased its margin from turn three on 4/8,
and ranked the successful history above a same-group failed history
on only 5/15 comparisons across six mixed groups. Group-macro
outcome ranking was 30.56% (six-group bootstrap interval
[6.67, 60.00]%). It preferred the original instruction on 33/48
turn-six histories overall, including 28/40 failures. Exact coverage,
both prompt orderings, and per-group scores are in
`qwen3_route_match/policy_development_analysis.json`. The expert
route gain therefore does **not** support a preterminal process
reward. No online RL or val-unseen is justified for that reward.

## Distinct terminal route-fidelity screen (frozen before scoring)

A terminal scalar could behave differently from a preterminal
potential. Test the same frozen Qwen3 route matcher on six views
uniformly sampled from the **complete executed motion history**,
using indices `floor(i*T/5+0.5)` for `i=0..5` and final motion turn
`T`. The source STOP response, outcome, geodesic distance, and goal
coordinates are excluded from the teacher input. Score six repeated
initial views as the no-motion control. Retain the same exact-start
wrong-goal instruction, both A/B orderings, model hashes, and 13
development groups. The source contains 11 successful terminal
histories and 19 success/failure comparisons in six mixed groups.
Before any candidate n=4 policy run, require at least 9/11 positive
correct-instruction terminal margins, at least 13/19 within-group
success-over-failure ranks, and at least 7/11 successful trajectories
whose terminal margin exceeds their own initial-view margin. Report
group and scene macros, order consistency, STOP/timeout strata, and
group-bootstrap intervals. Passing would authorize only a matched
two-step terminal-reward wiring smoke and then a 64-step n=4 paired
navigation screen; it would not rescue the failed dense process
reward or establish a navigation gain. Failing stops this teacher
reward path.

The fixed terminal screen passed two of three gates but failed the
required evidence-gain gate. The terminal score preferred the
correct instruction on 9/11 successful trajectories and ranked a
same-group success above a failure on 13/19 comparisons, with a
six-group bootstrap interval [47.37, 89.47]% for that ranking. Only
5/11 successful trajectories increased their margin above their
own no-motion view (required 7/11). A/B ordering agreed on only
30/52 terminal histories. Thus this absolute terminal score is not
qualified as a semantic reward; no online or val-unseen run follows
from this screen. Exact terminal results and per-group margins are
in `qwen3_route_match/terminal_development_analysis.json`.

## Group-relative terminal preference hypothesis (new protocol)

The failed absolute-gain gate does not test whether a **within-prompt
rank** can provide group-four credit: every rollout of a group shares
the same initial scene and two instructions, so a fixed initial
preference and A/B letter bias cancel in a same-group score
difference. This is a separate reward construction, not a waiver of
the failed terminal rule. Freeze the teacher, natural exact-start
wrong-goal selection, six-frame terminal sampling, and two A/B
orderings. On the 55 already cached fit groups, use the 220 terminal
histories and **71** success/failure pairs across 21 mixed groups.
The earlier 62 count applied only to routes with a preterminal
six-turn view; all 220 fit routes have a terminal view. This label-only
denominator correction was made before the complete fit scoring run.
Before further online work, require the score of a successful rollout
to exceed a failed rollout in **at least 48/71** same-group pairs and
at least 65% group-macro accuracy. Report seed, scene, STOP/timeout,
and pair-cluster intervals. No model/prompt/hyperparameter is fitted
or selected on these scores. The earlier development 13/19 rank is
exploratory hypothesis generation, not an independent audit. If fit
fails, stop this hypothesis. If it passes, define a bounded group
rank bonus from the four order-averaged terminal margins and run only
a same-data, same-seed, same-rollout-budget n=4 two-step wiring smoke,
then a 64-step paired 256-episode navigation screen against the
group-four outcome control. The bonus scale must be fixed against
the existing environment reward range before training; the small
val-unseen screen selects no paper-level claim.

The fixed fit group-relative terminal screen passed: among 55
complete four-rollout groups and 71 same-group success/failure pairs,
the frozen teacher ranked success higher **53/71** times (74.65%),
with 74.21% group-macro and 75.81% scene-macro ranking. The
21-group bootstrap interval is [62.86, 84.93]%. Successful
terminal histories had a positive original-versus-wrong goal margin
on 34/39 routes, but failures also did on 106/181; individual
positivity is therefore unsuitable as a success reward. These are
previously collected R2R-train policy rollouts, not a navigation
improvement. The development-only hypothesis-generating group rank
was 13/19; its own absolute-gain gate failed. The exact 220-route
fit cache and analysis are in `qwen3_route_match/terminal_fit_*`.
Proceed only to a bounded **within-group rank** bonus with standard
n=4 and a matched destination-only control. Fix the reward scale and
validate grouping/gradient wiring before running policy updates.

### Failure-group reward relevance check (fixed before CPU analysis)

Most n=4 training groups contain no successes, so a rank bonus
must also order **failed** trajectories usefully. Using only frozen
fit terminal teacher scores and simulator labels for offline
evaluation, compare failures within one four-rollout group when
their terminal geodesic distances differ by at least 1.5 m. The
source has 33 all-failure groups. Among them, 80 eligible
same-terminal-mode pairs span 30 groups; 67 cross-mode pairs span
24 groups. The teacher never receives terminal distance or outcome.
Require at least **56/80** same-mode pairs to rank the closer
trajectory higher and at least 65% macro accuracy over contributing
groups. Report cross-mode comparisons separately to expose STOP
versus timeout bias; do not select a mode-specific score afterward.
The prior mode-stratified ordinal reward reached 66.12% on its own
screen but lost 5.47 SR points in a matched 256-episode navigation
pilot, so even passing this offline check would only justify a
two-step wiring smoke and matched pilot, not an expected gain.
If this failure-group check fails, stop the proposed group-rank
online reward despite its success/failure ranking result.

The failure-group check passed on the frozen fit records: **60/80**
closer-over-farther same-mode pairs (75.0%) and 69.56% group-macro
across 30 contributing groups. Cross-mode ranking was 41/67; the
online bonus should therefore compare failed rollouts only inside
the same terminal-mode stratum. This is train-scene evidence only.

### Coverage and same-scene fallback (frozen before scoring)

The exact-same-start counterfactual exists for only **81/256** rows of
the original n=4 pilot dataset, including 21/64 in the first 64
steps. To avoid training with a mostly disabled reward, first test a
natural wrong-goal instruction from the *same scene* with a start at
least 1 m different and goal at least 4 m away. Select the eligible
candidate with the nearest start position, breaking ties by a fixed
SHA-256 rank. Keep the original 55 fit groups, 220 cached policy
routes, six-frame terminal sampling, both A/B prompt orders, teacher
weights, and all outcome labels. The new counterfactual is a coverage
fallback, not a teacher retraining step. The group-relative score
should cancel shared initial-scene preference, but this transfer is
unproven until measured.

Before seeing its model scores, require at least **48/71**
same-group success-over-failure pairs with at least 65% group-macro,
and at least **56/80** closer-over-farther same-terminal-mode pairs
among all-failure groups with at least 65% group-macro. Stop this
full-coverage n=4 online reward path if either fit criterion fails;
then consider a separately paired control/candidate on the eligible
subset. Passing authorizes only an n=4 two-step wiring smoke and a
same-data, same-seed 64-step/256-val-unseen screen against the
existing n=4 destination-only control. Use the exact-same-start
counterfactual when available, otherwise this fallback. This is
exploratory reused train-scene screening; even a positive 256-episode
screen requires independent seeds and full val-unseen confirmation.

The same-scene, different-start fallback **failed** before online use:
the unchanged teacher ranked a success above a same-group failure on
42/71 pairs (59.2%) with 57.94% group-macro, below both fixed gates.
No reward or policy training used this fallback. Its frozen 220-route
outputs and counterfactual manifest are retained for audit.

The restricted follow-up draws **256 distinct exact-same-start
eligible episodes** from the existing 4,000-row R2R train parquet,
balanced round-robin over train scenes with SHA-256 ranking. Both
destination-only control and group-relative candidate must train from
the same navigation-SFT checkpoint, in exactly this row order, with
seed 11, group size four, and 64 optimizer steps. The negative
instruction is selected by the earlier exact-start SHA rule. For
online scoring, send six evenly spaced initial/turn observations as
336-pixel JPEGs at quality 82, matching the offline replay cache.
For all-failure groups, score only same-terminal-mode rollouts; assign a
zero-sum rank bonus in [-0.5,+0.5]. Any group with a successful rollout
keeps the outcome-only reward. This makes the minimum success reward
(2) exceed the maximum failure bonus (+0.5). A two-step wiring run must
first show four samples per episode, finite reward tensors, no missing
teacher scores in eligible all-failure strata, and nonzero actor
gradients. Evaluate both 64-step models on the same fixed 256
val-unseen episodes with full unique-ID coverage and zero inference
errors. Only positive paired SR **and** SPL merits multi-seed/full
val-unseen scaling; the small screen is exploratory and cannot alone
support a CVPR claim. A group-size >4 diagnostic is deferred until an
n=4 algorithmic gain is observed.

The conditional larger-budget dataset is now frozen **before** the
256-episode navigation result: the same scene-round-robin SHA rule
extends the 256 rows to 512 distinct exact-start rows from the
original R2R 4,000-row parquet. The first 256 IDs and wrong-goal
choices match the pilot exactly. It spans 54 train scenes with 3–11
rows per scene; parquet SHA-256 is
`d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f`.
The ID-only `qwen3_route_match/online_exact512_manifest.json` is in
the repository. This dataset is **not yet used for training**. If and
only if the matched 64-step n=4 pilot has positive paired SR and SPL
on the fixed 256 val-unseen episodes, train both destination-only and
Qwen-group-reward arms for 128 steps from the same SFT checkpoint on
this 512-row dataset at seeds 11, 22, and 33, holding the four-rollout
budget and optimizer settings identical. Evaluate all six checkpoints
on the same complete 1,839-episode val-unseen set with exact coverage,
zero inference errors, per-seed paired SR/SPL, scene-and-seed
uncertainty intervals, and the 1,583 episodes outside the reused
256-episode screen reported separately. Do not present a screen-only
or one-seed increase as a confirmed navigation improvement.

The scale execution is now scripted in
`run_qwen_group_scale_conditional.sh`. It waits for the matched
64-step group-four pilot and checks the fixed 256-episode manifest,
complete coverage, zero inference errors, and strictly positive paired
SR and SPL before allocating scale compute. If the gate fails, it
records `no_pilot_gain` and does not train scale models. If it passes,
it runs the three same-seed control/candidate pairs sequentially while
reusing one 16-simulator service per arm. Candidate training places
the frozen Qwen teacher on GPU 1, Habitat on GPU 0, and the policy on
GPUs 2 and 3. Once training ends, each seed's two models evaluate in
parallel: model GPUs 1 and 3 with four Habitat shards each on GPUs 0
and 2. The full-set analyzer checks all six 1,839-episode labels and
reports the 1,583 episodes outside the reused pilot screen separately.
Training reward and optimizer audits precede inference. These scripts
are staged; their presence is not evidence of pilot or scale results.
