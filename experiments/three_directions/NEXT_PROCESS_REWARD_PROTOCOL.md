# Next group-four representation-to-reward experiment

Status: representation screens and a new group-four offline screen are in
progress. No new policy training run or navigation gain is claimed here.
The mode-stratified
ordinal pilot failed its matched 256-episode screen (SR -5.47, SPL -5.23
percentage points) and did not enter the 128-step scale suite.

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
SACA (Li et al., arXiv:2603.09740) and semantic progress by
Progress-Think (Wang et al., arXiv:2511.17097). Any later paper claim
must compare with these methods; the proposed distinction here is the
separate STOP calibration gate and explicit verification of turn-wise
credit assignment in this ActiveVLN implementation.

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
