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
