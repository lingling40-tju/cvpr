# Next group-four representation-to-reward experiment

Status: design and stage gates only. No representation checkpoint, new
training run, or navigation gain is claimed here. The mode-stratified
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
