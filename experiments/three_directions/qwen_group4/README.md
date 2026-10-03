# Frozen route representation to group-four reward

The local Qwen3-VL-8B teacher scores six ordered observations of a
completed VLN policy route against two natural instructions with the
same start and different goals. The score averages A/B first-token
logit margins over both instruction orderings. It contains no simulator
distance or success signal. The teacher is frozen throughout policy
training.

The exploratory offline fit screen ranked a success above a failure
within the same four-rollout group in 53/71 comparisons. In
all-failure groups, it ranked the closer of two trajectories with the
same terminal mode in 60/80 comparisons where the geodesic distance
gap was at least 1.5 m. An attempted same-scene, different-start
counterfactual fell to 42/71 success/failure comparisons and was
rejected before online training. These are R2R **train-scene**
diagnostics and establish no navigation improvement.

`prepare_qwen3_group4_dataset.py` selected 256 distinct rows from the
existing R2R 4,000-row parquet, balancing 54 training scenes to 3–5
rows each. Every selected instruction has an exact-same-start natural
wrong goal at least 4 m away. The dataset SHA-256 is
`6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69`.
The ID-only mapping and source hashes are in
`../ordinal_progress/policy_preference/qwen3_route_match/online_exact256_manifest.json`;
the parquet and licensed images remain on `wanghaozhihuoshanyun`.

The standard comparison uses `rollout.n=4`, seed 11, the same dataset
and row order, navigation-SFT initialization, 64 optimizer steps,
and an unchanged destination-only reward as the control. The Qwen
candidate modifies **only** all-failure groups: within each terminal
mode, it converts frozen route margins to a zero-sum rank bonus in
[-0.5,+0.5]. Mixed success groups retain the destination reward.
The success floor of 2 exceeds the maximum failure bonus. The source
hook is checksum guarded, lives in a separate ActiveVLN tree, and
encodes online views as 336-pixel quality-82 JPEGs like the offline
replay cache.

The live teacher service reproduced the frozen offline logit margin
exactly on all 40 original cached JPEG trajectories shared by the
selected dataset and fit split (10 four-rollout groups; maximum
absolute difference 0). This checks service weights, prompt, option
order averaging, and counterfactual mapping. Fresh simulator images
still require the two-step wiring audit; this replay check does not
establish navigation performance (`replay_parity.json`).

An exploratory confidence gate retained a route pair only when both
A/B instruction orderings agreed on its rank. On fit scenes,
closer-failure ranking rose from 60/80 to 54/66 retained pairs, but
on the development scenes it was 11/16 without the gate and 9/13
with it. This provides no evidence of a transferable failure-reward
gain, so the ongoing online pilot keeps the frozen ungated rank
(`order_consensus_diagnostic.json`).

`analyze_confidence_gate.py` tests a second **post hoc** reward idea
using only the existing train-scene cache. It computes a score-gap
threshold of 5.5 from the upper median absolute margin of the 80 fit
same-mode failure pairs, then gives each same-mode failure pair a
bounded, opposite-signed vote only when its margin clears that gap.
Of the original 80 fit and 16 development pairs, this direct gate
retains 40 and 7; the teacher orders 33/40 and 7/7 correctly. The
resulting group reward orders 44/54 fit and 8/9 development pairs
where it assigns different rewards. This analysis idea followed
inspection of related development results; these reused train-scene
figures cannot validate generalization. It was **not** the reward in
the completed ungated pilot and has no navigation result yet
(`confidence_gate_diagnostic.json`).
The corresponding `confident_pair_reward.py` defines the next n=4
algorithm: pairs below the 5.5-point Qwen margin
gap contribute no reward, while qualifying same-mode failure pairs
cast opposite-signed votes bounded to [-0.5,+0.5]. Mixed-success
groups still use only outcome reward. Unit checks include exact
reward-value parity with all 33 fit and seven development
all-failure groups in the frozen cache
(`test_confident_pair_reward.py`). It is installed only in the
isolated confidence-candidate tree; no policy-training result is yet
available.
The ungated 256-episode pilot had negative paired SR and SPL, so
`run_confident_conditional.sh` observed its `no_pilot_gain` marker and
copied only source code and small data into a separate
`ActiveVLN_qwen_confident_20261004` tree with the checksum-pinned
confidence reward. Its next stages run a two-step audit and then a
matched 64-step n=4 candidate. It evaluates that checkpoint and the original
same-data n=4 control concurrently on the newly frozen fifth
256-episode screen. The new audit recomputes pair votes independently
and checks training-row identity, reward totals, teacher requests,
and gradients. The script passed local syntax and frozen-cache parity
checks and has prepared the isolated tree. Its two-step n=4 wiring test
passed an independent audit: eight matched groups, 27 frozen-teacher
requests for 27 failures, eight nonzero ordinal-reward rollouts, and
nonzero actor gradients at both steps (`confident_2step_audit.json`).
The 64-step confidence candidate has started. There is no completed
confidence-candidate navigation result yet.
An immutable first-20-step rollout-prefix diagnostic now checks the live
candidate without using another GPU. Among 80 matched n=4 train groups,
42 had no successful rollout and 22 received a nonzero confidence-pair
rank. Within the same terminal mode and a 1.5 m simulator-distance gap,
the raw teacher ordered the nearer failure correctly in 87/114 pairs.
The frozen 5.5-point gap retained 45 pairs and ordered 44 correctly;
the *applied group reward* distinguished 54 pairs and ordered 51
correctly. Those pairs share groups and are not independent samples.
This is a train-scene signal with reduced reward coverage, not a
val-unseen navigation result or a reason to alter the running threshold
(`confident_onpolicy_step20.json`).

`analyze_confident_onpolicy.py` also freezes an **interim train-scene
diagnostic** from the first 24 steps of the now completed ungated
candidate, pairing every train episode with the matched control and
hashing only the first 24 rollout lines. In its 51 all-failure groups,
147 same-mode failed-route pairs have at least 1.5 m terminal distance
separation. The current teacher rank orders 104/147 closer routes
correctly. The previously frozen 5.5-point gap retains 63 direct
pairs and orders 55/63 correctly; hypothetical confidence votes order
69/84 pairs where they differ. The lower coverage and train-scene
labels matter. This does not predict navigation gain or authorize
scale-up (`current_onpolicy_step24.json`).

The same frozen 5.5-point gate was rechecked over the first 40 steps
without changing the reward or training run. Across 242 same-mode
failure pairs separated by at least 1.5 m, the raw teacher orders
159/242 correctly. The absolute-gap gate retains 100 pairs and orders
74/100 correctly; hypothetical confidence votes order 93/128 pairs
where they differ. These cumulative rates conceal a weaker later
slice: subtracting the first-24-step counts leaves steps 25–40 at
55/95 raw, 19/37 retained high-gap, and 24/44 confidence-vote pairs.
Different episodes occupy this later slice, so the change cannot be
attributed specifically to policy drift. All distances are train-scene
diagnostic labels, and the frozen threshold has not been revised from
them. This weakens the case for the confidence fallback but does not
replace its paired val-unseen navigation test
(`current_onpolicy_step40.json`).

A CPU-only calibration check used the same fixed first 40 training steps.
It fitted one ridge-regularized linear pairwise score from the teacher's
two prompt-order margins, executed-action count, and turn count, using
simulator distance only as a training label. A hash partition reserved
whole train scenes from fitting. On 76 failure pairs in those held-out
train scenes, the raw teacher ordered 42 correctly and the fitted
score ordered 45 correctly. The three-pair difference is weak, and
these are still train scenes. No policy has been trained with this
calibration; it does not justify another GPU run or a navigation claim
(`fit_margin_calibrator.py`, `margin_calibration_step40.json`).

For a possible next reward candidate, `prepare_next_val_manifest.py`
froze a separate 256-episode screen from the 1,839 val-unseen IDs
**before** the current candidate's navigation result. It excludes all
four earlier, mutually disjoint 256-episode development screens
(1,024 IDs total), selects only by scene and hashed episode ID, and
does not inspect policy outputs. The fifth screen contains 256 IDs
from the 815 remaining episodes across eight scenes; its manifest
SHA-256 is
`e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1`.
Three val-unseen scenes have no episodes left after the earlier
screens, and the complete 1,839 set has been analyzed in prior work.
This is a fresh **development slice**, not an independent test or
paper-level confirmation (`next_val256_manifest.json`). It has not
been used for inference.

Resource schedule: Habitat on GPU0; the frozen teacher on GPU1;
two policy actors on GPU2/3. The 64-step control runs first while the
teacher service is prepared on otherwise idle GPU1. A single
`run_qwen_group_followup.sh` watcher then audits the control, runs a
two-step candidate wiring test, trains the 64-step candidate if that
passes, and evaluates candidate and control simultaneously on the
same 256 val-unseen IDs with two GPU pairs. Four-sample rollout cost
is kept identical in both arms. Group sizes above four are reserved
for a small diagnostic if the group-four algorithm shows a real
paired navigation gain.

The matched control finished 64 steps. Its training audit confirms
256 distinct train episodes, four rollouts per episode, 1,024 total
rollouts, 152 all-failure groups, and 57 nonzero-gradient steps
(`control_training_audit.json`). This is not a held-out result.
The first automatic candidate-service start failed after the control
audit because the already running teacher process had inherited the
service startup file descriptor and held its lock. No candidate
training started during that failure. Both service startup scripts
now close that descriptor in background children. The teacher was
restarted, the lock was verified free, and the candidate and scale
watchers were restarted. The two-step candidate wiring run then
finished and passed its audit: eight matched four-rollout groups,
26 frozen-teacher requests for 26 failed trajectories, 16 nonzero
ordinal reward rollouts, and nonzero actor gradients at both steps
(`candidate_2step_audit.json`). The 64-step candidate run then finished
and passed its independent training audit: 256 matched four-rollout
groups, 796 frozen-teacher requests for 796 failures, 433 nonzero
ordinal reward rollouts, and nonzero actor gradients at all 64 steps
(`candidate_64step_audit.json`). The candidate/control evaluation has
started in parallel on the fixed 256-episode val-unseen screen.
The fixed 256-episode val-unseen screen then completed with identical
unique episode IDs in both arms and zero inference errors. The Qwen
candidate reached 63/256 success (24.61% SR, 24.42% SPL), versus
68/256 (26.56% SR, 26.25% SPL) for its same-data n=4 control. Paired
changes are -1.95 SR and -1.83 SPL percentage points. The nine-scene
bootstrap intervals cross zero; this screen establishes neither a
reliable loss nor a gain. It fails the prespecified strictly positive
SR-and-SPL scale gate, so the Qwen group-rank method is not expanded to
three seeds or full 1,839-episode evaluation. The exact per-episode
success, SPL, and terminal-distance values are preserved without images
(`paired_qwen_vs_control.json`, `paired_qwen_episodes.jsonl`). This
val-unseen screen has been reused in earlier development and is not an
independent paper test.

`run_qwen_full_recheck_after_confident.sh` is a conditional sensitivity
check using those **existing** seed-11 n=4 checkpoints. A prior group-four
optimizer candidate was negative on a 256-episode screen but positive on
the complete 1,839 episodes, so the small screen can miss a scene-dependent
effect. This check waits for the confidence candidate's own fixed-256
result. If that candidate passes its positive-SR-and-SPL gate, the check
skips to leave all GPUs for the planned three-seed scale. If it fails,
the script runs the original Qwen candidate and same-data control in
parallel GPU lanes on the complete 1,839 manifest, then recomputes
paired metrics on all episodes and on the 1,583 episodes outside its
reused 256 screen (`analyze_qwen_full_recheck.py`). This is a **post hoc,
one-seed development sensitivity check**, not an independent test or a
new predeclared scale gate; it cannot by itself establish a publishable
gain. No larger group size or new policy training is scheduled by it.

The confidence fallback has a separate conditional scale plan. Its
watcher first requires the fifth, fixed 256-episode screen to show
strictly positive paired SR and SPL for the 64-step confidence candidate.
Only then does it train three matched n=4 control/candidate seeds for
128 steps on the frozen 512-row extension. Each seed is audited for
identical train episodes, four rollouts per episode, reward components,
teacher requests, and nonzero gradients. Full val-unseen evaluation
then runs each candidate/control pair concurrently on two GPU lanes,
with 1,839 unique episodes per checkpoint. The analysis reports the
three paired seed effects on all 1,839 episodes and separately on the
1,583 outside the fifth screen. All service ports and checkpoint names
are isolated from the original candidate. The confidence scale watcher
does not use a GPU while it waits and exits without scale training if
the fallback pilot fails its navigation gate. The fallback pilot result
and scale decision are below.

The confidence fallback has now completed its separate seed-11,
64-step group-four training and independent train-row audit
(`confident_64step_audit.json`). The audit found 256 distinct episode
groups and 1,024 rollouts, including 146 all-failure groups; 87 of
those groups had an active ordinal reward. It verified 828/828 teacher
requests for failed rollouts, 196 confident pair comparisons, and
nonzero actor gradients in all 64 steps. The fixed fifth 256-episode
candidate/control val-unseen evaluation then completed in parallel.
Both models covered the exact same 256 unique IDs with zero inference
errors. The confidence candidate reached 80/256 (31.25% SR, 30.78%
SPL), versus 88/256 (34.38% SR, 33.92% SPL) for the same-data
group-four outcome control. Paired changes were -3.13 SR and -3.14
SPL percentage points; the eight-scene bootstrap intervals cross
zero. This fails the prespecified positive SR-and-SPL scale gate, so
the confidence three-seed extension did not launch. The exact compact
paired rows and summary are `paired_confident_episodes.jsonl` and
`paired_confident_vs_control.json`. A separate local recomputation
checked all 256 manifest IDs, both metrics, and discordant successes.
The complete 64-step train-scene diagnostic
(`confident_onpolicy_step64.json`) examined 346 same-end-mode pairs
with at least 1.5 m simulator distance separation. The raw Qwen score
ranked the nearer failed rollout correctly in 230/346; the fixed 5.5
score-gap filter retained 160 pairs and ranked 124/160 correctly.
The actual zero-sum gated reward ordered 191/346 eligible pairs and
ranked 144/191 of those correctly. These are correlated on-policy
*training* pairs, with ties outside coverage; the result is neither
an independent verifier accuracy estimate nor a navigation gain.
An additional **post hoc** termination-mode audit of that same fixed
256-item screen (`analyze_confident_stop_modes.py`,
`confident_stop_mode_diagnostic.json`) found 143 candidate failures
and 109 control failures with no evaluator forced-stop reason.
The paired candidate-only/control-only counts for this event are
58/24, a +13.28-point difference; an eight-scene bootstrap interval
is [+6.61,+21.05] points. A missing forced-stop reason is consistent
with a model-selected STOP, but these compact records do not contain
an independent action trace, and the post hoc association cannot
causally explain the SR gap. It reinforces the need to validate STOP
separately from progress before using a learned process reward.
The same-row training-rollout audit
(`analyze_confident_train_stops.py`, `confident_train_stop_modes.json`)
provides a related, less selected check: across 256 matched four-sample
episode groups, the candidate had 295 failed voluntary STOPs among
1,024 rollouts versus 263 for the control (+3.13 points). The two
arms sampled different trajectories and had different success totals,
so this is supporting descriptive evidence, not a causal STOP effect.
Because the confidence pilot failed, the conditional original-Qwen
full-1,839 sensitivity recheck ran on two inference/Sim GPU lanes
using the previously trained seed-11 checkpoints. Both arms covered
the same 1,839 unique episodes with zero inference errors. The Qwen
group-rank candidate reached 471/1,839 success (25.61% SR, 25.49% SPL)
versus 562/1,839 (30.56% SR, 30.21% SPL) for the same-data group-four
control. Paired changes were -4.95 SR and -4.72 SPL points; the
11-scene bootstrap intervals were [-6.85,-2.31] and [-6.60,-2.10]
points. On the 1,583 episodes outside the reused original screen,
paired changes were -4.93 SR and -4.69 SPL points. The compact
per-episode records are `paired_qwen_full_episodes.jsonl`; summaries
and the raw-stat audit are `paired_qwen_full_recheck.json` and
`qwen_full_recheck_analysis.json`. A separate local recomputation of
the compact records recovered both paired effects. This recheck was
chosen after seeing the negative 256-item screen; even its complement
is development evidence from one seed and one decode. It reinforces
the negative navigation finding and does not trigger scale-up. The
isolated privileged turn-wise mechanism smoke started after this
four-GPU evaluation released its resources.

To keep any scale-up independent of that screen, a 512-row exact-start
extension was fixed in advance. Its first 256 rows and wrong goals are
identical to the pilot; the 512-row parquet SHA-256 is
`d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f`.
The ID-only mapping is
`../ordinal_progress/policy_preference/qwen3_route_match/online_exact512_manifest.json`.
No model has trained on this extension yet. A positive paired SR and
SPL screen would trigger matched n=4 control/candidate training at
three seeds for 128 steps, followed by full 1,839-episode val-unseen
evaluation and a separate report for the 1,583 episodes outside the
reused screen.

The scale dataset was staged in both isolated training trees.
`run_qwen_group_scale_conditional.sh` wrote `no_pilot_gain` and exited
without scale training after the negative paired screen.
`start_qwen_group_scale_services.sh` and
`run_qwen_group_scale_training.sh` require the positive pilot gate marker. The
teacher server accepts either the 256-row pilot or 512-row scale
manifest while verifying the corresponding parquet hash.
