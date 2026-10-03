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
figures cannot validate generalization. It is **not** the reward in
the running pilot and has no navigation result
(`confidence_gate_diagnostic.json`).
The corresponding `confident_pair_reward.py` is staged locally as a
possible next n=4 algorithm: pairs below the 5.5-point Qwen margin
gap contribute no reward, while qualifying same-mode failure pairs
cast opposite-signed votes bounded to [-0.5,+0.5]. Mixed-success
groups still use only outcome reward. Unit checks include exact
reward-value parity with all 33 fit and seven development
all-failure groups in the frozen cache
(`test_confident_pair_reward.py`). It has **not** been installed in
the remote trainer or used for policy training.

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
(`candidate_2step_audit.json`). The 64-step candidate run has started.
There is **no candidate navigation result** yet. The fixed
256-episode screen must have exact ID
coverage, zero inference errors, and positive paired SR and SPL
before a three-seed, full-1,839-episode confirmation is considered.
Repeated use of the 256-episode screen remains exploratory.

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

The scale dataset has been staged in both isolated training trees.
`run_qwen_group_scale_conditional.sh` is running as a watcher; it
starts scale training only if the 256-episode paired navigation screen
has strictly positive SR and SPL. No scale training has started.
`start_qwen_group_scale_services.sh` and
`run_qwen_group_scale_training.sh` require the positive pilot gate marker. The
teacher server accepts either the 256-row pilot or 512-row scale
manifest while verifying the corresponding parquet hash.
