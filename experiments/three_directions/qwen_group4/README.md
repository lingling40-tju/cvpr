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

As of 2026-10-03 16:56 UTC, the matched control is running and the
Qwen teacher service is healthy. There is **no candidate navigation
result** yet. The fixed 256-episode screen must have exact ID
coverage, zero inference errors, and positive paired SR and SPL
before a three-seed, full-1,839-episode confirmation is considered.
Repeated use of the 256-episode screen remains exploratory.
