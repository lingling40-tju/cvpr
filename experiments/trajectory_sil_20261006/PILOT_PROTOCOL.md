# Positive trajectory update: frozen candidate protocol

This is preparation for a **new** algorithm comparison if the running n=8
sensitivity and n=4 three-seed experiments do not establish a useful gain.
No model has been trained or evaluated with this candidate. It is an on-policy,
positive-only policy-gradient update inspired by self-imitation, **not** the
replay-buffer Self-Imitation Learning algorithm.

## Mechanism and matched comparison

- Fit: `fit512_manifest.json`, 512 R2R-train episodes in 45 fit scenes, fixed
  64 optimizer steps, four sampled trajectories per instruction, seed 11.
- Both arms start from the same Qwen2.5-VL-3B navigation SFT checkpoint and
  receive the same simulator terminal reward: distance-weighted success base
  15 plus nDTW base 5. No semantic verifier or per-turn geodesic reward.
- Both arms use one PPO epoch, the same rollout budget and learning rate,
  sequence-mean/token-mean loss aggregation, and a reference-policy KL loss
  coefficient of 0.01. A two-step real-environment smoke test must first
  confirm these settings fit GPU memory and produce finite nonzero gradients.
- Control: standard GRPO advantage from that terminal reward. Candidate:
  `positive_trajectory_advantage.py`. It gives positive actor credit only to
  trajectories that beat the other three in terminal quality. Simulator
  success has priority over an unsuccessful high-nDTW rollout; a failure must
  earn terminal score at least 2.5. The positive gap is divided by 5 and
  capped at 1.5. Equal-quality groups have zero actor update. All generated
  action tokens in a trajectory share the weight; observation tokens have
  zero weight.
- Primary comparison: paired SR and SPL on `development256.json`, with all
  256 unique episode IDs in each arm and zero inference errors. The fixed
  expansion gate is **both** paired SR and SPL at least +2 percentage points.
  Report episode-level deltas and scene-cluster uncertainty. If the gate
  fails, do not expand this candidate on the same development screen.
- Both 256-ID screens were resolved against Habitat's actual 10,819-episode
  R2R-train dataset without missing IDs. Habitat prefixes each scene path
  with `data/scene_datasets/`; strip that prefix when comparing to the frozen
  manifests' `mp3d/...` scene IDs. Each screen covers its eight frozen scenes.
  The derived `eval_train_scene_subset.py` passed `--validate-only` in the
  actual Habitat environment for both roles (256 unique episodes, 64 in each
  of four shards, eight scenes), and refused a development manifest supplied
  under the reserved role. This does not exercise model inference.
  `analyze_train_scene_pair.py` independently checks raw shard coverage and
  recounts paired SR/SPL; a temporary synthetic 256-episode I/O preflight
  recovered exactly one discordant success (+0.390625 points for both metrics).
  Synthetic data are not a navigation result.
- If the gate passes, replicate seeds 22 and 33 at 512 rows × 128 steps,
  n=4, on matched fit data. Use `reserved256.json` only once after choices
  are frozen. Earlier SFT and exploration may have seen the train scenes, so
  even the reserved screen is not an untouched generalization benchmark.

The running n=8 and n=4 post-result experiments have priority for GPU
resources. This protocol does not start another training job. Its reward
source is privileged simulator navigation geometry, not a deployable
instruction-grounded semantic verifier. No human labels are requested.

An isolated remote source copy is staged at
`/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006`.
It excludes the live tree's checkpoints and contains the hash-verified fit512
parquet and both frozen manifests. `prepare_positive_source.py` patched only
that copy: trainer SHA-256 changed from `ca3e7ec596f4c5cc13b6b574a3f71cd9040db6a34090e8776f2ce44a8288354b`
to `50356c80a1fda653e10f4332d128b611d29a4268598023d37c66513c26f31cdb`;
the active n=4 trainer retained its original hash. CPU integration through
the actual trainer passed for both the flagged positive-only branch and the
unflagged GRPO control branch. No GPU training or model inference has been
launched for this fallback.
