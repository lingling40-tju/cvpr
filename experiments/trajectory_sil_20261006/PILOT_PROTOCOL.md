# Positive trajectory update: frozen candidate protocol

This is a **new** algorithm comparison after the n=8 sensitivity and n=4
three-seed expansion did not establish a stable gain. No navigation result
exists yet for this candidate. It is an on-policy,
positive-only policy-gradient update inspired by self-imitation, **not** the
replay-buffer Self-Imitation Learning algorithm.

## Mechanism and matched comparison

- Fit: `fit512_manifest.json`, 512 R2R-train episodes in 45 fit scenes, fixed
  64 optimizer steps with eight instructions per step and four sampled
  trajectories per instruction (32 concurrent Habitat actors), seed 11.
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
- If the gate passes, train **both** arms for seeds 11, 22, and 33 at 512
  rows × 128 steps, n=4, on matched fit data. The 64-step seed-11 pilot is
  not pooled with these longer runs. Use `reserved256.json` only once after
  choices are frozen. Earlier SFT and exploration may have seen the train
  scenes, so even the reserved screen is not an untouched generalization
  benchmark.

The completed n=8 and n=4 post-result experiments had priority for GPU
resources. The pilot started only after their suites completed and the GPUs
were idle. Its reward source is privileged simulator navigation geometry,
not a deployable
instruction-grounded semantic verifier. No human labels are requested.

An isolated remote source copy is staged at
`/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006`.
It excludes the live tree's checkpoints and contains the hash-verified fit512
parquet and both frozen manifests. `prepare_positive_source.py` patched only
that copy: trainer SHA-256 changed from `ca3e7ec596f4c5cc13b6b574a3f71cd9040db6a34090e8776f2ce44a8288354b`
to `50356c80a1fda653e10f4332d128b611d29a4268598023d37c66513c26f31cdb`;
the active n=4 trainer retained its original hash. CPU integration through
the actual trainer passed for both the flagged positive-only branch and the
unflagged GRPO control branch.

The staged `start_positive_service.sh` and `run_positive_train.sh` require
both the n=8 suite and n=4 three-seed suite to finish before they can run.
The service uses a separate port (5085) and 32 Habitat actors; the training
script gives both arms the same eight-row batches, 64-step fit and KL setting.
These scripts passed shell syntax checks, and Hydra accepted the new batch,
loss aggregation, KL and nDTW settings in CPU-only configuration mode.
`run_positive_development_eval.sh` is restricted to the development screen;
its validator and paired analyzer passed a temporary synthetic 256-episode
raw-shard pipeline check. The reserved screen remains unopened. Real two-step
smoke tests precede the 64-step comparison; no fallback navigation inference
has run.

The staged `run_positive_pilot_suite.sh` runs the two arms sequentially only
after the existing n=8 and n=4 suites finish. `audit_positive_train.py`
checks exact optimizer-step coverage, finite KL metrics, and positive actor
advantages separately from gradient magnitude; a KL gradient alone does not
count as evidence that the reward signal trained the policy. The orchestrator
stops its own Habitat service before model evaluation and writes a frozen
dual-metric gate result. It has passed shell and synthetic audit checks and is
running on the GPUs.

## Pre-pilot reward configuration repair

The first **control-only two-step smoke** finished both optimizer steps with
nonzero actor gradients, then failed the frozen reward-range audit: logged
maximum sequence scores were 21.101 and 21.279, above the stated
success-15-plus-nDTW-5 upper bound. Simulator logs showed an additional
`success_floor: 2.0` on successful rollouts. The initially staged command had
`semantic_success_floor=2` even though `semantic_reward_weight=0`; the
semantic wrapper adds that floor independently of the verifier weight.
The failed smoke and checkpoint are preserved remotely under
`pre_floor_fix_20261007` names, with the log hash and correction recorded in
[`smoke_reward_range_recovery.json`](smoke_reward_range_recovery.json).

Before any candidate optimizer step or navigation evaluation, the run command
was corrected to `semantic_success_floor=0` for **both** arms. This restores
the frozen reward formula and the existing 0--20 audit bound; no performance
metric or development threshold was changed. The matching two-arm smoke then
completed from the original SFT initialization: both arms covered exactly two
optimizer steps, with nonzero actor gradients and positive advantages in both
steps; the candidate advantages were nonnegative and both arms had maximum
terminal score 19.279. The compact [control](real_smoke/control_smoke_audit.json)
and [candidate](real_smoke/candidate_smoke_audit.json) audits include training
log hashes. The matched 64-step training has begun; the reserved screen is
still closed.
