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

The prospective [`run_positive_scale_train.sh`](run_positive_scale_train.sh)
implements that 128-step budget for either arm and seeds 11/22/33. It refuses
to run unless the pilot suite is complete and the original development gate
records both paired metrics at least +2 points. It has not been launched;
the script is staged before seeing the pilot navigation result.
The scale command explicitly sets two data epochs: 512 rows at eight rows per
batch provide 64 steps per pass, and the trainer's epoch loop does not extend
itself when only `total_training_steps` is raised. This CPU source/config
check corrects the future command to deliver its already stated 128-step
budget; the live 64-step pilot still uses one pass.

[`run_positive_scale_if_pass.sh`](run_positive_scale_if_pass.sh) provides the
conditional relay: it waits for the pilot's live process and completion lock,
independently verifies the exported development episodes and gate before
allocating training GPUs, and records `scale.skipped` if either metric misses
the original threshold. All six 128-step runs and their training audits must
finish before [`run_positive_reserved_eval.sh`](run_positive_reserved_eval.sh)
can run. Each seed's two models are evaluated concurrently on GPUs 0/1,
ports 8136/8137, with four GPU-2 Habitat shards per model. A `reserved.opened`
marker records the start of this fixed six-model procedure and prevents
further scale training; it does not imply that inference has finished.
The pilot gate's `reserved_screen_opened: false` records its historical state
at the development decision and is not rewritten when the later screen opens.

Each reserved pair is independently recounted before
[`analyze_positive_scale.py`](analyze_positive_scale.py) reports the equal-seed
mean, sample standard deviation, and descriptive seed-and-scene bootstrap
intervals. A CPU-only synthetic raw-stat-to-three-seed-report pipeline
recovered the known +0.78125-point mean and rejected duplicate episode rows;
the 128-step training-audit parser also passed synthetic input. None of these
checks are model training or navigation results. The relay is staged before
seeing any pilot navigation metric and cannot bypass its frozen gate.
Its watcher is now waiting remotely; six-run scale training and reserved
inference have not started. The 128-step-capable local training auditor is
deployed as `tools/audit_positive_scale_train.py`, leaving the running pilot's
existing audit file unchanged. A direct premature reserved-evaluation call
was rejected before creating its result directory.

[`verify_positive_compact.py`](verify_positive_compact.py) independently
recounts the exported episode rows, validates their frozen scene/episode
mapping, reconciles both arm validators, and checks paired SR/SPL,
scene-bootstrap intervals, and the original gate decision. It imports no code
from the raw-stat analyzer. A CPU-only synthetic raw-stat-to-export pipeline
with one discordant success recovered exactly +0.390625 SR/SPL points and
matching intervals; this is a parser check, not navigation evidence.

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

The staged `run_positive_pilot_suite.sh` trains the two arms sequentially only
after the existing n=8 and n=4 suites finish. `audit_positive_train.py`
checks exact optimizer-step coverage, finite KL metrics, and positive actor
advantages separately from gradient magnitude; a KL gradient alone does not
count as evidence that the reward signal trained the policy. The orchestrator
stops its own Habitat service before model evaluation and writes a frozen
dual-metric gate result. It has passed shell and synthetic audit checks and is
running on the GPUs.

## Pre-inference parallel evaluation amendment

Before either pilot model ran navigation evaluation (control training at
23/64 steps), the existing development-evaluation entry point was changed
to delegate to [`run_positive_development_pair.sh`](run_positive_development_pair.sh).
Its first control/candidate call evaluates the fixed pair concurrently;
the suite's second call returns after both arm validators without launching
duplicate models. The running pilot suite file was kept byte-identical.
[`development_parallel_amendment.json`](development_parallel_amendment.json)
records its hash, the preserved original helper hash, the new helper hashes,
and zero navigation-result rows at this amendment.

[`run_positive_development_model.sh`](run_positive_development_model.sh)
uses GPU0/port8135 for control and GPU1/port8138 for candidate. Each still
has four GPU2 Habitat shards, with the same checkpoint, 256 episode IDs,
image preprocessing, seed 11, generation settings, and per-model inference
budget. The parent requires both completed 64-step training audits and
available GPUs before launching; the existing raw analysis, independent
recount, and +2/+2-point gate remain in force. CPU-only temporary model
stubs confirmed concurrent starts, refusal before training completion,
no second-call duplication, and failure propagation without a pair
completion marker. A premature call to the actual deployed entry point
was also refused. These scheduling checks are not model inference or
evidence of a measured speedup. Reserved inference remains unopened.

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
log hashes. Both matched 64-step runs have now completed. The independent
training audits cover every optimizer step, with finite KL metrics and
nonzero actor gradients on all 64 steps in each arm. Candidate advantages
are nonnegative, with logged maximum terminal score 19.589; the control's
maximum is 19.500. These are optimization diagnostics, not navigation
benefits. The reserved screen is still closed.

## Evaluation reference-split recovery

The first parallel development evaluation failed in Habitat's nDTW
measurement during `env.reset()`: the evaluator changed the dataset split
to `train` but left nDTW's independent reference split at `val_unseen`.
All eight shard logs contain a missing train-episode key. Four partial
stat files (two distinct episode IDs, each evaluated by both arms) were
produced before failure; these were archived with the failed logs and are
excluded from the restarted screen. The complete failure record and log
hashes are in [`ndtw_reference_recovery.json`](ndtw_reference_recovery.json).

The evaluator now sets `TASK.NDTW.SPLIT=train` and verifies reference
locations for all 256 frozen episode IDs before inference. A real Habitat
reset check covered each of the four IDs that caused failure, with zero
model calls and zero navigation actions; see
[`repaired_reset_preflight.json`](repaired_reset_preflight.json).
[`run_positive_eval_recovery.sh`](run_positive_eval_recovery.sh) resumes
from the existing audited checkpoints, runs the same two-model parallel
evaluation, and independently recounts the compact episodes before writing
the suite completion marker. It does not invoke training. The original
evaluator, failed outputs, and launcher state remain preserved remotely.
The manifest, policy checkpoints, action prompts, generation parameters,
SR/SPL formulas, and original +2/+2-point gate are unchanged. No complete
paired navigation result exists yet; reserved inference remains unopened.
