# Credit assignment × progress reward: frozen group-four screen

This follow-up separates two changes that were coupled in the first
turn-level RLOO pilot. All four arms share the same Qwen2.5-VL-3B SFT
initialization, seed 11, 256 R2R-train rows, 64 optimization steps,
four sampled rollouts per row, and 12-turn/36-command budgets.

| Credit assignment | Terminal outcome only | Terminal outcome + training-only geodesic progress |
| --- | --- | --- |
| Trajectory-level GRPO | Existing exact group-four control | New dense_grpo arm |
| Turn-level return-to-go / leave-one-out | New terminal arm | Existing turn_rloo combined arm |

The dense process term is simulator geodesic progress and is unavailable
as a deployed semantic reward. This factorial is a mechanism test; the
previous combined-arm val-seen screen was negative, and no gain is
presumed for this new screen.

## Frozen evaluation

The new 256-episode R2R val-seen manifest has 50 scenes and SHA-256
c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325.
Its episode IDs have zero overlap with the preceding 256-episode
turn_rloo screen. Six numeric IDs are reused between the 256 training
rows and this val-seen split (575, 585, 654, 671, 677, 678), but all six
map to different scenes in the authoritative train and val-seen datasets.
The training rows explicitly specify split=train; the numeric ID alone is
not a global trajectory key. See train_val_id_audit.json. The four
checkpoints will each receive one stochastic
decode at seed 11, four Habitat shards, exact ID coverage, and zero
inference errors. The primary paired measures are SR and SPL, each
against the existing outcome-only GRPO control. A candidate advances
to matched three-seed 512-row, 128-step replication only if both paired
SR and SPL reach +2.0 percentage points on this frozen screen.

This is still val-seen development data and only one training seed.
Val-unseen has already been used repeatedly during method development,
so a positive screen here cannot establish unseen-scene generalization.
No further human labels are required for this optimizer test.

## Source and run status

The terminal-only arm changes one argument in the independently
audited turn_rloo_advantage adapter: progress_weight=0.0. A CPU check
proved its advantages invariant to arbitrary changes in the process
tensor, preserved the observation mask, and retained terminal outcome
contrast. A two-step real-environment smoke finished with nonzero
actor gradient norms 0.030 and 0.047. Its 64-step run has started.
At step 34, raw TensorBoard recorded an exact zero actor gradient and all
terminal scores were zero. This is an expected possible batch under a
sparse outcome-only reward, not a training crash. The diagnostic audit
was corrected during the run to require all 64 optimizer steps, finite
nonnegative norms, and at least one nonzero update; it now reports every
zero-or-console-rounded step instead of requiring all steps to be
nonzero. This changes no rollout, reward, optimizer, or checkpoint code.

The reward-only arm keeps the baseline trajectory-level GRPO
normalization and changes its scalar group score to terminal outcome/15
plus 0.5 times the sum of executed-turn geodesic progress. The
simulator/token alignment is validated by the existing turn adapter.
CPU checks established baseline advantage parity when progress is zero,
a masked observation token, and a nonzero all-failure-group progress
contrast. Its two-step smoke and 64-step run are queued after the
terminal-only run.

GPU 0/1 hold the two-GPU actor, GPU 2 runs the Habitat service, and
GPU 3 still hosts an earlier independent service. These jobs therefore
use one training lane with CPU preparation overlapping GPU work; three
independent two-GPU training jobs cannot safely run concurrently on
four A800 GPUs in this configuration. The continuation watcher has a
lock and completion/failure markers.
It stops each training service before starting the next phase. Once
training releases GPUs 0/1, two vLLM servers evaluate separate arms
concurrently on ports 8126/8127; each uses four Habitat shards on GPU 2.
The two waves cover all four models with the same frozen episode IDs,
decode seed, and per-arm validation checks. Parallel scheduling changes
throughput, not the reward or model comparison.

## Scale-up readiness (no result implied)

The existing outcome-only GRPO controls under
`ActiveVLN_three_directions_20261002/verl_checkpoints/oracle_exact512_control_128_seed{11,22,33}`
have completed checkpoints at step 128. Their run configs record group size 4,
the same 512-row train dataset SHA-256
`d664a6b14a51c6660a010280d6cf3eae232648789db8e40f167464eba062142f`,
and seeds 11/22/33. The parquet has 512 unique train-split episode IDs;
each control has a nonzero actor gradient at step 128. These controls could
save retraining if a candidate passes the frozen gate, after checking full
source/config parity and evaluating all arms on the same episode manifest.
A read-only source diff against the terminal-RLOO tree found the same
navigation prompt and training YAML. The differences are the intended
process-reward logging/tensor and advantage branches, plus a waypoint-map
bounds guard. The three control training logs agree on the SFT base,
4-rollout sampling, 12-turn/36-command budget, 15-point success reward,
zero nDTW and semantic weights, and 128 steps. The map guard affects
out-of-bounds waypoint painting, so any later comparison will use the
same guarded evaluator for every checkpoint. No scale-up candidate has
been launched on the basis of this inventory.

## Reproduction

- factorial_protocol.json, terminal_protocol.json, dense_protocol.json:
  pre-result specifications and source hashes.
- val_seen256_manifest.json: exact held-out episode IDs and scene IDs.
- terminal_advantage.patch, dense_grpo_trainer.patch: minimal changes
  against the prior isolated turn_rloo source.
- terminal_run_train.sh, dense_run_train.sh, terminal_start_service.sh,
  dense_start_service.sh: exact remote launch settings.
- continue_factorial.sh, run_factorial_suite.sh,
  audit_training_gradients.py: the sequential trainer/evaluator and
  exact-step gradient audit.
- terminal_2_train_audit.json: real two-step smoke evidence.
- train_val_id_audit.json: six split-local numeric ID collisions, all in
  different scenes; no same-scene collision in this audit.
- export_factorial_compact.py: exports exact four-arm per-episode results
  only after suite completion. verify_factorial_compact.py independently
  recounts paired SR/SPL and descriptive scene-cluster intervals from
  that compact export. Both were frozen before factorial outcomes.

The shared val-seen evaluator, full-label validator, and paired
analyzer are preserved in ../turn_rloo_20261005. Apply that package's Habitat waypoint-map bounds guard before rerunning the frozen
evaluation. Raw model weights, Matterport3D scans, and private
human-review CSVs are excluded.
