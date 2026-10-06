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
presumed for this new screen. The turn-level RLOO implementation divides
each turn advantage across its generated action tokens, while GRPO
applies a normalized trajectory advantage to each action token. This
changes effective update magnitude in addition to temporal credit. A
pre-result, descriptive check of the first 37 optimizer steps found
mean raw TensorBoard actor gradient norms of 0.058 (terminal RLOO) and
1.659 (outcome GRPO control), with respectively 2 and 4 exact zero
updates. These are different policy trajectories, so the norm contrast
is not a causal estimate; any navigation difference must be reported as
a comparison of the complete update rules, not credit assignment alone.

## Frozen evaluation

The new 256-episode R2R val-seen manifest has 50 scenes and SHA-256
c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325.
Its episode IDs have zero overlap with the preceding 256-episode
turn_rloo screen. Six numeric IDs are reused between the 256 training
rows and this val-seen split (575, 585, 654, 671, 677, 678), but all six
map to different scenes in the authoritative train and val-seen datasets.
The training rows explicitly specify split=train; the numeric ID alone is
not a global trajectory key. See train_val_id_audit.json. The four
checkpoints each received one stochastic
decode at seed 11, four Habitat shards, exact ID coverage, and zero
inference errors. The primary paired measures are SR and SPL, each
against the existing outcome-only GRPO control. A candidate advances
to matched three-seed 512-row, 128-step replication only if both paired
SR and SPL reach +2.0 percentage points on this frozen screen.
Because three candidates share this development screen, selecting any
passing arm also selects on observed noise; the pilot threshold alone
cannot establish a reliable gain.

This is still val-seen development data and only one training seed.
Val-unseen has already been used repeatedly during method development,
so a positive screen here cannot establish unseen-scene generalization.
No further human labels are required for this optimizer test.

## Frozen factorial result

The four-arm suite completed. Each arm covered all 256 distinct episodes
in the frozen manifest, with zero inference errors. An independent
recount from the compact per-episode export agrees with the paired
analysis (differences below numerical precision). Percentage-point
differences are paired against the outcome-only GRPO control:

| Arm | Successes / 256 | SR difference (pp) | SPL difference (pp) | Joint gate |
| --- | ---: | ---: | ---: | --- |
| Outcome-only GRPO control | 97 | reference | reference | — |
| Combined turn RLOO + progress | 96 | −0.391 | −0.137 | fail |
| Terminal-only turn RLOO | 98 | +0.391 | +0.679 | fail |
| Dense-progress GRPO | 106 | +3.516 | +1.509 | fail |

Dense-progress GRPO improved SR on this development screen but did not
reach the frozen +2.0 pp SPL threshold. Its descriptive 95% scene-cluster
bootstrap intervals were [−2.24, +9.26] pp for SR and [−4.21, +7.17]
pp for SPL. All three candidates therefore failed the joint gate; none
is being scaled from this screen. This one-seed val-seen result does not
establish an unseen-scene improvement or a deployable semantic reward.

A post-result decomposition of the compact records helps localize the
SR--SPL gap without identifying a cause. Dense GRPO and control both
succeed on 71 episodes, where dense GRPO's mean path is 8.31 m versus
7.38 m for control; these episodes contribute −1.38 pp to the overall
paired SPL difference. Dense-only successes (35) contribute +12.94 pp,
while control-only successes (26) contribute −10.05 pp. The categories
condition on model outcomes, so this is descriptive and cannot justify
changing the frozen gate or tuning a new reward on this screen.

The independent recount initially stopped because the compact exporter
serialized an omitted raw `episode_id` field as JSON `null`, while the
verifier expected an absent key. After seeing the outcome, the verifier
was repaired to accept only a null or matching raw ID. The required
manifest order, expected shard files, per-episode data, and metric
formulas were unchanged. The original failure log is preserved on the
remote host. The repaired verifier passed both locally and remotely;
`factorial_compact.json`, `factorial_decision.json`,
`factorial_independent_recount.json`, and four `*.validated.json` files
record the result. This schema repair must not be described as a fully
unchanged pre-result verifier.

## Source and run status

The terminal-only arm changes one argument in the independently
audited turn_rloo_advantage adapter: progress_weight=0.0. A CPU check
proved its advantages invariant to arbitrary changes in the process
tensor, preserved the observation mask, and retained terminal outcome
contrast. A two-step real-environment smoke finished with nonzero
actor gradient norms 0.030 and 0.047. The 64-step run completed with
a saved step-64 checkpoint. `terminal_64_train_audit.json` verifies all
64 optimizer steps and 58 nonzero-gradient steps. Raw TensorBoard
independently confirms exact zero updates at steps 25, 34, 38, 42, 50,
and 58, each corresponding to an all-zero terminal-score batch. These
are expected possible batches under sparse outcome-only reward, not
training crashes. The diagnostic audit was corrected during the run to
report zero updates rather than require every batch to update. This
changed no rollout, reward, optimizer, or checkpoint code.

The reward-only arm keeps the baseline trajectory-level GRPO
normalization and changes its scalar group score to terminal outcome/15
plus 0.5 times the sum of executed-turn geodesic progress. The
simulator/token alignment is validated by the existing turn adapter.
Each eligible movement turn contributes
$(d_{t-1}-d_t)/\max(d_0,3\,\mathrm{m})$; a response containing STOP
receives zero auxiliary credit. Summing eligible deltas largely
telescopes to net distance reduction. GRPO still broadcasts one
trajectory-level contrast across action tokens, so this arm tests extra
ranking information for failed routes, not within-trajectory temporal
credit assignment. The separate turn-level arms test that update rule.
CPU checks established baseline advantage parity when progress is zero,
a masked observation token, and a nonzero all-failure-group progress
contrast. The terminal-only handoff closed Habitat port 5057 and
started the dense arm service on port 5058. Its two-step real-environment
smoke completed with actor gradient norms 1.981 and 2.006; the complete
`dense_2_train_audit.json` confirms both steps. The frozen 64-step
training run completed with the same train-file hash, seed 11, n=4,
and process weight 0.5. `dense_64_train_audit.json` verifies exactly
64 optimizer steps and 64 nonzero actor-gradient steps. The frozen
TensorBoard recount in `process_signal_audit.json` found ten dense-arm
batches with zero terminal score but nonzero advantage and actor
gradient; the outcome-only control and terminal-only RLOO had seven
and six zero-terminal-score batches, respectively, with zero gradient
on those same batches. These are training-signal diagnostics across
different policy trajectories, not navigation effects. The dense-arm
navigation result is reported in the factorial table above.

During training, GPU 0/1 held the two-GPU actor and GPU 2 ran the
Habitat service. These jobs therefore
use one training lane with CPU preparation overlapping GPU work; three
independent two-GPU training jobs cannot safely run concurrently on
four A800 GPUs in this configuration. The continuation watcher has a
lock and completion/failure markers.
It stops each training service before starting the next phase. The
four-arm evaluation suite completed using two vLLM servers on
ports 8126/8127; each used four Habitat shards on GPU 2.
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

`scale_replication_protocol.json` fixes the conditional 512-row,
128-step, three-seed extension and full 1839-episode val-unseen pairing
before the factorial outcome. `scale_control_reuse_audit.json` records
that the exact three existing step-128 control checkpoints were already
evaluated at their matching decode seeds on the same full manifest;
the completed validator checked all 1839 unique episode IDs and zero
inference errors for each seed. A scaled candidate must use that same
evaluator source and manifest before its metrics can be paired with
these controls. The full val-unseen split has been used in prior
development, so this extension remains a larger development
replication rather than a clean generalization test.
The generic `verify_three_seed_scale_raw.py` was frozen before new scale
outcomes. It checks each arm's exact shard files and validator, exports
compact paired episodes, and recomputes per-seed and three-seed SR/SPL.
A read-only backtest on the already completed oracle/control raw data
reproduced candidate/control successes 513/450, 525/545, and 585/492,
with mean paired SR +2.4651 and SPL +2.5108 percentage points. This
checks the verifier against known data; it is not evidence for a new
candidate.

## Conditional next mechanism, frozen before factorial outcomes

The active RLOO adapter gives each action token the turn contrast divided
by the turn's token count. To test whether this unintended update-scale
change masks a useful temporal credit signal, a separate CPU-checked
contingency assigns each action token the same standardized, active-peer
leave-one-out terminal contrast. This keeps group size four, terminal
feedback, 256 training rows, seed 11, and 64 steps; it excludes process
rewards. The source is `normalized_terminal_rloo_contingency.py` and the
pre-result specification is `normalized_terminal_protocol.json`.

Because no arm in the factorial passed its joint SR/SPL gate,
this method proceeded to a real two-step smoke.
Its separate 256-episode, 38-scene val-seen manifest has SHA-256
`39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b`
and zero episode-ID overlap with the two earlier frozen 256-item screens.
It is still development data. The real two-step smoke passed, and the
64-step training completed on 2026-10-06; the paired third-screen
evaluation remains pending, and no gain is presumed. An isolated remote source tree at
`ActiveVLN_norm_terminal_rloo_20261006` now contains only source and
symlinked data, without copied checkpoints. The fail-closed patch hash
audit is `normalized_source_patch.json`; a synthetic four-rollout batch
passed through the actual adapter on CPU, including token alignment and
masked observations. `normalized_run_train.sh` and
`normalized_start_service.sh` supplied the completed training run.
`verify_normalized_terminal.py` was frozen before this conditional
outcome; it requires exact four-shard coverage for both arms and
independently recounts paired SR/SPL from raw episode stats.
A separate locked watcher, `continue_normalized_if_needed.sh` (PID file
`ActiveVLN_norm_terminal_rloo_20261006/runlogs/conditional_chain/chain.launcher.pid`),
ran after factorial suite completion. It first ran
the frozen four-arm compact exporter and independent recount, checking
agreement with the original paired analysis. A passing factorial arm
causes it to exit for scale-up; only an independently confirmed all-fail
result starts the isolated normalized-RLOO smoke, training, and third
screen evaluation. Any recount error fails closed. The active factorial
source hash remains unchanged.

`normalized_scale_protocol.json` freezes a conditional 512-row,
128-step, three-seed continuation before the third-screen navigation
outcome. `normalized_scale_train.sh` preserves the pilot algorithm,
group size 4, and source hashes. The separate
`normalized_scale_if_pass.sh` watcher waits for the independent
third-screen recount and starts training only if both paired SR and
SPL reach the predeclared +2.0 percentage-point pilot gate. It otherwise
records a skip, leaving the distinct GAE fallback to run. The existing
same-seed controls may be reused for the 1839-episode comparison only
after the checkpoint, seed, evaluator, manifest, and coverage audit in
`scale_control_reuse_audit.json`; candidate-side parity and an
independent raw recount remain required. The larger val-unseen run will
remain development replication because that split was used earlier.
`normalized_scale_full_eval.sh` is staged for that conditional branch:
it uses the audited control evaluator, evaluates two candidate seeds at
once and the third afterward, then requires the independent raw
three-seed recount before writing an evaluation completion marker.

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
- terminal_2_train_audit.json, terminal_64_train_audit.json,
  dense_2_train_audit.json, and dense_64_train_audit.json: completed
  real-run gradient evidence.
- audit_process_signal.py and process_signal_audit.json: frozen
  TensorBoard recount of zero-terminal-score batches, actor gradients,
  and selected advantages across the completed 64 steps. This measures
  training signal, not navigation SR.
- train_val_id_audit.json: six split-local numeric ID collisions, all in
  different scenes; no same-scene collision in this audit.
- export_factorial_compact.py: exports exact four-arm per-episode results
  only after suite completion. verify_factorial_compact.py independently
  recounts paired SR/SPL and descriptive scene-cluster intervals from
  that compact export. Both were initially frozen before factorial
  outcomes; the verifier needed the post-outcome schema-only repair
  documented above.

The shared val-seen evaluator, full-label validator, and paired
analyzer are preserved in ../turn_rloo_20261005. Apply that package's Habitat waypoint-map bounds guard before rerunning the frozen
evaluation. Raw model weights, Matterport3D scans, and private
human-review CSVs are excluded.
