# Sampling-seed scope of the frozen positive-trajectory expansion

This audit covers the current positive-trajectory source tree and the first
recorded batch of its three completed 128-step runs. It does not establish
the sampling behavior of earlier experiments in other source trees.

## Observed configuration and recorded trajectories

The training script passes the configured seed to both the data loader and
the rollout engine. The standard SamplingParams constructor logs seeds 11
or 22 as expected. In the VLN agent, however, the cloned parameters used
for actual requests set `seed = None`, `n = 1`, and cap each response at
512 tokens. Clearing the request seed avoids restarting each group's four
requests at the same sampling state. The FSDP/vLLM sharding manager separately
initializes its GPU generation state with `1000 + DP rank`; it saves and
restores the training RNG state around generation.

The [source and first-batch audit](scale128/sampling_seed_provenance_three_completed.json)
records hashes of all five inspected source files, the auditor, and each
complete raw rollout file. Control-11, candidate-11, and control-22 have
identical first-batch signatures for all 32 trajectories. The signature
includes episode ID, generated responses, executed actions, reward,
success, end reason, and final distance; timing is excluded. Within each
of the eight episode groups, all four signatures are distinct. Thus this
batch has group-four diversity while the checked configured-seed starts
share the same recorded actions and outcomes.

The separate [executed-action export](scale128/first_batch_executed_actions_three_completed.json)
checks this diversity without generated text: all eight groups in each
of the three first batches have four distinct flattened action sequences,
four distinct turn-bounded action sequences, and four distinct terminal
reward/success/distance tuples. Its [read-only auditor](audit_positive_executed_actions.py)
ties the records to the previous sampling-provenance and raw rollout
hashes. A local recount of the exported actions reproduces every group
count and all 32 shared episode/action sequences across the three starts.
This is first-batch behavioral diversity, not a claim about later batches
or independent rollout streams.

The [raw budget audit](scale128/rollout_budget_local_recount_three_completed.json)
also establishes identical per-step episode membership. The first actor
gradient norms nevertheless differ between control-11 and control-22
(approximately 1.2135 and 1.2254 in the stored TensorBoard exports).
The audit does not show that entire trainings are identical, that every
configured seed is ignored, or what causes later run variation. In
particular, engine and training RNG settings remain separately configured.
No live CUDA RNG state was inspected.

Report this expansion as **three configured-seed replications with a common
initial GPU generation-state rule**. Independence of all rollout sampling
streams is unestablished. Keep the original seed labels, source identities,
checkpoints, budget, and evaluation schedule. Finish all six matched runs
and the planned evaluations under this frozen runtime; do not relabel their
variance as fully independent sampling-seed variance.

To repeat the read-only audit after all six runs complete, use a fresh output:

```sh
CUDA_VISIBLE_DEVICES="" python3 tools/audit_positive_sampling_seed.py \
  --root "$PWD" --output runlogs/positive_scale/sampling_seed_provenance_six_completed.json
```

## Unapplied correction for a future isolated experiment

[configured_rollout_rng.patch](configured_rollout_rng.patch) changes only
the manager's initial generation-state seed to `configured rollout.seed +
DP rank` and requires that configuration field. TP ranks in the same DP
group use the same stream initialization. The agent's per-request seed
stays unset so successive requests consume the advancing state. Training
RNG capture/restore is retained. The original manager source SHA is
`e1204549388f04853e89a8baffc608c27b9fa4fd5cfa95eb332fb67e822e637d`;
the patched source SHA is
`f1a6c9d0ee5315a64fd0178cf039266ce00eea5ef107f8991aa2ae46e72fe9e0`.
The patch has **not been applied to the running experiment**.

The [CPU fixture](check_configured_rollout_rng_patch_cpu.py) applies the
actual patch in memory, checks both full-source hashes and Python syntax,
and executes the constructor RNG statements with a CPU `torch.Generator`.
The [recorded check](scale128/configured_rollout_rng_patch_cpu_check.json)
confirms common legacy states for seeds 11/22, distinct patched states,
repeatability at the same seed/rank, distinct DP-rank states, restoration
of the training state, and rejection of a missing configured seed. This
is a constructor-subset check, not CUDA, model-inference, or navigation
validation. Reproduce it without importing or changing the live manager:

```sh
CUDA_VISIBLE_DEVICES="" python3 check_configured_rollout_rng_patch_cpu.py \
  --source /path/to/original/verl/workers/sharding_manager/fsdp_vllm.py \
  --patch configured_rollout_rng.patch --output /tmp/fresh_rng_cpu_check.json
```

Any future use requires a fresh isolated source tree with the guarded
original hash, a frozen patched identity and budget, and the existing GPU
locks after the current suites release GPU0/1/2. Before a new pilot, run
real two-step group-four smoke tests at configured seeds 11 and 22. Record
actual initial CUDA generation-state hashes, check TP agreement within each
DP group and variation across configured seeds, retain within-group
trajectory diversity, and verify finite masked actor updates. The CPU
fixture does not substitute for these checks. Do not start another GPU
job while the current expansion and its evaluations own those resources.
