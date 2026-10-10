# ReMax-style n=4 exploratory evaluation

This separate 64-update run compared a ReMax-style greedy baseline (one
`do_sample=False` trajectory) with four sampled trajectories per prompt. It
used the Qwen2.5-VL-3B SFT initialization, 512 fit rows, configured seed 11,
group size four, and KL coefficient 0.1. The 64 updates include a two-update
smoke and 62 resumed updates. The smoke gradient audit covers updates 1--2;
TensorBoard scalar events cover all 64 steps, with resumed console
cross-checks for steps 3--64. `training_combined_audit64.json` records the
source hashes and checks this split without rewriting the training log.

On the frozen manifest `412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241`
(256 episodes, 8 train scenes), the ReMax candidate has 39 successes and the
same-precision FP16 SFT reference has 91. Paired changes are -20.3125 SR and
-19.21886697597057 SPL percentage points. Both models cover the exact 256
manifest IDs and have zero inference errors. The independently rerun compact
verifier reproduces the paired metrics and scene bootstrap. The fixed
per-metric +2-point advancement gate fails; no expansion is justified.

This reused development screen is not a clean generalization test. The
single configured seed does not establish independent rollout streams. The
result is a large regression versus initialization, not evidence that ReMax
or group size four is generally unsuitable.

## Audit trail

- `evaluation_identity.json` and its hash bind the model, data, FP16 decode,
  analysis sources, and fixed gate. The versioned v3 identity changes only
  the wrapper/audit implementation; inference, metric code, and thresholds
  are unchanged.
- The combined training audit reads the two-step smoke audit and 64-step
  TensorBoard events, then matches resumed log scalars for steps 3--64.
- `checkpoint_identity.json` validates the four safetensors header payload
  byte ranges against the index total size; it does not hash tensor values.
- `remax_vs_sft_episodes.jsonl`, the report, candidate/SFT validators,
  dtype proofs, and independent recount support exact coverage and metrics.
- `audit_attempts/v1` and `audit_attempts/v2` preserve pre-inference
  failures. V1 incorrectly required the resumed log alone to contain smoke
  steps 1--2; v2 incorrectly compared full shard file bytes (including
  headers) with tensor payload bytes. Neither attempt opened candidate
  inference. The corrected v3 runner retains both historical records.
- `SHA256SUMS` authenticates the local archive contents.

The manifest used for the independent recount is included as
`reserved256_manifest.json`.
