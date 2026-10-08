# Additional SFT reference at matched inference precision

This additional diagnostic was frozen at **2026-10-08 08:11:05 UTC**, before
either SFT reference or the scale reserved screen had produced an episode.
Candidate-22 was training; the first freeze snapshot recorded step 97/128.
The original training, six-model evaluation, auto-dtype SFT schedule,
checkpoints, episode manifests and selection rules remain as frozen.

## Actual evidence motivating the addition

The [CPU configuration-resolution report](scale128/inference_dtype_cpu_resolution.json)
calls the actual installed vLLM 0.8.5.post1 resolver with the local Hugging
Face configurations and GPU visibility disabled. `EngineArgs.dtype`
defaults to `auto`. The unchanged initial SFT has configuration dtype
`bfloat16`, resolving to BF16. The completed control-11 step-128 export
has configuration dtype `float32`, resolving to FP16. Explicit `half`
resolves both to FP16. Both existing 64-step pilot engine startup logs
actually record FP16.

This identifies a precision mismatch in the default configuration.
It has not measured the SFT engine or a navigation effect from precision.
Each future engine's actual startup configuration is checked separately;
configuration resolution alone does not substitute for that check.

## Frozen additional evaluation

The [identity](positive_matched_precision_identity.json) has SHA-256
`72eb92266ca551fb0f300b96a7525769044b397483cd77b337db5279e56bb000`.
The new [suite](run_positive_matched_sft_suite.sh) waits for both original
scale/reserved and extra suites to complete and release their locks.
It first verifies actual FP16 startup logs for all reused trained models
on each screen, and actual BF16 logs for the original native SFT diagnostics.
Source, parent-protocol, evaluator, compatibility-shim and installed
vLLM configuration hashes are guarded.

The [new SFT worker](run_positive_matched_sft_model.sh) loads the unchanged,
fully hash-verified initial SFT using **`--dtype half`**. It adds one shared
reference on development256, reserved256 and full val-unseen1839: **2,351
additional SFT episodes**. Every planned trained raw evaluation is reused;
there is no additional inference of the six trained models. GPU0/1 host
models, GPU2 hosts four Habitat shards per model, and GPU3 is excluded.
Ports 8141/8142 and the existing per-GPU locks isolate this diagnostic.

The frozen generation parameters and episode IDs are unchanged: decode
seed 11, temperature 0.2, top-p 0.8, 512 response tokens, 12 turns,
76,800 image pixels, server context 16,384 and memory utilization 0.72.
The worker requires actual FP16 startup configuration before starting
Habitat, exact complete raw coverage, finite metrics and zero inference
errors before completing a model. Its final log hash is recorded after
its own server has shut down. Original raw analysis and the separate
compact verifier must agree before `suite.completed` is written.

New raw SFT outputs belong to `runlogs/positive_matched_precision_<role>/`.
Original native SFT outputs remain in `runlogs/positive_extra_<role>/`.
Both use the same checkpoint label, so role and result-root identity must
accompany every copied validator or record. The new suite additionally
writes `verified_precision_comparisons.json`, tying each comparison to
the new freeze, actual precision proofs and report hashes.

## Preflight and interpretation

The [real CPU preflight](scale128/precision_reference_cpu_preflight.json)
verified the new source freeze, parsed an actual old control engine log,
rejected an incorrect BF16 expectation and reused the identical proof.
It made zero new model calls. An unlaunched first draft was corrected to
hash logs after server shutdown; its [identity](scale128/precision_prelaunch_v1_identity.json)
and all four source files are preserved remotely under
`runlogs/positive_precision_prelaunch_v1/`. No SFT inference preceded
the correction or final freeze.

The waiting relay was actually launched at 2026-10-08 08:12:40 UTC;
its [launch record](scale128/precision_reference_launch.json) records PID
3745877 and the exact freeze digest. A live waiting process is not an
evaluation result.

Algorithm-versus-initialization comparisons must use the matched FP16
reference and retain the native BF16 results as a separate diagnostic.
This is one shared decode, with zero independent SFT training seeds.
The [configured rollout-seed limitation](SAMPLING_SEED_SCOPE.md), prior
adaptive validation use, post-pilot design and stochastic scheduling
limitations remain. No improvement over initialization is claimed until
the actual matched-precision results and independent recount are available.
