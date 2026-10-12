# LoRA-RLOO n=4 development pilot (2026-10-12)

A single configured-seed (11), 64-global-update LoRA-RLOO pilot was resumed from a two-update smoke run. It used the frozen 512-row R2R-train fit set, n=4, rank-16 LoRA on q_proj/v_proj, and the existing Qwen2.5-VL-3B navigation SFT initialization. The combined gradient audit covers all global steps 1–64 with nonzero actor gradients.

The pre-frozen comparison used the same 256-episode / 8-scene R2R-train development manifest and FP16 inference for the candidate, SFT reference, and matched GRPO anchor. Candidate evaluation has exact manifest coverage (four shards, 256 episodes) and zero inference errors.

| Comparison | Candidate | Reference | Paired ΔSR | Paired ΔSPL | Frozen gate |
|---|---:|---:|---:|---:|---|
| LoRA-RLOO vs FP16 SFT | 95/256 | 109/256 | −5.47 pp | −5.34 pp | Fail |
| LoRA-RLOO vs matched GRPO | 95/256 | 47/256 | +18.75 pp | +17.56 pp | Pass |

The scale rule required both SR and SPL to improve by at least 2 points versus both references. Because the candidate regressed against SFT, it is not eligible for expansion despite the gain over the weaker GRPO anchor. Independent compact recounts reproduce both paired reports. This is one seed on adaptively reused development scenes, not a clean test or generalization result. It does not demonstrate a deployable semantic reward.

Inference used the SFT base model with the step-64 adapter explicitly loaded by vLLM, FP16, seed 11, context 16384, and the frozen decoding settings. The actual engine startup proof and served-adapter identity are retained. The initial orchestration failure occurred before any evaluation episodes were generated; its diagnosis and evidence are retained separately.

The evaluation directory contains paired per-episode compact records, reports, independent recounts, frozen identities and completion evidence. The training directory contains the combined training audit and checkpoint payload identity (metadata and hashes, not full weights). The validation directory contains per-shard summaries and local independent recounts. The sources directory contains the exact evaluator, analyzer, compact verifier, validator, dtype verifier, and launcher used.
