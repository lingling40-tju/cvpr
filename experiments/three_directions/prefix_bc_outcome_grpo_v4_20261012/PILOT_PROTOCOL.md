# Expert-aligned prefix BC + outcome GRPO pilot

## Rationale
Earlier action-level REINFORCE/RLOO and dense geometry pilots underperformed the matched Qwen SFT. This pilot adds a narrowly scoped token-level imitation term only for the contiguous generated action prefix that exactly matches the frozen reference action sequence after the already-executed history. At the first action mismatch, the mask ends; later turns receive no imitation label. The regularizer uses existing on-policy outputs and R2R training actions; it adds no model inference, human labels, or simulator reward. Outcome GRPO remains unchanged.

## Frozen setup
- Same Qwen2.5-VL-3B R2R SFT initialization and standard outcome GRPO baseline.
- R2R fit512 only, batch 8, group size 4, configured seed 11, learning rate 1e-6, KL coefficient 0.1, 64 optimizer updates including the two-update smoke/resume.
- Auxiliary loss: 0.05 times mean negative log probability over tokens belonging to exactly expert-aligned contiguous action turns. No loss after the first divergent turn; no update from empty masks.
- Same development256 (8 train scenes) and paired SFT raw outputs; require exact episode coverage, zero inference errors, and paired SR/SPL both at least +2 percentage points vs SFT. Development was adaptively reused, so this is exploratory only.
- No scale run unless both frozen metrics pass. No reserved or val-unseen use for tuning.

## Fail-closed checks
The mask requires exact turn/token span count and batch metadata. Exact string mismatch terminates the prefix mask, not the row. We log aligned rows/turns and token fraction. The smoke must have at least one aligned turn, finite auxiliary loss, nonzero actor gradients, and finite KL before 64-step continuation. The method is not a semantic verifier and uses reference action supervision from R2R training data.

## Versioned smoke repair (2026-10-12)
The first v1 smoke failed closed before any optimizer update because the rollout worker removes `history_actions` from returned metadata. Its failed log and identity are retained under `runlogs/archive_failed_v1/`. A separately identified v2 source attaches the already-frozen expert history to the trainer batch immediately before repeat/union; this does not change labels, mask rule, loss, reward, data split, budget, or gate. V2 requires a fresh two-step smoke and its own source identity before any continuation.


## Versioned actor-mask repair (2026-10-12)
V2 completed two GRPO updates with finite metrics, but actor-side `expert_prefix_bc_tokens` was zero although the trainer-side prefix mask was nonempty. V2 is retained in `runlogs/archive_smoke_v2/` and does not validate the auxiliary objective. V3 uses the already validated trainer-side action-token mask directly in the BC term; the mask is constructed from action spans and intersected with the frozen GRPO response mask before dispatch. V3 must show nonzero actor-side BC tokens, finite nonzero BC loss, finite KL, and nonzero actor gradient in both smoke updates before continuation.


## Versioned sparse-prefix smoke gate (2026-10-12)
V3 had finite KL and nonzero actor gradients on both updates, but only update 2 had actor-visible BC tokens/loss (1.312 mean tokens, BC loss 0.010); update 1 had zero. Its frozen both-updates BC gate therefore failed and its log is retained in `runlogs/archive_smoke_v3/`. V4 keeps the same source/loss/data and changes only the implementation smoke acceptance to account for naturally sparse aligned prefixes: across the two smoke updates at least one must show actor-visible BC tokens and finite nonzero BC loss, while both updates must have finite KL and nonzero actor gradients. Empty masks contribute zero BC loss and do not create an auxiliary update. This is an implementation smoke criterion only, not a navigation efficacy gate.
