# Completed positive-trajectory replication

All six 512-row, 128-step, n=4 runs and the reserved/full/SFT evaluations
completed before this archive. Three configured-seed replications share
the original initial GPU generation rule; sampling-stream independence
has not been established. Native BF16 SFT diagnostics are preserved
separately from the matched FP16 reference.

## Verified navigation results

| Screen | Shared FP16 SFT successes | Candidate / control successes | Candidate minus control SR / SPL, pp |
| --- | ---: | --- | --- |
| Development256, 64 steps | 109 | 51 / 17 | +13.28 / +13.28 |
| Reserved256, 128 steps | 91 | 35/24; 23/23; 30/27 | mean +1.82 / +1.76 |
| Full1839, 128 steps | 555 | 243/223; 220/216; 254/217 | mean +1.11 / +1.07 |

On full1839 the matched FP16 SFT obtains SR 30.18%, SPL 29.23%.
Candidate mean SR/SPL are 13.00%/12.95%; its mean differences from SFT
are **-17.18/-16.28 pp**. Thus the observed candidate advantage over the
weak updated control does not exceed initialization. This direction is
not expanded further. All per-model episode sets are complete and final
inference-error counts are zero.

## Evidence layout and recount

- `runlogs/positive_matched_precision`: matched-FP16 episode exports,
  reports, remote recounts, three new local independent recounts and
  the actual-engine precision proof chain.
- `runlogs/positive_extra`: original native-SFT diagnostics and the
  independent six-model full raw recount with its paired episode exports.
- `runlogs/positive_scale`: reserved paired records and frozen training audits.
- `runlogs/positive_remaining_evidence`: the final three scalar, checkpoint
  structure and rollout-budget archives; those are training diagnostics.
- `prepared_data`: the fixed manifests and identities. No images or weights
  are included. `archive_transfer_inventory.json` checks the transferred
  source bytes and records pre-transfer actual log/config hash checks.

`../verify_positive_sft_compact.py` was run separately for development,
reserved and val_unseen with each screen's own SFT/trained validators.
The full check also reconciled all six trained models to the original
three-seed raw recount and seed/scene intervals. All checks passed.
The unchanged native outputs remain diagnostics; the paper uses matched
FP16 comparisons. This is exploratory: screens were reused adaptively,
the reserved scenes may have been seen in SFT, and there is one shared
SFT decode with zero independent SFT training seeds.

## Post-hoc diagnostic

The [generated-turn/termination export](posthoc_terminal_proposals/) checks all seven completed models without model calls. Updated arms have median five saved assistant turns versus twelve for SFT. This is a hypothesis-generating proposal/termination description, not an executed-action trace or a causal explanation, and changes no frozen gate.

## Budget-convention audit

The [source and recorded-outcome audit](posthoc_budget_alignment/) finds zero training outcome reward for budget exhaustion even at a logged terminal distance below 3 m (230/314 occurrences in two seed-11 fit runs), while the evaluator forces STOP at the turn limit (238 of the shared SFT's 555 full-split successes). All 8,192 compact records and the SFT counts pass independent local recount. This is a protocol difference and mechanism hypothesis, not a causal effect or repaired navigation result.


The budget audit also includes a [within-group reward-order recount](posthoc_budget_alignment/within_group_terminal_reward_order.json):
114/145 of 1,024 groups per arm contain a farther failed STOP rewarded
above a <3 m timeout (215/270 pairs). All 485 pairs pass independent compact
recount. Distance order is not instruction-following ground truth or a
causal effect; this informs only a possible later, separately matched
reward/credit mechanism if the current pilots fail.
