# Conditional fallback: success-triggered same-episode resampling

This is a prospective experiment to run only if both ongoing optimizer pilots
fail their predeclared fixed-256 gain gate. The existing ActiveVLN agent code
already implements this option; this experiment is an ablation of that
implementation, not a claim to have invented dynamic sampling.

## Motivation

The destination-only two-sample 128-step seed-11 control had equal returns in
348/512 episode groups. With GRPO's within-group comparison, these ties can
remove the policy-gradient signal even when the trajectories differ. The
four-sample pilot increases the group size for every episode. This fallback
instead keeps two final samples per episode and resamples a group only when
neither original trajectory reaches the goal. The implementation permits at
most two extra two-sample attempts and keeps the last sampled group. It does
not ensure a successful group; all attempts and simulator cost must be
counted. It might bias training toward lucky successful rollouts or worsen
held-out performance.

## Paired protocol

1. Wait until both four-sample and KL-anchored pilots finish their fixed-256
   evaluations and fail the same-data control gate (paired SR > 0 and SPL >=
   0). The existing evaluation watcher must have released their dedicated
   simulators before this fallback begins.
2. Use the exact `branch_pilot_train.parquet` (SHA-256
   `a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3`),
   seed 11, the SFT initializer, the destination-only reward, two final
   trajectories per episode, and 64 steps. The sole intervention against
   `branch_control64` is `enable_dynamic_sampling=true` with
   `max_sample_attempts=2`. Use the original-reward 16-slot simulator service.
3. Run two steps first. Audit exact per-step episode pairing, no prefix
   replay, finite rewards and gradients, the code's
   `from_dynamic_sampling` marker, and the number of extra simulator
   trajectories parsed from the logged resampling attempts. Then run the
   64-step pilot with the same audit.
4. Evaluate on the frozen 256 val-unseen manifest, with exact episode
   coverage and zero inference errors. Compare paired SR/SPL against
   `branch_control64`. A positive pilot can trigger three-seed 128-step
   expansion and complete 1,839-episode evaluation; the pilot alone is a
   screening result, not a paper claim.

`run_dynamic_resampling_conditional.sh` implements the first four stages.
The script stays idle while either current optimizer experiment is still
pending or eligible for expansion. No result is recorded yet.
