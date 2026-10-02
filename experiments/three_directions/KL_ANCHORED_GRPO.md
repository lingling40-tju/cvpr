# Conditional alternative: anchor destination-only GRPO to the SFT policy

This is a prospective optimization experiment following negative semantic,
policy-prefix, distance-progress, and reference-route pilots. It keeps the
destination-only reward, two rollouts per episode, train rows, SFT
initializer, action budget, optimizer learning rate, and evaluation manifest
fixed. The only training intervention is an actor KL term relative to the
frozen SFT reference policy.

## Motivation and limits

On complete 1,839-episode val-unseen, the SFT initializer has 544 successes
(29.58% SR). The earlier destination-only 64-step controls have 479, 566,
and 502 successes for seeds 11, 22, and 33; their mean SR is 28.04%.
Thus, two seeds degrade relative to SFT while one improves. These checkpoints
used `algorithm.kl_ctrl.kl_coef=0` and
`actor_rollout_ref.actor.use_kl_loss=false`. The observation does not show
that KL will help; train rows and stochastic decodes also vary.

The [verl GRPO documentation](https://verl.readthedocs.io/en/latest/algo/grpo.html)
describes an actor KL penalty and coefficient 0.001. This test sets
`use_kl_loss=true`, `kl_loss_coef=0.001`, and leaves reward KL at zero,
using the existing `low_var_kl` estimator. It does not introduce a new
reward or semantic verifier. A penalty could prevent useful adaptation as
well as harmful drift, so only held-out SR/SPL decide whether it helps.

## Matched test

1. Start an isolated eight-simulator original-reward service on port 5014,
   GPU 1, while the four-sample run uses GPUs 2/3 and its own service.
2. Run a two-step seed-11 smoke on the exact `branch_pilot_train.parquet`
   (SHA-256 `a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3`).
   Require the same episode set at each step as `branch_control64`, two
   trajectories per episode, no prefix replay, finite destination rewards,
   finite actor KL loss, nonzero actor updates, and a saved checkpoint.
3. If the smoke passes, train 64 steps and audit all 256 unique train
   episodes and 512 rollouts. Evaluate on the frozen 256 val-unseen episodes
   against the same-data destination-only control, with exact coverage and
   zero inference errors. Expansion requires paired SR strictly higher and
   SPL nondecreasing; also report the SFT baseline.
4. A passing pilot triggers matched three-seed 128-step training on the
   fixed 512-row train data and complete 1,839 val-unseen evaluation. Report
   each seed, scene uncertainty, the 1,583 episodes outside the fixed screen,
   and inference and training costs. A single-seed pilot is not a paper claim.

The two-step smoke completed on the isolated service. Both steps used the
same four episode IDs as the two-sample destination-only control, with two
trajectories per episode and no replayed prefix. All eight groups had
distinct trajectories and four had return variance. The actor gradient
norms were 1.049 and 1.473, and the step-2 checkpoint was saved. The
resolved trainer configuration shows `use_kl_loss=True`, coefficient
0.001, and reward KL coefficient zero; the local actor code adds the KL
term to policy loss. The console rounds both observed `actor/kl_loss`
values to `0.000`, so the smoke confirms wiring but cannot measure the
penalty's magnitude. See `kl_anchor_smoke/paired_train_audit.json`.

The matched 64-step seed-11 pilot is running on GPUs 0 and 1, concurrent
with the four-sample pilot on GPUs 2 and 3. No held-out navigation result
exists for this arm yet.
The full-run auditor records zero-gradient steps and requires each to have
all four episode groups tied in return, while still requiring finite
gradients and at least one effective update overall.
