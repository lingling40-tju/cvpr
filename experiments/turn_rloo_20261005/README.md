# Turn-level RLOO mechanism test (2026-10-05)

## Why this test exists

Two independent human reviewers labeled the same 49 selected, blinded
EventTrace transitions. They agreed on 33/49 three-way labels (Cohen's
kappa 0.459); 16 remain unresolved. Their short rationales expose two
structural problems with a before/after event reward: movement **toward** an
instruction landmark can be mistaken for completion, and earlier route
history needed to know whether a landmark was passed is absent from the
two supplied images. The selected cases cannot estimate population
accuracy, and we do not join the disputed labels to model predictions.
[`human_review_agreement.json`](human_review_agreement.json) gives the
answer-free aggregate counts and SHA-256 hashes of the two private CSVs.
No additional human annotations are needed for this optimization test.

Trajectory-level GRPO gives the same outcome advantage to all action
tokens of a rollout. A value-based PPO/GAE critic would be a natural way to
assign time-dependent credit, but this ActiveVLN environment does not
currently load a Qwen2.5-VL token-classification critic: the installed
Transformers mapping has no such model, and its TRL value-head fallback
is absent. We therefore test an actor-only, **turn-level return-to-go with
a leave-one-out baseline**, inspired by [RLOO](https://arxiv.org/abs/2402.14740)
and [REINFORCE++](https://arxiv.org/abs/2501.03262). This is our adaptation
to executed VLN turns; those papers do not establish its efficacy for VLN.
[PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438), and
[VLN-CE's public code](https://github.com/jacobkrantz/VLN-CE) are comparison
context. V-trace could improve asynchronous throughput but does not repair
an unobservable or misplaced event label.

## Frozen experiment

- Source: isolated remote tree
  `/Knowin/foundation/haozhiwang/whz/ActiveVLN_turn_rloo_20261005`.
- Start: the same Qwen2.5-VL-3B navigation SFT checkpoint as the existing
  group-four GRPO control.
- Train: the same 256 R2R-train episode rows as the control, SHA-256
  `6b34052cba8befed916a50c5a8ca4eb74c38b0f620e15cdacb2734d48a207d69`,
  seed 11, four sampled rollouts per row, 64 optimizer steps, 12-turn and
  36-command budgets, and the same actor learning rate and PPO clipping.
- Reward: simulator destination success at the actual terminal turn plus
  normalized **privileged geodesic progress** for executed non-STOP turns.
  Fixed coefficients are terminal outcome / 15, progress weight 0.5,
  per-turn discount 1.0. The latter is a training diagnostic and cannot
  be called a deployable semantic reward.
- Advantage at turn *t*: that rollout's future return from *t* minus the
  mean future return of the other still-active rollouts from the same
  four-sample group. A lone surviving rollout gets zero relative update.
  The turn's advantage is divided equally among its generated action
  tokens; observation tokens receive zero. This differs from broadcasting
  one sequence score to all turns. Training still uses the actor's PPO
  clipped objective.
- Evaluation: fixed 256 R2R **val-seen** episodes from 53 scenes, with
  a pre-result manifest SHA-256
  `d9ba66de3fb4fc070cae9449decb1427e80672d61523891af6f8d8ad45ad6a31`.
  Four Habitat shards evaluate the step-64 GRPO control and candidate on
  the same episode IDs, one stochastic decode each. Exact coverage and
  zero inference errors are required. This development split does not
  establish generalization to unseen scenes. Val-unseen has been reused
  heavily in prior method design and is not used for this screen.

The synthetic adapter check passed turn differentiation, STOP credit,
observation masking, and group-size enforcement. An actual two-step
environment/optimizer smoke finished with actor gradient norms 0.053 and
0.037. The 64-step run and automatic paired evaluation were started;
**no navigation result was available when this protocol was frozen**.
The watcher stops the isolated training Habitat service after training,
audits all 64 gradient steps, then evaluates control and candidate and
recounts paired SR/SPL and a descriptive scene bootstrap interval. Do not
interpret a one-seed positive screen as a confirmed improvement. Advance
to independent seed replication only if paired SR and SPL are each at
least +2 percentage points with exact coverage and no inference errors;
otherwise stop this mechanism. Even if it passes, keep the privileged
reward claim separate from observation-grounded semantics.

## Reproduction files

- [`turn_rloo_advantage.py`](turn_rloo_advantage.py): estimator and actual
  ActiveVLN action-span adapter.
- [`patch_trainer.py`](patch_trainer.py): hash-guarded isolated trainer patch.
- [`start_service.sh`](start_service.sh), [`run_train.sh`](run_train.sh):
  isolated simulator and train entry points.
- [`eval_val_seen_subset.py`](eval_val_seen_subset.py),
  [`run_val_seen_eval.sh`](run_val_seen_eval.sh),
  [`run_pair_after_train.sh`](run_pair_after_train.sh), and
  [`analyze_pair.py`](analyze_pair.py): frozen paired evaluation and audit.
- [`val_seen256_manifest.json`](val_seen256_manifest.json): exact episode IDs.
