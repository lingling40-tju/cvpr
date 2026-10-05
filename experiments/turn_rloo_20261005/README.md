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
0.037. **No navigation result was available when this protocol was
frozen.** The 64-step run subsequently finished with nonzero actor
gradients at every step (minimum 0.009). The watcher stopped the
isolated training Habitat service, then evaluated control and candidate
on the frozen 256-episode manifest. Both models have exact coverage
and zero inference errors.

| Arm | Successes / 256 | SR | SPL |
| --- | ---: | ---: | ---: |
| Destination-only GRPO control | 78 | 30.47% | 29.92% |
| Turn-level return-to-go / leave-one-out + privileged progress | 68 | 26.56% | 26.20% |

The candidate minus control difference is **-3.91 SR** and **-3.72
SPL percentage points**. Candidate-only successes number 15, versus
25 control-only successes. An independent recount of compact
per-episode statistics agrees exactly on the paired point estimates;
its descriptive 10,000-resample scene-cluster 95% intervals are
[-8.46, 0.40] SR and [-8.28, 0.68] SPL points. This is one training
seed and one stochastic decode on val-seen development episodes, and
the intervals are descriptive rather than confirmatory. The candidate
changes both temporal credit assignment and training reward, so this
comparison cannot identify the effect of RLOO alone. It gives no
unseen-scene or deployable semantic-reward result.

The prespecified advancement rule required at least +2 percentage
points in **both** paired SR and SPL, exact coverage, and zero inference
errors. It failed; no more seeds or val-unseen run were launched for
this mechanism. The first control evaluation stopped after 131 episode
records when a legacy Habitat waypoint-map drawer indexed one pixel
past the image boundary. We added only a bounds check to that map
overlay, resumed with the same checkpoint, decoding settings and
checksummed manifest, and reused the completed episode files. The
guard is supplied as `apply_map_boundary_fix.py`. It does not alter
navigation actions or reward metrics.

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

## Completed evaluation records

- `val_seen256_compact.json`: 256 paired episode rows with scene IDs,
  success, SPL, terminal distance, path length and early-stop reason.
- `control_validation.json`, `candidate_validation.json`,
  `train_audit.json`, and `suite.completed`: completion and gradient
  checks from the remote run.
- `paired_analysis_remote.json`: original paired analysis, including
  a 2,000-resample descriptive scene interval.
- `independent_recount.json`: independent local 10,000-resample
  recount. Reproduce it with:

  ```sh
  python3 experiments/turn_rloo_20261005/verify_turn_rloo_compact.py \
    experiments/turn_rloo_20261005/val_seen256_compact.json \
    experiments/turn_rloo_20261005/val_seen256_manifest.json \
    --output /tmp/turn_rloo_recount.json
  ```

The two human-label files remain private; the aggregate agreement
does not adjudicate their 16 disagreements or give model accuracy.
