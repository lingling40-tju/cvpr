# Instruction-conditioned arrival transitions: frozen exploratory protocol

Status (2026-10-05): **CPU-only source preflight, exact fit/development RGB
replay, and a four-microstep nonzero-gradient model smoke passed. The
fixed 1,024-microstep fit is running; no model development result, audit,
or navigation result is yet available.** This hypothesis was
chosen after the single-frame goal-region occupancy head failed its fixed
development recall gate. The same eight train-development scenes have been
used for method design. The eight train-audit scenes and their RGB remain
unopened.

## Why change the representation

The earlier head scored each state separately. It reached AUC 0.6053 on
288 train-development crossing trajectories and recalled only 8.33% of
near-goal states at a 4.81% pooled false-positive rate. A text-and-motion
shortcut reached AUC 0.5884; shuffling the images within each scene gave
0.5897. This is weak evidence that a single first-person image exposes
the geometric 3 m destination boundary. We therefore test whether a
**joint comparison of two observations** can detect arrival evidence
associated with the instruction. This targets a transition rather than
an absolute distance class. It is a different representation and would
eventually require a different reward than the failed 0.1 continuation
penalty.

## Frozen source and cost gate

Use only the audited group-four R2R-train rollouts from seeds 11/22/33,
the scene split and hashes of the earlier boundary source, and the
natural instruction. The source preflight found 1,184 fit and 288
development crossing pairs from 260 and 62 episode IDs. Within the
three-state context preceding the crossing, 897/202 trajectories have
an adjacent nonarrival pair with both states at least 5 m from the
goal; 195/41 have an outside-boundary stationary pair. Across all
fit/development trajectories, 283/92 have a real near-goal retreat of
at least 0.5 m that finishes outside 3.5 m, from 130/30 episode IDs.
The fit/development crossing source includes 155/41 episode IDs with
a natural exact-same-start instruction whose alternative goal is more
than 7 m from the true goal. Near-goal unsuccessful crossings come from
209/43 episode IDs. These counts make both success and hard-negative
checks possible; they say nothing about visual separability.

The preflight for four-state local context counted 4,145 fit and 1,008
development frames beyond the already verified 2,944 crossing RGB
images. The [frozen two-view capture manifest](ordinal_progress/policy_preference/multiview_event_source/capture_manifest.json)
selects only the states needed for crossing, far nonarrival, and real
retreat: 2,356 fit and 588 development additional frames. It reuses
all 2,944 verified crossing images byte-for-byte. The manifest and
[privileged pair labels](ordinal_progress/policy_preference/multiview_event_source/privileged_pair_labels.json)
are separate. The [source report](ordinal_progress/policy_preference/multiview_event_source/frozen_report.json)
passes all fixed coverage gates: development has 62 crossing, 53 far,
30 retreat, and 41 exact-start wrong-instruction episode IDs. Use the
same GPU evaluation lock and at most four Habitat shards; hash-check
the original rollouts, verify every selected state geodesic against
the source, and never pass those distances to a model. Freeze a
deterministic manifest and separate label file before further RGB
rendering. Do not open audit RGB during development.

The [collector](collect_multiview_event_frames.py),
[independent verifier](verify_multiview_event_replay.py), and
[launcher](run_multiview_event_replay.sh) follow that manifest. A
one-record smoke replayed four states, copied two old JPEGs unchanged,
rendered two new JPEGs, and matched both geodesic sources with zero
drift. The full four-shard replay then verified 1,391 fit and 351
development records, 4,724 and 1,164 state JPEGs, respectively; all
2,944 prior crossing JPEGs were byte-identical, the 2,944 extra states
were rendered, and maximum source geodesic drift was 0.0 m. The
[compact smoke and full audits](ordinal_progress/policy_preference/multiview_event_replay/)
pin the exact capture and label hashes. Audit-scene RGB remains unopened.

## Model and fixed development decision

At each pair, give the navigation SFT Qwen2.5-VL-3B both ordered RGB
images in one cross-modal prompt, the full instruction, and a
deterministic terminal clause extracted from the same text. The prompt
contains no absolute turn index, success flag, geodesic distance, or
source class. The same prompt format is used for crossing, far
nonarrival, real retreat, and exact-start wrong-instruction pairs.
Train a scalar arrival-evidence head with LoRA on `q_proj`/`v_proj`.
Use balanced scene-and-episode sampling. A crossing with its correct
instruction is positive; the other three categories are negative.
The wrong-instruction example uses exactly the same two images as the
corresponding crossing. No synthetic frame reversal counts as a real
retreat negative. A source record with ambiguous action replay or
missing images is excluded by a deterministic pre-replay rule, never
after observing a model score.

First require a two-image input and four-microstep nonzero-gradient
smoke. Fix the full budget to 1,024 microsteps with accumulation four
and checkpoint candidates at steps 256/512/768/1024. Choose one
checkpoint on a fixed, hash-selected eight-record-per-scene development
subset by pooled AUC, then evaluate the full development partition
once. Choose one threshold maximizing crossing recall subject to a
5% pooled negative FPR cap. The complete development gate requires:

1. At least 50 crossing episode IDs, 50 far-nonarrival IDs, 25 real
   retreat IDs, and 30 exact-start wrong-instruction IDs.
2. Crossing recall at least 55% and unsuccessful-crossing recall at
   least 50% at pooled negative FPR at most 5%.
3. Real-retreat FPR at most 10% and wrong-instruction FPR at most 12%.
4. Pooled AUC at least five points above a text-only two-view prompt
   control, and at least five points above a same-scene image-shuffle
   control. Report episode- and scene-macro metrics as well as pooled
   rates.

If any check fails, stop this representation, do not inspect audit,
and do not tune the same development screen. If all checks pass,
freeze the checkpoint and threshold before a single train-scene audit
on the reserved eight scenes. The audit requires the same recall and
FPR gates; the model must then be tested through a separate,
same-budget group-four online reward pilot. The previous failed
continuation penalty supplies no evidence for this new reward.
An n=8 sensitivity test is considered only after a learned n=4 reward
has a clear positive navigation result. No reused val-unseen result is
an independent final test, and a single AI blind annotator is not
human ground truth.

The [two-view scorer](train_multiview_event_lora.py) passed that
four-microstep fit-only smoke with one nonzero-gradient optimizer update
and 3,008 fit pairs; its fixed full budget is running under GPU 1's
shared lock. A [text-only control](fit_multiview_text_only.py) trained
on the same fit pair classes without opening any image and obtained
development AUC 0.5177; it recalled only 4.17% of crossings at its
4.04% pooled FPR. The [fixed image-pair shuffle](evaluate_multiview_image_shuffle.py)
will use the selected two-view checkpoint and the same development
threshold after training. These are development diagnostics, not
navigation improvements.
