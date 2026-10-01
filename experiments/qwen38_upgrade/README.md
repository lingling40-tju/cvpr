# Upgraded semantic verifier experiment

The original Qwen3-VL-8B-Instruct parser is held fixed. A loopback proxy sends
only the event verification image pair and instruction span to DashScope
`qwen3.8-max-0902`, with temperature 0 and thinking disabled. Motion gates,
reward scale, navigation actor initialization, and the R2R training data are
unchanged. The API key is kept outside this repository and is never logged.
The user explicitly authorized sending these academic experiment images to
DashScope. Matterport3D images remain subject to its
[academic terms](https://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf).

## Frozen audit

The original 49 examples from 12 val-unseen episodes retain exactly the same
single-AI labels and image hashes. `old_448/` and `qwen38_448/` replay the
same image pairs after the 448-pixel JPEG encoding used by the training
client. Against those labels, three-way agreement is 27/49 for the local 8B
verifier and 30/49 for Qwen3.8-Max. Predictions of event completion that
match the label are 7/15 and 6/12, respectively; clear negatives predicted
as completed fall from 4 to 2. This small selected audit has **no independent
human ground truth** and does not establish verifier precision.

An earlier Qwen3.8-Max replay used raw 640 x 480 images
(`audit_640_qwen38_*`), which do not match training input. It is retained only
as an audit trail and excluded from the comparison above. A stricter temporal
prompt on that resolution (`strict_640/`) did not improve the completion
decisions. A medium-reasoning request exceeded the 150-second online timeout,
so the 256-step experiment uses the non-thinking configuration.

## Training and held-out test

The two-step `smoke/` check has 8/8 diverse GRPO groups, 5 groups with varied
returns, nonzero actor gradients, 14/16 rollouts with parsed events, and no
semantic-service errors. `../run_qwen38_256_suite.sh` runs 256 optimizer steps
for each destination-only and EventTrace arm at seeds 11, 22, and 33; this is
four times the previous per-arm budget. It records API counters, rejects
verifier errors, and checks trajectory diversity. The seven-checkpoint
1,839-episode val-unseen analysis is prepared in `../run_qwen38_full_suite.sh`.
Training rollout success is not a held-out navigation result. The paper must
not claim a gain until all runs and full-split evaluations are complete.
