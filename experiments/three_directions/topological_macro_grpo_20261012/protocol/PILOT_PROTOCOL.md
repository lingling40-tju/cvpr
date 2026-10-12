# Topological macro-action GRPO pilot

## Motivation and method

Earlier VLM action-token and turn-level RL pilots did not beat the matched SFT navigator. This pilot tests a structurally different policy: ETP-R1's topological graph planner selects frontier/waypoint macro actions and a low-level controller routes between nodes. GRPO is applied to the graph policy, reducing the number of learned decisions in a long navigation episode.

Reference implementation: ETP-R1, official repository commit `a94b5c8fe20d1631e9e150c430a925543eb1cba9`, arXiv:2512.20940. The upstream default group is eight; this pilot fixes the group at four. A one-line R2R reward correction changes the terminal success threshold from 1.5 m to 3.0 m so the training success/SPL reward matches the R2R-CE evaluator's configured success radius. The RxR branch is unchanged. All other reward terms and the graph planner remain from the pinned implementation.

## Frozen data and compute

- Reuse the existing R2R-train `fit512` and `development256` episode IDs. CPU verification confirms 512 fit IDs and 256 development IDs map exactly to the local R2R-CE episode database, with zero overlap.
- Fit: 512 episodes across 45 scenes. Development: 256 episodes across 8 scenes. The development screen has been adaptively reused and is not a clean test.
- Training: R2R DAgger checkpoint initialization; configured seed 11; four GPUs in one distributed job; two environments per GPU (eight episode groups per update); GRPO group size 4; one update epoch; 64 optimizer updates. This yields 8 x 4 x 64 = 2048 recorded rollout trajectories when every environment group completes. A separate two-update multimodal smoke validates the stack and is excluded from the 64-update pilot budget.
- Candidate and DAgger checkpoint will be evaluated on the exact 256 development IDs. The existing Qwen SFT raw per-episode export on those same IDs is a second comparator; no new Qwen inference is needed.
- Greedy evaluation, identical Habitat scene assets, success radius 3.0 m, and per-episode IDs/SR/SPL are required. Reserved and val-unseen splits remain closed.

## Decision gate

The graph-GRPO candidate must exceed both (a) its ETP-R1 R2R DAgger initialization and (b) the current Qwen SFT reference by at least +2.0 percentage points in paired SR and paired SPL. All four metric comparisons must pass to justify a separate multi-seed scale study. Otherwise stop this mechanism without threshold tuning or evaluation on another held-out screen.

## Interpretation limits

This is a one-configured-seed exploratory pilot on a reused development split. The graph policy uses a different architecture and pretrained checkpoint than the Qwen VLM, so comparisons to Qwen are system-level, while the paired ETP DAgger comparison isolates the RFT stage. Passing a single screen is not robust generalization evidence. The method uses simulator outcomes and is not a semantic-reward deployment claim.

## Pre-run resolved implementation settings (frozen before the GPU smoke)

The official R2R final-RFT wrapper settings are used where the protocol had not yet recorded an exact value: learning rate 2e-5, warmup 0, cosine minimum LR ratio 0.25, GRPO beta 0.04, epsilon 0.2, dropout 0.10 with all dropouts and sampling dropout enabled, AMP disabled, max gradient norm 2.0, and temperature 2.0 to 1.0. This pilot sets `sample_num=4`, `update_epochs=1`, `log_every=1`, and two environments per rank (four ranks), so each logged interval is one optimizer update. It retains the official R2R choices `max_traj_len=15`, `max_text_len=150`, location noise 0.5, waypoint augmentation, teleport backtracking, and sliding enabled. The exact task/train/smoke YAMLs and their SHA256 values are recorded in `runlogs/freeze/resolved_pilot_v2.json`; the first path-invalid config and its hash are retained and not executed. Checkpoint output directories are unique per attempt.
