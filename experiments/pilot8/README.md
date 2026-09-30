# Eight-episode feasibility pilot

The source is the 2026-09-30 ActiveVLN semantic reward experiment on R2R **training** episodes. Both arms started from the same Qwen2.5-VL-3B SFT checkpoint. Each arm ran two optimizer steps, four episodes per step and two rollouts per episode. The event lists in `curated_events.json` were reviewed manually. The frozen event verifier was Qwen3-VL-8B-Instruct.

`summary.json`, `control_rollout.jsonl`, and `event_rollout.jsonl` are copied from the original run, with no reconstructed or invented trajectories. The note `实验记录.md` describes setup, diagnostics, and limitations in detail. The original model checkpoints are large and are not included in this paper repository.

The 13/16 success counts per arm are training rollout outcomes. Differences in mean return measure different reward definitions; they are not evidence of a navigation gain. The 11/17 confirmed events from expert replays are diagnostics on the same curated episodes, not a verifier accuracy estimate.
