# GAE/PPO fallback: source and resource preflight

Checked read-only on `wanghaozhihuoshanyun` while the frozen dense-GRPO
trainer was running. This is feasibility evidence, **not** a trained method
or navigation result.

- The installed Verl trainer has `adv_estimator=gae`, uses a learned
  `CriticWorker`, and computes masked token-level GAE in
  `verl/trainer/ppo/core_algos.py`. Its recurrence skips observation-token
  gaps when `action_mask` is present. The current VLN YAML selects GRPO;
  changing the estimator alone would activate a separate critic.
- The default critic path is an unrelated DeepSeek 7B model and must be
  overridden. `verl/utils/model.py:load_valuehead_model` first tries
  `AutoModelForTokenClassification`, then a TRL value-head fallback.
  CPU-only model-family checks for the current Qwen2.5-VL-3B SFT model
  found `model_type=qwen2_5_vl`, no token-classification mapping, a
  Vision2Seq mapping, and **no `trl` package** in the training environment.
  Thus stock GAE cannot be launched safely with this model as-is.
- The trainer's stock `main_ppo.py` maps actor and critic to the same Ray
  GPU pool. During the active two-GPU dense-GRPO run, GPUs 0 and 1 used
  roughly 43 GiB each; this is not a critic memory benchmark. GPU 3 was
  mostly idle, but assigning a separate critic pool would require an
  isolated source change. No critic allocation or package installation
  was attempted during the live experiment.

If the current frozen methods fail, a genuinely different actor-critic
pilot would first need an isolated multimodal value-head load/forward
smoke, exact action/observation masking and turn-reward alignment checks,
and a measured two-step memory/gradient smoke. Only then should a fixed
budget, group-four GAE/PPO comparison be launched. Its training reward
and evaluation split must be frozen before looking at its outcomes.
