# Multimodal actor--critic fallback: CPU preflight only

This is preparation for a genuinely different optimizer if the frozen
normalized terminal-RLOO pilot also fails. It has **not** been trained or
evaluated as a VLN method. The active normalized pilot remains isolated.

The installed Verl `gae` path needs a token-value critic. The stock
`load_valuehead_model` cannot load the Qwen2.5-VL-3B SFT checkpoint as a
critic: its token-classification auto mapping is unavailable and TRL is
absent. The Qwen2.5-VL vision-to-text model itself can serve as a value
model by replacing its tied language-model head with an independent
one-dimensional linear head and then disabling future weight tying.
Its unchanged multimodal forward emits `(batch, sequence, 1)` logits,
the shape expected by Verl's critic worker.

Two CPU-only checks ran in the exact remote `activevln_train_env`
(Transformers 4.51.3) with no CUDA device visible:

1. `cpu_value_head_smoke.py` builds a tiny randomly initialized
   Qwen2.5-VL with one synthetic image. Changing the image changes the
   scalar value, and backward propagation gives nonzero value-head and
   vision-patch gradients. Synthetic GAE action-token advantages and
   returns are invariant to arbitrary value perturbations on three
   masked observation tokens. Exact output is in
   `cpu_value_head_smoke.json`.
2. `real_model_cpu_forward.py` loads the existing 3B SFT checkpoint
   from the training host, replaces the output head, and forwards one
   synthetic image. The value shape is `(1, 8, 1)`, the image has a
   nonzero effect, and the new head gets a nonzero gradient. The 3B
   backbone was frozen for this CPU gradient check. Exact output is in
   `real_model_cpu_forward.json`.

These checks establish an input/shape path only. They do **not** prove
that a full 3B critic can train with FSDP, that its full backbone gets
correct gradients, that Ray can allocate the critic without contesting
the actor/Habitat GPUs, or that GAE improves navigation. Those are
required before a fixed-budget real smoke and pilot. The trainer's
`compute_response_mask` uses `action_mask` when present, and the GAE
recurrence skips masked observation spans; this needs a real rollout
alignment audit as well.

`gae_multimodal_critic.patch` is an unapplied source proposal. It adds
the scalar Qwen2.5-VL loader and an optional separate Ray critic pool;
without `trainer.critic_gpus_per_node`, the existing pool mapping is
unchanged. A read-only `patch --dry-run -p1` passed against the current
isolated normalized source, whose `verl/utils/model.py` and
`verl/trainer/main_ppo.py` SHA-256 values are
`8bb8da222fd646c8391c239a18f209f26a53f7c941b3f05bf367eacda87abde4`
and `c26468885b5ebf701be47c177be45c2ab48de72adaeb34d6f3f573e4b84925b8`.
The patch SHA-256 is
`4dfc754916942d358b0be63cedbdd3658a20178e4418165b24de178465b36b56`.
The live normalized source was not modified. A future GPU smoke would
expose physical GPUs 0, 1, and 3 to a fresh local Ray process, assigning
two logical GPUs to actor/rollout and one to critic while the Habitat
service remains on physical GPU 2. Placement and memory must be checked
at runtime, not inferred from this static patch.

The three previously frozen 256-item val-seen screens have zero
episode-ID overlap and cover 768 of the source split's 778 episodes.
Therefore another disjoint 256-episode val-seen screen is unavailable.
A future GAE pilot must freeze a different development evaluation
source and its limitations before training; reusing an old screen would
be adaptive model selection. No additional human annotation is needed.
