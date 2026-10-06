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

The three previously frozen 256-item val-seen screens have zero
episode-ID overlap and cover 768 of the source split's 778 episodes.
Therefore another disjoint 256-episode val-seen screen is unavailable.
A future GAE pilot must freeze a different development evaluation
source and its limitations before training; reusing an old screen would
be adaptive model selection. No additional human annotation is needed.
