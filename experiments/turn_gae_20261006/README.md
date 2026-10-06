# Multimodal actor--critic fallback: interface preflight

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

An additional `real_model_gpu_gradient_smoke.py` used only physical GPU 3
after checking its memory use. The real 3B SFT checkpoint processed one
synthetic image and backpropagated through the scalar head, text
embeddings, and vision patch projection, all with finite nonzero
gradients. Peak PyTorch allocation was 15.39 GiB. Its exact output is
`real_model_gpu_gradient_smoke.json`. The active normalized trainer
continued from step 21 to step 24 on GPUs 0/1, and GPU 3 returned to
its 5.0 GiB baseline after the smoke.

After copying source to an independent `ActiveVLN_turn_gae_20261006`
tree, `patched_loader_gpu_smoke.py` exercised the **actual** patched
Verl loader. Its first synthetic-image run failed: the new scalar head
was FP32 but its hidden states were BF16. Before any GAE training or
navigation evaluation, we changed the new head construction to use
`torch_dtype`. The repeated GPU3 run then passed with nonzero head,
vision, and text gradients; see `patched_loader_gpu_smoke.json`.
The protocol records this pre-result technical amendment and the
earlier patch hash. Reapplying the corrected patch to pristine copies
of all three base files reproduced the isolated tree byte for byte;
`gae_source_isolation.json` records the hashes. The active normalized
tree was not changed.

A second pre-result source check found that Verl's batch validation
counted all three Ray-visible GPUs as actor GPUs, while the proposed
resource pools assign two to the actor and one to the critic. The
isolated GAE patch now validates actor and critic batch sizes against
their respective pool sizes. It again applied to pristine base files
and matched the isolated tree byte for byte. Both changes are listed
in `gae_conditional_protocol.json`; no GAE optimizer step or navigation
outcome existed at either amendment.

These checks establish an input/shape and one full-model gradient path
only. They do **not** prove that a full 3B critic can train with FSDP,
that Ray can allocate the critic without contesting
the actor/Habitat GPUs, or that GAE improves navigation. Those are
required before a fixed-budget real smoke and pilot. The trainer's
`compute_response_mask` uses `action_mask` when present, and the GAE
recurrence skips masked observation spans; this needs a real rollout
alignment audit as well.

A source trace found that the VLN agent writes each environment reward
on the last generated action token of its turn, while the naive reward
manager adds `env_reward` and returns zero additional score for R2R.
This is the intended GAE alignment, but an actual rollout may expose
an edge case. The isolated patch therefore aborts before GAE if any
nonzero token reward falls on a masked observation or padding token.

`gae_multimodal_critic.patch` is an unapplied source proposal. It adds
the scalar Qwen2.5-VL loader, an optional separate Ray critic pool, and
the fail-closed reward-mask assertion;
without `trainer.critic_gpus_per_node`, the existing pool mapping is
unchanged. A read-only `patch --dry-run -p1` passed against the current
isolated normalized source, whose `verl/utils/model.py`,
`verl/trainer/main_ppo.py`, and `verl/trainer/ppo/ray_trainer.py`
SHA-256 values are
`8bb8da222fd646c8391c239a18f209f26a53f7c941b3f05bf367eacda87abde4`
`c26468885b5ebf701be47c177be45c2ab48de72adaeb34d6f3f573e4b84925b8`,
and `ca3e7ec596f4c5cc13b6b574a3f71cd9040db6a34090e8776f2ce44a8288354b`.
The patch SHA-256 is
`9a79685cf412d6ba69eb8a0a255d37e51cbffbe50bb5383383ec59f9fb294b54`.
The live normalized source was not modified. A future GPU smoke would
expose physical GPUs 0, 1, and 3 to a fresh local Ray process, assigning
two logical GPUs to actor/rollout and one to critic while the Habitat
service remains on physical GPU 2. Placement and memory must be checked
at runtime, not inferred from this static patch.

The three previously frozen 256-item val-seen screens have zero
episode-ID overlap and cover 768 of the source split's 778 episodes.
Therefore another disjoint 256-episode val-seen screen is unavailable.
`gae_conditional_protocol.json` therefore freezes the complete 778-item
val-seen split, its source-order manifest, the same control checkpoint,
training budget and GAE settings, and the joint +2/+2 pp gate before any
GAE training. Every episode overlaps the union of prior development
screens except ten, so this comparison remains adaptive development
evidence despite using the entire split. It must not be called a clean
test or used to claim unseen-scene generalization. No additional human
annotation is needed.

The isolated GAE tree now has a conditional watcher. It waits for the
normalized-terminal suite to finish, checks all 256 episode records with
`verify_normalized_terminal.py`, and compares the independent SR/SPL
recount with the suite decision. A passing normalized result skips GAE;
otherwise it starts a separate Habitat service and runs only a real
two-step GAE smoke. `audit_gae_gradients.py` then requires actor and
critic gradients at both optimizer steps. The watcher stops after this
smoke so that actual Ray placement, reward masking, and memory can be
checked before committing the frozen 64-step budget. Neither a GAE
optimizer step nor a GAE navigation result was available when this
watcher was installed.
