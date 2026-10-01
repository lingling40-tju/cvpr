# Step-64 wrapper incident

While the branch and recovery 64-step jobs were active, their
`run_direction_pilot.sh` file was copied over in place to add future seed and
dataset options. Both shells subsequently resumed reading at a changed byte
offset and exited with code 127 (`trainer.logger=[console,tensorboard]: command
not found`). Their `train.log` files were truncated to that error. This is an
orchestration mistake; the original failure markers and truncated logs are
kept on the experiment server.

Both checkpoints were saved before the wrappers exited. The independent
`validate_checkpoint_tensorboard.py` check confirms for each arm: 64
contiguous original rollout records, 256 two-sample groups, nonidentical
trajectories, nonzero return variance, 64 TensorBoard `actor/grad_norm`
events with at least one nonzero gradient, and complete nonempty model shards.
The recorded validation JSON files are in this directory. They validate the
saved training artifacts; they do not substitute for held-out evaluation.

The remaining counterfactual run was launched with the corrected runner after
the first two jobs ended. The runner file must not be changed while that shell
is active. The branch and recovery checkpoints are being evaluated on the
fixed 256-episode val-unseen manifest using spare GPUs while counterfactual
training continues.
