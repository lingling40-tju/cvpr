# Reward prototype integration

The implementation was developed as a modification of [ActiveVLN](https://github.com/arvillion/ActiveVLN) at commit `3a0c63b00e4f42c828cc74c3554afce17641da60`, which is licensed under Apache-2.0. `activevln.patch` contains modifications to tracked ActiveVLN files; `semantic_reward/` contains the new environment wrapper and state tracker; `tools/` contains the verifier service and pilot scripts. Preserve the upstream Apache-2.0 license and notices when redistributing a combined checkout.

The scripts encode the original cloud checkout paths and are provided to document the actual pilot command. Adapt paths and model locations for a fresh deployment. The reward verifier accesses simulator displacement; the navigation actor does not.
