"""CPU-only Qwen2.5-VL scalar-head and masked-GAE feasibility smoke.

This tiny randomly initialized model checks the interface, not a trained
critic or navigation result. Run with the same Transformers/Verl environment
as the remote trainer and with CUDA_VISIBLE_DEVICES empty.
"""

import json

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from verl.trainer.ppo.core_algos import compute_gae_advantage_return


def make_tiny_value_model() -> Qwen2_5_VLForConditionalGeneration:
    vision = {
        "depth": 1,
        "hidden_size": 64,
        "hidden_act": "silu",
        "intermediate_size": 128,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 64,
        "fullatt_block_indexes": [0],
    }
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vision_config=vision,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        rope_scaling={"mrope_section": [2, 3, 3], "rope_type": "default", "type": "default"},
        tie_word_embeddings=True,
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    embedding = model.model.embed_tokens.weight
    model.lm_head = torch.nn.Linear(config.hidden_size, 1, bias=True)
    torch.nn.init.normal_(model.lm_head.weight, std=1e-3)
    torch.nn.init.zeros_(model.lm_head.bias)
    model.config.tie_word_embeddings = False
    assert model.model.embed_tokens.weight is embedding
    assert model.lm_head.weight is not embedding
    return model


def main() -> None:
    torch.manual_seed(20261006)
    torch.set_num_threads(2)
    model = make_tiny_value_model()
    ids = torch.tensor([[1, 122, 120, 120, 120, 120, 2, 3]])
    position_ids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    image_grid = torch.tensor([[1, 4, 4]])
    image = torch.randn(16, 3 * 2 * 14 * 14)

    def value(image_tensor: torch.Tensor) -> torch.Tensor:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=position_ids,
            pixel_values=image_tensor,
            image_grid_thw=image_grid,
            use_cache=False,
            return_dict=True,
        )
        assert output.logits.shape == (1, 8, 1)
        assert torch.isfinite(output.logits).all()
        return output.logits[0, -2, 0]

    score = value(image)
    image_effect = (value(image + 0.1) - score).abs().item()
    (score - 1.0).square().backward()
    head_gradient = model.lm_head.weight.grad.norm().item()
    vision_gradient = model.visual.patch_embed.proj.weight.grad.norm().item()
    assert image_effect > 0 and head_gradient > 0 and vision_gradient > 0

    # The trainer's response mask is the action mask. Observation tokens at
    # positions 2, 3, and 6 must not alter action-token advantages.
    action_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 1]], dtype=torch.float32)
    rewards = torch.zeros_like(action_mask)
    rewards[0, -1] = 1.0
    values = torch.zeros_like(action_mask)
    baseline_adv, baseline_return = compute_gae_advantage_return(
        rewards, values, action_mask, gamma=0.99, lam=0.95
    )
    perturbed_values = values.clone()
    perturbed_values[0, [2, 3, 6]] = torch.tensor([11.0, -17.0, 23.0])
    changed_adv, changed_return = compute_gae_advantage_return(
        rewards, perturbed_values, action_mask, gamma=0.99, lam=0.95
    )
    selected = action_mask.bool()
    assert torch.allclose(baseline_adv[selected], changed_adv[selected], atol=1e-6)
    assert torch.allclose(baseline_return[selected], changed_return[selected], atol=1e-6)

    print(json.dumps({
        "schema": "qwen25vl_tiny_scalar_head_gae_cpu_smoke_v1",
        "value_shape": [1, 8, 1],
        "image_effect_abs": image_effect,
        "head_gradient_norm": head_gradient,
        "vision_gradient_norm": vision_gradient,
        "action_advantages_invariant_to_observation_values": True,
        "action_returns_invariant_to_observation_values": True,
        "scope": "tiny synthetic CPU model and tensors only; no 3B load or navigation result",
    }, indent=2))


if __name__ == "__main__":
    main()
