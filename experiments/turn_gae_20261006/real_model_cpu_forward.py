"""CPU-only SFT Qwen2.5-VL scalar-value-head load and image forward.

Run on the training host with the activevln_train_env interpreter. This
checks only model loading and the multimodal forward contract; it does not
train a 3B critic, exercise FSDP, or estimate navigation performance.
"""

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForVision2Seq


MODEL = Path("/Knowin/foundation/haozhiwang/whz/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn")


def main() -> None:
    torch.manual_seed(20261006)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    assert config.model_type == "qwen2_5_vl" and config.tie_word_embeddings
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.requires_grad_(False)
    embedding = model.model.embed_tokens.weight
    model.lm_head = torch.nn.Linear(config.hidden_size, 1, bias=True, dtype=torch.bfloat16)
    torch.nn.init.normal_(model.lm_head.weight, std=1e-2)
    torch.nn.init.zeros_(model.lm_head.bias)
    model.config.tie_word_embeddings = False
    assert model.model.embed_tokens.weight is embedding
    assert model.lm_head.weight is not embedding
    model.eval()

    ids = torch.tensor([[1, config.vision_start_token_id] + [config.image_token_id] * 4 + [2, 3]])
    positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    image_grid = torch.tensor([[1, 4, 4]])
    vision = config.vision_config
    image = torch.randn(
        16,
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size,
        dtype=torch.bfloat16,
    )

    def score(pixels: torch.Tensor) -> torch.Tensor:
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            pixel_values=pixels,
            image_grid_thw=image_grid,
            use_cache=False,
            return_dict=True,
        )
        assert output.logits.shape == (1, 8, 1)
        assert torch.isfinite(output.logits).all()
        return output.logits[0, -2, 0].float()

    first = score(image)
    with torch.no_grad():
        shifted = score(image + torch.tensor(0.25, dtype=image.dtype))
    image_effect = (shifted - first.detach()).abs().item()
    (first - 1.0).square().backward()
    head_gradient = model.lm_head.weight.grad.float().norm().item()
    assert image_effect > 0 and head_gradient > 0
    print(json.dumps({
        "schema": "qwen25vl_3b_scalar_head_cpu_forward_v1",
        "checkpoint": str(MODEL),
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "value_shape": [1, 8, 1],
        "image_effect_abs": image_effect,
        "head_gradient_norm": head_gradient,
        "scope": "3B SFT load, one synthetic image forward, head-only gradient on CPU; no FSDP or full-critic gradient",
    }, indent=2))


if __name__ == "__main__":
    main()
