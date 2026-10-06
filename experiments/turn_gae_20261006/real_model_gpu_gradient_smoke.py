"""Single synthetic-image 3B value-model backward on one isolated GPU.

Run only with CUDA_VISIBLE_DEVICES=3 on the training host after checking
that physical GPU 3 has ample free memory. No Ray process, optimizer step,
checkpoint write, navigation data, or Habitat environment is involved.
"""

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForVision2Seq


MODEL = Path("/Knowin/foundation/haozhiwang/whz/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one CUDA-visible GPU")
    torch.manual_seed(20261006)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    embedding = model.model.embed_tokens.weight
    model.lm_head = torch.nn.Linear(config.hidden_size, 1, bias=True, dtype=torch.bfloat16)
    torch.nn.init.normal_(model.lm_head.weight, std=1e-3)
    torch.nn.init.zeros_(model.lm_head.bias)
    model.config.tie_word_embeddings = False
    assert model.model.embed_tokens.weight is embedding
    assert model.lm_head.weight is not embedding
    model.to("cuda").train()
    torch.cuda.reset_peak_memory_stats()

    ids = torch.tensor([[1, config.vision_start_token_id] + [config.image_token_id] * 4 + [2, 3]], device="cuda")
    positions = torch.arange(ids.shape[1], device="cuda").view(1, 1, -1).expand(3, 1, -1)
    image_grid = torch.tensor([[1, 4, 4]], device="cuda")
    vision = config.vision_config
    image = torch.randn(
        16,
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=positions,
        pixel_values=image,
        image_grid_thw=image_grid,
        use_cache=False,
        return_dict=True,
    )
    assert output.logits.shape == (1, 8, 1)
    loss = (output.logits[0, -2, 0].float() - 1.0).square()
    loss.backward()
    head_grad = model.lm_head.weight.grad.float().norm().item()
    vision_grad = model.visual.patch_embed.proj.weight.grad.float().norm().item()
    text_grad = model.model.embed_tokens.weight.grad.float().norm().item()
    assert all(torch.isfinite(torch.tensor(x)) and x > 0 for x in [head_grad, vision_grad, text_grad])
    print(json.dumps({
        "schema": "qwen25vl_3b_scalar_head_single_gpu_gradient_smoke_v1",
        "checkpoint": str(MODEL),
        "visible_cuda_devices": torch.cuda.device_count(),
        "value_shape": [1, 8, 1],
        "head_gradient_norm": head_grad,
        "vision_gradient_norm": vision_grad,
        "text_embedding_gradient_norm": text_grad,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope": "one synthetic image, full-model backward on isolated GPU; no FSDP, optimizer, rollout or navigation result",
    }, indent=2))


if __name__ == "__main__":
    main()
