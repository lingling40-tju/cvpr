"""Exercise the isolated Verl Qwen2.5-VL value loader on one GPU.

Run from ActiveVLN_turn_gae_20261006 with its directory on PYTHONPATH
and CUDA_VISIBLE_DEVICES=3. Synthetic image only; no Ray or optimizer.
"""

import json
from pathlib import Path

import torch
from transformers import AutoConfig
from verl.utils.model import load_valuehead_model


MODEL = Path("/Knowin/foundation/haozhiwang/whz/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one CUDA-visible GPU")
    torch.manual_seed(20261006)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config.num_labels = 1
    model = load_valuehead_model(str(MODEL), torch.bfloat16, config, False)
    assert model.config.tie_word_embeddings is False
    assert model.lm_head.out_features == 1
    model.to("cuda").train()
    torch.cuda.reset_peak_memory_stats()

    ids = torch.tensor([[1, config.vision_start_token_id] + [config.image_token_id] * 4 + [2, 3]], device="cuda")
    positions = torch.arange(ids.shape[1], device="cuda").view(1, 1, -1).expand(3, 1, -1)
    grid = torch.tensor([[1, 4, 4]], device="cuda")
    visual = config.vision_config
    image = torch.randn(
        16,
        visual.in_channels * visual.temporal_patch_size * visual.patch_size * visual.patch_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=positions,
        pixel_values=image,
        image_grid_thw=grid,
        use_cache=False,
        return_dict=True,
    )
    assert output.logits.shape == (1, 8, 1)
    loss = (output.logits[0, -2, 0].float() - 1.0).square()
    loss.backward()
    metrics = {
        "head": model.lm_head.weight.grad.float().norm().item(),
        "vision": model.visual.patch_embed.proj.weight.grad.float().norm().item(),
        "text": model.model.embed_tokens.weight.grad.float().norm().item(),
    }
    assert all(torch.isfinite(torch.tensor(v)) and v > 0 for v in metrics.values())
    print(json.dumps({
        "schema": "qwen25vl_3b_patched_verl_loader_gpu_smoke_v1",
        "checkpoint": str(MODEL),
        "value_shape": [1, 8, 1],
        "gradient_norms": metrics,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope": "isolated patched loader, one synthetic image, full-model backward on GPU3; no FSDP, Ray or navigation result",
    }, indent=2))


if __name__ == "__main__":
    main()
