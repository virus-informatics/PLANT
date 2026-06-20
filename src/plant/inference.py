"""Inference helpers for PLANT."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch


@torch.no_grad()
def embed_sequences(
    model,
    dataloader,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> np.ndarray:
    """Return PLANT latent coordinates for virus sequences."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    mixed_dtype = None
    if torch.cuda.is_available() and device.type == "cuda":
        if use_bf16 and torch.cuda.is_bf16_supported():
            mixed_dtype = torch.bfloat16
        elif use_fp16:
            mixed_dtype = torch.float16
    ctx = (
        torch.autocast(device_type="cuda", dtype=mixed_dtype)
        if mixed_dtype is not None
        else nullcontext()
    )
    with ctx:
        for batch in dataloader:
            outputs = model(
                input_ids_virus=batch["input_ids_virus"].to(device),
                attention_mask_virus=batch["attention_mask_virus"].to(device),
            )
            outs.append(outputs.hidden_state_virus.float().cpu().numpy())
    return np.concatenate(outs, axis=0)
