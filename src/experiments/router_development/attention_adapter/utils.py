from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    bsz, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    return x.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

def block_forward_hidden(block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    out = block(hidden_states, use_cache=False)
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "hidden_states"):
        return out.hidden_states
    raise TypeError(f"Unexpected transformer block output type: {type(out)}")

def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))