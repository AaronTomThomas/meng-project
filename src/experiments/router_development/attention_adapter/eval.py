from __future__ import annotations

from typing import Dict, List

from experiments.router_development.attention_adapter.utils import lm_loss
import torch
import torch.nn as nn

from experiments.router_development.attention_adapter.adapters import AdapterModel
from experiments.router_development.attention_adapter.utils import lm_loss

@torch.no_grad()
def eval_baseline(model: nn.Module, chunks: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    n_examples = chunks.shape[0]
    for start in range(0, n_examples, batch_size):
        input_ids = chunks[start : start + batch_size].to(device)
        logits = model(input_ids).logits
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])
    return sum(losses) / max(1, n_examples)



@torch.no_grad()
def eval_wrapped(
    wrapped: AdapterModel,
    chunks: torch.Tensor,
    batch_size: int,
    *,
    collect_stats: bool = True,
) -> Dict[str, float]:
    wrapped.set_peft_eval_mode()
    losses: List[float] = []
    stats_accum: Dict[str, float] = {}
    n_stats_batches = 0
    n_examples = chunks.shape[0]
    for start in range(0, n_examples, batch_size):
        input_ids = chunks[start : start + batch_size].to(wrapped.device)
        logits = wrapped(input_ids)
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])
        if collect_stats:
            stats = wrapped.peft_stats(input_ids)
            for k, v in stats.items():
                stats_accum[k] = stats_accum.get(k, 0.0) + float(v)
            n_stats_batches += 1
    out = {"loss": sum(losses) / max(1, n_examples)}
    if collect_stats:
        for k, v in stats_accum.items():
            out[k] = v / max(1, n_stats_batches)
    return out
