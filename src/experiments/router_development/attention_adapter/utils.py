from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
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


def model_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    return output.logits if hasattr(output, "logits") else output


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_count = int((shift_labels != -100).sum().item())
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, token_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_model_family(model_name_or_path: str) -> str:
    lower = model_name_or_path.lower()
    if "pythia" in lower or "gpt-neox" in lower:
        return "pythia"
    return "gpt2"


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
