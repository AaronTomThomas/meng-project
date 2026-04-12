"""Shared utilities for language model probe experiments."""

from __future__ import annotations

import hashlib
import math
import random
from typing import List, Sequence

import torch

from experiments.attention_learners import build_learners


class LearnerRegistry:
    """Factory/registry that provides callable learner implementations."""

    def __init__(self, learner_names: Sequence[str]):
        if not learner_names:
            raise ValueError("learner_names must be non-empty")
        self._names: List[str] = list(learner_names)
        self._impls = build_learners(self._names)

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def predict(
        self,
        learner: str,
        q: torch.Tensor,
        Kctx: torch.Tensor,
        Vctx: torch.Tensor,
        cfg,
    ) -> torch.Tensor:
        if learner not in self._impls:
            raise ValueError(f"Unknown learner '{learner}'")
        return self._impls[learner](q, Kctx, Vctx, cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first_tensor(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def short_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10]


def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    B, L, D = x.shape
    if D != num_heads * head_dim:
        raise ValueError("Last dim must match num_heads * head_dim")
    return x.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)


def causal_soft_attention_from_qkv(
    q: torch.Tensor,  # (B,H,L,Dh)
    k: torch.Tensor,  # (B,H,L,Dh)
    v: torch.Tensor,  # (B,H,L,Dh)
) -> torch.Tensor:
    B, H, L, Dh = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)

    causal = torch.triu(
        torch.ones(L, L, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
