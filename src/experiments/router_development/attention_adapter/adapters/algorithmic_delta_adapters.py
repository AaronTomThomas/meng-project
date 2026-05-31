from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from experiments.attention_learners import (
    BaseAttentionLearner,
    LEARNERS,
    LearnerHyperParams,
    build_learners,
)
from experiments.language_model_probes.probe_utils import merge_heads, split_heads
from experiments.router_development.attention_adapter.adapters.base import AdapterModel
from experiments.router_development.attention_adapter.adapters.utils import delta_stats


def parse_csv(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    if value.strip() == "all":
        raise ValueError("'all' is only valid for head_indices, not integer CSV fields")
    return [int(item) for item in parse_csv(value)]


def parse_head_indices(value: str, num_heads: int) -> list[int]:
    if value.strip() == "all":
        return list(range(num_heads))
    heads = parse_int_csv(value)
    if not heads:
        raise ValueError("head_indices cannot be empty")
    for head_idx in heads:
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"head_idx={head_idx} out of range for num_heads={num_heads}")
    return heads


class ScalarAlgorithmicDeltaRouter(nn.Module):
    def __init__(self, num_actions: int, alpha_scale: float):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.zeros(num_actions))
        self.alpha_scale = float(alpha_scale)

    def forward(self, router_input: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _hidden = router_input.shape
        alpha = self.alpha_scale * torch.tanh(self.raw_alpha)
        return alpha.view(1, 1, -1).expand(bsz, seq_len, -1)


class MLPAlgorithmicDeltaRouter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        dropout: float,
        alpha_scale: float,
    ):
        super().__init__()
        self.alpha_scale = float(alpha_scale)
        dims = [hidden_size, *hidden_dims, num_actions]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        final = nn.Linear(dims[-2], dims[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, router_input: torch.Tensor) -> torch.Tensor:
        return self.alpha_scale * torch.tanh(self.net(router_input))


def build_algorithmic_delta_router(
    *,
    mode: str,
    hidden_size: int,
    num_actions: int,
    hidden_dims: Sequence[int],
    dropout: float,
    alpha_scale: float,
) -> nn.Module:
    if mode == "scalar":
        return ScalarAlgorithmicDeltaRouter(num_actions=num_actions, alpha_scale=alpha_scale)
    if mode == "mlp":
        return MLPAlgorithmicDeltaRouter(
            hidden_size=hidden_size,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
            dropout=dropout,
            alpha_scale=alpha_scale,
        )
    raise ValueError(f"Unknown adapter_mode={mode!r}; choices=['scalar', 'mlp']")


class GPT2AlgorithmicDeltaAdapter(AdapterModel):
    """
    GPT-2 algorithmic attention-delta adapter.

    For each edited layer, the pre-output-projection attention value is changed as:

        z_new = z_soft + sum_a alpha_a(x) * (z_a - z_soft)

    Candidate learner outputs z_a are fixed, no-grad algorithmic directions. The
    trainable parameters are only the routers that produce alpha_a(x).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        cfg: LearnerHyperParams,
        layer_indices: Sequence[int],
        head_indices: str,
        candidate_learners: Sequence[str],
        adapter_mode: str,
        router_hidden_dims: Sequence[int],
        router_dropout: float,
        router_input: str,
        alpha_scale: float,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(layer_idx) for layer_idx in layer_indices)
        self.layer_set = set(self.layer_indices)
        self.candidate_learners = list(candidate_learners)
        self.router_input = router_input

        if self.router_input not in {"ln1", "residual"}:
            raise ValueError("router_input must be one of: ln1, residual")
        unknown = sorted(set(self.candidate_learners) - set(LEARNERS))
        if unknown:
            raise ValueError(f"Unknown candidate learners {unknown}; available={LEARNERS}")
        if not self.candidate_learners:
            raise ValueError("candidate_learners cannot be empty")

        for param in self.model.parameters():
            param.requires_grad_(False)

        hidden_size = int(model.config.n_embd)
        num_heads = int(model.config.n_head)
        head_dim = hidden_size // num_heads
        if hidden_size != num_heads * head_dim:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_indices = parse_head_indices(head_indices, num_heads)
        self.learner_instances = build_learners(self.candidate_learners)
        self.adapters = nn.ModuleDict(
            {
                str(layer_idx): build_algorithmic_delta_router(
                    mode=adapter_mode,
                    hidden_size=hidden_size,
                    num_actions=len(self.candidate_learners),
                    hidden_dims=router_hidden_dims,
                    dropout=router_dropout,
                    alpha_scale=alpha_scale,
                )
                for layer_idx in self.layer_indices
            }
        )
        for param in self.adapters.parameters():
            param.requires_grad_(True)

        self._block_inputs: Dict[int, torch.Tensor] = {}
        self._latest_alphas: Dict[int, torch.Tensor] = {}
        self._latest_deltas: Dict[int, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return next(self.adapters.parameters()).device

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        self.adapters.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()
        self.adapters.eval()

    def _make_block_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            self._block_inputs[layer_idx] = inputs[0]

        return hook

    @torch.no_grad()
    def _apply_learner_to_heads(
        self,
        *,
        learner: BaseAttentionLearner,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        bsz, group_size, seq_len, dim = q.shape
        dv = v.size(-1)
        outputs: list[torch.Tensor] = []
        for pos in range(seq_len):
            q_pos = q[:, :, pos, :].reshape(bsz * group_size, dim)
            k_ctx = k[:, :, : pos + 1, :].reshape(bsz * group_size, pos + 1, dim)
            v_ctx = v[:, :, : pos + 1, :].reshape(bsz * group_size, pos + 1, dv)
            pred = learner(q_pos.float(), k_ctx.float(), v_ctx.float(), self.cfg)
            outputs.append(pred.to(dtype=v.dtype).view(bsz, group_size, dv))
        return torch.stack(outputs, dim=2)

    @torch.no_grad()
    def compute_candidate_deltas(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        z_soft_heads: torch.Tensor,
    ) -> torch.Tensor:
        q_sel = q[:, self.head_indices, :, :].detach()
        k_sel = k[:, self.head_indices, :, :].detach()
        v_sel = v[:, self.head_indices, :, :].detach()
        z_sel = z_soft_heads[:, self.head_indices, :, :].detach()

        deltas = []
        for name in self.candidate_learners:
            pred = self._apply_learner_to_heads(
                learner=self.learner_instances[name],
                q=q_sel,
                k=k_sel,
                v=v_sel,
            )
            deltas.append((pred.float() - z_sel.float()).unsqueeze(1))
        return torch.cat(deltas, dim=1)

    def compute_delta_and_alpha(
        self,
        *,
        layer_idx: int,
        z_soft: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.model.transformer.h[layer_idx]
        residual = hidden_states
        with torch.no_grad():
            x_ln1 = block.ln_1(hidden_states)
            qkv = block.attn.c_attn(x_ln1)
            q_raw, k_raw, v_raw = qkv.split(block.attn.split_size, dim=2)
            q = split_heads(q_raw, self.num_heads, self.head_dim)
            k = split_heads(k_raw, self.num_heads, self.head_dim)
            v = split_heads(v_raw, self.num_heads, self.head_dim)
            z_soft_heads = split_heads(z_soft.detach(), self.num_heads, self.head_dim)
            deltas = self.compute_candidate_deltas(q=q, k=k, v=v, z_soft_heads=z_soft_heads)

        if self.router_input == "ln1":
            router_input = x_ln1.detach()
        elif self.router_input == "residual":
            router_input = residual.detach()
        else:
            raise ValueError(f"Unknown router_input={self.router_input!r}")

        alpha = self.adapters[str(layer_idx)](router_input).float()
        weighted_delta = (
            alpha.permute(0, 2, 1).unsqueeze(2).unsqueeze(-1) * deltas
        ).sum(dim=1)

        delta_heads = torch.zeros_like(z_soft_heads)
        for local_head, head_idx in enumerate(self.head_indices):
            delta_heads[:, head_idx, :, :] = weighted_delta[:, local_head, :, :]
        return merge_heads(delta_heads).to(dtype=z_soft.dtype), alpha

    def _make_c_proj_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if layer_idx not in self._block_inputs:
                raise RuntimeError(
                    f"Missing cached GPT-2 block input for layer {layer_idx}. "
                    "The block pre-hook did not fire before attn.c_proj."
                )
            z_soft = inputs[0]
            delta, alpha = self.compute_delta_and_alpha(
                layer_idx=layer_idx,
                z_soft=z_soft,
                hidden_states=self._block_inputs[layer_idx],
            )
            self._latest_deltas[layer_idx] = delta.detach()
            self._latest_alphas[layer_idx] = alpha.detach()
            return (z_soft + delta,) + inputs[1:]

        return hook

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        self._block_inputs.clear()
        self._latest_alphas.clear()
        self._latest_deltas.clear()
        handles: list[torch.utils.hooks.RemovableHandle] = []
        try:
            for layer_idx in self.layer_indices:
                block = self.model.transformer.h[layer_idx]
                handles.append(block.register_forward_pre_hook(self._make_block_pre_hook(layer_idx)))
                handles.append(block.attn.c_proj.register_forward_pre_hook(self._make_c_proj_pre_hook(layer_idx)))
            return self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
            self._block_inputs.clear()

    def _alpha_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        all_abs: list[torch.Tensor] = []
        for layer_idx, alpha in self._latest_alphas.items():
            alpha_f = alpha.float()
            all_abs.append(alpha_f.abs().mean())
            stats[f"layer_{layer_idx}_alpha_abs_mean"] = float(alpha_f.abs().mean().item())
            stats[f"layer_{layer_idx}_alpha_abs_max"] = float(alpha_f.abs().max().item())
            for action_idx, name in enumerate(self.candidate_learners):
                stats[f"layer_{layer_idx}_alpha_{name}_mean"] = float(alpha_f[..., action_idx].mean().item())
                stats[f"layer_{layer_idx}_alpha_{name}_abs_mean"] = float(
                    alpha_f[..., action_idx].abs().mean().item()
                )
        stats["alpha_abs_mean"] = float(torch.stack(all_abs).mean().item()) if all_abs else 0.0
        return stats

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}
        self.set_peft_eval_mode()
        _ = self(input_ids)
        stats = self._alpha_stats()
        stats.update(delta_stats(self._latest_deltas))
        return stats


__all__ = [
    "GPT2AlgorithmicDeltaAdapter",
    "MLPAlgorithmicDeltaRouter",
    "ScalarAlgorithmicDeltaRouter",
    "parse_csv",
    "parse_head_indices",
    "parse_int_csv",
]
