from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig
from experiments.router_development.attention_adapter.utils import (
    block_forward_hidden,
    split_heads,
)


class AdapterModel(nn.Module, ABC):
    """Common interface used by the training loop for custom and official PEFT adapters."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def set_peft_train_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_peft_eval_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        raise NotImplementedError


class BottleneckDeltaAdapter(nn.Module):
    """AKAZA/FreeZ bottleneck delta adapter with exact no-op initialization."""

    def __init__(self, hidden_size: int, bottleneck_dim: int, dropout: float, output_scale: float):
        super().__init__()
        self.output_scale = float(output_scale)
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, adapter_input: torch.Tensor) -> torch.Tensor:
        adapter_dtype = self.down.weight.dtype
        if adapter_input.dtype != adapter_dtype:
            adapter_input = adapter_input.to(adapter_dtype)

        h = self.down(adapter_input)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.output_scale * torch.tanh(self.up(h))

class GPT2AKAZAAdapter(AdapterModel):
    """GPT-2 pre-c_proj z-space AKAZA intervention."""

    def __init__(self, *, model: nn.Module, cfg: AdapterFineTuneConfig, layer_indices: Sequence[int]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)
        self.layer_set = set(self.layer_indices)
        for p in self.model.parameters():
            p.requires_grad_(False)
        hidden_size = int(model.config.n_embd)
        self.adapters = nn.ModuleDict(
            {
                str(layer_idx): BottleneckDeltaAdapter(
                    hidden_size=hidden_size,
                    bottleneck_dim=cfg.bottleneck_dim,
                    dropout=cfg.adapter_dropout,
                    output_scale=cfg.output_scale,
                )
                for layer_idx in self.layer_indices
            }
        )
        for p in self.adapters.parameters():
            p.requires_grad_(True)

    @property
    def device(self) -> torch.device:
        return next(self.adapters.parameters()).device

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        self.adapters.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()
        self.adapters.eval()

    def adapter_input_for_block(self, *, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        # The adapter is conditioned on frozen pre-attention features x = LN1(h).
        # Gradients update only the bottleneck delta map, not the base transformer path.
        return block.ln_1(hidden_states).detach()

    def compute_delta(self, *, layer_idx: int, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        adapter_input = self.adapter_input_for_block(block=block, hidden_states=hidden_states)
        return self.adapters[str(layer_idx)](adapter_input)

    def attention_parts(self, *, block: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = block.attn
        residual = hidden_states
        x_ln1 = block.ln_1(hidden_states)
        qkv = attn.c_attn(x_ln1)
        q_raw, k_raw, v_raw = qkv.split(attn.split_size, dim=2)
        q = split_heads(q_raw, attn.num_heads).float()
        k = split_heads(k_raw, attn.num_heads).float()
        v = split_heads(v_raw, attn.num_heads).float()

        # Recompute causal attention z = softmax(QK^T / sqrt(d), causal) V in
        # pre-c_proj space, then merge heads back to [batch, seq, hidden].
        seq_len = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(q.shape[-1]))
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        z_heads = torch.matmul(torch.softmax(scores, dim=-1), v).float()
    
        bsz, num_heads, _, head_dim = z_heads.shape
        z_soft = z_heads.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, num_heads * head_dim).to(x_ln1.dtype)

        return residual, z_soft

    def forward_edited_block(self, hidden_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.model.transformer.h[layer_idx]
        residual, z_soft = self.attention_parts(block=block, hidden_states=hidden_states)

        delta = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)
        attn = block.attn

        # AKAZA edits the attention value before c_proj: h' = h + c_proj(z + Delta(x)).
        hidden_states = residual + attn.resid_dropout(attn.c_proj(z_soft + delta.to(z_soft.dtype)))
        residual = hidden_states
        hidden_states = residual + block.mlp(block.ln_2(hidden_states))
        return hidden_states, delta

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        transformer = self.model.transformer
        input_ids = input_ids.to(self.device)
        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        first_edit = min(self.layer_indices)
        with torch.no_grad():
            hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
            hidden_states = transformer.drop(hidden_states)
            for i in range(first_edit):
                hidden_states = block_forward_hidden(transformer.h[i], hidden_states)
        for i in range(first_edit, len(transformer.h)):
            if i in self.layer_set:
                hidden_states, _ = self.forward_edited_block(hidden_states, i)
            else:
                hidden_states = block_forward_hidden(transformer.h[i], hidden_states)
        return self.model.lm_head(transformer.ln_f(hidden_states))


    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}
        self.set_peft_eval_mode()
        transformer = self.model.transformer
        input_ids = input_ids.to(self.device)
        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        first_edit = min(self.layer_indices)
        hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
        hidden_states = transformer.drop(hidden_states)
        for i in range(first_edit):
            hidden_states = block_forward_hidden(transformer.h[i], hidden_states)
        deltas: Dict[int, torch.Tensor] = {}
        for i in range(first_edit, len(transformer.h)):
            if i in self.layer_set:
                hidden_states, delta = self.forward_edited_block(hidden_states, i)
                deltas[i] = delta
            else:
                hidden_states = block_forward_hidden(transformer.h[i], hidden_states)
        return delta_stats(deltas)



class PythiaAKAZAAdapter(AdapterModel):
    """Pythia/GPT-NeoX AKAZA intervention using pre-hooks on attention.dense."""

    def __init__(self, *, model: nn.Module, cfg: AdapterFineTuneConfig, layer_indices: Sequence[int]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)
        self.layer_set = set(self.layer_indices)
        self._current_layer_idx: int | None = None
        self._latest_deltas: Dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for p in self.model.parameters():
            p.requires_grad_(False)
        hidden_size = int(model.config.hidden_size)
        self.adapters = nn.ModuleDict(
            {
                str(layer_idx): BottleneckDeltaAdapter(
                    hidden_size=hidden_size,
                    bottleneck_dim=cfg.bottleneck_dim,
                    dropout=cfg.adapter_dropout,
                    output_scale=cfg.output_scale,
                )
                for layer_idx in self.layer_indices
            }
        )
        for layer_idx in self.layer_indices:
            dense = self.model.gpt_neox.layers[layer_idx].attention.dense
            self._handles.append(dense.register_forward_pre_hook(self._make_dense_pre_hook(layer_idx)))


    @property
    def device(self) -> torch.device:
        return next(self.adapters.parameters()).device

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        self.adapters.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()
        self.adapters.eval()


    def _make_dense_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            z = inputs[0]
            layer = self.model.gpt_neox.layers[layer_idx]
            adapter_input = self._adapter_input_for_layer(layer)
            delta = self.adapters[str(layer_idx)](adapter_input).to(z.dtype)
            self._latest_deltas[layer_idx] = delta.detach()
            # Pythia exposes the same pre-output-projection attention value as
            # attention.dense input, so the hook implements z -> z + Delta(LN1(h)).
            return (z + delta,) + inputs[1:]

        return hook
    

    def _capture_layer_input(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        self._current_layer_idx = layer_idx
        self._current_hidden_states = hidden_states

    def _adapter_input_for_layer(self, layer: nn.Module) -> torch.Tensor:
        hidden_states = self._current_hidden_states
        # Match the GPT-2 AKAZA conditioning: x = input_layernorm(h), detached.
        return layer.input_layernorm(hidden_states).detach()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        self._latest_deltas = {}
        handles = []

        def make_layer_pre_hook(layer_idx: int):
            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                self._capture_layer_input(layer_idx, inputs[0])

            return hook

        for layer_idx in self.layer_indices:
            handles.append(self.model.gpt_neox.layers[layer_idx].register_forward_pre_hook(make_layer_pre_hook(layer_idx)))
        try:
            return self.model(input_ids).logits
        finally:
            for handle in handles:
                handle.remove()
            self._current_layer_idx = None

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}
        self.set_peft_eval_mode()
        _ = self(input_ids)
        return delta_stats(self._latest_deltas)



class OfficialPEFTAdapter(AdapterModel):
    """Interface adapter for Hugging Face PEFT models."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids.to(self.device)).logits

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        for name, module in self.model.named_modules():
            if "lora_dropout" in name:
                module.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        del input_ids
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return {}
        vec = torch.cat([p.detach().reshape(-1).float().cpu() for p in params])
        return {
            "peft_param_abs_mean": float(vec.abs().mean().item()),
            "peft_param_l2_rms": float(vec.pow(2).mean().sqrt().item()),
        }


def delta_stats(deltas: Dict[int, torch.Tensor]) -> Dict[str, float]:
    if not deltas:
        return {}
    flat = torch.cat([delta.detach().reshape(-1).float().cpu() for delta in deltas.values()])
    return {
        "delta_abs_mean": float(flat.abs().mean().item()),
        "delta_l2_rms": float(flat.pow(2).mean().sqrt().item()),
    }
