from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

from experiments.router_development.attention_adapter.adapters.utils import delta_stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig


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
        self._adapter_inputs: Dict[int, torch.Tensor] = {}
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

    def adapter_input_for_block(self, *, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        # The adapter is conditioned on frozen pre-attention features x = LN1(h).
        # Gradients update only the bottleneck delta map, not the base transformer path.
        return block.ln_1(hidden_states).detach()

    def compute_delta(self, *, layer_idx: int, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        adapter_input = self.adapter_input_for_block(block=block, hidden_states=hidden_states)
        return self.adapters[str(layer_idx)](adapter_input)

    def _make_block_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            self._adapter_inputs[layer_idx] = inputs[0]

        return hook

    def _make_c_proj_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if layer_idx not in self._adapter_inputs:
                raise RuntimeError(
                    f"Missing cached GPT-2 block input for layer {layer_idx}. "
                    "The block pre-hook did not fire before attn.c_proj."
                )
            z = inputs[0]
            block = self.model.transformer.h[layer_idx]
            delta = self.compute_delta(
                layer_idx=layer_idx,
                block=block,
                hidden_states=self._adapter_inputs[layer_idx],
            ).to(dtype=z.dtype, device=z.device)
            self._latest_deltas[layer_idx] = delta.detach()
            # GPT-2 c_proj receives the merged pre-output-projection attention value z.
            return (z + delta,) + inputs[1:]

        return hook

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        self._adapter_inputs.clear()
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
            self._adapter_inputs.clear()

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}
        self.set_peft_eval_mode()
        _ = self(input_ids)
        return delta_stats(self._latest_deltas)



class PythiaAKAZAAdapter(AdapterModel):
    """Pythia/GPT-NeoX AKAZA intervention using pre-hooks on attention.dense."""

    def __init__(self, *, model: nn.Module, cfg: AdapterFineTuneConfig, layer_indices: Sequence[int]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)
        self.layer_set = set(self.layer_indices)
        self._adapter_inputs: Dict[int, torch.Tensor] = {}
        self._latest_deltas: Dict[int, torch.Tensor] = {}

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

    @property
    def device(self) -> torch.device:
        return next(self.adapters.parameters()).device

    def remove_hooks(self) -> None:
        """Compatibility no-op: hooks are now scoped to each forward call."""
        return None

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        self.adapters.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()
        self.adapters.eval()


    def _make_dense_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if layer_idx not in self._adapter_inputs:
                raise RuntimeError(
                    f"Missing cached Pythia layer input for layer {layer_idx}. "
                    "The layer pre-hook did not fire before attention.dense."
                )
            z = inputs[0]
            layer = self.model.gpt_neox.layers[layer_idx]
            adapter_input = self._adapter_input_for_layer(layer_idx=layer_idx, layer=layer)
            delta = self.adapters[str(layer_idx)](adapter_input).to(z.dtype)
            self._latest_deltas[layer_idx] = delta.detach()
            # Pythia exposes the same pre-output-projection attention value as
            # attention.dense input, so the hook implements z -> z + Delta(LN1(h)).
            return (z + delta,) + inputs[1:]

        return hook
    

    def _capture_layer_input(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        self._adapter_inputs[layer_idx] = hidden_states

    def _adapter_input_for_layer(self, *, layer_idx: int, layer: nn.Module) -> torch.Tensor:
        hidden_states = self._adapter_inputs[layer_idx]
        # Match the GPT-2 AKAZA conditioning: x = input_layernorm(h), detached.
        return layer.input_layernorm(hidden_states).detach()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        self._adapter_inputs.clear()
        self._latest_deltas.clear()
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_layer_pre_hook(layer_idx: int):
            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                self._capture_layer_input(layer_idx, inputs[0])

            return hook

        for layer_idx in self.layer_indices:
            layer = self.model.gpt_neox.layers[layer_idx]
            handles.append(layer.register_forward_pre_hook(make_layer_pre_hook(layer_idx)))
            handles.append(layer.attention.dense.register_forward_pre_hook(self._make_dense_pre_hook(layer_idx)))
        try:
            return self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
            self._adapter_inputs.clear()

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

