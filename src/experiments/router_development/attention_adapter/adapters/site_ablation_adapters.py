from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from experiments.router_development.attention_adapter.adapters.akaza_adapters import BottleneckDeltaAdapter
from experiments.router_development.attention_adapter.adapters.base import AdapterModel
from experiments.router_development.attention_adapter.adapters.utils import delta_stats
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import create_causal_mask


INTERVENTION_SITES = (
    "z_pre_cproj",
    "attn_post_cproj",
    "residual_pre_block",
    "residual_post_block",
    "mlp_post",
)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    bsz, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    return x.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, num_heads * head_dim)


def block_forward_hidden(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    causal_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if causal_mask is None:
        out = block(hidden_states, use_cache=False, position_ids=position_ids)
    else:
        out = block(
            hidden_states,
            None,
            causal_mask,
            None,
            use_cache=False,
            position_ids=position_ids,
        )
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "hidden_states"):
        return out.hidden_states
    raise TypeError(f"Unexpected GPT-2 block output type: {type(out)}")


def causal_soft_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, _, seq_len, head_dim = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(head_dim))
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class GPT2SiteAblationAdapter(AdapterModel):
    """GPT-2 matched bottleneck adapters at alternative Transformer intervention sites."""

    def __init__(self, *, model: nn.Module, cfg: Any, layer_indices: Sequence[int]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.layer_indices = sorted(int(x) for x in layer_indices)
        self.layer_set = set(self.layer_indices)
        self.intervention_site = cfg.intervention_site
        self._latest_deltas: Dict[int, torch.Tensor] = {}

        if self.intervention_site not in INTERVENTION_SITES:
            raise ValueError(
                f"Unknown intervention_site={self.intervention_site!r}; "
                f"choices={list(INTERVENTION_SITES)}"
            )

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
        if self.cfg.adapter_input == "residual":
            adapter_input = hidden_states
        elif self.cfg.adapter_input == "ln1":
            adapter_input = block.ln_1(hidden_states)
        else:
            raise ValueError(f"Unknown adapter_input={self.cfg.adapter_input!r}")
        return adapter_input.detach() if self.cfg.detach_adapter_input else adapter_input

    def compute_delta(self, *, layer_idx: int, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        adapter_input = self.adapter_input_for_block(block=block, hidden_states=hidden_states)
        return self.adapters[str(layer_idx)](adapter_input)

    def attention_parts(self, *, block: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = block.attn
        x_ln1 = block.ln_1(hidden_states)
        qkv = attn.c_attn(x_ln1)
        q_raw, k_raw, v_raw = qkv.split(attn.split_size, dim=2)
        num_heads = attn.num_heads
        q = split_heads(q_raw, num_heads).float()
        k = split_heads(k_raw, num_heads).float()
        v = split_heads(v_raw, num_heads).float()
        z_soft_heads = causal_soft_attention(q, k, v).float()
        return hidden_states, merge_heads(z_soft_heads).to(x_ln1.dtype)

    def run_normal_block_with_mlp_delta(
        self,
        *,
        block: nn.Module,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        x_ln1 = block.ln_1(hidden_states)
        attn_outputs = block.attn(
            x_ln1,
            None,
            causal_mask,
            use_cache=False,
            position_ids=position_ids,
        )
        attn_output = attn_outputs[0] if isinstance(attn_outputs, (tuple, list)) else attn_outputs
        hidden_states = residual + attn_output

        residual = hidden_states
        x_ln2 = block.ln_2(hidden_states)
        feed_forward_hidden_states = block.mlp(x_ln2)
        feed_forward_hidden_states = feed_forward_hidden_states + delta.to(feed_forward_hidden_states.dtype)
        return residual + feed_forward_hidden_states

    def forward_edited_block(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.model.transformer.h[layer_idx]
        site = self.intervention_site

        if site == "residual_pre_block":
            delta = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)
            hidden_states = hidden_states + delta.to(hidden_states.dtype)
            return block_forward_hidden(
                block,
                hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
            ), delta

        if site == "residual_post_block":
            delta = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)
            hidden_states = block_forward_hidden(
                block,
                hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
            )
            return hidden_states + delta.to(hidden_states.dtype), delta

        if site == "mlp_post":
            delta = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)
            return self.run_normal_block_with_mlp_delta(
                block=block,
                hidden_states=hidden_states,
                delta=delta,
                causal_mask=causal_mask,
                position_ids=position_ids,
            ), delta

        residual, z_soft = self.attention_parts(block=block, hidden_states=hidden_states)
        delta = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)
        attn = block.attn

        if site == "z_pre_cproj":
            attn_output = attn.c_proj(z_soft + delta.to(z_soft.dtype))
            attn_output = attn.resid_dropout(attn_output)
        elif site == "attn_post_cproj":
            attn_output = attn.c_proj(z_soft)
            attn_output = attn_output + delta.to(attn_output.dtype)
            attn_output = attn.resid_dropout(attn_output)
        else:
            raise ValueError(f"Unhandled intervention_site={site!r}")

        hidden_states = residual + attn_output
        residual = hidden_states
        x_ln2 = block.ln_2(hidden_states)
        feed_forward_hidden_states = block.mlp(x_ln2)
        return residual + feed_forward_hidden_states, delta

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        transformer = self.model.transformer
        input_ids = input_ids.to(self.device)
        self._latest_deltas.clear()

        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        inputs_embeds = transformer.wte(input_ids)
        causal_mask = create_causal_mask(
            config=transformer.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )

        first_edit = min(self.layer_indices)
        with torch.no_grad():
            hidden_states = inputs_embeds + transformer.wpe(position_ids).to(inputs_embeds.device)
            hidden_states = transformer.drop(hidden_states)
            for i in range(first_edit):
                hidden_states = block_forward_hidden(
                    transformer.h[i],
                    hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                )

        for i in range(first_edit, len(transformer.h)):
            if i in self.layer_set:
                hidden_states, delta = self.forward_edited_block(
                    hidden_states,
                    i,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                )
                self._latest_deltas[i] = delta.detach()
            else:
                hidden_states = block_forward_hidden(
                    transformer.h[i],
                    hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                )

        hidden_states = transformer.ln_f(hidden_states)
        return self.model.lm_head(hidden_states)

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        if input_ids is None:
            return {}
        self.set_peft_eval_mode()
        _ = self(input_ids)
        return delta_stats(self._latest_deltas)
