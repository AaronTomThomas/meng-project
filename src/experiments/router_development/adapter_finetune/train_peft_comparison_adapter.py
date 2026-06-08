from __future__ import annotations

"""
Compare AKAZA/FreeZ against official PEFT baselines on frozen GPT-2.

This script keeps the same experimental protocol as your site-ablation script:
  - GPT-2 base weights are frozen.
  - Train / validation / test splits are loaded separately.
  - Validation selects the checkpoint.
  - Test is evaluated only after best-val checkpoint selection unless
    --eval_test_during_training is explicitly enabled.
  - All methods should be no-op at initialization.

Implemented methods:
  1. akaza_freez
     Your custom pre-projection attention-output correction:

         z_new = z_soft + delta_z(x)
         attn_out = attn.c_proj(z_new)

  2. lora
     Official Hugging Face PEFT LoRA via get_peft_model + LoraConfig.

  3. ia3
     Official Hugging Face PEFT IA3 via get_peft_model + IA3Config.

  4. mlp_attention
     Frozen GPT-2 where selected layers keep attn.c_attn and attn.c_proj, but
     replace the soft-attention computation Q,K,V -> z with a trainable MLP:

         q,k,v = attn.c_attn(ln1(h))
         z_new = MLP(concat(q, k, v))
         attn_out = attn.c_proj(z_new)

Why AKAZA is custom:
  PEFT LoRA/IA3 modify existing module weights or activations. Your method edits
  the intermediate pre-c_proj attention tensor z, which is not an exposed module
  boundary in GPT-2. So the AKAZA/FreeZ path remains custom, while PEFT baselines
  use official external implementations.

Install dependency for PEFT baselines:

    uv add peft

Example AKAZA/FreeZ run:

PYTHONPATH=src uv run python src/experiments/router_development/adapter_finetune/train_peft_comparison_adapter.py \
  --method akaza_freez \
  --model_name openai-community/gpt2 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --val_split validation \
  --test_split test \
  --text_field text \
  --max_train_texts 1000 \
  --max_val_texts 200 \
  --max_test_texts 200 \
  --block_size 96 \
  --batch_size 4 \
  --max_train_chunks 2048 \
  --max_val_chunks 512 \
  --max_test_chunks 512 \
  --layer_indices 6,7,8,9,10,11 \
  --bottleneck_dim 4 \
  --adapter_dropout 0.05 \
  --adapter_input ln1 \
  --detach_adapter_input \
  --output_scale 0.05 \
  --peft_l2 1e-5 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --epochs 500 \
  --patience 30 \
  --device cuda \
  --output_path outputs/adapter_finetune/peft_compare_akaza_freez.pt

Example LoRA run, roughly parameter-matched to 41,496 AKAZA params:

PYTHONPATH=src uv run python src/experiments/router_development/adapter_finetune/train_peft_comparison_adapter.py \
  --method lora \
  --peft_target_profile attn_c_proj \
  --lora_rank 4 \
  --lora_alpha 4 \
  --lora_dropout 0.05 \
  --model_name openai-community/gpt2 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --val_split validation \
  --test_split test \
  --text_field text \
  --max_train_texts 1000 \
  --max_val_texts 200 \
  --max_test_texts 200 \
  --block_size 96 \
  --batch_size 4 \
  --max_train_chunks 2048 \
  --max_val_chunks 512 \
  --max_test_chunks 512 \
  --layer_indices 6,7,8,9,10,11 \
  --peft_l2 1e-5 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --epochs 500 \
  --patience 30 \
  --device cuda \
  --output_path outputs/adapter_finetune/peft_compare_lora_attn_cproj_r4.pt

Useful LoRA parameter budgets on GPT-2 small, layers 6-11:
  - --peft_target_profile attn_c_proj --lora_rank 4  => 36,864 trainable params
  - --peft_target_profile attn_c_proj --lora_rank 5  => 46,080 trainable params
  - --peft_target_profile attn_c_attn --lora_rank 2  => 36,864 trainable params
  - --peft_target_profile attn_c_attn --lora_rank 3  => 55,296 trainable params
  - --peft_target_profile attn_both   --lora_rank 1  => 27,648 trainable params
  - --peft_target_profile attn_both   --lora_rank 2  => 55,296 trainable params
"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.attention_learners import LearnerHyperParams
from experiments.gpt2_probe_utils import load_and_pack_texts


METHODS = ["akaza_freez", "lora", "ia3", "mlp_attention"]

# These are GPT-2 module suffixes under model.transformer.h.<layer>.
LORA_TARGET_PROFILES: Dict[str, List[str]] = {
    "attn_c_attn": ["attn.c_attn"],
    "attn_c_proj": ["attn.c_proj"],
    "attn_both": ["attn.c_attn", "attn.c_proj"],
    "mlp": ["mlp.c_fc", "mlp.c_proj"],
    "block_all": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
}

# GPT-2 has packed qkv in attn.c_attn, so IA3 over attention is not a pure k/v-only
# IA3 intervention. It is still a useful official PEFT baseline.
IA3_TARGET_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "ia3_standard": {
        "target_modules": ["attn.c_attn", "mlp.c_fc"],
        "feedforward_modules": ["mlp.c_fc"],
    },
    "attn_c_attn": {
        "target_modules": ["attn.c_attn"],
        "feedforward_modules": [],
    },
    "mlp_c_fc": {
        "target_modules": ["mlp.c_fc"],
        "feedforward_modules": ["mlp.c_fc"],
    },
}

ALL_TARGET_PROFILES = sorted(set(LORA_TARGET_PROFILES) | set(IA3_TARGET_PROFILES))


@dataclass
class PEFTComparisonConfig(LearnerHyperParams):
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    text_field: str = "text"

    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

    max_train_texts: int = 1000
    max_val_texts: int = 200
    max_test_texts: int = 200

    block_size: int = 96
    batch_size: int = 4

    max_train_chunks: int = 2048
    max_val_chunks: int = 512
    max_test_chunks: int = 512

    layer_indices: str = "6,7,8,9,10,11"
    method: str = "akaza_freez"

    # AKAZA / FreeZ options.
    bottleneck_dim: int = 4
    adapter_dropout: float = 0.05
    adapter_input: str = "ln1"  # ln1 | residual
    detach_adapter_input: bool = True
    output_scale: float = 0.05

    # Full attention replacement control.
    mlp_attention_hidden_dim: int = 768
    mlp_attention_depth: int = 2
    mlp_attention_dropout: float = 0.05
    mlp_attention_zero_init: bool = True

    # Official PEFT options.
    peft_target_profile: str = "attn_c_proj"

    # LoRA options.
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.05
    lora_bias: str = "none"  # none | all | lora_only

    # IA3 options.
    ia3_init_weights: bool = True

    # Regularization over trainable PEFT/custom adapter params.
    peft_l2: float = 1e-5
    peft_l1: float = 0.0

    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 500
    patience: int = 30
    eval_every: int = 1
    log_every_steps: int = 10
    grad_clip: float = 1.0

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: str = "outputs/adapter_finetune/peft_comparison.pt"
    cache_dir: str = "outputs/adapter_finetune/cache/"
    skip_freeze_check: bool = False
    eval_test_during_training: bool = False

    # Compatibility fields for load_and_pack_texts.
    split: str = "train"
    max_texts: int = 1000
    max_chunks: int = 2048


def parse_csv(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x) for x in parse_csv(value)]


def batch_slices(n: int, batch_size: int) -> Sequence[slice]:
    return [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # [B, T, C] -> [B, H, T, D]
    bsz, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    return x.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    # [B, H, T, D] -> [B, T, C]
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, num_heads * head_dim)


def block_forward_hidden(block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    out = block(hidden_states, use_cache=False)

    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        return out[0]

    if hasattr(out, "hidden_states"):
        return out.hidden_states

    raise TypeError(f"Unexpected GPT-2 block output type: {type(out)}")


def causal_soft_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q/k/v: [B, H, T, D]
    _, _, seq_len, head_dim = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(head_dim))

    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(
        ~mask.view(1, 1, seq_len, seq_len),
        torch.finfo(scores.dtype).min,
    )

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
    )


def ppl_from_nll(nll: float) -> float:
    return float(math.exp(nll))


def relative_ppl_reduction(delta_nll: float) -> float:
    # delta_nll = baseline_nll - adapted_nll
    return float(1.0 - math.exp(-delta_nll))


def target_module_names(layer_indices: Sequence[int], suffixes: Sequence[str]) -> List[str]:
    return [f"transformer.h.{layer_idx}.{suffix}" for layer_idx in layer_indices for suffix in suffixes]


def trainable_named_parameters(module: nn.Module) -> List[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in module.named_parameters() if p.requires_grad]


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for _, p in trainable_named_parameters(module)]


def trainable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in trainable_named_parameters(module)}


@torch.no_grad()
def load_trainable_state_dict(module: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    named = dict(module.named_parameters())
    missing = [name for name in state if name not in named]
    if missing:
        raise KeyError(f"State contains unknown parameter names: {missing[:10]}")

    for name, value in state.items():
        named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def peft_regularizer(module: nn.Module, l2: float, l1: float) -> torch.Tensor | None:
    params = trainable_parameters(module)
    if not params or (l2 <= 0 and l1 <= 0):
        return None

    vec = torch.cat([p.reshape(-1) for p in params])
    reg = torch.zeros((), device=vec.device, dtype=vec.dtype)
    if l2 > 0:
        reg = reg + l2 * vec.pow(2).mean()
    if l1 > 0:
        reg = reg + l1 * vec.abs().mean()
    return reg


@torch.no_grad()
def clone_frozen_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters()
        if not p.requires_grad
    }


@torch.no_grad()
def assert_frozen_unchanged(module: nn.Module, before: Dict[str, torch.Tensor]) -> None:
    changed = []
    for name, p in module.named_parameters():
        if name not in before:
            continue
        after = p.detach().cpu()
        if not torch.equal(before[name], after):
            max_diff = (before[name] - after).abs().max().item()
            changed.append((name, max_diff))

    assert not changed, f"Frozen parameters changed: {changed[:10]}"
    print("[freeze_check] OK: frozen parameters unchanged after optimizer step.")


def assert_trainable_scope(wrapped: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    trainable = trainable_named_parameters(wrapped)
    trainable_count = sum(p.numel() for _, p in trainable)

    print()
    print(f"[freeze_check] trainable params: {trainable_count}")
    print(f"[freeze_check] trainable parameter tensors: {len(trainable)}")
    for name, p in trainable[:20]:
        print(f"  trainable: {name} shape={tuple(p.shape)} numel={p.numel()}")
    if len(trainable) > 20:
        print(f"  ... {len(trainable) - 20} more trainable tensors")

    assert trainable, "No trainable parameters found."

    trainable_ids = {id(p) for _, p in trainable}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert optimizer_ids == trainable_ids, (
        "Optimizer params do not exactly equal trainable params: "
        f"missing={len(trainable_ids - optimizer_ids)} extra={len(optimizer_ids - trainable_ids)}"
    )
    print("[freeze_check] OK: optimizer contains exactly trainable params.")


def assert_no_frozen_grads(wrapped: nn.Module) -> None:
    frozen_with_grads = [
        name
        for name, p in wrapped.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]
    trainable_with_grads = [
        name
        for name, p in wrapped.named_parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert not frozen_with_grads, f"Frozen params received grads: {frozen_with_grads[:10]}"
    assert trainable_with_grads, "No trainable params received gradients."
    print("[freeze_check] OK: gradients only exist on trainable params.")


class BottleneckDeltaAdapter(nn.Module):
    """
    AKAZA / FreeZ bottleneck z-adapter.

        delta = output_scale * tanh(W_up(dropout(GELU(W_down(x)))))

    W_up is zero-initialized, so this starts as an exact no-op.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int,
        dropout: float,
        output_scale: float,
    ):
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
        h = self.down(adapter_input)
        h = F.gelu(h)
        h = self.dropout(h)
        raw = self.up(h)
        return self.output_scale * torch.tanh(raw)


class AKAZAFreeZModel(nn.Module):
    """
    Custom pre-c_proj z-space intervention.

    This remains custom because official PEFT modules do not expose GPT-2's
    intermediate pre-projection attention output z as a tunable module boundary.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        cfg: PEFTComparisonConfig,
        layer_indices: Sequence[int],
    ):
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
        if self.cfg.adapter_input == "residual":
            adapter_input = hidden_states
        elif self.cfg.adapter_input == "ln1":
            adapter_input = block.ln_1(hidden_states)
        else:
            raise ValueError(f"Unknown adapter_input={self.cfg.adapter_input!r}")

        if self.cfg.detach_adapter_input:
            adapter_input = adapter_input.detach()
        return adapter_input

    def compute_delta(self, *, layer_idx: int, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        adapter_input = self.adapter_input_for_block(block=block, hidden_states=hidden_states)
        return self.adapters[str(layer_idx)](adapter_input)

    def attention_parts(self, *, block: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = block.attn
        residual = hidden_states
        x_ln1 = block.ln_1(hidden_states)

        qkv = attn.c_attn(x_ln1)
        q_raw, k_raw, v_raw = qkv.split(attn.split_size, dim=2)

        num_heads = attn.num_heads
        q = split_heads(q_raw, num_heads).float()
        k = split_heads(k_raw, num_heads).float()
        v = split_heads(v_raw, num_heads).float()

        z_soft_heads = causal_soft_attention(q, k, v).float()
        z_soft = merge_heads(z_soft_heads).to(x_ln1.dtype)
        return residual, z_soft

    def forward_edited_block(self, hidden_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.model.transformer.h[layer_idx]
        residual, z_soft = self.attention_parts(block=block, hidden_states=hidden_states)
        delta = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)

        attn = block.attn
        z_new = z_soft + delta.to(z_soft.dtype)
        attn_output = attn.c_proj(z_new)
        attn_output = attn.resid_dropout(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        x_ln2 = block.ln_2(hidden_states)
        feed_forward_hidden_states = block.mlp(x_ln2)
        hidden_states = residual + feed_forward_hidden_states

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

        hidden_states = transformer.ln_f(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits

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

        stats: Dict[str, float] = {}
        all_abs: List[torch.Tensor] = []
        all_l2: List[torch.Tensor] = []
        all_max: List[torch.Tensor] = []

        for i in range(first_edit, len(transformer.h)):
            if i in self.layer_set:
                hidden_states, delta = self.forward_edited_block(hidden_states, i)
                abs_mean = delta.abs().mean()
                l2_rms = delta.pow(2).mean().sqrt()
                abs_max = delta.abs().max()

                all_abs.append(abs_mean)
                all_l2.append(l2_rms)
                all_max.append(abs_max)

                stats[f"layer_{i}_delta_abs_mean"] = float(abs_mean.item())
                stats[f"layer_{i}_delta_abs_max"] = float(abs_max.item())
                stats[f"layer_{i}_delta_l2_rms"] = float(l2_rms.item())
            else:
                hidden_states = block_forward_hidden(transformer.h[i], hidden_states)

        if all_abs:
            stats["delta_abs_mean"] = float(torch.stack(all_abs).mean().item())
            stats["delta_abs_max"] = float(torch.stack(all_max).max().item())
            stats["delta_l2_rms"] = float(torch.stack(all_l2).mean().item())
        return stats


class QKVToZMLP(nn.Module):
    """
    Token-wise MLP replacement for GPT-2's soft attention map.

    The input is concat(q, k, v) after GPT-2's packed c_attn projection, with
    heads merged back to [B, T, C]. The output is the replacement pre-c_proj
    attention value z.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        mlp_hidden_dim: int,
        depth: int,
        dropout: float,
        zero_init_output: bool,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("mlp_attention_depth must be >= 1")

        layers: List[nn.Module] = []
        in_dim = 3 * hidden_size
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, hidden_size))
        self.net = nn.Sequential(*layers)

        linear_layers = [module for module in self.net if isinstance(module, nn.Linear)]
        for module in linear_layers:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)

        if zero_init_output:
            nn.init.zeros_(linear_layers[-1].weight)
            nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q/k/v are [B, H, T, D]. Merge heads before the token-wise MLP.
        x = torch.cat([merge_heads(q), merge_heads(k), merge_heads(v)], dim=-1)
        return self.net(x)


class GPT2MLPAttentionReplacementModel(nn.Module):
    """
    Frozen GPT-2 with selected layers' soft attention replaced by QKV -> z MLPs.

    The GPT-2 q/k/v projection matrix and output projection are kept frozen.
    Only the MLPs that predict the pre-output-projection attention value z are
    trainable.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        cfg: PEFTComparisonConfig,
        layer_indices: Sequence[int],
    ):
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
                str(layer_idx): QKVToZMLP(
                    hidden_size=hidden_size,
                    mlp_hidden_dim=cfg.mlp_attention_hidden_dim,
                    depth=cfg.mlp_attention_depth,
                    dropout=cfg.mlp_attention_dropout,
                    zero_init_output=cfg.mlp_attention_zero_init,
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

    def attention_qkv_and_teacher_z(
        self,
        *,
        block: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = block.attn
        residual = hidden_states
        x_ln1 = block.ln_1(hidden_states)

        qkv = attn.c_attn(x_ln1)
        q_raw, k_raw, v_raw = qkv.split(attn.split_size, dim=2)

        num_heads = attn.num_heads
        q = split_heads(q_raw, num_heads).float()
        k = split_heads(k_raw, num_heads).float()
        v = split_heads(v_raw, num_heads).float()
        z_soft = merge_heads(causal_soft_attention(q, k, v).float()).to(x_ln1.dtype)
        return residual, q, k, v, z_soft

    def forward_edited_block(self, hidden_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block = self.model.transformer.h[layer_idx]
        residual, q, k, v, z_soft = self.attention_qkv_and_teacher_z(block=block, hidden_states=hidden_states)

        attn = block.attn
        z_pred = self.adapters[str(layer_idx)](q, k, v).to(z_soft.dtype)
        attn_output = attn.c_proj(z_pred)
        attn_output = attn.resid_dropout(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        x_ln2 = block.ln_2(hidden_states)
        feed_forward_hidden_states = block.mlp(x_ln2)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, z_pred, z_soft

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
                hidden_states, _, _ = self.forward_edited_block(hidden_states, i)
            else:
                hidden_states = block_forward_hidden(transformer.h[i], hidden_states)

        hidden_states = transformer.ln_f(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits

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

        stats: Dict[str, float] = {}
        all_mse: List[torch.Tensor] = []
        all_cos: List[torch.Tensor] = []
        all_abs: List[torch.Tensor] = []
        all_l2: List[torch.Tensor] = []

        for i in range(first_edit, len(transformer.h)):
            if i in self.layer_set:
                hidden_states, z_pred, z_soft = self.forward_edited_block(hidden_states, i)
                mse = F.mse_loss(z_pred.float(), z_soft.float())
                cos = F.cosine_similarity(
                    z_pred.float().reshape(-1, z_pred.shape[-1]),
                    z_soft.float().reshape(-1, z_soft.shape[-1]),
                    dim=-1,
                ).mean()
                abs_mean = z_pred.float().abs().mean()
                l2_rms = z_pred.float().pow(2).mean().sqrt()

                all_mse.append(mse)
                all_cos.append(cos)
                all_abs.append(abs_mean)
                all_l2.append(l2_rms)

                stats[f"layer_{i}_z_mse_vs_soft"] = float(mse.item())
                stats[f"layer_{i}_z_cos_vs_soft"] = float(cos.item())
                stats[f"layer_{i}_z_pred_abs_mean"] = float(abs_mean.item())
                stats[f"layer_{i}_z_pred_l2_rms"] = float(l2_rms.item())
            else:
                hidden_states = block_forward_hidden(transformer.h[i], hidden_states)

        if all_mse:
            stats["z_mse_vs_soft"] = float(torch.stack(all_mse).mean().item())
            stats["z_cos_vs_soft"] = float(torch.stack(all_cos).mean().item())
            stats["z_pred_abs_mean"] = float(torch.stack(all_abs).mean().item())
            stats["z_pred_l2_rms"] = float(torch.stack(all_l2).mean().item())
        return stats


class OfficialPEFTWrapper(nn.Module):
    """Small wrapper that exposes the same interface as AKAZAFreeZModel."""

    def __init__(self, model: nn.Module, cfg: PEFTComparisonConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        return self.model(input_ids).logits

    def set_peft_train_mode(self) -> None:
        # Keep frozen GPT-2 dropout disabled. For LoRA, enable only LoRA dropout.
        self.model.eval()
        for name, module in self.model.named_modules():
            if "lora_dropout" in name:
                module.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        del input_ids
        trainable = trainable_parameters(self)
        if not trainable:
            return {}
        vec = torch.cat([p.detach().reshape(-1).float().cpu() for p in trainable])
        return {
            "peft_param_abs_mean": float(vec.abs().mean().item()),
            "peft_param_abs_max": float(vec.abs().max().item()),
            "peft_param_l2_rms": float(vec.pow(2).mean().sqrt().item()),
        }


def build_official_peft_model(
    *,
    model: nn.Module,
    cfg: PEFTComparisonConfig,
    layer_indices: Sequence[int],
) -> OfficialPEFTWrapper:
    try:
        from peft import IA3Config, LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "This method requires Hugging Face PEFT. Install it with `uv add peft` "
            "or `pip install peft`."
        ) from exc

    if cfg.method == "lora":
        if cfg.peft_target_profile not in LORA_TARGET_PROFILES:
            raise ValueError(
                f"LoRA profile {cfg.peft_target_profile!r} not in {sorted(LORA_TARGET_PROFILES)}"
            )
        suffixes = LORA_TARGET_PROFILES[cfg.peft_target_profile]
        target_modules = target_module_names(layer_indices, suffixes)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            fan_in_fan_out=True,  # GPT-2 Conv1D stores weights as fan_in, fan_out.
            bias=cfg.lora_bias,
            init_lora_weights=True,  # Default LoRA no-op init: B matrix is zero.
        )

    elif cfg.method == "ia3":
        if cfg.peft_target_profile not in IA3_TARGET_PROFILES:
            raise ValueError(
                f"IA3 profile {cfg.peft_target_profile!r} not in {sorted(IA3_TARGET_PROFILES)}"
            )
        profile = IA3_TARGET_PROFILES[cfg.peft_target_profile]
        target_modules = target_module_names(layer_indices, profile["target_modules"])
        feedforward_modules = target_module_names(layer_indices, profile["feedforward_modules"])

        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules if feedforward_modules else None,
            fan_in_fan_out=True,  # GPT-2 Conv1D stores weights as fan_in, fan_out.
            init_ia3_weights=cfg.ia3_init_weights,
        )
    else:
        raise ValueError(f"Unsupported official PEFT method: {cfg.method}")

    print()
    print(f"[peft] method={cfg.method}")
    print(f"[peft] target_profile={cfg.peft_target_profile}")
    print(f"[peft] target_modules={target_modules}")
    if cfg.method == "ia3":
        print(f"[peft] feedforward_modules={feedforward_modules}")

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return OfficialPEFTWrapper(peft_model, cfg)


def load_chunks_for_split(
    cfg: PEFTComparisonConfig,
    tokenizer,
    *,
    split: str,
    max_texts: int,
    max_chunks: int,
) -> torch.Tensor:
    split_cfg = replace(cfg, split=split, max_texts=max_texts, max_chunks=max_chunks)
    chunks = load_and_pack_texts(split_cfg, tokenizer, text_field=cfg.text_field).cpu()
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks


@torch.no_grad()
def eval_baseline(
    model: nn.Module,
    chunks: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    label: str = "baseline",
    log_every_batches: int = 10,
) -> float:
    model.eval()
    losses: List[float] = []
    n_examples = chunks.shape[0]
    batches = batch_slices(n_examples, batch_size)

    for batch_idx, sl in enumerate(batches, start=1):
        input_ids = chunks[sl].to(device)
        logits = model(input_ids).logits
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])
        if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == len(batches)):
            done = sl.stop
            running = sum(losses) / max(1, done)
            print(
                f"[{label}] eval_batch={batch_idx}/{len(batches)} "
                f"examples={done}/{n_examples} loss={running:.6f}",
                flush=True,
            )

    return sum(losses) / max(1, n_examples)


@torch.no_grad()
def eval_wrapped(
    wrapped: AKAZAFreeZModel | GPT2MLPAttentionReplacementModel | OfficialPEFTWrapper,
    chunks: torch.Tensor,
    batch_size: int,
    *,
    collect_stats: bool = True,
    label: str = "wrapped",
    log_every_batches: int = 10,
) -> Dict[str, float]:
    wrapped.set_peft_eval_mode()

    losses: List[float] = []
    stats_accum: Dict[str, float] = {}
    n_stats_batches = 0
    n_examples = chunks.shape[0]
    batches = batch_slices(n_examples, batch_size)

    for batch_idx, sl in enumerate(batches, start=1):
        input_ids = chunks[sl].to(wrapped.device)
        logits = wrapped(input_ids)
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])

        if collect_stats:
            stats = wrapped.peft_stats(input_ids)
            for k, v in stats.items():
                stats_accum[k] = stats_accum.get(k, 0.0) + float(v)
            n_stats_batches += 1
        if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == len(batches)):
            done = sl.stop
            running = sum(losses) / max(1, done)
            print(
                f"[{label}] eval_batch={batch_idx}/{len(batches)} "
                f"examples={done}/{n_examples} loss={running:.6f}",
                flush=True,
            )

    out = {"loss": sum(losses) / max(1, n_examples)}
    if collect_stats:
        for k, v in stats_accum.items():
            out[k] = v / max(1, n_stats_batches)
    return out


def build_wrapped_model(
    *,
    model: nn.Module,
    cfg: PEFTComparisonConfig,
    layer_indices: Sequence[int],
) -> AKAZAFreeZModel | GPT2MLPAttentionReplacementModel | OfficialPEFTWrapper:
    if cfg.method == "akaza_freez":
        return AKAZAFreeZModel(model=model, cfg=cfg, layer_indices=layer_indices)
    if cfg.method == "mlp_attention":
        return GPT2MLPAttentionReplacementModel(model=model, cfg=cfg, layer_indices=layer_indices)
    if cfg.method in {"lora", "ia3"}:
        return build_official_peft_model(model=model, cfg=cfg, layer_indices=layer_indices)
    raise ValueError(f"Unknown method={cfg.method!r}; choices={METHODS}")


def train(cfg: PEFTComparisonConfig) -> None:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method={cfg.method!r}; choices={METHODS}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)

    layer_indices = sorted(parse_int_csv(cfg.layer_indices))
    if not layer_indices:
        raise ValueError("--layer_indices cannot be empty")

    print("[config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print(f"  parsed_layer_indices: {layer_indices}")

    print()
    print(f"[model] loading tokenizer: {cfg.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading base model: {cfg.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    print(f"[model] loaded base model on {device}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.transformer.h)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")

    print()
    print("[data] loading train/val/test chunks separately", flush=True)
    print(
        f"[data] loading train split={cfg.train_split} "
        f"max_texts={cfg.max_train_texts} max_chunks={cfg.max_train_chunks}",
        flush=True,
    )
    train_chunks = load_chunks_for_split(
        cfg,
        tokenizer,
        split=cfg.train_split,
        max_texts=cfg.max_train_texts,
        max_chunks=cfg.max_train_chunks,
    )
    print(f"[data] loaded train chunks={train_chunks.shape[0]} block_size={train_chunks.shape[1]}", flush=True)
    print(
        f"[data] loading val split={cfg.val_split} "
        f"max_texts={cfg.max_val_texts} max_chunks={cfg.max_val_chunks}",
        flush=True,
    )
    val_chunks = load_chunks_for_split(
        cfg,
        tokenizer,
        split=cfg.val_split,
        max_texts=cfg.max_val_texts,
        max_chunks=cfg.max_val_chunks,
    )
    print(f"[data] loaded val chunks={val_chunks.shape[0]} block_size={val_chunks.shape[1]}", flush=True)
    print(
        f"[data] loading test split={cfg.test_split} "
        f"max_texts={cfg.max_test_texts} max_chunks={cfg.max_test_chunks}",
        flush=True,
    )
    test_chunks = load_chunks_for_split(
        cfg,
        tokenizer,
        split=cfg.test_split,
        max_texts=cfg.max_test_texts,
        max_chunks=cfg.max_test_chunks,
    )
    print(f"[data] loaded test chunks={test_chunks.shape[0]} block_size={test_chunks.shape[1]}", flush=True)

    print(f"[data] train chunks={train_chunks.shape[0]} block_size={train_chunks.shape[1]}")
    print(f"[data] val   chunks={val_chunks.shape[0]} block_size={val_chunks.shape[1]}")
    print(f"[data] test  chunks={test_chunks.shape[0]} block_size={test_chunks.shape[1]}")

    print("[baseline] evaluating train split", flush=True)
    baseline_train = eval_baseline(model, train_chunks, cfg.batch_size, device, label="baseline/train")
    print(f"[baseline] train_loss={baseline_train:.6f}", flush=True)
    print("[baseline] evaluating val split", flush=True)
    baseline_val = eval_baseline(model, val_chunks, cfg.batch_size, device, label="baseline/val")
    print(f"[baseline] val_loss={baseline_val:.6f}", flush=True)
    print("[baseline] evaluating test split", flush=True)
    baseline_test = eval_baseline(model, test_chunks, cfg.batch_size, device, label="baseline/test")
    print(f"[baseline] test_loss={baseline_test:.6f}", flush=True)

    print(f"[model] building wrapped method={cfg.method}", flush=True)
    wrapped = build_wrapped_model(model=model, cfg=cfg, layer_indices=layer_indices).to(device)
    num_trainable = sum(p.numel() for p in trainable_parameters(wrapped))
    print(f"[model] trainable_params={num_trainable}", flush=True)

    print("[wrapped@init] evaluating train split", flush=True)
    init_train = eval_wrapped(wrapped, train_chunks, cfg.batch_size, collect_stats=False, label="wrapped_init/train")
    print(f"[wrapped@init] train_loss={init_train['loss']:.6f}", flush=True)
    print("[wrapped@init] evaluating val split", flush=True)
    init_val = eval_wrapped(wrapped, val_chunks, cfg.batch_size, collect_stats=True, label="wrapped_init/val")
    print(f"[wrapped@init] val_loss={init_val['loss']:.6f}", flush=True)
    print("[wrapped@init] evaluating test split", flush=True)
    init_test = eval_wrapped(wrapped, test_chunks, cfg.batch_size, collect_stats=False, label="wrapped_init/test")
    print(f"[wrapped@init] test_loss={init_test['loss']:.6f}", flush=True)

    print()
    print(
        f"[baseline] train_loss={baseline_train:.6f} "
        f"val_loss={baseline_val:.6f} test_loss={baseline_test:.6f}"
    )
    print(
        f"[wrapped@init] train_loss={init_train['loss']:.6f} "
        f"val_loss={init_val['loss']:.6f} test_loss={init_test['loss']:.6f}"
    )
    print(
        f"[init_delta] train={init_train['loss'] - baseline_train:+.8f} "
        f"val={init_val['loss'] - baseline_val:+.8f} "
        f"test={init_test['loss'] - baseline_test:+.8f}"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters(wrapped),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    if not cfg.skip_freeze_check:
        assert_trainable_scope(wrapped, optimizer)
        frozen_before_training = clone_frozen_state(wrapped)
    else:
        frozen_before_training = {}

    best_val = float("inf")
    best_state: Dict[str, Any] | None = None
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        wrapped.set_peft_train_mode()

        perm = torch.randperm(train_chunks.shape[0])
        total_loss = 0.0
        total_examples = 0
        running_loss = 0.0
        running_examples = 0

        for sl in batch_slices(perm.numel(), cfg.batch_size):
            idx = perm[sl]
            input_ids = train_chunks[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(input_ids)
            loss = lm_loss(logits, input_ids)

            reg = peft_regularizer(wrapped, l2=cfg.peft_l2, l1=cfg.peft_l1)
            if reg is not None:
                loss = loss + reg

            loss.backward()

            if epoch == 1 and total_examples == 0 and not cfg.skip_freeze_check:
                assert_no_frozen_grads(wrapped)

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(wrapped), max_norm=cfg.grad_clip)

            optimizer.step()

            if epoch == 1 and total_examples == 0 and not cfg.skip_freeze_check:
                assert_frozen_unchanged(wrapped, frozen_before_training)

            total_loss += float(loss.item()) * input_ids.shape[0]
            total_examples += input_ids.shape[0]
            running_loss += float(loss.item()) * input_ids.shape[0]
            running_examples += input_ids.shape[0]
            global_step += 1

            if cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
                running_avg = running_loss / max(1, running_examples)
                epoch_avg = total_loss / max(1, total_examples)
                print(
                    f"[step {global_step:06d}] "
                    f"epoch={epoch:03d} "
                    f"examples={total_examples}/{train_chunks.shape[0]} "
                    f"batch_loss={float(loss.item()):.6f} "
                    f"running_loss={running_avg:.6f} "
                    f"epoch_loss={epoch_avg:.6f}",
                    flush=True,
                )
                running_loss = 0.0
                running_examples = 0

        train_loss = total_loss / max(1, total_examples)
        row: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}

        do_eval = epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs
        if do_eval:
            print(f"[eval] epoch={epoch:03d} evaluating val split", flush=True)
            val_metrics = eval_wrapped(wrapped, val_chunks, cfg.batch_size, collect_stats=True, label=f"eval/epoch_{epoch:03d}/val")
            val_loss = val_metrics["loss"]
            val_imp = baseline_val - val_loss

            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_improvement"] = val_imp
            row["val_relative_ppl_reduction"] = relative_ppl_reduction(val_imp)

            test_metrics = None
            test_imp = None
            if cfg.eval_test_during_training:
                print(f"[eval] epoch={epoch:03d} evaluating test split", flush=True)
                test_metrics = eval_wrapped(wrapped, test_chunks, cfg.batch_size, collect_stats=False, label=f"eval/epoch_{epoch:03d}/test")
                test_imp = baseline_test - test_metrics["loss"]
                row["test_loss_exploratory"] = test_metrics["loss"]
                row["test_improvement_exploratory"] = test_imp
                row["test_relative_ppl_reduction_exploratory"] = relative_ppl_reduction(test_imp)

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                bad_epochs = 0
                best_state = {
                    "trainable_state_dict": trainable_state_dict(wrapped),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }
            else:
                bad_epochs += 1

            msg = (
                f"[epoch {epoch:03d}] "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"val_imp={val_imp:.6f} "
                f"val_ppl_red={100.0 * relative_ppl_reduction(val_imp):.3f}% "
            )
            if cfg.eval_test_during_training and test_metrics is not None and test_imp is not None:
                msg += (
                    f"test={test_metrics['loss']:.6f} "
                    f"test_imp={test_imp:.6f} "
                f"test_ppl_red={100.0 * relative_ppl_reduction(test_imp):.3f}% "
            )
            msg += (
                f"peft_abs={val_metrics.get('delta_abs_mean', val_metrics.get('z_pred_abs_mean', val_metrics.get('peft_param_abs_mean', 0.0))):.6f} "
                f"peft_l2={val_metrics.get('delta_l2_rms', val_metrics.get('z_pred_l2_rms', val_metrics.get('peft_param_l2_rms', 0.0))):.6f}"
            )
            print(msg, flush=True)

            history.append(row)

            if bad_epochs >= cfg.patience:
                print(f"[early_stop] no val improvement for {bad_epochs} evals", flush=True)
                break
        else:
            print(f"[epoch {epoch:03d}] train={train_loss:.6f}", flush=True)
            history.append(row)

    if best_state is None:
        best_state = {
            "trainable_state_dict": trainable_state_dict(wrapped),
            "epoch": cfg.epochs,
            "val_loss": init_val["loss"],
        }

    load_trainable_state_dict(wrapped, best_state["trainable_state_dict"])

    print("[best] evaluating train split", flush=True)
    best_train_metrics = eval_wrapped(wrapped, train_chunks, cfg.batch_size, collect_stats=False, label="best/train")
    print("[best] evaluating val split", flush=True)
    best_val_metrics = eval_wrapped(wrapped, val_chunks, cfg.batch_size, collect_stats=True, label="best/val")
    print("[best] evaluating test split", flush=True)
    best_test_metrics = eval_wrapped(wrapped, test_chunks, cfg.batch_size, collect_stats=True, label="best/test")

    train_imp = baseline_train - best_train_metrics["loss"]
    val_imp = baseline_val - best_val_metrics["loss"]
    test_imp = baseline_test - best_test_metrics["loss"]

    summary = {
        "config": asdict(cfg),
        "method": cfg.method,
        "peft_target_profile": cfg.peft_target_profile,
        "layer_indices": layer_indices,
        "num_trainable_params": num_trainable,
        "trainable_param_names": [name for name, _ in trainable_named_parameters(wrapped)],
        "train_chunks": int(train_chunks.shape[0]),
        "val_chunks": int(val_chunks.shape[0]),
        "test_chunks": int(test_chunks.shape[0]),
        "block_size": int(train_chunks.shape[1]),
        "baseline_train_loss": baseline_train,
        "baseline_val_loss": baseline_val,
        "baseline_test_loss": baseline_test,
        "baseline_train_ppl": ppl_from_nll(baseline_train),
        "baseline_val_ppl": ppl_from_nll(baseline_val),
        "baseline_test_ppl": ppl_from_nll(baseline_test),
        "wrapped_init_train_loss": init_train["loss"],
        "wrapped_init_val_loss": init_val["loss"],
        "wrapped_init_test_loss": init_test["loss"],
        "wrapped_init_train_delta_vs_baseline": init_train["loss"] - baseline_train,
        "wrapped_init_val_delta_vs_baseline": init_val["loss"] - baseline_val,
        "wrapped_init_test_delta_vs_baseline": init_test["loss"] - baseline_test,
        "best_epoch": int(best_state["epoch"]),
        "best_train_loss": best_train_metrics["loss"],
        "best_val_loss": best_val_metrics["loss"],
        "best_test_loss": best_test_metrics["loss"],
        "best_train_improvement_nats_per_token": train_imp,
        "best_val_improvement_nats_per_token": val_imp,
        "best_test_improvement_nats_per_token": test_imp,
        "best_train_relative_ppl_reduction": relative_ppl_reduction(train_imp),
        "best_val_relative_ppl_reduction": relative_ppl_reduction(val_imp),
        "best_test_relative_ppl_reduction": relative_ppl_reduction(test_imp),
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "history": history,
    }

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[done] saving checkpoint: {out}", flush=True)
    torch.save(
        {
            "summary": summary,
            "trainable_state_dict": best_state["trainable_state_dict"],
        },
        out,
    )

    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"[done] wrote {out}")
    print(f"[done] wrote {summary_path}")
    print(
        f"[best] method={cfg.method} profile={cfg.peft_target_profile} "
        f"epoch={summary['best_epoch']} "
        f"val_loss={summary['best_val_loss']:.6f} "
        f"test_loss={summary['best_test_loss']:.6f} "
        f"val_imp={summary['best_val_improvement_nats_per_token']:.6f} nats/token "
        f"test_imp={summary['best_test_improvement_nats_per_token']:.6f} nats/token "
        f"test_ppl_red={100.0 * summary['best_test_relative_ppl_reduction']:.3f}%"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare custom pre-c_proj AKAZA/FreeZ adapters against official "
            "Hugging Face PEFT LoRA/IA3 baselines under the same frozen-GPT2 protocol."
        )
    )

    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_field", type=str, default="text")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--max_train_texts", type=int, default=1000)
    parser.add_argument("--max_val_texts", type=int, default=200)
    parser.add_argument("--max_test_texts", type=int, default=200)

    parser.add_argument("--block_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--max_train_chunks", type=int, default=2048)
    parser.add_argument("--max_val_chunks", type=int, default=512)
    parser.add_argument("--max_test_chunks", type=int, default=512)

    parser.add_argument("--layer_indices", type=str, default="6,7,8,9,10,11")
    parser.add_argument("--method", type=str, default="akaza_freez", choices=METHODS)

    # AKAZA / FreeZ options.
    parser.add_argument("--bottleneck_dim", type=int, default=4)
    parser.add_argument("--adapter_dropout", type=float, default=0.05)
    parser.add_argument("--adapter_input", type=str, default="ln1", choices=["ln1", "residual"])
    parser.add_argument("--detach_adapter_input", action="store_true")
    parser.add_argument("--no_detach_adapter_input", dest="detach_adapter_input", action="store_false")
    parser.set_defaults(detach_adapter_input=True)
    parser.add_argument("--output_scale", type=float, default=0.05)
    parser.add_argument("--mlp_attention_hidden_dim", type=int, default=768)
    parser.add_argument("--mlp_attention_depth", type=int, default=2)
    parser.add_argument("--mlp_attention_dropout", type=float, default=0.05)
    parser.add_argument("--mlp_attention_zero_init", action="store_true")
    parser.add_argument("--no_mlp_attention_zero_init", dest="mlp_attention_zero_init", action="store_false")
    parser.set_defaults(mlp_attention_zero_init=True)

    # Official PEFT options.
    parser.add_argument("--peft_target_profile", type=str, default="attn_c_proj", choices=ALL_TARGET_PROFILES)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--ia3_init_weights", action="store_true")
    parser.add_argument("--no_ia3_init_weights", dest="ia3_init_weights", action="store_false")
    parser.set_defaults(ia3_init_weights=True)

    parser.add_argument("--peft_l2", type=float, default=1e-5)
    parser.add_argument("--peft_l1", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=10,
        help="Print training loss every N optimizer steps. Set <=0 to disable.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--skip_freeze_check", action="store_true")
    parser.add_argument(
        "--eval_test_during_training",
        action="store_true",
        help="Exploratory only. Leave disabled for final reporting.",
    )

    # Present only because load_and_pack_texts expects a LearnerHyperParams-like config.
    parser.add_argument("--beta_soft", type=float, default=6.0)
    parser.add_argument("--k_sharp", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--k_linear_local", type=int, default=16)
    parser.add_argument("--ridge_lambda", type=float, default=1e-1)
    parser.add_argument("--k_knn_mean", type=int, default=4)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = PEFTComparisonConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        max_train_texts=args.max_train_texts,
        max_val_texts=args.max_val_texts,
        max_test_texts=args.max_test_texts,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_train_chunks=args.max_train_chunks,
        max_val_chunks=args.max_val_chunks,
        max_test_chunks=args.max_test_chunks,
        layer_indices=args.layer_indices,
        method=args.method,
        bottleneck_dim=args.bottleneck_dim,
        adapter_dropout=args.adapter_dropout,
        adapter_input=args.adapter_input,
        detach_adapter_input=args.detach_adapter_input,
        output_scale=args.output_scale,
        mlp_attention_hidden_dim=args.mlp_attention_hidden_dim,
        mlp_attention_depth=args.mlp_attention_depth,
        mlp_attention_dropout=args.mlp_attention_dropout,
        mlp_attention_zero_init=args.mlp_attention_zero_init,
        peft_target_profile=args.peft_target_profile,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        ia3_init_weights=args.ia3_init_weights,
        peft_l2=args.peft_l2,
        peft_l1=args.peft_l1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        log_every_steps=args.log_every_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        output_path=args.output_path,
        skip_freeze_check=args.skip_freeze_check,
        eval_test_during_training=args.eval_test_during_training,
        local_kernel_beta=args.beta_soft,
        k_sharp=args.k_sharp,
        window_size=args.window_size,
        k_linear_local=args.k_linear_local,
        ridge_lambda=args.ridge_lambda,
        k_knn_mean=args.k_knn_mean,
        split=args.train_split,
        max_texts=args.max_train_texts,
        max_chunks=args.max_train_chunks,
    )

    train(cfg)


if __name__ == "__main__":
    main()
