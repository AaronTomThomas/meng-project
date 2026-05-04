from __future__ import annotations

"""
Evaluate whether a trained AKAZA/FreeZ correction can be explained by a dictionary
of candidate attention-learning algorithm directions.

This is the natural follow-up to eval_akaza_delta_decomposition.py.
The previous script asked:

  Is delta mostly parallel to z, i.e. gate-like, or orthogonal/directional?

This script asks:

  If delta is directional, is it directional in ways explained by known candidate
  attention learners?

For each edited layer and token, AKAZA produces:

    z_akaza = z_soft + delta

For each candidate learner a, we compute a candidate attention-output vector:

    z_a

and define an algorithmic direction:

    d_a = z_a - z_soft

Then we fit a local ridge projection:

    alpha* = argmin_alpha ||delta - sum_a alpha_a d_a||_2^2
                         + lambda ||alpha||_2^2

This decomposes the AKAZA correction into:

    delta_alg = sum_a alpha_a d_a
    delta_residual = delta - delta_alg

Evaluation modes, without retraining:

  1. none
       z_new = z_soft
       Frozen baseline sanity check.

  2. full
       z_new = z_soft + delta
       Original AKAZA.

  3. projection
       z_new = z_soft + delta_alg
       Keeps only the component explained by the candidate learner dictionary.

  4. residual
       z_new = z_soft + delta_residual
       Keeps only the component not explained by the candidate learner dictionary.

  5. top1_abs_cosine
       z_new = z_soft + best one-dimensional scaled learner direction
       Useful interpretability stress test: can one candidate direction explain the
       correction locally?

The thesis-useful quantity is recovered gain fraction:

    recovered_gain_fraction =
        (L_baseline - L_projection) / (L_baseline - L_full)

If this is low, the current dictionary does not explain AKAZA's gains. That is not
necessarily bad: it says AKAZA discovered useful correction directions outside the
hand-designed learner basis. If expanding the dictionary raises this value, the
added learners explain more of AKAZA's learned algorithmic behaviour.

Example:

PYTHONPATH=src uv run python src/experiments/router_development/adapter_finetune/eval_akaza_algorithmic_direction_decomposition.py \
  --checkpoint_path outputs/adapter_finetune/akaza_freez_b4_scale1_seed0_equiv_gated.pt \
  --device cuda \
  --output_path outputs/adapter_finetune/akaza_freez_b4_scale1_seed0_alg_direction_decomp.json \
  --candidate_learners sharp,window_soft,topk_soft,knn_mean,uniform_window,temp_sharp,temp_smooth,recency_soft
"""

import argparse
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.router_development.adapter_finetune.train_peft_comparison_adapter import (
    AKAZAFreeZModel,
    PEFTComparisonConfig,
    block_forward_hidden,
    eval_baseline,
    lm_loss,
    load_chunks_for_split,
    load_trainable_state_dict,
    parse_int_csv,
    relative_ppl_reduction,
)


BASE_DECOMP_MODES = ["none", "full", "projection", "residual", "top1_abs_cosine"]
DEFAULT_CANDIDATE_LEARNERS = [
    # Hard / sparse retrieval.
    "sharp",
    "topk_soft_4",
    "knn_mean_4",

    # Locality / prefix baselines.
    "window_soft_16",
    "uniform_window_16",
    "prefix_mean",

    # Temperature and recency perturbations.
    "temp_2",
    "temp_0p5",
    "recency_soft_0p05",

    # Copy / anti-copy style probes.
    "self_value",
    "prev_value",
    "exclude_self_soft",
    "dissimilar_soft",

    # Rank/tail probes. These are useful if AKAZA is suppressing the top match or
    # extracting mid-ranked context rather than simply sharpening/softening.
    "exclude_topk_soft_1",
    "exclude_topk_soft_4",
    "rank_band_soft_1_4",
    "rank_band_mean_1_4",
    "reverse_rank_band_soft_1_4",

    # Probability-shape probes.
    "prob_power_2",
    "prob_power_0p5",

    # Value-aware probes: attends to high/low norm values, not just q-k similarity.
    "value_norm_soft_0p5",
    "anti_value_norm_soft_0p5",
]


def safe_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def parse_csv(s: str | None) -> List[str]:
    if s is None or not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_number_token(s: str) -> float:
    """Parse compact CLI-safe numeric tokens like 0p5, 0p05, 2, 4p0."""
    return float(s.replace("p", "."))


def int_suffix(name: str, prefix: str, default: int) -> int:
    if name == prefix:
        return int(default)
    if name.startswith(prefix + "_"):
        return int(name[len(prefix) + 1 :])
    raise ValueError(f"{name!r} does not match prefix {prefix!r}")


def float_suffix(name: str, prefix: str, default: float) -> float:
    if name == prefix:
        return float(default)
    if name.startswith(prefix + "_"):
        return parse_number_token(name[len(prefix) + 1 :])
    raise ValueError(f"{name!r} does not match prefix {prefix!r}")


def two_int_suffix(name: str, prefix: str) -> tuple[int, int]:
    """Parse names like band_soft_4_16 into (4, 16)."""
    stem = prefix + "_"
    if not name.startswith(stem):
        raise ValueError(f"{name!r} does not match prefix {prefix!r}")
    parts = name[len(stem) :].split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected {prefix}_LO_HI, got {name!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 0 or hi <= lo:
        raise ValueError(f"Invalid band in {name!r}; expected 0 <= LO < HI")
    return lo, hi


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B, T, C] -> [B, H, T, D]."""
    bsz, seq_len, width = x.shape
    if width % num_heads != 0:
        raise ValueError(f"width={width} is not divisible by num_heads={num_heads}")
    head_dim = width // num_heads
    return x.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, D] -> [B, T, C]."""
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, num_heads * head_dim)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return [1, 1, T, T] bool mask where True means attention is allowed."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).view(1, 1, seq_len, seq_len)


def finite_neg(dtype: torch.dtype) -> float:
    # Avoid literal -inf because some older kernels/half paths can produce NaNs
    # when a row contains many masked entries.
    if dtype in {torch.float16, torch.bfloat16}:
        return -1e4
    return -1e9


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~mask, finite_neg(scores.dtype))
    return torch.softmax(scores, dim=-1)


def attention_scores_from_block(block: nn.Module, hidden_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GPT-2 q/k/v and causal attention scores for one block.

    Returns:
        scores: [B, H, T, T], already scaled and causally masked with large negatives
        q:      [B, H, T, D]
        k:      [B, H, T, D]
        v:      [B, H, T, D]

    This intentionally assumes no padding attention_mask because these experiments
    use packed fixed-length WikiText chunks.
    """
    attn = block.attn
    x_ln1 = block.ln_1(hidden_states)
    qkv = attn.c_attn(x_ln1)
    query, key, value = qkv.split(attn.split_size, dim=2)

    num_heads = int(attn.num_heads)
    q = split_heads(query, num_heads)
    k = split_heads(key, num_heads)
    v = split_heads(value, num_heads)

    scores = torch.matmul(q, k.transpose(-1, -2))

    if getattr(attn, "scale_attn_weights", True):
        scores = scores / math.sqrt(v.size(-1))

    if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
        # HuggingFace GPT2Attention uses layer_idx + 1 when this option is enabled.
        scores = scores / float(layer_idx + 1)

    mask = causal_mask(scores.size(-1), scores.device)
    scores = scores.masked_fill(~mask, finite_neg(scores.dtype))
    return scores, q, k, v


def apply_weights(weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """weights [B,H,T,T], v [B,H,T,D] -> merged z [B,T,C]."""
    z_heads = torch.matmul(weights, v)
    return merge_heads(z_heads)


def one_hot_backshift_weights(seq_len: int, back: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [1,1,T,T] weights selecting token t-back, clipped at 0."""
    rows = torch.arange(seq_len, device=device)
    cols = (rows - int(back)).clamp_min(0)
    weights = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    weights[rows, cols] = 1.0
    return weights.view(1, 1, seq_len, seq_len)


def strict_past_mask_with_first_self(seq_len: int, device: torch.device) -> torch.Tensor:
    """Causal mask excluding self; first token falls back to self to avoid empty rows."""
    idx = torch.arange(seq_len, device=device)
    i = idx.view(seq_len, 1)
    j = idx.view(1, seq_len)
    mask = j < i
    mask[0, 0] = True
    return mask.view(1, 1, seq_len, seq_len)


def make_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    i = idx.view(seq_len, 1)
    j = idx.view(1, seq_len)
    dist = i - j
    allowed = (dist >= 0) & (dist < int(window_size))
    return allowed.view(1, 1, seq_len, seq_len)


def make_band_mask(seq_len: int, lo: int, hi: int, device: torch.device) -> torch.Tensor:
    """Causal distance band mask: allow tokens with lo <= t_query - t_key < hi.

    Empty early rows fall back to self so every query has at least one valid key.
    """
    idx = torch.arange(seq_len, device=device)
    i = idx.view(seq_len, 1)
    j = idx.view(1, seq_len)
    dist = i - j
    allowed = (dist >= int(lo)) & (dist < int(hi))
    row_empty = ~allowed.any(dim=-1)
    if row_empty.any():
        allowed[row_empty, idx[row_empty]] = True
    return allowed.view(1, 1, seq_len, seq_len)


def make_older_than_mask(seq_len: int, min_age: int, device: torch.device) -> torch.Tensor:
    """Allow only keys at least min_age positions behind the query.

    Empty early rows fall back to self.
    """
    idx = torch.arange(seq_len, device=device)
    i = idx.view(seq_len, 1)
    j = idx.view(1, seq_len)
    dist = i - j
    allowed = dist >= int(min_age)
    row_empty = ~allowed.any(dim=-1)
    if row_empty.any():
        allowed[row_empty, idx[row_empty]] = True
    return allowed.view(1, 1, seq_len, seq_len)


def make_recency_bias(seq_len: int, device: torch.device, strength: float) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    i = idx.view(seq_len, 1)
    j = idx.view(1, seq_len)
    dist = (i - j).clamp_min(0).float()
    # Negative distance bias: closer tokens receive less penalty.
    return (-float(strength) * dist).view(1, 1, seq_len, seq_len)


def topk_mask_from_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return bool mask with top-k entries per query row."""
    k_eff = max(1, min(int(k), scores.size(-1)))
    _, idx = torch.topk(scores, k=k_eff, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return mask


def bottomk_mask_from_scores(scores: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Return bool mask with bottom-k valid entries per query row."""
    k_eff = max(1, min(int(k), scores.size(-1)))
    large_pos = 1e4 if scores.dtype in {torch.float16, torch.bfloat16} else 1e9
    masked_scores = scores.masked_fill(~mask, large_pos)
    _, idx = torch.topk(-masked_scores, k=k_eff, dim=-1)
    out = torch.zeros_like(scores, dtype=torch.bool)
    out.scatter_(dim=-1, index=idx, value=True)
    return out & mask


def rank_decay_weights(scores: torch.Tensor, mask: torch.Tensor, power: float, *, descending: bool) -> torch.Tensor:
    """Rank-based attention weights over valid keys.

    descending=True ranks most similar first. descending=False ranks least similar
    first. Weight is 1 / rank**power after causal masking.
    """
    fill = finite_neg(scores.dtype) if descending else -finite_neg(scores.dtype)
    masked_scores = scores.masked_fill(~mask, fill)
    order = torch.argsort(masked_scores, dim=-1, descending=descending)
    rank_values = torch.arange(1, scores.size(-1) + 1, device=scores.device, dtype=torch.float32)
    rank_values = rank_values.view(*([1] * (scores.dim() - 1)), scores.size(-1))
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks.scatter_(dim=-1, index=order, src=rank_values.expand_as(scores.float()))
    weights = ranks.pow(-float(power)).masked_fill(~mask, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights.to(scores.dtype)


def rank_band_mask_from_scores(
    scores: torch.Tensor,
    mask: torch.Tensor,
    lo: int,
    hi: int,
    *,
    descending: bool,
) -> torch.Tensor:
    """Select a band of valid keys by similarity rank.

    lo/hi are zero-indexed half-open rank bounds. rank_band_* ranks highest scores
    first; reverse_rank_band_* ranks lowest scores first. Empty rows fall back to
    the best valid key under the same ranking direction.
    """
    fill = finite_neg(scores.dtype) if descending else -finite_neg(scores.dtype)
    masked_scores = scores.masked_fill(~mask, fill)
    order = torch.argsort(masked_scores, dim=-1, descending=descending)
    rank_values = torch.arange(scores.size(-1), device=scores.device, dtype=torch.long)
    rank_values = rank_values.view(*([1] * (scores.dim() - 1)), scores.size(-1))
    ranks = torch.empty_like(order)
    ranks.scatter_(dim=-1, index=order, src=rank_values.expand_as(order))
    selected = (ranks >= int(lo)) & (ranks < int(hi)) & mask

    row_empty = ~selected.any(dim=-1, keepdim=True)
    if row_empty.any():
        fallback = topk_mask_from_scores(scores, 1) & mask if descending else bottomk_mask_from_scores(scores, mask, 1)
        selected = selected | (row_empty & fallback)
    return selected


def exclude_topk_mask_from_scores(scores: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Keep valid keys except the top-k most similar ones; empty rows fall back to self/top1."""
    top = topk_mask_from_scores(scores, k) & mask
    selected = mask & ~top
    row_empty = ~selected.any(dim=-1, keepdim=True)
    if row_empty.any():
        fallback = topk_mask_from_scores(scores, 1) & mask
        selected = selected | (row_empty & fallback)
    return selected


def exclude_bottomk_mask_from_scores(scores: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Keep valid keys except the bottom-k least similar ones; empty rows fall back to top1."""
    bottom = bottomk_mask_from_scores(scores, mask, k)
    selected = mask & ~bottom
    row_empty = ~selected.any(dim=-1, keepdim=True)
    if row_empty.any():
        fallback = topk_mask_from_scores(scores, 1) & mask
        selected = selected | (row_empty & fallback)
    return selected


def prob_power_weights(scores: torch.Tensor, mask: torch.Tensor, power: float) -> torch.Tensor:
    """Raise ordinary softmax probabilities to a power, then renormalize."""
    weights = masked_softmax(scores, mask).float().clamp_min(1e-30).pow(float(power)).masked_fill(~mask, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights.to(scores.dtype)


def value_norm_biased_weights(scores: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, strength: float) -> torch.Tensor:
    """Soft attention biased toward high/low value-norm keys.

    strength > 0 favours high-norm value vectors; strength < 0 favours low-norm.
    This tests value-aware retrieval directions that pure q-k learners cannot express.
    """
    v_norm = v.float().pow(2).sum(dim=-1).sqrt()  # [B,H,T]
    v_norm = (v_norm - v_norm.mean(dim=-1, keepdim=True)) / v_norm.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    bias = float(strength) * v_norm.unsqueeze(-2).to(scores.dtype)  # key-axis bias [B,H,1,T]
    return masked_softmax(scores + bias, mask)


def positive_linear_weights(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """A simple non-softmax linear-weight learner.

    This is intentionally a rough dictionary element rather than a claim that this
    exactly matches older weighted_linear code. It keeps only positive shifted
    causal scores and normalizes them linearly.
    """
    masked_scores = scores.masked_fill(~mask, finite_neg(scores.dtype))
    row_min = masked_scores.masked_fill(~mask, 0.0).amin(dim=-1, keepdim=True)
    shifted = (scores - row_min).clamp_min(0.0).masked_fill(~mask, 0.0)
    denom = shifted.sum(dim=-1, keepdim=True).clamp_min(eps)
    return shifted / denom


def candidate_attention_outputs(
    *,
    block: nn.Module,
    hidden_states: torch.Tensor,
    layer_idx: int,
    candidate_names: Sequence[str],
    window_size: int,
    top_k: int,
    temp_sharp: float,
    temp_smooth: float,
    recency_strength: float,
) -> Dict[str, torch.Tensor]:
    """Compute candidate learner z outputs for one block.

    All outputs are merged pre-c_proj attention outputs with shape [B, T, C].
    """
    scores, _q, _k, v = attention_scores_from_block(block, hidden_states, layer_idx)
    seq_len = scores.size(-1)
    base_mask = causal_mask(seq_len, scores.device)
    out: Dict[str, torch.Tensor] = {}

    for name in candidate_names:
        if name == "soft":
            weights = masked_softmax(scores, base_mask)
            out[name] = apply_weights(weights, v)

        elif name == "sharp":
            # Hard argmax retrieval within the causal prefix.
            idx = scores.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scores)
            weights.scatter_(dim=-1, index=idx, value=1.0)
            weights = weights.masked_fill(~base_mask, 0.0)
            row_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights / row_sum, v)

        elif name.startswith("window_soft"):
            w = int_suffix(name, "window_soft", window_size)
            mask = make_window_mask(seq_len, w, scores.device)
            weights = masked_softmax(scores, mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("topk_soft"):
            k = int_suffix(name, "topk_soft", top_k)
            top_mask = topk_mask_from_scores(scores, k) & base_mask
            weights = masked_softmax(scores, top_mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("knn_mean"):
            k = int_suffix(name, "knn_mean", top_k)
            top_mask = topk_mask_from_scores(scores, k) & base_mask
            weights = top_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("uniform_window"):
            w = int_suffix(name, "uniform_window", window_size)
            mask = make_window_mask(seq_len, w, scores.device)
            weights = mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("band_soft_"):
            lo, hi = two_int_suffix(name, "band_soft")
            mask = make_band_mask(seq_len, lo, hi, scores.device)
            weights = masked_softmax(scores, mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("uniform_band_"):
            lo, hi = two_int_suffix(name, "uniform_band")
            mask = make_band_mask(seq_len, lo, hi, scores.device)
            weights = mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("old_soft_"):
            min_age = int(name[len("old_soft_") :])
            mask = make_older_than_mask(seq_len, min_age, scores.device)
            weights = masked_softmax(scores, mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("old_mean_"):
            min_age = int(name[len("old_mean_") :])
            mask = make_older_than_mask(seq_len, min_age, scores.device)
            weights = mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name == "prefix_mean":
            weights = base_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name == "self_value":
            weights = one_hot_backshift_weights(seq_len, 0, scores.device, v.dtype)
            out[name] = apply_weights(weights, v)

        elif name == "prev_value":
            weights = one_hot_backshift_weights(seq_len, 1, scores.device, v.dtype)
            out[name] = apply_weights(weights, v)

        elif name.startswith("back_value_"):
            back = int(name[len("back_value_") :])
            weights = one_hot_backshift_weights(seq_len, back, scores.device, v.dtype)
            out[name] = apply_weights(weights, v)

        elif name == "exclude_self_soft":
            mask = strict_past_mask_with_first_self(seq_len, scores.device)
            weights = masked_softmax(scores, mask)
            out[name] = apply_weights(weights, v)

        elif name == "dissimilar_soft":
            weights = masked_softmax(-scores, base_mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("bottomk_soft_"):
            k = int(name[len("bottomk_soft_") :])
            bottom_mask = bottomk_mask_from_scores(scores, base_mask, k)
            weights = masked_softmax(scores, bottom_mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("bottomk_mean_"):
            k = int(name[len("bottomk_mean_") :])
            bottom_mask = bottomk_mask_from_scores(scores, base_mask, k)
            weights = bottom_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("exclude_topk_soft_"):
            k = int(name[len("exclude_topk_soft_") :])
            selected = exclude_topk_mask_from_scores(scores, base_mask, k)
            weights = masked_softmax(scores, selected)
            out[name] = apply_weights(weights, v)

        elif name.startswith("exclude_topk_mean_"):
            k = int(name[len("exclude_topk_mean_") :])
            selected = exclude_topk_mask_from_scores(scores, base_mask, k)
            weights = selected.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("exclude_bottomk_soft_"):
            k = int(name[len("exclude_bottomk_soft_") :])
            selected = exclude_bottomk_mask_from_scores(scores, base_mask, k)
            weights = masked_softmax(scores, selected)
            out[name] = apply_weights(weights, v)

        elif name.startswith("exclude_bottomk_mean_"):
            k = int(name[len("exclude_bottomk_mean_") :])
            selected = exclude_bottomk_mask_from_scores(scores, base_mask, k)
            weights = selected.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("rank_band_soft_"):
            lo, hi = two_int_suffix(name, "rank_band_soft")
            selected = rank_band_mask_from_scores(scores, base_mask, lo, hi, descending=True)
            weights = masked_softmax(scores, selected)
            out[name] = apply_weights(weights, v)

        elif name.startswith("rank_band_mean_"):
            lo, hi = two_int_suffix(name, "rank_band_mean")
            selected = rank_band_mask_from_scores(scores, base_mask, lo, hi, descending=True)
            weights = selected.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("reverse_rank_band_soft_"):
            lo, hi = two_int_suffix(name, "reverse_rank_band_soft")
            selected = rank_band_mask_from_scores(scores, base_mask, lo, hi, descending=False)
            weights = masked_softmax(scores, selected)
            out[name] = apply_weights(weights, v)

        elif name.startswith("reverse_rank_band_mean_"):
            lo, hi = two_int_suffix(name, "reverse_rank_band_mean")
            selected = rank_band_mask_from_scores(scores, base_mask, lo, hi, descending=False)
            weights = selected.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("prob_power_"):
            power = float_suffix(name, "prob_power", 1.0)
            weights = prob_power_weights(scores, base_mask, power)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("value_norm_soft_"):
            strength = float_suffix(name, "value_norm_soft", 0.5)
            weights = value_norm_biased_weights(scores, v, base_mask, strength)
            out[name] = apply_weights(weights, v)

        elif name.startswith("anti_value_norm_soft_"):
            strength = float_suffix(name, "anti_value_norm_soft", 0.5)
            weights = value_norm_biased_weights(scores, v, base_mask, -strength)
            out[name] = apply_weights(weights, v)

        elif name.startswith("rank_decay_"):
            power = float_suffix(name, "rank_decay", 1.0)
            weights = rank_decay_weights(scores, base_mask, power, descending=True)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name.startswith("reverse_rank_decay_"):
            power = float_suffix(name, "reverse_rank_decay", 1.0)
            weights = rank_decay_weights(scores, base_mask, power, descending=False)
            out[name] = apply_weights(weights.to(v.dtype), v)

        elif name in {"temp_sharp", "temp_smooth"} or name.startswith("temp_"):
            if name == "temp_sharp":
                multiplier = float(temp_sharp)
            elif name == "temp_smooth":
                multiplier = float(temp_smooth)
            else:
                multiplier = float_suffix(name, "temp", 1.0)
            weights = masked_softmax(scores * multiplier, base_mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("recency_soft"):
            strength = float_suffix(name, "recency_soft", recency_strength)
            bias = make_recency_bias(seq_len, scores.device, strength).to(scores.dtype)
            weights = masked_softmax(scores + bias, base_mask)
            out[name] = apply_weights(weights, v)

        elif name.startswith("anti_recency_soft"):
            strength = float_suffix(name, "anti_recency_soft", recency_strength)
            bias = (-make_recency_bias(seq_len, scores.device, strength)).to(scores.dtype)
            weights = masked_softmax(scores + bias, base_mask)
            out[name] = apply_weights(weights, v)

        elif name == "linear_scores":
            weights = positive_linear_weights(scores, base_mask)
            out[name] = apply_weights(weights.to(v.dtype), v)

        else:
            raise ValueError(
                f"Unknown candidate learner {name!r}. Supported forms include: soft, sharp, "
                "window_soft[_W], topk_soft[_K], knn_mean[_K], uniform_window[_W], "
                "band_soft_LO_HI, uniform_band_LO_HI, old_soft_N, old_mean_N, "
                "prefix_mean, self_value, prev_value, back_value_N, exclude_self_soft, "
                "dissimilar_soft, bottomk_soft_K, bottomk_mean_K, exclude_topk_soft_K, "
                "exclude_topk_mean_K, exclude_bottomk_soft_K, exclude_bottomk_mean_K, "
                "rank_band_soft_LO_HI, rank_band_mean_LO_HI, reverse_rank_band_soft_LO_HI, "
                "reverse_rank_band_mean_LO_HI, rank_decay_P, reverse_rank_decay_P, "
                "prob_power_P, value_norm_soft_S, anti_value_norm_soft_S, temp_M, "
                "temp_sharp, temp_smooth, recency_soft[_S], anti_recency_soft[_S], "
                "linear_scores. Numeric decimals use p, e.g. temp_0p5."
            )

    return out


def ridge_project_delta(
    delta: torch.Tensor,
    directions: torch.Tensor,
    *,
    ridge_lambda: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project delta onto local dictionary directions.

    Args:
        delta:      [B, T, C]
        directions: [B, T, A, C]

    Returns:
        projected_delta: [B, T, C]
        alpha:           [B, T, A]

    Numerical note:
        Rich dictionaries often contain duplicate or zero directions, especially at
        early sequence positions. A tiny absolute ridge is not always enough once
        direction scales vary, so the diagonal stabilizer is relative to the local
        Gram scale. Failed rows fall back to a pseudo-inverse solve.
    """
    bsz, seq_len, num_candidates, width = directions.shape
    d = torch.nan_to_num(delta.float()).reshape(-1, width)  # [N, C]
    D = torch.nan_to_num(directions.float()).reshape(-1, num_candidates, width)  # [N, A, C]

    G = torch.matmul(D, D.transpose(-1, -2))  # [N, A, A]
    rhs = torch.einsum("nac,nc->na", D, d)  # [N, A]

    eye = torch.eye(num_candidates, device=delta.device, dtype=torch.float32).view(1, num_candidates, num_candidates)

    # Scale-relative ridge. If all dictionary directions are zero for a token,
    # mean_diag is zero and we fall back to eps as an absolute stabilizer.
    mean_diag = G.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(float(eps))
    diag_jitter = (float(ridge_lambda) * mean_diag).view(-1, 1, 1) + float(eps)
    G_reg = G + diag_jitter * eye

    solve_out = torch.linalg.solve_ex(G_reg, rhs.unsqueeze(-1))
    # PyTorch versions differ slightly: some expose .result, older/newer docs may
    # show tuple-like access. Be defensive so this script works across envs.
    if hasattr(solve_out, "result"):
        solve_result = solve_out.result
        solve_info = solve_out.info
    elif hasattr(solve_out, "solution"):
        solve_result = solve_out.solution
        solve_info = solve_out.info
    else:
        solve_result, solve_info = solve_out

    alpha = solve_result.squeeze(-1)

    failed = solve_info != 0
    if failed.any():
        # Rare but possible with large, redundant dictionaries. Pseudo-inverse is
        # slower but robust, and only applied to failed token rows.
        G_bad = G_reg[failed]
        rhs_bad = rhs[failed]
        alpha_bad = torch.matmul(torch.linalg.pinv(G_bad), rhs_bad.unsqueeze(-1)).squeeze(-1)
        alpha = alpha.clone()
        alpha[failed] = alpha_bad

    alpha = torch.nan_to_num(alpha)
    projected = torch.einsum("na,nac->nc", alpha, D)

    return projected.reshape(bsz, seq_len, width).to(delta.dtype), alpha.reshape(bsz, seq_len, num_candidates)


def top1_scaled_projection(
    delta: torch.Tensor,
    directions: torch.Tensor,
    *,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best single scaled direction by absolute cosine with delta.

    Args:
        delta:      [B, T, C]
        directions: [B, T, A, C]

    Returns:
        top1_delta: [B, T, C]
        top1_idx:   [B, T]
    """
    d = delta.float()
    D = directions.float()
    dot = torch.einsum("btac,btc->bta", D, d)
    D_norm = D.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
    d_norm = d.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    cos = dot / (D_norm * d_norm)
    top_idx = cos.abs().argmax(dim=-1)  # [B, T]

    gather_idx = top_idx[..., None, None].expand(*top_idx.shape, 1, D.size(-1))
    top_dir = D.gather(dim=2, index=gather_idx).squeeze(2)  # [B, T, C]
    top_dot = (top_dir * d).sum(dim=-1, keepdim=True)
    top_norm2 = top_dir.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = top_dot / top_norm2
    return (alpha * top_dir).to(delta.dtype), top_idx


class AlgorithmicDecomposedAKAZAFreeZModel(AKAZAFreeZModel):
    """AKAZA wrapper with evaluation-time algorithmic projection modes."""

    def __init__(
        self,
        *,
        model: nn.Module,
        cfg: PEFTComparisonConfig,
        layer_indices: Sequence[int],
        candidate_names: Sequence[str],
        decomp_mode: str = "full",
        ridge_lambda: float = 1e-4,
        eps: float = 1e-8,
        window_size: int = 16,
        top_k: int = 4,
        temp_sharp: float = 2.0,
        temp_smooth: float = 0.5,
        recency_strength: float = 0.05,
    ):
        super().__init__(model=model, cfg=cfg, layer_indices=layer_indices)
        self.candidate_names = list(candidate_names)
        self.decomp_mode = decomp_mode
        self.ridge_lambda = float(ridge_lambda)
        self.eps = float(eps)
        self.window_size = int(window_size)
        self.top_k = int(top_k)
        self.temp_sharp = float(temp_sharp)
        self.temp_smooth = float(temp_smooth)
        self.recency_strength = float(recency_strength)

        if self.decomp_mode not in BASE_DECOMP_MODES:
            raise ValueError(f"Unknown decomp_mode={self.decomp_mode!r}; choices={BASE_DECOMP_MODES}")
        if not self.candidate_names:
            raise ValueError("candidate_names must be non-empty")
        if "soft" in self.candidate_names:
            raise ValueError("Do not include 'soft' as a candidate direction; it gives a zero direction.")

    def set_decomp_mode(self, mode: str) -> None:
        if mode not in BASE_DECOMP_MODES:
            raise ValueError(f"Unknown decomp_mode={mode!r}; choices={BASE_DECOMP_MODES}")
        self.decomp_mode = mode

    def candidate_directions(
        self,
        *,
        block: nn.Module,
        hidden_states: torch.Tensor,
        layer_idx: int,
        z_soft: torch.Tensor,
    ) -> torch.Tensor:
        cand = candidate_attention_outputs(
            block=block,
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            candidate_names=self.candidate_names,
            window_size=self.window_size,
            top_k=self.top_k,
            temp_sharp=self.temp_sharp,
            temp_smooth=self.temp_smooth,
            recency_strength=self.recency_strength,
        )
        dirs = [cand[name].to(z_soft.dtype) - z_soft for name in self.candidate_names]
        return torch.stack(dirs, dim=2)  # [B, T, A, C]

    def select_delta(
        self,
        *,
        block: nn.Module,
        hidden_states: torch.Tensor,
        layer_idx: int,
        z_soft: torch.Tensor,
        delta_full: torch.Tensor,
    ) -> torch.Tensor:
        if self.decomp_mode == "none":
            return torch.zeros_like(delta_full)
        if self.decomp_mode == "full":
            return delta_full

        directions = self.candidate_directions(
            block=block,
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            z_soft=z_soft,
        )

        if self.decomp_mode in {"projection", "residual"}:
            delta_projection, _alpha = ridge_project_delta(
                delta_full,
                directions,
                ridge_lambda=self.ridge_lambda,
                eps=self.eps,
            )
            if self.decomp_mode == "projection":
                return delta_projection
            return delta_full - delta_projection

        if self.decomp_mode == "top1_abs_cosine":
            delta_top1, _top_idx = top1_scaled_projection(delta_full, directions, eps=self.eps)
            return delta_top1

        raise ValueError(f"Unknown decomp_mode={self.decomp_mode!r}")

    def forward_edited_block(self, hidden_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.model.transformer.h[layer_idx]
        residual, z_soft = self.attention_parts(block=block, hidden_states=hidden_states)
        delta_full = self.compute_delta(layer_idx=layer_idx, block=block, hidden_states=hidden_states)
        delta_used = self.select_delta(
            block=block,
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            z_soft=z_soft,
            delta_full=delta_full,
        ).to(z_soft.dtype)

        attn = block.attn
        z_new = z_soft + delta_used
        attn_output = attn.c_proj(z_new)
        attn_output = attn.resid_dropout(attn_output)
        hidden_states = residual + attn_output

        residual2 = hidden_states
        x_ln2 = block.ln_2(hidden_states)
        hidden_states = residual2 + block.mlp(x_ln2)

        return hidden_states, delta_used

    @torch.no_grad()
    def algorithmic_stats_for_batch(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """Collect geometric attribution stats along the full AKAZA trajectory."""
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

        old_mode = self.decomp_mode

        total_delta_energy = torch.tensor(0.0, device=input_ids.device)
        total_projection_energy = torch.tensor(0.0, device=input_ids.device)
        total_residual_energy = torch.tensor(0.0, device=input_ids.device)
        delta_projection_cos_sum = torch.tensor(0.0, device=input_ids.device)
        n_projection_cos = 0

        alpha_sum = torch.zeros(len(self.candidate_names), device=input_ids.device)
        alpha_abs_sum = torch.zeros(len(self.candidate_names), device=input_ids.device)
        cosine_sum = torch.zeros(len(self.candidate_names), device=input_ids.device)
        abs_cosine_sum = torch.zeros(len(self.candidate_names), device=input_ids.device)
        top_abs_cos_counts = torch.zeros(len(self.candidate_names), device=input_ids.device)
        top_abs_alpha_counts = torch.zeros(len(self.candidate_names), device=input_ids.device)
        n_token_layer = 0

        try:
            # Stats should describe the actual trained AKAZA trajectory.
            self.decomp_mode = "full"

            for i in range(first_edit, len(transformer.h)):
                if i in self.layer_set:
                    block = transformer.h[i]
                    residual, z_soft = self.attention_parts(block=block, hidden_states=hidden_states)
                    delta = self.compute_delta(layer_idx=i, block=block, hidden_states=hidden_states).to(z_soft.dtype)
                    directions = self.candidate_directions(
                        block=block,
                        hidden_states=hidden_states,
                        layer_idx=i,
                        z_soft=z_soft,
                    )
                    projection, alpha = ridge_project_delta(
                        delta,
                        directions,
                        ridge_lambda=self.ridge_lambda,
                        eps=self.eps,
                    )
                    residual_delta = delta - projection

                    d = delta.float()
                    p = projection.float()
                    r = residual_delta.float()
                    D = directions.float()

                    total_delta_energy = total_delta_energy + d.pow(2).sum()
                    total_projection_energy = total_projection_energy + p.pow(2).sum()
                    total_residual_energy = total_residual_energy + r.pow(2).sum()

                    dp_dot = (d * p).sum(dim=-1)
                    dp_den = d.pow(2).sum(dim=-1).sqrt().clamp_min(self.eps) * p.pow(2).sum(dim=-1).sqrt().clamp_min(self.eps)
                    delta_projection_cos_sum = delta_projection_cos_sum + (dp_dot / dp_den).sum()
                    n_projection_cos += int(dp_dot.numel())

                    dot = torch.einsum("btac,btc->bta", D, d)
                    D_norm = D.pow(2).sum(dim=-1).sqrt().clamp_min(self.eps)
                    d_norm = d.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
                    cos = dot / (D_norm * d_norm)

                    alpha_sum = alpha_sum + alpha.sum(dim=(0, 1))
                    alpha_abs_sum = alpha_abs_sum + alpha.abs().sum(dim=(0, 1))
                    cosine_sum = cosine_sum + cos.sum(dim=(0, 1))
                    abs_cosine_sum = abs_cosine_sum + cos.abs().sum(dim=(0, 1))

                    top_abs_cos = cos.abs().argmax(dim=-1)
                    top_abs_alpha = alpha.abs().argmax(dim=-1)
                    top_abs_cos_counts.scatter_add_(0, top_abs_cos.flatten(), torch.ones_like(top_abs_cos.flatten(), dtype=torch.float32))
                    top_abs_alpha_counts.scatter_add_(0, top_abs_alpha.flatten(), torch.ones_like(top_abs_alpha.flatten(), dtype=torch.float32))
                    n_token_layer += int(top_abs_cos.numel())

                    # Continue with full AKAZA hidden trajectory.
                    attn = block.attn
                    z_new = z_soft + delta
                    attn_output = attn.c_proj(z_new)
                    attn_output = attn.resid_dropout(attn_output)
                    hidden_states = residual + attn_output

                    residual2 = hidden_states
                    x_ln2 = block.ln_2(hidden_states)
                    hidden_states = residual2 + block.mlp(x_ln2)
                else:
                    hidden_states = block_forward_hidden(transformer.h[i], hidden_states)
        finally:
            self.decomp_mode = old_mode

        out: Dict[str, float] = {}
        if n_token_layer == 0:
            return out

        total_delta_energy = total_delta_energy.clamp_min(self.eps)
        out["alg_projection_energy_over_delta_energy"] = safe_float(total_projection_energy / total_delta_energy)
        out["alg_residual_energy_over_delta_energy"] = safe_float(total_residual_energy / total_delta_energy)
        out["delta_projection_cosine_mean"] = safe_float(delta_projection_cos_sum / max(1, n_projection_cos))

        denom = float(max(1, n_token_layer))
        for j, name in enumerate(self.candidate_names):
            out[f"candidate/{name}/alpha_mean"] = safe_float(alpha_sum[j] / denom)
            out[f"candidate/{name}/alpha_abs_mean"] = safe_float(alpha_abs_sum[j] / denom)
            out[f"candidate/{name}/cosine_mean"] = safe_float(cosine_sum[j] / denom)
            out[f"candidate/{name}/abs_cosine_mean"] = safe_float(abs_cosine_sum[j] / denom)
            out[f"candidate/{name}/top_abs_cosine_frac"] = safe_float(top_abs_cos_counts[j] / denom)
            out[f"candidate/{name}/top_abs_alpha_frac"] = safe_float(top_abs_alpha_counts[j] / denom)

        return out


@torch.no_grad()
def eval_algorithmic_decomp(
    wrapped: AlgorithmicDecomposedAKAZAFreeZModel,
    chunks: torch.Tensor,
    batch_size: int,
    *,
    mode: str,
    collect_algorithmic_stats: bool = False,
) -> Dict[str, float]:
    wrapped.set_decomp_mode(mode)
    wrapped.set_peft_eval_mode()

    losses: List[float] = []
    stats_accum: Dict[str, float] = {}
    n_stats_batches = 0
    n_examples = chunks.shape[0]

    for sl in range(0, n_examples, batch_size):
        input_ids = chunks[sl : min(sl + batch_size, n_examples)].to(wrapped.device)
        logits = wrapped(input_ids)
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])

        if collect_algorithmic_stats:
            stats = wrapped.algorithmic_stats_for_batch(input_ids)
            for k, v in stats.items():
                stats_accum[k] = stats_accum.get(k, 0.0) + float(v)
            n_stats_batches += 1

    out = {"loss": sum(losses) / max(1, n_examples)}
    if collect_algorithmic_stats:
        for k, v in stats_accum.items():
            out[k] = v / max(1, n_stats_batches)
    return out


def cfg_from_summary(summary: Dict[str, Any]) -> PEFTComparisonConfig:
    cfg_dict = dict(summary["config"])
    valid_fields = {f.name for f in fields(PEFTComparisonConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return PEFTComparisonConfig(**filtered)


def add_recovery_metrics(split_modes: Dict[str, Any]) -> None:
    baseline_loss = split_modes["baseline"]["loss"]
    full_imp = split_modes["full"]["improvement_nats_per_token"]
    denom = full_imp if abs(full_imp) > 1e-12 else float("nan")

    for mode in ["projection", "residual", "top1_abs_cosine"]:
        imp = split_modes[mode]["improvement_nats_per_token"]
        split_modes[mode]["gain_recovered_over_full_gain"] = imp / denom

    projection_imp = split_modes["projection"]["improvement_nats_per_token"]
    residual_imp = split_modes["residual"]["improvement_nats_per_token"]
    split_modes["gain_recovery_summary"] = {
        "projection_gain_over_full_gain": projection_imp / denom,
        "residual_gain_over_full_gain": residual_imp / denom,
        "projection_plus_residual_gain_over_full_gain": (projection_imp + residual_imp) / denom,
        "top1_abs_cosine_gain_over_full_gain": split_modes["top1_abs_cosine"]["improvement_nats_per_token"] / denom,
        "baseline_loss": baseline_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AKAZA algorithmic direction decomposition from a trained checkpoint."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="val_test", choices=["train", "val", "test", "val_test", "all"])
    parser.add_argument("--max_train_chunks", type=int, default=None)
    parser.add_argument("--max_val_chunks", type=int, default=None)
    parser.add_argument("--max_test_chunks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--candidate_learners", type=str, default=",".join(DEFAULT_CANDIDATE_LEARNERS))
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--temp_sharp", type=float, default=2.0)
    parser.add_argument("--temp_smooth", type=float, default=0.5)
    parser.add_argument("--recency_strength", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "summary" not in payload or "trainable_state_dict" not in payload:
        raise KeyError("Checkpoint must contain 'summary' and 'trainable_state_dict'.")

    summary = payload["summary"]
    if summary.get("method") != "akaza_freez":
        raise ValueError(f"Expected an akaza_freez checkpoint, got method={summary.get('method')!r}")

    cfg = cfg_from_summary(summary)
    cfg.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_train_chunks is not None:
        cfg.max_train_chunks = args.max_train_chunks
    if args.max_val_chunks is not None:
        cfg.max_val_chunks = args.max_val_chunks
    if args.max_test_chunks is not None:
        cfg.max_test_chunks = args.max_test_chunks

    device = torch.device(cfg.device)
    layer_indices = summary.get("layer_indices") or parse_int_csv(cfg.layer_indices)
    layer_indices = sorted(int(x) for x in layer_indices)
    candidate_names = parse_csv(args.candidate_learners)

    print("[loaded]")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  method: {summary.get('method')}")
    print(f"  best_epoch: {summary.get('best_epoch')}")
    print(f"  layer_indices: {layer_indices}")
    print(f"  bottleneck_dim: {cfg.bottleneck_dim}")
    print(f"  output_scale: {cfg.output_scale}")
    print(f"  adapter_input: {cfg.adapter_input}")
    print(f"  detach_adapter_input: {cfg.detach_adapter_input}")
    print(f"  candidate_learners: {candidate_names}")
    print(f"  ridge_lambda: {args.ridge_lambda}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = AlgorithmicDecomposedAKAZAFreeZModel(
        model=model,
        cfg=cfg,
        layer_indices=layer_indices,
        candidate_names=candidate_names,
        decomp_mode="full",
        ridge_lambda=args.ridge_lambda,
        eps=args.eps,
        window_size=args.window_size,
        top_k=args.top_k,
        temp_sharp=args.temp_sharp,
        temp_smooth=args.temp_smooth,
        recency_strength=args.recency_strength,
    ).to(device)
    load_trainable_state_dict(wrapped, payload["trainable_state_dict"])

    print()
    print("[data] loading chunks")
    chunks_by_split: Dict[str, torch.Tensor] = {}

    need_train = args.eval_split in {"train", "all"}
    need_val = args.eval_split in {"val", "val_test", "all"}
    need_test = args.eval_split in {"test", "val_test", "all"}

    if need_train:
        chunks_by_split["train"] = load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.train_split,
            max_texts=cfg.max_train_texts,
            max_chunks=cfg.max_train_chunks,
        )
    if need_val:
        chunks_by_split["val"] = load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.val_split,
            max_texts=cfg.max_val_texts,
            max_chunks=cfg.max_val_chunks,
        )
    if need_test:
        chunks_by_split["test"] = load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.test_split,
            max_texts=cfg.max_test_texts,
            max_chunks=cfg.max_test_chunks,
        )

    for split_name, chunks in chunks_by_split.items():
        print(f"  {split_name}: chunks={chunks.shape[0]} block_size={chunks.shape[1]}")

    results: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source_summary_best_epoch": summary.get("best_epoch"),
        "source_summary_best_val_loss": summary.get("best_val_loss"),
        "source_summary_best_test_loss": summary.get("best_test_loss"),
        "config": asdict(cfg),
        "layer_indices": layer_indices,
        "candidate_learners": candidate_names,
        "decomposition_config": {
            "ridge_lambda": args.ridge_lambda,
            "eps": args.eps,
            "window_size": args.window_size,
            "top_k": args.top_k,
            "temp_sharp": args.temp_sharp,
            "temp_smooth": args.temp_smooth,
            "recency_strength": args.recency_strength,
        },
        "modes": {},
    }

    print()
    print("[eval]")
    for split_name, chunks in chunks_by_split.items():
        baseline_loss = eval_baseline(model, chunks, cfg.batch_size, device)
        results["modes"].setdefault(split_name, {})
        results["modes"][split_name]["baseline"] = {
            "loss": baseline_loss,
            "ppl": math.exp(baseline_loss),
            "improvement_nats_per_token": 0.0,
            "relative_ppl_reduction": 0.0,
        }

        print(f"\n[{split_name}] baseline_loss={baseline_loss:.6f} ppl={math.exp(baseline_loss):.3f}")

        for mode in BASE_DECOMP_MODES:
            collect_stats = split_name in {"val", "test"} and mode == "full"
            metrics = eval_algorithmic_decomp(
                wrapped,
                chunks,
                cfg.batch_size,
                mode=mode,
                collect_algorithmic_stats=collect_stats,
            )
            loss = metrics["loss"]
            imp = baseline_loss - loss
            ppl_red = relative_ppl_reduction(imp)

            row = {
                "loss": loss,
                "ppl": math.exp(loss),
                "improvement_nats_per_token": imp,
                "relative_ppl_reduction": ppl_red,
                **{k: v for k, v in metrics.items() if k != "loss"},
            }
            results["modes"][split_name][mode] = row

            print(
                f"  mode={mode:16s} "
                f"loss={loss:.6f} "
                f"imp={imp:.6f} "
                f"ppl_red={100.0 * ppl_red:.3f}%"
            )

        add_recovery_metrics(results["modes"][split_name])
        rec = results["modes"][split_name]["gain_recovery_summary"]
        print(
            "  recovery: "
            f"projection/full={rec['projection_gain_over_full_gain']:.3f} "
            f"residual/full={rec['residual_gain_over_full_gain']:.3f} "
            f"sum/full={rec['projection_plus_residual_gain_over_full_gain']:.3f} "
            f"top1/full={rec['top1_abs_cosine_gain_over_full_gain']:.3f}"
        )

        stats = results["modes"][split_name].get("full", {})
        if "alg_projection_energy_over_delta_energy" in stats:
            print(
                "  geometric stats: "
                f"projection_energy/delta={100.0 * stats['alg_projection_energy_over_delta_energy']:.2f}% "
                f"residual_energy/delta={100.0 * stats['alg_residual_energy_over_delta_energy']:.2f}% "
                f"delta_projection_cos={stats['delta_projection_cosine_mean']:.4f}"
            )
            print("  top candidates by abs-cosine frequency:")
            freqs = []
            for name in candidate_names:
                key = f"candidate/{name}/top_abs_cosine_frac"
                if key in stats:
                    freqs.append((name, stats[key]))
            for name, frac in sorted(freqs, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {name:16s} {100.0 * frac:.2f}%")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print()
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
