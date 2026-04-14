"""Shared helpers for GPT-2 attention intervention experiments."""

from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset

from experiments.language_model_probes.probe_utils import (
    causal_soft_attention_from_qkv,
    first_tensor,
    merge_heads,
    short_hash,
    split_heads,
)


@torch.no_grad()
def get_input_embeddings_gpt2(model, input_ids: torch.Tensor) -> torch.Tensor:
    device = input_ids.device
    _, seqlen = input_ids.shape
    transformer = model.transformer
    pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    tok_emb = transformer.wte(input_ids)
    pos_emb = transformer.wpe(pos_ids)
    hidden_states = transformer.drop(tok_emb + pos_emb)
    return hidden_states


@torch.no_grad()
def get_block_input_gpt2(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    x = get_input_embeddings_gpt2(model, input_ids)
    blocks = model.transformer.h
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"layer_idx={layer_idx} invalid for {len(blocks)} blocks")
    for l in range(layer_idx):
        x = first_tensor(blocks[l](x, use_cache=False))
    return x


@torch.no_grad()
def extract_head_qkv_and_teacher_outputs_gpt2(model, x_in: torch.Tensor, layer_idx: int):
    block = model.transformer.h[layer_idx]
    attn_module = block.attn
    h_ln1 = block.ln_1(x_in)
    qkv = attn_module.c_attn(h_ln1)
    split_size = attn_module.split_size
    q_raw, k_raw, v_raw = qkv.split(split_size, dim=2)

    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    q = split_heads(q_raw, num_heads, head_dim)
    k = split_heads(k_raw, num_heads, head_dim)
    v = split_heads(v_raw, num_heads, head_dim)
    z_teacher = causal_soft_attention_from_qkv(q, k, v)
    zcat_teacher = merge_heads(z_teacher)
    return h_ln1, q, k, v, z_teacher, zcat_teacher, block, attn_module


@torch.no_grad()
def continue_from_modified_block_gpt2(
    model,
    block,
    x_in: torch.Tensor,
    zcat_mod: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    attn_out = block.attn.c_proj(zcat_mod)
    attn_out = block.attn.resid_dropout(attn_out)
    x = x_in + attn_out
    residual = x
    x_ln2 = block.ln_2(x)
    mlp_out = block.mlp(x_ln2)
    x = residual + mlp_out
    for l in range(layer_idx + 1, len(model.transformer.h)):
        x = first_tensor(model.transformer.h[l](x, use_cache=False))
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def _text_cache_key(cfg, text_field: str) -> str:
    return (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.split}|"
        f"text={text_field}|max_texts={cfg.max_texts}|block={cfg.block_size}|"
        f"max_chunks={cfg.max_chunks}"
    )


def _block_cache_key(cfg) -> str:
    return (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.split}|"
        f"block={cfg.block_size}|max_chunks={cfg.max_chunks}|layer={cfg.layer_idx}"
    )


def load_and_pack_texts(cfg, tokenizer, text_field: str | None = None) -> torch.Tensor:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_field = text_field or getattr(cfg, "text_field", "text")

    cache_path = cache_dir / f"{short_hash(_text_cache_key(cfg, text_field))}__chunks.pt"
    if cache_path.exists():
        print(f"[cache] loading chunks from {cache_path}")
        return torch.load(cache_path)

    print("[data] loading dataset...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    token_blocks = []
    total_texts = 0
    print("[data] tokenizing and packing texts...")
    for ex in ds:
        text = ex[text_field]
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.numel() < cfg.block_size:
            continue
        n_blocks = ids.numel() // cfg.block_size
        ids = ids[: n_blocks * cfg.block_size].view(n_blocks, cfg.block_size)
        token_blocks.append(ids)
        total_texts += 1
        if total_texts >= cfg.max_texts:
            break
    if not token_blocks:
        raise ValueError("No usable token blocks found.")
    chunks = torch.cat(token_blocks, dim=0)[: cfg.max_chunks]
    print(f"[data] packed {chunks.shape[0]} chunks from {total_texts} texts")
    torch.save(chunks.cpu(), cache_path)
    return chunks


@torch.no_grad()
def run_to_block_and_cache_tensors(model, chunks: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{short_hash(_block_cache_key(cfg))}__layer{cfg.layer_idx}__block_tensors.pt"
    )
    if cache_path.exists():
        print(f"[cache] loading block tensors from {cache_path}")
        return torch.load(cache_path)

    print("[cache-build] extracting block tensors...")
    n_chunks = chunks.shape[0]
    n_batches = math.ceil(n_chunks / cfg.batch_size)
    x_in_all = []
    q_all = []
    k_all = []
    v_all = []
    z_teacher_all = []
    zcat_teacher_all = []
    for batch_idx, start in enumerate(range(0, n_chunks, cfg.batch_size)):
        batch_ids = list(range(start, min(start + cfg.batch_size, n_chunks)))
        batch_input_ids = chunks[batch_ids].to(cfg.device)
        x_in = get_block_input_gpt2(model, batch_input_ids, cfg.layer_idx)
        _, q, k, v, z_teacher, zcat_teacher, _, _ = extract_head_qkv_and_teacher_outputs_gpt2(
            model, x_in, cfg.layer_idx
        )
        x_in_all.append(x_in.cpu())
        q_all.append(q.cpu())
        k_all.append(k.cpu())
        v_all.append(v.cpu())
        z_teacher_all.append(z_teacher.cpu())
        zcat_teacher_all.append(zcat_teacher.cpu())
        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            print(
                f"[cache-build] batch {batch_idx + 1}/{n_batches} chunks {batch_ids[0]}..{batch_ids[-1]}"
            )
    out = {
        "x_in": torch.cat(x_in_all, dim=0),
        "q": torch.cat(q_all, dim=0),
        "k": torch.cat(k_all, dim=0),
        "v": torch.cat(v_all, dim=0),
        "z_teacher": torch.cat(z_teacher_all, dim=0),
        "zcat_teacher": torch.cat(zcat_teacher_all, dim=0),
    }
    torch.save(out, cache_path)
    print(f"[cache] saved block tensors to {cache_path}")
    return out


def parse_head_indices(head_indices: str, n_heads: int) -> List[int]:
    if head_indices == "all":
        return list(range(n_heads))
    out = [int(x.strip()) for x in head_indices.split(",") if x.strip()]
    if not out:
        raise ValueError("Parsed empty head index list")
    for h in out:
        if h < 0 or h >= n_heads:
            raise ValueError(f"Head index {h} out of range [0, {n_heads - 1}]")
    return out


def parse_manual_head_groups(s: str) -> List[List[int]]:
    s = s.strip()
    if not s:
        return []
    groups: List[List[int]] = []
    for chunk in s.split(";"):
        grp = [int(x.strip()) for x in chunk.split(",") if x.strip()]
        if grp:
            groups.append(grp)
    return groups


def build_head_groups(
    selected_heads: List[int],
    group_size: int,
    strategy: str,
    manual_head_groups: str,
    max_head_groups: int,
    seed: int | None = None,
) -> List[List[int]]:
    if strategy == "manual":
        groups = parse_manual_head_groups(manual_head_groups)
        if not groups:
            raise ValueError("manual_head_groups required when strategy=manual")
        selected = set(selected_heads)
        for grp in groups:
            for h in grp:
                if h not in selected:
                    raise ValueError(
                        f"Manual head {h} not present in selected head set {selected_heads}"
                    )
    else:
        if group_size <= 0:
            raise ValueError("head_group_size must be positive")
        if group_size > len(selected_heads):
            raise ValueError(
                f"head_group_size={group_size} > number of selected heads={len(selected_heads)}"
            )
        if strategy == "contiguous":
            groups = [
                selected_heads[i : i + group_size]
                for i in range(0, len(selected_heads), group_size)
                if len(selected_heads[i : i + group_size]) == group_size
            ]
        elif strategy == "random":
            rng = random.Random(seed)
            shuffled = selected_heads[:]
            rng.shuffle(shuffled)
            groups = [
                sorted(shuffled[i : i + group_size])
                for i in range(0, len(shuffled), group_size)
                if len(shuffled[i : i + group_size]) == group_size
            ]
        else:
            raise ValueError(f"Unknown head_group_strategy={strategy}")
    if max_head_groups > 0:
        groups = groups[:max_head_groups]
    if not groups:
        raise ValueError("No head groups were generated")
    return groups


def head_slice(head_idx: int, head_dim: int) -> slice:
    return slice(head_idx * head_dim, (head_idx + 1) * head_dim)


def mean_next_token_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    if not positions:
        raise ValueError("positions must be non-empty")
    device = logits.device
    pos_t = torch.tensor(list(positions), device=device, dtype=torch.long)
    target_t = pos_t + 1
    log_probs = F.log_softmax(logits[:, pos_t, :], dim=-1)
    targets = input_ids[:, target_t]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean(dim=-1)


def build_candidate_assignments(
    replace_mode: str,
    learner_names: Sequence[str],
    group_size: int,
) -> List[Tuple[str, ...]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if replace_mode == "multi_head_single_pos_shared":
        return [(learner,) * group_size for learner in learner_names]
    if replace_mode == "multi_head_single_pos_per_head":
        return list(itertools.product(learner_names, repeat=group_size))
    raise ValueError(f"Unsupported replace_mode={replace_mode}")


def candidate_name(assign: Sequence[str]) -> str:
    return "+".join(assign)


def add_shared_cli_args(
    parser: argparse.ArgumentParser,
    *,
    include_text_field: bool,
    replace_mode_choices: Sequence[str],
    default_replace_mode: str,
    default_position_stride: int,
    head_group_strategy_choices: Sequence[str],
) -> argparse.ArgumentParser:
    """Attach shared CLI options for GPT-2 probe scripts."""

    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    if include_text_field:
        parser.add_argument("--text_field", type=str, default="text")

    parser.add_argument("--max_texts", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=64)

    parser.add_argument("--layer_idx", type=int, default=4)
    parser.add_argument("--head_indices", type=str, default="all")
    parser.add_argument("--min_context", type=int, default=16)
    parser.add_argument("--position_stride", type=int, default=default_position_stride)

    parser.add_argument(
        "--replace_mode",
        type=str,
        default=default_replace_mode,
        choices=replace_mode_choices,
    )
    parser.add_argument("--head_group_size", type=int, default=2)
    parser.add_argument(
        "--head_group_strategy",
        type=str,
        default=head_group_strategy_choices[0],
        choices=head_group_strategy_choices,
    )
    parser.add_argument("--manual_head_groups", type=str, default="")
    parser.add_argument("--max_head_groups", type=int, default=0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="outputs/head_counterfactual_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/head_counterfactual_results")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser
