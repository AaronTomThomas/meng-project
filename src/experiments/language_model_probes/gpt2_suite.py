#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.learners import predict_with_learner


LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]


# ============================================================
# Config
# ============================================================

@dataclass
class EvalConfig:
    L: int
    d: int
    dv: int
    batch_size: int
    sigma: float
    device: str
    beta_soft: float
    k_sharp: int
    k_linear_local: int
    ridge_lambda: float
    min_context: int
    window_size: int


@dataclass
class BenchConfig:
    # model/data
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "validation"
    text_field: str = "text"

    # packing
    max_texts: int = 200
    block_size: int = 96
    batch_size: int = 4
    max_chunks: int = 64

    # intervention target
    layer_idx: int = 4
    head_indices: str = "all"
    min_context: int = 16
    position_stride: int = 8

    # learner hyperparams
    beta_soft: float = 6.0
    k_sharp: int = 4
    window_size: int = 16
    k_linear_local: int = 16
    ridge_lambda: float = 1e-1

    # intervention mode
    replace_mode: str = "single_head_single_pos"
    # options:
    #   single_head_single_pos
    #   multi_head_single_pos_shared
    #   multi_head_single_pos_per_head

    head_group_size: int = 2
    head_group_strategy: str = "contiguous"
    # options:
    #   contiguous
    #   random
    #   manual

    manual_head_groups: str = ""
    # e.g. "0,1;2,3;4,5"

    max_head_groups: int = 0
    # 0 = no cap

    # misc
    seed: int = 0
    cache_dir: str = "outputs/head_counterfactual_cache"
    output_dir: str = "outputs/head_counterfactual_results"
    save_results: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

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


def cache_stem(cfg: BenchConfig) -> str:
    key = json.dumps(asdict(cfg), sort_keys=True)
    return short_hash(key)


def result_stem(cfg: BenchConfig) -> str:
    key = json.dumps(asdict(cfg), sort_keys=True)
    return short_hash(key)


def parse_head_indices(head_indices: str, n_heads: int) -> List[int]:
    if head_indices == "all":
        return list(range(n_heads))

    out = [int(x.strip()) for x in head_indices.split(",") if x.strip()]
    if not out:
        raise ValueError("Parsed empty head index list.")

    for h in out:
        if h < 0 or h >= n_heads:
            raise ValueError(f"Head index {h} out of range [0, {n_heads - 1}]")
    return out


def parse_manual_head_groups(s: str) -> List[List[int]]:
    groups = []
    s = s.strip()
    if not s:
        return groups
    for grp in s.split(";"):
        g = [int(x.strip()) for x in grp.split(",") if x.strip()]
        if not g:
            continue
        groups.append(g)
    return groups


def build_eval_cfg(cfg: BenchConfig, head_dim: int) -> EvalConfig:
    return EvalConfig(
        L=cfg.block_size,
        d=head_dim,
        dv=head_dim,
        batch_size=1,
        sigma=0.0,
        device=cfg.device,
        beta_soft=cfg.beta_soft,
        k_sharp=cfg.k_sharp,
        k_linear_local=cfg.k_linear_local,
        ridge_lambda=cfg.ridge_lambda,
        min_context=cfg.min_context,
        window_size=cfg.window_size,
    )


def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    # x: (B, L, D)
    B, L, D = x.shape
    assert D == num_heads * head_dim
    return x.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, Dh)
    B, H, L, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)


def causal_soft_attention_from_qkv(
    q: torch.Tensor,  # (B,H,L,Dh)
    k: torch.Tensor,  # (B,H,L,Dh)
    v: torch.Tensor,  # (B,H,L,Dh)
) -> torch.Tensor:
    _, _, L, Dh = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)
    causal = torch.triu(
        torch.ones(L, L, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    z = torch.matmul(attn, v)
    return z


# ============================================================
# Head group construction
# ============================================================

def generate_head_groups(
    head_indices: List[int],
    group_size: int,
    strategy: str,
    manual_head_groups: str,
    max_head_groups: int,
    seed: int,
) -> List[List[int]]:
    if strategy == "manual":
        groups = parse_manual_head_groups(manual_head_groups)
        if not groups:
            raise ValueError("head_group_strategy=manual but no manual_head_groups were provided.")
        for g in groups:
            for h in g:
                if h not in head_indices:
                    raise ValueError(f"Manual group head {h} not in selected head_indices={head_indices}")
        return groups

    if group_size <= 0:
        raise ValueError("head_group_size must be positive.")

    if group_size > len(head_indices):
        raise ValueError(
            f"head_group_size={group_size} > number of selected heads={len(head_indices)}"
        )

    if strategy == "contiguous":
        groups = []
        for i in range(0, len(head_indices), group_size):
            g = head_indices[i:i + group_size]
            if len(g) == group_size:
                groups.append(g)
    elif strategy == "random":
        rng = random.Random(seed)
        shuffled = head_indices[:]
        rng.shuffle(shuffled)
        groups = []
        for i in range(0, len(shuffled), group_size):
            g = shuffled[i:i + group_size]
            if len(g) == group_size:
                groups.append(sorted(g))
    else:
        raise ValueError(f"Unknown head_group_strategy={strategy}")

    if max_head_groups > 0:
        groups = groups[:max_head_groups]

    if not groups:
        raise ValueError("No head groups were generated.")

    return groups


# ============================================================
# Data loading / packing
# ============================================================

def load_and_pack_texts(cfg: BenchConfig, tokenizer) -> torch.Tensor:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem_key = (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.split}|"
        f"text={cfg.text_field}|max_texts={cfg.max_texts}|block={cfg.block_size}|"
        f"max_chunks={cfg.max_chunks}"
    )
    cache_path = cache_dir / f"{short_hash(stem_key)}__chunks.pt"

    if cache_path.exists():
        print(f"[cache] loading chunks from {cache_path}")
        return torch.load(cache_path)

    print("[data] loading dataset...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)

    token_blocks = []
    total_texts = 0

    print("[data] tokenizing and packing texts...")
    for ex in ds:
        text = ex[cfg.text_field]
        if not isinstance(text, str) or not text.strip():
            continue

        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]

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


# ============================================================
# GPT-2 block extraction helpers
# ============================================================

@torch.no_grad()
def get_input_embeddings_gpt2(model, input_ids: torch.Tensor) -> torch.Tensor:
    device = input_ids.device
    _, seqlen = input_ids.shape

    transformer = model.transformer
    pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    tok_emb = transformer.wte(input_ids)
    pos_emb = transformer.wpe(pos_ids)
    hidden_states = tok_emb + pos_emb
    hidden_states = transformer.drop(hidden_states)
    return hidden_states


@torch.no_grad()
def run_to_block_input_gpt2(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
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
def cache_block_tensors(model, chunks: torch.Tensor, cfg: BenchConfig) -> Dict[str, torch.Tensor]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem_key = (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.split}|"
        f"block={cfg.block_size}|max_chunks={cfg.max_chunks}|"
        f"layer={cfg.layer_idx}|device_independent=true"
    )
    cache_path = cache_dir / f"{short_hash(stem_key)}__layer{cfg.layer_idx}__block_tensors.pt"

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

        x_in = run_to_block_input_gpt2(model, batch_input_ids, cfg.layer_idx)
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
            print(f"[cache-build] batch {batch_idx + 1}/{n_batches} chunks {batch_ids[0]}..{batch_ids[-1]}")

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


# ============================================================
# Continue model after intervention
# ============================================================

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
    logits = model.lm_head(x)
    return logits


# ============================================================
# Learner prediction helpers
# ============================================================

@torch.no_grad()
def predict_head_output_for_all_learners(
    q: torch.Tensor,  # (1,H,L,Dh)
    k: torch.Tensor,  # (1,H,L,Dh)
    v: torch.Tensor,  # (1,H,L,Dh)
    head_idx: int,
    pos: int,
    eval_cfg: EvalConfig,
) -> Dict[str, torch.Tensor]:
    q_t = q[:, head_idx, pos, :]         # (1,Dh)
    Kctx = k[:, head_idx, :pos + 1, :]   # (1,T,Dh)
    Vctx = v[:, head_idx, :pos + 1, :]   # (1,T,Dh)

    out = {}
    for learner in LEARNERS:
        pred = predict_with_learner(learner, q_t, Kctx, Vctx, eval_cfg)  # (1,Dh)
        out[learner] = pred[0]
    return out


# ============================================================
# Intervention evaluation
# ============================================================

@torch.no_grad()
def evaluate_single_head_single_pos(
    model,
    input_ids: torch.Tensor,
    x_in: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    zcat_teacher: torch.Tensor,
    block,
    attn_module,
    layer_idx: int,
    head_idx: int,
    pos: int,
    cfg: BenchConfig,
) -> Dict[str, float]:
    head_dim = attn_module.head_dim
    eval_cfg = build_eval_cfg(cfg, head_dim)

    pred_map = predict_head_output_for_all_learners(q, k, v, head_idx, pos, eval_cfg)

    M = len(LEARNERS)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    for m, learner in enumerate(LEARNERS):
        zcat_rep[m, pos, start:end] = pred_map[learner]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_rep,
        layer_idx=layer_idx,
    )

    target_next = input_ids[0, pos + 1].repeat(M)
    log_probs = F.log_softmax(logits[:, pos, :], dim=-1)
    nll = -log_probs[torch.arange(M, device=logits.device), target_next]

    return {learner: nll[m].item() for m, learner in enumerate(LEARNERS)}


@torch.no_grad()
def evaluate_multi_head_single_pos_shared(
    model,
    input_ids: torch.Tensor,
    x_in: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    zcat_teacher: torch.Tensor,
    block,
    attn_module,
    layer_idx: int,
    head_group: Sequence[int],
    pos: int,
    cfg: BenchConfig,
) -> Dict[str, float]:
    head_dim = attn_module.head_dim
    eval_cfg = build_eval_cfg(cfg, head_dim)

    pred_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    for head_idx in head_group:
        pred_map = predict_head_output_for_all_learners(q, k, v, head_idx, pos, eval_cfg)
        for learner, pred in pred_map.items():
            pred_cache[(head_idx, learner)] = pred

    M = len(LEARNERS)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    for m, learner in enumerate(LEARNERS):
        for head_idx in head_group:
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            zcat_rep[m, pos, start:end] = pred_cache[(head_idx, learner)]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_rep,
        layer_idx=layer_idx,
    )

    target_next = input_ids[0, pos + 1].repeat(M)
    log_probs = F.log_softmax(logits[:, pos, :], dim=-1)
    nll = -log_probs[torch.arange(M, device=logits.device), target_next]

    return {learner: nll[m].item() for m, learner in enumerate(LEARNERS)}


@torch.no_grad()
def evaluate_multi_head_single_pos_per_head(
    model,
    input_ids: torch.Tensor,
    x_in: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    zcat_teacher: torch.Tensor,
    block,
    attn_module,
    layer_idx: int,
    head_group: Sequence[int],
    pos: int,
    cfg: BenchConfig,
) -> Dict[str, float]:
    head_dim = attn_module.head_dim
    eval_cfg = build_eval_cfg(cfg, head_dim)

    pred_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    for head_idx in head_group:
        pred_map = predict_head_output_for_all_learners(q, k, v, head_idx, pos, eval_cfg)
        for learner, pred in pred_map.items():
            pred_cache[(head_idx, learner)] = pred

    assignments = list(product(LEARNERS, repeat=len(head_group)))
    assignment_names = ["+".join(a) for a in assignments]

    M = len(assignments)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    for m, assignment in enumerate(assignments):
        for head_idx, learner in zip(head_group, assignment):
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            zcat_rep[m, pos, start:end] = pred_cache[(head_idx, learner)]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_rep,
        layer_idx=layer_idx,
    )

    target_next = input_ids[0, pos + 1].repeat(M)
    log_probs = F.log_softmax(logits[:, pos, :], dim=-1)
    nll = -log_probs[torch.arange(M, device=logits.device), target_next]

    return {assignment_names[m]: nll[m].item() for m in range(M)}


# ============================================================
# Main benchmark loop
# ============================================================

@torch.no_grad()
def run_benchmark(cfg: BenchConfig) -> Dict[str, object]:
    print("[setup] loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("This script currently supports GPT-2 style models only.")

    print("[setup] loading and packing text...")
    chunks = load_and_pack_texts(cfg, tokenizer).to(cfg.device)
    cached = cache_block_tensors(model, chunks, cfg)

    n_heads = cached["q"].shape[1]
    head_indices = parse_head_indices(cfg.head_indices, n_heads)

    dummy_x = cached["x_in"][:1].to(cfg.device)
    _, _, _, _, _, _, block, attn_module = extract_head_qkv_and_teacher_outputs_gpt2(
        model, dummy_x, cfg.layer_idx
    )

    mode = cfg.replace_mode
    if mode == "single_head_single_pos":
        intervention_units = [[h] for h in head_indices]
    elif mode in ("multi_head_single_pos_shared", "multi_head_single_pos_per_head"):
        intervention_units = generate_head_groups(
            head_indices=head_indices,
            group_size=cfg.head_group_size,
            strategy=cfg.head_group_strategy,
            manual_head_groups=cfg.manual_head_groups,
            max_head_groups=cfg.max_head_groups,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown replace_mode={mode}")

    print(f"[setup] total chunks={chunks.shape[0]}, layer={cfg.layer_idx}, mode={mode}")
    print(f"[setup] selected heads={head_indices}")
    print(f"[setup] intervention units={intervention_units[:10]}")
    if len(intervention_units) > 10:
        print(f"[setup] ... ({len(intervention_units)} total units)")

    losses_all = []
    unit_labels_all = []
    pos_ids_all = []

    total_examples = 0
    for chunk_id in range(chunks.shape[0]):
        input_ids_1 = chunks[chunk_id:chunk_id + 1]
        x_in_1 = cached["x_in"][chunk_id:chunk_id + 1].to(cfg.device)
        q_1 = cached["q"][chunk_id:chunk_id + 1].to(cfg.device)
        k_1 = cached["k"][chunk_id:chunk_id + 1].to(cfg.device)
        v_1 = cached["v"][chunk_id:chunk_id + 1].to(cfg.device)
        zcat_teacher_1 = cached["zcat_teacher"][chunk_id:chunk_id + 1].to(cfg.device)

        L = input_ids_1.shape[1]

        for pos in range(cfg.min_context, L - 1, cfg.position_stride):
            for unit in intervention_units:
                if mode == "single_head_single_pos":
                    losses_dict = evaluate_single_head_single_pos(
                        model=model,
                        input_ids=input_ids_1,
                        x_in=x_in_1,
                        q=q_1,
                        k=k_1,
                        v=v_1,
                        zcat_teacher=zcat_teacher_1,
                        block=block,
                        attn_module=attn_module,
                        layer_idx=cfg.layer_idx,
                        head_idx=unit[0],
                        pos=pos,
                        cfg=cfg,
                    )
                    candidate_names = LEARNERS
                    unit_label = f"head={unit[0]}"
                elif mode == "multi_head_single_pos_shared":
                    losses_dict = evaluate_multi_head_single_pos_shared(
                        model=model,
                        input_ids=input_ids_1,
                        x_in=x_in_1,
                        q=q_1,
                        k=k_1,
                        v=v_1,
                        zcat_teacher=zcat_teacher_1,
                        block=block,
                        attn_module=attn_module,
                        layer_idx=cfg.layer_idx,
                        head_group=unit,
                        pos=pos,
                        cfg=cfg,
                    )
                    candidate_names = LEARNERS
                    unit_label = f"group={','.join(map(str, unit))}"
                elif mode == "multi_head_single_pos_per_head":
                    losses_dict = evaluate_multi_head_single_pos_per_head(
                        model=model,
                        input_ids=input_ids_1,
                        x_in=x_in_1,
                        q=q_1,
                        k=k_1,
                        v=v_1,
                        zcat_teacher=zcat_teacher_1,
                        block=block,
                        attn_module=attn_module,
                        layer_idx=cfg.layer_idx,
                        head_group=unit,
                        pos=pos,
                        cfg=cfg,
                    )
                    candidate_names = list(losses_dict.keys())
                    unit_label = f"group={','.join(map(str, unit))}"
                else:
                    raise RuntimeError("Unreachable mode branch.")

                losses_vec = torch.tensor(
                    [[losses_dict[name] for name in candidate_names]],
                    dtype=torch.float32,
                )

                losses_all.append(losses_vec)
                unit_labels_all.append(unit_label)
                pos_ids_all.append(pos)
                total_examples += 1

        if chunk_id % 8 == 0 or chunk_id == chunks.shape[0] - 1:
            print(f"[bench] chunk {chunk_id + 1}/{chunks.shape[0]} examples so far={total_examples}")

    return {
        "config": asdict(cfg),
        "losses": torch.cat(losses_all, dim=0),
        "unit_labels": unit_labels_all,
        "pos_ids": torch.tensor(pos_ids_all, dtype=torch.long),
        "candidate_names": candidate_names,
    }


# ============================================================
# Summaries
# ============================================================

def summarize(results: Dict[str, object]) -> None:
    losses: torch.Tensor = results["losses"]
    unit_labels: List[str] = results["unit_labels"]
    pos_ids: torch.Tensor = results["pos_ids"]
    candidate_names: List[str] = results["candidate_names"]

    mean_losses = losses.mean(dim=0)
    oracle = losses.min(dim=-1).values.mean().item()
    best_fixed = mean_losses.min().item()
    winner_idx = mean_losses.argmin().item()
    winners = losses.argmin(dim=-1)

    print("\n=== Fixed candidate benchmark ===")
    for i, name in enumerate(candidate_names):
        print(f"  {name:<32} {mean_losses[i].item():.6f}")
    print(f"  {'best_fixed':<32} {best_fixed:.6f} ({candidate_names[winner_idx]})")
    print(f"  {'oracle':<32} {oracle:.6f}")
    print(f"  {'oracle_gap':<32} {best_fixed - oracle:.6f}")

    print("\n=== Winner fractions ===")
    for i, name in enumerate(candidate_names):
        frac = (winners == i).float().mean().item()
        print(f"  {name:<32} {frac:.3f}")

    sorted_losses, _ = torch.sort(losses, dim=-1)
    margins = sorted_losses[:, 1] - sorted_losses[:, 0]

    print("\n=== Oracle margin stats ===")
    print(f"  {'mean margin':<32} {margins.mean().item():.6f}")
    print(f"  {'median margin':<32} {margins.median().item():.6f}")
    print(f"  {'frac margin < 1e-4':<32} {(margins < 1e-4).float().mean().item():.4f}")
    print(f"  {'frac margin < 1e-3':<32} {(margins < 1e-3).float().mean().item():.4f}")
    print(f"  {'frac margin < 1e-2':<32} {(margins < 1e-2).float().mean().item():.4f}")

    print("\n=== Winner fractions by unit (first 12) ===")
    unique_units = sorted(set(unit_labels))
    for unit in unique_units[:12]:
        mask = torch.tensor([u == unit for u in unit_labels], dtype=torch.bool)
        winners_u = winners[mask]
        stats = {
            name: (winners_u == i).float().mean().item()
            for i, name in enumerate(candidate_names)
        }
        stats_str = "  ".join(f"{k}:{v:.2f}" for k, v in stats.items())
        print(f"  {unit:<20} {stats_str}")

    print("\n=== Winner fractions by position (first 10 positions present) ===")
    for pos in sorted(torch.unique(pos_ids).tolist())[:10]:
        mask = pos_ids == pos
        winners_p = winners[mask]
        stats = {
            name: (winners_p == i).float().mean().item()
            for i, name in enumerate(candidate_names)
        }
        stats_str = "  ".join(f"{k}:{v:.2f}" for k, v in stats.items())
        print(f"  pos={pos:>3}  {stats_str}")


def save_results(results: Dict[str, object], cfg: BenchConfig) -> None:
    if not cfg.save_results:
        return

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = result_stem(cfg)
    path = output_dir / f"{stem}__results.pt"
    torch.save(results, path)
    print(f"\n[save] saved results to {path}")


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # model/data
    p.add_argument("--model_name", type=str, default="openai-community/gpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--text_field", type=str, default="text")

    # packing
    p.add_argument("--max_texts", type=int, default=200)
    p.add_argument("--block_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_chunks", type=int, default=64)

    # intervention target
    p.add_argument("--layer_idx", type=int, default=4)
    p.add_argument("--head_indices", type=str, default="all")
    p.add_argument("--min_context", type=int, default=16)
    p.add_argument("--position_stride", type=int, default=8)

    # learner hyperparams
    p.add_argument("--beta_soft", type=float, default=6.0)
    p.add_argument("--k_sharp", type=int, default=4)
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--k_linear_local", type=int, default=16)
    p.add_argument("--ridge_lambda", type=float, default=1e-1)

    # intervention mode
    p.add_argument(
        "--replace_mode",
        type=str,
        default="single_head_single_pos",
        choices=[
            "single_head_single_pos",
            "multi_head_single_pos_shared",
            "multi_head_single_pos_per_head",
        ],
    )
    p.add_argument("--head_group_size", type=int, default=2)
    p.add_argument(
        "--head_group_strategy",
        type=str,
        default="contiguous",
        choices=["contiguous", "random", "manual"],
    )
    p.add_argument("--manual_head_groups", type=str, default="")
    p.add_argument("--max_head_groups", type=int, default=0)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default="outputs/head_counterfactual_cache")
    p.add_argument("--output_dir", type=str, default="outputs/head_counterfactual_results")
    p.add_argument("--save_results", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = BenchConfig(**vars(args))
    set_seed(cfg.seed)

    print("[config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    if cfg.replace_mode == "multi_head_single_pos_per_head":
        n_assignments = len(LEARNERS) ** cfg.head_group_size
        print(f"[note] per-head mode will evaluate {n_assignments} assignments per example")

    results = run_benchmark(cfg)
    summarize(results)
    save_results(results, cfg)

    print("\n=== Interpretation guide ===")
    print("1) If oracle beats best_fixed by a nontrivial margin, there is headroom for adaptive selection.")
    print("2) In multi_head_single_pos_shared, the oracle asks whether one learner suits a whole head-group at a token.")
    print("3) In multi_head_single_pos_per_head, the oracle asks whether different heads in the same group want different learners.")
    print("4) If per-head oracle beats shared oracle by a lot, that is stronger evidence for intra-token heterogeneity across heads.")


if __name__ == "__main__":
    main()