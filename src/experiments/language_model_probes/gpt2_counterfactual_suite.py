#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.synthetic_alignment.config import EvalConfig
from experiments.learners import predict_with_learner

LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]

def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    # x: (B, L, D)
    B, L, D = x.shape
    assert D == num_heads * head_dim
    return x.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()  # (B,H,L,Dh)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, Dh)
    B, H, L, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)  # (B,L,D)


def causal_soft_attention_from_qkv(
    q: torch.Tensor,   # (B,H,L,Dh)
    k: torch.Tensor,   # (B,H,L,Dh)
    v: torch.Tensor,   # (B,H,L,Dh)
) -> torch.Tensor:
    """
    Vanilla causal soft attention in head space.
    Returns:
      z: (B,H,L,Dh)
    """
    B, H, L, Dh = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)   # (B,H,L,L)

    causal = torch.triu(
        torch.ones(L, L, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)   # (B,H,L,L)
    z = torch.matmul(attn, v)              # (B,H,L,Dh)
    return z


@dataclass
class BenchConfig:
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "validation"
    text_field: str = "text"

    max_texts: int = 200
    block_size: int = 96
    batch_size: int = 4
    max_chunks: int = 64

    layer_idx: int = 4
    head_indices: str = "all"
    min_context: int = 16
    position_stride: int = 8

    beta_soft: float = 6.0
    k_sharp: int = 4
    window_size: int = 16
    k_linear_local: int = 16
    ridge_lambda: float = 1e-1

    seed: int = 0
    cache_dir: str = "outputs/head_counterfactual_cache"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
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
    key = (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.split}|"
        f"maxtexts={cfg.max_texts}|block={cfg.block_size}|maxchunks={cfg.max_chunks}|"
        f"layer={cfg.layer_idx}|heads={cfg.head_indices}|"
        f"minctx={cfg.min_context}|stride={cfg.position_stride}|"
        f"beta={cfg.beta_soft}|ksharp={cfg.k_sharp}|"
        f"window={cfg.window_size}|klin={cfg.k_linear_local}|ridge={cfg.ridge_lambda}|"
        f"seed={cfg.seed}"
    )
    return short_hash(key)


def parse_head_indices(head_indices: str, n_heads: int) -> List[int]:
    if head_indices == "all":
        return list(range(n_heads))
    out = [int(x.strip()) for x in head_indices.split(",") if x.strip()]
    for h in out:
        if h < 0 or h >= n_heads:
            raise ValueError(f"Head index {h} out of range [0, {n_heads-1}]")
    return out


def make_eval_cfg(cfg: BenchConfig, head_dim: int) -> EvalConfig:
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


def load_and_pack_texts(cfg: BenchConfig, tokenizer) -> torch.Tensor:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_stem(cfg)
    cache_path = cache_dir / f"{stem}__chunks.pt"

    if cache_path.exists():
        print(f"[cache] loading chunks from {cache_path}")
        return torch.load(cache_path)

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)

    token_blocks = []
    total_texts = 0

    print("[data] tokenizing and packing texts...")
    for ex in ds:
        text = ex[cfg.text_field]
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

    chunks = torch.cat(token_blocks, dim=0)
    chunks = chunks[: cfg.max_chunks]
    print(f"[data] packed {chunks.shape[0]} chunks from {total_texts} texts")
    torch.save(chunks.cpu(), cache_path)
    return chunks


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

    h_ln1 = block.ln_1(x_in)  # (B,L,D)
    qkv = attn_module.c_attn(h_ln1)  # (B,L,3D)
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
def cache_block_tensors(model, chunks: torch.Tensor, cfg: BenchConfig):
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_stem(cfg)
    cache_path = cache_dir / f"{stem}__layer{cfg.layer_idx}__block_tensors.pt"

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
            print(f"[cache-build] batch {batch_idx+1}/{n_batches} chunks {batch_ids[0]}..{batch_ids[-1]}")

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


@torch.no_grad()
def continue_from_modified_block_gpt2(model, block, x_in: torch.Tensor, zcat_mod: torch.Tensor, layer_idx: int):
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


@torch.no_grad()
def evaluate_query_head(
    model,
    input_ids: torch.Tensor,
    x_in: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z_teacher: torch.Tensor,
    zcat_teacher: torch.Tensor,
    block,
    attn_module,
    layer_idx: int,
    head_idx: int,
    pos: int,
    cfg: BenchConfig,
):
    head_dim = attn_module.head_dim
    eval_cfg = make_eval_cfg(cfg, head_dim)

    q_t = q[:, head_idx, pos, :]
    Kctx = k[:, head_idx, :pos + 1, :]
    Vctx = v[:, head_idx, :pos + 1, :]

    learner_preds = {}
    for learner in LEARNERS:
        learner_preds[learner] = predict_with_learner(learner, q_t, Kctx, Vctx, eval_cfg)

    M = len(LEARNERS)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    for m, learner in enumerate(LEARNERS):
        zcat_rep[m, pos, start:end] = learner_preds[learner][0]

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
def run_benchmark(cfg: BenchConfig):
    print("[setup] loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("This script is GPT-2 style specific.")

    print("[setup] loading and packing text...")
    chunks = load_and_pack_texts(cfg, tokenizer).to(cfg.device)
    cached = cache_block_tensors(model, chunks, cfg)

    # Head metadata from cached tensors
    n_heads = cached["q"].shape[1]
    head_indices = parse_head_indices(cfg.head_indices, n_heads)

    print(f"[setup] total chunks={chunks.shape[0]}, layer={cfg.layer_idx}, heads={head_indices}")

    # Need block/module objects for continuation
    dummy_x = cached["x_in"][:1].to(cfg.device)
    _, _, _, _, _, _, block, attn_module = extract_head_qkv_and_teacher_outputs_gpt2(model, dummy_x, cfg.layer_idx)

    losses_all = []
    head_ids_all = []
    pos_ids_all = []

    total_examples = 0
    for chunk_id in range(chunks.shape[0]):
        input_ids_1 = chunks[chunk_id:chunk_id + 1]
        x_in_1 = cached["x_in"][chunk_id:chunk_id + 1].to(cfg.device)
        q_1 = cached["q"][chunk_id:chunk_id + 1].to(cfg.device)
        k_1 = cached["k"][chunk_id:chunk_id + 1].to(cfg.device)
        v_1 = cached["v"][chunk_id:chunk_id + 1].to(cfg.device)
        z_teacher_1 = cached["z_teacher"][chunk_id:chunk_id + 1].to(cfg.device)
        zcat_teacher_1 = cached["zcat_teacher"][chunk_id:chunk_id + 1].to(cfg.device)

        L = input_ids_1.shape[1]

        for pos in range(cfg.min_context, L - 1, cfg.position_stride):
            for head_idx in head_indices:
                losses_dict = evaluate_query_head(
                    model=model,
                    input_ids=input_ids_1,
                    x_in=x_in_1,
                    q=q_1,
                    k=k_1,
                    v=v_1,
                    z_teacher=z_teacher_1,
                    zcat_teacher=zcat_teacher_1,
                    block=block,
                    attn_module=attn_module,
                    layer_idx=cfg.layer_idx,
                    head_idx=head_idx,
                    pos=pos,
                    cfg=cfg,
                )
                losses_vec = torch.tensor([[losses_dict[l] for l in LEARNERS]], dtype=torch.float32)
                losses_all.append(losses_vec)
                head_ids_all.append(torch.tensor([head_idx], dtype=torch.long))
                pos_ids_all.append(torch.tensor([pos], dtype=torch.long))
                total_examples += 1

        if chunk_id % 8 == 0 or chunk_id == chunks.shape[0] - 1:
            print(f"[bench] chunk {chunk_id+1}/{chunks.shape[0]} examples so far={total_examples}")

    return {
        "losses": torch.cat(losses_all, dim=0),
        "head_ids": torch.cat(head_ids_all, dim=0),
        "pos_ids": torch.cat(pos_ids_all, dim=0),
    }


def summarize(results: Dict[str, torch.Tensor]):
    losses = results["losses"]
    head_ids = results["head_ids"]
    pos_ids = results["pos_ids"]

    mean_losses = losses.mean(dim=0)
    oracle = losses.min(dim=-1).values.mean().item()
    best_fixed = mean_losses.min().item()
    winner = mean_losses.argmin().item()

    print("\n=== Fixed learner benchmark ===")
    for i, learner in enumerate(LEARNERS):
        print(f"  {learner:<16} {mean_losses[i].item():.6f}")
    print(f"  {'best_fixed':<16} {best_fixed:.6f} ({LEARNERS[winner]})")
    print(f"  {'oracle':<16} {oracle:.6f}")
    print(f"  {'oracle_gap':<16} {best_fixed - oracle:.6f}")

    winners = losses.argmin(dim=-1)
    print("\n=== Winner fractions ===")
    for i, learner in enumerate(LEARNERS):
        frac = (winners == i).float().mean().item()
        print(f"  {learner:<16} {frac:.3f}")

    sorted_losses, _ = torch.sort(losses, dim=-1)
    margins = sorted_losses[:, 1] - sorted_losses[:, 0]
    print("\n=== Oracle margin stats ===")
    print(f"  mean margin          {margins.mean().item():.6f}")
    print(f"  median margin        {margins.median().item():.6f}")
    print(f"  frac margin < 1e-4   {(margins < 1e-4).float().mean().item():.4f}")
    print(f"  frac margin < 1e-3   {(margins < 1e-3).float().mean().item():.4f}")
    print(f"  frac margin < 1e-2   {(margins < 1e-2).float().mean().item():.4f}")

    print("\n=== Winner fractions by head ===")
    for head in sorted(torch.unique(head_ids).tolist()):
        mask = (head_ids == head)
        winners_h = winners[mask]
        stats = {learner: (winners_h == i).float().mean().item() for i, learner in enumerate(LEARNERS)}
        stats_str = "  ".join(f"{k}:{v:.2f}" for k, v in stats.items())
        print(f"  head={head:>2}  {stats_str}")

    print("\n=== Winner fractions by position (first 10 positions present) ===")
    for pos in sorted(torch.unique(pos_ids).tolist())[:10]:
        mask = (pos_ids == pos)
        winners_p = winners[mask]
        stats = {learner: (winners_p == i).float().mean().item() for i, learner in enumerate(LEARNERS)}
        stats_str = "  ".join(f"{k}:{v:.2f}" for k, v in stats.items())
        print(f"  pos={pos:>3}  {stats_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text_field", type=str, default="text")

    parser.add_argument("--max_texts", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=64)

    parser.add_argument("--layer_idx", type=int, default=4)
    parser.add_argument("--head_indices", type=str, default="all")
    parser.add_argument("--min_context", type=int, default=16)
    parser.add_argument("--position_stride", type=int, default=8)

    parser.add_argument("--beta_soft", type=float, default=6.0)
    parser.add_argument("--k_sharp", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--k_linear_local", type=int, default=16)
    parser.add_argument("--ridge_lambda", type=float, default=1e-1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="outputs/head_counterfactual_cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    cfg = BenchConfig(**vars(args))

    set_seed(cfg.seed)
    results = run_benchmark(cfg)
    summarize(results)

    print("\n=== Interpretation ===")
    print("This benchmark supports useful real-NLP heterogeneity only if:")
    print("  1) oracle beats best fixed by a nontrivial margin")
    print("  2) winner fractions are mixed rather than soft dominating by construction")
    print("  3) this remains true across layers / heads / corpora")
    print("If soft dominates almost everything, learned K/V geometry may already make one learner sufficient.")


if __name__ == "__main__":
    main()