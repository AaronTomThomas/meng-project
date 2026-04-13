import argparse
from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

from experiments.attention_learners import LearnerHyperParams
from experiments.language_model_probes.probe_utils import LearnerRegistry, causal_soft_attention_from_qkv, merge_heads, split_heads
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import math

BASE_LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]
LEARNER_REGISTRY = LearnerRegistry(BASE_LEARNERS)


@dataclass
class SequenceOracleConfig(LearnerHyperParams):
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "validation"

    max_texts: int = 200
    block_size: int = 96
    batch_size: int = 4
    max_chunks: int = 64

    layer_idx: int = 4
    head_indices: str = "all"
    min_context: int = 16
    position_stride: int = 1

    replace_mode: str = "multi_head_single_pos_per_head" #"multi_head_single_pos_shared/multi_head_single_pos_per_head"
    head_group_size: int = 12
    head_group_strategy: str = "contiguous"
    manual_head_groups: str = ""
    max_head_groups: int = 0

    save_results: bool = False
    output_dir: str = "outputs/head_counterfactual_results"
    cache_dir: str = "outputs/head_counterfactual_cache"

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def cache_stem(cfg: SequenceOracleConfig) -> str:
    key = (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.split}|"
        f"maxtexts={cfg.max_texts}|block={cfg.block_size}|maxchunks={cfg.max_chunks}|"
        f"layer={cfg.layer_idx}|heads={cfg.head_indices}|"
        f"minctx={cfg.min_context}|stride={cfg.position_stride}|"
        f"beta={cfg.beta_soft}|ksharp={cfg.k_sharp}|"
        f"window={cfg.window_size}|klin={cfg.k_linear_local}|ridge={cfg.ridge_lambda}|"
        f"seed={cfg.seed}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:10]

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
def get_block_input_gpt2(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    x = get_input_embeddings_gpt2(model, input_ids)
    blocks = model.transformer.h
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"layer_idx={layer_idx} invalid for {len(blocks)} blocks")
    for l in range(layer_idx):
        x = blocks[l](x, use_cache=False)
        x = x[0] if isinstance(x, tuple) else x
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
    
def run_to_block_and_cache_tensors(model, chunks: torch.Tensor, cfg: SequenceOracleConfig):
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
        x = model.transformer.h[l](x, use_cache=False)
        x = x[0] if isinstance(x, tuple) else x
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits

def load_and_pack_texts(cfg: SequenceOracleConfig, tokenizer) -> torch.Tensor:

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
        text = ex["text"]
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

def build_head_groups(
    selected_heads: List[int],
    group_size: int,
    strategy: str,
    manual_head_groups: str,
    max_head_groups: int
) -> List[List[int]]:
    if strategy == "manual":
        if not manual_head_groups.strip():
            raise ValueError("manual_head_groups required when strategy=manual")
        groups = []
        for chunk in manual_head_groups.split(";"):
            grp = [int(x.strip()) for x in chunk.split(",") if x.strip()]
            groups.append(grp)
    elif strategy == "contiguous":
        groups = []
        for i in range(0, len(selected_heads), group_size):
            grp = selected_heads[i:i + group_size]
            if len(grp) == group_size:
                groups.append(grp)
    else:
        raise ValueError(f"Unknown head_group_strategy={strategy}")

    if max_head_groups > 0:
        groups = groups[:max_head_groups]
    return groups


def head_slice(head_idx: int, head_dim: int) -> slice:
    return slice(head_idx * head_dim, (head_idx + 1) * head_dim)

@torch.no_grad()
def mean_next_token_nll(
    logits: torch.Tensor, 
    input_ids: torch.Tensor, 
    positions: List[int]
) -> torch.Tensor:
    if len(positions) == 0:
        raise ValueError("positions must be non-empty")
    
    device = logits.device
    pos_t = torch.tensor(positions, device=device, dtype=torch.long)
    target_t = pos_t + 1

    log_probs = F.log_softmax(logits[:, pos_t, :], dim=-1)
    targets = input_ids[:, target_t]    
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean(dim=-1)


@torch.no_grad()
def apply_sequence_assignment(
    zcat_teacher: torch.Tensor,
    group: List[int],
    positions: List[int],
    per_pos_assignment: Dict[int, Tuple[str, ...]],
    preds: Dict[int, Dict[Tuple[str, ...], Dict[int, torch.Tensor]]],
    head_dim: int,
) -> torch.Tensor:
    out = zcat_teacher.clone()
    for pos in positions: 
        assignment = per_pos_assignment[pos]
        for h in group:
            out[0, pos, head_slice(h, head_dim)] = preds[pos][assignment][h]
    return out


@torch.no_grad()
def best_local_assignment_per_position(
    model,
    block,
    x_in: torch.Tensor,
    input_ids: torch.Tensor,
    zcat_teacher: torch.Tensor,
    group: List[int],
    positions: List[int],
    assignments: List[Tuple[str, ...]],
    preds: Dict[int, Dict[Tuple[str, ...], Dict[int, torch.Tensor]]],
    head_dim: int,
    layer_idx: int,
) -> Dict[int, Tuple[str, ...]]:
    """Isolated local oracle for eachc position: i.e choose the assignment that
    minimizes NLL when only that one position is replaced"""
    best = {}
    for pos in positions: 
        cand_zcats = []
        for assignment in assignments: 
            zcat_mod = zcat_teacher.clone()
            for h in group:
                # replacing the attention output for this head at this position with the learner prediction
                zcat_mod[0, pos, head_slice(h, head_dim)] = preds[pos][assignment][h]
            cand_zcats.append(zcat_mod)
        zcat_batch = torch.cat(cand_zcats, dim= 0)
        x_rep = x_in.repeat(len(assignments), 1, 1)
        logits = continue_from_modified_block_gpt2(
            model=model,
            block=block,
            x_in=x_rep,
            zcat_mod=zcat_batch,
            layer_idx=layer_idx,
        )
        losses = mean_next_token_nll(logits, input_ids.repeat(len(assignments), 1), [pos])
        best_idx = int(losses.argmin().item())
        best[pos] = assignments[best_idx]
    return best


@torch.no_grad()
def greedy_sequence_assignment(
    model,
    block,
    x_in: torch.Tensor,
    input_ids: torch.Tensor,
    zcat_teacher: torch.Tensor,
    group: List[int],
    positions: List[int],
    assignments: List[Tuple[str, ...]],
    preds: Dict[int, Dict[Tuple[str, ...], Dict[int, torch.Tensor]]],
    head_dim: int,
    layer_idx: int,
) -> Dict[int, Tuple[str, ...]]:
    """
    Greedy sequence oracle:
    commit left-to-right. At each position, choose the assignment that minimizes
    suffix mean NLL from current position onward, given earlier committed choices.
    Future positions remain teacher until chosen.
    """

    committed = {}
    current_zcat = zcat_teacher.clone()
    for i, pos in enumerate(positions):
        suffix_positions = positions[i:]
        cand_zcats = []
        for assignment in assignments:
            z = current_zcat.clone()
            for h in group:
                z[0, pos, head_slice(h, head_dim)] = preds[pos][assignment][h]
            cand_zcats.append(z)

        zcat_batch = torch.cat(cand_zcats, dim=0)
        x_rep = x_in.repeat(len(assignments), 1, 1)
        logits = continue_from_modified_block_gpt2(
            model=model,
            block=block,
            x_in=x_rep,
            zcat_mod=zcat_batch,
            layer_idx=layer_idx,
        )
        losses = mean_next_token_nll(logits, input_ids.repeat(len(assignments), 1), suffix_positions)
        best_idx = int(losses.argmin().item())
        best_assign = assignments[best_idx]
        committed[pos] = best_assign
        for h in group:
            current_zcat[0, pos, head_slice(h, head_dim)] = preds[pos][best_assign][h]

    return committed

def evaluate_head_group(
    model,
    block,
    input_ids: torch.Tensor,
    x_in: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    zcat_teacher: torch.Tensor,
    group: List[int],
    cfg: SequenceOracleConfig,
    head_dim: int,
    assignments: List[Tuple[str, ...]],
    positions: List[int],
):
    preds = {}
    for pos in positions:
        q_pos = {}
        ctx = {}
        for h in group:
            q_pos[h] = q[:, h, pos, :]
            ctx[h] = (
                k[:, h, :pos + 1, :],
                v[:, h, :pos + 1, :],
            )
        preds[pos] = {}
        for assignment in assignments:
            head_pred = {}
            for h, learner in zip(group, assignment):
                Kctx, Vctx = ctx[h]
                pred = LEARNER_REGISTRY.predict(learner, q_pos[h], Kctx, Vctx, cfg)[0]
                head_pred[h] = pred
            preds[pos][assignment] = head_pred
    
    # sequence-level fixed assignment
    fixed_losses = {}
    for assignment in assignments:
        per_pos_assign = {pos: assignment for pos in positions}
        zcat_mod = apply_sequence_assignment(
            zcat_teacher=zcat_teacher,
            group=group,
            positions=positions,
            per_pos_assignment=per_pos_assign,
            preds=preds,
            head_dim=head_dim,
        )
        logits = continue_from_modified_block_gpt2(
            model=model,
            block=block,
            x_in=x_in,
            zcat_mod=zcat_mod,
            layer_idx=cfg.layer_idx
        )
        fixed_losses[assignment] = float(mean_next_token_nll(logits, input_ids, positions).item())

    # locally find best learner then apply jointly and measure loss
    local_best_assign = best_local_assignment_per_position(
        model=model,
        block=block,
        x_in=x_in,
        input_ids=input_ids,
        zcat_teacher=zcat_teacher,
        group=group,
        positions=positions,
        assignments=assignments,
        preds=preds,
        head_dim=head_dim,
        layer_idx=cfg.layer_idx,
    )

    zcat_local_joint = apply_sequence_assignment(
        zcat_teacher=zcat_teacher,
        group=group,
        positions=positions,
        per_pos_assignment=local_best_assign,
        preds=preds,
        head_dim=head_dim,
    )

    logits_local_joint = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_in,
        zcat_mod=zcat_local_joint,
        layer_idx=cfg.layer_idx,
    )
    local_joint_loss = float(mean_next_token_nll(logits_local_joint, input_ids, positions).item())


    # greedily assign

    greedy_assign = greedy_sequence_assignment(
        model=model,
        block=block,
        x_in=x_in,
        input_ids=input_ids,
        zcat_teacher=zcat_teacher,
        group=group,
        positions=positions,
        assignments=assignments,
        preds=preds,
        head_dim=head_dim,
        layer_idx=cfg.layer_idx,
    )
    zcat_greedy = apply_sequence_assignment(
        zcat_teacher=zcat_teacher,
        group=group,
        positions=positions,
        per_pos_assignment=greedy_assign,
        preds=preds,
        head_dim=head_dim,
    )
    logits_greedy = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_in,
        zcat_mod=zcat_greedy,
        layer_idx=cfg.layer_idx,
    )
    greedy_loss = float(mean_next_token_nll(logits_greedy, input_ids, positions).item())

    return {
        "fixed_losses": fixed_losses,
        "local_joint_loss": local_joint_loss,
        "greedy_loss": greedy_loss,
        "local_best_assign": local_best_assign,
        "greedy_assign": greedy_assign,
    }


def run_oracle_eval(cfg: SequenceOracleConfig):
    print("[setup] loading tokenizer/model")
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
    block_tensors = run_to_block_and_cache_tensors(model, chunks, cfg)

    n_heads = block_tensors["q"].shape[1]
    head_dim = block_tensors["q"].shape[-1]

    if cfg.head_indices == "all":
        selected_heads = list(range(n_heads))
    else:
        selected_heads = [int(x.strip()) for x in cfg.head_indices.split(",") if x.strip()]
        
    for h in selected_heads:
        if h < 0 or h >= n_heads:
            raise ValueError(f"Head index {h} out of range [0, {n_heads-1}]")
    
    head_groups = build_head_groups(
        selected_heads=selected_heads,
        group_size=cfg.head_group_size,
        strategy=cfg.head_group_strategy,
        manual_head_groups=cfg.manual_head_groups,
        max_head_groups=cfg.max_head_groups,
    )

    if cfg.replace_mode == "multi_head_single_pos_shared":
        assignments =  [(learner,) * cfg.head_group_size for learner in BASE_LEARNERS]
    elif cfg.replace_mode == "multi_head_single_pos_per_head":
        assignments =  list(itertools.product(BASE_LEARNERS, repeat=cfg.head_group_size))
    else:
        raise ValueError(f"Unsupported replace_mode={cfg.replace_mode}")
    
    print(f"[setup] total chunks={chunks.shape[0]}, layer={cfg.layer_idx}, mode={cfg.replace_mode}")
    print(f"[setup] selected heads={selected_heads}")
    print(f"[setup] intervention units={head_groups}")
    print(f"[setup] candidate assignments={len(assignments)}")

    dummy_x = block_tensors["x_in"][:1].to(cfg.device)
    _, _, _, _, _, _, block, attn_module = extract_head_qkv_and_teacher_outputs_gpt2(model, dummy_x, cfg.layer_idx)
    assert head_dim == attn_module.head_dim
    
    positions_template = list(range(cfg.min_context, cfg.block_size - 1, cfg.position_stride))

    fixed_losses_all = []
    local_joint_all = []
    greedy_all = []
    group_ids_all = []

    total_examples = 0
    for chunk_id in range(chunks.shape[0]):
        input_ids_1 = chunks[chunk_id:chunk_id + 1]
        x_in_1 = block_tensors["x_in"][chunk_id:chunk_id + 1].to(cfg.device)
        q_1 = block_tensors["q"][chunk_id:chunk_id + 1].to(cfg.device)
        k_1 = block_tensors["k"][chunk_id:chunk_id + 1].to(cfg.device)
        v_1 = block_tensors["v"][chunk_id:chunk_id + 1].to(cfg.device)
        zcat_teacher_1 = block_tensors["zcat_teacher"][chunk_id:chunk_id + 1].to(cfg.device)

        positions = [p for p in positions_template if p < input_ids_1.shape[1] - 1]

        for group_idx, group in enumerate(head_groups):
            print(f"Chunk {chunk_id} : evaluating heads {group}")
            out = evaluate_head_group(
                model=model,
                block=block,
                input_ids=input_ids_1,
                x_in=x_in_1,
                q=q_1,
                k=k_1,
                v=v_1,
                zcat_teacher=zcat_teacher_1,
                group=group,
                cfg=cfg,
                head_dim=head_dim,
                assignments=assignments,
                positions=positions,
            )

            fixed_losses_vec = torch.tensor(
                [[out["fixed_losses"][assignment] for assignment in assignments]],
                dtype=torch.float32,
            )
            fixed_losses_all.append(fixed_losses_vec)
            local_joint_all.append(torch.tensor([out["local_joint_loss"]], dtype=torch.float32))
            greedy_all.append(torch.tensor([out["greedy_loss"]], dtype=torch.float32))
            group_ids_all.append(torch.tensor([group_idx], dtype=torch.long))
            total_examples += 1
        if chunk_id % 8 == 0 or chunk_id == chunks.shape[0] - 1:
            print(f"[bench] chunk {chunk_id+1}/{chunks.shape[0]} examples so far={total_examples}")
    return {
        "assignments": assignments,
        "groups": head_groups,
        "fixed_losses": torch.cat(fixed_losses_all, dim=0),
        "local_joint_losses": torch.cat(local_joint_all, dim=0),
        "greedy_losses": torch.cat(greedy_all, dim=0),
        "unit_ids": torch.cat(group_ids_all, dim=0),
    }



def candidate_name(assign: Sequence[str]) -> str:
    return "+".join(assign)

def summarize(results: Dict[str, torch.Tensor]):
    assignments = results["assignments"]
    units = results["groups"]
    fixed_losses = results["fixed_losses"]
    local_joint_losses = results["local_joint_losses"]
    greedy_losses = results["greedy_losses"]
    unit_ids = results["unit_ids"]

    mean_fixed = fixed_losses.mean(dim=0)
    best_fixed_idx = int(mean_fixed.argmin().item())
    best_fixed_name = candidate_name(assignments[best_fixed_idx])
    best_fixed_loss = float(mean_fixed[best_fixed_idx].item())

    mean_local_joint = float(local_joint_losses.mean().item())
    mean_greedy = float(greedy_losses.mean().item())

    print("\n=== Sequence-level fixed assignment benchmark ===")
    for i, assign in enumerate(assignments):
        print(f"  {candidate_name(assign):<32} {mean_fixed[i].item():.6f}")
    print(f"  {'best_fixed':<32} {best_fixed_loss:.6f} ({best_fixed_name})")

    print("\n=== Sequence-level heterogeneous selectors ===")
    print(f"  {'local_then_joint':<32} {mean_local_joint:.6f}")
    print(f"  {'greedy_suffix_oracle':<32} {mean_greedy:.6f}")

    print("\n=== Gaps vs best fixed ===")
    print(f"  {'local_then_joint_gap':<32} {best_fixed_loss - mean_local_joint:.6f}")
    print(f"  {'greedy_suffix_gap':<32} {best_fixed_loss - mean_greedy:.6f}")

    winners = fixed_losses.argmin(dim=-1)
    print("\n=== Best fixed assignment winner fractions across sequence-level examples ===")
    for i, assign in enumerate(assignments):
        frac = (winners == i).float().mean().item()
        print(f"  {candidate_name(assign):<32} {frac:.3f}")

    print("\n=== Best fixed assignment winner fractions by unit ===")
    for unit_idx, unit in enumerate(units):
        mask = (unit_ids == unit_idx)
        w = winners[mask]
        stats = {candidate_name(assign): (w == i).float().mean().item() for i, assign in enumerate(assignments)}
        stats_str = "  ".join(f"{k}:{v:.2f}" for k, v in stats.items())
        print(f"  group={','.join(map(str, unit)):<12} {stats_str}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--block_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=64)

    parser.add_argument("--layer_idx", type=int, default=4)
    parser.add_argument("--head_indices", type=str, default="all")
    parser.add_argument("--position_stride", type=int, default=8)
    parser.add_argument("--replace_mode", type=str, default="multi_head_single_pos_per_head",
                        choices=["multi_head_single_pos_shared", "multi_head_single_pos_per_head"])
    parser.add_argument("--head_group_size", type=int, default=2)
    parser.add_argument("--head_group_strategy", type=str, default="contiguous",
                        choices=["contiguous", "manual"])
    parser.add_argument("--manual_head_groups", type=str, default="")
    parser.add_argument("--max_head_groups", type=int, default=0)


    cfg = SequenceOracleConfig()
    print("[config]")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    results = run_oracle_eval(cfg)


    summarize(results)
    
if __name__ == "__main__":
    main()
