#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.language_model_probes.gpt2_probe_utils import (
    add_shared_cli_args,
    build_candidate_assignments,
    build_head_groups,
    candidate_name,
    continue_from_modified_block_gpt2,
    extract_head_qkv_and_teacher_outputs_gpt2,
    head_slice,
    load_and_pack_texts,
    mean_next_token_nll,
    parse_head_indices,
    run_to_block_and_cache_tensors,
)
from experiments.language_model_probes.probe_utils import LearnerRegistry, short_hash


BASE_LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]
LEARNER_REGISTRY = LearnerRegistry(BASE_LEARNERS)

@dataclass
class SuiteConfig:
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

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def result_stem(cfg: SuiteConfig) -> str:
    key = json.dumps(asdict(cfg), sort_keys=True)
    return short_hash(key)


@torch.no_grad()
def predict_head_output_for_all_learners(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_idx: int,
    pos: int,
    cfg: SuiteConfig,
) -> Dict[str, torch.Tensor]:
    q_t = q[:, head_idx, pos, :]
    Kctx = k[:, head_idx, :pos + 1, :] 
    Vctx = v[:, head_idx, :pos + 1, :]

    out = {}
    for learner in BASE_LEARNERS:
        pred = LEARNER_REGISTRY.predict(learner, q_t, Kctx, Vctx, cfg)[0]
        out[learner] = pred
    return out

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
    cfg: SuiteConfig,
) -> Dict[str, float]:
    head_dim = attn_module.head_dim
    pred_map = predict_head_output_for_all_learners(q, k, v, head_idx, pos, cfg)

    M = len(BASE_LEARNERS)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    head_slice_idx = head_slice(head_idx, head_dim)
    for m, learner in enumerate(BASE_LEARNERS):
        zcat_rep[m, pos, head_slice_idx] = pred_map[learner]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_rep,
        layer_idx=layer_idx,
    )

    losses = mean_next_token_nll(logits, input_ids.repeat(M, 1), [pos])
    return {learner: float(losses[m].item()) for m, learner in enumerate(BASE_LEARNERS)}


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
    cfg: SuiteConfig,
) -> Dict[str, float]:
    head_dim = attn_module.head_dim
    pred_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    for head_idx in head_group:
        pred_map = predict_head_output_for_all_learners(q, k, v, head_idx, pos, cfg)
        for learner, pred in pred_map.items():
            pred_cache[(head_idx, learner)] = pred

    M = len(BASE_LEARNERS)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    for m, learner in enumerate(BASE_LEARNERS):
        for head_idx in head_group:
            zcat_rep[m, pos, head_slice(head_idx, head_dim)] = pred_cache[(head_idx, learner)]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_rep,
        layer_idx=layer_idx,
    )

    losses = mean_next_token_nll(logits, input_ids.repeat(M, 1), [pos])
    return {learner: float(losses[m].item()) for m, learner in enumerate(BASE_LEARNERS)}


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
    cfg: SuiteConfig,
    assignments: Sequence[Tuple[str, ...]],
    assignment_names: Sequence[str],
) -> Dict[str, float]:
    head_dim = attn_module.head_dim

    pred_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    for head_idx in head_group:
        pred_map = predict_head_output_for_all_learners(q, k, v, head_idx, pos, cfg)
        for learner, pred in pred_map.items():
            pred_cache[(head_idx, learner)] = pred

    M = len(assignments)
    x_rep = x_in.repeat(M, 1, 1)
    zcat_rep = zcat_teacher.repeat(M, 1, 1)

    for m, assignment in enumerate(assignments):
        for head_idx, learner in zip(head_group, assignment):
            zcat_rep[m, pos, head_slice(head_idx, head_dim)] = pred_cache[(head_idx, learner)]

    logits = continue_from_modified_block_gpt2(
        model=model,
        block=block,
        x_in=x_rep,
        zcat_mod=zcat_rep,
        layer_idx=layer_idx,
    )

    losses = mean_next_token_nll(logits, input_ids.repeat(M, 1), [pos])
    return {assignment_names[m]: float(losses[m].item()) for m in range(M)}


# ============================================================
# Main benchmark loop
# ============================================================

@torch.no_grad()
def run_benchmark(cfg: SuiteConfig) -> Dict[str, object]:
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
    chunks = load_and_pack_texts(cfg, tokenizer, text_field=cfg.text_field).to(cfg.device)
    cached = run_to_block_and_cache_tensors(model, chunks, cfg)

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
        intervention_units = build_head_groups(
            selected_heads=head_indices,
            group_size=cfg.head_group_size,
            strategy=cfg.head_group_strategy,
            manual_head_groups=cfg.manual_head_groups,
            max_head_groups=cfg.max_head_groups,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown replace_mode={mode}")

    if not intervention_units:
        raise ValueError("No intervention units generated")

    if mode == "multi_head_single_pos_per_head":
        per_head_assignments = build_candidate_assignments(
            replace_mode=mode,
            learner_names=BASE_LEARNERS,
            group_size=cfg.head_group_size,
        )
        per_head_assignment_names = [candidate_name(a) for a in per_head_assignments]
        candidate_names = per_head_assignment_names
    else:
        per_head_assignments = None
        per_head_assignment_names = None
        candidate_names = BASE_LEARNERS

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
                    unit_label = f"group={','.join(map(str, unit))}"
                elif mode == "multi_head_single_pos_per_head":
                    if per_head_assignments is None or per_head_assignment_names is None:
                        raise RuntimeError("per-head assignments were not initialized")
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
                        assignments=per_head_assignments,
                        assignment_names=per_head_assignment_names,
                    )
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


def save_results(results: Dict[str, object], cfg: SuiteConfig) -> None:
    if not cfg.save_results:
        return

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = result_stem(cfg)
    path = output_dir / f"{stem}__results.pt"
    torch.save(results, path)
    print(f"\n[save] saved results to {path}")




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

    cfg = SuiteConfig(**vars(args))
    set_seed(cfg.seed)

    print("[config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    if cfg.replace_mode == "multi_head_single_pos_per_head":
        n_assignments = len(BASE_LEARNERS) ** cfg.head_group_size
        print(f"[note] per-head mode will evaluate {n_assignments} assignments per example")

    results = run_benchmark(cfg)
    summarize(results)
    save_results(results, cfg)

if __name__ == "__main__":
    main()
