from __future__ import annotations

"""
Token-level bridge between TTR readout improvability and trained AKAZA deltas.

This probe is intentionally standalone and GPT-2-specific. It reuses the
existing language-model probe machinery for frozen q/K/V readouts and the
attention-adapter checkpoint plumbing for trained AKAZA/FreeZ checkpoints.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.attention_learners import LearnerHyperParams, build_learners
from experiments.gpt2_probe_utils import (
    continue_from_modified_block_gpt2,
    extract_head_qkv_and_teacher_outputs_gpt2,
    get_block_input_gpt2,
)
from experiments.router_development.attention_adapter.data import load_chunks_for_split
from experiments.router_development.attention_adapter.probes.eval_algorithmic_decomp import (
    CANONICAL_LEARNERS,
    DEFAULT_CANDIDATE_LEARNERS,
    candidate_attention_outputs,
    cfg_from_summary,
    jsonable_dataclass_dict,
    parse_csv,
    parse_int_csv,
    ridge_project_delta,
)
from experiments.router_development.attention_adapter.adapters.akaza_adapters import (
    GPT2AKAZAAdapter,
)
from experiments.router_development.attention_adapter.trainer import TrainableParameters


def token_next_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return next-token NLL for positions 0..T-2 with shape [B,T-1]."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Average ranks for a 1D tensor, with ties receiving their mean rank."""
    if x.ndim != 1:
        raise ValueError(f"average_ranks expects a 1D tensor, got shape {tuple(x.shape)}")
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks_sorted = torch.empty_like(sorted_x, dtype=torch.float64)
    n = sorted_x.numel()
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * float(i + j - 1)
        i = j
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman correlation with finite filtering and average-tie ranks."""
    x = x.detach().flatten().double().cpu()
    y = y.detach().flatten().double().cpu()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return float("nan")

    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.square().sum().sqrt() * ry.square().sum().sqrt()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((rx * ry).sum().item() / denom.item())


def top_decile_enrichment(score: torch.Tensor, value: torch.Tensor) -> Dict[str, float]:
    """Compare value means for top-10% score tokens against the remaining tokens."""
    score = score.detach().flatten().float().cpu()
    value = value.detach().flatten().float().cpu()
    mask = torch.isfinite(score) & torch.isfinite(value)
    score = score[mask]
    value = value[mask]
    if score.numel() < 10:
        return {
            "threshold": float("nan"),
            "top_mean": float("nan"),
            "rest_mean": float("nan"),
            "ratio": float("nan"),
            "difference": float("nan"),
            "top_count": 0.0,
            "rest_count": 0.0,
        }

    threshold = torch.quantile(score, 0.9)
    top = score >= threshold
    rest = ~top
    if not bool(top.any()) or not bool(rest.any()):
        return {
            "threshold": float(threshold.item()),
            "top_mean": float("nan"),
            "rest_mean": float("nan"),
            "ratio": float("nan"),
            "difference": float("nan"),
            "top_count": float(top.sum().item()),
            "rest_count": float(rest.sum().item()),
        }

    top_mean = value[top].mean()
    rest_mean = value[rest].mean()
    return {
        "threshold": float(threshold.item()),
        "top_mean": float(top_mean.item()),
        "rest_mean": float(rest_mean.item()),
        "ratio": float((top_mean / rest_mean).item()) if abs(float(rest_mean.item())) > 1e-12 else float("nan"),
        "difference": float((top_mean - rest_mean).item()),
        "top_count": float(top.sum().item()),
        "rest_count": float(rest.sum().item()),
    }


def safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def c_proj_linear_delta(block, delta: torch.Tensor) -> torch.Tensor:
    """Apply GPT-2 attn.c_proj to a delta without counting the projection bias."""
    zeros = torch.zeros_like(delta)
    return block.attn.c_proj(delta) - block.attn.c_proj(zeros)


def summarize_vectors(g_ttr: torch.Tensor, g_akaza: torch.Tensor, delta_norm: torch.Tensor, delta_wo_norm: torch.Tensor) -> Dict[str, Any]:
    return {
        "n_token_layer": int(g_ttr.numel()),
        "mean_G_TTR": float(g_ttr.float().mean().item()),
        "mean_G_AKAZA": float(g_akaza.float().mean().item()),
        "mean_delta_norm": float(delta_norm.float().mean().item()),
        "mean_delta_WO_norm": float(delta_wo_norm.float().mean().item()),
        "corr_G_TTR_delta_norm": spearman_corr(g_ttr, delta_norm),
        "corr_G_TTR_delta_WO_norm": spearman_corr(g_ttr, delta_wo_norm),
        "corr_G_TTR_G_AKAZA": spearman_corr(g_ttr, g_akaza),
        "top_decile_delta_norm_enrichment": top_decile_enrichment(g_ttr, delta_norm),
        "top_decile_delta_WO_norm_enrichment": top_decile_enrichment(g_ttr, delta_wo_norm),
        "top_decile_G_AKAZA_enrichment": top_decile_enrichment(g_ttr, g_akaza),
    }


@torch.no_grad()
def eval_ttr_bridge(
    *,
    model,
    wrapped: GPT2AKAZAAdapter,
    chunks: torch.Tensor,
    layer_indices: Sequence[int],
    candidate_names: Sequence[str],
    learner_hparams: LearnerHyperParams,
    batch_size: int,
    device: torch.device,
    ridge_lambda: float,
    eps: float,
    token_csv_path: str | None = None,
) -> Dict[str, Any]:
    learner_instances = build_learners(list(candidate_names))

    per_layer: Dict[int, Dict[str, list[torch.Tensor] | list[float] | Dict[str, int]]] = {
        int(layer_idx): {
            "g_ttr": [],
            "g_akaza": [],
            "delta_norm": [],
            "delta_wo_norm": [],
            "projection_energy_num": [],
            "projection_energy_den": [],
            "projection_gain": [],
            "full_layer_gain": [],
            "best_counts": {name: 0 for name in candidate_names},
        }
        for layer_idx in layer_indices
    }

    token_csv_fh = None
    token_writer = None
    if token_csv_path:
        token_csv_fh = Path(token_csv_path).open("w", newline="")
        token_writer = csv.DictWriter(
            token_csv_fh,
            fieldnames=[
                "chunk_idx",
                "layer_idx",
                "token_pos",
                "G_TTR",
                "best_learner",
                "G_AKAZA",
                "delta_norm",
                "delta_WO_norm",
            ],
        )
        token_writer.writeheader()

    try:
        n_examples = int(chunks.shape[0])
        for start in range(0, n_examples, batch_size):
            input_ids = chunks[start : start + batch_size].to(device)
            bsz, seq_len = input_ids.shape

            baseline_logits = model(input_ids=input_ids, use_cache=False).logits
            baseline_nll = token_next_nll(baseline_logits, input_ids)

            wrapped.set_peft_eval_mode()
            akaza_logits = wrapped(input_ids)
            akaza_nll = token_next_nll(akaza_logits, input_ids)
            g_akaza = baseline_nll - akaza_nll
            latest_deltas = {int(k): v.detach() for k, v in wrapped._latest_deltas.items()}

            for layer_idx in layer_indices:
                layer_idx = int(layer_idx)
                block = model.transformer.h[layer_idx]
                x_in = get_block_input_gpt2(model, input_ids, layer_idx)
                _h_ln1, _q, _k, _v, _z_teacher, zcat_teacher, _block, _attn = (
                    extract_head_qkv_and_teacher_outputs_gpt2(model, x_in, layer_idx)
                )

                candidate_outputs = candidate_attention_outputs(
                    model=model,
                    layer_idx=layer_idx,
                    hidden_states=x_in,
                    candidate_names=candidate_names,
                    learner_hparams=learner_hparams,
                    learner_instances=learner_instances,
                )

                candidate_gains: list[torch.Tensor] = []
                for name in candidate_names:
                    logits_a = continue_from_modified_block_gpt2(
                        model=model,
                        block=block,
                        x_in=x_in,
                        zcat_mod=candidate_outputs[name].to(dtype=zcat_teacher.dtype),
                        layer_idx=layer_idx,
                    )
                    nll_a = token_next_nll(logits_a, input_ids)
                    candidate_gains.append(baseline_nll - nll_a)

                gain_stack = torch.stack(candidate_gains, dim=-1)  # [B,T-1,A]
                g_ttr, best_idx = gain_stack.max(dim=-1)

                delta = latest_deltas[layer_idx].to(device=device, dtype=zcat_teacher.dtype)
                delta_tok = delta[:, :-1, :]
                delta_norm = delta_tok.float().norm(dim=-1)
                delta_wo_norm = c_proj_linear_delta(block, delta)[:, :-1, :].float().norm(dim=-1)

                directions = torch.stack(
                    [
                        candidate_outputs[name].to(dtype=zcat_teacher.dtype) - zcat_teacher
                        for name in candidate_names
                    ],
                    dim=2,
                )
                projection, _alpha = ridge_project_delta(
                    delta,
                    directions,
                    ridge_lambda=ridge_lambda,
                    eps=eps,
                )
                full_layer_logits = continue_from_modified_block_gpt2(
                    model=model,
                    block=block,
                    x_in=x_in,
                    zcat_mod=zcat_teacher + delta,
                    layer_idx=layer_idx,
                )
                projection_logits = continue_from_modified_block_gpt2(
                    model=model,
                    block=block,
                    x_in=x_in,
                    zcat_mod=zcat_teacher + projection,
                    layer_idx=layer_idx,
                )
                full_layer_gain = baseline_nll - token_next_nll(full_layer_logits, input_ids)
                projection_gain = baseline_nll - token_next_nll(projection_logits, input_ids)

                layer_acc = per_layer[layer_idx]
                layer_acc["g_ttr"].append(g_ttr.detach().cpu())
                layer_acc["g_akaza"].append(g_akaza.detach().cpu())
                layer_acc["delta_norm"].append(delta_norm.detach().cpu())
                layer_acc["delta_wo_norm"].append(delta_wo_norm.detach().cpu())
                layer_acc["projection_energy_num"].append(projection.float().square().sum().detach().cpu())
                layer_acc["projection_energy_den"].append(delta.float().square().sum().detach().cpu())
                layer_acc["projection_gain"].append(projection_gain.detach().cpu())
                layer_acc["full_layer_gain"].append(full_layer_gain.detach().cpu())

                best_counts = layer_acc["best_counts"]
                for j, name in enumerate(candidate_names):
                    best_counts[name] += int((best_idx == j).sum().item())

                if token_writer is not None:
                    g_ttr_cpu = g_ttr.detach().cpu()
                    g_akaza_cpu = g_akaza.detach().cpu()
                    delta_norm_cpu = delta_norm.detach().cpu()
                    delta_wo_norm_cpu = delta_wo_norm.detach().cpu()
                    best_idx_cpu = best_idx.detach().cpu()
                    for b in range(bsz):
                        chunk_idx = start + b
                        for pos in range(seq_len - 1):
                            token_writer.writerow(
                                {
                                    "chunk_idx": chunk_idx,
                                    "layer_idx": layer_idx,
                                    "token_pos": pos,
                                    "G_TTR": float(g_ttr_cpu[b, pos].item()),
                                    "best_learner": candidate_names[int(best_idx_cpu[b, pos].item())],
                                    "G_AKAZA": float(g_akaza_cpu[b, pos].item()),
                                    "delta_norm": float(delta_norm_cpu[b, pos].item()),
                                    "delta_WO_norm": float(delta_wo_norm_cpu[b, pos].item()),
                                }
                            )

            print(f"[eval] processed chunks {start + bsz}/{n_examples}")
    finally:
        if token_csv_fh is not None:
            token_csv_fh.close()

    layer_summaries: Dict[str, Any] = {}
    global_parts: Dict[str, list[torch.Tensor]] = {
        "g_ttr": [],
        "g_akaza": [],
        "delta_norm": [],
        "delta_wo_norm": [],
        "projection_gain": [],
        "full_layer_gain": [],
    }
    total_projection_energy_num = 0.0
    total_projection_energy_den = 0.0

    for layer_idx, layer_acc in per_layer.items():
        g_ttr = torch.cat(layer_acc["g_ttr"], dim=0).flatten()
        g_akaza = torch.cat(layer_acc["g_akaza"], dim=0).flatten()
        delta_norm = torch.cat(layer_acc["delta_norm"], dim=0).flatten()
        delta_wo_norm = torch.cat(layer_acc["delta_wo_norm"], dim=0).flatten()
        projection_gain = torch.cat(layer_acc["projection_gain"], dim=0).flatten()
        full_layer_gain = torch.cat(layer_acc["full_layer_gain"], dim=0).flatten()

        projection_energy_num = float(torch.stack(layer_acc["projection_energy_num"]).sum().item())
        projection_energy_den = float(torch.stack(layer_acc["projection_energy_den"]).sum().item())
        full_gain_sum = float(full_layer_gain.sum().item())
        projection_gain_sum = float(projection_gain.sum().item())

        summary = summarize_vectors(g_ttr, g_akaza, delta_norm, delta_wo_norm)
        summary["best_learner_counts"] = dict(layer_acc["best_counts"])
        summary["projection_recovery"] = {
            "projection_energy_over_delta_energy": (
                projection_energy_num / projection_energy_den
                if abs(projection_energy_den) > 1e-12
                else float("nan")
            ),
            "projection_gain_over_full_layer_gain": (
                projection_gain_sum / full_gain_sum
                if abs(full_gain_sum) > 1e-12
                else float("nan")
            ),
            "mean_projection_gain": float(projection_gain.mean().item()),
            "mean_full_layer_gain": float(full_layer_gain.mean().item()),
        }
        layer_summaries[str(layer_idx)] = summary

        total_projection_energy_num += projection_energy_num
        total_projection_energy_den += projection_energy_den
        for key, value in (
            ("g_ttr", g_ttr),
            ("g_akaza", g_akaza),
            ("delta_norm", delta_norm),
            ("delta_wo_norm", delta_wo_norm),
            ("projection_gain", projection_gain),
            ("full_layer_gain", full_layer_gain),
        ):
            global_parts[key].append(value)

    global_g_ttr = torch.cat(global_parts["g_ttr"])
    global_g_akaza = torch.cat(global_parts["g_akaza"])
    global_delta_norm = torch.cat(global_parts["delta_norm"])
    global_delta_wo_norm = torch.cat(global_parts["delta_wo_norm"])
    global_projection_gain = torch.cat(global_parts["projection_gain"])
    global_full_layer_gain = torch.cat(global_parts["full_layer_gain"])
    global_full_gain_sum = float(global_full_layer_gain.sum().item())
    global_projection_gain_sum = float(global_projection_gain.sum().item())

    global_summary = summarize_vectors(
        global_g_ttr,
        global_g_akaza,
        global_delta_norm,
        global_delta_wo_norm,
    )
    global_summary["projection_recovery"] = {
        "projection_energy_over_delta_energy": (
            total_projection_energy_num / total_projection_energy_den
            if abs(total_projection_energy_den) > 1e-12
            else float("nan")
        ),
        "projection_gain_over_full_layer_gain": (
            global_projection_gain_sum / global_full_gain_sum
            if abs(global_full_gain_sum) > 1e-12
            else float("nan")
        ),
        "mean_projection_gain": float(global_projection_gain.mean().item()),
        "mean_full_layer_gain": float(global_full_layer_gain.mean().item()),
    }

    return {
        "global": global_summary,
        "layers": layer_summaries,
    }


def write_layer_csv(path: str, layers: Dict[str, Any]) -> None:
    fieldnames = [
        "layer_idx",
        "n_token_layer",
        "mean_G_TTR",
        "mean_G_AKAZA",
        "mean_delta_norm",
        "mean_delta_WO_norm",
        "corr_G_TTR_delta_norm",
        "corr_G_TTR_delta_WO_norm",
        "corr_G_TTR_G_AKAZA",
        "projection_energy_over_delta_energy",
        "projection_gain_over_full_layer_gain",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for layer_idx, metrics in sorted(layers.items(), key=lambda kv: int(kv[0])):
            recovery = metrics["projection_recovery"]
            writer.writerow(
                {
                    "layer_idx": int(layer_idx),
                    "n_token_layer": metrics["n_token_layer"],
                    "mean_G_TTR": metrics["mean_G_TTR"],
                    "mean_G_AKAZA": metrics["mean_G_AKAZA"],
                    "mean_delta_norm": metrics["mean_delta_norm"],
                    "mean_delta_WO_norm": metrics["mean_delta_WO_norm"],
                    "corr_G_TTR_delta_norm": metrics["corr_G_TTR_delta_norm"],
                    "corr_G_TTR_delta_WO_norm": metrics["corr_G_TTR_delta_WO_norm"],
                    "corr_G_TTR_G_AKAZA": metrics["corr_G_TTR_G_AKAZA"],
                    "projection_energy_over_delta_energy": recovery["projection_energy_over_delta_energy"],
                    "projection_gain_over_full_layer_gain": recovery["projection_gain_over_full_layer_gain"],
                }
            )


def main() -> None:
    default_hp = LearnerHyperParams()
    parser = argparse.ArgumentParser(description="Evaluate the token-level TTR bridge for GPT-2 AKAZA checkpoints.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--token_csv_path", type=str, default=None)
    parser.add_argument("--layer_csv_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_train_chunks", type=int, default=None)
    parser.add_argument("--max_val_chunks", type=int, default=None)
    parser.add_argument("--max_test_chunks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate_learners", type=str, default=",".join(DEFAULT_CANDIDATE_LEARNERS))
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--local_kernel_beta", type=float, default=default_hp.local_kernel_beta)
    parser.add_argument("--window_size", type=int, default=default_hp.window_size)
    parser.add_argument("--k_knn_mean", type=int, default=default_hp.k_knn_mean)
    parser.add_argument("--learner_ridge_lambda", type=float, default=default_hp.ridge_lambda)
    parser.add_argument("--k_linear_local", type=int, default=default_hp.k_linear_local)
    parser.add_argument("--k_sharp", type=int, default=default_hp.k_sharp)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "summary" not in payload or "trainable_state_dict" not in payload:
        raise KeyError("Checkpoint must contain 'summary' and 'trainable_state_dict'.")

    summary = payload["summary"]
    if summary.get("method") != "akaza_freez":
        raise ValueError(f"Expected an akaza_freez checkpoint, got method={summary.get('method')!r}")

    cfg = cfg_from_summary(summary)
    if cfg.model_family != "gpt2":
        raise ValueError(
            "TTR bridge evaluation currently supports GPT-2 AKAZA checkpoints only; "
            f"got model_family={cfg.model_family!r}."
        )
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

    layer_indices = summary.get("layer_indices") or parse_int_csv(cfg.layer_indices)
    layer_indices = sorted(int(x) for x in layer_indices)

    candidate_names = parse_csv(args.candidate_learners)
    if not candidate_names:
        raise ValueError("candidate_learners must be non-empty")
    unknown = sorted(set(candidate_names) - set(CANONICAL_LEARNERS))
    if unknown:
        raise ValueError(
            f"Unknown candidate learner(s): {unknown}. "
            f"Imported canonical registry contains: {CANONICAL_LEARNERS}."
        )
    if "soft" in candidate_names:
        raise ValueError("Do not include 'soft' as a candidate; it is the frozen reference readout.")

    learner_hparams = LearnerHyperParams(
        local_kernel_beta=args.local_kernel_beta,
        window_size=args.window_size,
        k_knn_mean=args.k_knn_mean,
        ridge_lambda=args.learner_ridge_lambda,
        k_linear_local=args.k_linear_local,
        k_sharp=args.k_sharp,
    )

    device = torch.device(cfg.device)
    print("[loaded]")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  layer_indices: {layer_indices}")
    print(f"  candidate_learners: {candidate_names}")
    print(f"  eval_split: {args.eval_split}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped = GPT2AKAZAAdapter(model=model, cfg=cfg, layer_indices=layer_indices).to(device)
    TrainableParameters(params=[], frozen_before_training={}, check_frozen=False).load_trainable_state_dict(
        wrapped,
        payload["trainable_state_dict"],
    )
    wrapped.set_peft_eval_mode()

    split_name = {
        "train": cfg.train_split,
        "val": cfg.val_split,
        "test": cfg.test_split,
    }[args.eval_split]
    max_chunks = {
        "train": cfg.max_train_chunks,
        "val": cfg.max_val_chunks,
        "test": cfg.max_test_chunks,
    }[args.eval_split]

    chunks = load_chunks_for_split(cfg, tokenizer, split=split_name, max_chunks=max_chunks)
    print(f"[data] {args.eval_split}: chunks={chunks.shape[0]} block_size={chunks.shape[1]}")

    metrics = eval_ttr_bridge(
        model=model,
        wrapped=wrapped,
        chunks=chunks,
        layer_indices=layer_indices,
        candidate_names=candidate_names,
        learner_hparams=learner_hparams,
        batch_size=cfg.batch_size,
        device=device,
        ridge_lambda=args.ridge_lambda,
        eps=args.eps,
        token_csv_path=args.token_csv_path,
    )

    results: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source_summary_best_epoch": summary.get("best_epoch"),
        "source_summary_best_val_loss": summary.get("best_val_loss"),
        "source_summary_best_test_loss": summary.get("best_test_loss"),
        "config": jsonable_dataclass_dict(cfg),
        "eval_split": args.eval_split,
        "dataset_split": split_name,
        "n_chunks": int(chunks.shape[0]),
        "block_size": int(chunks.shape[1]),
        "layer_indices": layer_indices,
        "candidate_learners": candidate_names,
        "canonical_learner_hparams": asdict(learner_hparams),
        "bridge_definition": {
            "G_TTR": "max_a(base_token_nll - token_nll_after_replacing_layer_z_with_candidate_readout_a)",
            "G_AKAZA": "base_token_nll - token_nll_after_full_trained_AKAZA_forward",
            "delta_WO_norm": "norm(attn.c_proj(delta) - attn.c_proj(0)) at the same layer/token",
            "token_positions": "next-token positions 0..block_size-2",
        },
        **metrics,
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    if args.layer_csv_path:
        write_layer_csv(args.layer_csv_path, metrics["layers"])

    print("[summary]")
    print(f"  corr(G_TTR, ||delta W_O||): {metrics['global']['corr_G_TTR_delta_WO_norm']:.6f}")
    print(f"  corr(G_TTR, G_AKAZA): {metrics['global']['corr_G_TTR_G_AKAZA']:.6f}")
    print(f"  wrote: {out}")
    if args.token_csv_path:
        print(f"  token_csv: {args.token_csv_path}")
    if args.layer_csv_path:
        print(f"  layer_csv: {args.layer_csv_path}")


if __name__ == "__main__":
    main()
