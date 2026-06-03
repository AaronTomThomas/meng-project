from __future__ import annotations

"""
Token-level geometric bridge between beneficial counterfactual learner readouts
and trained AKAZA/FreeZ corrections.

Question:
    For each layer/token/candidate learner, if replacing z_soft with the
    candidate readout z_a improves frozen-model next-token NLL, is z_a closer
    to the trained AKAZA-corrected readout z_soft + delta than z_soft itself?

This is intentionally GPT-2-specific and mirrors the plumbing used by the
existing TTR bridge probe:
    - load a trained GPT-2 AKAZA/FreeZ checkpoint,
    - compute candidate readouts from frozen q/K/V geometry,
    - patch each candidate readout into the frozen model,
    - compare each candidate direction d_a = z_a - z_soft to the trained
      AKAZA direction delta.

The key metrics are computed in both:
    1. pre-projection z-space, and
    2. post-output-projection residual-effect space, using frozen attn.c_proj.

A candidate is "closer than soft" when:

    ||z_a - z_AKAZA|| < ||z_soft - z_AKAZA||

equivalently:

    ||d_a - delta|| < ||delta||

and similarly after the frozen output projection W_O.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.attention_learners import LearnerHyperParams, build_learners
from experiments.gpt2_probe_utils import (
    continue_from_modified_block_gpt2,
    extract_head_qkv_and_teacher_outputs_gpt2,
    get_block_input_gpt2,
)
from experiments.router_development.attention_adapter.adapters.akaza_adapters import (
    GPT2AKAZAAdapter,
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
)
from experiments.router_development.attention_adapter.trainer import TrainableParameters


METRIC_NAMES = [
    "candidate_gain",
    "is_beneficial",
    "G_AKAZA",
    "delta_norm_z",
    "delta_norm_WO",
    "candidate_direction_norm_z",
    "candidate_direction_norm_WO",
    "soft_to_akaza_z",
    "candidate_to_akaza_z",
    "distance_ratio_z",
    "relative_closeness_z",
    "closer_than_soft_z",
    "cosine_to_akaza_z",
    "projection_frac_onto_akaza_z",
    "soft_to_akaza_WO",
    "candidate_to_akaza_WO",
    "distance_ratio_WO",
    "relative_closeness_WO",
    "closer_than_soft_WO",
    "cosine_to_akaza_WO",
    "projection_frac_onto_akaza_WO",
]

CORR_PAIRS = [
    ("candidate_gain", "relative_closeness_z"),
    ("candidate_gain", "relative_closeness_WO"),
    ("candidate_gain", "cosine_to_akaza_z"),
    ("candidate_gain", "cosine_to_akaza_WO"),
    ("candidate_gain", "closer_than_soft_z"),
    ("candidate_gain", "closer_than_soft_WO"),
]


def token_next_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return next-token NLL for positions 0..T-2 with shape [B,T-1]."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def c_proj_linear_delta(block, delta: torch.Tensor) -> torch.Tensor:
    """
    Apply GPT-2 attn.c_proj to a delta without counting the projection bias.

    GPT-2's Conv1D projection supports arbitrary leading dimensions, so this
    works for [B,T,D] and [B,T,A,D].
    """
    zeros = torch.zeros_like(delta)
    return block.attn.c_proj(delta) - block.attn.c_proj(zeros)


def safe_div(num: torch.Tensor, den: torch.Tensor, eps: float) -> torch.Tensor:
    return num / den.clamp_min(eps)


class RunningStats:
    """Streaming means and Pearson correlations for masked tensor metrics."""

    def __init__(self) -> None:
        self.n_group = 0
        self.metric_sum: DefaultDict[str, float] = defaultdict(float)
        self.metric_count: DefaultDict[str, int] = defaultdict(int)
        self.pair_stats: DefaultDict[tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {
                "n": 0.0,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "sum_x2": 0.0,
                "sum_y2": 0.0,
                "sum_xy": 0.0,
            }
        )

    def add(self, metrics: Mapping[str, torch.Tensor], mask: torch.Tensor) -> None:
        mask = mask.bool()
        self.n_group += int(mask.sum().item())

        for name in METRIC_NAMES:
            value = metrics[name]
            valid = mask & torch.isfinite(value)
            if not bool(valid.any()):
                continue
            selected = value[valid].double()
            self.metric_sum[name] += float(selected.sum().item())
            self.metric_count[name] += int(selected.numel())

        for x_name, y_name in CORR_PAIRS:
            x = metrics[x_name]
            y = metrics[y_name]
            valid = mask & torch.isfinite(x) & torch.isfinite(y)
            if not bool(valid.any()):
                continue
            xv = x[valid].double()
            yv = y[valid].double()
            s = self.pair_stats[(x_name, y_name)]
            s["n"] += float(xv.numel())
            s["sum_x"] += float(xv.sum().item())
            s["sum_y"] += float(yv.sum().item())
            s["sum_x2"] += float(xv.square().sum().item())
            s["sum_y2"] += float(yv.square().sum().item())
            s["sum_xy"] += float((xv * yv).sum().item())

    def _mean(self, name: str) -> float:
        count = self.metric_count.get(name, 0)
        if count == 0:
            return float("nan")
        return self.metric_sum[name] / float(count)

    def _pearson(self, x_name: str, y_name: str) -> float:
        s = self.pair_stats[(x_name, y_name)]
        n = s["n"]
        if n < 2:
            return float("nan")

        cov = s["sum_xy"] - (s["sum_x"] * s["sum_y"] / n)
        var_x = s["sum_x2"] - (s["sum_x"] * s["sum_x"] / n)
        var_y = s["sum_y2"] - (s["sum_y"] * s["sum_y"] / n)
        denom = math.sqrt(max(var_x, 0.0)) * math.sqrt(max(var_y, 0.0))
        if denom <= 0.0:
            return float("nan")
        return cov / denom

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": int(self.n_group)}
        for name in METRIC_NAMES:
            out[f"mean_{name}"] = self._mean(name)

        for x_name, y_name in CORR_PAIRS:
            out[f"pearson_{x_name}_vs_{y_name}"] = self._pearson(x_name, y_name)

        # Friendlier aliases for headline quantities.
        out["frac_beneficial"] = out["mean_is_beneficial"]
        out["frac_closer_than_soft_z"] = out["mean_closer_than_soft_z"]
        out["frac_closer_than_soft_WO"] = out["mean_closer_than_soft_WO"]
        return out


class NestedAccumulator:
    """
    Stores streaming summaries for:
      - global
      - per layer
      - per learner
      - per layer/learner

    Each scope is split into:
      - all candidate-token pairs
      - beneficial pairs, candidate_gain > threshold
      - nonbeneficial pairs, candidate_gain <= threshold
    """

    def __init__(self) -> None:
        self.global_stats: DefaultDict[str, RunningStats] = defaultdict(RunningStats)
        self.layer_stats: DefaultDict[str, DefaultDict[str, RunningStats]] = defaultdict(
            lambda: defaultdict(RunningStats)
        )
        self.learner_stats: DefaultDict[str, DefaultDict[str, RunningStats]] = defaultdict(
            lambda: defaultdict(RunningStats)
        )
        self.layer_learner_stats: DefaultDict[
            str, DefaultDict[str, DefaultDict[str, RunningStats]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(RunningStats)))

    def add(
        self,
        *,
        layer_idx: int,
        learner_name: str,
        metrics: Mapping[str, torch.Tensor],
        beneficial_mask: torch.Tensor,
    ) -> None:
        all_mask = torch.ones_like(beneficial_mask, dtype=torch.bool)
        groups = {
            "all": all_mask,
            "beneficial": beneficial_mask,
            "nonbeneficial": ~beneficial_mask,
        }

        layer_key = str(int(layer_idx))
        for group_name, mask in groups.items():
            self.global_stats[group_name].add(metrics, mask)
            self.layer_stats[layer_key][group_name].add(metrics, mask)
            self.learner_stats[learner_name][group_name].add(metrics, mask)
            self.layer_learner_stats[layer_key][learner_name][group_name].add(metrics, mask)

    def summary(self) -> Dict[str, Any]:
        return {
            "global": {
                group: stats.summary()
                for group, stats in sorted(self.global_stats.items())
            },
            "layers": {
                layer: {
                    group: stats.summary()
                    for group, stats in sorted(group_stats.items())
                }
                for layer, group_stats in sorted(
                    self.layer_stats.items(), key=lambda kv: int(kv[0])
                )
            },
            "learners": {
                learner: {
                    group: stats.summary()
                    for group, stats in sorted(group_stats.items())
                }
                for learner, group_stats in sorted(self.learner_stats.items())
            },
            "layer_learners": {
                layer: {
                    learner: {
                        group: stats.summary()
                        for group, stats in sorted(group_stats.items())
                    }
                    for learner, group_stats in sorted(learner_stats.items())
                }
                for layer, learner_stats in sorted(
                    self.layer_learner_stats.items(), key=lambda kv: int(kv[0])
                )
            },
        }


def build_metric_tensors(
    *,
    candidate_gain: torch.Tensor,
    g_akaza: torch.Tensor,
    direction_tok: torch.Tensor,
    delta_tok: torch.Tensor,
    direction_wo_tok: torch.Tensor,
    delta_wo_tok: torch.Tensor,
    eps: float,
    gain_threshold: float,
) -> Dict[str, torch.Tensor]:
    """
    All returned tensors have shape [B,T-1].
    """

    delta_norm_z = delta_tok.float().norm(dim=-1)
    delta_norm_wo = delta_wo_tok.float().norm(dim=-1)
    direction_norm_z = direction_tok.float().norm(dim=-1)
    direction_norm_wo = direction_wo_tok.float().norm(dim=-1)

    candidate_to_akaza_z = (direction_tok.float() - delta_tok.float()).norm(dim=-1)
    candidate_to_akaza_wo = (direction_wo_tok.float() - delta_wo_tok.float()).norm(dim=-1)

    distance_ratio_z = safe_div(candidate_to_akaza_z, delta_norm_z, eps)
    distance_ratio_wo = safe_div(candidate_to_akaza_wo, delta_norm_wo, eps)

    relative_closeness_z = 1.0 - distance_ratio_z
    relative_closeness_wo = 1.0 - distance_ratio_wo

    closer_z = (candidate_to_akaza_z < delta_norm_z).float()
    closer_wo = (candidate_to_akaza_wo < delta_norm_wo).float()

    cosine_z = F.cosine_similarity(
        direction_tok.float(),
        delta_tok.float(),
        dim=-1,
        eps=eps,
    )
    cosine_wo = F.cosine_similarity(
        direction_wo_tok.float(),
        delta_wo_tok.float(),
        dim=-1,
        eps=eps,
    )

    projection_frac_z = (
        (direction_tok.float() * delta_tok.float()).sum(dim=-1)
        / delta_tok.float().square().sum(dim=-1).clamp_min(eps)
    )
    projection_frac_wo = (
        (direction_wo_tok.float() * delta_wo_tok.float()).sum(dim=-1)
        / delta_wo_tok.float().square().sum(dim=-1).clamp_min(eps)
    )

    is_beneficial = (candidate_gain > gain_threshold).float()

    return {
        "candidate_gain": candidate_gain.float(),
        "is_beneficial": is_beneficial,
        "G_AKAZA": g_akaza.float(),
        "delta_norm_z": delta_norm_z,
        "delta_norm_WO": delta_norm_wo,
        "candidate_direction_norm_z": direction_norm_z,
        "candidate_direction_norm_WO": direction_norm_wo,
        "soft_to_akaza_z": delta_norm_z,
        "candidate_to_akaza_z": candidate_to_akaza_z,
        "distance_ratio_z": distance_ratio_z,
        "relative_closeness_z": relative_closeness_z,
        "closer_than_soft_z": closer_z,
        "cosine_to_akaza_z": cosine_z,
        "projection_frac_onto_akaza_z": projection_frac_z,
        "soft_to_akaza_WO": delta_norm_wo,
        "candidate_to_akaza_WO": candidate_to_akaza_wo,
        "distance_ratio_WO": distance_ratio_wo,
        "relative_closeness_WO": relative_closeness_wo,
        "closer_than_soft_WO": closer_wo,
        "cosine_to_akaza_WO": cosine_wo,
        "projection_frac_onto_akaza_WO": projection_frac_wo,
    }


def maybe_write_token_rows(
    *,
    token_writer: csv.DictWriter | None,
    start: int,
    layer_idx: int,
    learner_name: str,
    metrics: Mapping[str, torch.Tensor],
) -> None:
    if token_writer is None:
        return

    bsz, t_minus_1 = metrics["candidate_gain"].shape
    cpu_metrics = {
        name: value.detach().cpu()
        for name, value in metrics.items()
        if name in {
            "candidate_gain",
            "is_beneficial",
            "G_AKAZA",
            "delta_norm_z",
            "delta_norm_WO",
            "candidate_direction_norm_z",
            "candidate_direction_norm_WO",
            "relative_closeness_z",
            "relative_closeness_WO",
            "closer_than_soft_z",
            "closer_than_soft_WO",
            "cosine_to_akaza_z",
            "cosine_to_akaza_WO",
            "projection_frac_onto_akaza_z",
            "projection_frac_onto_akaza_WO",
            "distance_ratio_z",
            "distance_ratio_WO",
        }
    }

    for b in range(bsz):
        chunk_idx = start + b
        for pos in range(t_minus_1):
            row = {
                "chunk_idx": chunk_idx,
                "layer_idx": int(layer_idx),
                "token_pos": int(pos),
                "learner": learner_name,
            }
            for name, tensor in cpu_metrics.items():
                row[name] = float(tensor[b, pos].item())
            token_writer.writerow(row)


@torch.no_grad()
def eval_akaza_learner_closeness(
    *,
    model,
    wrapped: GPT2AKAZAAdapter,
    chunks: torch.Tensor,
    layer_indices: Sequence[int],
    candidate_names: Sequence[str],
    learner_hparams: LearnerHyperParams,
    batch_size: int,
    device: torch.device,
    eps: float,
    gain_threshold: float,
    token_csv_path: str | None = None,
) -> Dict[str, Any]:
    learner_instances = build_learners(list(candidate_names))
    acc = NestedAccumulator()

    token_csv_fh = None
    token_writer = None
    if token_csv_path:
        token_path = Path(token_csv_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_csv_fh = token_path.open("w", newline="")
        token_writer = csv.DictWriter(
            token_csv_fh,
            fieldnames=[
                "chunk_idx",
                "layer_idx",
                "token_pos",
                "learner",
                "candidate_gain",
                "is_beneficial",
                "G_AKAZA",
                "delta_norm_z",
                "delta_norm_WO",
                "candidate_direction_norm_z",
                "candidate_direction_norm_WO",
                "relative_closeness_z",
                "relative_closeness_WO",
                "closer_than_soft_z",
                "closer_than_soft_WO",
                "cosine_to_akaza_z",
                "cosine_to_akaza_WO",
                "projection_frac_onto_akaza_z",
                "projection_frac_onto_akaza_WO",
                "distance_ratio_z",
                "distance_ratio_WO",
            ],
        )
        token_writer.writeheader()

    try:
        n_examples = int(chunks.shape[0])
        for start in range(0, n_examples, batch_size):
            input_ids = chunks[start : start + batch_size].to(device)
            bsz, _seq_len = input_ids.shape

            baseline_logits = model(input_ids=input_ids, use_cache=False).logits
            baseline_nll = token_next_nll(baseline_logits, input_ids)

            wrapped.set_peft_eval_mode()
            akaza_logits = wrapped(input_ids)
            akaza_nll = token_next_nll(akaza_logits, input_ids)
            g_akaza = baseline_nll - akaza_nll

            latest_deltas = {int(k): v.detach() for k, v in wrapped._latest_deltas.items()}

            for layer_idx_raw in layer_indices:
                layer_idx = int(layer_idx_raw)
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

                delta = latest_deltas[layer_idx].to(device=device, dtype=zcat_teacher.dtype)
                delta_tok = delta[:, :-1, :]
                delta_wo = c_proj_linear_delta(block, delta)
                delta_wo_tok = delta_wo[:, :-1, :]

                for learner_name in candidate_names:
                    z_candidate = candidate_outputs[learner_name].to(dtype=zcat_teacher.dtype)

                    logits_candidate = continue_from_modified_block_gpt2(
                        model=model,
                        block=block,
                        x_in=x_in,
                        zcat_mod=z_candidate,
                        layer_idx=layer_idx,
                    )
                    candidate_nll = token_next_nll(logits_candidate, input_ids)
                    candidate_gain = baseline_nll - candidate_nll

                    direction = z_candidate - zcat_teacher
                    direction_tok = direction[:, :-1, :]
                    direction_wo = c_proj_linear_delta(block, direction)
                    direction_wo_tok = direction_wo[:, :-1, :]

                    metrics = build_metric_tensors(
                        candidate_gain=candidate_gain,
                        g_akaza=g_akaza,
                        direction_tok=direction_tok,
                        delta_tok=delta_tok,
                        direction_wo_tok=direction_wo_tok,
                        delta_wo_tok=delta_wo_tok,
                        eps=eps,
                        gain_threshold=gain_threshold,
                    )

                    beneficial_mask = candidate_gain > gain_threshold
                    acc.add(
                        layer_idx=layer_idx,
                        learner_name=learner_name,
                        metrics=metrics,
                        beneficial_mask=beneficial_mask,
                    )

                    maybe_write_token_rows(
                        token_writer=token_writer,
                        start=start,
                        layer_idx=layer_idx,
                        learner_name=learner_name,
                        metrics=metrics,
                    )

            print(f"[eval] processed chunks {start + bsz}/{n_examples}")
    finally:
        if token_csv_fh is not None:
            token_csv_fh.close()

    return acc.summary()


def write_layer_learner_csv(path: str, layer_learners: Mapping[str, Any]) -> None:
    fieldnames = [
        "layer_idx",
        "learner",
        "group",
        "n",
        "mean_candidate_gain",
        "frac_beneficial",
        "mean_G_AKAZA",
        "mean_delta_norm_z",
        "mean_delta_norm_WO",
        "mean_candidate_direction_norm_z",
        "mean_candidate_direction_norm_WO",
        "frac_closer_than_soft_z",
        "frac_closer_than_soft_WO",
        "mean_relative_closeness_z",
        "mean_relative_closeness_WO",
        "mean_cosine_to_akaza_z",
        "mean_cosine_to_akaza_WO",
        "mean_projection_frac_onto_akaza_z",
        "mean_projection_frac_onto_akaza_WO",
        "pearson_candidate_gain_vs_relative_closeness_z",
        "pearson_candidate_gain_vs_relative_closeness_WO",
        "pearson_candidate_gain_vs_cosine_to_akaza_z",
        "pearson_candidate_gain_vs_cosine_to_akaza_WO",
    ]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for layer_idx, learner_map in sorted(layer_learners.items(), key=lambda kv: int(kv[0])):
            for learner_name, group_map in sorted(learner_map.items()):
                for group_name, s in sorted(group_map.items()):
                    writer.writerow(
                        {
                            "layer_idx": int(layer_idx),
                            "learner": learner_name,
                            "group": group_name,
                            "n": s["n"],
                            "mean_candidate_gain": s["mean_candidate_gain"],
                            "frac_beneficial": s["frac_beneficial"],
                            "mean_G_AKAZA": s["mean_G_AKAZA"],
                            "mean_delta_norm_z": s["mean_delta_norm_z"],
                            "mean_delta_norm_WO": s["mean_delta_norm_WO"],
                            "mean_candidate_direction_norm_z": s["mean_candidate_direction_norm_z"],
                            "mean_candidate_direction_norm_WO": s["mean_candidate_direction_norm_WO"],
                            "frac_closer_than_soft_z": s["frac_closer_than_soft_z"],
                            "frac_closer_than_soft_WO": s["frac_closer_than_soft_WO"],
                            "mean_relative_closeness_z": s["mean_relative_closeness_z"],
                            "mean_relative_closeness_WO": s["mean_relative_closeness_WO"],
                            "mean_cosine_to_akaza_z": s["mean_cosine_to_akaza_z"],
                            "mean_cosine_to_akaza_WO": s["mean_cosine_to_akaza_WO"],
                            "mean_projection_frac_onto_akaza_z": s[
                                "mean_projection_frac_onto_akaza_z"
                            ],
                            "mean_projection_frac_onto_akaza_WO": s[
                                "mean_projection_frac_onto_akaza_WO"
                            ],
                            "pearson_candidate_gain_vs_relative_closeness_z": s[
                                "pearson_candidate_gain_vs_relative_closeness_z"
                            ],
                            "pearson_candidate_gain_vs_relative_closeness_WO": s[
                                "pearson_candidate_gain_vs_relative_closeness_WO"
                            ],
                            "pearson_candidate_gain_vs_cosine_to_akaza_z": s[
                                "pearson_candidate_gain_vs_cosine_to_akaza_z"
                            ],
                            "pearson_candidate_gain_vs_cosine_to_akaza_WO": s[
                                "pearson_candidate_gain_vs_cosine_to_akaza_WO"
                            ],
                        }
                    )


def print_headline(metrics: Mapping[str, Any]) -> None:
    g = metrics["global"]
    all_s = g.get("all", {})
    ben_s = g.get("beneficial", {})
    non_s = g.get("nonbeneficial", {})

    def get(s: Mapping[str, Any], key: str) -> float:
        value = s.get(key, float("nan"))
        return float(value) if value is not None else float("nan")

    print("[summary]")
    print(f"  all pairs: n={int(all_s.get('n', 0))}")
    print(f"  beneficial fraction: {get(all_s, 'frac_beneficial'):.6f}")
    print("  closer-than-soft fraction:")
    print(f"    all          z={get(all_s, 'frac_closer_than_soft_z'):.6f}  WO={get(all_s, 'frac_closer_than_soft_WO'):.6f}")
    print(f"    beneficial  z={get(ben_s, 'frac_closer_than_soft_z'):.6f}  WO={get(ben_s, 'frac_closer_than_soft_WO'):.6f}")
    print(f"    nonbenef    z={get(non_s, 'frac_closer_than_soft_z'):.6f}  WO={get(non_s, 'frac_closer_than_soft_WO'):.6f}")
    print("  mean cosine to AKAZA:")
    print(f"    beneficial  z={get(ben_s, 'mean_cosine_to_akaza_z'):.6f}  WO={get(ben_s, 'mean_cosine_to_akaza_WO'):.6f}")
    print(f"    nonbenef    z={get(non_s, 'mean_cosine_to_akaza_z'):.6f}  WO={get(non_s, 'mean_cosine_to_akaza_WO'):.6f}")


def main() -> None:
    default_hp = LearnerHyperParams()

    parser = argparse.ArgumentParser(
        description="Evaluate whether beneficial learner readouts move closer to trained AKAZA corrections."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--layer_learner_csv_path", type=str, default=None)
    parser.add_argument("--token_csv_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_train_chunks", type=int, default=None)
    parser.add_argument("--max_val_chunks", type=int, default=None)
    parser.add_argument("--max_test_chunks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate_learners", type=str, default=",".join(DEFAULT_CANDIDATE_LEARNERS))
    parser.add_argument("--gain_threshold", type=float, default=0.0)
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
            "This closeness probe currently supports GPT-2 AKAZA checkpoints only; "
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
        raise ValueError("Do not include 'soft' as a candidate; it is the reference readout.")

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
    print(f"  gain_threshold: {args.gain_threshold}")

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

    metrics = eval_akaza_learner_closeness(
        model=model,
        wrapped=wrapped,
        chunks=chunks,
        layer_indices=layer_indices,
        candidate_names=candidate_names,
        learner_hparams=learner_hparams,
        batch_size=cfg.batch_size,
        device=device,
        eps=args.eps,
        gain_threshold=args.gain_threshold,
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
        "gain_threshold": args.gain_threshold,
        "definitions": {
            "candidate_gain": "base_token_nll - token_nll_after_replacing_layer_z_with_candidate_readout",
            "beneficial": "candidate_gain > gain_threshold",
            "delta": "trained AKAZA correction at the same layer/token",
            "closer_than_soft_z": "||d_a - delta|| < ||delta||, where d_a = z_candidate - z_soft",
            "relative_closeness_z": "1 - ||d_a - delta|| / ||delta||",
            "closer_than_soft_WO": "same test after applying frozen attn.c_proj linearly to d_a and delta",
            "relative_closeness_WO": "1 - ||d_a W_O - delta W_O|| / ||delta W_O||",
            "token_positions": "next-token positions 0..block_size-2",
        },
        **metrics,
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    if args.layer_learner_csv_path:
        write_layer_learner_csv(args.layer_learner_csv_path, metrics["layer_learners"])

    print_headline(metrics)
    print(f"  wrote: {out}")
    if args.layer_learner_csv_path:
        print(f"  layer_learner_csv: {args.layer_learner_csv_path}")
    if args.token_csv_path:
        print(f"  token_csv: {args.token_csv_path}")


if __name__ == "__main__":
    main()
