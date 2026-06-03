from __future__ import annotations

"""
Token-level geometric bridge between counterfactual learner readouts and
trained AKAZA/FreeZ corrections.

Main question:
    For each layer/token/candidate learner, when replacing z_soft with a
    candidate readout z_a improves frozen-model next-token NLL, does the
    candidate movement d_a = z_a - z_soft point toward the trained AKAZA
    correction delta?

This script reports two families of metrics:

1. Endpoint closeness:
       ||d_a - delta|| < ||delta||

   This asks whether the actual candidate endpoint z_a is closer to the
   AKAZA-corrected readout z_soft + delta than z_soft itself is.

   This is strict and can fail even if d_a points in the right direction but
   has the wrong magnitude.

2. Ray / scale-free closeness:
       alpha* = <d_a, delta> / ||d_a||^2
       ||alpha* d_a - delta|| < ||delta||

   This asks whether the candidate learner direction spans a ray that points
   toward the AKAZA correction, allowing for an optimal scalar rescaling.

Both metric families are computed in:
    - pre-projection z-space,
    - post-output-projection residual-effect space, after frozen attn.c_proj.

This is intentionally GPT-2-specific and uses the same checkpoint/config/data
plumbing as the existing bridge probes.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Mapping, Sequence

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

    # Endpoint metrics.
    "endpoint_distance_ratio_z",
    "endpoint_distance_ratio_WO",
    "endpoint_relative_closeness_z",
    "endpoint_relative_closeness_WO",
    "endpoint_closer_than_soft_z",
    "endpoint_closer_than_soft_WO",

    # Directional metrics.
    "cosine_to_akaza_z",
    "cosine_to_akaza_WO",
    "projection_frac_onto_akaza_z",
    "projection_frac_onto_akaza_WO",

    # Ray/scale-free metrics.
    "ray_alpha_z",
    "ray_alpha_WO",
    "ray_alpha_positive_z",
    "ray_alpha_positive_WO",
    "ray_distance_ratio_z",
    "ray_distance_ratio_WO",
    "ray_relative_closeness_z",
    "ray_relative_closeness_WO",
    "ray_closer_than_soft_z",
    "ray_closer_than_soft_WO",
    "positive_ray_closer_than_soft_z",
    "positive_ray_closer_than_soft_WO",
]

CORR_PAIRS = [
    ("candidate_gain", "endpoint_relative_closeness_z"),
    ("candidate_gain", "endpoint_relative_closeness_WO"),
    ("candidate_gain", "endpoint_closer_than_soft_z"),
    ("candidate_gain", "endpoint_closer_than_soft_WO"),

    ("candidate_gain", "cosine_to_akaza_z"),
    ("candidate_gain", "cosine_to_akaza_WO"),

    ("candidate_gain", "ray_relative_closeness_z"),
    ("candidate_gain", "ray_relative_closeness_WO"),
    ("candidate_gain", "ray_closer_than_soft_z"),
    ("candidate_gain", "ray_closer_than_soft_WO"),
    ("candidate_gain", "positive_ray_closer_than_soft_z"),
    ("candidate_gain", "positive_ray_closer_than_soft_WO"),
]


def token_next_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return next-token NLL for positions 0..T-2 with shape [B,T-1]."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def c_proj_linear_delta(block, delta: torch.Tensor) -> torch.Tensor:
    """
    Apply GPT-2 attn.c_proj to a delta without counting the projection bias.

    GPT-2's Conv1D projection accepts arbitrary leading dimensions, so this
    works for [B,T,D] and [B,T,A,D].
    """
    zeros = torch.zeros_like(delta)
    return block.attn.c_proj(delta) - block.attn.c_proj(zeros)


def safe_div(num: torch.Tensor, den: torch.Tensor, eps: float) -> torch.Tensor:
    return num / den.clamp_min(eps)


def endpoint_and_ray_metrics(
    *,
    direction: torch.Tensor,
    delta: torch.Tensor,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute endpoint and ray metrics for tensors of shape [B,T,D].

    direction = d_a = z_candidate - z_soft
    delta     = trained AKAZA correction
    """
    direction = direction.float()
    delta = delta.float()

    delta_norm = delta.norm(dim=-1)
    direction_norm = direction.norm(dim=-1)

    endpoint_dist = (direction - delta).norm(dim=-1)
    endpoint_ratio = safe_div(endpoint_dist, delta_norm, eps)
    endpoint_rel = 1.0 - endpoint_ratio
    endpoint_closer = (endpoint_dist < delta_norm).float()

    cosine = F.cosine_similarity(direction, delta, dim=-1, eps=eps)
    projection_frac = (
        (direction * delta).sum(dim=-1)
        / delta.square().sum(dim=-1).clamp_min(eps)
    )

    # Best scalar alpha that maps direction to delta in least squares:
    #     min_alpha ||alpha direction - delta||^2
    alpha = (
        (direction * delta).sum(dim=-1)
        / direction.square().sum(dim=-1).clamp_min(eps)
    )
    ray_point = alpha.unsqueeze(-1) * direction
    ray_dist = (ray_point - delta).norm(dim=-1)
    ray_ratio = safe_div(ray_dist, delta_norm, eps)
    ray_rel = 1.0 - ray_ratio
    ray_closer = (ray_dist < delta_norm).float()

    alpha_positive = (alpha > 0).float()
    positive_ray_closer = ((alpha > 0) & (ray_dist < delta_norm)).float()

    return {
        "delta_norm": delta_norm,
        "direction_norm": direction_norm,

        "endpoint_distance_ratio": endpoint_ratio,
        "endpoint_relative_closeness": endpoint_rel,
        "endpoint_closer_than_soft": endpoint_closer,

        "cosine_to_akaza": cosine,
        "projection_frac_onto_akaza": projection_frac,

        "ray_alpha": alpha,
        "ray_alpha_positive": alpha_positive,
        "ray_distance_ratio": ray_ratio,
        "ray_relative_closeness": ray_rel,
        "ray_closer_than_soft": ray_closer,
        "positive_ray_closer_than_soft": positive_ray_closer,
    }


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

        out["frac_beneficial"] = out["mean_is_beneficial"]

        out["frac_endpoint_closer_than_soft_z"] = out["mean_endpoint_closer_than_soft_z"]
        out["frac_endpoint_closer_than_soft_WO"] = out["mean_endpoint_closer_than_soft_WO"]

        out["frac_ray_alpha_positive_z"] = out["mean_ray_alpha_positive_z"]
        out["frac_ray_alpha_positive_WO"] = out["mean_ray_alpha_positive_WO"]

        out["frac_ray_closer_than_soft_z"] = out["mean_ray_closer_than_soft_z"]
        out["frac_ray_closer_than_soft_WO"] = out["mean_ray_closer_than_soft_WO"]

        out["frac_positive_ray_closer_than_soft_z"] = out[
            "mean_positive_ray_closer_than_soft_z"
        ]
        out["frac_positive_ray_closer_than_soft_WO"] = out[
            "mean_positive_ray_closer_than_soft_WO"
        ]

        return out


class NestedAccumulator:
    """
    Streaming summaries for:
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

    z_metrics = endpoint_and_ray_metrics(
        direction=direction_tok,
        delta=delta_tok,
        eps=eps,
    )
    wo_metrics = endpoint_and_ray_metrics(
        direction=direction_wo_tok,
        delta=delta_wo_tok,
        eps=eps,
    )

    is_beneficial = (candidate_gain > gain_threshold).float()

    return {
        "candidate_gain": candidate_gain.float(),
        "is_beneficial": is_beneficial,
        "G_AKAZA": g_akaza.float(),

        "delta_norm_z": z_metrics["delta_norm"],
        "delta_norm_WO": wo_metrics["delta_norm"],
        "candidate_direction_norm_z": z_metrics["direction_norm"],
        "candidate_direction_norm_WO": wo_metrics["direction_norm"],

        "endpoint_distance_ratio_z": z_metrics["endpoint_distance_ratio"],
        "endpoint_distance_ratio_WO": wo_metrics["endpoint_distance_ratio"],
        "endpoint_relative_closeness_z": z_metrics["endpoint_relative_closeness"],
        "endpoint_relative_closeness_WO": wo_metrics["endpoint_relative_closeness"],
        "endpoint_closer_than_soft_z": z_metrics["endpoint_closer_than_soft"],
        "endpoint_closer_than_soft_WO": wo_metrics["endpoint_closer_than_soft"],

        "cosine_to_akaza_z": z_metrics["cosine_to_akaza"],
        "cosine_to_akaza_WO": wo_metrics["cosine_to_akaza"],
        "projection_frac_onto_akaza_z": z_metrics["projection_frac_onto_akaza"],
        "projection_frac_onto_akaza_WO": wo_metrics["projection_frac_onto_akaza"],

        "ray_alpha_z": z_metrics["ray_alpha"],
        "ray_alpha_WO": wo_metrics["ray_alpha"],
        "ray_alpha_positive_z": z_metrics["ray_alpha_positive"],
        "ray_alpha_positive_WO": wo_metrics["ray_alpha_positive"],
        "ray_distance_ratio_z": z_metrics["ray_distance_ratio"],
        "ray_distance_ratio_WO": wo_metrics["ray_distance_ratio"],
        "ray_relative_closeness_z": z_metrics["ray_relative_closeness"],
        "ray_relative_closeness_WO": wo_metrics["ray_relative_closeness"],
        "ray_closer_than_soft_z": z_metrics["ray_closer_than_soft"],
        "ray_closer_than_soft_WO": wo_metrics["ray_closer_than_soft"],
        "positive_ray_closer_than_soft_z": z_metrics["positive_ray_closer_than_soft"],
        "positive_ray_closer_than_soft_WO": wo_metrics["positive_ray_closer_than_soft"],
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
    cpu_metrics = {name: value.detach().cpu() for name, value in metrics.items()}

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
                *METRIC_NAMES,
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

                if layer_idx not in latest_deltas:
                    raise KeyError(
                        f"AKAZA wrapper did not record a delta for layer {layer_idx}. "
                        f"Recorded layers: {sorted(latest_deltas)}"
                    )

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

        "frac_endpoint_closer_than_soft_z",
        "frac_endpoint_closer_than_soft_WO",
        "mean_endpoint_relative_closeness_z",
        "mean_endpoint_relative_closeness_WO",

        "mean_cosine_to_akaza_z",
        "mean_cosine_to_akaza_WO",
        "mean_projection_frac_onto_akaza_z",
        "mean_projection_frac_onto_akaza_WO",

        "mean_ray_alpha_z",
        "mean_ray_alpha_WO",
        "frac_ray_alpha_positive_z",
        "frac_ray_alpha_positive_WO",
        "frac_ray_closer_than_soft_z",
        "frac_ray_closer_than_soft_WO",
        "frac_positive_ray_closer_than_soft_z",
        "frac_positive_ray_closer_than_soft_WO",
        "mean_ray_relative_closeness_z",
        "mean_ray_relative_closeness_WO",

        "pearson_candidate_gain_vs_endpoint_relative_closeness_z",
        "pearson_candidate_gain_vs_endpoint_relative_closeness_WO",
        "pearson_candidate_gain_vs_cosine_to_akaza_z",
        "pearson_candidate_gain_vs_cosine_to_akaza_WO",
        "pearson_candidate_gain_vs_ray_relative_closeness_z",
        "pearson_candidate_gain_vs_ray_relative_closeness_WO",
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
                            "mean_candidate_direction_norm_z": s[
                                "mean_candidate_direction_norm_z"
                            ],
                            "mean_candidate_direction_norm_WO": s[
                                "mean_candidate_direction_norm_WO"
                            ],

                            "frac_endpoint_closer_than_soft_z": s[
                                "frac_endpoint_closer_than_soft_z"
                            ],
                            "frac_endpoint_closer_than_soft_WO": s[
                                "frac_endpoint_closer_than_soft_WO"
                            ],
                            "mean_endpoint_relative_closeness_z": s[
                                "mean_endpoint_relative_closeness_z"
                            ],
                            "mean_endpoint_relative_closeness_WO": s[
                                "mean_endpoint_relative_closeness_WO"
                            ],

                            "mean_cosine_to_akaza_z": s["mean_cosine_to_akaza_z"],
                            "mean_cosine_to_akaza_WO": s["mean_cosine_to_akaza_WO"],
                            "mean_projection_frac_onto_akaza_z": s[
                                "mean_projection_frac_onto_akaza_z"
                            ],
                            "mean_projection_frac_onto_akaza_WO": s[
                                "mean_projection_frac_onto_akaza_WO"
                            ],

                            "mean_ray_alpha_z": s["mean_ray_alpha_z"],
                            "mean_ray_alpha_WO": s["mean_ray_alpha_WO"],
                            "frac_ray_alpha_positive_z": s["frac_ray_alpha_positive_z"],
                            "frac_ray_alpha_positive_WO": s["frac_ray_alpha_positive_WO"],
                            "frac_ray_closer_than_soft_z": s[
                                "frac_ray_closer_than_soft_z"
                            ],
                            "frac_ray_closer_than_soft_WO": s[
                                "frac_ray_closer_than_soft_WO"
                            ],
                            "frac_positive_ray_closer_than_soft_z": s[
                                "frac_positive_ray_closer_than_soft_z"
                            ],
                            "frac_positive_ray_closer_than_soft_WO": s[
                                "frac_positive_ray_closer_than_soft_WO"
                            ],
                            "mean_ray_relative_closeness_z": s[
                                "mean_ray_relative_closeness_z"
                            ],
                            "mean_ray_relative_closeness_WO": s[
                                "mean_ray_relative_closeness_WO"
                            ],

                            "pearson_candidate_gain_vs_endpoint_relative_closeness_z": s[
                                "pearson_candidate_gain_vs_endpoint_relative_closeness_z"
                            ],
                            "pearson_candidate_gain_vs_endpoint_relative_closeness_WO": s[
                                "pearson_candidate_gain_vs_endpoint_relative_closeness_WO"
                            ],
                            "pearson_candidate_gain_vs_cosine_to_akaza_z": s[
                                "pearson_candidate_gain_vs_cosine_to_akaza_z"
                            ],
                            "pearson_candidate_gain_vs_cosine_to_akaza_WO": s[
                                "pearson_candidate_gain_vs_cosine_to_akaza_WO"
                            ],
                            "pearson_candidate_gain_vs_ray_relative_closeness_z": s[
                                "pearson_candidate_gain_vs_ray_relative_closeness_z"
                            ],
                            "pearson_candidate_gain_vs_ray_relative_closeness_WO": s[
                                "pearson_candidate_gain_vs_ray_relative_closeness_WO"
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

    print("  endpoint closer-than-soft fraction:")
    print(
        f"    all          z={get(all_s, 'frac_endpoint_closer_than_soft_z'):.6f}  "
        f"WO={get(all_s, 'frac_endpoint_closer_than_soft_WO'):.6f}"
    )
    print(
        f"    beneficial  z={get(ben_s, 'frac_endpoint_closer_than_soft_z'):.6f}  "
        f"WO={get(ben_s, 'frac_endpoint_closer_than_soft_WO'):.6f}"
    )
    print(
        f"    nonbenef    z={get(non_s, 'frac_endpoint_closer_than_soft_z'):.6f}  "
        f"WO={get(non_s, 'frac_endpoint_closer_than_soft_WO'):.6f}"
    )

    print("  ray closer-than-soft fraction:")
    print(
        f"    all          z={get(all_s, 'frac_ray_closer_than_soft_z'):.6f}  "
        f"WO={get(all_s, 'frac_ray_closer_than_soft_WO'):.6f}"
    )
    print(
        f"    beneficial  z={get(ben_s, 'frac_ray_closer_than_soft_z'):.6f}  "
        f"WO={get(ben_s, 'frac_ray_closer_than_soft_WO'):.6f}"
    )
    print(
        f"    nonbenef    z={get(non_s, 'frac_ray_closer_than_soft_z'):.6f}  "
        f"WO={get(non_s, 'frac_ray_closer_than_soft_WO'):.6f}"
    )

    print("  positive-ray closer-than-soft fraction:")
    print(
        f"    beneficial  z={get(ben_s, 'frac_positive_ray_closer_than_soft_z'):.6f}  "
        f"WO={get(ben_s, 'frac_positive_ray_closer_than_soft_WO'):.6f}"
    )
    print(
        f"    nonbenef    z={get(non_s, 'frac_positive_ray_closer_than_soft_z'):.6f}  "
        f"WO={get(non_s, 'frac_positive_ray_closer_than_soft_WO'):.6f}"
    )

    print("  mean cosine to AKAZA:")
    print(
        f"    beneficial  z={get(ben_s, 'mean_cosine_to_akaza_z'):.6f}  "
        f"WO={get(ben_s, 'mean_cosine_to_akaza_WO'):.6f}"
    )
    print(
        f"    nonbenef    z={get(non_s, 'mean_cosine_to_akaza_z'):.6f}  "
        f"WO={get(non_s, 'mean_cosine_to_akaza_WO'):.6f}"
    )

    print("  mean ray relative closeness:")
    print(
        f"    beneficial  z={get(ben_s, 'mean_ray_relative_closeness_z'):.6f}  "
        f"WO={get(ben_s, 'mean_ray_relative_closeness_WO'):.6f}"
    )
    print(
        f"    nonbenef    z={get(non_s, 'mean_ray_relative_closeness_z'):.6f}  "
        f"WO={get(non_s, 'mean_ray_relative_closeness_WO'):.6f}"
    )


def main() -> None:
    default_hp = LearnerHyperParams()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether beneficial learner directions align with trained "
            "AKAZA corrections, including scale-free ray metrics."
        )
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
            "endpoint_closer_than_soft_z": "||d_a - delta|| < ||delta||",
            "endpoint_relative_closeness_z": "1 - ||d_a - delta|| / ||delta||",
            "ray_alpha_z": "<d_a, delta> / ||d_a||^2",
            "ray_closer_than_soft_z": "||alpha*d_a - delta|| < ||delta|| using least-squares alpha",
            "positive_ray_closer_than_soft_z": "ray_closer_than_soft_z and alpha > 0",
            "WO_metrics": "same tests after applying frozen attn.c_proj linearly to d_a and delta",
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
