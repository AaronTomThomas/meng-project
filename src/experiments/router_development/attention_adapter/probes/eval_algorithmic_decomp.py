from __future__ import annotations

"""
Canonical learner-dictionary decomposition for trained AKAZA/FreeZ corrections.

This version intentionally imports the original learner registry instead of
reimplementing the learners in this script. The canonical learners live in:

    src/experiments/attention_learners.py

Depending on how the repo is launched, the import path is usually either:

    from experiments.attention_learners import ...        # with PYTHONPATH=src

or:

    from src.experiments.attention_learners import ...    # with PYTHONPATH=.

This script supports both via a small import fallback.

The decomposition question is:

    How much of a trained AKAZA/FreeZ correction delta is explained by the
    span of the original TTR learner-induced directions?

For each edited layer and token, AKAZA produces:

    z_akaza = z_soft + delta

For each canonical learner a, this script computes:

    z_a,   d_a = z_a - z_soft

where z_a is obtained by literally calling the imported learner object on the
causal q/K/V context for each head and token.

Then it fits a local ridge projection:

    alpha* = argmin_alpha ||delta - sum_a alpha_a d_a||_2^2
                         + lambda ||alpha||_2^2

Evaluation modes, without retraining:

  1. none
       z_new = z_soft
       Frozen baseline sanity check.

  2. full
       z_new = z_soft + delta
       Original AKAZA.

  3. projection
       z_new = z_soft + delta_alg
       Keeps only the component explained by the canonical learner dictionary.

  4. residual
       z_new = z_soft + delta_residual
       Keeps only the component not explained by the canonical learner dictionary.

  5. top1_abs_cosine
       z_new = z_soft + best one-dimensional scaled canonical learner direction.

Important:
    This is a fidelity-first diagnostic. It is slower than the old vectorised
    probe script because the imported learner API is single-query/context. That
    is deliberate: this script should answer the thesis question using exactly
    the learners from the original TTR learner dictionary.

Example:

PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.probes.eval_algorithmic_decomp \
  --checkpoint_path outputs/attention_adapter/gpt2_akaza.pt \
  --device cuda \
  --output_path outputs/attention_adapter/gpt2_akaza_algorithmic_decomp.json \
  --candidate_learners sharp,linear_global,window_soft,knn_mean,linear_attention,weighted_linear
"""

import argparse
import json
import math
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Usual repo invocation: PYTHONPATH=src uv run python ...
    from experiments.attention_learners import (  # type: ignore
        LEARNERS,
        LearnerHyperParams,
        BaseAttentionLearner,
        build_learners,
    )
except ModuleNotFoundError:
    # Alternative invocation from repo root with PYTHONPATH=.
    from src.experiments.attention_learners import (  # type: ignore
        LEARNERS,
        LearnerHyperParams,
        BaseAttentionLearner,
        build_learners,
    )

from experiments.router_development.attention_adapter.adapters.akaza_adapters import (
    GPT2AKAZAAdapter,
)
from experiments.router_development.attention_adapter.config import (
    AKAZAFreeZConfig,
    AdapterMethod,
)
from experiments.router_development.attention_adapter.data import load_chunks_for_split
from experiments.router_development.attention_adapter.eval import eval_baseline
from experiments.router_development.attention_adapter.trainer import TrainableParameters
from experiments.router_development.attention_adapter.utils import lm_loss
from experiments.gpt2_probe_utils import extract_head_qkv_and_teacher_outputs_gpt2
from experiments.language_model_probes.probe_utils import merge_heads


BASE_DECOMP_MODES = ["none", "full", "projection", "residual", "top1_abs_cosine"]

# Source of truth: imported registry from experiments.attention_learners.
CANONICAL_LEARNERS = list(LEARNERS)

# soft is the reference z_soft, so including it would create a zero direction.
DEFAULT_CANDIDATE_LEARNERS = [name for name in CANONICAL_LEARNERS if name != "soft"]


def safe_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def parse_csv(s: str | None) -> List[str]:
    if s is None or not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv(s: str) -> List[int]:
    return [int(x) for x in parse_csv(s)]


def relative_ppl_reduction(delta_nll: float) -> float:
    return float(1.0 - math.exp(-delta_nll))


def jsonable_dataclass_dict(obj: Any) -> Dict[str, Any]:
    out = asdict(obj)
    for key, value in list(out.items()):
        if isinstance(value, Enum):
            out[key] = value.value
    return out


@torch.no_grad()
def apply_imported_learner_to_qkv(
    learner: BaseAttentionLearner,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: LearnerHyperParams,
) -> torch.Tensor:
    """Apply one imported canonical learner to every head/token.

    The imported learner interface is single-query/context:

        q:    [B, D]
        Kctx: [B, n, D]
        Vctx: [B, n, Dv]

    GPT-2 q/k/v here are:

        q, k, v: [B, H, T, D]

    For each token t, we flatten B and H into one batch dimension and call the
    imported learner on the causal prefix 0..t. This preserves the exact learner
    semantics from experiments.attention_learners instead of duplicating logic.

    Returns:
        merged z_a: [B, T, C]
    """
    bsz, num_heads, seq_len, dim = q.shape
    dv = v.size(-1)
    outputs: list[torch.Tensor] = []

    for t in range(seq_len):
        q_t = q[:, :, t, :].reshape(bsz * num_heads, dim)
        Kctx = k[:, :, : t + 1, :].reshape(bsz * num_heads, t + 1, dim)
        Vctx = v[:, :, : t + 1, :].reshape(bsz * num_heads, t + 1, dv)

        # Some learners solve linear systems. Running in float32 is more stable
        # and matches the old diagnostic style. Return to original dtype after.
        y = learner(
            q_t.float(),
            Kctx.float(),
            Vctx.float(),
            cfg,
        )
        y = y.to(dtype=v.dtype).view(bsz, num_heads, dv)
        outputs.append(y)

    out_heads = torch.stack(outputs, dim=2)  # [B,H,T,Dv]
    return merge_heads(out_heads)


@torch.no_grad()
def candidate_attention_outputs(
    *,
    model: nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    candidate_names: Sequence[str],
    learner_hparams: LearnerHyperParams,
    learner_instances: Mapping[str, BaseAttentionLearner],
) -> Dict[str, torch.Tensor]:
    """Compute canonical imported-learner z outputs for one block.

    All outputs are merged pre-c_proj attention outputs with shape [B, T, C].
    Supported names are exactly the imported LEARNERS registry.
    """
    _h_ln1, q, k, v, _z_teacher, _zcat_teacher, _block, _attn = extract_head_qkv_and_teacher_outputs_gpt2(
        model,
        hidden_states,
        layer_idx,
    )
    out: Dict[str, torch.Tensor] = {}

    for name in candidate_names:
        if name not in learner_instances:
            raise ValueError(
                f"Unknown canonical learner {name!r}. "
                f"Imported registry contains: {CANONICAL_LEARNERS}."
            )
        out[name] = apply_imported_learner_to_qkv(
            learner_instances[name],
            q,
            k,
            v,
            learner_hparams,
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
    """Best single scaled direction by absolute cosine with delta."""
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


class AlgorithmicDecomposedAKAZAFreeZModel(GPT2AKAZAAdapter):
    """AKAZA wrapper with evaluation-time canonical learner projection modes."""

    def __init__(
        self,
        *,
        model: nn.Module,
        cfg: AKAZAFreeZConfig,
        layer_indices: Sequence[int],
        candidate_names: Sequence[str],
        decomp_mode: str = "full",
        ridge_lambda: float = 1e-4,
        eps: float = 1e-8,
        learner_hparams: LearnerHyperParams | None = None,
    ):
        super().__init__(model=model, cfg=cfg, layer_indices=layer_indices)
        self.candidate_names = list(candidate_names)
        self.decomp_mode = decomp_mode
        self.ridge_lambda = float(ridge_lambda)
        self.eps = float(eps)
        self.learner_hparams = learner_hparams or LearnerHyperParams()

        if self.decomp_mode not in BASE_DECOMP_MODES:
            raise ValueError(f"Unknown decomp_mode={self.decomp_mode!r}; choices={BASE_DECOMP_MODES}")
        if not self.candidate_names:
            raise ValueError("candidate_names must be non-empty")
        unknown = sorted(set(self.candidate_names) - set(CANONICAL_LEARNERS))
        if unknown:
            raise ValueError(
                f"Unknown candidate learner(s): {unknown}. "
                f"Imported canonical registry contains: {CANONICAL_LEARNERS}."
            )
        if "soft" in self.candidate_names:
            raise ValueError("Do not include 'soft' as a candidate direction; it gives a zero direction.")

        self.learner_instances = build_learners(self.candidate_names)
        self._algorithmic_stats_accum: Dict[str, Any] | None = None

    def set_decomp_mode(self, mode: str) -> None:
        if mode not in BASE_DECOMP_MODES:
            raise ValueError(f"Unknown decomp_mode={mode!r}; choices={BASE_DECOMP_MODES}")
        self.decomp_mode = mode

    def candidate_directions(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        z_soft: torch.Tensor,
    ) -> torch.Tensor:
        cand = candidate_attention_outputs(
            model=self.model,
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            candidate_names=self.candidate_names,
            learner_hparams=self.learner_hparams,
            learner_instances=self.learner_instances,
        )
        dirs = [cand[name].to(z_soft.dtype) - z_soft for name in self.candidate_names]
        return torch.stack(dirs, dim=2)  # [B, T, A, C]

    def select_delta(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        z_soft: torch.Tensor,
        delta_full: torch.Tensor,
    ) -> torch.Tensor:
        if self.decomp_mode == "none":
            return torch.zeros_like(delta_full)
        if self.decomp_mode == "full":
            return delta_full

        directions = self.candidate_directions(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
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

    def _make_c_proj_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if layer_idx not in self._adapter_inputs:
                raise RuntimeError(
                    f"Missing cached GPT-2 block input for layer {layer_idx}. "
                    "The block pre-hook did not fire before attn.c_proj."
                )
            z_soft = inputs[0]
            hidden_states = self._adapter_inputs[layer_idx]
            block = self.model.transformer.h[layer_idx]
            delta_full = self.compute_delta(
                layer_idx=layer_idx,
                block=block,
                hidden_states=hidden_states,
            ).to(dtype=z_soft.dtype, device=z_soft.device)

            if self._algorithmic_stats_accum is not None:
                self._accumulate_algorithmic_stats(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    z_soft=z_soft,
                    delta=delta_full,
                )

            delta_used = self.select_delta(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                z_soft=z_soft,
                delta_full=delta_full,
            ).to(dtype=z_soft.dtype, device=z_soft.device)
            self._latest_deltas[layer_idx] = delta_used.detach()
            return (z_soft + delta_used,) + inputs[1:]

        return hook

    def _accumulate_algorithmic_stats(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        z_soft: torch.Tensor,
        delta: torch.Tensor,
    ) -> None:
        accum = self._algorithmic_stats_accum
        if accum is None:
            return
        block = self.model.transformer.h[layer_idx]
        directions = self.candidate_directions(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
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

        accum["total_delta_energy"] = accum["total_delta_energy"] + d.pow(2).sum()
        accum["total_projection_energy"] = accum["total_projection_energy"] + p.pow(2).sum()
        accum["total_residual_energy"] = accum["total_residual_energy"] + r.pow(2).sum()

        dp_dot = (d * p).sum(dim=-1)
        dp_den = d.pow(2).sum(dim=-1).sqrt().clamp_min(self.eps) * p.pow(2).sum(dim=-1).sqrt().clamp_min(self.eps)
        accum["delta_projection_cos_sum"] = accum["delta_projection_cos_sum"] + (dp_dot / dp_den).sum()
        accum["n_projection_cos"] += int(dp_dot.numel())

        dot = torch.einsum("btac,btc->bta", D, d)
        D_norm = D.pow(2).sum(dim=-1).sqrt().clamp_min(self.eps)
        d_norm = d.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        cos = dot / (D_norm * d_norm)

        accum["alpha_sum"] = accum["alpha_sum"] + alpha.sum(dim=(0, 1))
        accum["alpha_abs_sum"] = accum["alpha_abs_sum"] + alpha.abs().sum(dim=(0, 1))
        accum["cosine_sum"] = accum["cosine_sum"] + cos.sum(dim=(0, 1))
        accum["abs_cosine_sum"] = accum["abs_cosine_sum"] + cos.abs().sum(dim=(0, 1))

        top_abs_cos = cos.abs().argmax(dim=-1)
        top_abs_alpha = alpha.abs().argmax(dim=-1)
        ones = torch.ones_like(top_abs_cos.flatten(), dtype=torch.float32)
        accum["top_abs_cos_counts"].scatter_add_(0, top_abs_cos.flatten(), ones)
        accum["top_abs_alpha_counts"].scatter_add_(0, top_abs_alpha.flatten(), ones)
        accum["n_token_layer"] += int(top_abs_cos.numel())

    @torch.no_grad()
    def algorithmic_stats_for_batch(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """Collect geometric attribution stats along the inherited AKAZA hook path."""
        self.set_peft_eval_mode()
        input_ids = input_ids.to(self.device)

        old_mode = self.decomp_mode
        self._algorithmic_stats_accum = {
            "total_delta_energy": torch.tensor(0.0, device=input_ids.device),
            "total_projection_energy": torch.tensor(0.0, device=input_ids.device),
            "total_residual_energy": torch.tensor(0.0, device=input_ids.device),
            "delta_projection_cos_sum": torch.tensor(0.0, device=input_ids.device),
            "n_projection_cos": 0,
            "alpha_sum": torch.zeros(len(self.candidate_names), device=input_ids.device),
            "alpha_abs_sum": torch.zeros(len(self.candidate_names), device=input_ids.device),
            "cosine_sum": torch.zeros(len(self.candidate_names), device=input_ids.device),
            "abs_cosine_sum": torch.zeros(len(self.candidate_names), device=input_ids.device),
            "top_abs_cos_counts": torch.zeros(len(self.candidate_names), device=input_ids.device),
            "top_abs_alpha_counts": torch.zeros(len(self.candidate_names), device=input_ids.device),
            "n_token_layer": 0,
        }

        try:
            self.decomp_mode = "full"
            _ = self(input_ids)
            accum = self._algorithmic_stats_accum
        finally:
            self.decomp_mode = old_mode
            self._algorithmic_stats_accum = None

        if accum is None:
            return {}

        out: Dict[str, float] = {}
        n_token_layer = int(accum["n_token_layer"])
        if n_token_layer == 0:
            return out

        total_delta_energy = accum["total_delta_energy"].clamp_min(self.eps)
        out["alg_projection_energy_over_delta_energy"] = safe_float(
            accum["total_projection_energy"] / total_delta_energy
        )
        out["alg_residual_energy_over_delta_energy"] = safe_float(
            accum["total_residual_energy"] / total_delta_energy
        )
        out["delta_projection_cosine_mean"] = safe_float(
            accum["delta_projection_cos_sum"] / max(1, int(accum["n_projection_cos"]))
        )

        denom = float(max(1, n_token_layer))
        for j, name in enumerate(self.candidate_names):
            out[f"candidate/{name}/alpha_mean"] = safe_float(accum["alpha_sum"][j] / denom)
            out[f"candidate/{name}/alpha_abs_mean"] = safe_float(accum["alpha_abs_sum"][j] / denom)
            out[f"candidate/{name}/cosine_mean"] = safe_float(accum["cosine_sum"][j] / denom)
            out[f"candidate/{name}/abs_cosine_mean"] = safe_float(accum["abs_cosine_sum"][j] / denom)
            out[f"candidate/{name}/top_abs_cosine_frac"] = safe_float(accum["top_abs_cos_counts"][j] / denom)
            out[f"candidate/{name}/top_abs_alpha_frac"] = safe_float(accum["top_abs_alpha_counts"][j] / denom)

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


def cfg_from_summary(summary: Dict[str, Any]) -> AKAZAFreeZConfig:
    cfg_dict = dict(summary["config"])
    cfg_dict["method"] = AdapterMethod.AKAZA_FREEZ
    valid_fields = {f.name for f in fields(AKAZAFreeZConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return AKAZAFreeZConfig(**filtered)


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
    default_hp = LearnerHyperParams()

    parser = argparse.ArgumentParser(
        description="Evaluate AKAZA decomposition using the imported canonical attention learner registry."
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

    # Projection ridge: stabilises the local least-squares projection of AKAZA
    # delta onto the learner-direction dictionary.
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)

    # These are passed directly into the imported LearnerHyperParams dataclass.
    parser.add_argument(
        "--local_kernel_beta",
        "--beta_soft",
        dest="local_kernel_beta",
        type=float,
        default=default_hp.local_kernel_beta,
        help="Local kernel beta for weighted local-linear learners. --beta_soft is kept as a deprecated alias.",
    )
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
            "Algorithmic decomposition is currently integrated for GPT-2 AKAZA checkpoints only; "
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

    device = torch.device(cfg.device)
    layer_indices = summary.get("layer_indices") or parse_int_csv(cfg.layer_indices)
    layer_indices = sorted(int(x) for x in layer_indices)

    candidate_names = parse_csv(args.candidate_learners)
    unknown = sorted(set(candidate_names) - set(CANONICAL_LEARNERS))
    if unknown:
        raise ValueError(
            f"Unknown candidate learner(s): {unknown}. "
            f"Imported canonical registry contains: {CANONICAL_LEARNERS}."
        )
    if "soft" in candidate_names:
        raise ValueError("Do not include 'soft' as a candidate direction; it gives a zero direction.")

    learner_hparams = LearnerHyperParams(
        local_kernel_beta=args.local_kernel_beta,
        window_size=args.window_size,
        k_knn_mean=args.k_knn_mean,
        ridge_lambda=args.learner_ridge_lambda,
        k_linear_local=args.k_linear_local,
        k_sharp=args.k_sharp,
    )

    print("[loaded]")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  method: {summary.get('method')}")
    print(f"  best_epoch: {summary.get('best_epoch')}")
    print(f"  layer_indices: {layer_indices}")
    print(f"  bottleneck_dim: {cfg.bottleneck_dim}")
    print(f"  output_scale: {cfg.output_scale}")
    print(f"  imported_learners: {CANONICAL_LEARNERS}")
    print(f"  candidate_learners: {candidate_names}")
    print(f"  projection_ridge_lambda: {args.ridge_lambda}")
    print(f"  learner_hparams: {learner_hparams}")

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
        learner_hparams=learner_hparams,
    ).to(device)
    TrainableParameters(params=[], frozen_before_training={}, check_frozen=False).load_trainable_state_dict(
        wrapped,
        payload["trainable_state_dict"],
    )

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
            max_chunks=cfg.max_train_chunks,
        )
    if need_val:
        chunks_by_split["val"] = load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.val_split,
            max_chunks=cfg.max_val_chunks,
        )
    if need_test:
        chunks_by_split["test"] = load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.test_split,
            max_chunks=cfg.max_test_chunks,
        )

    for split_name, chunks in chunks_by_split.items():
        print(f"  {split_name}: chunks={chunks.shape[0]} block_size={chunks.shape[1]}")

    results: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source_summary_best_epoch": summary.get("best_epoch"),
        "source_summary_best_val_loss": summary.get("best_val_loss"),
        "source_summary_best_test_loss": summary.get("best_test_loss"),
        "config": jsonable_dataclass_dict(cfg),
        "layer_indices": layer_indices,
        "imported_canonical_learners": CANONICAL_LEARNERS,
        "candidate_learners": candidate_names,
        "decomposition_config": {
            "projection_ridge_lambda": args.ridge_lambda,
            "eps": args.eps,
            "canonical_learner_hparams": asdict(learner_hparams),
            "implementation_note": "Candidate z_a outputs are computed by calling imported experiments.attention_learners learner instances over causal q/K/V prefixes.",
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
