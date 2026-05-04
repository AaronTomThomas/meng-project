"""Utilities for first-order distillation routing."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


class GainVectorMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            layers.append(torch.nn.GELU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            current_dim = hidden_dim
        self.backbone = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_gain_vector_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> GainVectorMLP:
    return GainVectorMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_float_csv(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def compute_feature_normalizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def normalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std


def batch_slices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield slice(start, min(start + batch_size, total))


def best_fixed_action(costs: torch.Tensor) -> int:
    """Return the action with the lowest mean cost on the supplied rows."""

    mean_costs = costs.float().mean(dim=0)
    return int(mean_costs.argmin().item())


def evaluate_actions(
    costs: torch.Tensor,
    actions: torch.Tensor,
    best_action: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize exact-cost performance for a row-aligned action policy."""

    costs = costs.float().cpu()
    actions = actions.long().cpu()
    row_ids = torch.arange(costs.shape[0])
    chosen_cost = costs[row_ids, actions]
    oracle_cost, oracle_action = costs.min(dim=1)
    if best_action is None:
        best_action = oracle_action
    else:
        best_action = best_action.long().cpu()

    return {
        "mean_cost": float(chosen_cost.mean().item()),
        "mean_regret": float((chosen_cost - oracle_cost).mean().item()),
        "oracle_cost": float(oracle_cost.mean().item()),
        "action_accuracy": float((actions == best_action).float().mean().item()),
    }


def gap_closure(router_loss: float, best_fixed_loss: float, oracle_loss: float) -> float:
    """Fraction of the fixed-policy oracle gap closed by the router."""

    gap = float(best_fixed_loss) - float(oracle_loss)
    if abs(gap) < 1e-12:
        return 0.0
    return (float(best_fixed_loss) - float(router_loss)) / gap


def selected_group_ids(dataset: dict, requested: str) -> list[int]:
    all_groups = sorted(int(x) for x in dataset["group_ids"].unique().tolist())
    if not requested.strip():
        return all_groups
    wanted = [int(x.strip()) for x in requested.split(",") if x.strip()]
    return [group_id for group_id in wanted if group_id in all_groups]


def get_split_indices(dataset: dict, split: str) -> torch.Tensor:
    split_indices = dataset["split_indices"]
    if split not in split_indices:
        raise KeyError(f"Unknown split '{split}'. Available: {sorted(split_indices)}")
    return split_indices[split].long()


def get_ordered_split_indices(dataset: dict, split: str) -> torch.Tensor:
    indices = get_split_indices(dataset, split)
    if indices.numel() == 0:
        return indices
    chunk_ids = dataset["chunk_ids"][indices]
    group_ids = dataset["group_ids"][indices]
    positions = dataset["positions"][indices]
    order = sorted(
        range(indices.numel()),
        key=lambda idx: (
            int(chunk_ids[idx].item()),
            int(group_ids[idx].item()),
            int(positions[idx].item()),
        ),
    )
    return indices[torch.tensor(order, dtype=torch.long)]


def best_fixed_soft(dataset: dict, split_idx: torch.Tensor) -> int:
    action_names = list(dataset["action_names"])
    if "soft" in action_names:
        return action_names.index("soft")
    return best_fixed_action(dataset["costs"][split_idx].float())


def candidate_action_indices(action_names: Sequence[str], candidate_names: Sequence[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(action_names)}
    missing = [name for name in candidate_names if name not in name_to_idx]
    if missing:
        raise KeyError(f"Candidate actions missing from dataset: {missing}")
    return [name_to_idx[name] for name in candidate_names]


def build_exact_pairwise_targets(
    costs: torch.Tensor,
    base_action_idx: int,
    candidate_indices: Sequence[int],
) -> torch.Tensor:
    base_cost = costs[:, base_action_idx].unsqueeze(-1)
    candidate_costs = costs[:, list(candidate_indices)]
    return (base_cost - candidate_costs).float()


def distillation_loss(
    pred_gains: torch.Tensor,
    first_order_targets: torch.Tensor,
    exact_targets: torch.Tensor | None,
    exact_loss_weight: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred_gains, first_order_targets)
    if exact_targets is not None and exact_loss_weight > 0:
        loss = loss + float(exact_loss_weight) * F.smooth_l1_loss(pred_gains, exact_targets)
    return loss


def choose_budget_actions(
    predicted_gains: torch.Tensor,
    base_action_idx: int,
    candidate_indices: Sequence[int],
    threshold: float,
) -> torch.Tensor:
    best_gain, best_idx = predicted_gains.max(dim=-1)
    chosen = torch.tensor(list(candidate_indices), dtype=torch.long, device=predicted_gains.device)[best_idx]
    actions = torch.full((predicted_gains.shape[0],), int(base_action_idx), dtype=torch.long, device=predicted_gains.device)
    mask = best_gain >= float(threshold)
    actions[mask] = chosen[mask]
    return actions


def threshold_for_budget(best_gain: torch.Tensor, budget_fraction: float) -> float:
    if budget_fraction <= 0:
        return float("inf")
    if budget_fraction >= 1:
        return float("-inf")
    num_examples = best_gain.numel()
    keep = max(1, int(round(budget_fraction * num_examples)))
    top_vals = torch.topk(best_gain, k=keep, largest=True).values
    return float(top_vals[-1].item())


def tune_budget_policy(
    predicted_gains: torch.Tensor,
    costs: torch.Tensor,
    best_action: torch.Tensor,
    base_action_idx: int,
    candidate_indices: Sequence[int],
    budget_grid: Sequence[float],
) -> tuple[float, float, dict[str, float], torch.Tensor]:
    best_budget = 0.0
    best_threshold = float("inf")
    best_metrics: dict[str, float] | None = None
    best_actions: torch.Tensor | None = None
    best_gain = predicted_gains.max(dim=-1).values
    for budget in budget_grid:
        threshold = threshold_for_budget(best_gain, float(budget))
        actions = choose_budget_actions(
            predicted_gains=predicted_gains,
            base_action_idx=base_action_idx,
            candidate_indices=candidate_indices,
            threshold=threshold,
        ).cpu()
        metrics = evaluate_actions(costs, actions, best_action=best_action)
        metrics["switch_rate"] = float((actions != base_action_idx).float().mean().item())
        if best_metrics is None or metrics["mean_cost"] < best_metrics["mean_cost"]:
            best_budget = float(budget)
            best_threshold = float(threshold)
            best_metrics = metrics
            best_actions = actions
    assert best_metrics is not None and best_actions is not None
    return best_budget, best_threshold, best_metrics, best_actions


def summarize_eval(
    costs: torch.Tensor,
    actions: torch.Tensor,
    best_action: torch.Tensor,
    base_action_idx: int,
    action_names: Sequence[str],
) -> dict[str, object]:
    router_metrics = evaluate_actions(costs, actions, best_action=best_action)
    base_actions = torch.full((costs.shape[0],), int(base_action_idx), dtype=torch.long)
    base_metrics = evaluate_actions(costs, base_actions, best_action=best_action)
    best_fixed_idx = best_fixed_action(costs)
    best_fixed_actions = torch.full((costs.shape[0],), int(best_fixed_idx), dtype=torch.long)
    best_fixed_metrics = evaluate_actions(costs, best_fixed_actions, best_action=best_action)
    router_metrics["switch_rate"] = float((actions != base_action_idx).float().mean().item())
    router_metrics["improvement_vs_base_policy"] = float(base_metrics["mean_cost"] - router_metrics["mean_cost"])
    router_metrics["improvement_vs_best_fixed"] = float(best_fixed_metrics["mean_cost"] - router_metrics["mean_cost"])
    router_metrics["gap_closure_vs_best_fixed"] = gap_closure(
        router_loss=router_metrics["mean_cost"],
        best_fixed_loss=best_fixed_metrics["mean_cost"],
        oracle_loss=router_metrics["oracle_cost"],
    )
    return {
        "router": router_metrics,
        "base_policy": {
            **base_metrics,
            "action_idx": int(base_action_idx),
            "action_name": action_names[base_action_idx],
        },
        "best_fixed": {
            **best_fixed_metrics,
            "action_idx": int(best_fixed_idx),
            "action_name": action_names[best_fixed_idx],
        },
    }
