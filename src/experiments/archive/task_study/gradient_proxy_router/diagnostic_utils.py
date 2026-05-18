from __future__ import annotations

import math
from typing import Any, List, Sequence

import torch


def as_tensor(x: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    out = x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x)
    return out.to(dtype=dtype) if dtype is not None else out


def safe_mean(x: torch.Tensor) -> float:
    x = x.detach().float().cpu().reshape(-1)
    return 0.0 if x.numel() == 0 else float(x.mean().item())


def safe_frac(mask: torch.Tensor) -> float:
    mask = mask.detach().bool().cpu().reshape(-1)
    return 0.0 if mask.numel() == 0 else float(mask.float().mean().item())


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.strip().split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x) for x in parse_csv(value)]


def find_action_index(action: str, action_names: Sequence[str]) -> int:
    action = action.strip()

    if action.isdigit():
        idx = int(action)
        if 0 <= idx < len(action_names):
            return idx
        raise ValueError(f"Action index {idx} out of range for {len(action_names)} actions.")

    if action in action_names:
        return action_names.index(action)

    lowered = [name.lower() for name in action_names]
    if action.lower() in lowered:
        return lowered.index(action.lower())

    raise ValueError(f"Could not find action {action!r}. Available: {list(action_names)}")


def resolve_candidate_indices(
    candidate_actions: str,
    action_names: Sequence[str],
    default_idx: int,
) -> List[int]:
    if candidate_actions.strip() in {"", "all", "nondefault"}:
        return [i for i in range(len(action_names)) if i != default_idx]

    out: List[int] = []
    for name in parse_csv(candidate_actions):
        idx = find_action_index(name, action_names)
        if idx != default_idx and idx not in out:
            out.append(idx)

    if not out:
        raise ValueError("No non-default candidate actions selected.")
    return out


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().cpu().reshape(-1)
    y = y.detach().float().cpu().reshape(-1)
    keep = torch.isfinite(x) & torch.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.numel() < 2:
        return 0.0

    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) < 1e-12:
        return 0.0
    return float(((x * y).sum() / denom).item())


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().float().cpu().reshape(-1)
    target = target.detach().float().cpu().reshape(-1)
    keep = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[keep]
    target = target[keep]
    if target.numel() < 2:
        return 0.0

    sse = ((pred - target) ** 2).sum()
    sst = ((target - target.mean()) ** 2).sum().clamp_min(1e-12)
    return float((1.0 - sse / sst).item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    keep = torch.isfinite(pred) & torch.isfinite(target)
    return 0.0 if keep.sum().item() == 0 else float(torch.mean((pred[keep] - target[keep]) ** 2).item())


def pairwise_ranking_accuracy(scores: torch.Tensor, target: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    target = target.detach().float().cpu()
    if scores.ndim != 2 or target.ndim != 2 or target.shape[1] < 2:
        return 0.0

    correct = 0.0
    total = 0.0
    for i in range(target.shape[1]):
        for j in range(i + 1, target.shape[1]):
            true_order = torch.sign(target[:, i] - target[:, j])
            pred_order = torch.sign(scores[:, i] - scores[:, j])
            keep = true_order != 0
            if keep.any():
                correct += (pred_order[keep] == true_order[keep]).float().sum().item()
                total += keep.float().sum().item()
    return float(correct / total) if total > 0 else 0.0


def actionwise_cosine(scores: torch.Tensor, target: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    target = target.detach().float().cpu()
    if scores.ndim != 2 or target.ndim != 2 or scores.shape[1] < 2:
        return 0.0

    scores_c = scores - scores.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    denom = scores_c.norm(dim=1).clamp_min(1e-8) * target_c.norm(dim=1).clamp_min(1e-8)
    return float(((scores_c * target_c).sum(dim=1) / denom).mean().item())


def threshold_for_budget(scores: torch.Tensor, budget: float) -> float:
    scores = scores.detach().float().reshape(-1)
    if scores.numel() == 0 or budget <= 0:
        return float("inf")

    k = min(scores.numel(), max(1, int(math.ceil(budget * scores.numel()))))
    return float(torch.topk(scores, k=k).values.min().item())


def standardize_with_stats(
    x: torch.Tensor,
    train_idx: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean = x[train_idx].mean(dim=0, keepdim=True)
    std = x[train_idx].std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std, {"mean": mean, "std": std}


def ridge_fit_augmented(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    x_train = x_train.double()
    y_train = y_train.double()

    ones = torch.ones(x_train.shape[0], 1, dtype=x_train.dtype)
    xb = torch.cat([x_train, ones], dim=1)
    eye = torch.eye(xb.shape[1], dtype=x_train.dtype)
    eye[-1, -1] = 0.0

    return torch.linalg.solve(xb.T @ xb + ridge_lambda * eye, xb.T @ y_train).float()


def ridge_predict(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
    return torch.cat([x, ones], dim=1) @ w
