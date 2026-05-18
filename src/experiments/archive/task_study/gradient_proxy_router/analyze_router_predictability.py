from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.archive.task_study.gradient_proxy_router.diagnostic_utils import (
    actionwise_cosine,
    as_tensor as _as_tensor,
    find_action_index,
    mse,
    pairwise_ranking_accuracy,
    parse_csv,
    parse_int_csv,
    pearson_corr,
    r2_score,
    resolve_candidate_indices,
    ridge_fit_augmented,
    ridge_predict,
    safe_frac as _safe_frac,
    safe_mean as _safe_mean,
    standardize_with_stats,
    threshold_for_budget,
)


ROUTER_FEATURE_KEY = "sequence_only_scalar_features_gradient_state"


BASE_FEATURE_KEY = "features"
SUPPORTED_FEATURE_KEYS = (BASE_FEATURE_KEY, ROUTER_FEATURE_KEY)


def normalize_action_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return str(x[0])
        return "+".join(str(v) for v in x)
    return str(x)


def infer_action_names(data: Dict[str, Any], num_actions: int) -> List[str]:
    for key in [
        "action_names",
        "assignment_names",
        "candidate_action_names",
        "actions",
        "assignments",
    ]:
        if key in data:
            names = [normalize_action_name(x) for x in data[key]]
            if len(names) == num_actions:
                return names

    return [f"action_{i}" for i in range(num_actions)]


def get_primary_tensor(data: Dict[str, Any], keys: Sequence[str]) -> torch.Tensor:
    for key in keys:
        if key in data:
            return _as_tensor(data[key], dtype=torch.float32)
    raise KeyError(f"Could not find any of these keys in dataset: {keys}")


def get_costs(data: Dict[str, Any]) -> torch.Tensor:
    return get_primary_tensor(
        data,
        keys=[
            "costs",
            "exact_costs",
            "exact_losses",
            "losses",
        ],
    )


def get_first_order_gains(data: Dict[str, Any]) -> torch.Tensor:
    return get_primary_tensor(
        data,
        keys=[
            "first_order_gains",
            "linearized_gains",
            "gradient_proxy_gains",
        ],
    )


def compute_exact_gains(costs: torch.Tensor, default_idx: int) -> torch.Tensor:
    """
    Gain convention: higher is better, and default action has gain exactly zero.
    """
    costs = costs.float()
    return costs[:, default_idx : default_idx + 1] - costs


def get_features(data: Dict[str, Any], feature_key: str) -> torch.Tensor:
    if feature_key not in SUPPORTED_FEATURE_KEYS:
        raise ValueError(
            f"Unsupported feature key {feature_key!r}. "
            f"Supported deployable feature keys: {list(SUPPORTED_FEATURE_KEYS)}"
        )
    if feature_key not in data:
        available = [key for key in SUPPORTED_FEATURE_KEYS if key in data]
        raise KeyError(
            f"Feature key {feature_key!r} not found. "
            f"Supported feature keys present in this dataset: {available}"
        )
    return _as_tensor(data[feature_key], dtype=torch.float32)


def build_selection_mask(
    data: Dict[str, Any],
    n: int,
    group_ids: str,
    layer_ids: str,
) -> torch.Tensor:
    mask = torch.ones(n, dtype=torch.bool)

    group_filter = parse_int_csv(group_ids)
    if group_filter:
        if "group_ids" not in data:
            raise KeyError("--group_ids was passed, but dataset has no 'group_ids' key.")
        gids = _as_tensor(data["group_ids"], dtype=torch.long).reshape(-1)
        allowed = torch.tensor(group_filter, dtype=torch.long)
        mask &= torch.isin(gids, allowed)

    layer_filter = parse_int_csv(layer_ids)
    if layer_filter:
        layer_key = None
        for key in ["layer_ids", "layers", "layer_idx"]:
            if key in data:
                layer_key = key
                break
        if layer_key is None:
            raise KeyError("--layer_ids was passed, but dataset has no layer id key.")
        lids = _as_tensor(data[layer_key], dtype=torch.long).reshape(-1)
        allowed = torch.tensor(layer_filter, dtype=torch.long)
        mask &= torch.isin(lids, allowed)

    if int(mask.sum().item()) == 0:
        raise ValueError("Selection mask is empty. Check --group_ids / --layer_ids.")

    return mask


def deterministic_split_from_chunks(
    data: Dict[str, Any],
    n: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    if "chunk_ids" in data:
        chunk_ids = _as_tensor(data["chunk_ids"], dtype=torch.long).reshape(-1)
        unique_chunks = torch.unique(chunk_ids).tolist()
        rng = random.Random(seed)
        rng.shuffle(unique_chunks)

        n_chunks = len(unique_chunks)
        n_train = int(0.7 * n_chunks)
        n_val = int(0.15 * n_chunks)

        train_chunks = set(unique_chunks[:n_train])
        val_chunks = set(unique_chunks[n_train : n_train + n_val])
        test_chunks = set(unique_chunks[n_train + n_val :])

        train = torch.tensor([int(c.item()) in train_chunks for c in chunk_ids], dtype=torch.bool)
        val = torch.tensor([int(c.item()) in val_chunks for c in chunk_ids], dtype=torch.bool)
        test = torch.tensor([int(c.item()) in test_chunks for c in chunk_ids], dtype=torch.bool)

        return {
            "train": train.nonzero(as_tuple=False).flatten(),
            "val": val.nonzero(as_tuple=False).flatten(),
            "test": test.nonzero(as_tuple=False).flatten(),
        }

    idx = torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(n, generator=g)]

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def normalize_split_key(key: str) -> str:
    key = key.lower()
    if key in {"train", "training"}:
        return "train"
    if key in {"val", "valid", "validation"}:
        return "val"
    if key in {"test", "testing"}:
        return "test"
    return key


def get_split_indices(
    data: Dict[str, Any],
    n: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    if "split_indices" not in data:
        return deterministic_split_from_chunks(data, n=n, seed=seed)

    raw = data["split_indices"]

    if not isinstance(raw, dict):
        raise ValueError("'split_indices' exists but is not a dict.")

    out: Dict[str, torch.Tensor] = {}

    for key, value in raw.items():
        norm_key = normalize_split_key(str(key))
        if norm_key not in {"train", "val", "test"}:
            continue
        out[norm_key] = _as_tensor(value, dtype=torch.long).reshape(-1)

    missing = {"train", "val", "test"} - set(out)
    if missing:
        raise ValueError(f"split_indices missing required splits: {sorted(missing)}")

    return out


def restrict_split_indices(
    full_split_indices: Dict[str, torch.Tensor],
    selection_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Converts original dataset indices into local indices after filtering by selection_mask.
    """
    n = selection_mask.numel()
    selected_orig = selection_mask.nonzero(as_tuple=False).flatten()

    out = {}
    for split_name, original_indices in full_split_indices.items():
        split_flag = torch.zeros(n, dtype=torch.bool)
        split_flag[original_indices.long()] = True

        local_mask = split_flag[selected_orig]
        out[split_name] = local_mask.nonzero(as_tuple=False).flatten()

    return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()

        layers: List[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden_dims(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(x) for x in parse_csv(value)]


def train_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    hidden_dims: Sequence[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> MLP:
    torch.manual_seed(seed)

    model = MLP(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    x_dev = x.to(device)
    y_dev = y.to(device)
    train_idx_dev = train_idx.to(device)
    val_idx_dev = val_idx.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")

    g = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(epochs):
        model.train()

        perm_cpu = train_idx[torch.randperm(train_idx.numel(), generator=g)]
        for start in range(0, perm_cpu.numel(), batch_size):
            batch_idx = perm_cpu[start : start + batch_size].to(device)

            pred = model(x_dev[batch_idx])
            loss = F.mse_loss(pred, y_dev[batch_idx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_dev[val_idx_dev])
            val_loss = F.mse_loss(val_pred, y_dev[val_idx_dev]).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()


@torch.no_grad()
def model_predict(model: MLP, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    preds = []
    for start in range(0, x.shape[0], batch_size):
        preds.append(model(x[start : start + batch_size]).detach().cpu())
    return torch.cat(preds, dim=0)


def evaluate_oracle_gain(
    costs: torch.Tensor,
    default_idx: int,
    candidate_indices: Sequence[int],
    action_names: Sequence[str],
    eps: float,
) -> Dict[str, Any]:
    allowed_indices = [default_idx] + list(candidate_indices)
    allowed_names = [action_names[i] for i in allowed_indices]

    c = costs[:, allowed_indices].float()
    base = c[:, 0]

    exact_gains = base.unsqueeze(1) - c

    oracle_cost, oracle_local_idx = c.min(dim=1)
    oracle_gain = base - oracle_cost

    best_fixed_costs = c.mean(dim=0)
    best_fixed_cost, best_fixed_local_idx = best_fixed_costs.min(dim=0)
    best_fixed_gain = base.mean() - best_fixed_cost

    nondefault_oracle = (oracle_local_idx != 0) & (oracle_gain > eps)

    return {
        "num_examples": int(costs.shape[0]),
        "allowed_actions": allowed_names,
        "base_action": action_names[default_idx],
        "base_mean_cost": float(base.mean().item()),
        "oracle_mean_cost": float(oracle_cost.mean().item()),
        "oracle_improvement_vs_base": float((base.mean() - oracle_cost.mean()).item()),
        "oracle_best_nondefault_rate": _safe_frac(nondefault_oracle),
        "oracle_mean_gain": float(oracle_gain.mean().item()),
        "oracle_mean_positive_gain": _safe_mean(oracle_gain[oracle_gain > eps]),
        "oracle_positive_gain_rate": _safe_frac(oracle_gain > eps),
        "best_fixed_action": allowed_names[int(best_fixed_local_idx.item())],
        "best_fixed_mean_cost": float(best_fixed_cost.item()),
        "best_fixed_improvement_vs_base": float(best_fixed_gain.item()),
        "oracle_gap_vs_best_fixed": float((best_fixed_cost - oracle_cost.mean()).item()),
        "exact_gain_by_action_mean": {
            allowed_names[i]: float(exact_gains[:, i].mean().item())
            for i in range(len(allowed_names))
        },
        "exact_gain_by_action_positive_rate": {
            allowed_names[i]: _safe_frac(exact_gains[:, i] > eps)
            for i in range(len(allowed_names))
        },
    }


def evaluate_first_order_alignment(
    costs: torch.Tensor,
    first_order_gains: torch.Tensor,
    default_idx: int,
    candidate_indices: Sequence[int],
    action_names: Sequence[str],
    eps: float,
) -> Dict[str, Any]:
    allowed_indices = [default_idx] + list(candidate_indices)
    allowed_names = [action_names[i] for i in allowed_indices]

    c = costs[:, allowed_indices].float()
    base = c[:, 0]
    exact_gains = base.unsqueeze(1) - c

    fo = first_order_gains[:, allowed_indices].float().clone()
    fo[:, 0] = 0.0

    nd_exact = exact_gains[:, 1:]
    nd_fo = fo[:, 1:]

    exact_best = exact_gains.argmax(dim=1)
    fo_best = fo.argmax(dim=1)

    exact_best_nondefault = nd_exact.max(dim=1).values > eps
    fo_best_nondefault = nd_fo.max(dim=1).values > eps

    out = {
        "allowed_actions": allowed_names,
        "flat_pearson_all_allowed": pearson_corr(fo, exact_gains),
        "flat_pearson_nondefault": pearson_corr(nd_fo, nd_exact),
        "r2_first_order_vs_exact_nondefault": r2_score(nd_fo, nd_exact),
        "mse_first_order_vs_exact_nondefault": mse(nd_fo, nd_exact),
        "pairwise_rank_acc_all_allowed": pairwise_ranking_accuracy(fo, exact_gains),
        "pairwise_rank_acc_nondefault": pairwise_ranking_accuracy(nd_fo, nd_exact),
        "actionwise_cosine_all_allowed": actionwise_cosine(fo, exact_gains),
        "actionwise_cosine_nondefault": actionwise_cosine(nd_fo, nd_exact),
        "top1_action_match_all_allowed": _safe_frac(fo_best == exact_best),
        "switch_sign_match": _safe_frac(fo_best_nondefault == exact_best_nondefault),
        "first_order_switch_rate": _safe_frac(fo_best_nondefault),
        "exact_switchable_rate": _safe_frac(exact_best_nondefault),
    }

    for local_i, action_name in enumerate(allowed_names):
        out[f"pearson_{action_name}"] = pearson_corr(fo[:, local_i], exact_gains[:, local_i])
        out[f"sign_match_{action_name}"] = _safe_frac(
            torch.sign(fo[:, local_i]) == torch.sign(exact_gains[:, local_i])
        )

    return out


def evaluate_prediction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    split_idx: torch.Tensor,
    action_names: Sequence[str],
) -> Dict[str, Any]:
    p = pred[split_idx].float()
    y = target[split_idx].float()

    out: Dict[str, Any] = {
        "mse": mse(p, y),
        "r2": r2_score(p, y),
        "flat_pearson": pearson_corr(p, y),
        "pairwise_rank_acc": pairwise_ranking_accuracy(p, y),
        "actionwise_cosine": actionwise_cosine(p, y),
    }

    if y.shape[1] > 1:
        out["top1_action_match"] = _safe_frac(p.argmax(dim=1) == y.argmax(dim=1))
    else:
        out["top1_action_match"] = 1.0

    for i, name in enumerate(action_names):
        out[f"mse_{name}"] = mse(p[:, i], y[:, i])
        out[f"r2_{name}"] = r2_score(p[:, i], y[:, i])
        out[f"pearson_{name}"] = pearson_corr(p[:, i], y[:, i])

    return out


def evaluate_budget_policy(
    pred_candidate_gains: torch.Tensor,
    target_candidate_gains: torch.Tensor,
    exact_candidate_gains: torch.Tensor,
    costs: torch.Tensor,
    default_idx: int,
    candidate_indices: Sequence[int],
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    budget_grid: Sequence[float],
    target_name: str,
    first_order_candidate_gains: torch.Tensor | None = None,
) -> List[Dict[str, Any]]:
    """
    Uses validation predicted max gain to choose threshold for each budget.
    Then evaluates exact-cost policy on test.

    Policy:
      - choose candidate with highest predicted gain for the supervised target
      - switch if max predicted gain >= threshold
      - otherwise default
    """
    candidate_indices_t = torch.tensor(candidate_indices, dtype=torch.long)

    pred = pred_candidate_gains.float().cpu()
    target = target_candidate_gains.float().cpu()
    exact_gain = exact_candidate_gains.float().cpu()
    first_order = (
        None if first_order_candidate_gains is None else first_order_candidate_gains.float().cpu()
    )

    base_cost = costs[:, default_idx].float().cpu()
    candidate_costs = costs[:, candidate_indices_t].float().cpu()

    pred_best_score, pred_best_local = pred.max(dim=1)
    chosen_candidate_cost = candidate_costs.gather(1, pred_best_local.unsqueeze(1)).squeeze(1)

    target_chosen = target.gather(1, pred_best_local.unsqueeze(1)).squeeze(1)
    exact_gain_chosen = exact_gain.gather(1, pred_best_local.unsqueeze(1)).squeeze(1)
    first_order_chosen = (
        None
        if first_order is None
        else first_order.gather(1, pred_best_local.unsqueeze(1)).squeeze(1)
    )

    out = []

    for budget in budget_grid:
        threshold = threshold_for_budget(pred_best_score[val_idx], budget)

        switch = pred_best_score >= threshold
        switch_test = switch[test_idx]

        policy_cost = base_cost[test_idx].clone()
        policy_cost[switch_test] = chosen_candidate_cost[test_idx][switch_test]

        base_test = base_cost[test_idx]
        improvement = base_test.mean() - policy_cost.mean()

        switched_exact_gains = exact_gain_chosen[test_idx][switch_test]
        switched_pred_scores = pred_best_score[test_idx][switch_test]
        switched_target = target_chosen[test_idx][switch_test]

        row = {
            "budget_fraction": float(budget),
            "validation_threshold": float(threshold),
            "test_switch_rate": _safe_frac(switch_test),
            "test_policy_mean_cost": float(policy_cost.mean().item()),
            "test_base_mean_cost": float(base_test.mean().item()),
            "test_improvement_vs_base": float(improvement.item()),
            "test_exact_positive_switch_rate": _safe_frac(switched_exact_gains > 0),
            "test_mean_exact_gain_on_switches": _safe_mean(switched_exact_gains),
            f"test_mean_predicted_{target_name}_gain_on_switches": _safe_mean(
                switched_pred_scores
            ),
            f"test_mean_true_{target_name}_gain_on_switches": _safe_mean(switched_target),
            "num_test_switches": int(switch_test.sum().item()),
        }

        if target_name == "first_order":
            row["test_mean_predicted_fo_gain_on_switches"] = _safe_mean(
                switched_pred_scores
            )
            row["test_mean_true_fo_gain_on_switches"] = _safe_mean(switched_target)

        if first_order_chosen is not None:
            switched_first_order = first_order_chosen[test_idx][switch_test]
            row["test_mean_true_first_order_gain_on_switches"] = _safe_mean(
                switched_first_order
            )

        out.append(row)

    return out


def run_supervised_predictability_study(
    features: torch.Tensor,
    target_gains: torch.Tensor,
    exact_gains: torch.Tensor,
    costs: torch.Tensor,
    split_indices: Dict[str, torch.Tensor],
    candidate_indices: Sequence[int],
    default_idx: int,
    candidate_action_names: Sequence[str],
    args: argparse.Namespace,
    target_name: str,
    first_order_gains: torch.Tensor | None = None,
) -> Dict[str, Any]:
    x = features.float().cpu()
    y = target_gains[:, candidate_indices].float().cpu()
    exact_y = exact_gains[:, candidate_indices].float().cpu()
    first_order_y = (
        None if first_order_gains is None else first_order_gains[:, candidate_indices].float().cpu()
    )

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    x_std, _standardizer = standardize_with_stats(x, train_idx)

    out: Dict[str, Any] = {
        "feature_key": args.feature_key,
        "feature_dim": int(x.shape[1]),
        "target": target_name,
        "candidate_actions": list(candidate_action_names),
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
        "num_test": int(test_idx.numel()),
        "models": {},
    }

    models = set(parse_csv(args.models))

    if "ridge" in models:
        w = ridge_fit_augmented(
            x_train=x_std[train_idx],
            y_train=y[train_idx],
            ridge_lambda=args.ridge_lambda,
        )
        pred = ridge_predict(x_std, w)

        out["models"]["ridge"] = {
            "ridge_lambda": args.ridge_lambda,
            "train": evaluate_prediction_metrics(pred, y, train_idx, candidate_action_names),
            "val": evaluate_prediction_metrics(pred, y, val_idx, candidate_action_names),
            "test": evaluate_prediction_metrics(pred, y, test_idx, candidate_action_names),
            "budget_policy_test_exact_cost": evaluate_budget_policy(
                pred_candidate_gains=pred,
                target_candidate_gains=y,
                exact_candidate_gains=exact_y,
                costs=costs,
                default_idx=default_idx,
                candidate_indices=candidate_indices,
                val_idx=val_idx,
                test_idx=test_idx,
                budget_grid=[float(x) for x in parse_csv(args.budget_grid)],
                target_name=target_name,
                first_order_candidate_gains=first_order_y,
            ),
        }

    if "mlp" in models:
        hidden_dims = parse_hidden_dims(args.hidden_dims)

        mlp = train_mlp(
            x=x_std,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )

        pred = model_predict(mlp, x_std)

        out["models"]["mlp"] = {
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "train": evaluate_prediction_metrics(pred, y, train_idx, candidate_action_names),
            "val": evaluate_prediction_metrics(pred, y, val_idx, candidate_action_names),
            "test": evaluate_prediction_metrics(pred, y, test_idx, candidate_action_names),
            "budget_policy_test_exact_cost": evaluate_budget_policy(
                pred_candidate_gains=pred,
                target_candidate_gains=y,
                exact_candidate_gains=exact_y,
                costs=costs,
                default_idx=default_idx,
                candidate_indices=candidate_indices,
                val_idx=val_idx,
                test_idx=test_idx,
                budget_grid=[float(x) for x in parse_csv(args.budget_grid)],
                target_name=target_name,
                first_order_candidate_gains=first_order_y,
            ),
        }

    return out


def analyze_dataset(dataset_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"[load] {dataset_path}")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dataset at {dataset_path} to be a dict.")

    costs = get_costs(data)
    first_order_gains = get_first_order_gains(data)
    features = get_features(data, args.feature_key)

    if costs.ndim != 2:
        raise ValueError(f"Expected costs to be [N, A], got {tuple(costs.shape)}")
    if first_order_gains.shape != costs.shape:
        raise ValueError(
            f"Expected first_order_gains shape to match costs. "
            f"costs={tuple(costs.shape)}, first_order_gains={tuple(first_order_gains.shape)}"
        )
    if features.shape[0] != costs.shape[0]:
        raise ValueError(
            f"Expected features and costs to have same N. "
            f"features={tuple(features.shape)}, costs={tuple(costs.shape)}"
        )

    n, num_actions = costs.shape
    action_names = infer_action_names(data, num_actions)

    default_idx = find_action_index(args.default_action, action_names)
    candidate_indices = resolve_candidate_indices(
        candidate_actions=args.candidate_actions,
        action_names=action_names,
        default_idx=default_idx,
    )
    candidate_action_names = [action_names[i] for i in candidate_indices]

    selection_mask = build_selection_mask(
        data=data,
        n=n,
        group_ids=args.group_ids,
        layer_ids=args.layer_ids,
    )

    full_splits = get_split_indices(data, n=n, seed=args.seed)
    local_splits = restrict_split_indices(full_splits, selection_mask)

    selected = selection_mask.nonzero(as_tuple=False).flatten()

    costs_sel = costs[selected]
    first_order_sel = first_order_gains[selected]
    exact_gain_sel = compute_exact_gains(costs_sel, default_idx=default_idx)
    features_sel = features[selected]

    for split_name, idx in local_splits.items():
        if idx.numel() == 0:
            raise ValueError(
                f"Split {split_name!r} is empty after filtering. "
                f"Relax --group_ids/--layer_ids or rebuild split_indices."
            )

    config = data.get("config", {})
    layer_label = args.layer_label
    if not layer_label:
        if isinstance(config, dict) and "layer_idx" in config:
            layer_label = str(config["layer_idx"])
        else:
            layer_label = dataset_path.stem

    print(f"[info] layer_label={layer_label}")
    print(f"[info] selected examples={costs_sel.shape[0]}/{n}")
    print(f"[info] actions={action_names}")
    print(f"[info] default={action_names[default_idx]} index={default_idx}")
    print(f"[info] candidates={candidate_action_names}")

    oracle = evaluate_oracle_gain(
        costs=costs_sel,
        default_idx=default_idx,
        candidate_indices=candidate_indices,
        action_names=action_names,
        eps=args.gain_eps,
    )

    first_order_alignment = evaluate_first_order_alignment(
        costs=costs_sel,
        first_order_gains=first_order_sel,
        default_idx=default_idx,
        candidate_indices=candidate_indices,
        action_names=action_names,
        eps=args.gain_eps,
    )

    first_order_predictability = run_supervised_predictability_study(
        features=features_sel,
        target_gains=first_order_sel,
        exact_gains=exact_gain_sel,
        costs=costs_sel,
        split_indices=local_splits,
        candidate_indices=candidate_indices,
        default_idx=default_idx,
        candidate_action_names=candidate_action_names,
        args=args,
        target_name="first_order",
        first_order_gains=first_order_sel,
    )

    exact_predictability = run_supervised_predictability_study(
        features=features_sel,
        target_gains=exact_gain_sel,
        exact_gains=exact_gain_sel,
        costs=costs_sel,
        split_indices=local_splits,
        candidate_indices=candidate_indices,
        default_idx=default_idx,
        candidate_action_names=candidate_action_names,
        args=args,
        target_name="exact",
        first_order_gains=first_order_sel,
    )

    result = {
        "dataset_path": str(dataset_path),
        "layer_label": layer_label,
        "selection": {
            "group_ids": args.group_ids,
            "layer_ids": args.layer_ids,
            "num_selected": int(costs_sel.shape[0]),
        },
        "action_names": action_names,
        "default_action": action_names[default_idx],
        "candidate_actions": candidate_action_names,
        "oracle_gain": oracle,
        "first_order_exact_alignment": first_order_alignment,
        "deployable_feature_predictability_of_first_order": first_order_predictability,
        "deployable_feature_predictability_of_exact": exact_predictability,
    }

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose, for a gradient-proxy router dataset: "
            "1) oracle gain, 2) first-order/exact alignment, "
            "3) deployable feature predictability of first-order gains, "
            "4) deployable feature predictability of exact downstream gains."
        )
    )

    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more .pt datasets. Pass one per layer if you built layer-specific datasets.",
    )
    parser.add_argument(
        "--feature_key",
        type=str,
        choices=SUPPORTED_FEATURE_KEYS,
        required=True,
        help=(
            "Feature tensor to train from. Use 'features' for base/current-position "
            f"diagnostics or {ROUTER_FEATURE_KEY!r} for the deployable router features."
        ),
    )

    parser.add_argument("--default_action", type=str, default="soft")
    parser.add_argument(
        "--candidate_actions",
        type=str,
        default="nondefault",
        help="Comma-separated action names/indices, or 'nondefault'. Example: window_soft",
    )

    parser.add_argument(
        "--group_ids",
        type=str,
        default="",
        help="Optional comma-separated group ids to keep, e.g. '1'.",
    )
    parser.add_argument(
        "--layer_ids",
        type=str,
        default="",
        help="Optional comma-separated layer ids to keep if dataset contains layer_ids.",
    )
    parser.add_argument(
        "--layer_label",
        type=str,
        default="",
        help="Optional label for reporting when running a single layer dataset.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="ridge",
        help="Comma-separated from: ridge,mlp. Default: ridge",
    )
    parser.add_argument("--ridge_lambda", type=float, default=1.0)

    parser.add_argument("--hidden_dims", type=str, default="256,128")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument(
        "--budget_grid",
        type=str,
        default="0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.10,0.15",
        help="Budget fractions for validation-thresholded switch policy.",
    )
    parser.add_argument("--gain_eps", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output_path", type=str, required=True)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results = []

    for raw_path in args.dataset_paths:
        result = analyze_dataset(Path(raw_path), args)
        results.append(result)

    output = {
        "config": vars(args),
        "results": results,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"[done] wrote {output_path}")

    for result in results:
        label = result["layer_label"]
        oracle = result["oracle_gain"]
        align = result["first_order_exact_alignment"]

        print()
        print(f"[summary] {label}")
        print(f"  oracle_improvement_vs_base: {oracle['oracle_improvement_vs_base']:.8f}")
        print(f"  oracle_best_nondefault_rate: {oracle['oracle_best_nondefault_rate']:.6f}")
        print(f"  first_order/exact pearson nondefault: {align['flat_pearson_nondefault']:.6f}")
        print(f"  first_order/exact pairwise acc nondefault: {align['pairwise_rank_acc_nondefault']:.6f}")

        for result_key, target_label in [
            ("deployable_feature_predictability_of_first_order", "first_order"),
            ("deployable_feature_predictability_of_exact", "exact"),
        ]:
            pred = result[result_key]["models"]
            for model_name, metrics in pred.items():
                test = metrics["test"]
                print(f"  {model_name} test R2 predicting {target_label}: {test['r2']:.6f}")
                print(
                    f"  {model_name} test pearson predicting {target_label}: "
                    f"{test['flat_pearson']:.6f}"
                )

                budget_rows = metrics["budget_policy_test_exact_cost"]
                best_budget = max(budget_rows, key=lambda r: r["test_improvement_vs_base"])
                print(
                    f"  {model_name} best {target_label}-target budget exact improvement: "
                    f"{best_budget['test_improvement_vs_base']:.8f} "
                    f"at switch_rate={best_budget['test_switch_rate']:.6f}"
                )


if __name__ == "__main__":
    main()
