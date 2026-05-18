"""Focused sweep for first-order gain distillation routers."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from experiments.archive.task_study.gradient_proxy_router.datasets import (
    DEFAULT_ROUTER_DATASET_PATH,
    FIRST_ORDER_GAIN_KEY,
    ROUTER_FEATURE_KEY,
    RouterDatasetBuildConfig,
    add_gradient_state_dataset,
    add_sequence_state_features,
    build_or_load_combiner_dataset,
)


BASE_ACTION_NAME = "soft"
CANONICAL_CANDIDATE_ACTION = "window_soft"
CANONICAL_GROUP_ID = 1

EXACT_LOSS_WEIGHTS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
SWITCH_BUDGET_GRIDS = (
    (0.005, 0.01, 0.015, 0.02, 0.03),
    (0.01, 0.015, 0.02, 0.025, 0.03),
    (0.015, 0.02, 0.025, 0.03, 0.04),
)

ROUTER_HIDDEN_DIM = 128
ROUTER_DEPTH = 2
ROUTER_DROPOUT = 0.1
ROUTER_LEARNING_RATE = 1e-3
ROUTER_WEIGHT_DECAY = 1e-4
ROUTER_EPOCHS = 20
ROUTER_GRAD_CLIP = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


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


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    mean_costs = costs.float().mean(dim=0)
    return int(mean_costs.argmin().item())


def evaluate_actions(
    costs: torch.Tensor,
    actions: torch.Tensor,
    best_action: torch.Tensor | None = None,
) -> dict[str, float]:
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
    gap = float(best_fixed_loss) - float(oracle_loss)
    if abs(gap) < 1e-12:
        return 0.0
    return (float(best_fixed_loss) - float(router_loss)) / gap


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


def build_sweep_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the canonical gradient-proxy router sweep.")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_ROUTER_DATASET_PATH)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/deployable_routers/gradient_proxy_router/focused_sweep")
    return parser


def model_kwargs(input_dim: int, output_dim: int) -> dict[str, Any]:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": ROUTER_HIDDEN_DIM,
        "depth": ROUTER_DEPTH,
        "dropout": ROUTER_DROPOUT,
    }


@torch.no_grad()
def predict_outputs(model: torch.nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    model.eval()
    for batch_slice in batch_slices(features.shape[0], batch_size):
        rows.append(model(features[batch_slice].to(device)))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def apply_group_models(
    checkpoint_groups: dict[str, Any],
    dataset: dict[str, Any],
    ordered_idx: torch.Tensor,
    base_action_idx: int,
    model_kwargs: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features = dataset[ROUTER_FEATURE_KEY][ordered_idx].float()
    split_group_ids = dataset["group_ids"][ordered_idx].long()
    actions = torch.full((ordered_idx.numel(),), int(base_action_idx), dtype=torch.long)

    payload = checkpoint_groups.get(str(CANONICAL_GROUP_ID))
    if payload is None:
        return actions
    row_idx = torch.nonzero(split_group_ids == CANONICAL_GROUP_ID, as_tuple=False).squeeze(-1)
    if row_idx.numel() == 0:
        return actions

    model = build_gain_vector_model(**model_kwargs).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    feature_mean = payload["feature_mean"].to(device)
    feature_std = payload["feature_std"].to(device)
    pred_rows: list[torch.Tensor] = []
    for batch_slice in batch_slices(row_idx.numel(), batch_size):
        idx_slice = row_idx[batch_slice]
        xb = normalize_features(features[idx_slice].to(device), feature_mean, feature_std)
        pred_rows.append(model(xb))
    pred_gains = torch.cat(pred_rows, dim=0)
    actions[row_idx] = choose_budget_actions(
        predicted_gains=pred_gains,
        base_action_idx=base_action_idx,
        candidate_indices=payload["candidate_indices"],
        threshold=float(payload["score_threshold"]),
    ).cpu()

    return actions


def summarize_actions(
    dataset: dict[str, Any],
    ordered_idx: torch.Tensor,
    actions: torch.Tensor,
    base_action_idx: int,
) -> dict[str, Any]:
    summary = summarize_eval(
        costs=dataset["costs"][ordered_idx].float(),
        actions=actions,
        best_action=dataset["best_action"][ordered_idx].long(),
        base_action_idx=base_action_idx,
        action_names=list(dataset["action_names"]),
    )
    return summary


@torch.no_grad()
def evaluate_full_split(
    checkpoint_groups: dict[str, Any],
    dataset: dict[str, Any],
    batch_size: int,
    device: torch.device,
    model_kwargs: dict[str, Any],
    base_action_idx: int,
) -> dict[str, Any]:
    ordered_idx = get_ordered_split_indices(dataset, "test")
    actions = apply_group_models(
        checkpoint_groups=checkpoint_groups,
        dataset=dataset,
        ordered_idx=ordered_idx,
        base_action_idx=base_action_idx,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        device=device,
    )
    return summarize_actions(dataset, ordered_idx, actions, base_action_idx)


def train_canonical_sweep(
    dataset_path: str | Path,
    batch_size: int = 256,
    seed: int = 0,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str | Path = "outputs/deployable_routers/gradient_proxy_router/focused_sweep",
) -> list[dict[str, Any]]:
    set_seed(seed)
    dataset_path = Path(dataset_path)
    device = torch.device(device)
    output_dir = ensure_dir(output_dir)

    dataset = torch.load(dataset_path, map_location="cpu")
    if ROUTER_FEATURE_KEY not in dataset:
        raise KeyError(f"Dataset is missing canonical router features '{ROUTER_FEATURE_KEY}'")
    if FIRST_ORDER_GAIN_KEY not in dataset:
        raise KeyError(f"Dataset is missing canonical gain targets '{FIRST_ORDER_GAIN_KEY}'")

    candidate_names = [CANONICAL_CANDIDATE_ACTION]
    candidate_indices = candidate_action_indices(dataset["action_names"], candidate_names)

    train_idx = get_ordered_split_indices(dataset, "train")
    val_idx = get_ordered_split_indices(dataset, "val")
    base_action_idx = list(dataset["action_names"]).index(BASE_ACTION_NAME)
    input_dim = int(dataset[ROUTER_FEATURE_KEY].shape[1])
    output_dim = len(candidate_indices)
    router_model_kwargs = model_kwargs(input_dim=input_dim, output_dim=output_dim)

    configs = list(itertools.product(EXACT_LOSS_WEIGHTS, SWITCH_BUDGET_GRIDS))
    results: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []

    for exact_loss_weight, budget_grid in configs:
        checkpoint_groups: dict[str, Any] = {}
        group_summaries: list[dict[str, Any]] = []
        for group_id in [CANONICAL_GROUP_ID]:
            group_train_idx = train_idx[dataset["group_ids"][train_idx] == group_id]
            group_val_idx = val_idx[dataset["group_ids"][val_idx] == group_id]
            if group_train_idx.numel() == 0 or group_val_idx.numel() == 0:
                continue

            train_features_raw = dataset[ROUTER_FEATURE_KEY][group_train_idx].float()
            val_features_raw = dataset[ROUTER_FEATURE_KEY][group_val_idx].float()
            feature_mean, feature_std = compute_feature_normalizer(train_features_raw)
            train_features = normalize_features(train_features_raw, feature_mean, feature_std)
            val_features = normalize_features(val_features_raw, feature_mean, feature_std)

            train_gain_targets = dataset[FIRST_ORDER_GAIN_KEY][group_train_idx][:, candidate_indices].float()
            train_costs = dataset["costs"][group_train_idx].float()
            val_costs = dataset["costs"][group_val_idx].float()
            val_best = dataset["best_action"][group_val_idx].long()
            exact_targets = None
            if exact_loss_weight > 0:
                exact_targets = build_exact_pairwise_targets(
                    costs=train_costs,
                    base_action_idx=base_action_idx,
                    candidate_indices=candidate_indices,
                )

            model = build_gain_vector_model(**router_model_kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=ROUTER_LEARNING_RATE, weight_decay=ROUTER_WEIGHT_DECAY)
            best_payload: dict[str, Any] | None = None

            for _ in range(ROUTER_EPOCHS):
                perm = torch.randperm(train_features.shape[0])
                model.train()
                for batch_slice in batch_slices(train_features.shape[0], batch_size):
                    batch_ids = perm[batch_slice]
                    xb = train_features[batch_ids].to(device)
                    yb = train_gain_targets[batch_ids].to(device)
                    zb = exact_targets[batch_ids].to(device) if exact_targets is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(xb)
                    loss = distillation_loss(pred, yb, zb, exact_loss_weight=exact_loss_weight)
                    loss.backward()
                    if ROUTER_GRAD_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), ROUTER_GRAD_CLIP)
                    optimizer.step()

                val_pred = predict_outputs(model, val_features, batch_size, device).cpu()
                best_budget, best_threshold, val_metrics, _ = tune_budget_policy(
                    predicted_gains=val_pred,
                    costs=val_costs,
                    best_action=val_best,
                    base_action_idx=base_action_idx,
                    candidate_indices=candidate_indices,
                    budget_grid=budget_grid,
                )
                if best_payload is None or val_metrics["mean_cost"] < best_payload["best_val_metrics"]["mean_cost"]:
                    best_payload = {
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "feature_mean": feature_mean.detach().cpu(),
                        "feature_std": feature_std.detach().cpu(),
                        "candidate_indices": list(candidate_indices),
                        "candidate_names": list(candidate_names),
                        "budget_fraction": float(best_budget),
                        "score_threshold": float(best_threshold),
                        "best_val_metrics": val_metrics,
                    }

            assert best_payload is not None
            checkpoint_groups[str(group_id)] = best_payload
            group_summaries.append(
                {
                    "group_id": int(group_id),
                    "best_val_cost": float(best_payload["best_val_metrics"]["mean_cost"]),
                    "best_val_switch_rate": float(best_payload["best_val_metrics"]["switch_rate"]),
                    "budget_fraction": float(best_payload["budget_fraction"]),
                    "score_threshold": float(best_payload["score_threshold"]),
                }
            )

        full_eval = evaluate_full_split(
            checkpoint_groups=checkpoint_groups,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            model_kwargs=router_model_kwargs,
            base_action_idx=base_action_idx,
        )
        record = {
            "exact_loss_weight": float(exact_loss_weight),
            "budget_grid": list(budget_grid),
            "group_id": CANONICAL_GROUP_ID,
            "candidate_action": CANONICAL_CANDIDATE_ACTION,
            "router": full_eval["router"],
            "base_policy": full_eval["base_policy"],
            "best_fixed": full_eval["best_fixed"],
            "per_group": group_summaries,
        }
        results.append(record)
        checkpoint_path = output_dir / f"distill_w{exact_loss_weight:.3f}_b{len(results):02d}.pt"
        torch.save(
            {
                "training_scope": "gradient_proxy_router",
                "dataset_path": str(dataset_path),
                "base_action_idx": int(base_action_idx),
                "base_action_name": dataset["action_names"][base_action_idx],
                "action_names": list(dataset["action_names"]),
                "model_kwargs": router_model_kwargs,
                "group_models": checkpoint_groups,
                "summary_rows": group_summaries,
                "config": {
                    "group_id": CANONICAL_GROUP_ID,
                    "candidate_action": CANONICAL_CANDIDATE_ACTION,
                    "exact_loss_weights": list(EXACT_LOSS_WEIGHTS),
                    "switch_budget_grids": [list(grid) for grid in SWITCH_BUDGET_GRIDS],
                    "batch_size": int(batch_size),
                    "seed": int(seed),
                    "device": str(device),
                },
            },
            checkpoint_path,
        )
        checkpoint_records.append({"path": str(checkpoint_path), "record": record})
        print(
            f"exact_loss_weight={exact_loss_weight:.3f} "
            f"router_cost={record['router']['mean_cost']:.6f} "
            f"improvement={record['router']['improvement_vs_best_fixed']:.6f} "
            f"switch_rate={record['router']['switch_rate']:.4f}"
        )

    results.sort(key=lambda row: row["router"]["mean_cost"])
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if checkpoint_records:
        best_record = min(checkpoint_records, key=lambda row: row["record"]["router"]["mean_cost"])
        (output_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "best_checkpoint_path": best_record["path"],
                    "best_record": best_record["record"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    findings_lines = ["Influence Distillation Focused Sweep", "", "Best Overall"]
    if results:
        best = results[0]
        findings_lines.append(
            f"group_id={best['group_id']} "
            f"action={best['candidate_action']} "
            f"exact_loss_weight={best['exact_loss_weight']:.3f} "
            f"router_cost={best['router']['mean_cost']:.6f} "
            f"best_fixed={best['best_fixed']['mean_cost']:.6f} "
            f"improvement={best['router']['improvement_vs_best_fixed']:.6f} "
            f"switch_rate={best['router']['switch_rate']:.4f}"
        )
        findings_lines.extend(["", "Top 10 Runs"])
        for row in results[:10]:
            findings_lines.append(
                f"exact={row['exact_loss_weight']:.3f} "
                f"router={row['router']['mean_cost']:.6f} "
                f"impr={row['router']['improvement_vs_best_fixed']:.6f} "
                f"switch_rate={row['router']['switch_rate']:.4f} "
                f"budget_grid={','.join(f'{x:.3f}' for x in row['budget_grid'])}"
            )
        positive = [row for row in results if row["router"]["improvement_vs_best_fixed"] > 0]
        findings_lines.extend(["", f"Positive Runs: {len(positive)}/{len(results)}"])
    (output_dir / "findings.txt").write_text("\n".join(findings_lines) + "\n", encoding="utf-8")
    return results


def sweep_main(argv: list[str] | None = None) -> None:
    train_canonical_sweep(**vars(build_sweep_arg_parser().parse_args(argv)))


@torch.no_grad()
def checkpoint_actions(
    checkpoint: dict[str, Any],
    dataset: dict[str, Any],
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_idx = get_ordered_split_indices(dataset, split)
    ordered_idx = ordered_idx[dataset["group_ids"][ordered_idx] == CANONICAL_GROUP_ID]
    actions = apply_group_models(
        checkpoint_groups=checkpoint["group_models"],
        dataset=dataset,
        ordered_idx=ordered_idx,
        base_action_idx=int(checkpoint["base_action_idx"]),
        model_kwargs=checkpoint["model_kwargs"],
        batch_size=batch_size,
        device=device,
    )
    return ordered_idx, actions


EXISTING_SEQUENCE_DATASETS = {
    4: Path("outputs/influence_router/group6_sequence_only_scalar_dataset.pt"),
    11: Path("outputs/deployable_routers/gradient_proxy_router/last_layer/group6_sequence_only_scalar_dataset_layer11.pt"),
}


def build_layer_robustness_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run layer/seed robustness checks for the gradient-proxy router.")
    parser.add_argument("--layers", type=str, default="4,11")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output_root", type=str, default="outputs/deployable_routers/gradient_proxy_router/layer_robustness")
    parser.add_argument("--dataset_device", type=str, default="cuda")
    parser.add_argument("--sweep_device", type=str, default="cpu")
    parser.add_argument("--rebuild_datasets", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    return parser


def dataset_paths(output_root: Path, layer_idx: int) -> tuple[Path, Path, Path]:
    layer_dir = output_root / f"layer{layer_idx}"
    return (
        layer_dir / f"combiner_dataset_group6_layer{layer_idx}.pt",
        layer_dir / f"gradient_state_dataset_group6_layer{layer_idx}.pt",
        layer_dir / f"group6_sequence_only_scalar_dataset_layer{layer_idx}.pt",
    )


def rebuild_dataset(output_root: Path, layer_idx: int, device: str, force: bool) -> Path:
    combiner_path, gradient_path, sequence_path = dataset_paths(output_root, layer_idx)
    if sequence_path.exists() and not force:
        return sequence_path
    existing_path = EXISTING_SEQUENCE_DATASETS.get(layer_idx)
    if existing_path is not None and existing_path.exists() and not force:
        return existing_path

    build_or_load_combiner_dataset(
        RouterDatasetBuildConfig(
            model_name="openai-community/gpt2",
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            split="validation",
            text_field="text",
            max_texts=200,
            block_size=96,
            batch_size=4,
            max_chunks=64,
            layer_idx=layer_idx,
            head_indices="all",
            min_context=16,
            position_stride=1,
            replace_mode="multi_head_single_pos_shared",
            head_group_size=6,
            head_group_strategy="contiguous",
            seed=0,
            device=device,
            output_path=str(combiner_path),
            rebuild=True,
        ),
    )
    add_gradient_state_dataset(
        base_dataset_path=combiner_path,
        output_path=gradient_path,
        device=device,
    )
    add_sequence_state_features(
        base_dataset_path=gradient_path,
        output_path=sequence_path,
    )
    return sequence_path


def run_sweep(dataset_path: Path, output_dir: Path, seed: int, device: str) -> dict:
    train_canonical_sweep(
        dataset_path=dataset_path,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )
    return json.loads((output_dir / "best_checkpoint.json").read_text(encoding="utf-8"))


@torch.no_grad()
def row_metrics(checkpoint_path: str, dataset_path: Path, device: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    dataset = torch.load(dataset_path, map_location="cpu")
    base_action_idx = int(checkpoint["base_action_idx"])
    ordered_idx, actions = checkpoint_actions(
        checkpoint=checkpoint,
        dataset=dataset,
        split="test",
        batch_size=256,
        device=torch.device(device),
    )
    costs = dataset["costs"][ordered_idx].float()

    row_ids = torch.arange(costs.shape[0])
    return {
        "chunk_ids": dataset["chunk_ids"][ordered_idx].long(),
        "router_cost": costs[row_ids, actions],
        "soft_cost": costs[:, base_action_idx],
        "oracle_cost": costs.min(dim=1).values,
        "switched": (actions != base_action_idx).float(),
    }


def bootstrap_gain(row_data: dict[str, torch.Tensor], samples: int, seed: int) -> dict[str, float]:
    if samples <= 0:
        return {}
    generator = torch.Generator().manual_seed(seed)
    unique_chunks = row_data["chunk_ids"].unique()
    gains: list[float] = []
    n_chunks = unique_chunks.numel()
    for _ in range(samples):
        sampled = unique_chunks[torch.randint(0, n_chunks, (n_chunks,), generator=generator)]
        masks = [(row_data["chunk_ids"] == chunk_id) for chunk_id in sampled]
        mask = torch.stack(masks, dim=0)
        router = row_data["router_cost"].unsqueeze(0).expand(n_chunks, -1)[mask].mean()
        soft = row_data["soft_cost"].unsqueeze(0).expand(n_chunks, -1)[mask].mean()
        gains.append(float((soft - router).item()))
    gains_sorted = sorted(gains)
    lo = gains_sorted[int(0.025 * (samples - 1))]
    hi = gains_sorted[int(0.975 * (samples - 1))]
    return {"bootstrap_gain_p025": lo, "bootstrap_gain_p975": hi}


def summarize_layer(rows: list[dict]) -> dict[str, object]:
    gains = [row["router_gain"] for row in rows]
    return {
        "num_seeds": len(rows),
        "mean_router_gain": mean(gains),
        "std_router_gain": pstdev(gains) if len(gains) > 1 else 0.0,
        "positive_seeds": sum(gain > 0 for gain in gains),
        "rows": rows,
    }


def layer_robustness_main(argv: list[str] | None = None) -> None:
    args = build_layer_robustness_arg_parser().parse_args(argv)
    layers = parse_int_csv(args.layers)
    seeds = parse_int_csv(args.seeds)
    output_root = Path(args.output_root)
    summary: dict[str, object] = {
        "layers": layers,
        "seeds": seeds,
        "output_root": str(output_root),
        "per_layer": {},
    }

    for layer_idx in layers:
        dataset_path = rebuild_dataset(output_root, layer_idx, args.dataset_device, force=args.rebuild_datasets)
        layer_rows: list[dict] = []
        for seed in seeds:
            run_dir = output_root / f"layer{layer_idx}" / f"seed{seed}"
            best_payload = run_sweep(dataset_path, run_dir, seed=seed, device=args.sweep_device)
            record = best_payload["best_record"]
            router = record["router"]
            base = record["base_policy"]
            row_data = row_metrics(best_payload["best_checkpoint_path"], dataset_path, args.sweep_device)
            row = {
                "layer_idx": layer_idx,
                "seed": seed,
                "dataset_path": str(dataset_path),
                "best_checkpoint_path": best_payload["best_checkpoint_path"],
                "oracle_cost": router["oracle_cost"],
                "router_mean_cost": router["mean_cost"],
                "soft_mean_cost": base["mean_cost"],
                "router_gain": router["improvement_vs_base_policy"],
                "switch_rate": router["switch_rate"],
                "exact_loss_weight": record["exact_loss_weight"],
                "budget_grid": record["budget_grid"],
                "selected": record["per_group"],
                **bootstrap_gain(row_data, samples=args.bootstrap_samples, seed=args.bootstrap_seed + layer_idx * 100 + seed),
            }
            layer_rows.append(row)
            print(json.dumps(row, indent=2))
        summary["per_layer"][str(layer_idx)] = summarize_layer(layer_rows)  # type: ignore[index]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
