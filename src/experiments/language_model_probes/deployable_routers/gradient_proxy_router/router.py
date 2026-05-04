"""Focused sweep for first-order gain distillation routers."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from experiments.language_model_probes.deployable_routers.gradient_proxy_router.datasets import (
    RouterDatasetBuildConfig,
    RouterFeatureConfig,
    add_gradient_state_dataset,
    add_sequence_state_features,
    build_or_load_combiner_dataset,
)
from experiments.language_model_probes.deployable_routers.gradient_proxy_router.utils import (
    batch_slices,
    best_fixed_soft,
    build_gain_vector_model,
    build_exact_pairwise_targets,
    candidate_action_indices,
    choose_budget_actions,
    compute_feature_normalizer,
    distillation_loss,
    ensure_dir,
    get_ordered_split_indices,
    normalize_features,
    parse_float_csv,
    parse_int_csv,
    parse_str_csv,
    selected_group_ids,
    set_seed,
    summarize_eval,
    tune_budget_policy,
)


def build_sweep_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep focused distillation-router configs.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--feature_key", type=str, default="features_gradient_state")
    parser.add_argument("--gain_key", type=str, default="first_order_gains")
    parser.add_argument("--group_ids", type=str, default="1")
    parser.add_argument("--candidate_actions", type=str, default="window_soft")
    parser.add_argument("--exact_loss_weights", type=str, default="0.0,0.05,0.1,0.2,0.5")
    parser.add_argument("--budget_grids", type=str, default="0.001,0.002,0.005,0.01,0.02|0.005,0.01,0.02,0.05,0.08,0.10")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/deployable_routers/gradient_proxy_router/focused_sweep")
    return parser


def parse_grid_list(text: str) -> list[list[float]]:
    return [parse_float_csv(chunk) for chunk in text.split("|") if chunk.strip()]


def model_kwargs_from_args(args: argparse.Namespace, input_dim: int, output_dim: int) -> dict[str, Any]:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
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
    feature_key: str,
    ordered_idx: torch.Tensor,
    base_action_idx: int,
    model_kwargs: dict[str, Any],
    batch_size: int,
    device: torch.device,
    group_ids: list[int] | None = None,
) -> torch.Tensor:
    features = dataset[feature_key][ordered_idx].float()
    split_group_ids = dataset["group_ids"][ordered_idx].long()
    actions = torch.full((ordered_idx.numel(),), int(base_action_idx), dtype=torch.long)
    selected_groups = group_ids if group_ids is not None else sorted(int(x) for x in split_group_ids.unique().tolist())

    for group_id in selected_groups:
        payload = checkpoint_groups.get(str(group_id))
        if payload is None:
            continue
        row_idx = torch.nonzero(split_group_ids == int(group_id), as_tuple=False).squeeze(-1)
        if row_idx.numel() == 0:
            continue
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
    feature_key: str,
    batch_size: int,
    device: torch.device,
    model_kwargs: dict[str, Any],
    base_action_idx: int,
) -> dict[str, Any]:
    ordered_idx = get_ordered_split_indices(dataset, "test")
    actions = apply_group_models(
        checkpoint_groups=checkpoint_groups,
        dataset=dataset,
        feature_key=feature_key,
        ordered_idx=ordered_idx,
        base_action_idx=base_action_idx,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        device=device,
    )
    return summarize_actions(dataset, ordered_idx, actions, base_action_idx)


def train_sweep(
    dataset_path: str | Path,
    feature_key: str = "features_gradient_state",
    gain_key: str = "first_order_gains",
    group_ids: str = "1",
    candidate_actions: str = "window_soft",
    exact_loss_weights: str = "0.0,0.05,0.1,0.2,0.5",
    budget_grids: str = "0.001,0.002,0.005,0.01,0.02|0.005,0.01,0.02,0.05,0.08,0.10",
    hidden_dim: int = 128,
    depth: int = 2,
    dropout: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str | Path = "outputs/deployable_routers/gradient_proxy_router/focused_sweep",
) -> list[dict[str, Any]]:
    args = argparse.Namespace(
        dataset_path=str(dataset_path),
        feature_key=feature_key,
        gain_key=gain_key,
        group_ids=group_ids,
        candidate_actions=candidate_actions,
        exact_loss_weights=exact_loss_weights,
        budget_grids=budget_grids,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        grad_clip=grad_clip,
        seed=seed,
        device=str(device),
        output_dir=str(output_dir),
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)

    dataset = torch.load(Path(args.dataset_path), map_location="cpu")
    if args.feature_key not in dataset:
        raise KeyError(f"Unknown feature key '{args.feature_key}'")
    if args.gain_key not in dataset:
        raise KeyError(f"Unknown gain key '{args.gain_key}'")

    exact_loss_weights = parse_float_csv(args.exact_loss_weights)
    budget_grids = parse_grid_list(args.budget_grids)
    group_ids = selected_group_ids(dataset, args.group_ids)
    candidate_names = parse_str_csv(args.candidate_actions)
    candidate_indices = candidate_action_indices(dataset["action_names"], candidate_names)

    train_idx = get_ordered_split_indices(dataset, "train")
    val_idx = get_ordered_split_indices(dataset, "val")
    base_action_idx = best_fixed_soft(dataset, train_idx)
    input_dim = int(dataset[args.feature_key].shape[1])
    output_dim = len(candidate_indices)
    model_kwargs = model_kwargs_from_args(args, input_dim=input_dim, output_dim=output_dim)

    configs = list(itertools.product(exact_loss_weights, budget_grids))
    results: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []

    for exact_loss_weight, budget_grid in configs:
        checkpoint_groups: dict[str, Any] = {}
        group_summaries: list[dict[str, Any]] = []
        for group_id in group_ids:
            group_train_idx = train_idx[dataset["group_ids"][train_idx] == group_id]
            group_val_idx = val_idx[dataset["group_ids"][val_idx] == group_id]
            if group_train_idx.numel() == 0 or group_val_idx.numel() == 0:
                continue

            train_features_raw = dataset[args.feature_key][group_train_idx].float()
            val_features_raw = dataset[args.feature_key][group_val_idx].float()
            feature_mean, feature_std = compute_feature_normalizer(train_features_raw)
            train_features = normalize_features(train_features_raw, feature_mean, feature_std)
            val_features = normalize_features(val_features_raw, feature_mean, feature_std)

            train_gain_targets = dataset[args.gain_key][group_train_idx][:, candidate_indices].float()
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

            model = build_gain_vector_model(**model_kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            best_payload: dict[str, Any] | None = None

            for _ in range(args.epochs):
                perm = torch.randperm(train_features.shape[0])
                model.train()
                for batch_slice in batch_slices(train_features.shape[0], args.batch_size):
                    batch_ids = perm[batch_slice]
                    xb = train_features[batch_ids].to(device)
                    yb = train_gain_targets[batch_ids].to(device)
                    zb = exact_targets[batch_ids].to(device) if exact_targets is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(xb)
                    loss = distillation_loss(pred, yb, zb, exact_loss_weight=exact_loss_weight)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                val_pred = predict_outputs(model, val_features, args.batch_size, device).cpu()
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
            feature_key=args.feature_key,
            batch_size=args.batch_size,
            device=device,
            model_kwargs=model_kwargs,
            base_action_idx=base_action_idx,
        )
        record = {
            "exact_loss_weight": float(exact_loss_weight),
            "budget_grid": list(budget_grid),
            "group_ids": list(group_ids),
            "candidate_actions": list(candidate_names),
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
                "dataset_path": args.dataset_path,
                "feature_key": args.feature_key,
                "gain_key": args.gain_key,
                "base_action_idx": int(base_action_idx),
                "base_action_name": dataset["action_names"][base_action_idx],
                "action_names": list(dataset["action_names"]),
                "model_kwargs": model_kwargs,
                "group_models": checkpoint_groups,
                "summary_rows": group_summaries,
                "args": vars(args),
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
            f"group_ids={','.join(str(x) for x in best['group_ids'])} "
            f"actions={','.join(best['candidate_actions'])} "
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
    train_sweep(**vars(build_sweep_arg_parser().parse_args(argv)))



# Checkpoint evaluation.
def build_eval_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a first-order gain distillation router.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--group_ids", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def action_histogram(actions: torch.Tensor, action_names: list[str]) -> dict[str, dict[str, float]]:
    counts = torch.bincount(actions.long(), minlength=len(action_names)).float()
    total = float(max(actions.numel(), 1))
    return {
        action_names[idx]: {
            "count": float(counts[idx].item()),
            "fraction": float(counts[idx].item() / total),
        }
        for idx in range(len(action_names))
    }


@torch.no_grad()
def checkpoint_actions(
    checkpoint: dict[str, Any],
    dataset: dict[str, Any],
    split: str,
    batch_size: int,
    device: torch.device,
    group_ids: str | list[int] = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    requested_groups = group_ids if isinstance(group_ids, list) else selected_group_ids(dataset, group_ids)
    ordered_idx = get_ordered_split_indices(dataset, split)
    group_tensor = torch.tensor(requested_groups, dtype=torch.long)
    ordered_idx = ordered_idx[torch.isin(dataset["group_ids"][ordered_idx], group_tensor)]
    actions = apply_group_models(
        checkpoint_groups=checkpoint["group_models"],
        dataset=dataset,
        feature_key=checkpoint["feature_key"],
        ordered_idx=ordered_idx,
        base_action_idx=int(checkpoint["base_action_idx"]),
        model_kwargs=checkpoint["model_kwargs"],
        batch_size=batch_size,
        device=device,
        group_ids=list(requested_groups),
    )
    return ordered_idx, actions


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path = "",
    split: str = "test",
    batch_size: int = 256,
    group_ids: str = "",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_path = Path(dataset_path or checkpoint["dataset_path"])
    dataset = torch.load(Path(dataset_path), map_location="cpu")
    base_action_idx = int(checkpoint["base_action_idx"])
    ordered_idx, actions = checkpoint_actions(
        checkpoint=checkpoint,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        device=device,
        group_ids=group_ids,
    )

    summary = summarize_actions(dataset, ordered_idx, actions, base_action_idx)
    summary.update(
        {
            "scope": dataset["metadata"].get("evaluation_scope", "pointwise_local_next_token_nll"),
            "training_scope": checkpoint["training_scope"],
            "split": split,
            "dataset_path": str(dataset_path),
            "checkpoint_path": str(checkpoint_path),
            "feature_key": checkpoint["feature_key"],
            "gain_key": checkpoint["gain_key"],
            "action_names": list(dataset["action_names"]),
            "diagnostics": {
                "chosen_action_histogram": action_histogram(actions, list(dataset["action_names"])),
            },
        }
    )
    return summary


def eval_main(argv: list[str] | None = None) -> None:
    args = build_eval_arg_parser().parse_args(argv)
    set_seed(args.seed)
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        group_ids=args.group_ids,
        device=args.device,
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


# Layer/seed robustness experiment.
BUDGET_GRIDS = (
    "0.005,0.01,0.015,0.02,0.03|"
    "0.01,0.015,0.02,0.025,0.03|"
    "0.015,0.02,0.025,0.03,0.04"
)
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
        RouterFeatureConfig(),
    )
    add_gradient_state_dataset(
        base_dataset_path=combiner_path,
        output_path=gradient_path,
        device=device,
    )
    add_sequence_state_features(
        base_dataset_path=gradient_path,
        output_path=sequence_path,
        base_feature_key="features_gradient_state",
        output_feature_key="sequence_only_scalar_features_gradient_state",
        window_size=4,
        summary_stats=("mean", "std", "max", "delta_to_mean"),
        sequence_feature_patterns=(
            "attn_",
            "topk_recency",
            "pred_norm_",
            "pred_teacher_",
            "pred_cos_",
            "pred_l2_",
            "absolute_position",
            "normalized_position",
            "context_length",
        ),
        include_history_fraction=True,
    )
    return sequence_path


def run_sweep(dataset_path: Path, output_dir: Path, seed: int, device: str) -> dict:
    train_sweep(
        dataset_path=dataset_path,
        feature_key="sequence_only_scalar_features_gradient_state",
        group_ids="1",
        candidate_actions="window_soft",
        exact_loss_weights="0.05,0.08,0.10,0.12,0.15,0.20",
        budget_grids=BUDGET_GRIDS,
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
