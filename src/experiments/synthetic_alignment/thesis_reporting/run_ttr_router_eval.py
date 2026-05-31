from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from experiments.synthetic_alignment.config import RouterExperimentConfig
from experiments.synthetic_alignment.router_dataset import RouterDatasetBuilder
from experiments.synthetic_alignment.experiment_routers import (
    evaluate_logistic_router,
    evaluate_loss_router,
    train_logistic_router,
    train_loss_mlp_router,
)


DEFAULT_TASK_NAMES = [
    "piecewise_linear",
    "shifted_local_map",
    "smooth_nonlinear_local",
]

DEFAULT_ROUTER_LEARNERS = [
    "soft",
    "sharp",
    "window_soft",
    "local_linear_attention",
]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset_three_way(dataset, seed: int, train_frac: float, val_frac: float):
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    n_total = dataset.X.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)

    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return (
        dataset.subset(train_idx),
        dataset.subset(val_idx),
        dataset.subset(test_idx),
    )


def standardize_train_val_test(train_ds, val_ds, test_ds, eps: float = 1e-6):
    mu = train_ds.X.mean(dim=0, keepdim=True)
    std = train_ds.X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)

    x_train = (train_ds.X - mu) / std
    x_val = (val_ds.X - mu) / std
    x_test = (test_ds.X - mu) / std

    return x_train, x_val, x_test


def compute_regrets(losses: torch.Tensor) -> torch.Tensor:
    return losses - losses.min(dim=-1, keepdim=True).values


def normalize_taskwise_regrets(
    regrets_train: torch.Tensor,
    task_ids_train: torch.Tensor,
    regrets_val: torch.Tensor,
    task_ids_val: torch.Tensor,
    regrets_test: torch.Tensor,
    task_ids_test: torch.Tensor,
    eps: float = 1e-8,
):
    regrets_train_norm = torch.zeros_like(regrets_train)
    regrets_val_norm = torch.zeros_like(regrets_val)
    regrets_test_norm = torch.zeros_like(regrets_test)

    unique_train = torch.unique(task_ids_train)

    for t in unique_train.tolist():
        train_mask = task_ids_train == t
        val_mask = task_ids_val == t
        test_mask = task_ids_test == t

        scale = regrets_train[train_mask].mean().item()
        scale = max(scale, eps)

        regrets_train_norm[train_mask] = regrets_train[train_mask] / scale
        if val_mask.any():
            regrets_val_norm[val_mask] = regrets_val[val_mask] / scale
        if test_mask.any():
            regrets_test_norm[test_mask] = regrets_test[test_mask] / scale

    return regrets_train_norm, regrets_val_norm, regrets_test_norm


def per_task_metrics(
    pred: torch.Tensor,
    losses: torch.Tensor,
    task_ids: torch.Tensor,
    task_to_idx: dict[str, int],
) -> list[dict[str, float | str]]:
    if pred.device != losses.device:
        pred = pred.to(losses.device)
    if task_ids.device != losses.device:
        task_ids = task_ids.to(losses.device)

    idx_to_task = {v: k for k, v in task_to_idx.items()}
    rows = []

    for idx, task_name in sorted(idx_to_task.items()):
        mask = task_ids == idx
        if not mask.any():
            continue

        losses_t = losses[mask]
        pred_t = pred[mask]

        routed_loss = losses_t[
            torch.arange(losses_t.shape[0], device=losses.device),
            pred_t,
        ].mean().item()

        oracle_loss = losses_t.min(dim=-1).values.mean().item()
        best_fixed_loss = losses_t.mean(dim=0).min().item()

        gap_closed = (
            (best_fixed_loss - routed_loss)
            / max(best_fixed_loss - oracle_loss, 1e-8)
        )

        rows.append(
            {
                "task": task_name,
                "best_fixed_loss": best_fixed_loss,
                "routed_loss": routed_loss,
                "oracle_loss": oracle_loss,
                "oracle_gap_closed_frac": gap_closed,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thesis Chapter 3 TTR readout router diagnostic."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/synthetic_alignment/ch3_ttr_readouts",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASK_NAMES),
        help="Comma-separated task names.",
    )
    parser.add_argument(
        "--learners",
        type=str,
        default=",".join(DEFAULT_ROUTER_LEARNERS),
        help="Comma-separated router action learners.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n_batches_per_task", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--dv", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--min_context", type=int, default=8)
    parser.add_argument("--k_sharp", type=int, default=4)
    parser.add_argument("--k_knn_mean", type=int, default=4)
    parser.add_argument("--k_linear_local", type=int, default=16)
    parser.add_argument("--ridge_lambda", type=float, default=1e-1)
    parser.add_argument("--local_kernel_beta", type=float, default=1.0)
    parser.add_argument("--logistic_epochs", type=int, default=4000)
    parser.add_argument("--loss_mlp_epochs", type=int, default=4000)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = parse_str_list(args.tasks)
    learner_names = parse_str_list(args.learners)
    seeds = parse_int_list(args.seeds)

    per_task_rows = []
    aggregate_rows = []

    for seed in seeds:
        print(f"\n=== router seed {seed} ===")
        set_seed(seed)

        cfg = RouterExperimentConfig(
            L=args.L,
            d=args.d,
            dv=args.dv,
            batch_size=args.batch_size,
            sigma=args.sigma,
            device=args.device,
            local_kernel_beta=args.local_kernel_beta,
            k_sharp=args.k_sharp,
            k_knn_mean=args.k_knn_mean,
            k_linear_local=args.k_linear_local,
            ridge_lambda=args.ridge_lambda,
            min_context=args.min_context,
        )

        builder = RouterDatasetBuilder(cfg=cfg, learners=learner_names)
        dataset = builder.build(
            task_names=task_names,
            n_batches_per_task=args.n_batches_per_task,
        )

        train_ds, val_ds, test_ds = split_dataset_three_way(
            dataset=dataset,
            seed=seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
        x_train, x_val, x_test = standardize_train_val_test(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
        )

        print(
            f"Dataset sizes: train={train_ds.X.shape[0]}, "
            f"val={val_ds.X.shape[0]}, test={test_ds.X.shape[0]}"
        )

        # Logistic winner classifier.
        print("Training logistic router")
        logistic = train_logistic_router(
            x_train,
            train_ds.y,
            x_val,
            val_ds.y,
            lr=1e-2,
            weight_decay=1e-4,
            epochs=args.logistic_epochs,
        )

        logistic_eval = evaluate_logistic_router(
            logistic,
            x_test,
            test_ds.y,
            test_ds.losses,
        )

        for row in per_task_metrics(
            pred=logistic_eval["pred"],
            losses=test_ds.losses,
            task_ids=test_ds.task_ids,
            task_to_idx=test_ds.task_to_idx,
        ):
            row.update({"seed": seed, "router": "logistic"})
            per_task_rows.append(row)

        aggregate_rows.append(
            {
                "seed": seed,
                "router": "logistic",
                "acc": logistic_eval["acc"],
                "routed_loss": logistic_eval["routed_loss"],
                "best_fixed_loss": logistic_eval["best_fixed_loss"],
                "oracle_loss": logistic_eval["oracle_loss"],
                "oracle_gap_closed_frac": logistic_eval["oracle_gap_closed_frac"],
            }
        )

        # Loss-aware MLP router.
        print("Training loss-aware MLP router")
        train_regrets = compute_regrets(train_ds.losses)
        val_regrets = compute_regrets(val_ds.losses)
        test_regrets = compute_regrets(test_ds.losses)

        train_regrets_norm, val_regrets_norm, _ = normalize_taskwise_regrets(
            train_regrets,
            train_ds.task_ids,
            val_regrets,
            val_ds.task_ids,
            test_regrets,
            test_ds.task_ids,
        )

        loss_mlp = train_loss_mlp_router(
            x_train,
            train_regrets_norm,
            x_val,
            val_regrets_norm,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=args.loss_mlp_epochs,
            hidden_dim=64,
        )

        loss_mlp_eval = evaluate_loss_router(
            loss_mlp,
            x_test,
            test_ds.y,
            test_ds.losses,
        )

        for row in per_task_metrics(
            pred=loss_mlp_eval["pred"],
            losses=test_ds.losses,
            task_ids=test_ds.task_ids,
            task_to_idx=test_ds.task_to_idx,
        ):
            row.update({"seed": seed, "router": "loss_mlp"})
            per_task_rows.append(row)

        aggregate_rows.append(
            {
                "seed": seed,
                "router": "loss_mlp",
                "acc": loss_mlp_eval["acc"],
                "routed_loss": loss_mlp_eval["routed_loss"],
                "best_fixed_loss": loss_mlp_eval["best_fixed_loss"],
                "oracle_loss": loss_mlp_eval["oracle_loss"],
                "oracle_gap_closed_frac": loss_mlp_eval["oracle_gap_closed_frac"],
            }
        )

    per_task_df = pd.DataFrame(per_task_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)

    per_task_df.to_csv(out_dir / "router_per_task_by_seed.csv", index=False)
    aggregate_df.to_csv(out_dir / "router_aggregate_by_seed.csv", index=False)

    per_task_summary = (
        per_task_df.groupby(["router", "task"], as_index=False)
        .agg(
            best_fixed_loss_mean=("best_fixed_loss", "mean"),
            best_fixed_loss_std=("best_fixed_loss", "std"),
            routed_loss_mean=("routed_loss", "mean"),
            routed_loss_std=("routed_loss", "std"),
            oracle_loss_mean=("oracle_loss", "mean"),
            oracle_loss_std=("oracle_loss", "std"),
            oracle_gap_closed_frac_mean=("oracle_gap_closed_frac", "mean"),
            oracle_gap_closed_frac_std=("oracle_gap_closed_frac", "std"),
        )
        .sort_values(["router", "task"])
    )
    per_task_summary.to_csv(out_dir / "router_per_task_summary.csv", index=False)

    aggregate_summary = (
        aggregate_df.groupby(["router"], as_index=False)
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            routed_loss_mean=("routed_loss", "mean"),
            routed_loss_std=("routed_loss", "std"),
            best_fixed_loss_mean=("best_fixed_loss", "mean"),
            best_fixed_loss_std=("best_fixed_loss", "std"),
            oracle_loss_mean=("oracle_loss", "mean"),
            oracle_loss_std=("oracle_loss", "std"),
            oracle_gap_closed_frac_mean=("oracle_gap_closed_frac", "mean"),
            oracle_gap_closed_frac_std=("oracle_gap_closed_frac", "std"),
        )
        .sort_values("router")
    )
    aggregate_summary.to_csv(out_dir / "router_aggregate_summary.csv", index=False)

    print("\n[saved]")
    for path in sorted(out_dir.glob("router_*.csv")):
        print(f"  {path}")

    print("\n=== Router aggregate summary ===")
    print(aggregate_summary.to_string(index=False))

    print("\n=== Router per-task summary ===")
    print(per_task_summary.to_string(index=False))


if __name__ == "__main__":
    main()