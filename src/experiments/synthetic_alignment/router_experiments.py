"""Training/evaluation helpers for router models on synthetic alignment tasks."""

from typing import Dict, List

import torch

from experiments.synthetic_alignment.config import RouterExperimentConfig
from experiments.synthetic_alignment.router_dataset import RouterDatasetBuilder
from experiments.experiment_routers import (
    evaluate_logistic_router,
    evaluate_loss_router,
    train_logistic_router,
    train_loss_mlp_router,
)

ROUTER_MODE_IMPL = {
    "logistic": {
        "train": lambda Xt, yt, Xv, yv: train_logistic_router(Xt, yt, Xv, yv),
        "eval": evaluate_logistic_router,
        "train_label_key": "y",
    },
    "loss_mlp": {
        "train": lambda Xt, Lt, Xv, Lv: train_loss_mlp_router(Xt, Lt, Xv, Lv),
        "eval": evaluate_loss_router,
        "train_label_key": "losses",
    },
}


def evaluate_router_per_task(
    pred: torch.Tensor,
    losses: torch.Tensor,
    task_ids: torch.Tensor,
    task_to_idx: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Compute routed/oracle/best-fixed losses per task."""
    if pred.device != losses.device:
        pred = pred.to(losses.device)
    if task_ids.device != losses.device:
        task_ids = task_ids.to(losses.device)

    idx_to_task = {v: k for k, v in task_to_idx.items()}
    out = {}

    for idx in sorted(idx_to_task.keys()):
        mask = task_ids == idx
        if not mask.any():
            continue

        losses_t = losses[mask]
        pred_t = pred[mask]
        routed_loss = losses_t[
            torch.arange(losses_t.shape[0], device=losses.device),
            pred_t
        ].mean().item()

        oracle_loss = losses_t.min(dim=-1).values.mean().item()
        best_fixed_loss = losses_t.mean(dim=0).min().item()
        gap_closed = (best_fixed_loss - routed_loss) / max(best_fixed_loss - oracle_loss, 1e-8)

        out[idx_to_task[idx]] = {
            "best_fixed_loss": best_fixed_loss,
            "routed_loss": routed_loss,
            "oracle_loss": oracle_loss,
            "oracle_gap_closed_frac": gap_closed,
        }

    return out


def evaluate_per_task_best_fixed_baseline(
    losses_train: torch.Tensor,
    task_ids_train: torch.Tensor,
    losses_test: torch.Tensor,
    task_ids_test: torch.Tensor,
    task_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Select each task's best fixed learner on train and apply to test queries."""
    device = losses_train.device
    idx_to_task = {v: k for k, v in task_to_idx.items()}
    chosen = {}

    for idx in sorted(idx_to_task.keys()):
        mask = task_ids_train == idx
        if not mask.any():
            continue
        best_idx = losses_train[mask].mean(dim=0).argmin().item()
        chosen[idx] = best_idx

    pred = torch.empty(losses_test.shape[0], dtype=torch.long, device=device)
    for idx in sorted(idx_to_task.keys()):
        if idx not in chosen:
            continue
        mask = task_ids_test == idx
        if mask.any():
            pred[mask] = chosen[idx]
    return pred


def _set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_eval_payload(
    eval_out: Dict[str, object],
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    metrics = {k: v for k, v in eval_out.items() if k not in ("pred", "pred_regrets")}
    pred = eval_out.get("pred")
    extras: Dict[str, torch.Tensor] = {}
    if "pred_regrets" in eval_out:
        extras["pred_regrets"] = eval_out["pred_regrets"]
    return pred, metrics, extras


def _compute_regrets(losses: torch.Tensor) -> torch.Tensor:
    oracle = losses.min(dim=-1, keepdim=True).values
    return losses - oracle


def _normalize_taskwise_regrets(
    regrets_train: torch.Tensor,
    task_ids_train: torch.Tensor,
    regrets_test: torch.Tensor,
    task_ids_test: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    regrets_train_norm = torch.zeros_like(regrets_train)
    regrets_test_norm = torch.zeros_like(regrets_test)
    unique_train = torch.unique(task_ids_train)
    task_scale: Dict[str, float] = {}

    for t in unique_train.tolist():
        train_mask = task_ids_train == t
        test_mask = task_ids_test == t
        scale = regrets_train[train_mask].mean().item()
        scale = max(scale, eps)
        regrets_train_norm[train_mask] = regrets_train[train_mask] / scale
        if test_mask.any():
            regrets_test_norm[test_mask] = regrets_test[test_mask] / scale
        task_scale[str(t)] = scale

    unique_test = torch.unique(task_ids_test).tolist()
    train_ids_set = set(unique_train.tolist())
    for t in unique_test:
        if t in train_ids_set:
            continue
        test_mask = task_ids_test == t
        if not test_mask.any():
            continue
        scale = regrets_test[test_mask].mean().item()
        scale = max(scale, eps)
        regrets_test_norm[test_mask] = regrets_test[test_mask] / scale
        task_scale[str(t)] = scale

    return regrets_train_norm, regrets_test_norm, task_scale


def run_router_experiment(
    cfg: RouterExperimentConfig,
    router_mode: str,
    task_names: List[str],
    learners: List[str],
    n_batches_per_task: int,
    seed: int = 0,
    train_frac: float = 0.8,
) -> Dict[str, object]:
    """Train a router (logistic or loss-MLP) and report aggregate metrics."""

    print(f"Collecting data for {router_mode} router evaluation")
    _set_random_seed(seed)

    builder = RouterDatasetBuilder(cfg=cfg, learners=learners)
    dataset = builder.build(
        task_names=task_names,
        n_batches_per_task=n_batches_per_task,
    )
    train_ds, test_ds = dataset.split(train_frac=train_frac, seed=seed)
    X_train, X_test, _, _ = train_ds.standardize_with(test_ds)

    if router_mode not in ROUTER_MODE_IMPL:
        raise ValueError(f"Unknown router_mode '{router_mode}'.")

    impl = ROUTER_MODE_IMPL[router_mode]
    extra_stats: Dict[str, object] = {}

    if impl["train_label_key"] == "y":
        train_labels = train_ds.y 
        test_labels = test_ds.y
    else:
        train_regrets = _compute_regrets(train_ds.losses)
        test_regrets = _compute_regrets(test_ds.losses)
        train_labels, test_labels, scale_stats = _normalize_taskwise_regrets(
            regrets_train=train_regrets,
            task_ids_train=train_ds.task_ids,
            regrets_test=test_regrets,
            task_ids_test=test_ds.task_ids,
        )
        extra_stats["regret_scale"] = scale_stats


    print(f"Training {router_mode} router...")
    model = impl["train"](X_train, train_labels, X_test, test_labels)
    train_eval = impl["eval"](model, X_train, train_ds.y, train_ds.losses)
    test_eval = impl["eval"](model, X_test, test_ds.y, test_ds.losses)

    train_pred, train_metrics, train_extras = _extract_eval_payload(train_eval)
    test_pred, test_metrics, test_extras = _extract_eval_payload(test_eval)

    task_metrics = evaluate_router_per_task(
        pred=test_pred,
        losses=test_ds.losses,
        task_ids=test_ds.task_ids,
        task_to_idx=test_ds.task_to_idx,
    )

    return {
        "router_mode": router_mode,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "task_metrics": task_metrics,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
        "prediction_extras": {
            "train": train_extras,
            "test": test_extras,
        },
        "dataset_summary": {
            "n_total": dataset.X.shape[0],
            "n_features": dataset.X.shape[1],
            "n_learners": dataset.losses.shape[1],
            "n_train": train_ds.X.shape[0],
            "n_test": test_ds.X.shape[0],
            "task_to_idx": dataset.task_to_idx,
        },
        "extra_stats": extra_stats,
    }


def report_router_result(title: str, result: Dict[str, object]) -> None:
    print(f"\n=== Router ({title}) test metrics ===")
    metrics = result["test_metrics"]
    for name, value in metrics.items():
        print(f"{name:>24}: {value:.6f}")

    print("Per-task oracle gap closure (%):")
    for task_name in sorted(result["task_metrics"].keys()):
        stats = result["task_metrics"][task_name]
        gap = stats.get("oracle_gap_closed_frac", 0.0) * 100
        print(
            f"  {task_name:<24} best_fixed={stats['best_fixed_loss']:.6f} "
            f"routed={stats['routed_loss']:.6f} oracle={stats['oracle_loss']:.6f} "
            f"gap_closed={gap:5.1f}"
        )

if __name__ == "__main__":

    ROUTER_LEARNERS: List[str] = ["soft", "sharp", "window_soft", "weighted_linear"]


    cfg = RouterExperimentConfig(
        L=128,
        d=32,
        dv=16,
        batch_size=128,
        sigma=0.05,
        beta_soft=6.0,
        k_sharp=4,
        k_linear_local=16,
        ridge_lambda=1e-1,
        min_context=8,
    )

    router_out_logistic = run_router_experiment(
        cfg,
        router_mode="logistic",
        task_names=["piecewise_linear", "prototype_lookup", "smooth_nonlinear_local", "shifted_local_map"],
        learners=ROUTER_LEARNERS,
        n_batches_per_task=8,
        seed=0,
    )

    router_out_loss_mlp = run_router_experiment(
        cfg,
        router_mode="loss_mlp",
        task_names=["piecewise_linear", "prototype_lookup", "smooth_nonlinear_local", "shifted_local_map"],
        learners=ROUTER_LEARNERS,
        n_batches_per_task=8,
        seed=0,
    )

    report_router_result("logistic", router_out_logistic)
    report_router_result("loss-aware mlp", router_out_loss_mlp)
