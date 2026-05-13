
# ============================================================
# ROUTER MODELS
# ============================================================

from typing import Callable, Dict

import torch
import torch.nn.functional as F

class LogisticRouter(torch.nn.Module):
    """
    Predicts winner logits directly.
    Used for winner-class classification.
    """
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LossMLPRouter(torch.nn.Module):
    """
    Predicts per-learner losses.
    Used for loss-aware routing.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# ============================================================
# TRAINING HELPERS
# ============================================================

def train_router_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_step: Callable[[int], float],
    eval_metric: Callable[[], float],
    epochs: int,
    maximize_metric: bool,
    metric_name: str,
    log_every: int | None = None,
) -> None:

    best_metric = -float("inf") if maximize_metric else float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        train_loss = train_step(epoch)
        metric_val = eval_metric()

        if log_every and epoch % log_every == 0:
            direction = "max" if maximize_metric else "min"
            print(
                f"Epoch {epoch:05d} | train loss: {train_loss:.4f} | "
                f"{metric_name} ({direction}): {metric_val:.4f}"
            )

        improved = (
            metric_val > best_metric if maximize_metric else metric_val < best_metric
        )
        if improved:
            best_metric = metric_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)



# ============================================================
# ROUTER TRAINING
# ============================================================

def train_logistic_router(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 4000,
) -> LogisticRouter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    model = LogisticRouter(X_train.shape[1], int(y_train.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(_: int) -> float:
        model.train()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def eval_metric() -> float:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(dim=-1)
            return (val_pred == y_val).float().mean().item()

    train_router_model(
        model=model,
        optimizer=opt,
        train_step=train_step,
        eval_metric=eval_metric,
        epochs=epochs,
        maximize_metric=True,
        metric_name="val_acc",
    )
    return model


def train_loss_mlp_router(
    X_train: torch.Tensor,
    losses_train: torch.Tensor,
    X_val: torch.Tensor,
    losses_val: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 4000,
    hidden_dim: int = 64,
) -> LossMLPRouter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.to(device)
    losses_train = losses_train.to(device)
    X_val = X_val.to(device)
    losses_val = losses_val.to(device)

    model = LossMLPRouter(
        in_dim=X_train.shape[1],
        out_dim=losses_train.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(epoch: int) -> float:
        model.train()
        pred_losses = model(X_train)
        loss = F.mse_loss(pred_losses, losses_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def eval_metric() -> float:
        model.eval()
        with torch.no_grad():
            val_pred_losses = model(X_val)
            val_pred_idx = val_pred_losses.argmin(dim=-1)
            return losses_val[
                torch.arange(losses_val.shape[0], device=device),
                val_pred_idx,
            ].mean().item()

    train_router_model(
        model=model,
        optimizer=opt,
        train_step=train_step,
        eval_metric=eval_metric,
        epochs=epochs,
        maximize_metric=False,
        metric_name="val_routed_loss",
        log_every=500,
    )
    return model


# ============================================================
# ROUTER EVALUATION
# ============================================================

def compute_router_metrics(
    pred: torch.Tensor,
    y: torch.Tensor,
    losses: torch.Tensor,
) -> Dict[str, float]:
    device = losses.device
    pred_device = pred.to(device)
    y_device = y.to(device)

    acc = (pred_device == y_device).float().mean().item()
    routed_loss = losses[
        torch.arange(losses.shape[0], device=device),
        pred_device,
    ].mean().item()

    oracle_loss = losses.min(dim=-1).values.mean().item()
    best_fixed_loss = losses.mean(dim=0).min().item()
    gap_closed = (best_fixed_loss - routed_loss) / max(best_fixed_loss - oracle_loss, 1e-8)

    return {
        "acc": acc,
        "routed_loss": routed_loss,
        "best_fixed_loss": best_fixed_loss,
        "oracle_loss": oracle_loss,
        "oracle_gap_closed_frac": gap_closed,
    }

@torch.no_grad()
def evaluate_logistic_router(
    model: LogisticRouter,
    X: torch.Tensor,
    y: torch.Tensor,
    losses: torch.Tensor,
):
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)
    losses = losses.to(device)

    logits = model(X)
    pred = logits.argmax(dim=-1)
    metrics = compute_router_metrics(pred, y, losses)
    metrics["pred"] = pred.cpu()
    return metrics


@torch.no_grad()
def evaluate_loss_router(
    model: LossMLPRouter,
    X: torch.Tensor,
    y: torch.Tensor,
    losses: torch.Tensor,
):
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)
    losses = losses.to(device)

    pred_regrets = model(X)
    pred = pred_regrets.argmin(dim=-1)
    metrics = compute_router_metrics(pred, y, losses)
    metrics["pred"] = pred.cpu()
    metrics["pred_regrets"] = pred_regrets.cpu()
    return metrics
