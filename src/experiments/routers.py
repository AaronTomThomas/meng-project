
# ============================================================
# ROUTER MODELS
# ============================================================

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
# 3.8) ROUTER TRAINING
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    model = LogisticRouter(X_train.shape[1], int(y_train.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = -1.0

    for _ in range(epochs):
        model.train()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(dim=-1)
            val_acc = (val_pred == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def train_loss_mlp_router(
    X_train: torch.Tensor,
    losses_train: torch.Tensor,
    X_val: torch.Tensor,
    losses_val: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 8000,
    hidden_dim: int = 64,
) -> LossMLPRouter:
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    best_state = None
    best_val_routed_loss = float("inf")

    for i in range(epochs):
        model.train()
        pred_losses = model(X_train)
        loss = F.mse_loss(pred_losses, losses_train)
        if (i) % 500 == 0:
            print(f"Epoch {i}, train MSE loss: {loss.item():.4f}")
            
            print("Best val routed loss so far: {:.4f}".format(best_val_routed_loss))

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred_losses = model(X_val)
            val_pred_idx = val_pred_losses.argmin(dim=-1)
            val_routed_loss = losses_val[
                torch.arange(losses_val.shape[0], device=device),
                val_pred_idx
            ].mean().item()

        if val_routed_loss < best_val_routed_loss:
            best_val_routed_loss = val_routed_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
            

    model.load_state_dict(best_state)
    return model


# ============================================================
# ROUTER EVALUATION
# ============================================================

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

    acc = (pred == y).float().mean().item()
    routed_loss = losses[torch.arange(losses.shape[0], device=device), pred].mean().item()

    oracle_loss = losses.min(dim=-1).values.mean().item()
    best_fixed_loss = losses.mean(dim=0).min().item()

    gap_closed = (best_fixed_loss - routed_loss) / max(best_fixed_loss - oracle_loss, 1e-8)

    return {
        "acc": acc,
        "routed_loss": routed_loss,
        "best_fixed_loss": best_fixed_loss,
        "oracle_loss": oracle_loss,
        "oracle_gap_closed_frac": gap_closed,
        "pred": pred.cpu(),
    }


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

    pred_losses = model(X)
    pred = pred_losses.argmin(dim=-1)

    acc = (pred == y).float().mean().item()
    routed_loss = losses[torch.arange(losses.shape[0], device=device), pred].mean().item()

    oracle_loss = losses.min(dim=-1).values.mean().item()
    best_fixed_loss = losses.mean(dim=0).min().item()

    gap_closed = (best_fixed_loss - routed_loss) / max(best_fixed_loss - oracle_loss, 1e-8)

    return {
        "acc": acc,
        "routed_loss": routed_loss,
        "best_fixed_loss": best_fixed_loss,
        "oracle_loss": oracle_loss,
        "oracle_gap_closed_frac": gap_closed,
        "pred": pred.cpu(),
        "pred_losses": pred_losses.cpu(),
    }

@torch.no_grad()
def evaluate_router_per_task(
    pred: torch.Tensor,              # (N,)
    losses: torch.Tensor,            # (N,M)
    task_ids: torch.Tensor,          # (N,)
    task_to_idx: dict,
):
    """
    Evaluate routed loss / oracle gap closed per task.
    """
    if pred.device != losses.device:
        pred = pred.to(losses.device)
    if task_ids.device != losses.device:
        task_ids = task_ids.to(losses.device)

    idx_to_task = {v: k for k, v in task_to_idx.items()}
    out = {}

    for t in sorted(idx_to_task.keys()):
        mask = (task_ids == t)
        if mask.sum().item() == 0:
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

        out[idx_to_task[t]] = {
            "best_fixed_loss": best_fixed_loss,
            "routed_loss": routed_loss,
            "oracle_loss": oracle_loss,
            "oracle_gap_closed_frac": gap_closed,
        }

    return out

@torch.no_grad()
def evaluate_per_task_best_fixed_baseline(
    losses_train: torch.Tensor,
    task_ids_train: torch.Tensor,
    losses_test: torch.Tensor,
    task_ids_test: torch.Tensor,
    task_to_idx: dict,
):
    """
    Choose, for each task, the learner with best average train loss.
    Evaluate on test.
    """
    idx_to_task = {v: k for k, v in task_to_idx.items()}
    chosen = {}

    for t in sorted(idx_to_task.keys()):
        mask = (task_ids_train == t)
        best_idx = losses_train[mask].mean(dim=0).argmin().item()
        chosen[t] = best_idx

    pred = torch.empty(losses_test.shape[0], dtype=torch.long)
    for t in sorted(idx_to_task.keys()):
        mask = (task_ids_test == t)
        pred[mask.cpu()] = chosen[t]

    return pred