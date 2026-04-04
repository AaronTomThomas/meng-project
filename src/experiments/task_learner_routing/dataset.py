


from typing import Dict, List

from experiments.task_learner_routing.config import EvalConfig
from experiments.task_learner_routing.learners import predict_with_learner, sharp_topk_predict, soft_kernel_predict, window_soft_predict
from experiments.task_learner_routing.tasks import TASK_FNS
import torch
import torch.nn.functional as F



@torch.no_grad()
def _topk_cosine_stats(
    q: torch.Tensor,         # (B,d)
    Kctx: torch.Tensor,      # (B,n,d)
    k_top: int,
    beta_soft: float,
):
    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(Kctx, dim=-1)
    sims = torch.einsum("bd,bnd->bn", qn, Kn)  # cosine sims

    k_eff = min(k_top, Kctx.shape[1])
    top_vals, top_idx = torch.topk(sims, k=k_eff, dim=-1)

    soft_w = F.softmax(beta_soft * sims, dim=-1)  # over all context
    top_soft_w = F.softmax(beta_soft * top_vals, dim=-1)

    return sims, top_vals, top_idx, soft_w, top_soft_w


@torch.no_grad()
def _weighted_local_fit_residual(
    q: torch.Tensor,         # (B,d)
    Kctx: torch.Tensor,      # (B,n,d)
    Vctx: torch.Tensor,      # (B,n,dv)
    k_top: int,
    ridge_lambda: float,
    beta: float,
):
    """
    Returns weighted local fit residual over retrieved neighbours:
      residual = weighted MSE of fitted centred local linear model on the neighbour set.
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]
    k_eff = min(k_top, n)

    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(Kctx, dim=-1)
    scores = torch.einsum("bd,bnd->bn", qn, Kn)

    top_scores, top_idx = torch.topk(scores, k=k_eff, dim=-1)

    K_top = Kctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_eff, d)
    )
    V_top = Vctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_eff, dv)
    )

    w = F.softmax(beta * top_scores, dim=-1)             # (B,k)
    w_col = w[:, :, None]

    k_bar = (w_col * K_top).sum(dim=1)                   # (B,d)
    v_bar = (w_col * V_top).sum(dim=1)                   # (B,dv)

    Kc = K_top - k_bar[:, None, :]                       # (B,k,d)
    Vc = V_top - v_bar[:, None, :]                       # (B,k,dv)

    sqrt_w = torch.sqrt(w + 1e-8)[:, :, None]
    Xw = Kc * sqrt_w
    Yw = Vc * sqrt_w

    XT = Xw.transpose(1, 2)
    XTX = XT @ Xw
    XTY = XT @ Yw

    I = torch.eye(d, device=Kctx.device, dtype=Kctx.dtype)[None, :, :]
    theta = torch.linalg.solve(XTX + ridge_lambda * I, XTY)  # (B,d,dv)

    Vc_hat = torch.einsum("bkd,bdv->bkv", Kc, theta)
    resid = ((Vc_hat - Vc) ** 2).mean(dim=-1)            # (B,k)
    weighted_resid = (w * resid).sum(dim=-1)             # (B,)

    return weighted_resid

@torch.no_grad()
def _weighted_local_stats(
    q: torch.Tensor,         # (B,d)
    Kctx: torch.Tensor,      # (B,n,d)
    Vctx: torch.Tensor,      # (B,n,dv)
    k_top: int,
    ridge_lambda: float,
    beta: float,
):
    """
    Returns local weighted stats used both for prediction and router features.

    Outputs:
      yhat          : weighted local linear prediction          (B,dv)
      k_bar         : weighted local key mean                  (B,d)
      v_bar         : weighted local value mean                (B,dv)
      qc_norm       : ||q - k_bar||                            (B,)
      corr_norm     : ||yhat - v_bar||                         (B,)
      fit_resid     : weighted local fit residual              (B,)
      top_scores    : retrieved cosine scores                  (B,k)
      top_weights   : soft weights over retrieved neighbours   (B,k)
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]
    k_eff = min(k_top, n)

    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(Kctx, dim=-1)
    scores = torch.einsum("bd,bnd->bn", qn, Kn)

    top_scores, top_idx = torch.topk(scores, k=k_eff, dim=-1)

    K_top = Kctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_eff, d)
    )  # (B,k,d)

    V_top = Vctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_eff, dv)
    )  # (B,k,dv)

    w = F.softmax(beta * top_scores, dim=-1)                  # (B,k)
    w_col = w[:, :, None]

    k_bar = (w_col * K_top).sum(dim=1)                        # (B,d)
    v_bar = (w_col * V_top).sum(dim=1)                        # (B,dv)

    Kc = K_top - k_bar[:, None, :]                            # (B,k,d)
    Vc = V_top - v_bar[:, None, :]                            # (B,k,dv)
    qc = q - k_bar                                            # (B,d)

    sqrt_w = torch.sqrt(w + 1e-8)[:, :, None]
    Xw = Kc * sqrt_w
    Yw = Vc * sqrt_w

    XT = Xw.transpose(1, 2)
    XTX = XT @ Xw
    XTY = XT @ Yw

    I = torch.eye(d, device=Kctx.device, dtype=Kctx.dtype)[None, :, :]
    theta = torch.linalg.solve(XTX + ridge_lambda * I, XTY)   # (B,d,dv)

    yhat = v_bar + torch.einsum("bd,bdv->bv", qc, theta)      # (B,dv)

    Vc_hat = torch.einsum("bkd,bdv->bkv", Kc, theta)
    resid = ((Vc_hat - Vc) ** 2).mean(dim=-1)                 # (B,k)
    fit_resid = (w * resid).sum(dim=-1)                       # (B,)

    qc_norm = qc.norm(dim=-1)                                 # (B,)
    corr_norm = (yhat - v_bar).norm(dim=-1)                   # (B,)

    return {
        "yhat": yhat,
        "k_bar": k_bar,
        "v_bar": v_bar,
        "qc_norm": qc_norm,
        "corr_norm": corr_norm,
        "fit_resid": fit_resid,
        "top_scores": top_scores,
        "top_weights": w,
    }

@torch.no_grad()
def _topk_recency_stats(
    top_idx: torch.Tensor,   # (B,k)
    n_ctx: int,
    window_size: int,
):
    """
    Returns recency stats for retrieved neighbours.
    Larger lag = older neighbour.
    """
    device = top_idx.device
    B, k = top_idx.shape

    # current query index is n_ctx, retrieved positions are in [0, n_ctx-1]
    lags = (n_ctx - 1 - top_idx).to(torch.float32)  # (B,k)

    mean_lag = lags.mean(dim=-1)
    std_lag = lags.std(dim=-1, unbiased=False)

    recent_frac = (lags < window_size).float().mean(dim=-1)

    return mean_lag, std_lag, recent_frac


@torch.no_grad()
def compute_router_features(
    q: torch.Tensor,         # (B,d)
    Kctx: torch.Tensor,      # (B,n,d)
    Vctx: torch.Tensor,      # (B,n,dv)
    cfg: EvalConfig,
) -> torch.Tensor:
    """
    Per-query handcrafted features.

    Old features:
      0  max cosine similarity
      1  top1-top2 similarity gap
      2  mean top-k similarity
      3  std top-k similarity
      4  soft entropy over all context
      5  effective support size 1/sum(w^2)
      6  nearest-neighbour distance = 1 - max_sim
      7  weighted variance of retrieved values
      8  local covariance trace
      9  local covariance condition surrogate log(lambda_max/lambda_min)
      10 weighted local fit residual

    New features:
      11 ||y_soft - y_window_soft||
      12 ||y_soft - y_sharp||
      13 ||y_soft - y_weighted_linear||
      14 ||y_sharp - y_weighted_linear||
      15 ||q - k_bar||                         (query shift from local support centre)
      16 ||y_wlin - v_bar||                    (local correction magnitude)
      17 fit_resid / (v_var + eps)             (fit-vs-copy ratio)
      18 window_soft entropy
      19 sharp top-k entropy
      20 sharp top-1 mass
      21 mean window similarity
      22 ||y_window_soft - y_weighted_linear||
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]

    # ---------- base retrieval stats ----------
    k_feat = min(max(cfg.k_linear_local, cfg.k_sharp, cfg.k_knn_mean, 4), n)

    sims, top_vals, top_idx, soft_w, top_soft_w = _topk_cosine_stats(
        q, Kctx, k_feat, cfg.beta_soft
    )

    K_top = Kctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_feat, d)
    )
    V_top = Vctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_feat, dv)
    )

    # 0 max cosine sim
    max_sim = top_vals[:, 0]

    # 1 top1-top2 gap
    if k_feat >= 2:
        top_gap = top_vals[:, 0] - top_vals[:, 1]
    else:
        top_gap = torch.zeros_like(max_sim)

    # 2,3 mean/std top-k similarity
    mean_topk_sim = top_vals.mean(dim=-1)
    std_topk_sim = top_vals.std(dim=-1, unbiased=False)

    # 4 soft entropy over all context
    soft_entropy = -(soft_w * soft_w.clamp_min(1e-8).log()).sum(dim=-1)

    # 5 effective support size
    eff_support = 1.0 / (soft_w.pow(2).sum(dim=-1) + 1e-8)

    # 6 nn distance
    nn_dist = 1.0 - max_sim

    # 7 weighted variance of retrieved values
    v_bar_top = (top_soft_w[:, :, None] * V_top).sum(dim=1)                     # (B,dv)
    v_var = (top_soft_w[:, :, None] * (V_top - v_bar_top[:, None, :]).pow(2)).sum(dim=(1, 2))

    # 8,9 local covariance stats
    k_bar_top = (top_soft_w[:, :, None] * K_top).sum(dim=1)                     # (B,d)
    Kc_top = K_top - k_bar_top[:, None, :]
    sqrt_w_top = torch.sqrt(top_soft_w + 1e-8)[:, :, None]
    Xw_top = Kc_top * sqrt_w_top
    cov = torch.einsum("bkd,bke->bde", Xw_top, Xw_top) / max(k_feat, 1)

    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-8)
    cov_trace = eigvals.sum(dim=-1)
    cov_log_cond = torch.log(eigvals[:, -1] / eigvals[:, 0])

    # legal query shift from top-k weighted centre
    qc_top = q - k_bar_top
    qc_top_norm = qc_top.norm(dim=-1)

    # richer value-dispersion stats
    v_dev = (V_top - v_bar_top[:, None, :]).norm(dim=-1)   # (B,k)
    v_std = torch.sqrt(v_var + 1e-8)
    v_max_dev = v_dev.max(dim=-1).values

    # anisotropy / spectral shape
    eig_ratio_12 = eigvals[:, -1] / eigvals[:, -2].clamp_min(1e-8)
    p = eigvals / eigvals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    effective_rank = torch.exp(-(p * p.clamp_min(1e-8).log()).sum(dim=-1))

    # recency stats of top-k retrieved neighbours
    mean_topk_lag, std_topk_lag, recent_frac_topk = _topk_recency_stats(
        top_idx=top_idx,
        n_ctx=n,
        window_size=cfg.window_size,
    )
    
    # ---------- window-soft specific features ----------
    w_eff = min(cfg.window_size, n)
    K_win = Kctx[:, -w_eff:, :]
    qn = F.normalize(q, dim=-1)
    Kn_win = F.normalize(K_win, dim=-1)
    win_scores = torch.einsum("bd,bwd->bw", qn, Kn_win)
    win_weights = F.softmax(cfg.beta_soft * win_scores, dim=-1)

    # 18 window-soft entropy
    window_entropy = -(win_weights * win_weights.clamp_min(1e-8).log()).sum(dim=-1)

    # 21 mean window similarity
    mean_window_sim = win_scores.mean(dim=-1)

    # ---------- sharp-specific concentration ----------
    sharp_top_vals, _ = torch.topk(sims, k=min(cfg.k_sharp, n), dim=-1)
    sharp_weights = F.softmax(sharp_top_vals, dim=-1)

    # 19 sharp top-k entropy
    sharp_entropy = -(sharp_weights * sharp_weights.clamp_min(1e-8).log()).sum(dim=-1)

    # 20 sharp top-1 mass
    sharp_top1_mass = sharp_weights[:, 0]

    feats = torch.stack(
        [
            max_sim,            # 0
            top_gap,            # 1
            mean_topk_sim,      # 2
            std_topk_sim,       # 3
            soft_entropy,       # 4
            eff_support,        # 5
            nn_dist,            # 6
            v_var,              # 7
            cov_trace,          # 8
            cov_log_cond,       # 9
            window_entropy,     # 10
            sharp_entropy,      # 11
            sharp_top1_mass,    # 12
            mean_window_sim,    # 13
            qc_top_norm,        # 14
            mean_topk_lag,      # 15
            std_topk_lag,       # 16
            recent_frac_topk,   # 17
            v_std,              # 18
            v_max_dev,          # 19
            eig_ratio_12,       # 20
            effective_rank,     # 21
        ],
        dim=-1
    )
    return feats


@torch.no_grad()
def collect_router_dataset(
    cfg: EvalConfig,
    task_names: List[str],
    learners: List[str],
    n_batches_per_task: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Build a dataset of:
      X: per-query features
      y: oracle winner label over `learners`
      losses: per-learner losses for each query
      task_id: integer task id
    """
    X_all = []
    y_all = []
    losses_all = []
    task_id_all = []

    task_to_idx = {name: i for i, name in enumerate(task_names)}

    for task_name in task_names:
        for _ in range(n_batches_per_task):
            K, V = TASK_FNS[task_name](cfg)
            B, L, _ = K.shape

            for i in range(cfg.min_context, L):
                q = K[:, i, :]
                Kctx = K[:, :i, :]
                Vctx = V[:, :i, :]
                target = V[:, i, :]

                feats = compute_router_features(q, Kctx, Vctx, cfg)  # (B,F)

                per_learner_losses = []
                for learner in learners:
                    yhat = predict_with_learner(learner, q, Kctx, Vctx, cfg)
                    mse = ((yhat - target) ** 2).mean(dim=-1)         # (B,)
                    per_learner_losses.append(mse)

                per_learner_losses = torch.stack(per_learner_losses, dim=-1)  # (B,M)
                labels = per_learner_losses.argmin(dim=-1)                     # (B,)

                X_all.append(feats.cpu())
                y_all.append(labels.cpu())
                losses_all.append(per_learner_losses.cpu())

                task_ids = torch.full((B,), task_to_idx[task_name], dtype=torch.long)
                task_id_all.append(task_ids)

    X = torch.cat(X_all, dim=0)               # (N,F)
    y = torch.cat(y_all, dim=0)               # (N,)
    losses = torch.cat(losses_all, dim=0)     # (N,M)
    task_ids = torch.cat(task_id_all, dim=0)  # (N,)

    return {
        "X": X,
        "y": y,
        "losses": losses,
        "task_ids": task_ids,
        "task_to_idx": task_to_idx,
    }



def standardize_train_test(X_train: torch.Tensor, X_test: torch.Tensor):
    mu = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (X_train - mu) / std, (X_test - mu) / std, mu, std


def train_test_split_dataset(ds: Dict[str, torch.Tensor], train_frac: float = 0.8, seed: int = 0):
    N = ds["X"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(train_frac * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    out = {}
    for k in ["X", "y", "losses", "task_ids"]:
        out[f"{k}_train"] = ds[k][train_idx]
        out[f"{k}_test"] = ds[k][test_idx]
    out["task_to_idx"] = ds["task_to_idx"]
    return out