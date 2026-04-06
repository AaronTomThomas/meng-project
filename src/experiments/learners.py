from experiments.synthetic_alignment.config import EvalConfig
import torch
import torch.nn.functional as F


@torch.no_grad()
def weighted_local_linear_predict(
    q: torch.Tensor,
    Kctx: torch.Tensor,
    Vctx: torch.Tensor,
    k_top: int,
    ridge_lambda: float,
    beta: float,
) -> torch.Tensor:
    """
    Weighted centred local linear regression.

    Steps:
      1. retrieve top-k neighbours using cosine similarity
      2. compute soft weights over those neighbours
      3. centre keys/values by weighted mean
      4. fit weighted ridge from centred keys -> centred values
      5. predict query value as local mean + local linear correction

    This is much stronger than the current local_linear baseline.
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]
    k_eff = min(k_top, n)

    # cosine retrieval
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

    # soft weights over retrieved neighbours
    w = F.softmax(beta * top_scores, dim=-1)  # (B,k)

    # weighted neighbourhood mean
    w_col = w[:, :, None]                     # (B,k,1)
    k_bar = (w_col * K_top).sum(dim=1)        # (B,d)
    v_bar = (w_col * V_top).sum(dim=1)        # (B,dv)

    # centred coordinates
    Kc = K_top - k_bar[:, None, :]            # (B,k,d)
    Vc = V_top - v_bar[:, None, :]            # (B,k,dv)
    qc = q - k_bar                            # (B,d)

    # weighted ridge
    # X_w = sqrt(w) * Kc, Y_w = sqrt(w) * Vc
    sqrt_w = torch.sqrt(w + 1e-8)[:, :, None]   # (B,k,1)
    Xw = Kc * sqrt_w                            # (B,k,d)
    Yw = Vc * sqrt_w                            # (B,k,dv)

    XT = Xw.transpose(1, 2)                     # (B,d,k)
    XTX = XT @ Xw                               # (B,d,d)
    XTY = XT @ Yw                               # (B,d,dv)

    I = torch.eye(d, device=Kctx.device, dtype=Kctx.dtype)[None, :, :]
    theta = torch.linalg.solve(XTX + ridge_lambda * I, XTY)  # (B,d,dv)

    # local affine prediction around neighbourhood mean
    yhat = v_bar + torch.einsum("bd,bdv->bv", qc, theta)
    return yhat

@torch.no_grad()
def soft_kernel_predict(
    q: torch.Tensor,      # (B,d)
    Kctx: torch.Tensor,   # (B,n,d)
    Vctx: torch.Tensor,   # (B,n,dv)
    beta: float,
) -> torch.Tensor:
    d = q.shape[-1]
    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(Kctx, dim=-1)
    scores = torch.einsum("bd,bnd->bn", qn, Kn)
    # scores = torch.einsum("bd,bnd->bn", q, Kctx) / math.sqrt(d)
    w = F.softmax(beta * scores, dim=-1)
    yhat = torch.einsum("bn,bnv->bv", w, Vctx)
    return yhat


@torch.no_grad()
def sharp_topk_predict(
    q: torch.Tensor,
    Kctx: torch.Tensor,
    Vctx: torch.Tensor,
    k_top: int,
) -> torch.Tensor:
    B, n, d = Kctx.shape
    k_eff = min(k_top, n)
    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(Kctx, dim=-1)
    scores = torch.einsum("bd,bnd->bn", qn, Kn)
    # scores = torch.einsum("bd,bnd->bn", q, Kctx) / math.sqrt(d)

    top_scores, top_idx = torch.topk(scores, k=k_eff, dim=-1)
    V_top = Vctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_eff, Vctx.shape[-1])
    )

    w = F.softmax(top_scores, dim=-1)
    yhat = torch.einsum("bk,bkv->bv", w, V_top)
    return yhat

@torch.no_grad()
def window_soft_predict(
    q: torch.Tensor,      # (B,d)
    Kctx: torch.Tensor,   # (B,n,d)
    Vctx: torch.Tensor,   # (B,n,dv)
    beta: float,
    window_size: int,
) -> torch.Tensor:
    """
    Softmax attention restricted to the last W context tokens.
    Standard locality-biased retrieval expert.
    """
    B, n, d = Kctx.shape
    w_eff = min(window_size, n)

    K_win = Kctx[:, -w_eff:, :]   # (B,w,d)
    V_win = Vctx[:, -w_eff:, :]   # (B,w,dv)

    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(K_win, dim=-1)
    scores = torch.einsum("bd,bwd->bw", qn, Kn)

    weights = F.softmax(beta * scores, dim=-1)
    yhat = torch.einsum("bw,bwv->bv", weights, V_win)
    return yhat

@torch.no_grad()
def knn_mean_predict(
    q: torch.Tensor,
    Kctx: torch.Tensor,
    Vctx: torch.Tensor,
    k_top: int,
) -> torch.Tensor:
    """
    Hard kNN retrieval with uniform averaging.
    No softmax weighting after retrieval.
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]
    k_eff = min(k_top, n)

    qn = F.normalize(q, dim=-1)
    Kn = F.normalize(Kctx, dim=-1)
    scores = torch.einsum("bd,bnd->bn", qn, Kn)

    _, top_idx = torch.topk(scores, k=k_eff, dim=-1)

    V_top = Vctx.gather(
        dim=1,
        index=top_idx[:, :, None].expand(B, k_eff, dv)
    )  # (B,k,dv)

    yhat = V_top.mean(dim=1)  # uniform average
    return yhat

@torch.no_grad()
def global_linear_ridge_predict(
    q: torch.Tensor,
    Kctx: torch.Tensor,
    Vctx: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """
    Global linear ridge on all past points.
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]

    ones = torch.ones(B, n, 1, device=Kctx.device, dtype=Kctx.dtype)
    X = torch.cat([Kctx, ones], dim=-1)  # (B,n,d+1)

    q_aug = torch.cat(
        [q, torch.ones(B, 1, device=q.device, dtype=q.dtype)],
        dim=-1
    )[:, None, :]  # (B,1,d+1)

    XT = X.transpose(1, 2)
    XTX = XT @ X
    XTY = XT @ Vctx

    I = torch.eye(d + 1, device=Kctx.device, dtype=Kctx.dtype)[None, :, :]
    theta = torch.linalg.solve(XTX + ridge_lambda * I, XTY)

    yhat = (q_aug @ theta).squeeze(1)
    return yhat

@torch.no_grad()
def linear_attention_predict(
    q: torch.Tensor,      # (B,d)
    Kctx: torch.Tensor,   # (B,n,d)
    Vctx: torch.Tensor,   # (B,n,dv)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Linear attention / kernel attention style predictor.

    Uses a positive feature map phi(x)=elu(x)+1 and computes:
        yhat = (phi(q)^T sum_j phi(k_j) v_j^T) / (phi(q)^T sum_j phi(k_j))
    """
    # positive feature map
    phi_q = F.elu(q) + 1.0              # (B,d)
    phi_K = F.elu(Kctx) + 1.0           # (B,n,d)

    # memory tensor: sum_j phi(k_j) outer v_j
    KV = torch.einsum("bnd,bnv->bdv", phi_K, Vctx)   # (B,d,dv)

    # normalizer state: sum_j phi(k_j)
    Z = phi_K.sum(dim=1)                                # (B,d)

    num = torch.einsum("bd,bdv->bv", phi_q, KV)         # (B,dv)
    den = torch.einsum("bd,bd->b", phi_q, Z).unsqueeze(-1)  # (B,1)

    yhat = num / (den + eps)
    return yhat


@torch.no_grad()
def predict_with_learner(
    learner_name: str,
    q: torch.Tensor,
    Kctx: torch.Tensor,
    Vctx: torch.Tensor,
    cfg: EvalConfig,
) -> torch.Tensor:
    if learner_name == "soft":
        return soft_kernel_predict(q, Kctx, Vctx, beta=cfg.beta_soft)
    elif learner_name == "sharp":
        return sharp_topk_predict(q, Kctx, Vctx, k_top=cfg.k_sharp)
    elif learner_name == "linear_global":
        return global_linear_ridge_predict(q, Kctx, Vctx, ridge_lambda=cfg.ridge_lambda)
    elif learner_name == "linear_attention":
        return linear_attention_predict(q, Kctx, Vctx)
    elif learner_name == "weighted_linear":
        return weighted_local_linear_predict(
            q, Kctx, Vctx,
            k_top=cfg.k_linear_local,
            ridge_lambda=cfg.ridge_lambda,
            beta=cfg.beta_soft,
        )
    elif learner_name == "knn_mean":
        return knn_mean_predict(q, Kctx, Vctx, k_top=cfg.k_knn_mean)
    elif learner_name == "window_soft":
        return window_soft_predict(
            q, Kctx, Vctx,
            beta=cfg.beta_soft,
            window_size=cfg.window_size,
        )
    else:
        raise ValueError(f"Unknown learner: {learner_name}")
    
LEARNERS = ["soft", "sharp", "linear_global", "window_soft", "knn_mean", "linear_attention", "weighted_linear"]
ROUTER_LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]
