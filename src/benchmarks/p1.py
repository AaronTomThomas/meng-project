import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# 1) CONFIG
# ============================================================

@dataclass
class EvalConfig:
    L: int = 128
    d: int = 32
    dv: int = 16
    batch_size: int = 128
    sigma: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # learner hyperparams
    beta_soft: float = 6.0
    k_sharp: int = 4
    k_linear_local: int = 16
    ridge_lambda: float = 1e-1
    min_context: int = 8
    retention_decay: float = 0.9
    window_size: int = 16
    k_knn_mean: int = 4

def _randn(*shape, cfg: EvalConfig):
    return torch.randn(*shape, device=cfg.device, dtype=cfg.dtype)


# ============================================================
# 2) TASK FAMILIES
# ============================================================

@torch.no_grad()
def generate_support_query_shifted_local_map(
    cfg: EvalConfig,
    segment_len: int = 16,
    regime_dims: int = 8,
    x_dim: int = 8,
    support_scale: float = 1.0,
    query_shift: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Anti-copy local-map task.

    Each segment c has its own linear map A_c.
    Within a segment:
      - first S-1 tokens are support inputs x ~ N(0, support_scale^2 I)
      - last token is a query input x_q ~ N(mu, I) with a shifted mean
      - values are v = A_c x + noise

    Key idea:
      - support outputs are NOT directly reusable for the query
      - nearest-neighbour averaging should be biased
      - fitting a local linear map should help

    Returns:
      K: (B,L,d)
      V: (B,L,dv)
      query_mask: (B,L) boolean, True where evaluation should happen
    """
    assert cfg.L % segment_len == 0
    C = cfg.L // segment_len
    B, L, d, dv = cfg.batch_size, cfg.L, cfg.d, cfg.dv
    device, dtype = cfg.device, cfg.dtype

    if x_dim + regime_dims > d:
        raise ValueError("Need x_dim + regime_dims <= d")

    # segment ids
    seg_id = torch.arange(L, device=device) // segment_len
    seg_id = seg_id[None, :].expand(B, -1).contiguous()  # (B,L)

    # unique regime sign patterns
    if C > (1 << regime_dims):
        raise ValueError(f"C={C} must be <= 2^regime_dims={1<<regime_dims}")

    idx = torch.randperm(1 << regime_dims, device=device)[:C]
    bits = ((idx[:, None] >> torch.arange(regime_dims, device=device)) & 1).to(dtype)
    signs = bits * 2 - 1  # (C, regime_dims)

    # hidden segment-specific linear maps on x-space only
    A = _randn(B, C, dv, x_dim, cfg=cfg) / math.sqrt(x_dim)

    K = torch.zeros(B, L, d, device=device, dtype=dtype)
    V = torch.zeros(B, L, dv, device=device, dtype=dtype)
    query_mask = torch.zeros(B, L, device=device, dtype=torch.bool)

    for c in range(C):
        start = c * segment_len
        end = start + segment_len

        # support tokens
        xs = support_scale * _randn(B, segment_len - 1, x_dim, cfg=cfg)

        # shifted query token
        mu = torch.zeros(x_dim, device=device, dtype=dtype)
        mu[0] = query_shift
        xq = mu[None, None, :] + _randn(B, 1, x_dim, cfg=cfg)

        x_seg = torch.cat([xs, xq], dim=1)  # (B,S,x_dim)

        # key = [regime tag | x | zeros]
        K[:, start:end, :regime_dims] = signs[c][None, None, :]
        K[:, start:end, regime_dims:regime_dims + x_dim] = x_seg

        # value = A_c x + noise
        A_c = A[:, c, :, :]                                  # (B,dv,x_dim)
        V_clean = torch.einsum("bvx,bsx->bsv", A_c, x_seg)   # (B,S,dv)
        V[:, start:end, :] = V_clean + cfg.sigma * _randn(B, segment_len, dv, cfg=cfg)

        # only evaluate on the shifted query position
        query_mask[:, end - 1] = True

    return K, V


@torch.no_grad()
def generate_global_linear(cfg: EvalConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One global linear map shared across the whole sequence.
    Should strongly favour global linear regression.
    """
    K = _randn(cfg.batch_size, cfg.L, cfg.d, cfg=cfg)
    A = _randn(cfg.batch_size, cfg.dv, cfg.d, cfg=cfg)
    V_clean = torch.einsum("bvd,bld->blv", A, K)
    V = V_clean + cfg.sigma * _randn(cfg.batch_size, cfg.L, cfg.dv, cfg=cfg)
    return K, V


@torch.no_grad()
def generate_piecewise_linear(
    cfg: EvalConfig,
    segment_len: int = 16,
    regime_dims: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Segment-specific linear maps, with regime identity encoded geometrically
    in the first regime_dims coordinates via unique sign patterns.
    """
    assert cfg.L % segment_len == 0
    C = cfg.L // segment_len
    if C > (1 << regime_dims):
        raise ValueError(f"C={C} must be <= 2^regime_dims={1<<regime_dims}")

    idx = torch.randperm(1 << regime_dims, device=cfg.device)[:C]
    bits = ((idx[:, None] >> torch.arange(regime_dims, device=cfg.device)) & 1).to(cfg.dtype)
    signs = bits * 2 - 1  # {-1, +1}

    seg_id = torch.arange(cfg.L, device=cfg.device) // segment_len
    seg_id = seg_id[None, :].expand(cfg.batch_size, -1).contiguous()

    K = _randn(cfg.batch_size, cfg.L, cfg.d, cfg=cfg)
    K[:, :, :regime_dims] *= signs[seg_id]

    A = _randn(cfg.batch_size, C, cfg.dv, cfg.d, cfg=cfg)
    A_tok = A.gather(
        dim=1,
        index=seg_id[:, :, None, None].expand(cfg.batch_size, cfg.L, cfg.dv, cfg.d)
    )
    V_clean = torch.einsum("blvd,bld->blv", A_tok, K)
    V = V_clean + cfg.sigma * _randn(cfg.batch_size, cfg.L, cfg.dv, cfg=cfg)
    return K, V


@torch.no_grad()
def generate_prototype_lookup(
    cfg: EvalConfig,
    n_proto: int = 32,
    proto_spread: float = 6.0,
    key_noise: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieval-like task:
    keys cluster around prototypes, values are arbitrary labels.
    Should favour sharp retrieval.
    """
    centers = proto_spread * _randn(cfg.batch_size, n_proto, cfg.d, cfg=cfg)
    proto_values = torch.sign(_randn(cfg.batch_size, n_proto, cfg.dv, cfg=cfg))

    proto_id = torch.randint(0, n_proto, (cfg.batch_size, cfg.L), device=cfg.device)

    K = centers.gather(
        dim=1,
        index=proto_id[:, :, None].expand(cfg.batch_size, cfg.L, cfg.d)
    )
    K = K + key_noise * _randn(cfg.batch_size, cfg.L, cfg.d, cfg=cfg)

    V_clean = proto_values.gather(
        dim=1,
        index=proto_id[:, :, None].expand(cfg.batch_size, cfg.L, cfg.dv)
    )
    V = V_clean + cfg.sigma * _randn(cfg.batch_size, cfg.L, cfg.dv, cfg=cfg)
    return K, V


@torch.no_grad()
def generate_smooth_nonlinear_local(
    cfg: EvalConfig,
    latent_dim: int = 2,
    u_scale: float = 3.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stronger locality-favouring nonlinear task.

    Idea:
    - latent U lives in a low-dim space over a wide range
    - K is a curved, folded embedding of U
    - V is a smooth nonlinear function of U
    - local neighbourhoods in K still carry useful information,
      but a single global linear map K -> V should fit poorly

    This should favour soft / local-linear methods more than global linear.
    """
    U = u_scale * _randn(cfg.batch_size, cfg.L, latent_dim, cfg=cfg)
    u1 = U[..., 0]
    u2 = U[..., 1]

    # Make key geometry directly meaningful
    K = torch.stack(
        [
            u1,
            u2,
            torch.sin(u1),
            torch.cos(u1),
            torch.sin(u2),
            torch.cos(u2),
        ],
        dim=-1
    )

    # pad/project gently to d dims
    Wk = _randn(cfg.batch_size, cfg.d, K.shape[-1], cfg=cfg) / math.sqrt(K.shape[-1])
    K = torch.einsum("bdm,blm->bld", Wk, K)
    K = 0.1 * K  # keep scales modest

    # Smooth bounded target
    V_feats = torch.stack(
        [
            torch.sin(0.8 * u1),
            torch.cos(0.8 * u2),
            torch.sin(0.5 * (u1 + u2)),
            torch.cos(0.5 * (u1 - u2)),
        ],
        dim=-1,
    )

    Wv = _randn(cfg.batch_size, cfg.dv, V_feats.shape[-1], cfg=cfg) / math.sqrt(V_feats.shape[-1])
    V = torch.einsum("bvm,blm->blv", Wv, V_feats)
    V = V + cfg.sigma * _randn(cfg.batch_size, cfg.L, cfg.dv, cfg=cfg)
    return K, V

TASK_FNS = {
    "global_linear": generate_global_linear,
    "piecewise_linear": generate_piecewise_linear,
    "prototype_lookup": generate_prototype_lookup,
    "smooth_nonlinear_local": generate_smooth_nonlinear_local,
    "shifted_local_map": generate_support_query_shifted_local_map,
}


# ============================================================
# 3) LEARNERS
# ============================================================
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

LEARNERS = ["soft", "sharp", "linear_global", "window_soft", "knn_mean", "linear_attention", "weighted_linear"]

# ============================================================
# 3.5) FEATURE EXTRACTION FOR ROUTER
# ============================================================

ROUTER_LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]


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
def compute_router_features(
    q: torch.Tensor,         # (B,d)
    Kctx: torch.Tensor,      # (B,n,d)
    Vctx: torch.Tensor,      # (B,n,dv)
    cfg: EvalConfig,
) -> torch.Tensor:
    """
    Per-query handcrafted features.

    Features:
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
    """
    B, n, d = Kctx.shape
    dv = Vctx.shape[-1]
    k_feat = min(max(cfg.k_linear_local, cfg.k_sharp, cfg.k_knn_mean, 4), n)

    sims, top_vals, top_idx, soft_w, top_soft_w = _topk_cosine_stats(
        q, Kctx, k_feat, cfg.beta_soft
    )

    # top-k neighbours
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
    soft_entropy = -(soft_w * (soft_w.clamp_min(1e-8).log())).sum(dim=-1)

    # 5 effective support size
    eff_support = 1.0 / (soft_w.pow(2).sum(dim=-1) + 1e-8)

    # 6 nn distance
    nn_dist = 1.0 - max_sim

    # 7 weighted variance of retrieved values
    v_bar = (top_soft_w[:, :, None] * V_top).sum(dim=1)                  # (B,dv)
    v_var = (top_soft_w[:, :, None] * (V_top - v_bar[:, None, :]).pow(2)).sum(dim=(1, 2))

    # 8,9 local covariance stats on retrieved keys
    k_bar = (top_soft_w[:, :, None] * K_top).sum(dim=1)                  # (B,d)
    Kc = K_top - k_bar[:, None, :]
    sqrt_w = torch.sqrt(top_soft_w + 1e-8)[:, :, None]
    Xw = Kc * sqrt_w                                                     # (B,k,d)
    cov = torch.einsum("bkd,bke->bde", Xw, Xw) / max(k_feat, 1)          # (B,d,d)

    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-8)                 # (B,d)
    cov_trace = eigvals.sum(dim=-1)
    cov_log_cond = torch.log(eigvals[:, -1] / eigvals[:, 0])

    # 10 weighted local fit residual
    fit_resid = _weighted_local_fit_residual(
        q, Kctx, Vctx,
        k_top=cfg.k_linear_local,
        ridge_lambda=cfg.ridge_lambda,
        beta=cfg.beta_soft,
    )

    feats = torch.stack(
        [
            max_sim,
            top_gap,
            mean_topk_sim,
            std_topk_sim,
            soft_entropy,
            eff_support,
            nn_dist,
            v_var,
            cov_trace,
            cov_log_cond,
            fit_resid,
        ],
        dim=-1
    )  # (B,11)

    return feats

# ============================================================
# 3.6) FEATURE DATASET COLLECTION
# ============================================================

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

# ============================================================
# 3.7) ROUTER MODELS
# ============================================================

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


def standardize_train_test(X_train: torch.Tensor, X_test: torch.Tensor):
    mu = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (X_train - mu) / std, (X_test - mu) / std, mu, std


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
    epochs: int = 4000,
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
# 3.9) ROUTER EVALUATION
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

# ============================================================
# 3.10) ROUTER EXPERIMENT RUNNER
# ============================================================

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


def run_router_experiment(
    cfg: EvalConfig,
    router_mode: str = "loss_mlp",   # "logistic" or "loss_mlp"
    task_names: List[str] = None,
    learners: List[str] = None,
    n_batches_per_task: int = 8,
    seed: int = 0,
):
    if task_names is None:
        task_names = list(TASK_FNS.keys())
    if learners is None:
        learners = ROUTER_LEARNERS

    print("\n=== Collecting feature dataset ===")
    ds = collect_router_dataset(
        cfg=cfg,
        task_names=task_names,
        learners=learners,
        n_batches_per_task=n_batches_per_task,
    )

    split = train_test_split_dataset(ds, train_frac=0.8, seed=seed)

    X_train_std, X_test_std, mu, std = standardize_train_test(
        split["X_train"], split["X_test"]
    )

    if router_mode == "logistic":
        print("\n=== Training logistic winner router ===")
        model = train_logistic_router(
            X_train=X_train_std,
            y_train=split["y_train"],
            X_val=X_test_std,
            y_val=split["y_test"],
            lr=1e-2,
            weight_decay=1e-4,
            epochs=4000,
        )

        train_metrics = evaluate_logistic_router(
            model,
            X_train_std,
            split["y_train"],
            split["losses_train"],
        )
        test_metrics = evaluate_logistic_router(
            model,
            X_test_std,
            split["y_test"],
            split["losses_test"],
        )

    elif router_mode == "loss_mlp":
        print("\n=== Training loss-predicting MLP router ===")
        model = train_loss_mlp_router(
            X_train=X_train_std,
            losses_train=split["losses_train"],
            X_val=X_test_std,
            losses_val=split["losses_test"],
            lr=1e-3,
            weight_decay=1e-4,
            epochs=8000,
            hidden_dim=64,
        )

        train_metrics = evaluate_loss_router(
            model,
            X_train_std,
            split["y_train"],
            split["losses_train"],
        )
        test_metrics = evaluate_loss_router(
            model,
            X_test_std,
            split["y_test"],
            split["losses_test"],
        )

    else:
        raise ValueError(f"Unknown router_mode: {router_mode}")

    print("\n=== Router results ===")
    print(f"router mode:           {router_mode}")
    print(f"learners:              {learners}")
    print(f"train acc:             {train_metrics['acc']:.4f}")
    print(f"test acc:              {test_metrics['acc']:.4f}")
    print(f"test best fixed loss:  {test_metrics['best_fixed_loss']:.6f}")
    print(f"test routed loss:      {test_metrics['routed_loss']:.6f}")
    print(f"test oracle loss:      {test_metrics['oracle_loss']:.6f}")
    print(f"oracle gap closed:     {100 * test_metrics['oracle_gap_closed_frac']:.2f}%")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split": split,
        "mu": mu,
        "std": std,
        "learners": learners,
        "router_mode": router_mode,
    }
# ============================================================
# 4) PER-QUERY PREDICTION
# ============================================================

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


# ============================================================
# 5) EVALUATION: FIXED LEARNERS, UNIFORM MIXTURE, ORACLE
# ============================================================

@torch.no_grad()
def evaluate_task_family_with_winners(
    task_name: str,
    cfg: EvalConfig,
) -> Dict[str, object]:
    """
    Returns:
      - mean MSE for each learner / uniform mix / oracle
      - oracle winner counts
      - oracle winner fractions
      - winner-by-position counts
    """
    K, V = TASK_FNS[task_name](cfg)
    B, L, _ = K.shape

    totals = {name: 0.0 for name in LEARNERS}
    totals["uniform_mix"] = 0.0
    totals["oracle"] = 0.0

    winner_counts = {name: 0 for name in LEARNERS}
    winner_by_pos = {name: torch.zeros(L, dtype=torch.long) for name in LEARNERS}

    count = 0
    total_queries = 0

    for i in range(cfg.min_context, L):
        q = K[:, i, :]
        Kctx = K[:, :i, :]
        Vctx = V[:, :i, :]
        target = V[:, i, :]

        preds = {}
        per_learner_mse = []

        for learner in LEARNERS:
            yhat = predict_with_learner(learner, q, Kctx, Vctx, cfg)
            preds[learner] = yhat
            mse = ((yhat - target) ** 2).mean(dim=-1)  # (B,)
            per_learner_mse.append(mse)
            totals[learner] += mse.mean().item()

        # uniform mixture
        mix_pred = torch.stack([preds[l] for l in LEARNERS], dim=0).mean(dim=0)
        mix_mse = ((mix_pred - target) ** 2).mean(dim=-1)
        totals["uniform_mix"] += mix_mse.mean().item()

        # oracle
        per_learner_mse = torch.stack(per_learner_mse, dim=-1)  # (B,M)
        oracle_vals, oracle_idx = per_learner_mse.min(dim=-1)   # (B,), (B,)
        totals["oracle"] += oracle_vals.mean().item()

        # winner counts
        for m, learner in enumerate(LEARNERS):
            wins = (oracle_idx == m)
            n_wins = wins.sum().item()
            winner_counts[learner] += n_wins
            winner_by_pos[learner][i] += n_wins

        count += 1
        total_queries += B

    mean_metrics = {k: v / count for k, v in totals.items()}
    winner_fractions = {k: winner_counts[k] / total_queries for k in LEARNERS}

    return {
        "metrics": mean_metrics,
        "winner_counts": winner_counts,
        "winner_fractions": winner_fractions,
        "winner_by_pos": winner_by_pos,
    }

@torch.no_grad()
def evaluate_task_family_full(
    task_name: str,
    cfg: EvalConfig,
) -> Dict[str, float]:
    """
    Returns mean MSE for:
      - each fixed learner
      - uniform mixture of learners
      - oracle per-query selector
    """
    K, V = TASK_FNS[task_name](cfg)
    B, L, _ = K.shape

    totals = {name: 0.0 for name in LEARNERS}
    totals["uniform_mix"] = 0.0
    totals["oracle"] = 0.0

    count = 0

    for i in range(cfg.min_context, L):
        q = K[:, i, :]
        Kctx = K[:, :i, :]
        Vctx = V[:, :i, :]
        target = V[:, i, :]

        preds = {}
        per_learner_mse = []

        for learner in LEARNERS:
            yhat = predict_with_learner(learner, q, Kctx, Vctx, cfg)
            preds[learner] = yhat
            mse = ((yhat - target) ** 2).mean(dim=-1)  # (B,)
            per_learner_mse.append(mse)
            totals[learner] += mse.mean().item()

        # uniform mixture
        mix_pred = torch.stack([preds[l] for l in LEARNERS], dim=0).mean(dim=0)
        mix_mse = ((mix_pred - target) ** 2).mean(dim=-1)
        totals["uniform_mix"] += mix_mse.mean().item()

        # oracle per-query selector
        per_learner_mse = torch.stack(per_learner_mse, dim=-1)  # (B,M)
        oracle_mse = per_learner_mse.min(dim=-1).values
        totals["oracle"] += oracle_mse.mean().item()

        count += 1

    return {k: v / count for k, v in totals.items()}


# ============================================================
# 6) RUN + REPORTING
# ============================================================


def print_results_table(results: Dict[str, Dict[str, float]]):
    cols = LEARNERS + ["uniform_mix", "oracle"]
    print("\nEvaluation results")
    header = f"{'task family':<24}" + "".join([f"{c:>16}" for c in cols]) + f"{'winner':>16}"
    print(header)
    print("-" * len(header))

    for task_name, vals in results.items():
        fixed_winner = min(LEARNERS, key=lambda x: vals[x])
        row = f"{task_name:<24}"
        for c in cols:
            row += f"{vals[c]:>16.6f}"
        row += f"{fixed_winner:>16}"
        print(row)


def print_oracle_gaps(results: Dict[str, Dict[str, float]]):
    print("\nOracle gap analysis")
    print("-------------------")
    for task_name, vals in results.items():
        best_fixed = min(vals[l] for l in LEARNERS)
        oracle = vals["oracle"]
        uniform = vals["uniform_mix"]

        oracle_gap_abs = best_fixed - oracle
        oracle_gap_rel = oracle_gap_abs / best_fixed if best_fixed > 0 else 0.0

        uniform_gap_abs = best_fixed - uniform
        uniform_gap_rel = uniform_gap_abs / best_fixed if best_fixed > 0 else 0.0

        print(
            f"{task_name:<24} "
            f"best_fixed={best_fixed:.6f}  "
            f"oracle={oracle:.6f}  "
            f"oracle_gain={oracle_gap_abs:.6f} ({100*oracle_gap_rel:.2f}%)  "
            f"uniform_mix={uniform:.6f}  "
            f"uniform_gap={uniform_gap_abs:.6f} ({100*uniform_gap_rel:.2f}%)"
        )

def print_winner_fractions(results: Dict[str, Dict[str, object]]):
    print("\nOracle winner fractions")
    print("-----------------------")
    header = f"{'task family':<24}" + "".join([f"{l:>16}" for l in LEARNERS])
    print(header)
    print("-" * len(header))

    for task_name, out in results.items():
        row = f"{task_name:<24}"
        fracs = out["winner_fractions"]
        for learner in LEARNERS:
            row += f"{fracs[learner]:>16.3f}"
        print(row)



def run_all(cfg: EvalConfig, log_winner_fractions: bool = True) -> Dict[str, Dict[str, float]]:
    results = {}
    for task_name in TASK_FNS.keys():
        results[task_name] = evaluate_task_family_full(task_name, cfg)
    if log_winner_fractions:
        winners_results = {}
        for task_name in TASK_FNS.keys():
            winners_results[task_name] = evaluate_task_family_with_winners(task_name, cfg)

        print_winner_fractions(winners_results)
    return results


# ============================================================
# 7) MULTI-SEED
# ============================================================

def run_all_multiseed(cfg: EvalConfig, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    cols = LEARNERS + ["uniform_mix", "oracle"]
    accum = {
        task: {c: 0.0 for c in cols}
        for task in TASK_FNS.keys()
    }

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        res = run_all(cfg, False)
        for task_name in res:
            for c in cols:
                accum[task_name][c] += res[task_name][c]

    n = len(seeds)
    avg = {
        task: {c: accum[task][c] / n for c in cols}
        for task in accum
    }
    return avg


# ============================================================
# 8) MAIN
# ============================================================

if __name__ == "__main__":
    cfg = EvalConfig(
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

    # print("\n=== Single-seed run ===")
    # single = run_all(cfg)
    # print_results_table(single)
    # print_oracle_gaps(single)

    # print("\n=== Multi-seed average ===")
    # multi = run_all_multiseed(cfg, seeds=[0, 1, 2, 3, 4])
    # print_results_table(multi)
    # print_oracle_gaps(multi)


        # ---- Feature -> winner prediction ----
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