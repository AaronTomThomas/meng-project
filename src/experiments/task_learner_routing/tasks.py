import math
from typing import Tuple

from experiments.task_learner_routing.config import EvalConfig, _randn
import torch


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
