import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------
# 1) Synthetic task generation
# ----------------------------

@dataclass
class TaskConfig:
    L: int = 512          # total sequence length
    S: int = 64           # segment length
    d: int = 64           # key dimension
    m: int = 8            # number of dims with fixed sign pattern
    dv: int = 64          # value/target dimension
    sigma: float = 0.05   # noise std
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def sample_unique_sign_patterns(C: int, m: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns S_c in {-1, +1}^m for c=0..C-1, all unique.
    Requires C <= 2^m.
    Shape: (C, m)
    """
    if C > (1 << m):
        raise ValueError(f"Need C <= 2^m, but got C={C}, m={m}, 2^m={1<<m}")

    # sample without replacement by sampling integers and decoding bits
    # map bit {0,1} -> {-1,+1}
    idx = torch.randperm(1 << m, device=device)[:C]
    bits = ((idx[:, None] >> torch.arange(m, device=device)) & 1).to(dtype)
    signs = bits * 2 - 1  # 0->-1, 1->+1
    return signs


@torch.no_grad()
def generate_piecewise_linear_sequence(
    cfg: TaskConfig,
    batch_size: int,
    *,
    return_segment_ids: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates (K, V, seg_id) where:
      - Sequence length L split into C=L/S segments of length S.
      - For each segment c:
          A_c ~ N(0, I) as a matrix in R^{dv x d}
          For each token i in segment c:
              z ~ N(0, I_d)
              k = z with first m dims sign-flipped by S_c
              v = A_c k + eps, eps ~ N(0, sigma^2 I)
    Shapes:
      K: (B, L, d)
      V: (B, L, dv)
      seg_id: (B, L)  (integer segment index)
    """
    assert cfg.L % cfg.S == 0
    C = cfg.L // cfg.S
    device, dtype = cfg.device, cfg.dtype

    # segment sign patterns S_c in {-1,+1}^m, unique across segments
    S_c = sample_unique_sign_patterns(C, cfg.m, device, dtype)  # (C, m)

    # segment-specific linear maps A_c ~ N(0, I): (B, C, dv, d)
    A = torch.randn(batch_size, C, cfg.dv, cfg.d, device=device, dtype=dtype)

    # base z ~ N(0, I): (B, L, d)
    z = torch.randn(batch_size, cfg.L, cfg.d, device=device, dtype=dtype)

    # assign each position to a segment
    seg_id = torch.arange(cfg.L, device=device) // cfg.S              # (L,)
    seg_id = seg_id[None, :].expand(batch_size, -1).contiguous()      # (B, L)

    # apply sign pattern in first m dims depending on segment
    K = z.clone()
    # gather S_c for each token: (B, L, m)
    signs = S_c[seg_id]  # indexing: seg_id is (B,L) -> signs (B,L,m)
    K[:, :, :cfg.m] = K[:, :, :cfg.m] * signs

    # compute V = A_c k + eps
    # gather A_c for each token: (B, L, dv, d)
    A_tok = A.gather(
        dim=1,
        index=seg_id[:, :, None, None].expand(batch_size, cfg.L, cfg.dv, cfg.d)
    )
    # v_i = A_c k_i : (B, L, dv)
    V_clean = torch.einsum("bldk,bld->blk", A_tok, K)
    eps = cfg.sigma * torch.randn_like(V_clean)
    V = V_clean + eps

    if return_segment_ids:
        return K, V, seg_id
    return K, V, torch.empty(0, device=device, dtype=torch.long)


# ------------------------------------------
# 2) Frozen single-head causal softmax attention
# ------------------------------------------

class FrozenSoftmaxAttention(nn.Module):
    """
    Single-head causal attention:
      q_i = Wq k_i
      k_j' = Wk k_j
      v_j' = Wv v_j
      attn(i,j) = softmax_j( beta * q_i^T k_j' / sqrt(dh) ), over j<=i-1
      out_i = Wo * sum_j attn(i,j) v_j'
    This produces a prediction \hat v_i in R^{dv}.
    """
    def __init__(
        self,
        d: int,
        dv: int,
        dh: int = None,
        beta: float = 5.0,
        identity_init: bool = True,
    ):
        super().__init__()
        dh = dh or d
        self.d = d
        self.dv = dv
        self.dh = dh
        self.beta = beta

        self.Wq = nn.Linear(d, dh, bias=False)
        self.Wk = nn.Linear(d, dh, bias=False)
        self.Wv = nn.Linear(dv, dv, bias=False)
        self.Wo = nn.Linear(dv, dv, bias=False)

        # "frozen state": we won't train these parameters.
        # If identity_init=True and dimensions match, this makes behavior easier to interpret:
        # similarity is directly based on key dot-products, and outputs average past values.
        if identity_init:
            if d == dh:
                nn.init.eye_(self.Wq.weight)
                nn.init.eye_(self.Wk.weight)
            else:
                nn.init.normal_(self.Wq.weight, std=1.0 / math.sqrt(d))
                nn.init.normal_(self.Wk.weight, std=1.0 / math.sqrt(d))
            nn.init.eye_(self.Wv.weight)
            nn.init.eye_(self.Wo.weight)
        else:
            nn.init.normal_(self.Wq.weight, std=1.0 / math.sqrt(d))
            nn.init.normal_(self.Wk.weight, std=1.0 / math.sqrt(d))
            nn.init.normal_(self.Wv.weight, std=1.0 / math.sqrt(dv))
            nn.init.normal_(self.Wo.weight, std=1.0 / math.sqrt(dv))

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        K: (B, L, d)   keys/features
        V: (B, L, dv)  observed values/targets (teacher forcing as context)
        Returns:
          Vhat: (B, L, dv) predictions; Vhat[:,0] will be ~0 since no past.
        """
        B, L, _ = K.shape

        Qh = self.Wq(K)  # (B, L, dh)
        Kh = self.Wk(K)  # (B, L, dh)
        Vp = self.Wv(V)  # (B, L, dv)

        # scores: (B, L, L)
        scores = torch.einsum("bih,bjh->bij", Qh, Kh) / math.sqrt(self.dh)
        scores = scores * self.beta

        # causal mask: only attend to strictly previous positions (j <= i-1)
        # diagonal and above -> -inf
        causal = torch.triu(torch.ones(L, L, device=K.device, dtype=torch.bool), diagonal=0)
        # causal[i,j]=True means j>=i, disallow. We want only j<i.
        scores = scores.masked_fill(causal[None, :, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, L, L); row i sums to 1 if any valid past, else NaN at i=0
        attn[:, 0, :] = 0.0  # define i=0 as no context -> zero weights

        # output: (B, L, dv)
        out = torch.einsum("bij,bjd->bid", attn, Vp)
        Vhat = self.Wo(out)
        return Vhat


# ----------------------------
# 3) Evaluation + plots
# ----------------------------

@torch.no_grad()
def evaluate_attention_on_task(
    cfg: TaskConfig,
    model: FrozenSoftmaxAttention,
    batch_size: int = 256,
) -> dict:
    K, V, seg_id = generate_piecewise_linear_sequence(cfg, batch_size)
    Vhat = model(K, V)

    # per-position squared error (B, L)
    mse_pos = ((Vhat - V) ** 2).mean(dim=-1)

    # average across batch
    mse_pos_mean = mse_pos.mean(dim=0)  # (L,)

    # within-segment index t = i % S
    t = torch.arange(cfg.L, device=cfg.device) % cfg.S
    mse_within = torch.zeros(cfg.S, device=cfg.device, dtype=cfg.dtype)
    counts = torch.zeros(cfg.S, device=cfg.device, dtype=cfg.dtype)
    for i in range(cfg.L):
        mse_within[t[i]] += mse_pos_mean[i]
        counts[t[i]] += 1
    mse_within = mse_within / counts  # (S,)

    return {
        "mse_pos_mean": mse_pos_mean.cpu(),
        "mse_within": mse_within.cpu(),
        "seg_id_example": seg_id[0].cpu(),
    }


from pathlib import Path

def plot_results(cfg: TaskConfig, results: dict, out_dir: str = "outputs/ttr_synth"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mse_pos = results["mse_pos_mean"].numpy()
    mse_within = results["mse_within"].numpy()

    # Plot 1: loss over absolute position (shows spikes at boundaries)
    plt.figure()
    plt.plot(mse_pos)
    plt.title("Per-position MSE (mean over batch)")
    plt.xlabel("Position i")
    plt.ylabel("MSE")
    for b in range(0, cfg.L + 1, cfg.S):
        plt.axvline(b, linewidth=0.5)
    plt.tight_layout()
    f1 = out_dir / "mse_per_position.png"
    plt.savefig(f1, dpi=200)
    plt.close()

    # Plot 2: average loss vs within-segment index (should decay if adapting)
    plt.figure()
    plt.plot(mse_within)
    plt.title("MSE vs within-segment index t (averaged across segments)")
    plt.xlabel("t = i mod S")
    plt.ylabel("MSE")
    plt.tight_layout()
    f2 = out_dir / "mse_within_segment.png"
    plt.savefig(f2, dpi=200)
    plt.close()

    print(f"[saved] {f1}")
    print(f"[saved] {f2}")


# ----------------------------
# 4) Run it
# ----------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TaskConfig(
        L=512,
        S=64,
        d=64,
        m=8,
        dv=64,
        sigma=0.05,
        device=device,
    )

    # Frozen softmax attention baseline
    attn = FrozenSoftmaxAttention(
        d=cfg.d,
        dv=cfg.dv,
        dh=cfg.d,          # keep dh=d for interpretability
        beta=8.0,          # higher beta => sharper kernel (more "local")
        identity_init=True # interpretability: dot-product on raw keys
    ).to(device)

    results = evaluate_attention_on_task(cfg, attn, batch_size=256)
    plot_results(cfg, results, out_dir="outputs/ttr_synth")
