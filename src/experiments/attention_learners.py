import math
from typing import Dict, List, Type
from experiments.synthetic_alignment.config import EvalConfig
import torch
import torch.nn.functional as F


class BaseAttentionLearner:
    """Callable interface for learners."""

    name: str

    def __call__(
        self,
        q: torch.Tensor,
        Kctx: torch.Tensor,
        Vctx: torch.Tensor,
        cfg: EvalConfig,
    ) -> torch.Tensor:
        return self.predict(q, Kctx, Vctx, cfg)

    @torch.no_grad()
    def predict(
        self,
        q: torch.Tensor,
        Kctx: torch.Tensor,
        Vctx: torch.Tensor,
        cfg: EvalConfig,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def loss(
        self,
        q: torch.Tensor,
        Kctx: torch.Tensor,
        Vctx: torch.Tensor,
        target: torch.Tensor,
        cfg: EvalConfig,
    ) -> torch.Tensor:
        pred = self.predict(q, Kctx, Vctx, cfg)
        return ((pred - target) ** 2).mean(dim=-1)


class SoftKernelLearner(BaseAttentionLearner):
    name = "soft"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        d = q.shape[-1]
        q_col = q.unsqueeze(-1)
        attn_logits = torch.matmul(Kctx, q_col).squeeze(-1) / math.sqrt(d)
        attn_weights = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights.unsqueeze(1), Vctx).squeeze(1)
        return out


class SharpTopKLearner(BaseAttentionLearner):
    name = "sharp"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        B, n, d = Kctx.shape
        k_top = cfg.k_sharp
        k_eff = min(k_top, n)
        q_col = q.unsqueeze(-1)
        scores = torch.matmul(Kctx, q_col).squeeze(-1) / math.sqrt(d)

        top_scores, top_idx = torch.topk(scores, k=k_eff, dim=-1)
        V_top = Vctx.gather(
            dim=1,
            index=top_idx[:, :, None].expand(B, k_eff, Vctx.shape[-1])
        )

        w = F.softmax(top_scores, dim=-1)
        out = torch.matmul(w.unsqueeze(1), V_top).squeeze(1)
        return out


class WindowSoftLearner(BaseAttentionLearner):
    name = "window_soft"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        B, n, d = Kctx.shape
        window_size = cfg.window_size
        w_eff = min(window_size, n)

        K_win = Kctx[:, -w_eff:, :]
        V_win = Vctx[:, -w_eff:, :]

        scores = torch.matmul(K_win, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(d)

        weights = F.softmax(scores, dim=-1)
        yhat = torch.matmul(weights.unsqueeze(1), V_win).squeeze(1)
        return yhat


class KNNMeanLearner(BaseAttentionLearner):
    name = "knn_mean"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        B, n, d = Kctx.shape
        dv = Vctx.shape[-1]
        k_top = cfg.k_knn_mean
        k_eff = min(k_top, n)

        scores = torch.matmul(Kctx, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(d)

        _, top_idx = torch.topk(scores, k=k_eff, dim=-1)

        V_top = Vctx.gather(
            dim=1,
            index=top_idx[:, :, None].expand(B, k_eff, dv)
        )

        yhat = V_top.mean(dim=1)
        return yhat


class LinearAttentionLearner(BaseAttentionLearner):
    name = "linear_attention"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        eps = 1e-6
        phi_q = F.elu(q) + 1.0
        phi_K = F.elu(Kctx) + 1.0

        KV = torch.matmul(phi_K.transpose(1, 2), Vctx)

        Z = phi_K.sum(dim=1)

        num = torch.matmul(phi_q.unsqueeze(1), KV).squeeze(1)
        den = torch.matmul(phi_q.unsqueeze(1), Z.unsqueeze(-1)).squeeze(-1)

        yhat = num / (den + eps)
        return yhat


class GlobalLinearRidgeLearner(BaseAttentionLearner):
    name = "linear_global"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        ridge_lambda = cfg.ridge_lambda
        B, n, d = Kctx.shape
        dv = Vctx.shape[-1]

        ones = torch.ones(B, n, 1, device=Kctx.device, dtype=Kctx.dtype)
        X = torch.cat([Kctx, ones], dim=-1)

        q_aug = torch.cat(
            [q, torch.ones(B, 1, device=q.device, dtype=q.dtype)],
            dim=-1
        )[:, None, :]

        XT = X.transpose(1, 2)
        XTX = XT @ X
        XTY = XT @ Vctx

        I = torch.eye(d + 1, device=Kctx.device, dtype=Kctx.dtype)[None, :, :]
        theta = torch.linalg.solve(XTX + ridge_lambda * I, XTY)

        yhat = (q_aug @ theta).squeeze(1)
        return yhat


class WeightedLocalLinearLearner(BaseAttentionLearner):
    name = "weighted_linear"

    @torch.no_grad()
    def predict(self, q, Kctx, Vctx, cfg):
        k_top = cfg.k_linear_local
        ridge_lambda = cfg.ridge_lambda
        beta = cfg.beta_soft
        B, n, d = Kctx.shape
        dv = Vctx.shape[-1]
        k_eff = min(k_top, n)

        qn = F.normalize(q, dim=-1)
        Kn = F.normalize(Kctx, dim=-1)
        scores = torch.matmul(Kn, qn.unsqueeze(-1)).squeeze(-1)

        top_scores, top_idx = torch.topk(scores, k=k_eff, dim=-1)

        K_top = Kctx.gather(
            dim=1,
            index=top_idx[:, :, None].expand(B, k_eff, d)
        )

        V_top = Vctx.gather(
            dim=1,
            index=top_idx[:, :, None].expand(B, k_eff, dv)
        )

        w = F.softmax(beta * top_scores, dim=-1)

        w_col = w[:, :, None]
        k_bar = (w_col * K_top).sum(dim=1)
        v_bar = (w_col * V_top).sum(dim=1)

        Kc = K_top - k_bar[:, None, :]
        Vc = V_top - v_bar[:, None, :]
        qc = q - k_bar

        sqrt_w = torch.sqrt(w + 1e-8)[:, :, None]
        Xw = Kc * sqrt_w
        Yw = Vc * sqrt_w

        XT = Xw.transpose(1, 2)
        XTX = XT @ Xw
        XTY = XT @ Yw

        I = torch.eye(d, device=Kctx.device, dtype=Kctx.dtype)[None, :, :]
        theta = torch.linalg.solve(XTX + ridge_lambda * I, XTY)

        yhat = v_bar + torch.matmul(qc.unsqueeze(1), theta).squeeze(1)
        return yhat


_LEARNER_CLASS_LIST: List[Type[BaseAttentionLearner]] = [
    SoftKernelLearner,
    SharpTopKLearner,
    GlobalLinearRidgeLearner,
    WindowSoftLearner,
    KNNMeanLearner,
    LinearAttentionLearner,
    WeightedLocalLinearLearner,
]

LEARNER_CLASSES: Dict[str, Type[BaseAttentionLearner]] = {
    cls.name: cls for cls in _LEARNER_CLASS_LIST
}

LEARNERS: List[str] = [cls.name for cls in _LEARNER_CLASS_LIST]


def build_learners(names: List[str] | None = None) -> Dict[str, BaseAttentionLearner]:
    if names is None:
        names = LEARNERS
    return {name: LEARNER_CLASSES[name]() for name in names}
    
def predict_with_learner(
    learner_name: str,
    q: torch.Tensor,
    Kctx: torch.Tensor,
    Vctx: torch.Tensor,
    cfg: EvalConfig,
) -> torch.Tensor:
    if learner_name not in LEARNER_CLASSES:
        raise ValueError(f"Unknown learner: {learner_name}")
    learner = LEARNER_CLASSES[learner_name]()
    return learner(q, Kctx, Vctx, cfg)
