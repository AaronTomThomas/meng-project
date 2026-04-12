from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from experiments.attention_learners import build_learners
from experiments.synthetic_alignment.synthetic_tasks import DEFAULT_TASK_REPO, TaskRepository 
import torch
import torch.nn.functional as F
import math
from experiments.synthetic_alignment.config import RouterExperimentConfig


@dataclass
class RouterDataset:
    X: torch.Tensor
    y: torch.Tensor
    losses: torch.Tensor
    task_ids: torch.Tensor
    task_to_idx: Dict[str, int]

    def subset(self, indices: torch.Tensor) -> "RouterDataset":
        return RouterDataset(
            X=self.X[indices],
            y=self.y[indices],
            losses=self.losses[indices],
            task_ids=self.task_ids[indices],
            task_to_idx=self.task_to_idx,
        )

    def split(
        self, 
        train_frac: float = 0.8, 
        seed: int = 0,
    ) -> Tuple[RouterDataset, RouterDataset]:
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        N = self.X.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)

        n_train = int(train_frac * N)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        return self.subset(train_idx), self.subset(test_idx)


    def standardize_with(
        self,
        other: RouterDataset,
        eps: float = 1e-6
    ):
        mu = self.X.mean(dim=0, keepdim=True)
        std = self.X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (self.X - mu) / std, (other.X - mu) / std, mu, std


class RouterDatasetBuilder: 
    
    def __init__(
        self,
        cfg: RouterExperimentConfig,
        learners: List[str],
        task_repo: TaskRepository | None = None,
    ) -> None:
        self.cfg = cfg
        self.learner_names = learners
        self.learners = build_learners(learners)
        self.task_repo = task_repo or DEFAULT_TASK_REPO
        self._reset_buffers()
        self._task_to_idx: Dict[str, int] = {}

    def _reset_buffers(self) -> None:
        self._X_all: List[torch.Tensor] = []
        self._y_all: List[torch.Tensor] = []
        self._losses_all: List[torch.Tensor] = []
        self._task_id_all: List[torch.Tensor] = []

    
    @torch.no_grad()
    def _topk_cosine_stats(
        self,
        q: torch.Tensor,
        Kctx: torch.Tensor,
        k_top: int,
        beta_soft: float,
    ):
        qn = F.normalize(q, dim=-1)
        Kn = F.normalize(Kctx, dim=-1)
        sims = torch.einsum("bd,bnd->bn", qn, Kn)

        k_eff = min(k_top, Kctx.shape[1])
        top_vals, top_idx = torch.topk(sims, k=k_eff, dim=-1)

        soft_w = F.softmax(beta_soft * sims, dim=-1)
        top_soft_w = F.softmax(beta_soft * top_vals, dim=-1)

        return sims, top_vals, top_idx, soft_w, top_soft_w

    @torch.no_grad()
    def _topk_recency_stats(
        self,
        top_idx: torch.Tensor,
        n_ctx: int,
        window_size: int,
    ):
        lags = (n_ctx - 1 - top_idx).to(torch.float32)

        mean_lag = lags.mean(dim=-1)
        std_lag = lags.std(dim=-1, unbiased=False)
        recent_frac = (lags < window_size).float().mean(dim=-1)

        return mean_lag, std_lag, recent_frac
    
    def _compute_features(
        self,
        q: torch.Tensor,
        Kctx: torch.Tensor,
        Vctx: torch.Tensor,
    ) -> torch.Tensor: 
        

        
        cfg = self.cfg 
        B, n, d = Kctx.shape
        dv = Vctx.shape[-1]
        k_feat = min(max(cfg.k_linear_local, cfg.k_sharp, cfg.k_knn_mean, 4), n)

        sims, top_vals, top_idx, soft_w, top_soft_w = self._topk_cosine_stats(
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

        max_sim = top_vals[:, 0]

        if k_feat >= 2:
            top_gap = top_vals[:, 0] - top_vals[:, 1]
        else:
            top_gap = torch.zeros_like(max_sim)

        mean_topk_lag, std_topk_lag, recent_frac_topk = self._topk_recency_stats(
            top_idx=top_idx,
            n_ctx=n,
            window_size=cfg.window_size,
        )

        mean_topk_sim = top_vals.mean(dim=-1)
        std_topk_sim = top_vals.std(dim=-1, unbiased=False)

        soft_entropy = -(soft_w * soft_w.clamp_min(1e-8).log()).sum(dim=-1)
        eff_support = 1.0 / (soft_w.pow(2).sum(dim=-1) + 1e-8)
        nn_dist = 1.0 - max_sim

        v_bar_top = (top_soft_w[:, :, None] * V_top).sum(dim=1)
        v_var = (top_soft_w[:, :, None] * (V_top - v_bar_top[:, None, :]).pow(2)).sum(dim=(1, 2))


        k_bar_top = (top_soft_w[:, :, None] * K_top).sum(dim=1)
        Kc_top = K_top - k_bar_top[:, None, :]
        sqrt_w_top = torch.sqrt(top_soft_w + 1e-8)[:, :, None]
        Xw_top = Kc_top * sqrt_w_top
        cov = Xw_top.transpose(1, 2) @ Xw_top / max(k_feat, 1)

        eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-8)
        cov_trace = eigvals.sum(dim=-1)
        cov_log_cond = torch.log(eigvals[:, -1] / eigvals[:, 0])


        qc_top = q - k_bar_top
        qc_top_norm = qc_top.norm(dim=-1)

        v_dev = (V_top - v_bar_top[:, None, :]).norm(dim=-1)
        v_std = torch.sqrt(v_var + 1e-8)
        v_max_dev = v_dev.max(dim=-1).values

        eig_ratio_12 = eigvals[:, -1] / eigvals[:, -2].clamp_min(1e-8)
        p = eigvals / eigvals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        effective_rank = torch.exp(-(p * p.clamp_min(1e-8).log()).sum(dim=-1))

        w_eff = min(cfg.window_size, n)
        K_win = Kctx[:, -w_eff:, :]
        qn = F.normalize(q, dim=-1)
        Kn_win = F.normalize(K_win, dim=-1)
        win_scores = torch.einsum("bd,bwd->bw", qn, Kn_win)
        win_weights = F.softmax(cfg.beta_soft * win_scores, dim=-1)

        window_entropy = -(win_weights * win_weights.clamp_min(1e-8).log()).sum(dim=-1)
        mean_window_sim = win_scores.mean(dim=-1)

        sharp_top_vals, _ = torch.topk(sims, k=min(cfg.k_sharp, n), dim=-1)
        sharp_weights = F.softmax(sharp_top_vals, dim=-1)

        sharp_entropy = -(sharp_weights * sharp_weights.clamp_min(1e-8).log()).sum(dim=-1)
        sharp_top1_mass = sharp_weights[:, 0]
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
                window_entropy,
                sharp_entropy,
                sharp_top1_mass,
                mean_window_sim,
                qc_top_norm,
                v_std,
                v_max_dev,
                eig_ratio_12,
                effective_rank,
                mean_topk_lag,
                std_topk_lag,
                recent_frac_topk
            ],
            dim=-1,
        )
        return feats

    def _append_examples_from_task(
        self, 
        task_idx: int, 
        K: torch.Tensor,
        V: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> None: 
        B, L, _ = K.shape
        for i in range(self.cfg.min_context, L):
            valid = query_mask[:, i]
            if not valid.any():
                continue
            q = K[valid, i, :]
            Kctx = K[valid, :i, :]
            Vctx = V[valid, :i, :]
            target = V[valid, i, :]
            feats = self._compute_features(q, Kctx, Vctx)

            losses = []
            for learner in self.learners.values():
                yhat = learner(q, Kctx, Vctx, self.cfg)
                mse = ((yhat - target) ** 2).mean(dim=-1)
                losses.append(mse)
            per_learner_losses = torch.stack(losses, dim=-1)

            labels = per_learner_losses.argmin(dim=-1)
            self._X_all.append(feats.detach().cpu())
            self._y_all.append(labels.detach().cpu())
            self._losses_all.append(per_learner_losses.detach().cpu())

            n_valid = valid.sum().item()
            task_ids = torch.full((n_valid,), task_idx, dtype=torch.long)
            self._task_id_all.append(task_ids)

    def build(
        self,
        task_names: List[str],
        n_batches_per_task: int = 8,
    ) -> RouterDataset: 
        if not task_names:
            raise ValueError("task_names must be non-empty")
        self._task_to_idx = {name: i for i, name in enumerate(task_names)}
        self._reset_buffers()

        for task_name in task_names:
            task = self.task_repo.get(task_name)
            print(f"building dataset for task : {task_name}")
            for _ in range(n_batches_per_task):
                task_out = task(self.cfg)
                K = task_out["K"]
                V = task_out["V"]
                B, L, _ = K.shape
                query_mask = task_out.get("query_mask", torch.ones(B, L, device=K.device, dtype=torch.bool))
                self._append_examples_from_task(
                    task_idx=self._task_to_idx[task_name],
                    K=K,
                    V=V,
                    query_mask=query_mask,
                )

        if not self._X_all:
            raise ValueError("No router samples were generated; check task/query settings")

        return RouterDataset(
            X=torch.cat(self._X_all, dim=0),
            y=torch.cat(self._y_all, dim=0),
            losses=torch.cat(self._losses_all, dim=0),
            task_ids=torch.cat(self._task_id_all, dim=0),
            task_to_idx=self._task_to_idx.copy(),
        )


@torch.no_grad()
def collect_router_dataset(
    cfg: RouterExperimentConfig,
    task_names: List[str],
    learners: List[str],
    n_batches_per_task: int = 8,
    task_repo: TaskRepository | None = None,
) -> RouterDataset:
    
    builder = RouterDatasetBuilder(
        cfg=cfg,
        learners=learners,
        task_repo=task_repo,
    )
    return builder.build(
        task_names=task_names,
        n_batches_per_task=n_batches_per_task,
    )
