from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import torch

from experiments.attention_learners import build_learners
from experiments.synthetic_alignment.config import EvalConfig
from experiments.synthetic_alignment.synthetic_tasks import DEFAULT_TASK_REPO


DEFAULT_TASK_NAMES = [
    "piecewise_linear",
    "shifted_local_map",
    "smooth_nonlinear_local",
]

DEFAULT_LEARNER_NAMES = [
    "soft",
    "sharp",
    "window_soft",
    "knn_mean",
    "linear_attention",
    "linear_global",
    "local_linear_attention",
]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_task(
    task_name: str,
    cfg: EvalConfig,
    learners: Dict[str, object],
) -> tuple[dict[str, float], dict[str, float], int]:
    task = DEFAULT_TASK_REPO.get(task_name)
    out = task(cfg)

    K = out["K"]
    V = out["V"]
    B, L, _ = K.shape

    query_mask = out.get("query_mask")
    if query_mask is None:
        query_mask = torch.ones(B, L, dtype=torch.bool, device=K.device)

    totals = {name: 0.0 for name in learners}
    totals["uniform_mix"] = 0.0
    totals["oracle"] = 0.0

    winner_counts = {name: 0 for name in learners}
    total_queries = 0

    for i in range(cfg.min_context, L):
        valid = query_mask[:, i]
        if not valid.any():
            continue

        q = K[valid, i, :]
        Kctx = K[valid, :i, :]
        Vctx = V[valid, :i, :]
        target = V[valid, i, :]

        n_valid = int(valid.sum().item())
        total_queries += n_valid

        predictions = []
        per_learner_mse = []

        for name, learner in learners.items():
            yhat = learner(q, Kctx, Vctx, cfg)
            predictions.append(yhat)

            mse = ((yhat - target) ** 2).mean(dim=-1)
            per_learner_mse.append(mse)
            totals[name] += mse.sum().item()

        mix_pred = torch.stack(predictions, dim=0).mean(dim=0)
        mix_mse = ((mix_pred - target) ** 2).mean(dim=-1)
        totals["uniform_mix"] += mix_mse.sum().item()

        loss_mat = torch.stack(per_learner_mse, dim=1)
        oracle_vals, oracle_idx = loss_mat.min(dim=1)
        totals["oracle"] += oracle_vals.sum().item()

        for idx, name in enumerate(learners):
            winner_counts[name] += int((oracle_idx == idx).sum().item())

    if total_queries == 0:
        raise RuntimeError(f"No valid queries for task {task_name}")

    mean_metrics = {k: v / total_queries for k, v in totals.items()}
    winner_fractions = {k: winner_counts[k] / total_queries for k in learners}

    return mean_metrics, winner_fractions, total_queries


def mode_or_first(values: Iterable[str]) -> str:
    series = pd.Series(list(values))
    modes = series.mode()
    if len(modes) == 0:
        return str(series.iloc[0])
    return str(modes.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thesis Chapter 3 TTR readout learner evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/synthetic_alignment/ch3_ttr_readouts",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASK_NAMES),
        help="Comma-separated task names.",
    )
    parser.add_argument(
        "--learners",
        type=str,
        default=",".join(DEFAULT_LEARNER_NAMES),
        help="Comma-separated learner names.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--dv", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--min_context", type=int, default=8)
    parser.add_argument("--k_sharp", type=int, default=4)
    parser.add_argument("--k_knn_mean", type=int, default=4)
    parser.add_argument("--k_linear_local", type=int, default=16)
    parser.add_argument("--ridge_lambda", type=float, default=1e-1)
    parser.add_argument("--local_kernel_beta", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = parse_str_list(args.tasks)
    learner_names = parse_str_list(args.learners)
    seeds = parse_int_list(args.seeds)

    metric_rows = []
    winner_rows = []
    gap_rows = []

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        set_seed(seed)

        cfg = EvalConfig(
            L=args.L,
            d=args.d,
            dv=args.dv,
            batch_size=args.batch_size,
            sigma=args.sigma,
            device=args.device,
            local_kernel_beta=args.local_kernel_beta,
            k_sharp=args.k_sharp,
            k_knn_mean=args.k_knn_mean,
            k_linear_local=args.k_linear_local,
            ridge_lambda=args.ridge_lambda,
            min_context=args.min_context,
        )

        learners = build_learners(learner_names)

        for task_name in task_names:
            print(f"Evaluating {task_name}")
            metrics, winners, n_queries = evaluate_task(task_name, cfg, learners)

            for method, mse in metrics.items():
                metric_rows.append(
                    {
                        "seed": seed,
                        "task": task_name,
                        "method": method,
                        "mse": mse,
                        "n_queries": n_queries,
                    }
                )

            for method, frac in winners.items():
                winner_rows.append(
                    {
                        "seed": seed,
                        "task": task_name,
                        "method": method,
                        "winner_fraction": frac,
                        "n_queries": n_queries,
                    }
                )

            best_fixed = min(metrics[m] for m in learner_names)
            best_fixed_method = min(learner_names, key=lambda m: metrics[m])
            oracle = metrics["oracle"]
            uniform_mix = metrics["uniform_mix"]

            gap_rows.append(
                {
                    "seed": seed,
                    "task": task_name,
                    "best_fixed_method": best_fixed_method,
                    "best_fixed_mse": best_fixed,
                    "oracle_mse": oracle,
                    "oracle_gain_abs": best_fixed - oracle,
                    "oracle_gain_rel_pct": 100.0 * (best_fixed - oracle) / best_fixed,
                    "uniform_mix_mse": uniform_mix,
                    "uniform_mix_gain_abs": best_fixed - uniform_mix,
                    "uniform_mix_gain_rel_pct": 100.0
                    * (best_fixed - uniform_mix)
                    / best_fixed,
                    "n_queries": n_queries,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    winners_df = pd.DataFrame(winner_rows)
    gaps_df = pd.DataFrame(gap_rows)

    metrics_df.to_csv(out_dir / "learner_metrics_by_seed.csv", index=False)
    winners_df.to_csv(out_dir / "winner_fractions_by_seed.csv", index=False)
    gaps_df.to_csv(out_dir / "oracle_gaps_by_seed.csv", index=False)

    metric_summary = (
        metrics_df.groupby(["task", "method"], as_index=False)
        .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"))
        .sort_values(["task", "method"])
    )
    metric_summary.to_csv(out_dir / "learner_metrics_summary.csv", index=False)

    winner_summary = (
        winners_df.groupby(["task", "method"], as_index=False)
        .agg(
            winner_fraction_mean=("winner_fraction", "mean"),
            winner_fraction_std=("winner_fraction", "std"),
        )
        .sort_values(["task", "method"])
    )
    winner_summary.to_csv(out_dir / "winner_fractions_summary.csv", index=False)

    gap_summary = (
        gaps_df.groupby(["task"], as_index=False)
        .agg(
            best_fixed_mse_mean=("best_fixed_mse", "mean"),
            best_fixed_mse_std=("best_fixed_mse", "std"),
            oracle_mse_mean=("oracle_mse", "mean"),
            oracle_mse_std=("oracle_mse", "std"),
            oracle_gain_abs_mean=("oracle_gain_abs", "mean"),
            oracle_gain_abs_std=("oracle_gain_abs", "std"),
            oracle_gain_rel_pct_mean=("oracle_gain_rel_pct", "mean"),
            oracle_gain_rel_pct_std=("oracle_gain_rel_pct", "std"),
            uniform_mix_mse_mean=("uniform_mix_mse", "mean"),
            uniform_mix_mse_std=("uniform_mix_mse", "std"),
            uniform_mix_gain_rel_pct_mean=("uniform_mix_gain_rel_pct", "mean"),
            uniform_mix_gain_rel_pct_std=("uniform_mix_gain_rel_pct", "std"),
        )
        .sort_values("task")
    )

    best_method_summary = (
        gaps_df.groupby("task")["best_fixed_method"]
        .agg(best_fixed_method_mode=mode_or_first)
        .reset_index()
    )
    gap_summary = best_method_summary.merge(gap_summary, on="task", how="right")
    gap_summary.to_csv(out_dir / "oracle_gaps_summary.csv", index=False)

    print("\n[saved]")
    for path in sorted(out_dir.glob("*.csv")):
        print(f"  {path}")

    print("\n=== Learner metric summary ===")
    print(metric_summary.to_string(index=False))

    print("\n=== Oracle gap summary ===")
    print(gap_summary.to_string(index=False))

    print("\n=== Winner fraction summary ===")
    print(winner_summary.to_string(index=False))


if __name__ == "__main__":
    main()