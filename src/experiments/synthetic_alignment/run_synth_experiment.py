

from typing import List, Dict

from experiments.synthetic_alignment.config import EvalConfig
from experiments.synthetic_alignment.dataset import collect_router_dataset, normalize_regrets_train_test, standardize_train_test, train_test_split_dataset
from experiments.learners import LEARNERS, ROUTER_LEARNERS, predict_with_learner
from experiments.routers import evaluate_logistic_router, evaluate_loss_router, evaluate_per_task_best_fixed_baseline, evaluate_router_per_task, train_logistic_router, train_loss_mlp_router
from experiments.synthetic_alignment.tasks import TASK_FNS
import torch


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
            epochs=8000,
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
        print("\n=== Preparing task-normalized regret targets ===")

        # raw regret targets
        oracle_train = split["losses_train"].min(dim=-1, keepdim=True).values
        oracle_test = split["losses_test"].min(dim=-1, keepdim=True).values

        regrets_train = split["losses_train"] - oracle_train
        regrets_test = split["losses_test"] - oracle_test

        regrets_train_norm, regrets_test_norm, regret_stats = normalize_regrets_train_test(
            regrets_train=regrets_train,
            task_ids_train=split["task_ids_train"],
            regrets_test=regrets_test,
            task_ids_test=split["task_ids_test"],
        )

        print("\n=== Training loss-predicting MLP router on normalized regrets ===")
        model = train_loss_mlp_router(
            X_train=X_train_std,
            losses_train=regrets_train_norm,
            X_val=X_test_std,
            losses_val=regrets_test_norm,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=8000,
            hidden_dim=64,
        )

        train_metrics = evaluate_loss_router(
            model,
            X_train_std,
            split["y_train"],
            split["losses_train"],   # evaluate on raw losses
        )
        test_metrics = evaluate_loss_router(
            model,
            X_test_std,
            split["y_test"],
            split["losses_test"],    # evaluate on raw losses
        )
    else:
        raise ValueError(f"Unknown router_mode: {router_mode}")



    pred_test = test_metrics["pred"]

    per_task_test = evaluate_router_per_task(
        pred=pred_test,
        losses=split["losses_test"],
        task_ids=split["task_ids_test"],
        task_to_idx=split["task_to_idx"],
    )
    print("\n=== Per-task router results (test) ===")
    for task_name, vals in per_task_test.items():
        print(
            f"{task_name:<24} "
            f"best_fixed={vals['best_fixed_loss']:.6f}  "
            f"routed={vals['routed_loss']:.6f}  "
            f"oracle={vals['oracle_loss']:.6f}  "
            f"gap_closed={100*vals['oracle_gap_closed_frac']:.2f}%"
        )
    print("\n=== Aggregate Router results ===")
    print(f"router mode:           {router_mode}")
    print(f"learners:              {learners}")
    print(f"train acc:             {train_metrics['acc']:.4f}")
    print(f"test acc:              {test_metrics['acc']:.4f}")
    print(f"test best fixed loss:  {test_metrics['best_fixed_loss']:.6f}")
    print(f"test routed loss:      {test_metrics['routed_loss']:.6f}")
    print(f"test oracle loss:      {test_metrics['oracle_loss']:.6f}")
    print(f"oracle gap closed:     {100 * test_metrics['oracle_gap_closed_frac']:.2f}%")


    baseline_pred = evaluate_per_task_best_fixed_baseline(
        losses_train=split["losses_train"],
        task_ids_train=split["task_ids_train"],
        losses_test=split["losses_test"],
        task_ids_test=split["task_ids_test"],
        task_to_idx=split["task_to_idx"],
    )

    baseline_per_task = evaluate_router_per_task(
        pred=baseline_pred,
        losses=split["losses_test"],
        task_ids=split["task_ids_test"],
        task_to_idx=split["task_to_idx"],
    )

    print("\n=== Per-task best-fixed baseline (test) ===")
    for task_name, vals in baseline_per_task.items():
        print(
            f"{task_name:<24} "
            f"best_fixed={vals['best_fixed_loss']:.6f}  "
            f"routed={vals['routed_loss']:.6f}  "
            f"oracle={vals['oracle_loss']:.6f}  "
            f"gap_closed={100*vals['oracle_gap_closed_frac']:.2f}%"
        )
        
    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "per_task_test": per_task_test,
        "baseline_per_task": baseline_per_task,
        "split": split,
        "mu": mu,
        "std": std,
        "learners": learners,
        "router_mode": router_mode,
    }

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

    print("\n=== Single-seed run ===")
    single = run_all(cfg)
    print_results_table(single)
    print_oracle_gaps(single)

    print("\n=== Multi-seed average ===")
    multi = run_all_multiseed(cfg, seeds=[0, 1, 2, 3, 4])
    print_results_table(multi)
    print_oracle_gaps(multi)


        # ---- Feature -> winner prediction ----
    router_out_logistic = run_router_experiment(
        cfg,
        router_mode="logistic",
        task_names=["piecewise_linear", "prototype_lookup", "smooth_nonlinear_local"],
        learners=ROUTER_LEARNERS,
        n_batches_per_task=8,
        seed=0,
    )

    router_out_loss_mlp = run_router_experiment(
        cfg,
        router_mode="loss_mlp",
        task_names=["piecewise_linear", "prototype_lookup", "smooth_nonlinear_local"],
        learners=ROUTER_LEARNERS,
        n_batches_per_task=8,
        seed=0,
    )