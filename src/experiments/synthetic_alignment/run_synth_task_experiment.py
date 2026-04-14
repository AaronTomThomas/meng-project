


from pathlib import Path
from typing import Dict, List

from experiments.attention_learners import build_learners
from experiments.synthetic_alignment.config import EvalConfig
from experiments.synthetic_alignment.synthetic_tasks import (
    DEFAULT_TASK_REPO,
    TaskRepository,
)
import torch

TASK_REPO: TaskRepository = DEFAULT_TASK_REPO

LEARNERS = build_learners([
            "soft",
            "sharp",
            "window_soft",
            "weighted_linear",
            "knn_mean",
            "linear_attention",
            "linear_global",
            ]) 

ROUTER_LEARNERS = build_learners([
    "soft",
    "sharp",
    "window_soft",
    "weighted_linear",
])



def evaluate_task(task_name: str, cfg: EvalConfig) -> Dict:
    print(f"Evaluating task {task_name}")

    task = TASK_REPO.get(task_name)
    out = task(cfg)
    K = out["K"]
    V = out["V"]
    B, L, _ = K.shape
    query_mask = out.get("query_mask")
    if query_mask is None:
        query_mask = torch.ones(B, L, dtype=torch.bool, device=K.device)

    totals = {name : 0.0 for name in LEARNERS.keys()}
    totals["uniform_mix"] = 0.0
    totals["oracle"] = 0.0 

    winner_counts = {name : 0 for name in LEARNERS.keys()}
    winner_by_position = {name: torch.zeros(L, dtype=torch.long ) for name in LEARNERS.keys()}
    
    count = 0
    total_queries = 0

    for i in range(cfg.min_context, L):
        valid = query_mask[:, i]
        if not valid.any():
            continue

        q = K[valid, i, :]
        Kctx = K[valid, :i, :]
        Vctx = V[valid, :i, :]
        target = V[valid, i, :]

        predictions = {}
        per_learner_mse = []

        for name, learner in LEARNERS.items():
            yhat = learner(q, Kctx, Vctx, cfg)
            predictions[name] = yhat
            mse = ((yhat - target) ** 2).mean(dim=-1)
            per_learner_mse.append(mse)
            totals[name] += mse.mean().item()

        mix_pred = torch.stack(list(predictions.values()), dim=0).mean(dim=0)
        mix_mse = ((mix_pred - target) ** 2).mean(dim=-1)
        totals["uniform_mix"] += mix_mse.mean().item()

        per_learner_mse = torch.stack(per_learner_mse, dim=0)
        oracle_vals, oracle_idx = per_learner_mse.min(dim=0)
        totals["oracle"] += oracle_vals.mean().item()

        n_valid = valid.sum().item()
        total_queries += n_valid

        for idx, learner_name in enumerate(LEARNERS.keys()):
            wins = (oracle_idx == idx)
            n_wins = wins.sum().item()
            winner_counts[learner_name] += n_wins
            winner_by_position[learner_name][i] += n_wins
        count += 1
    
    if count == 0:
        raise ValueError("No valid queries were found for this task; check query_mask or min_context")

    mean_metrics = {k: v / count for k, v in totals.items()}
    winner_fractions = {
        k: (winner_counts[k] / total_queries) if total_queries > 0 else 0.0
        for k in LEARNERS.keys()
    }

    return {
        "mean_metrics": mean_metrics,
        "winner_counts": winner_counts,
        "winner_fractions": winner_fractions,
        "winner_by_position": winner_by_position,
    }


def print_results(
    results: Dict[str, Dict[str, object]],
    log_stdout: bool = True,
    output_path: str = "outputs/synthetic_alignment_out.txt",
    section_title: str | None = None,
    append: bool = False,
) -> None:
    """Write experiment results to disk and optionally log to stdout."""

    cols = list(LEARNERS.keys()) + ["uniform_mix", "oracle"]

    sections: List[str] = []
    if section_title:
        sections.append(section_title)

    header = f"{'task family':<24}" + "".join([f"{c:>16}" for c in cols]) + f"{'winner':>16}"
    table_lines = ["Evaluation results", header, "-" * len(header)]
    for task_name, vals in results.items():
        metrics = vals["mean_metrics"]
        fixed_winner = min(LEARNERS.keys(), key=lambda x: metrics[x])
        row = f"{task_name:<24}"
        for c in cols:
            row += f"{metrics[c]:>16.6f}"
        row += f"{fixed_winner:>16}"
        table_lines.append(row)
    sections.append("\n".join(table_lines))

    gap_lines = ["Oracle gap analysis", "-------------------"]
    for task_name, vals in results.items():
        metrics = vals["mean_metrics"]
        best_fixed = min(metrics[l] for l in LEARNERS.keys())
        oracle = metrics["oracle"]
        uniform = metrics["uniform_mix"]
        oracle_gap_abs = best_fixed - oracle
        oracle_gap_rel = oracle_gap_abs / best_fixed if best_fixed > 0 else 0.0
        uniform_gap_abs = best_fixed - uniform
        uniform_gap_rel = uniform_gap_abs / best_fixed if best_fixed > 0 else 0.0
        gap_lines.append(
            f"{task_name:<24} "
            f"best_fixed={best_fixed:.6f}  "
            f"oracle={oracle:.6f}  "
            f"oracle_gain={oracle_gap_abs:.6f} ({100*oracle_gap_rel:.2f}%)  "
            f"uniform_mix={uniform:.6f}  "
            f"uniform_gap={uniform_gap_abs:.6f} ({100*uniform_gap_rel:.2f}%)"
        )
    sections.append("\n".join(gap_lines))

    winner_header = f"{'task family':<24}" + "".join([f"{l:>16}" for l in LEARNERS.keys()])
    winner_lines = ["Oracle winner fractions", "-----------------------", winner_header, "-" * len(winner_header)]
    for task_name, vals in results.items():
        row = f"{task_name:<24}"
        fracs = vals["winner_fractions"]
        for learner in LEARNERS.keys():
            row += f"{fracs[learner]:>16.3f}"
        winner_lines.append(row)
    sections.append("\n".join(winner_lines))

    report_text = "\n\n".join(sections)

    output_file = Path(output_path)
    if not output_file.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        output_file = project_root / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    prefix = ""
    if append and output_file.exists():
        prefix = "\n\n"

    with output_file.open(mode) as fh:
        fh.write(prefix + report_text + "\n")

    if log_stdout:
        print(report_text)


def run_all(cfg: EvalConfig) -> Dict[str, Dict]:
    results = {}
    for task_name in TASK_REPO.names():
        results[task_name] = evaluate_task(task_name, cfg)
    return results


def run_all_multiseed(cfg: EvalConfig, seeds: List[int]) -> Dict[str, Dict[str, object]]:
    if not seeds:
        raise ValueError("run_all_multiseed requires at least one seed")

    cols = list(LEARNERS.keys()) + ["uniform_mix", "oracle"]
    accumulated_results = {
        task_name: {
            "mean_metrics": {c: 0.0 for c in cols},
            "winner_fractions": {learner: 0.0 for learner in LEARNERS.keys()},
        }
        for task_name in TASK_REPO.names()
    }

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        results = run_all(cfg)
        for task_name, vals in results.items():
            for c in cols:
                accumulated_results[task_name]["mean_metrics"][c] += vals["mean_metrics"][c]
            for learner in LEARNERS.keys():
                accumulated_results[task_name]["winner_fractions"][learner] += vals["winner_fractions"][learner]

    n = len(seeds)
    averaged_results: Dict[str, Dict[str, object]] = {}
    for task_name, vals in accumulated_results.items():
        mean_metrics = {c: v / n for c, v in vals["mean_metrics"].items()}
        winner_fractions = {
            learner: vals["winner_fractions"][learner] / n
            for learner in LEARNERS.keys()
        }
        averaged_results[task_name] = {
            "mean_metrics": mean_metrics,
            "winner_fractions": winner_fractions,
        }
    return averaged_results


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

    log_stdout = True
    single = run_all(cfg)
    print_results(
        single,
        log_stdout=log_stdout,
        section_title="=== Single-seed run ===",
        append=False,
    )

    multiseed_list = [0, 1, 2, 3, 4]
    multi = run_all_multiseed(cfg, seeds=multiseed_list)
    seed_desc = ", ".join(str(seed) for seed in multiseed_list)
    print_results(
        multi,
        log_stdout=log_stdout,
        section_title=f"=== Multi-seed average (seeds: {seed_desc}) ===",
        append=True,
    )
