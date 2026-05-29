from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FAIRNESS_CONFIG_KEYS = (
    "model_name_or_path",
    "task",
    "max_length",
    "target_max_length",
    "epochs",
    "lr",
    "weight_decay",
    "batch_size",
    "gradient_accumulation_steps",
    "eval_batch_size",
    "selection_split_from_train",
)


def load_metrics(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["_metrics_path"] = str(path)
        rows.append(payload)
    return rows


def _nested(payload: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _fairness_signature(row: dict[str, Any]) -> dict[str, Any]:
    config = _nested(row, ("manifest", "config"), {})
    training = _nested(row, ("manifest", "training"), {})
    model = _nested(row, ("manifest", "model"), {})
    splits = _nested(row, ("manifest", "splits"), {})
    method = _nested(row, ("manifest", "model", "method"), row.get("method"))
    signature = {key: config.get(key) for key in FAIRNESS_CONFIG_KEYS}
    signature["model_family"] = model.get("model_family")
    if method != "full_finetune":
        signature["layer_indices"] = model.get("layer_indices")
    signature["effective_batch_size"] = training.get("effective_batch_size")
    signature["splits"] = splits
    return signature


def validate_fairness(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task"))
        model_name = str(_nested(row, ("manifest", "config", "model_name_or_path"), ""))
        comparison_group = "full_finetune_baseline" if row.get("method") == "full_finetune" else "peft_or_zero_shot"
        grouped[(task, model_name, comparison_group)].append(row)

    for (task, model_name, comparison_group), group in grouped.items():
        reference = _fairness_signature(group[0])
        mismatches = []
        for row in group[1:]:
            current = _fairness_signature(row)
            if current != reference:
                mismatches.append((row.get("_metrics_path"), current))
        if mismatches:
            paths = [str(path) for path, _ in mismatches[:5]]
            raise ValueError(
                f"Fairness check failed for task={task!r}, model={model_name!r}; "
                f"group={comparison_group!r}; "
                f"mismatched metrics files={paths}"
            )


def comparison_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in metrics_rows:
        task_meta = _nested(payload, ("manifest", "task"), {})
        main_metric = str(task_meta.get("main_metric") or payload.get("selection_metric") or "loss")
        metric_key = f"validation_{main_metric}"
        value = payload.get(metric_key)
        rows.append(
            {
                "task": payload.get("task"),
                "method": payload.get("method"),
                "seed": _nested(payload, ("manifest", "training", "seed")),
                "model_name_or_path": _nested(payload, ("manifest", "config", "model_name_or_path")),
                "report_split": "validation",
                "main_metric": main_metric,
                "main_metric_value": value,
                "trainable_parameters": payload.get("trainable_parameters"),
                "total_parameters": payload.get("total_parameters"),
                "best_epoch": payload.get("best_epoch"),
                "selection_metric": payload.get("selection_metric"),
                "best_selection_score": payload.get("best_selection_score"),
                "metrics_path": payload.get("_metrics_path"),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["task"], row["method"], row["model_name_or_path"], row["report_split"], row["main_metric"])
        grouped[key].append(row)

    aggregates = []
    for key, group in sorted(grouped.items()):
        values = [float(row["main_metric_value"]) for row in group if row["main_metric_value"] is not None]
        mean = sum(values) / len(values) if values else None
        std = None
        stderr = None
        if len(values) > 1 and mean is not None:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = math.sqrt(variance)
            stderr = std / math.sqrt(len(values))
        aggregates.append(
            {
                "task": key[0],
                "method": key[1],
                "model_name_or_path": key[2],
                "report_split": key[3],
                "main_metric": key[4],
                "n": len(values),
                "mean": mean,
                "std": std,
                "stderr": stderr,
                "seeds": ",".join(str(row["seed"]) for row in group),
                "trainable_parameters": group[0]["trainable_parameters"],
            }
        )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate fine-tuning evaluation metrics across methods/seeds.")
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--aggregate_json", type=Path, default=None)
    parser.add_argument("--allow_mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics_rows = load_metrics(args.metrics)
    if not args.allow_mismatch:
        validate_fairness(metrics_rows)
    rows = comparison_rows(metrics_rows)
    aggregates = aggregate_rows(rows)
    write_csv(args.output_csv, aggregates)
    if args.aggregate_json is not None:
        args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_json.write_text(json.dumps({"runs": rows, "aggregates": aggregates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
