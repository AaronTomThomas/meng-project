from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from experiments.router_development.attention_adapter.fine_tuning_evaluation.config import FineTuneEvalConfig
from experiments.router_development.attention_adapter.fine_tuning_evaluation.download_glue_data import ensure_sst2
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import GlueTaskSpec


@dataclass
class LoadedTaskData:
    train: Dataset | None
    val: Dataset | None
    report_val: Dataset | None
    test: Dataset | None
    split_names: dict[str, str | None]
    split_details: dict[str, dict[str, object]]

def _limit(ds: Dataset | None, limit: int | None) -> Dataset | None:
    if ds is None or limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def _split_selection_from_train(
    train: Dataset | None,
    *,
    fraction: float,
    seed: int,
) -> tuple[Dataset | None, Dataset | None]:
    if train is None or fraction <= 0.0:
        return train, None
    if fraction >= 1.0:
        raise ValueError("selection_split_from_train must be < 1.0")
    if len(train) < 2:
        raise ValueError("selection_split_from_train requires at least two train examples")
    selection_count = max(1, int(round(len(train) * fraction)))
    selection_count = min(selection_count, len(train) - 1)
    shuffled = train.shuffle(seed=seed)
    selection = shuffled.select(range(selection_count))
    remaining_train = shuffled.select(range(selection_count, len(shuffled)))
    return remaining_train, selection


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(text)


def _load_sst2_tsv(path: Path, *, has_labels: bool) -> Dataset:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            example: dict[str, object] = {"sentence": row["sentence"]}
            official_index = _parse_optional_int(row.get("index"))
            example["idx"] = row_idx if official_index is None else official_index
            if official_index is not None:
                example["index"] = official_index
            if has_labels:
                example["label"] = _parse_optional_int(row.get("label"))
            rows.append(example)
    return Dataset.from_list(rows)


def _load_local_glue_task(data_dir: str | Path, task: GlueTaskSpec) -> dict[str, Dataset]:
    if task.name != "sst2":
        raise ValueError(f"Unsupported GLUE task={task.name!r}; only 'sst2' is active")
    task_dir = ensure_sst2(data_dir)
    return {
        "train": _load_sst2_tsv(task_dir / task.train_file, has_labels=True),
        "validation": _load_sst2_tsv(task_dir / task.validation_file, has_labels=True),
        "test": _load_sst2_tsv(task_dir / task.test_file, has_labels=False),
    }


def load_task_data(task: GlueTaskSpec, cfg: FineTuneEvalConfig) -> LoadedTaskData:
    if cfg.selection_split_from_train < 0.0 or cfg.selection_split_from_train >= 1.0:
        raise ValueError("selection_split_from_train must be in [0.0, 1.0)")
    if task.name != "sst2":
        raise ValueError(f"Unsupported task={task.name!r}; only 'sst2' is active")
    raw = _load_local_glue_task(cfg.glue_data_dir, task)
    train = raw["train"]
    report_val = raw["validation"]
    selection_val = report_val
    train, train_selection = _split_selection_from_train(
        train,
        fraction=cfg.selection_split_from_train,
        seed=cfg.selection_split_seed if cfg.selection_split_seed is not None else cfg.seed,
    )
    if train_selection is not None:
        selection_val = train_selection
    test = raw["test"]
    train = _limit(train, cfg.max_train_examples)
    selection_val = _limit(selection_val, cfg.max_val_examples)
    report_val = _limit(report_val, cfg.max_val_examples)
    test = _limit(test, cfg.max_test_examples)
    split_names = {
        "train": task.train_file,
        "selection": "train_selection" if train_selection is not None else task.validation_file,
        "val": task.validation_file,
        "test": task.test_file,
    }
    split_details = {
        "train": {
            "source_split": split_names["train"],
            "num_examples": len(train),
            "has_labels": True,
        },
        "selection": {
            "source_split": split_names["selection"],
            "num_examples": len(selection_val),
            "has_labels": True,
            "is_train_derived": train_selection is not None,
            "selection_fraction_from_train": cfg.selection_split_from_train,
        },
        "validation": {
            "source_split": split_names["val"],
            "num_examples": len(report_val),
            "has_labels": True,
        },
        "test": {
            "source_split": split_names["test"],
            "num_examples": len(test),
            "has_labels": False,
            "source_kind": "official",
        },
    }
    return LoadedTaskData(
        train=train,
        val=selection_val,
        report_val=report_val,
        test=test,
        split_names=split_names,
        split_details=split_details,
    )
