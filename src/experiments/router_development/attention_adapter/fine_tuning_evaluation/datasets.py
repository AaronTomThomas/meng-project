from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from experiments.router_development.attention_adapter.fine_tuning_evaluation.config import FineTuneEvalConfig
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import TaskSpec


E2E_RAW_URLS = {
    "train": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/trainset.csv",
    "validation": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/devset.csv",
    "test": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/testset_w_refs.csv",
}


@dataclass
class LoadedTaskData:
    train: Dataset | None
    val: Dataset | None
    test: Dataset | None
    split_names: dict[str, str | None]


def _limit(ds: Dataset | None, limit: int | None) -> Dataset | None:
    if ds is None or limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def _has_usable_labels(ds: Dataset, task: TaskSpec) -> bool:
    if task.name == "sst2":
        return "label" in ds.column_names and any(int(x) >= 0 for x in ds["label"][: min(16, len(ds))])
    if task.name == "boolq":
        return "answer" in ds.column_names
    return True


def load_task_data(task: TaskSpec, cfg: FineTuneEvalConfig) -> LoadedTaskData:
    if task.name == "e2e_nlg":
        raw = load_dataset("csv", data_files=E2E_RAW_URLS)
    else:
        raw: DatasetDict = (
            load_dataset(task.dataset_name, task.dataset_config)
            if task.dataset_config
            else load_dataset(task.dataset_name)
        )
    train = raw[task.train_split] if task.train_split in raw else None
    val = raw[task.val_split] if task.val_split in raw else None
    test_name = task.test_split
    test = raw[test_name] if test_name and test_name in raw else None
    if test is None or not _has_usable_labels(test, task):
        test = val
        test_name = task.val_split
    return LoadedTaskData(
        train=_limit(train, cfg.max_train_examples),
        val=_limit(val, cfg.max_val_examples),
        test=_limit(test, cfg.max_test_examples),
        split_names={"train": task.train_split if train is not None else None, "val": task.val_split if val is not None else None, "test": test_name},
    )
