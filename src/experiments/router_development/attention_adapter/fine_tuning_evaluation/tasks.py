from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


Formatter = Callable[[Mapping[str, Any]], tuple[str, str]]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_name: str
    dataset_config: str | None
    train_split: str
    val_split: str
    test_split: str | None
    task_type: str
    main_metric: str
    formatter: Formatter
    candidates: dict[str, str] | None = None


def _sst2(example: Mapping[str, Any]) -> tuple[str, str]:
    prompt = f"Sentence: {example['sentence']}\nSentiment:"
    label = int(example["label"])
    return prompt, {0: " negative", 1: " positive"}[label]


def _boolq(example: Mapping[str, Any]) -> tuple[str, str]:
    prompt = f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
    return prompt, " yes" if bool(example["answer"]) else " no"


def _first_present(example: Mapping[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        if name in example and example[name] is not None:
            return str(example[name])
    raise KeyError(f"None of the expected fields are present: {names}; available={sorted(example)}")


def _e2e(example: Mapping[str, Any]) -> tuple[str, str]:
    mr = _first_present(example, ("meaning_representation", "meaning_representation_src", "mr"))
    ref = _first_present(example, ("human_reference", "reference", "target", "text", "ref"))
    return f"Meaning representation: {mr}\nDescription:", f" {ref.strip()}"


TASKS: dict[str, TaskSpec] = {
    "sst2": TaskSpec(
        name="sst2",
        dataset_name="nyu-mll/glue",
        dataset_config="sst2",
        train_split="train",
        val_split="validation",
        test_split="test",
        task_type="classification",
        main_metric="accuracy",
        formatter=_sst2,
        candidates={"negative": " negative", "positive": " positive"},
    ),
    "boolq": TaskSpec(
        name="boolq",
        dataset_name="google/boolq",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        test_split=None,
        task_type="classification",
        main_metric="accuracy",
        formatter=_boolq,
        candidates={"no": " no", "yes": " yes"},
    ),
    "e2e_nlg": TaskSpec(
        name="e2e_nlg",
        dataset_name="tuetschek/e2e_nlg",
        dataset_config=None,
        train_split="train",
        val_split="validation",
        test_split="test",
        task_type="generation",
        main_metric="loss",
        formatter=_e2e,
    ),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task={name!r}; choices={sorted(TASKS)}") from exc
