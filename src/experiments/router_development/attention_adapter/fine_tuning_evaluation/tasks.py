from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


Formatter = Callable[[Mapping[str, Any]], tuple[str, str | None]]


@dataclass(frozen=True)
class GlueTaskSpec:
    name: str
    glue_dir_name: str
    train_file: str
    validation_file: str
    test_file: str
    main_metric: str
    selection_metric: str
    selection_mode: str
    formatter: Formatter
    candidates: dict[str, str]
    submission_name: str
    submission_labels: dict[str, str]
    add_eos_to_target: bool = False
    score_normalization: str = "mean_token_logprob"
    evaluation_protocol: str = "decoder_lm_verbalized_classification"
    prompt_template: str = "Sentence: {sentence}\\nSentiment:"


def _sst2(example: Mapping[str, Any]) -> tuple[str, str | None]:
    prompt = f"Sentence: {example['sentence']}\nSentiment:"

    if "label" not in example or example["label"] is None:
        return prompt, None

    label = int(example["label"])
    if label < 0:
        return prompt, None

    return prompt, {0: " negative", 1: " positive"}.get(label)


def _rte(example: Mapping[str, Any]) -> tuple[str, str | None]:
    prompt = f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}\nEntailment:"

    label = example.get("label")
    if label is None:
        return prompt, None

    label_text = str(label).strip()
    if label_text in {"", "-1"}:
        return prompt, None

    return prompt, {
        "0": " entailment",
        "1": " not_entailment",
        "entailment": " entailment",
        "not_entailment": " not_entailment",
    }.get(label_text)


TASKS: dict[str, GlueTaskSpec] = {
    "sst2": GlueTaskSpec(
        name="sst2",
        glue_dir_name="SST-2",
        train_file="train.tsv",
        validation_file="dev.tsv",
        test_file="test.tsv",
        main_metric="accuracy",
        selection_metric="accuracy",
        selection_mode="max",
        formatter=_sst2,
        candidates={"negative": " negative", "positive": " positive"},
        submission_name="SST-2",
        submission_labels={"negative": "negative", "positive": "positive"},
        add_eos_to_target=False,
        score_normalization="mean_token_logprob",
    ),
    "rte": GlueTaskSpec(
        name="rte",
        glue_dir_name="RTE",
        train_file="train.tsv",
        validation_file="dev.tsv",
        test_file="test.tsv",
        main_metric="accuracy",
        selection_metric="accuracy",
        selection_mode="max",
        formatter=_rte,
        candidates={"entailment": " entailment", "not_entailment": " not_entailment"},
        submission_name="RTE",
        submission_labels={"entailment": "entailment", "not_entailment": "not_entailment"},
        add_eos_to_target=False,
        score_normalization="mean_token_logprob",
        prompt_template="Premise: {sentence1}\\nHypothesis: {sentence2}\\nEntailment:",
    ),
}


def get_task(name: str) -> GlueTaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task={name!r}; choices={sorted(TASKS)}") from exc


def format_example(task: GlueTaskSpec, example: Mapping[str, Any]) -> dict[str, Any]:
    prompt, target = task.formatter(example)
    return {"prompt": prompt, "target": target}
