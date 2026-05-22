from __future__ import annotations

from typing import Any, Mapping

from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import TaskSpec


def format_example(task: TaskSpec, example: Mapping[str, Any]) -> dict[str, Any]:
    prompt, target = task.formatter(example)
    return {"prompt": prompt, "target": target}

