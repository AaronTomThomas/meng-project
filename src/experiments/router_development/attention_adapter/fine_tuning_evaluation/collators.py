from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from experiments.router_development.attention_adapter.fine_tuning_evaluation.formatting import format_example
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import TaskSpec


@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    task: TaskSpec
    max_length: int
    target_max_length: int

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [format_example(self.task, example) for example in examples]
        encoded = [self._encode(row["prompt"], row["target"]) for row in rows]
        max_len = min(self.max_length, max(len(x["input_ids"]) for x in encoded))
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels = [], []
        for item in encoded:
            ids = item["input_ids"][:max_len]
            labs = item["labels"][:max_len]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _encode(self, prompt: str, target: str) -> dict[str, list[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False, truncation=True, max_length=self.target_max_length).input_ids
        available_prompt = max(1, self.max_length - len(target_ids))
        prompt_ids = prompt_ids[-available_prompt:]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if self.tokenizer.eos_token_id is not None and len(ids) < self.max_length:
            ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
        return {"input_ids": ids, "labels": labels}

