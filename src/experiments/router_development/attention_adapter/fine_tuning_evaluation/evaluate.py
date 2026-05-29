from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import GlueTaskSpec, format_example
from experiments.router_development.attention_adapter.utils import masked_lm_loss, model_logits


def accuracy(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def _encode_prompt_target(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    target: str,
    *,
    max_length: int,
    target_max_length: int,
    add_eos: bool,
) -> dict[str, list[int]]:
    eos_budget = 1 if add_eos and tokenizer.eos_token_id is not None else 0
    target_budget = min(target_max_length, max(1, max_length - eos_budget))
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=target_budget,
    ).input_ids
    available_prompt = max(0, max_length - len(target_ids) - eos_budget)
    prompt_ids = prompt_ids[-available_prompt:] if available_prompt > 0 else []
    ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if add_eos and tokenizer.eos_token_id is not None and len(ids) < max_length:
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    return {"input_ids": ids, "labels": labels}


@dataclass
class _GlueVerbalizerCollator:
    tokenizer: PreTrainedTokenizerBase
    task: GlueTaskSpec
    max_length: int
    target_max_length: int

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [format_example(self.task, example) for example in examples]
        encoded = [
            _encode_prompt_target(
                self.tokenizer,
                row["prompt"],
                row["target"],
                max_length=self.max_length,
                target_max_length=self.target_max_length,
                add_eos=self.task.add_eos_to_target,
            )
            for row in rows
        ]
        max_len = min(self.max_length, max(len(item["input_ids"]) for item in encoded))
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


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: Iterable[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    task: GlueTaskSpec,
    *,
    batch_size: int,
    max_length: int,
    target_max_length: int,
    device: torch.device,
) -> dict[str, float]:
    if not all(format_example(task, example)["target"] is not None for example in dataset):
        return {"examples": float(len(dataset)) if hasattr(dataset, "__len__") else 0.0}
    if hasattr(model, "set_peft_eval_mode"):
        model.set_peft_eval_mode()
    else:
        model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_GlueVerbalizerCollator(tokenizer, task, max_length, target_max_length),
    )
    total_loss = 0.0
    total_tokens = 0
    total_examples = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, tokens = masked_lm_loss(model_logits(model, input_ids), labels)
        total_loss += float(loss.item())
        total_tokens += tokens
        total_examples += input_ids.shape[0]
    loss = total_loss / max(1, total_tokens)
    return {"loss": loss, "nll": loss, "ppl": float(math.exp(min(20.0, loss))), "examples": float(total_examples)}


@torch.no_grad()
def score_candidates(
    model: torch.nn.Module,
    dataset: Iterable[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    task: GlueTaskSpec,
    *,
    max_length: int,
    target_max_length: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not task.candidates:
        return {}, []
    if hasattr(model, "set_peft_eval_mode"):
        model.set_peft_eval_mode()
    else:
        model.eval()
    predictions: list[str] = []
    references: list[str] = []
    reference_predictions: list[str] = []
    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        row = format_example(task, example)
        example_idx = example.get("idx", example.get("index", idx))
        prompt = row["prompt"]
        target = row["target"]
        has_reference = target is not None
        scores: dict[str, float] = {}
        for label_name, candidate in task.candidates.items():
            encoded = _encode_prompt_target(
                tokenizer,
                prompt,
                candidate,
                max_length=max_length,
                target_max_length=target_max_length,
                add_eos=task.add_eos_to_target,
            )
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
            label_tensor = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
            loss_sum, token_count = masked_lm_loss(model_logits(model, input_ids), label_tensor)
            if task.score_normalization == "mean_token_logprob":
                scores[label_name] = -float(loss_sum.item()) / max(1, token_count)
            elif task.score_normalization == "sum_logprob":
                scores[label_name] = -float(loss_sum.item())
            else:
                raise ValueError(
                    f"Unknown score_normalization={task.score_normalization!r}; "
                    "expected 'mean_token_logprob' or 'sum_logprob'"
                )
        prediction = max(scores, key=scores.get)
        reference = (
            next((name for name, text in task.candidates.items() if text == target), target.strip())
            if has_reference
            else None
        )
        predictions.append(prediction)
        if reference is not None:
            references.append(reference)
            reference_predictions.append(prediction)
        rows.append(
            {
                "idx": example_idx,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "target": target,
                "candidate_logprobs_per_token": scores,
            }
        )
    if references:
        return {"accuracy": accuracy(reference_predictions, references)}, rows
    return {"predicted_examples": float(len(predictions))}, rows
