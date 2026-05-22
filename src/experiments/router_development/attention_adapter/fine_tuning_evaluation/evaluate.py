from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from experiments.router_development.attention_adapter.fine_tuning_evaluation.collators import CausalLMCollator
from experiments.router_development.attention_adapter.fine_tuning_evaluation.formatting import format_example
from experiments.router_development.attention_adapter.fine_tuning_evaluation.metrics import accuracy, bleu_1_to_4, rouge_l
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import TaskSpec


def model_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    return output.logits if hasattr(output, "logits") else output


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_count = int((shift_labels != -100).sum().item())
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, token_count


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: Iterable[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    task: TaskSpec,
    *,
    batch_size: int,
    max_length: int,
    target_max_length: int,
    device: torch.device,
) -> dict[str, float]:
    if hasattr(model, "set_peft_eval_mode"):
        model.set_peft_eval_mode()
    else:
        model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer, task, max_length, target_max_length),
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
    task: TaskSpec,
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
    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        row = format_example(task, example)
        prompt = row["prompt"]
        target = row["target"]
        scores: dict[str, float] = {}
        for label_name, candidate in task.candidates.items():
            candidate_ids = tokenizer(candidate, add_special_tokens=False, truncation=True, max_length=target_max_length).input_ids
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_ids = prompt_ids[-max(1, max_length - len(candidate_ids)) :]
            ids = prompt_ids + candidate_ids
            labels = [-100] * len(prompt_ids) + candidate_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
            loss, token_count = masked_lm_loss(model_logits(model, input_ids), label_tensor)
            scores[label_name] = -float(loss.item()) / max(1, token_count)
        prediction = max(scores, key=scores.get)
        reference = next((name for name, text in task.candidates.items() if text == target), target.strip())
        predictions.append(prediction)
        references.append(reference)
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "target": target,
                "candidate_logprobs_per_token": scores,
            }
        )
    return {"accuracy": accuracy(predictions, references)}, rows


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    dataset: Iterable[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    task: TaskSpec,
    *,
    max_length: int,
    target_max_length: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if hasattr(model, "set_peft_eval_mode"):
        model.set_peft_eval_mode()
    else:
        model.eval()
    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict[str, Any]] = []
    eos = tokenizer.eos_token_id
    for idx, example in enumerate(dataset):
        row = format_example(task, example)
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False, truncation=True, max_length=max_length).input_ids
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated: list[int] = []
        for _ in range(target_max_length):
            logits = model_logits(model, input_ids)[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            if eos is not None and next_id == eos:
                break
            generated.append(next_id)
            if input_ids.shape[1] + 1 > max_length:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        reference = row["target"].strip()
        predictions.append(prediction)
        references.append(reference)
        rows.append({"idx": idx, "prompt": row["prompt"], "prediction": prediction, "reference": reference})
    metrics = {"bleu": bleu_1_to_4(predictions, references), "rouge_l": rouge_l(predictions, references)}
    return metrics, rows

