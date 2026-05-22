from __future__ import annotations

import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.router_development.attention_adapter.config import (
    AKAZAFreeZConfig,
    AdapterMethod,
    LoRAFineTuneConfig,
    ReFTFineTuneConfig,
)
from experiments.router_development.attention_adapter.fine_tuning_evaluation.collators import CausalLMCollator
from experiments.router_development.attention_adapter.fine_tuning_evaluation.config import FineTuneEvalConfig
from experiments.router_development.attention_adapter.fine_tuning_evaluation.datasets import load_task_data
from experiments.router_development.attention_adapter.fine_tuning_evaluation.evaluate import (
    evaluate_loss,
    generate_predictions,
    masked_lm_loss,
    model_logits,
    score_candidates,
)
from experiments.router_development.attention_adapter.fine_tuning_evaluation.io import append_jsonl, write_json, write_jsonl
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import get_task
from experiments.router_development.attention_adapter.models import DEFAULT_FAMILY_SPECS
from experiments.router_development.attention_adapter.peft_factory import build_wrapped_model
from experiments.router_development.attention_adapter.trainer import TrainableParameters, infer_num_layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_model_family(model_name_or_path: str) -> str:
    lower = model_name_or_path.lower()
    if "pythia" in lower or "gpt-neox" in lower:
        return "pythia"
    return "gpt2"


def _filtered_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def adapter_config_from_eval_config(cfg: FineTuneEvalConfig, model_family: str, layer_indices: str):
    method_name = "loreft" if cfg.method == "reft" else cfg.method
    method = AdapterMethod(method_name)
    defaults = DEFAULT_FAMILY_SPECS[model_family]
    common = {
        "method": method,
        "model_family": model_family,
        "model_name": cfg.model_name_or_path,
        "layer_indices": layer_indices,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "eval_every": cfg.eval_every,
        "seed": cfg.seed,
        "device": cfg.device,
        "output_path": str(Path(cfg.output_dir) / "best_checkpoint.pt"),
    }
    if method is AdapterMethod.AKAZA_FREEZ:
        return AKAZAFreeZConfig(
            **_filtered_kwargs(
                AKAZAFreeZConfig,
                {
                    **common,
                    "bottleneck_dim": cfg.bottleneck_dim or defaults.default_bottleneck_dim,
                    "adapter_dropout": cfg.adapter_dropout,
                    "output_scale": cfg.output_scale if cfg.output_scale is not None else defaults.default_output_scale,
                },
            )
        )
    if method is AdapterMethod.LORA:
        return LoRAFineTuneConfig(
            **_filtered_kwargs(
                LoRAFineTuneConfig,
                {
                    **common,
                    "peft_target_profile": cfg.peft_target_profile or defaults.default_peft_target_profile,
                    "lora_rank": cfg.lora_rank,
                    "lora_alpha": cfg.lora_alpha,
                    "lora_dropout": cfg.lora_dropout,
                    "lora_bias": cfg.lora_bias,
                },
            )
        )
    return ReFTFineTuneConfig(
        **_filtered_kwargs(
            ReFTFineTuneConfig,
            {
                **common,
                "reft_rank": cfg.reft_rank,
                "reft_dropout": cfg.reft_dropout,
                "reft_output_scale": cfg.reft_output_scale,
                "reft_position_mode": cfg.reft_position_mode,
                "reft_prefix_positions": cfg.reft_prefix_positions,
                "reft_suffix_positions": cfg.reft_suffix_positions,
            },
        )
    )


def build_model(cfg: FineTuneEvalConfig, device: torch.device) -> tuple[torch.nn.Module, str, list[int]]:
    model_family = cfg.model_family or infer_model_family(cfg.model_name_or_path)
    if model_family not in DEFAULT_FAMILY_SPECS:
        raise ValueError(f"Unknown model_family={model_family!r}; choices={sorted(DEFAULT_FAMILY_SPECS)}")
    defaults = DEFAULT_FAMILY_SPECS[model_family]
    layer_indices_text = cfg.layer_indices or defaults.default_layer_indices
    layer_indices = sorted(int(x.strip()) for x in layer_indices_text.split(",") if x.strip())
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path).to(device)
    if cfg.method == "base":
        for param in base.parameters():
            param.requires_grad_(True)
        return base, model_family, []
    n_layers = infer_num_layers(base, model_family)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")
    for param in base.parameters():
        param.requires_grad_(False)
    adapter_cfg = adapter_config_from_eval_config(cfg, model_family, layer_indices_text)
    return build_wrapped_model(model=base, cfg=adapter_cfg, layer_indices=layer_indices).to(device), model_family, layer_indices


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_accumulation_steps: int,
    grad_clip: float,
    epoch: int, 
) -> float:
    if hasattr(model, "set_peft_train_mode"):
        model.set_peft_train_mode()
    else:
        model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_tokens = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_sum, tokens = masked_lm_loss(model_logits(model, input_ids), labels)
        loss = loss_sum / max(1, tokens) / gradient_accumulation_steps
        loss.backward()
        if step % gradient_accumulation_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss_sum.item())
        total_tokens += tokens
        #print  loss every 500 steps
        if step % 500 == 0:
            print(f"[Epoch {epoch}] Step {step}/ {len(loader)}: loss = {total_loss / max(1, total_tokens)}")
    if len(loader) % gradient_accumulation_steps != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, total_tokens)


def save_checkpoint(path: Path, cfg: FineTuneEvalConfig, model: torch.nn.Module, summary: dict[str, Any]) -> None:
    scope = TrainableParameters(params=[], frozen_before_training={}, check_frozen=False)
    torch.save({"summary": summary, "config": cfg.to_json_dict(), "trainable_state_dict": scope.trainable_state_dict(model)}, path)


def load_checkpoint(path: Path, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload["trainable_state_dict"] if "trainable_state_dict" in payload else payload
    TrainableParameters(params=[], frozen_before_training={}, check_frozen=False).load_trainable_state_dict(model, state)


def run(cfg: FineTuneEvalConfig) -> dict[str, Any]:
    if not cfg.do_train and not cfg.do_eval:
        cfg.do_eval = True
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("run()")
    write_json(output_dir / "config.json", cfg.to_json_dict())

    task = get_task(cfg.task)
    data = load_task_data(task, cfg)
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, model_family, layer_indices = build_model(cfg, device)
    counts = parameter_counts(model)
    if cfg.checkpoint_path:
        load_checkpoint(Path(cfg.checkpoint_path), model)

    print("model built")
    train_loader = None
    if data.train is not None:
        train_loader = DataLoader(
            data.train,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=CausalLMCollator(tokenizer, task, cfg.max_length, cfg.target_max_length),
        )

    print("data loaders built")
    best_val_loss = float("inf")
    best_epoch = 0
    bad_evals = 0
    history_path = output_dir / "training_log.jsonl"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"


    # pre-fine tuning evaluation
    if cfg.do_eval and data.val is not None:
        val_metrics = evaluate_loss(
            model,
            data.val,
            tokenizer,
            task,
            batch_size=cfg.eval_batch_size,
            max_length=cfg.max_length,
            target_max_length=cfg.target_max_length,
            device=device,
        )
        print(f"Pre-training validation loss = {val_metrics['loss']}")
        write_json(output_dir / "pretrain_val_metrics.json", val_metrics)
    if cfg.do_train:
        if train_loader is None:
            raise ValueError(f"Task {task.name!r} has no train split")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("beginning training")
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                gradient_accumulation_steps=max(1, cfg.gradient_accumulation_steps),
                grad_clip=cfg.grad_clip,
                epoch=epoch,
            )
            print(f"Epoch {epoch}: Training loss = {train_loss}")
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                if data.val is not None:
                    val_metrics = evaluate_loss(
                        model,
                        data.val,
                        tokenizer,
                        task,
                        batch_size=cfg.eval_batch_size,
                        max_length=cfg.max_length,
                        target_max_length=cfg.target_max_length,
                        device=device,
                    )
                    row.update({f"val_{key}": value for key, value in val_metrics.items()})
                    val_loss = float(val_metrics["loss"])
                    print(f"Epoch {epoch}: Validation loss = {val_loss}")
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        bad_evals = 0
                        save_checkpoint(best_checkpoint_path, cfg, model, {"epoch": epoch, "val_loss": val_loss})
                    else:
                        bad_evals += 1
                append_jsonl(history_path, [row])
                if bad_evals >= cfg.patience:
                    break
            else:
                append_jsonl(history_path, [row])
        if best_checkpoint_path.exists():
            load_checkpoint(best_checkpoint_path, model)

    metrics: dict[str, Any] = {
        "task": task.name,
        "method": cfg.method,
        "model_family": model_family,
        "layer_indices": layer_indices,
        "split_names": data.split_names,
        **counts,
        "best_validation_loss": None if best_val_loss == float("inf") else best_val_loss,
        "best_epoch": best_epoch,
    }

    prediction_rows: list[dict[str, Any]] = []
    if cfg.do_eval:
        for split_key, split_data in (("validation", data.val), ("test", data.test)):
            if split_data is None:
                continue
            loss_metrics = evaluate_loss(
                model,
                split_data,
                tokenizer,
                task,
                batch_size=cfg.eval_batch_size,
                max_length=cfg.max_length,
                target_max_length=cfg.target_max_length,
                device=device,
            )
            metrics.update({f"{split_key}_{key}": value for key, value in loss_metrics.items()})
            if task.task_type == "classification":
                class_metrics, rows = score_candidates(
                    model,
                    split_data,
                    tokenizer,
                    task,
                    max_length=cfg.max_length,
                    target_max_length=cfg.target_max_length,
                    device=device,
                )
                metrics.update({f"{split_key}_{key}": value for key, value in class_metrics.items()})
                if split_key == "test":
                    prediction_rows = rows
            elif cfg.generate or split_key == "test":
                gen_metrics, rows = generate_predictions(
                    model,
                    split_data,
                    tokenizer,
                    task,
                    max_length=cfg.max_length,
                    target_max_length=cfg.target_max_length,
                    device=device,
                )
                metrics.update({f"{split_key}_{key}": value for key, value in gen_metrics.items()})
                if split_key == "test":
                    prediction_rows = rows
    write_json(output_dir / "metrics.json", metrics)
    if prediction_rows:
        write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    if cfg.do_train and not best_checkpoint_path.exists():
        save_checkpoint(best_checkpoint_path, cfg, model, metrics)
    return metrics
