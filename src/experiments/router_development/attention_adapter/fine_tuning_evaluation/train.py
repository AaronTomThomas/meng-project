from __future__ import annotations

import csv
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.router_development.attention_adapter.config import (
    config_from_values,
)
from experiments.router_development.attention_adapter.fine_tuning_evaluation.config import FineTuneEvalConfig
from experiments.router_development.attention_adapter.fine_tuning_evaluation.datasets import load_task_data
from experiments.router_development.attention_adapter.fine_tuning_evaluation.evaluate import (
    _GlueVerbalizerCollator,
    evaluate_loss,
    score_candidates,
)
from experiments.router_development.attention_adapter.fine_tuning_evaluation.tasks import get_task
from experiments.router_development.attention_adapter.models import DEFAULT_FAMILY_SPECS
from experiments.router_development.attention_adapter.trainer import TrainableParameters, build_adapter_model
from experiments.router_development.attention_adapter.utils import (
    append_jsonl,
    infer_model_family,
    masked_lm_loss,
    model_logits,
    parameter_counts,
    set_seed,
    write_json,
    write_jsonl,
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, object]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "is_dirty": bool(status),
        "status_short": status,
    }


def _selection_is_better(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return value > best + 1e-12
    if mode == "min":
        return value < best - 1e-12
    raise ValueError(f"selection_mode must be 'min' or 'max', got {mode!r}")


def _evaluate_task_metrics(
    model: torch.nn.Module,
    dataset,
    tokenizer,
    task,
    cfg: FineTuneEvalConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metrics = evaluate_loss(
        model,
        dataset,
        tokenizer,
        task,
        batch_size=cfg.eval_batch_size,
        max_length=cfg.max_length,
        target_max_length=cfg.target_max_length,
        device=device,
    )
    class_metrics, rows = score_candidates(
        model,
        dataset,
        tokenizer,
        task,
        max_length=cfg.max_length,
        target_max_length=cfg.target_max_length,
        device=device,
    )
    metrics.update(class_metrics)
    return metrics, rows


def _candidate_tokenizations(tokenizer, task) -> dict[str, list[int]]:
    return {
        label: tokenizer(verbalizer, add_special_tokens=False).input_ids
        for label, verbalizer in task.candidates.items()
    }


def _build_manifest(
    cfg: FineTuneEvalConfig,
    *,
    task,
    data,
    tokenizer,
    model_family: str,
    layer_indices: list[int],
    parameter_counts_payload: dict[str, int],
) -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "git": _git_metadata(),
        "packages": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "peft": _package_version("peft"),
        },
        "model": {
            "model_name_or_path": cfg.model_name_or_path,
            "model_family": model_family,
            "layer_indices": layer_indices,
            "method": cfg.method,
            "parameter_counts": parameter_counts_payload,
        },
        "tokenizer": {
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "task": {
            "name": task.name,
            "dataset_name": "glue",
            "glue_dir_name": task.glue_dir_name,
            "train_file": task.train_file,
            "validation_file": task.validation_file,
            "test_file": task.test_file,
            "main_metric": task.main_metric,
            "selection_metric": task.selection_metric,
            "selection_mode": task.selection_mode,
            "evaluation_protocol": task.evaluation_protocol,
            "prompt_template": task.prompt_template,
            "candidate_verbalizers": task.candidates,
            "candidate_score_normalization": task.score_normalization,
            "target_eos": task.add_eos_to_target,
            "candidate_tokenizations": _candidate_tokenizations(tokenizer, task),
        },
        "splits": data.split_details,
        "training": {
            "seed": cfg.seed,
            "selection_split_seed": cfg.selection_split_seed if cfg.selection_split_seed is not None else cfg.seed,
            "selection_split_from_train": cfg.selection_split_from_train,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "effective_batch_size": cfg.batch_size * max(1, cfg.gradient_accumulation_steps),
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "max_length": cfg.max_length,
            "target_max_length": cfg.target_max_length,
        },
        "config": cfg.to_json_dict(),
    }


def adapter_config_from_eval_config(cfg: FineTuneEvalConfig, model_family: str, layer_indices: str):
    method_name = "loreft" if cfg.method == "reft" else cfg.method
    defaults = DEFAULT_FAMILY_SPECS[model_family]
    return config_from_values(
        method_name,
        model_family=model_family,
        model_name=cfg.model_name_or_path,
        layer_indices=layer_indices,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.patience,
        eval_every=cfg.eval_every,
        seed=cfg.seed,
        device=cfg.device,
        output_path=str(Path(cfg.output_dir) / "best_checkpoint.pt"),
        bottleneck_dim=cfg.bottleneck_dim or defaults.default_bottleneck_dim,
        adapter_dropout=cfg.adapter_dropout,
        output_scale=cfg.output_scale if cfg.output_scale is not None else defaults.default_output_scale,
        peft_target_profile=cfg.peft_target_profile or defaults.default_peft_target_profile,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_bias=cfg.lora_bias,
        reft_rank=cfg.reft_rank,
        reft_dropout=cfg.reft_dropout,
        reft_output_scale=cfg.reft_output_scale,
        reft_position_mode=cfg.reft_position_mode,
        reft_prefix_positions=cfg.reft_prefix_positions,
        reft_suffix_positions=cfg.reft_suffix_positions,
    )


def build_model(cfg: FineTuneEvalConfig, device: torch.device) -> tuple[torch.nn.Module, str, list[int]]:
    model_family = cfg.model_family or infer_model_family(cfg.model_name_or_path)
    if model_family not in DEFAULT_FAMILY_SPECS:
        raise ValueError(f"Unknown model_family={model_family!r}; choices={sorted(DEFAULT_FAMILY_SPECS)}")
    defaults = DEFAULT_FAMILY_SPECS[model_family]
    layer_indices_text = cfg.layer_indices or defaults.default_layer_indices
    if cfg.method in {"zero_shot", "full_finetune"}:
        base = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path).to(device)
        trainable = cfg.method == "full_finetune"
        for param in base.parameters():
            param.requires_grad_(trainable)
        return base, model_family, []
    adapter_cfg = adapter_config_from_eval_config(cfg, model_family, layer_indices_text)
    model, layer_indices = build_adapter_model(adapter_cfg, device)
    return model, model_family, layer_indices


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
        if step % 10 == 0:
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


def write_test_outputs(output_dir: Path, task, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "prediction"])
        writer.writeheader()
        writer.writerows({"idx": row["idx"], "prediction": row["prediction"]} for row in rows)
    with (submissions_dir / f"{task.submission_name}.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "prediction"])
        writer.writerows([row["idx"], task.submission_labels.get(row["prediction"], row["prediction"])] for row in rows)


def run(cfg: FineTuneEvalConfig) -> dict[str, Any]:
    if cfg.do_train or not cfg.do_eval:
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

    train_loader = DataLoader(
        data.train,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_GlueVerbalizerCollator(tokenizer, task, cfg.max_length, cfg.target_max_length),
    )

    print("data loaders built")
    best_val_loss = float("inf")
    best_epoch = 0
    best_selection_score: float | None = None
    bad_evals = 0
    history_path = output_dir / "training_log.jsonl"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    manifest = _build_manifest(
        cfg,
        task=task,
        data=data,
        tokenizer=tokenizer,
        model_family=model_family,
        layer_indices=layer_indices,
        parameter_counts_payload=counts,
    )
    write_json(output_dir / "manifest.json", manifest)
    if cfg.do_eval:
        val_metrics, _ = _evaluate_task_metrics(
            model,
            data.val,
            tokenizer,
            task,
            cfg,
            device=device,
        )
        print(f"Pre-training selection loss = {val_metrics['loss']}")
        write_json(output_dir / "pretrain_selection_metrics.json", val_metrics)
    if cfg.do_train:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("do_train=True requires a trainable method; use --method full_finetune or a PEFT method")
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
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
                val_metrics, _ = _evaluate_task_metrics(model, data.val, tokenizer, task, cfg, device=device)
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                val_loss = float(val_metrics["loss"])
                if task.selection_metric not in val_metrics:
                    raise ValueError(
                        f"selection_metric={task.selection_metric!r} was not produced for task={task.name!r}; "
                        f"available={sorted(val_metrics)}"
                    )
                selection_score = float(val_metrics[task.selection_metric])
                row.update(selection_metric=task.selection_metric, selection_mode=task.selection_mode, selection_score=selection_score)
                print(f"Epoch {epoch}: Selection loss = {val_loss}; {task.selection_metric} = {selection_score}")
                if _selection_is_better(selection_score, best_selection_score, task.selection_mode):
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_selection_score = selection_score
                    bad_evals = 0
                    save_checkpoint(
                        best_checkpoint_path,
                        cfg,
                        model,
                        {
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "selection_metric": task.selection_metric,
                            "selection_mode": task.selection_mode,
                            "selection_score": selection_score,
                        },
                    )
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
        "split_details": data.split_details,
        **counts,
        "best_validation_loss": None if best_val_loss == float("inf") else best_val_loss,
        "best_epoch": best_epoch,
        "selection_metric": task.selection_metric,
        "selection_mode": task.selection_mode,
        "best_selection_score": best_selection_score,
        "manifest": manifest,
    }

    prediction_rows: list[dict[str, Any]] = []
    if cfg.do_eval:
        eval_splits = [("validation", data.report_val), ("test", data.test)]
        selection_details = data.split_details.get("selection", {})
        if selection_details.get("is_train_derived"):
            eval_splits.insert(0, ("selection", data.val))
        for split_key, split_data in eval_splits:
            if split_data is None:
                continue
            loss_metrics, rows = _evaluate_task_metrics(model, split_data, tokenizer, task, cfg, device=device)
            metrics.update({f"{split_key}_{key}": value for key, value in loss_metrics.items()})
            if split_key == "test":
                prediction_rows = rows
    write_json(output_dir / "metrics.json", metrics)
    if prediction_rows:
        write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
        write_test_outputs(output_dir, task, prediction_rows)
    if cfg.do_train and not best_checkpoint_path.exists():
        save_checkpoint(best_checkpoint_path, cfg, model, metrics)
    return metrics
