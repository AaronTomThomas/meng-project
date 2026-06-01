from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from experiments.attention_learners import LearnerHyperParams
from experiments.router_development.attention_adapter.adapters.site_ablation_adapters import (
    GPT2SiteAblationAdapter,
    INTERVENTION_SITES,
)
from experiments.router_development.attention_adapter.data import load_adapter_finetune_data
from experiments.router_development.attention_adapter.eval import eval_baseline, eval_wrapped
from experiments.router_development.attention_adapter.trainer import (
    TrainableParameters,
    infer_num_layers,
    parse_layer_indices,
    train_one_epoch,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SiteAblationFineTuneConfig(LearnerHyperParams):
    model_family: str = "gpt2"
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_revision: str | None = None
    text_field: str = "text"

    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

    block_size: int = 96
    batch_size: int = 4
    max_train_chunks: int = 2048
    max_val_chunks: int = 512
    max_test_chunks: int = 512

    layer_indices: str = "6,7,8,9,10,11"
    intervention_site: str = "z_pre_cproj"
    adapter_input: str = "ln1"
    detach_adapter_input: bool = True

    bottleneck_dim: int = 4
    adapter_dropout: float = 0.05
    output_scale: float = 0.05

    peft_l2: float = 1e-5
    peft_l1: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 500
    patience: int = 30
    eval_every: int = 1
    grad_clip: float = 1.0

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: str = "outputs/attention_adapter/site_ablation.pt"
    cache_dir: str = "outputs/attention_adapter/cache_gpt2/"
    skip_freeze_check: bool = False
    eval_test_during_training: bool = False

    split: str = "train"
    max_chunks: int = 2048


def relative_ppl_reduction(delta_nll: float) -> float:
    return float(1.0 - math.exp(-delta_nll))


def build_wrapped_model(
    *,
    cfg: SiteAblationFineTuneConfig,
    device: torch.device,
) -> tuple[GPT2SiteAblationAdapter, list[int]]:
    if cfg.model_family != "gpt2":
        raise ValueError("Site ablation currently supports only --model_family gpt2")
    if cfg.intervention_site not in INTERVENTION_SITES:
        raise ValueError(f"Unknown intervention_site={cfg.intervention_site!r}; choices={list(INTERVENTION_SITES)}")

    layer_indices = parse_layer_indices(cfg.layer_indices)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = infer_num_layers(model, cfg.model_family)
    print(f"[model] family={cfg.model_family} layers={n_layers}")
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")

    wrapped = GPT2SiteAblationAdapter(model=model, cfg=cfg, layer_indices=layer_indices).to(device)
    return wrapped, layer_indices


def train(cfg: SiteAblationFineTuneConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    layer_indices = parse_layer_indices(cfg.layer_indices)

    print("[config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print(f"  parsed_layer_indices: {layer_indices}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)

    data = load_adapter_finetune_data(cfg, tokenizer)
    baseline_train = eval_baseline(baseline_model, data.train, cfg.batch_size, device)
    baseline_val = eval_baseline(baseline_model, data.val, cfg.batch_size, device)
    baseline_test = eval_baseline(baseline_model, data.test, cfg.batch_size, device)
    del baseline_model

    wrapped, layer_indices = build_wrapped_model(cfg=cfg, device=device)
    trainable_params = [p for p in wrapped.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[model] trainable_params={num_trainable}")

    init_train = eval_wrapped(wrapped, data.train, cfg.batch_size, collect_stats=False)
    init_val = eval_wrapped(wrapped, data.val, cfg.batch_size, collect_stats=True)
    init_test = eval_wrapped(wrapped, data.test, cfg.batch_size, collect_stats=False)
    print()
    print(
        f"[baseline] train_loss={baseline_train:.6f} "
        f"val_loss={baseline_val:.6f} test_loss={baseline_test:.6f}"
    )
    print(
        f"[wrapped@init] train_loss={init_train['loss']:.6f} "
        f"val_loss={init_val['loss']:.6f} test_loss={init_test['loss']:.6f}"
    )
    print(
        f"[init_delta] train={init_train['loss'] - baseline_train:+.8f} "
        f"val={init_val['loss'] - baseline_val:+.8f} "
        f"test={init_test['loss'] - baseline_test:+.8f}"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scope = TrainableParameters.from_model(wrapped, optimizer, check_frozen=not cfg.skip_freeze_check)

    best_val = float("inf")
    best_state: Dict[str, Any] | None = None
    bad_epochs = 0
    history: list[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            wrapped=wrapped,
            train_chunks=data.train,
            optimizer=optimizer,
            scope=scope,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )

        row: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        do_eval = epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs
        if do_eval:
            val_metrics = eval_wrapped(wrapped, data.val, cfg.batch_size, collect_stats=True)
            val_loss = val_metrics["loss"]
            val_imp = baseline_val - val_loss
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_improvement"] = val_imp
            row["val_relative_ppl_reduction"] = relative_ppl_reduction(val_imp)

            if cfg.eval_test_during_training:
                test_metrics = eval_wrapped(wrapped, data.test, cfg.batch_size, collect_stats=False)
                test_imp = baseline_test - test_metrics["loss"]
                row["test_loss_exploratory"] = test_metrics["loss"]
                row["test_improvement_exploratory"] = test_imp
                row["test_relative_ppl_reduction_exploratory"] = relative_ppl_reduction(test_imp)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad_epochs = 0
                best_state = {
                    "trainable_state_dict": scope.trainable_state_dict(wrapped),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }
            else:
                bad_epochs += 1

            print(
                f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} "
                f"val_imp={val_imp:.6f} "
                f"val_ppl_red={100.0 * relative_ppl_reduction(val_imp):.3f}% "
                f"peft_abs={val_metrics.get('delta_abs_mean', 0.0):.6f} "
                f"peft_l2={val_metrics.get('delta_l2_rms', 0.0):.6f}"
            )
            history.append(row)
            if bad_epochs >= cfg.patience:
                print(f"[early_stop] no val improvement for {bad_epochs} evals")
                break
        else:
            print(f"[epoch {epoch:03d}] train={train_loss:.6f}")
            history.append(row)

    if best_state is None:
        best_state = {
            "trainable_state_dict": scope.trainable_state_dict(wrapped),
            "epoch": cfg.epochs,
            "val_loss": init_val["loss"],
        }
    scope.load_trainable_state_dict(wrapped, best_state["trainable_state_dict"])

    best_train_metrics = eval_wrapped(wrapped, data.train, cfg.batch_size, collect_stats=False)
    best_val_metrics = eval_wrapped(wrapped, data.val, cfg.batch_size, collect_stats=True)
    best_test_metrics = eval_wrapped(wrapped, data.test, cfg.batch_size, collect_stats=True)
    train_imp = baseline_train - best_train_metrics["loss"]
    val_imp = baseline_val - best_val_metrics["loss"]
    test_imp = baseline_test - best_test_metrics["loss"]

    summary = {
        "config": asdict(cfg),
        "model_family": cfg.model_family,
        "method": "site_ablation",
        "intervention_site": cfg.intervention_site,
        "adapter_input": cfg.adapter_input,
        "detach_adapter_input": cfg.detach_adapter_input,
        "layer_indices": layer_indices,
        "num_trainable_params": num_trainable,
        "trainable_param_names": [name for name, param in wrapped.named_parameters() if param.requires_grad],
        "official_dataset_splits": data.official_splits,
        "official_dataset_splits_checked": data.official_splits_checked,
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "test_split": cfg.test_split,
        "train_chunks": int(data.train.shape[0]),
        "val_chunks": int(data.val.shape[0]),
        "test_chunks": int(data.test.shape[0]),
        "block_size": data.block_size,
        "baseline_train_loss": baseline_train,
        "baseline_val_loss": baseline_val,
        "baseline_test_loss": baseline_test,
        "baseline_train_ppl": float(math.exp(baseline_train)),
        "baseline_val_ppl": float(math.exp(baseline_val)),
        "baseline_test_ppl": float(math.exp(baseline_test)),
        "wrapped_init_train_loss": init_train["loss"],
        "wrapped_init_val_loss": init_val["loss"],
        "wrapped_init_test_loss": init_test["loss"],
        "wrapped_init_train_delta_vs_baseline": init_train["loss"] - baseline_train,
        "wrapped_init_val_delta_vs_baseline": init_val["loss"] - baseline_val,
        "wrapped_init_test_delta_vs_baseline": init_test["loss"] - baseline_test,
        "best_epoch": int(best_state["epoch"]),
        "best_train_loss": best_train_metrics["loss"],
        "best_val_loss": best_val_metrics["loss"],
        "best_test_loss": best_test_metrics["loss"],
        "best_train_improvement_nats_per_token": train_imp,
        "best_val_improvement_nats_per_token": val_imp,
        "best_test_improvement_nats_per_token": test_imp,
        "best_train_relative_ppl_reduction": relative_ppl_reduction(train_imp),
        "best_val_relative_ppl_reduction": relative_ppl_reduction(val_imp),
        "best_test_relative_ppl_reduction": relative_ppl_reduction(test_imp),
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "history": history,
    }

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"summary": summary, "trainable_state_dict": best_state["trainable_state_dict"]}, out)
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[done] wrote {out}")
    print(f"[done] wrote {summary_path}")
    print(
        f"[best] model_family={cfg.model_family} method=site_ablation "
        f"site={cfg.intervention_site} epoch={summary['best_epoch']} "
        f"val_loss={summary['best_val_loss']:.6f} test_loss={summary['best_test_loss']:.6f} "
        f"val_imp={summary['best_val_improvement_nats_per_token']:.6f} nats/token "
        f"test_imp={summary['best_test_improvement_nats_per_token']:.6f} nats/token "
        f"test_ppl_red={100.0 * summary['best_test_relative_ppl_reduction']:.3f}%"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a GPT-2 site-ablation adapter using the attention_adapter data, "
            "evaluation, and checkpointing path."
        )
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model_family", type=str, default="gpt2", choices=["gpt2"])
    model_group.add_argument("--model_name", type=str, default="openai-community/gpt2")
    model_group.add_argument("--layer_indices", type=str, default="6,7,8,9,10,11")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset_name", type=str, default="wikitext")
    data_group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    data_group.add_argument("--dataset_revision", type=str, default=None)
    data_group.add_argument("--text_field", type=str, default="text")
    data_group.add_argument("--train_split", type=str, default="train")
    data_group.add_argument("--val_split", type=str, default="validation")
    data_group.add_argument("--test_split", type=str, default="test")
    data_group.add_argument("--block_size", type=int, required=True)
    data_group.add_argument("--batch_size", type=int, default=4)
    data_group.add_argument("--max_train_chunks", type=int, default=2048)
    data_group.add_argument("--max_val_chunks", type=int, default=512)
    data_group.add_argument("--max_test_chunks", type=int, default=512)
    data_group.add_argument("--cache_dir", type=str, default="outputs/attention_adapter/cache_gpt2/")

    adapter_group = parser.add_argument_group("site ablation adapter")
    adapter_group.add_argument("--intervention_site", type=str, default="z_pre_cproj", choices=list(INTERVENTION_SITES))
    adapter_group.add_argument("--adapter_input", type=str, default="ln1", choices=["ln1", "residual"])
    adapter_group.add_argument("--detach_adapter_input", action=argparse.BooleanOptionalAction, default=True)
    adapter_group.add_argument("--bottleneck_dim", type=int, default=4)
    adapter_group.add_argument("--adapter_dropout", type=float, default=0.05)
    adapter_group.add_argument("--output_scale", type=float, default=0.05)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--peft_l2", type=float, default=1e-5)
    train_group.add_argument("--peft_l1", type=float, default=0.0)
    train_group.add_argument("--lr", type=float, default=3e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-4)
    train_group.add_argument("--epochs", type=int, default=500)
    train_group.add_argument("--patience", type=int, default=30)
    train_group.add_argument("--eval_every", type=int, default=1)
    train_group.add_argument("--grad_clip", type=float, default=1.0)
    train_group.add_argument("--seed", type=int, default=0)
    train_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--skip_freeze_check", action="store_true")
    train_group.add_argument("--eval_test_during_training", action="store_true")
    train_group.add_argument("--output_path", type=str, required=True)

    return parser


def config_from_args(args: argparse.Namespace) -> SiteAblationFineTuneConfig:
    return SiteAblationFineTuneConfig(
        model_family=args.model_family,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        text_field=args.text_field,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_train_chunks=args.max_train_chunks,
        max_val_chunks=args.max_val_chunks,
        max_test_chunks=args.max_test_chunks,
        layer_indices=args.layer_indices,
        intervention_site=args.intervention_site,
        adapter_input=args.adapter_input,
        detach_adapter_input=args.detach_adapter_input,
        bottleneck_dim=args.bottleneck_dim,
        adapter_dropout=args.adapter_dropout,
        output_scale=args.output_scale,
        peft_l2=args.peft_l2,
        peft_l1=args.peft_l1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        output_path=args.output_path,
        cache_dir=args.cache_dir,
        skip_freeze_check=args.skip_freeze_check,
        eval_test_during_training=args.eval_test_during_training,
        split=args.train_split,
        max_chunks=args.max_train_chunks,
    )


def main() -> None:
    parser = build_arg_parser()
    train(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
