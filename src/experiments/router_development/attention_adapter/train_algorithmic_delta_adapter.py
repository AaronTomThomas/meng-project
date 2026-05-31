from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.attention_learners import LEARNERS, LearnerHyperParams
from experiments.router_development.attention_adapter.adapters.algorithmic_delta_adapters import (
    GPT2AlgorithmicDeltaAdapter,
    parse_csv,
    parse_int_csv,
)
from experiments.router_development.attention_adapter.data import load_adapter_finetune_data
from experiments.router_development.attention_adapter.models import DEFAULT_FAMILY_SPECS
from experiments.router_development.attention_adapter.trainer import (
    TrainableParameters,
    infer_num_layers,
    parse_layer_indices,
    train_one_epoch,
)
from experiments.router_development.attention_adapter.utils import lm_loss, set_seed


@dataclass
class AlgorithmicDeltaFineTuneConfig(LearnerHyperParams):
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
    head_indices: str = "all"
    candidate_learners: str = "sharp,window_soft,local_linear_attention"

    adapter_mode: str = "mlp"
    router_hidden_dims: str = "4"
    router_dropout: float = 0.05
    router_input: str = "ln1"
    alpha_scale: float = 0.25

    peft_l2: float = 1e-5
    peft_l1: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 150
    patience: int = 25
    eval_every: int = 1
    grad_clip: float = 1.0

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: str = "outputs/attention_adapter/algorithmic_delta.pt"
    cache_dir: str = "outputs/attention_adapter/cache_gpt2/"
    skip_freeze_check: bool = False
    eval_test_during_training: bool = False

    split: str = "train"
    max_chunks: int = 2048


def relative_ppl_reduction(delta_nll: float) -> float:
    return float(1.0 - math.exp(-delta_nll))


@torch.no_grad()
def eval_baseline_with_progress(
    model: torch.nn.Module,
    chunks: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    label: str,
    progress_every: int = 25,
) -> float:
    model.eval()
    losses: list[float] = []
    n_examples = chunks.shape[0]
    n_batches = math.ceil(n_examples / batch_size)
    print(f"[eval:{label}] baseline start chunks={n_examples} batches={n_batches}")
    for batch_idx, start in enumerate(range(0, n_examples, batch_size), start=1):
        input_ids = chunks[start : start + batch_size].to(device)
        logits = model(input_ids).logits
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])
        if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == n_batches:
            print(f"[eval:{label}] baseline batch {batch_idx}/{n_batches}")
    out = sum(losses) / max(1, n_examples)
    print(f"[eval:{label}] baseline done loss={out:.6f}")
    return out


@torch.no_grad()
def eval_wrapped_with_progress(
    wrapped: GPT2AlgorithmicDeltaAdapter,
    chunks: torch.Tensor,
    batch_size: int,
    *,
    label: str,
    collect_stats: bool = True,
    progress_every: int = 25,
) -> Dict[str, float]:
    wrapped.set_peft_eval_mode()
    losses: list[float] = []
    stats_accum: Dict[str, float] = {}
    n_stats_batches = 0
    n_examples = chunks.shape[0]
    n_batches = math.ceil(n_examples / batch_size)
    stats_text = " with stats" if collect_stats else ""
    print(f"[eval:{label}] wrapped start chunks={n_examples} batches={n_batches}{stats_text}")
    for batch_idx, start in enumerate(range(0, n_examples, batch_size), start=1):
        input_ids = chunks[start : start + batch_size].to(wrapped.device)
        logits = wrapped(input_ids)
        loss = lm_loss(logits, input_ids)
        losses.append(float(loss.item()) * input_ids.shape[0])
        if collect_stats:
            stats = wrapped.peft_stats(input_ids)
            for key, value in stats.items():
                stats_accum[key] = stats_accum.get(key, 0.0) + float(value)
            n_stats_batches += 1
        if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == n_batches:
            print(f"[eval:{label}] wrapped batch {batch_idx}/{n_batches}")

    out = {"loss": sum(losses) / max(1, n_examples)}
    if collect_stats:
        for key, value in stats_accum.items():
            out[key] = value / max(1, n_stats_batches)
    print(f"[eval:{label}] wrapped done loss={out['loss']:.6f}")
    return out


def build_wrapped_model(
    *,
    model: torch.nn.Module,
    cfg: AlgorithmicDeltaFineTuneConfig,
    layer_indices: list[int],
) -> GPT2AlgorithmicDeltaAdapter:
    if cfg.model_family != "gpt2":
        raise ValueError("Algorithmic delta adapter currently supports only model_family='gpt2'")
    candidate_learners = parse_csv(cfg.candidate_learners)
    if not candidate_learners:
        raise ValueError("candidate_learners cannot be empty")
    unknown = sorted(set(candidate_learners) - set(LEARNERS))
    if unknown:
        raise ValueError(f"Unknown candidate learners {unknown}; available={LEARNERS}")
    return GPT2AlgorithmicDeltaAdapter(
        model=model,
        cfg=cfg,
        layer_indices=layer_indices,
        head_indices=cfg.head_indices,
        candidate_learners=candidate_learners,
        adapter_mode=cfg.adapter_mode,
        router_hidden_dims=parse_int_csv(cfg.router_hidden_dims),
        router_dropout=cfg.router_dropout,
        router_input=cfg.router_input,
        alpha_scale=cfg.alpha_scale,
    )


def train(cfg: AlgorithmicDeltaFineTuneConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    if cfg.model_family not in DEFAULT_FAMILY_SPECS:
        raise ValueError(f"Unknown model_family={cfg.model_family!r}; choices={sorted(DEFAULT_FAMILY_SPECS)}")
    if cfg.model_family != "gpt2":
        raise ValueError("Algorithmic delta adapter currently supports only model_family='gpt2'")

    layer_indices = parse_layer_indices(cfg.layer_indices)
    print("[config]")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    print(f"  parsed_layer_indices: {layer_indices}")
    print(f"  parsed_candidate_learners: {parse_csv(cfg.candidate_learners)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    n_layers = infer_num_layers(model, cfg.model_family)
    print(f"[model] family={cfg.model_family} layers={n_layers}")
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")

    data = load_adapter_finetune_data(cfg, tokenizer)
    baseline_train = eval_baseline_with_progress(model, data.train, cfg.batch_size, device, label="train")
    baseline_val = eval_baseline_with_progress(model, data.val, cfg.batch_size, device, label="val")
    baseline_test = eval_baseline_with_progress(model, data.test, cfg.batch_size, device, label="test")

    wrapped = build_wrapped_model(model=model, cfg=cfg, layer_indices=layer_indices).to(device)
    trainable_params = [param for param in wrapped.parameters() if param.requires_grad]
    num_trainable = sum(param.numel() for param in trainable_params)
    print(f"[model] trainable_params={num_trainable}")

    init_train = eval_wrapped_with_progress(
        wrapped,
        data.train,
        cfg.batch_size,
        label="init_train",
        collect_stats=False,
    )
    init_val = eval_wrapped_with_progress(
        wrapped,
        data.val,
        cfg.batch_size,
        label="init_val",
        collect_stats=True,
    )
    init_test = eval_wrapped_with_progress(
        wrapped,
        data.test,
        cfg.batch_size,
        label="init_test",
        collect_stats=False,
    )
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
    history: List[Dict[str, Any]] = []

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
            val_metrics = eval_wrapped_with_progress(
                wrapped,
                data.val,
                cfg.batch_size,
                label=f"epoch_{epoch:03d}_val",
                collect_stats=True,
            )
            val_loss = val_metrics["loss"]
            val_imp = baseline_val - val_loss
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["val_improvement"] = val_imp
            row["val_relative_ppl_reduction"] = relative_ppl_reduction(val_imp)

            if cfg.eval_test_during_training:
                test_metrics = eval_wrapped_with_progress(
                    wrapped,
                    data.test,
                    cfg.batch_size,
                    label=f"epoch_{epoch:03d}_test",
                    collect_stats=False,
                )
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
                f"alpha_abs={val_metrics.get('alpha_abs_mean', 0.0):.6f} "
                f"delta_l2={val_metrics.get('delta_l2_rms', 0.0):.6f}"
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

    best_train_metrics = eval_wrapped_with_progress(
        wrapped,
        data.train,
        cfg.batch_size,
        label="best_train",
        collect_stats=False,
    )
    best_val_metrics = eval_wrapped_with_progress(
        wrapped,
        data.val,
        cfg.batch_size,
        label="best_val",
        collect_stats=True,
    )
    best_test_metrics = eval_wrapped_with_progress(
        wrapped,
        data.test,
        cfg.batch_size,
        label="best_test",
        collect_stats=True,
    )
    train_imp = baseline_train - best_train_metrics["loss"]
    val_imp = baseline_val - best_val_metrics["loss"]
    test_imp = baseline_test - best_test_metrics["loss"]

    summary = {
        "config": asdict(cfg),
        "model_family": cfg.model_family,
        "method": "algorithmic_delta",
        "layer_indices": layer_indices,
        "head_indices": cfg.head_indices,
        "candidate_learners": parse_csv(cfg.candidate_learners),
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
        f"[best] model_family={cfg.model_family} method=algorithmic_delta "
        f"epoch={summary['best_epoch']} val_loss={summary['best_val_loss']:.6f} "
        f"test_loss={summary['best_test_loss']:.6f} "
        f"val_imp={summary['best_val_improvement_nats_per_token']:.6f} nats/token "
        f"test_imp={summary['best_test_improvement_nats_per_token']:.6f} nats/token "
        f"test_ppl_red={100.0 * summary['best_test_relative_ppl_reduction']:.3f}%"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_FAMILY_SPECS["gpt2"]
    parser = argparse.ArgumentParser(
        description=(
            "Train a native attention_adapter algorithmic-delta adapter for GPT-2. "
            "Each selected layer mixes fixed canonical attention-learner deltas relative to soft attention."
        )
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model_family", type=str, default="gpt2", choices=["gpt2"])
    model_group.add_argument("--model_name", type=str, default=defaults.default_model_name)
    model_group.add_argument("--layer_indices", type=str, default=defaults.default_layer_indices)
    model_group.add_argument("--head_indices", type=str, default="all")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset_name", type=str, default="wikitext")
    data_group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    data_group.add_argument("--dataset_revision", type=str, default=None)
    data_group.add_argument("--text_field", type=str, default="text")
    data_group.add_argument("--train_split", type=str, default="train")
    data_group.add_argument("--val_split", type=str, default="validation")
    data_group.add_argument("--test_split", type=str, default="test")
    data_group.add_argument("--block_size", type=int, required=True)
    data_group.add_argument("--batch_size", type=int, default=defaults.default_batch_size)
    data_group.add_argument("--max_train_chunks", type=int, default=defaults.default_train_chunks)
    data_group.add_argument("--max_val_chunks", type=int, default=defaults.default_val_chunks)
    data_group.add_argument("--max_test_chunks", type=int, default=defaults.default_test_chunks)
    data_group.add_argument("--cache_dir", type=str, default=None)

    adapter_group = parser.add_argument_group("algorithmic delta")
    adapter_group.add_argument("--candidate_learners", type=str, default="sharp,window_soft,local_linear_attention")
    adapter_group.add_argument("--adapter_mode", type=str, default="mlp", choices=["scalar", "mlp"])
    adapter_group.add_argument("--router_hidden_dims", type=str, default="4")
    adapter_group.add_argument("--router_dropout", type=float, default=0.05)
    adapter_group.add_argument("--router_input", type=str, default="ln1", choices=["ln1", "residual"])
    adapter_group.add_argument("--alpha_scale", type=float, default=0.25)

    learner_group = parser.add_argument_group("attention learner hyperparameters")
    learner_group.add_argument("--local_kernel_beta", type=float, default=1.0)
    learner_group.add_argument("--window_size", type=int, default=16)
    learner_group.add_argument("--k_knn_mean", type=int, default=4)
    learner_group.add_argument("--ridge_lambda", type=float, default=1e-1)
    learner_group.add_argument("--k_linear_local", type=int, default=16)
    learner_group.add_argument("--k_sharp", type=int, default=2)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--peft_l2", type=float, default=1e-5)
    train_group.add_argument("--peft_l1", type=float, default=0.0)
    train_group.add_argument("--lr", type=float, default=3e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-4)
    train_group.add_argument("--epochs", type=int, default=150)
    train_group.add_argument("--patience", type=int, default=25)
    train_group.add_argument("--eval_every", type=int, default=1)
    train_group.add_argument("--grad_clip", type=float, default=1.0)
    train_group.add_argument("--seed", type=int, default=0)
    train_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--skip_freeze_check", action="store_true")
    train_group.add_argument("--eval_test_during_training", action="store_true")
    train_group.add_argument("--output_path", type=str, required=True)

    return parser


def config_from_args(args: argparse.Namespace) -> AlgorithmicDeltaFineTuneConfig:
    cache_dir = args.cache_dir or f"outputs/attention_adapter/cache_{args.model_family}/"
    return AlgorithmicDeltaFineTuneConfig(
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
        head_indices=args.head_indices,
        candidate_learners=args.candidate_learners,
        adapter_mode=args.adapter_mode,
        router_hidden_dims=args.router_hidden_dims,
        router_dropout=args.router_dropout,
        router_input=args.router_input,
        alpha_scale=args.alpha_scale,
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
        cache_dir=cache_dir,
        skip_freeze_check=args.skip_freeze_check,
        eval_test_during_training=args.eval_test_during_training,
        split=args.train_split,
        max_chunks=args.max_train_chunks,
        local_kernel_beta=args.local_kernel_beta,
        window_size=args.window_size,
        k_knn_mean=args.k_knn_mean,
        ridge_lambda=args.ridge_lambda,
        k_linear_local=args.k_linear_local,
        k_sharp=args.k_sharp,
    )


def main() -> None:
    parser = build_arg_parser()
    train(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
