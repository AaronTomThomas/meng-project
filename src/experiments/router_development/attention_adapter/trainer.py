from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from experiments.router_development.attention_adapter.data import load_adapter_finetune_data
from experiments.router_development.attention_adapter.eval import eval_baseline, eval_wrapped
from experiments.router_development.attention_adapter.peft_factory import METHODS, build_wrapped_model
from experiments.router_development.attention_adapter.models import DEFAULT_FAMILY_SPECS
from experiments.router_development.attention_adapter.utils import lm_loss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig, AdapterMethod


def infer_num_layers(model, model_family: str) -> int:
    if model_family == "gpt2":
        return len(model.transformer.h)
    if model_family == "pythia":
        if not hasattr(model, "gpt_neox") or not hasattr(model.gpt_neox, "layers"):
            raise ValueError("Pythia mode expects a GPT-NeoX-style model with model.gpt_neox.layers")
        return len(model.gpt_neox.layers)
    raise ValueError(f"Unknown model_family={model_family!r}")


@dataclass
class TrainableParameters:
    params: list[torch.nn.Parameter]
    frozen_before_training: dict[str, torch.Tensor]
    check_frozen: bool

    @classmethod
    def from_model(
        cls,
        wrapped: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        check_frozen: bool,
    ) -> "TrainableParameters":
        params = [p for p in wrapped.parameters() if p.requires_grad]
        if not check_frozen:
            return cls(params=params, frozen_before_training={}, check_frozen=False)
        named = [(name, p) for name, p in wrapped.named_parameters() if p.requires_grad]
        print(f"[freeze_check] trainable params: {sum(p.numel() for _, p in named)}")
        print(f"[freeze_check] trainable parameter tensors: {len(named)}")

        for name, p in named[:20]:
            print(f"  trainable: {name} shape={tuple(p.shape)} numel={p.numel()}")
        if len(named) > 20:
            print(f"  ... {len(named) - 20} more trainable tensors")
        assert named, "No trainable parameters found."

        trainable_ids = {id(p) for _, p in named}
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert optimizer_ids == trainable_ids, (
            "Optimizer params do not exactly equal trainable params: "
            f"missing={len(trainable_ids - optimizer_ids)} extra={len(optimizer_ids - trainable_ids)}"
        )
        print("[freeze_check] OK: optimizer contains exactly trainable params.")
        frozen = {
            name: p.detach().cpu().clone()
            for name, p in wrapped.named_parameters()
            if not p.requires_grad
        }
        return cls(params=params, frozen_before_training=frozen, check_frozen=True)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.params)
    

    def trainable_state_dict(self, wrapped: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: p.detach().cpu().clone()
            for name, p in wrapped.named_parameters()
            if p.requires_grad
        }
    
    def load_trainable_state_dict(self, wrapped: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
        named_params = dict(wrapped.named_parameters())
        expected = {name for name, p in wrapped.named_parameters() if p.requires_grad}
        actual = set(state)
        unknown = sorted(actual - set(named_params))
        if unknown:
            raise KeyError(f"State contains unknown parameter names: {unknown[:10]}")
        missing = sorted(expected - actual)
        if missing:
            raise KeyError(f"State is missing trainable parameter names: {missing[:10]}")
        extra = sorted(actual - expected)
        if extra:
            raise KeyError(f"State contains non-trainable parameter names: {extra[:10]}")
        with torch.no_grad():
            for name, value in state.items():
                named_params[name].copy_(value.to(device=named_params[name].device, dtype=named_params[name].dtype))

    
    def assert_first_backward_ok(self, wrapped: torch.nn.Module) -> None:
        if not self.check_frozen:
            return
        frozen_with_grads = [
            name for name, p in wrapped.named_parameters() if not p.requires_grad and p.grad is not None
        ]
        trainable_with_grads = [
            name for name, p in wrapped.named_parameters() if p.requires_grad and p.grad is not None
        ]
        assert not frozen_with_grads, f"Frozen params received grads: {frozen_with_grads[:10]}"
        assert trainable_with_grads, "No trainable params received gradients."
        print("[freeze_check] OK: gradients only exist on trainable params.")


    def assert_frozen_unchanged(self, wrapped: torch.nn.Module) -> None:
        if not self.check_frozen:
            return
        changed = []
        for name, p in wrapped.named_parameters():
            if name not in self.frozen_before_training:
                continue
            after = p.detach().cpu()
            if not torch.equal(self.frozen_before_training[name], after):
                changed.append((name, (self.frozen_before_training[name] - after).abs().max().item()))
        assert not changed, f"Frozen parameters changed: {changed[:10]}"
        print("[freeze_check] OK: frozen parameters unchanged after optimizer step.")



def train_one_epoch(
    *,
    wrapped: torch.nn.Module,
    train_chunks: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scope: TrainableParameters,
    cfg: AdapterFineTuneConfig,
    device: torch.device,
    epoch: int,
) -> float:
    wrapped.set_peft_train_mode()
    perm = torch.randperm(train_chunks.shape[0])
    total_loss = 0.0
    total_examples = 0

    for start in range(0, perm.numel(), cfg.batch_size):
        idx = perm[start : start + cfg.batch_size]
        input_ids = train_chunks[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = wrapped(input_ids)
        loss = lm_loss(logits, input_ids)

        if scope.params and (cfg.peft_l2 > 0 or cfg.peft_l1 > 0):
            vec = torch.cat([p.reshape(-1) for p in scope.params])
            reg = torch.zeros((), device=vec.device, dtype=vec.dtype)
            if cfg.peft_l2 > 0:
                reg = reg + cfg.peft_l2 * vec.pow(2).mean()
            if cfg.peft_l1 > 0:
                reg = reg + cfg.peft_l1 * vec.abs().mean()
            loss = loss + reg

        loss.backward()
        if epoch == 1 and total_examples == 0:
            scope.assert_first_backward_ok(wrapped)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(scope.params, max_norm=cfg.grad_clip)
        optimizer.step()
        if epoch == 1 and total_examples == 0:
            scope.assert_frozen_unchanged(wrapped)

        total_loss += float(loss.item()) * input_ids.shape[0]
        total_examples += input_ids.shape[0]

    return total_loss / max(1, total_examples)

def train(cfg: AdapterFineTuneConfig) -> None:
    if cfg.model_family not in DEFAULT_FAMILY_SPECS:
        raise ValueError(f"Unknown model_family={cfg.model_family!r}; choices={sorted(DEFAULT_FAMILY_SPECS)}")
    if not isinstance(cfg.method, AdapterMethod):
        choices = [method.value for method in METHODS]
        raise TypeError(f"cfg.method must be an AdapterMethod enum, got {cfg.method!r}; choices={choices}")
    if cfg.method not in METHODS:
        choices = [method.value for method in METHODS]
        raise ValueError(f"Unknown method={cfg.method!r}; choices={choices}")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    layer_indices = sorted(int(x.strip()) for x in cfg.layer_indices.split(",") if x.strip())
    if not layer_indices:
        raise ValueError("--layer_indices cannot be empty")

    print("[config]")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print(f"  parsed_layer_indices: {layer_indices}")



    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = infer_num_layers(model, cfg.model_family)
    print(f"[model] family={cfg.model_family} layers={n_layers}")
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")
        

    data = load_adapter_finetune_data(cfg, tokenizer)
    baseline_train = eval_baseline(model, data.train, cfg.batch_size, device)
    baseline_val = eval_baseline(model, data.val, cfg.batch_size, device)
    baseline_test = eval_baseline(model, data.test, cfg.batch_size, device)

    wrapped = build_wrapped_model(model=model, cfg=cfg, layer_indices=layer_indices).to(device)
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
            val_metrics = eval_wrapped(wrapped, data.val, cfg.batch_size, collect_stats=True)
            val_loss = val_metrics["loss"]
            val_imp = baseline_val - val_loss
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_improvement"] = val_imp
            row["val_relative_ppl_reduction"] = 1.0 - math.exp(-val_imp)

            if cfg.eval_test_during_training:
                test_metrics = eval_wrapped(wrapped, data.test, cfg.batch_size, collect_stats=False)
                test_imp = baseline_test - test_metrics["loss"]
                row["test_loss_exploratory"] = test_metrics["loss"]
                row["test_improvement_exploratory"] = test_imp
                row["test_relative_ppl_reduction_exploratory"] = 1.0 - math.exp(-test_imp)

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
                f"val_ppl_red={100.0 * (1.0 - math.exp(-val_imp)):.3f}% "
                f"peft_abs={val_metrics.get('delta_abs_mean', val_metrics.get('peft_param_abs_mean', 0.0)):.6f} "
                f"peft_l2={val_metrics.get('delta_l2_rms', val_metrics.get('peft_param_l2_rms', 0.0)):.6f}"
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
    config_dict = asdict(cfg)
    config_dict["method"] = cfg.method.value

    summary = {
        "config": config_dict,
        "model_family": cfg.model_family,
        "method": cfg.method.value,
        "peft_target_profile": getattr(cfg, "peft_target_profile", None),
        "layer_indices": layer_indices,
        "num_trainable_params": num_trainable,
        "trainable_param_names": [name for name, p in wrapped.named_parameters() if p.requires_grad],
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
        "best_train_relative_ppl_reduction": 1.0 - math.exp(-train_imp),
        "best_val_relative_ppl_reduction": 1.0 - math.exp(-val_imp),
        "best_test_relative_ppl_reduction": 1.0 - math.exp(-test_imp),
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
    profile_text = f" profile={cfg.peft_target_profile}" if hasattr(cfg, "peft_target_profile") else ""
    print(
        f"[best] model_family={cfg.model_family} method={cfg.method}{profile_text} "
        f"epoch={summary['best_epoch']} val_loss={summary['best_val_loss']:.6f} "
        f"test_loss={summary['best_test_loss']:.6f} "
        f"val_imp={summary['best_val_improvement_nats_per_token']:.6f} nats/token "
        f"test_imp={summary['best_test_improvement_nats_per_token']:.6f} nats/token "
        f"test_ppl_red={100.0 * summary['best_test_relative_ppl_reduction']:.3f}%"
    )
