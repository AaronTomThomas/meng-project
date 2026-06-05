from __future__ import annotations

"""
Benchmark inference latency for attention-adapter model types.

This benchmark uses the active attention_adapter implementations:
  - baseline frozen causal LM
  - AKAZA/FreeZ
  - LoRA
  - LoReFT

Example:

PYTHONPATH=src uv run python -m experiments.router_development.attention_adapter.probes.benchmark_inference \
  --model_family gpt2 \
  --model_name openai-community/gpt2 \
  --methods baseline,akaza_freez,lora,loreft \
  --batch_size 4 \
  --seq_len 96 \
  --warmup_steps 20 \
  --timed_steps 100 \
  --repeats 5 \
  --device cuda \
  --output_path outputs/attention_adapter/inference_benchmark.json
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from experiments.router_development.attention_adapter.config import (
    AdapterMethod,
    config_from_values,
)
from experiments.router_development.attention_adapter.models import DEFAULT_FAMILY_SPECS
from experiments.router_development.attention_adapter.peft_factory import build_wrapped_model
from experiments.router_development.attention_adapter.trainer import (
    TrainableParameters,
    infer_num_layers,
    parse_layer_indices,
)


BASELINE_METHOD = "baseline"
DEFAULT_METHODS = [BASELINE_METHOD, *(method.value for method in AdapterMethod)]
DTYPES = {"float32", "float16", "bfloat16"}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def dtype_context(dtype_name: str, device: torch.device):
    if dtype_name == "float32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type in {"cuda", "cpu"})


def build_base_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    n_layers = infer_num_layers(model, args.model_family)
    for layer_idx in args.layer_indices_parsed:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={n_layers}")
    return model


def method_checkpoint(args: argparse.Namespace, method: str) -> str:
    return getattr(args, f"{method}_checkpoint", "")


def maybe_load_checkpoint(wrapped: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "trainable_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain 'trainable_state_dict'")
    scope = TrainableParameters(params=[], frozen_before_training={}, check_frozen=False)
    scope.load_trainable_state_dict(wrapped, checkpoint["trainable_state_dict"])
    print(f"[checkpoint] loaded trainable weights from {checkpoint_path}")


def build_method_model(
    *,
    args: argparse.Namespace,
    method: str,
    device: torch.device,
) -> nn.Module:
    model = build_base_model(args, device)
    if method == BASELINE_METHOD:
        return model

    cfg = config_from_values(
        method,
        model_family=args.model_family,
        model_name=args.model_name,
        layer_indices=args.layer_indices,
        batch_size=args.batch_size,
        block_size=args.seq_len,
        device=str(device),
        bottleneck_dim=args.bottleneck_dim,
        adapter_dropout=args.adapter_dropout,
        output_scale=args.output_scale,
        peft_target_profile=args.peft_target_profile,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        reft_rank=args.reft_rank,
        reft_dropout=args.reft_dropout,
        reft_output_scale=args.reft_output_scale,
        reft_position_mode=args.reft_position_mode,
        reft_prefix_positions=args.reft_prefix_positions,
        reft_suffix_positions=args.reft_suffix_positions,
    )
    wrapped = build_wrapped_model(
        model=model,
        cfg=cfg,
        layer_indices=args.layer_indices_parsed,
    ).to(device)
    maybe_load_checkpoint(wrapped, method_checkpoint(args, method), device)
    wrapped.eval()
    if hasattr(wrapped, "set_peft_eval_mode"):
        wrapped.set_peft_eval_mode()
    return wrapped


@torch.inference_mode()
def run_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    if hasattr(output, "logits"):
        return output.logits
    return output


@torch.inference_mode()
def validate_forward(model: nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    logits = run_forward(model, input_ids)
    expected_prefix = tuple(input_ids.shape)
    if tuple(logits.shape[:2]) != expected_prefix:
        raise RuntimeError(f"Expected logits prefix {expected_prefix}, got {tuple(logits.shape)}")
    if logits.ndim != 3:
        raise RuntimeError(f"Expected 3D logits, got shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Forward pass produced non-finite logits")
    return {
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
    }


@torch.inference_mode()
def time_repeated_forwards(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    timed_steps: int,
    device: torch.device,
) -> float:
    synchronize_if_needed(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed_steps):
            logits = run_forward(model, input_ids)
        end.record()
        torch.cuda.synchronize(device)
        total_ms = float(start.elapsed_time(end))
    else:
        start_time = time.perf_counter()
        for _ in range(timed_steps):
            logits = run_forward(model, input_ids)
        total_ms = (time.perf_counter() - start_time) * 1000.0

    del logits
    return total_ms


@torch.inference_mode()
def benchmark_model(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int,
    timed_steps: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup_steps):
        _ = run_forward(model, input_ids)
    synchronize_if_needed(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    repeat_totals_ms = [
        time_repeated_forwards(
            model=model,
            input_ids=input_ids,
            timed_steps=timed_steps,
            device=device,
        )
        for _ in range(repeats)
    ]
    mean_total_ms = statistics.fmean(repeat_totals_ms)
    mean_ms = mean_total_ms / max(1, timed_steps)
    median_ms = statistics.median(total / max(1, timed_steps) for total in repeat_totals_ms)
    stdev_ms = (
        statistics.stdev(total / max(1, timed_steps) for total in repeat_totals_ms)
        if repeats > 1
        else 0.0
    )
    tokens = input_ids.numel() * timed_steps
    examples = input_ids.shape[0] * timed_steps
    return {
        "mean_total_ms": mean_total_ms,
        "mean_ms_per_forward": mean_ms,
        "median_ms_per_forward": float(median_ms),
        "stdev_ms_per_forward": float(stdev_ms),
        "tokens_per_second": tokens / (mean_total_ms / 1000.0),
        "examples_per_second": examples / (mean_total_ms / 1000.0),
    }


def print_results(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return

    fastest_ms = min(row["mean_ms_per_forward"] for row in rows)
    baseline = next((row for row in rows if row["method"] == BASELINE_METHOD), rows[0])
    baseline_ms = baseline["mean_ms_per_forward"]

    print()
    print("[benchmark]")
    print("method          trainable    mean_ms  median_ms  std_ms   tok/s      vs_base  vs_fast  peak_mem_mb")
    for row in rows:
        peak_mem = row.get("peak_memory_mb")
        peak_mem_text = "-" if peak_mem is None else f"{peak_mem:.1f}"
        print(
            f"{row['method']:<14} "
            f"{row['trainable_params']:>9} "
            f"{row['mean_ms_per_forward']:>10.3f} "
            f"{row['median_ms_per_forward']:>10.3f} "
            f"{row['stdev_ms_per_forward']:>7.3f} "
            f"{row['tokens_per_second']:>10.1f} "
            f"{row['mean_ms_per_forward'] / baseline_ms:>7.3f}x "
            f"{row['mean_ms_per_forward'] / fastest_ms:>7.3f}x "
            f"{peak_mem_text:>11}"
        )
    print(f"[baseline] slowdown is relative to method={baseline['method']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark inference time for attention_adapter AKAZA, LoRA, and LoReFT implementations."
    )
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--model_family", type=str, default="gpt2", choices=sorted(DEFAULT_FAMILY_SPECS))
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--layer_indices", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--timed_steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=sorted(DTYPES))
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--bottleneck_dim", type=int, default=0)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--output_scale", type=float, default=0.05)

    parser.add_argument("--peft_target_profile", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    parser.add_argument("--reft_rank", type=int, default=4)
    parser.add_argument("--reft_dropout", type=float, default=0.0)
    parser.add_argument("--reft_output_scale", type=float, default=1.0)
    parser.add_argument(
        "--reft_position_mode",
        type=str,
        default="all",
        choices=["all", "prefix", "suffix", "prefix_suffix"],
    )
    parser.add_argument("--reft_prefix_positions", type=int, default=0)
    parser.add_argument("--reft_suffix_positions", type=int, default=0)

    parser.add_argument("--akaza_freez_checkpoint", type=str, default="")
    parser.add_argument("--lora_checkpoint", type=str, default="")
    parser.add_argument("--loreft_checkpoint", type=str, default="")
    return parser


def resolve_defaults(args: argparse.Namespace) -> None:
    defaults = DEFAULT_FAMILY_SPECS[args.model_family]
    if not args.model_name:
        args.model_name = defaults.default_model_name
    if not args.layer_indices:
        args.layer_indices = defaults.default_layer_indices
    if args.batch_size <= 0:
        args.batch_size = defaults.default_batch_size
    if args.bottleneck_dim <= 0:
        args.bottleneck_dim = defaults.default_bottleneck_dim
    if not args.peft_target_profile:
        args.peft_target_profile = defaults.default_peft_target_profile

    if args.seq_len <= 0:
        raise ValueError("--seq_len must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")
    if args.timed_steps <= 0:
        raise ValueError("--timed_steps must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    args.layer_indices_parsed = parse_layer_indices(args.layer_indices)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    resolve_defaults(args)

    methods = parse_csv(args.methods)
    choices = set(DEFAULT_METHODS)
    unknown = [method for method in methods if method not in choices]
    if unknown:
        raise ValueError(f"Unknown methods={unknown}; choices={sorted(choices)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    print("[config]")
    print(f"  methods: {methods}")
    print(f"  model_family: {args.model_family}")
    print(f"  model_name: {args.model_name}")
    print(f"  layer_indices: {args.layer_indices_parsed}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  timed_steps: {args.timed_steps}")
    print(f"  repeats: {args.repeats}")
    print(f"  device: {device}")
    print(f"  dtype: {args.dtype}")

    rows: list[dict[str, Any]] = []
    input_ids: torch.Tensor | None = None

    for method in methods:
        print()
        print(f"[run] method={method}")
        model = build_method_model(args=args, method=method, device=device)
        vocab_size = int(model.config.vocab_size if hasattr(model, "config") else model.model.config.vocab_size)
        if input_ids is None:
            input_ids = torch.randint(
                low=0,
                high=vocab_size,
                size=(args.batch_size, args.seq_len),
                dtype=torch.long,
                device=device,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with dtype_context(args.dtype, device):
            validation = validate_forward(model, input_ids)
            metrics = benchmark_model(
                model=model,
                input_ids=input_ids,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                repeats=args.repeats,
                device=device,
            )

        row = {
            "method": method,
            "model_family": args.model_family,
            "model_name": args.model_name,
            "layer_indices": args.layer_indices_parsed,
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "warmup_steps": int(args.warmup_steps),
            "timed_steps": int(args.timed_steps),
            "repeats": int(args.repeats),
            "device": str(device),
            "dtype": args.dtype,
            "trainable_params": int(sum(p.numel() for p in trainable_parameters(model))),
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "peak_memory_mb": peak_memory_mb(device),
            **validation,
            **metrics,
        }
        rows.append(row)
        print(
            f"[result] method={method} mean_ms={row['mean_ms_per_forward']:.3f} "
            f"median_ms={row['median_ms_per_forward']:.3f} tok/s={row['tokens_per_second']:.1f}"
        )

        del model
        synchronize_if_needed(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results(rows)

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {k: v for k, v in vars(args).items() if k != "layer_indices_parsed"},
            "layer_indices": args.layer_indices_parsed,
            "results": rows,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
