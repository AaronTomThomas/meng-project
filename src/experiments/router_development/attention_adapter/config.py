from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from experiments.router_development.attention_adapter.models import DEFAULT_FAMILY_SPECS
import torch

from experiments.attention_learners import LearnerHyperParams

class AdapterMethod(str, Enum):
    AKAZA_FREEZ = "akaza_freez"
    LORA = "lora"

    LOREFT = "loreft"
    NOREFT = "noreft"
    DI_REFT = "direft"
    def __str__(self) -> str:
        return self.value

REFT_METHODS: set[AdapterMethod] = {
    AdapterMethod.LOREFT,
    AdapterMethod.NOREFT,
    AdapterMethod.DI_REFT,
}

@dataclass
class BaseAdapterFineTuneConfig(LearnerHyperParams):
    model_family: str = "gpt2"
    model_name: str = "openai-community/gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_revision: str | None = None
    text_field: str = "text"

    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

    max_train_texts: int = 1000
    max_val_texts: int = 200
    max_test_texts: int = 200


    block_size: int = 96
    batch_size: int = 4

    max_train_chunks: int = 2048
    max_val_chunks: int = 512
    max_test_chunks: int = 512

    layer_indices: str = "6,7,8,9,10,11"
    method: AdapterMethod = AdapterMethod.AKAZA_FREEZ

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
    output_path: str = "outputs/attention_adapter/peft_comparison.pt"
    cache_dir: str = "outputs/attention_adapter/cache/"
    skip_freeze_check: bool = False
    eval_test_during_training: bool = False

    split: str = "train"
    max_texts: int = 1000
    max_chunks: int = 2048

@dataclass
class AKAZAFreeZConfig(BaseAdapterFineTuneConfig):
    method: AdapterMethod = AdapterMethod.AKAZA_FREEZ
    bottleneck_dim: int = 4
    adapter_dropout: float = 0.05
    output_scale: float = 0.05
#

@dataclass
class LoRAFineTuneConfig(BaseAdapterFineTuneConfig):
    method: AdapterMethod = AdapterMethod.LORA
    peft_target_profile: str = "attn_c_proj"
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.05
    lora_bias: str = "none"


@dataclass
class ReFTFineTuneConfig(BaseAdapterFineTuneConfig):
    method: AdapterMethod = AdapterMethod.LOREFT
    reft_rank: int = 4
    reft_dropout: float = 0.05
    reft_output_scale: float = 1.0
    reft_position_mode: str = "all"
    reft_prefix_positions: int = 0
    reft_suffix_positions: int = 0

AdapterFineTuneConfig: TypeAlias = (
    AKAZAFreeZConfig
    | LoRAFineTuneConfig
    | ReFTFineTuneConfig
)

CONFIG_TYPES: dict[AdapterMethod, type[BaseAdapterFineTuneConfig]] = {
    AdapterMethod.AKAZA_FREEZ: AKAZAFreeZConfig,
    AdapterMethod.LORA: LoRAFineTuneConfig,

    AdapterMethod.LOREFT: ReFTFineTuneConfig,
    AdapterMethod.NOREFT: ReFTFineTuneConfig,
    AdapterMethod.DI_REFT: ReFTFineTuneConfig,
}


def config_from_args(args: argparse.Namespace) -> AdapterFineTuneConfig:
    try:
        method = args.method if isinstance(args.method, AdapterMethod) else AdapterMethod(args.method)
    except ValueError as exc:
        choices = sorted(method.value for method in CONFIG_TYPES)
        raise ValueError(f"Unknown method={args.method!r}; choices={choices}") from exc

    try:
        config_type = CONFIG_TYPES[args.method]
    except KeyError as exc:
        choices = sorted(method.value for method in CONFIG_TYPES)
        raise ValueError(f"Unknown method={args.method!r}; choices={choices}") from exc
    
    config_kwargs: dict[str, Any] = {

        "method": method,

        "model_family": args.model_family,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "text_field": args.text_field,

        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,

        "max_train_texts": args.max_train_texts,
        "max_val_texts": args.max_val_texts,
        "max_test_texts": args.max_test_texts,

        "max_train_chunks": args.max_train_chunks,
        "max_val_chunks": args.max_val_chunks,
        "max_test_chunks": args.max_test_chunks,

        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "layer_indices": args.layer_indices,

        "peft_l2": args.peft_l2,
        "peft_l1": args.peft_l1,

        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "eval_every": args.eval_every,
        "grad_clip": args.grad_clip,

        "seed": args.seed,
        "device": args.device,
        "output_path": args.output_path,
        "cache_dir": args.cache_dir or f"outputs/attention_adapter/cache_{args.model_family}/",
        "skip_freeze_check": args.skip_freeze_check,
        "eval_test_during_training": args.eval_test_during_training,

        "split": args.train_split,
        "max_texts": args.max_train_texts,
        "max_chunks": args.max_train_chunks,
    }

    if args.method is AdapterMethod.AKAZA_FREEZ:
        config_kwargs.update(
            bottleneck_dim=args.bottleneck_dim,
            adapter_dropout=args.adapter_dropout,
            output_scale=args.output_scale,
        )
        return config_type(**config_kwargs)
    if args.method is AdapterMethod.LORA:
        config_kwargs.update(
            peft_target_profile=args.peft_target_profile,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_bias=args.lora_bias,
        )
        return config_type(**config_kwargs)

    if method in REFT_METHODS:
        config_kwargs.update(
            reft_rank=args.reft_rank,
            reft_dropout=args.reft_dropout,
            reft_output_scale=args.reft_output_scale,
            reft_position_mode=args.reft_position_mode,
            reft_prefix_positions=args.reft_prefix_positions,
            reft_suffix_positions=args.reft_suffix_positions,
        )
        return config_type(**config_kwargs)
    
    raise ValueError(f"Unknown method={args.method!r}")
