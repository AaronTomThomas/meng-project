
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from experiments.router_development.attention_adapter.trainer import train
import torch


from experiments.router_development.attention_adapter.config import (
    AKAZAFreeZConfig,
    AdapterFineTuneConfig,
    AdapterMethod,
    LoRAFineTuneConfig,
    config_from_args,
)
from experiments.router_development.attention_adapter.models import (
    DEFAULT_FAMILY_SPECS,
    ModelFamilyDefaults,
)

@dataclass(frozen=True)
class CommandSpec:
    name: str
    family: str
    method: AdapterMethod
    help_text: str


COMMANDS = (
    CommandSpec(
        name="gpt2-akaza",
        family="gpt2",
        method=AdapterMethod.AKAZA_FREEZ,
        help_text="GPT-2 custom AKAZA/FreeZ pre-c_proj z-space adapter.",
    ),
    CommandSpec(
        name="gpt2-lora",
        family="gpt2",
        method=AdapterMethod.LORA,
        help_text="GPT-2 official Hugging Face PEFT LoRA baseline.",
    ),
    CommandSpec(
        name="pythia-akaza",
        family="pythia",
        method=AdapterMethod.AKAZA_FREEZ,
        help_text="Pythia/GPT-NeoX custom AKAZA/FreeZ pre-attention.dense adapter.",
    ),
    CommandSpec(
        name="pythia-lora",
        family="pythia",
        method=AdapterMethod.LORA,
        help_text="Pythia/GPT-NeoX official Hugging Face PEFT LoRA baseline.",
    ),
)


def add_model_args(parser: argparse.ArgumentParser, defaults: ModelFamilyDefaults) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("--model_name", type=str, default=defaults.default_model_name)
    group.add_argument("--layer_indices", type=str, default=defaults.default_layer_indices)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument("--dataset_name", type=str, default="wikitext")
    group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")

    group.add_argument("--dataset_revision", type=str, default=None)

    group.add_argument("--text_field", type=str, default="text")

    group.add_argument("--train_split", type=str, default="train")
    group.add_argument("--val_split", type=str, default="validation")
    group.add_argument("--test_split", type=str, default="test")

    group.add_argument("--max_train_texts", type=int, required=True)
    group.add_argument("--max_val_texts", type=int, required=True)
    group.add_argument("--max_test_texts", type=int, required=True)

    group.add_argument("--block_size", type=int, required=True)    
    group.add_argument("--batch_size", type=int, required=True)

    group.add_argument("--max_train_chunks", type=int, required=True)
    group.add_argument("--max_val_chunks", type=int, required=True)
    group.add_argument("--max_test_chunks", type=int, required=True)

    group.add_argument("--cache_dir", type=str, default=None)



def add_training_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--lr", type=float, default=3e-4)
    group.add_argument("--weight_decay", type=float, default=1e-4)

    group.add_argument("--epochs", type=int, required=True)
    group.add_argument("--patience", type=int, required=True)
    group.add_argument("--eval_every", type=int, default=1)

    group.add_argument("--grad_clip", type=float, default=1.0)
    group.add_argument("--peft_l2", type=float, default=1e-5)
    group.add_argument("--peft_l1", type=float, default=0.0)

    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    group.add_argument("--skip_freeze_check", action="store_true")
    group.add_argument("--eval_test_during_training", action="store_true")
    group.add_argument("--output_path", type=str, required=True)


def add_akaza_args(parser: argparse.ArgumentParser, defaults: ModelFamilyDefaults) -> None:
    group = parser.add_argument_group("AKAZA / FreeZ")
    group.add_argument("--bottleneck_dim", type=int, default=defaults.default_bottleneck_dim)
    group.add_argument("--adapter_dropout", type=float, default=0.05)
    group.add_argument("--output_scale", type=float, default=defaults.default_output_scale)


def add_lora_args(parser: argparse.ArgumentParser, defaults: ModelFamilyDefaults) -> None:
    group = parser.add_argument_group("LoRA")
    group.add_argument(
        "--peft_target_profile",
        type=str,
        default=defaults.default_peft_target_profile,
        choices=sorted(defaults.lora_target_profiles),
    )
    group.add_argument("--lora_rank", type=int, default=4)
    group.add_argument("--lora_alpha", type=int, default=4)
    group.add_argument("--lora_dropout", type=float, default=0.05)
    group.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

def add_method_args(
    parser: argparse.ArgumentParser,
    *,
    method: AdapterMethod,
    defaults: ModelFamilyDefaults,
) -> None:
    if method is AdapterMethod.AKAZA_FREEZ:
        add_akaza_args(parser, defaults)
        return
    if method is AdapterMethod.LORA:
        add_lora_args(parser, defaults)
        return


def add_command_parser(subparsers: argparse._SubParsersAction, spec: CommandSpec) -> None:
    defaults = DEFAULT_FAMILY_SPECS[spec.family]
    parser = subparsers.add_parser(spec.name, help=spec.help_text, description=spec.help_text)
    parser.set_defaults(model_family=spec.family, method=spec.method)
    add_model_args(parser, defaults)
    add_data_args(parser)
    add_training_args(parser)
    add_method_args(parser, method=spec.method, defaults=defaults)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one attention adapter fine-tuning baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for spec in COMMANDS:
        add_command_parser(subparsers, spec)
    return parser



    

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(config_from_args(args))




if __name__ == "__main__":
    main()
