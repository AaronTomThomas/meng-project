from __future__ import annotations

import argparse
import json

from experiments.router_development.attention_adapter.fine_tuning_evaluation.config import FineTuneEvalConfig
from experiments.router_development.attention_adapter.fine_tuning_evaluation.train import run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate attention-adapter methods on official GLUE files."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["zero_shot", "full_finetune", "akaza_freez", "lora", "loreft", "reft"])
    parser.add_argument("--task", type=str, required=True, choices=["sst2", "rte"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--glue_data_dir", type=str, default="glue_data")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--target_max_length", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--selection_split_from_train", type=float, default=0.0)
    parser.add_argument("--selection_split_seed", type=int, default=None)

    parser.add_argument("--model_family", type=str, default=None, choices=["gpt2", "pythia"])
    parser.add_argument("--layer_indices", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--bottleneck_dim", type=int, default=None)
    parser.add_argument("--adapter_dropout", type=float, default=0.05)
    parser.add_argument("--output_scale", type=float, default=None)

    parser.add_argument("--peft_target_profile", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    parser.add_argument("--reft_rank", type=int, default=4)
    parser.add_argument("--reft_dropout", type=float, default=0.05)
    parser.add_argument("--reft_output_scale", type=float, default=1.0)
    parser.add_argument("--reft_position_mode", type=str, default="all", choices=["all", "prefix", "suffix", "prefix_suffix"])
    parser.add_argument("--reft_prefix_positions", type=int, default=0)
    parser.add_argument("--reft_suffix_positions", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = FineTuneEvalConfig(**vars(args))
    metrics = run(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
