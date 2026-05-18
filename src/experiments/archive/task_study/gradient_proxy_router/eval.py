"""Checkpoint evaluation for the gradient-proxy router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.archive.task_study.gradient_proxy_router.datasets import (
    FIRST_ORDER_GAIN_KEY,
    ROUTER_FEATURE_KEY,
)
from experiments.archive.task_study.gradient_proxy_router.router import (
    CANONICAL_CANDIDATE_ACTION,
    CANONICAL_GROUP_ID,
    checkpoint_actions,
    summarize_actions,
)


def build_eval_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a first-order gain distillation router.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="")
    return parser


def action_histogram(actions: torch.Tensor, action_names: list[str]) -> dict[str, dict[str, float]]:
    counts = torch.bincount(actions.long(), minlength=len(action_names)).float()
    total = float(max(actions.numel(), 1))
    return {
        action_names[idx]: {
            "count": float(counts[idx].item()),
            "fraction": float(counts[idx].item() / total),
        }
        for idx in range(len(action_names))
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path = "",
    split: str = "test",
    batch_size: int = 256,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset_path = Path(dataset_path or checkpoint["dataset_path"])
    dataset = torch.load(Path(dataset_path), map_location="cpu")
    base_action_idx = int(checkpoint["base_action_idx"])
    ordered_idx, actions = checkpoint_actions(
        checkpoint=checkpoint,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        device=device,
    )

    summary = summarize_actions(dataset, ordered_idx, actions, base_action_idx)
    summary.update(
        {
            "scope": dataset["metadata"].get("evaluation_scope", "pointwise_local_next_token_nll"),
            "training_scope": checkpoint["training_scope"],
            "split": split,
            "dataset_path": str(dataset_path),
            "checkpoint_path": str(checkpoint_path),
            "router_features": ROUTER_FEATURE_KEY,
            "target": FIRST_ORDER_GAIN_KEY,
            "group_id": CANONICAL_GROUP_ID,
            "candidate_action": CANONICAL_CANDIDATE_ACTION,
            "action_names": list(dataset["action_names"]),
            "diagnostics": {
                "chosen_action_histogram": action_histogram(actions, list(dataset["action_names"])),
            },
        }
    )
    return summary


def eval_main(argv: list[str] | None = None) -> None:
    args = build_eval_arg_parser().parse_args(argv)
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    eval_main()
