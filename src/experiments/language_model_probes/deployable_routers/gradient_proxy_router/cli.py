"""Unified CLI for gradient-proxy router experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from experiments.language_model_probes.deployable_routers.gradient_proxy_router.datasets import (
    combiner_main,
    gradient_state_main,
    sequence_state_main,
)
from experiments.language_model_probes.deployable_routers.gradient_proxy_router.router import (
    eval_main,
    layer_robustness_main,
    sweep_main,
)


COMMANDS: dict[str, tuple[str, Callable[[list[str] | None], None]]] = {
    "build-combiner": ("Build exact-cost learner-combiner rows.", combiner_main),
    "add-gradient-state": ("Add first-order gain targets.", gradient_state_main),
    "add-sequence-features": ("Add compact temporal features.", sequence_state_main),
    "sweep": ("Train and select sparse gain-distillation routers.", sweep_main),
    "eval": ("Evaluate a saved router checkpoint.", eval_main),
    "layer-robustness": ("Run layer/seed robustness checks.", layer_robustness_main),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gradient-proxy router command facade.")
    parser.add_argument("command", choices=sorted(COMMANDS), help="Subcommand to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the selected subcommand.")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser = build_arg_parser()
        parser.print_help()
        print("\nsubcommands:")
        for name in sorted(COMMANDS):
            print(f"  {name:22s} {COMMANDS[name][0]}")
        return

    command = argv[0]
    if command not in COMMANDS:
        build_arg_parser().parse_args(argv)
        return

    _, fn = COMMANDS[command]
    fn(argv[1:])


if __name__ == "__main__":
    main()
