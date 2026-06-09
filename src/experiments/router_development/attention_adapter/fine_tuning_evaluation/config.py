from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


MethodName = Literal[
    "zero_shot",
    "full_finetune",
    "akaza_freez",
    "akaza_zconditioned",
    "akaza_fused",
    "lora",
    "loreft",
    "reft",
]
TaskName = Literal["sst2", "rte"]


@dataclass
class FineTuneEvalConfig:
    model_name_or_path: str
    method: MethodName
    task: TaskName
    output_dir: str

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    eval_batch_size: int = 8
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    max_test_examples: int | None = None
    max_length: int = 512
    target_max_length: int = 64
    gradient_accumulation_steps: int = 1
    eval_every: int = 1
    patience: int = 2
    checkpoint_path: str | None = None
    do_train: bool = False
    do_eval: bool = False
    selection_split_from_train: float = 0.0
    selection_split_seed: int | None = None
    glue_data_dir: str = "glue_data"

    model_family: str | None = None
    layer_indices: str | None = None
    grad_clip: float = 1.0

    bottleneck_dim: int | None = None
    adapter_dropout: float = 0.05
    output_scale: float | None = None

    peft_target_profile: str | None = None
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    reft_rank: int = 4
    reft_dropout: float = 0.05
    reft_output_scale: float = 1.0
    reft_position_mode: str = "all"
    reft_prefix_positions: int = 0
    reft_suffix_positions: int = 0

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)
