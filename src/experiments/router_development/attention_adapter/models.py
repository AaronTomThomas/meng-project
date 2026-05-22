from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelFamilyDefaults:

    name: str
    default_model_name: str
    default_layer_indices: str
    default_batch_size: int

    default_train_chunks: int
    default_val_chunks: int
    default_test_chunks: int

    default_bottleneck_dim: int
    default_output_scale: float

    default_epochs: int
    default_patience: int

    default_peft_target_profile: str
    lora_target_profiles: Dict[str, List[str]]

    target_prefix_template: str
    peft_fan_in_fan_out: bool

    def target_module_names(self, layer_indices: list[int], suffixes: list[str]) -> list[str]:
        return [
            self.target_prefix_template.format(layer_idx=layer_idx, suffix=suffix)
            for layer_idx in layer_indices
            for suffix in suffixes
        ]
    

GPT2_DEFAULTS = ModelFamilyDefaults(
    name="gpt2",
    default_model_name="openai-community/gpt2",
    default_layer_indices="6,7,8,9,10,11",

    default_batch_size=4,

    default_train_chunks=2048,
    default_val_chunks=512,
    default_test_chunks=512,

    default_bottleneck_dim=4,
    default_output_scale=1.0,
    default_epochs=500,
    default_patience=30,

    default_peft_target_profile="attn_c_proj",
    lora_target_profiles={
        "attn_c_attn": ["attn.c_attn"],
        "attn_c_proj": ["attn.c_proj"],
        "attn_both": ["attn.c_attn", "attn.c_proj"],
        "mlp": ["mlp.c_fc", "mlp.c_proj"],
        "block_all": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    },

    target_prefix_template="transformer.h.{layer_idx}.{suffix}",
    peft_fan_in_fan_out=True,
)


PYTHIA_DEFAULTS = ModelFamilyDefaults(
    name="pythia",
    default_model_name="EleutherAI/pythia-1b",
    default_layer_indices="10,11,12,13,14,15",
    default_batch_size=1,

    default_train_chunks=512,
    default_val_chunks=128,
    default_test_chunks=128,

    default_bottleneck_dim=1,
    default_output_scale=1.0,
    default_epochs=80,
    default_patience=8,

    default_peft_target_profile="attn_dense",
    lora_target_profiles={
        "attn_qkv": ["attention.query_key_value"],
        "attn_dense": ["attention.dense"],
        "attn_both": ["attention.query_key_value", "attention.dense"],
        "mlp": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "block_all": [
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    },
    target_prefix_template="gpt_neox.layers.{layer_idx}.{suffix}",
    peft_fan_in_fan_out=False,
)

DEFAULT_FAMILY_SPECS = {
    GPT2_DEFAULTS.name: GPT2_DEFAULTS,
    PYTHIA_DEFAULTS.name: PYTHIA_DEFAULTS,
}
