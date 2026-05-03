from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from experiments.router_development.attention_adapter.adapters import (
    GPT2AKAZAAdapter,
    OfficialPEFTAdapter,
    PythiaAKAZAAdapter,
)
from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig, AdapterMethod
from experiments.router_development.attention_adapter.models import (
    DEFAULT_FAMILY_SPECS,
    ModelFamilyDefaults,
)
from peft import LoraConfig, TaskType, get_peft_model


METHODS = tuple(AdapterMethod)


def build_wrapped_model(
    *,
    model: nn.Module,
    cfg: AdapterFineTuneConfig,
    layer_indices: Sequence[int],
):
    defaults = DEFAULT_FAMILY_SPECS[cfg.model_family]
    if not isinstance(cfg.method, AdapterMethod):
        choices = [method.value for method in METHODS]
        raise TypeError(f"cfg.method must be an AdapterMethod enum, got {cfg.method!r}; choices={choices}")
    if cfg.method is AdapterMethod.AKAZA_FREEZ:
        if cfg.model_family == "gpt2":
            return GPT2AKAZAAdapter(model=model, cfg=cfg, layer_indices=layer_indices)
        if cfg.model_family == "pythia":
            return PythiaAKAZAAdapter(model=model, cfg=cfg, layer_indices=layer_indices)
        raise ValueError(f"AKAZA is not implemented for model_family={cfg.model_family!r}")
    if cfg.method in {AdapterMethod.LORA}:
        return build_official_peft_model(model=model, cfg=cfg, defaults=defaults, layer_indices=layer_indices)
    choices = [method.value for method in METHODS]
    raise ValueError(f"Unknown method={cfg.method!r}; choices={choices}")


def build_official_peft_model(
    *,
    model: nn.Module,
    cfg: AdapterFineTuneConfig,
    defaults: ModelFamilyDefaults,
    layer_indices: Sequence[int],
) -> OfficialPEFTAdapter:
    if cfg.method is AdapterMethod.LORA:
        if cfg.peft_target_profile not in defaults.lora_target_profiles:
            raise ValueError(
                f"LoRA profile {cfg.peft_target_profile!r} not in {sorted(defaults.lora_target_profiles)}"
            )
        suffixes = defaults.lora_target_profiles[cfg.peft_target_profile]
        target_modules = defaults.target_module_names(list(layer_indices), suffixes)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            fan_in_fan_out=defaults.peft_fan_in_fan_out,
            bias=cfg.lora_bias,
            init_lora_weights=True,
        )
    else:
        raise ValueError(f"Unsupported official PEFT method: {cfg.method}")
    print(f"[peft] method={cfg.method}")
    print(f"[peft] family={defaults.name}")
    print(f"[peft] target_profile={cfg.peft_target_profile}")
    print(f"[peft] target_modules={target_modules}")

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return OfficialPEFTAdapter(peft_model)
