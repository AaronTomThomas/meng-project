

from typing import Dict

from experiments.router_development.attention_adapter.adapters.base import AdapterModel
import torch
from torch import nn


class OfficialPEFTAdapter(AdapterModel):
    """Interface adapter for Hugging Face PEFT models."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids.to(self.device)).logits

    def set_peft_train_mode(self) -> None:
        self.model.eval()
        for name, module in self.model.named_modules():
            if "lora_dropout" in name:
                module.train()

    def set_peft_eval_mode(self) -> None:
        self.model.eval()

    @torch.no_grad()
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        del input_ids
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return {}
        vec = torch.cat([p.detach().reshape(-1).float().cpu() for p in params])
        return {
            "peft_param_abs_mean": float(vec.abs().mean().item()),
            "peft_param_l2_rms": float(vec.pow(2).mean().sqrt().item()),
        }

