from abc import ABC, abstractmethod
from typing import Dict

from torch import nn
import torch


class AdapterModel(nn.Module, ABC):
    """Common interface used by the training loop for custom and official PEFT adapters."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def set_peft_train_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_peft_eval_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def peft_stats(self, input_ids: torch.Tensor | None = None) -> Dict[str, float]:
        raise NotImplementedError
