
from dataclasses import dataclass
from experiments.attention_learners import LearnerHyperParams
import torch


@dataclass
class EvalConfig(LearnerHyperParams):
    L: int = 128
    d: int = 32
    dv: int = 16
    batch_size: int = 128
    sigma: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    min_context: int = 8
    retention_decay: float = 0.9



@dataclass
class RouterExperimentConfig(EvalConfig):
    """Configuration for router experiments; currently inherits EvalConfig."""
    pass

def _randn(*shape, cfg: EvalConfig):
    return torch.randn(*shape, device=cfg.device, dtype=cfg.dtype)
