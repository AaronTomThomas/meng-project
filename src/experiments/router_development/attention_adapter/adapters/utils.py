from typing import Dict

import torch


def delta_stats(deltas: Dict[int, torch.Tensor]) -> Dict[str, float]:
    if not deltas:
        return {}
    flat = torch.cat([delta.detach().reshape(-1).float().cpu() for delta in deltas.values()])
    return {
        "delta_abs_mean": float(flat.abs().mean().item()),
        "delta_l2_rms": float(flat.pow(2).mean().sqrt().item()),
    }


