from torch import Tensor
import torch.nn as nn
import math
from torch.nn import functional as F

class SoftmaxKernelLearner(nn.Module):
    def __init__(self, dk: int):
        super().__init__()
        self.dk = dk
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None):
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_mask is not None:
            scores = scores + attn_mask.to(dtype=scores.dtype)
        A = F.softmax(scores, dim=-1)
        out = A @ v
        return out, A