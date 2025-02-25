import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Final Projection Block
class CodaBlock(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.norm(x))