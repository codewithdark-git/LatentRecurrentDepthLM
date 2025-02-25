import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from Model.prelude_Block import PreludeBlock
from Model.recurrent_Block import RecurrentBlock
from Model.codaBlock import CodaBlock

# Full Latent Recurrent Depth Model
class LatentRecurrentDepthLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.prelude = PreludeBlock(vocab_size, d_model, num_heads, dropout)
        self.recurrent = RecurrentBlock(d_model, num_heads, dropout)
        self.coda = CodaBlock(d_model, vocab_size)

    def forward(self, x: torch.Tensor, num_iterations: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.prelude(x, mask)
        recurrent_state = torch.zeros_like(hidden)
        for _ in range(num_iterations):
            hidden, recurrent_state = self.recurrent(hidden, recurrent_state, mask)
        return self.coda(hidden)