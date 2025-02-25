import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from Model.multi_head_Attention import MultiHeadAttention

# Recurrent Block (Processing Over Time)
class RecurrentBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.state_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, recurrent_state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        recurrent_state = self.state_proj(recurrent_state)
        x = x + recurrent_state
        attended = self.attention(self.norm1(x), mask)
        return x + attended + self.feed_forward(self.norm2(x)), x