import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from Model.multi_head_Attention import MultiHeadAttention


# Prelude Block (Initial Processing)
class PreludeBlock(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.token_embedding(x) + self.pos_encoding[:, :seq_len, :]
        attended = self.attention(self.norm1(x), mask)
        x = x + attended
        return x + self.feed_forward(self.norm2(x))