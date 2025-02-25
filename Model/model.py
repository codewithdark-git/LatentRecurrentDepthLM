import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from transformers import PretrainedConfig, PreTrainedModel
from Model.latent_Recurrent import LatentRecurrentDepthLM

# Configuration for the Latent Recurrent Depth Model
class LatentRecurrentDepthConfig(PretrainedConfig):
    model_type = "latent_recurrent_depth"

    def __init__(self, vocab_size=50257, d_model=768, num_heads=12, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout


# Hugging Face-Compatible Model Wrapper
class LatentRecurrentDepthModel(PreTrainedModel):
    config_class = LatentRecurrentDepthConfig
    base_model_prefix = "latent_recurrent_depth"

    def __init__(self, config: LatentRecurrentDepthConfig):
        super().__init__(config)
        self.latent_model = LatentRecurrentDepthLM(config.vocab_size, config.d_model, config.num_heads, config.dropout)
        self.init_weights()

    def forward(self, input_ids: torch.Tensor, num_iterations: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.latent_model(input_ids, num_iterations, mask)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 20,
        num_iterations: int = 3,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens given input_ids.

        Args:
          input_ids: torch.Tensor of shape (batch, seq_length) containing the prompt.
          max_length: The number of tokens to generate.
          num_iterations: The number of recurrent iterations to use in each forward pass.
          temperature: Temperature for scaling logits.
          top_k: If set, only sample from the top k tokens.

        Returns:
          generated: torch.Tensor containing the generated sequence.
        """
        generated = input_ids.clone()
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits from the model for the current sequence.
                logits = self.forward(generated, num_iterations=num_iterations)
                # Use only the logits for the last token in the sequence.
                next_token_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    # Top-k filtering
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probabilities = F.softmax(top_k_logits, dim=-1)
                    next_token = top_k_indices.gather(-1, torch.multinomial(probabilities, num_samples=1))
                else:
                    probabilities = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                # Optionally, break if the EOS token is generated.
                if next_token.item() == self.config.eos_token_id:
                    break
        return generated
