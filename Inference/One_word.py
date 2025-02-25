from transformers import AutoModel, AutoTokenizer
# Also here you can the AutoModelForCausalLM class

# Replace with your repository name on the Hugging Face Hub.
repo_id = "codewithdark/latent-recurrent-depth-lm"

# Load the model and tokenizer from the hub.
model = AutoModel.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Use the model for inference. For example, generate text:
prompt = "In the realm of language modeling"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids

outputs = model(input_ids, num_iterations=3)
logits = outputs  # (batch, seq, vocab_size)

# You can now sample from logits to generate text.
import torch
probs = torch.softmax(logits[:, -1, :], dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
generated_ids = torch.cat([input_ids, next_token], dim=1)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
