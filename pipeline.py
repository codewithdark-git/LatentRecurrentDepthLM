import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from trainer import Trainer
from dataset import TextDataset, load_dataset
from tokenizer import create_tokenizer
from push_to_hub import save_model_to_hub
from Model.model import LatentRecurrentDepthConfig, LatentRecurrentDepthModel


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load datasets
train_texts, val_texts = load_dataset()
print("Training tokenizer...")
train_tokenizer = create_tokenizer(train_texts)
val_tokenizer = create_tokenizer(val_texts)
tr_vocab_size = train_tokenizer.get_vocab_size()
vl_vocab_size = val_tokenizer.get_vocab_size()
print(f"Training Vocabulary size: {tr_vocab_size}")
print(f"Testing Vocabulary size: {vl_vocab_size}")

# Convert text to datasets
train_dataset = TextDataset(train_texts, train_tokenizer)
val_dataset = TextDataset(val_texts, val_tokenizer)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# Initialize Model
print("Initializing model...")
# Initialize model
print("Initializing model...")
config = LatentRecurrentDepthConfig(vocab_size=50257, d_model=768, num_heads=12, dropout=0.1)

# Instantiate the model
model = LatentRecurrentDepthModel(config)

# Start Training
trainer = Trainer(model, train_tokenizer)
num_epochs = 10

logger.info("Starting training...")
for epoch in range(num_epochs):
    loss = trainer.train(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    if (epoch + 1) % 2 == 0:
        sample = trainer.generate("The main purpose of", max_length=30)
        print(f"\nGenerated Sample:\n{sample}")

# Save the model weights
trainer.save("latent_lm_checkpoint")

# Save model the full model
trainer.save_model("latent_lm_checkpoint")

print("Training completed! Model saved as latent_lm_checkpoint.pt")

# Generate final samples
print("\nFinal text generation samples:")
prompts = ["The history of", "Scientists discovered", "In recent years,"]

for prompt in prompts:
    generated = trainer.generate(prompt, max_length=50)
    print(f"\nPrompt: {prompt}\nGenerated: {generated}")

# Evaluate model perplexity
perplexity = trainer.evaluate_perplexity(val_dataloader, num_iterations=3)
print(f"Perplexity on validation set: {perplexity:.2f}")

# Save the model and tokenizer to the hub
# before push the model and tokenizer to the hub you need to login -> huggingface-cli login -- token <Token here>
save_model_to_hub(model, train_tokenizer, "latent-recurrent-depth-lm")
print("Model and tokenizer saved to the hub!")

# inference locally
from Inference.Squence_Generator import generator

prompt = "In the realm of language modeling"
output = generator(prompt, num_interations=3, max_length=50, temperature=0.7)
print(f"Generated text: {output}")

