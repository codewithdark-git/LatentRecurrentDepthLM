# prompt: save the model with pretrain and tokenizer to push to hub

from huggingface_hub import HfApi, Repository
import os
from transformers import PreTrainedTokenizerFast
import torch

def save_model_to_hub(model, tokenizer, repo_id):
    
    # Initialize the Hugging Face API
    api = HfApi()

    repo_id = "codewithdark/latent-recurrent-depth-lm"  # Replace with your desired repo ID

    # Create the repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, private=False)
    except Exception as e:
        print(f"Repo likely already exists: {e}")

    # Create a local clone of the repo
    repo_local_path = f"SaveModel/latent-recurrent-depth-lm"
    repo = Repository(local_dir=repo_local_path, clone_from=repo_id)

    # Wrap the tokenizer with PreTrainedTokenizerFast for compatibility
    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer) # Wrap tokenizer

    # Save tokenizer using the wrapped tokenizer
    wrapped_tokenizer.save_pretrained(repo_local_path) # Save with wrapped tokenizer

    # Save model using torch.save
    # torch.save(model.state_dict(), os.path.join(repo_local_path, "model.pt"))
    model.save_pretrained(repo_local_path)

    # Add and commit files to the repo
    repo.push_to_hub(commit_message="Initial model and tokenizer commit")

    print(f"Model and tokenizer pushed to {repo_id}")