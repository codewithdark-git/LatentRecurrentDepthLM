import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# Load dataset
def load_dataset():
    print("Loading dataset...")
    train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5000]")["text"]
    val_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:5000]")["text"]

    # Remove empty strings and very short texts
    train_texts = [text for text in train_data if len(text.strip()) > 50]
    val_texts = [text for text in val_data if len(text.strip()) > 50]
    return train_texts, val_texts

# Dataset preparation
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        for text in texts:
            encoding = tokenizer.encode(text)
            if len(encoding) > max_length:
                encoding = encoding[:max_length]
            elif len(encoding) < max_length:
                encoding += [tokenizer.tokenizer.token_to_id("[PAD]")] * (max_length - len(encoding))
            self.encodings.append(encoding)
        self.encodings = torch.tensor(self.encodings)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = self.encodings[idx]
        return item[:-1], item[1:]


