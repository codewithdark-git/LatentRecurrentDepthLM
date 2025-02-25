import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# Trainer class
class Trainer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.losses = []

    def train(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            num_iterations = torch.randint(1, 5, (1,)).item()
            mask = self.create_attention_mask(inputs).to(self.device)
            outputs = self.model(inputs, num_iterations, mask)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        self.losses.append(avg_loss)
        return avg_loss
    
    def evaluate_perplexity(self, data_loader, num_iterations):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs, num_iterations=num_iterations)
                # Compute cross-entropy loss without reduction, to sum over tokens.
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='sum')
                total_loss += loss.item()
                total_tokens += targets.numel()
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()

    @staticmethod
    def create_attention_mask(x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask

    def generate(self, prompt, max_length=50, num_iterations=3, temperature=0.7):
        self.model.eval()
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(max_length):
                mask = self.create_attention_mask(input_ids).to(self.device)
                outputs = self.model(input_ids, num_iterations, mask)
                next_token_logits = outputs[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.tokenizer.tokenizer.token_to_id("[EOS]"):
                    break
        return self.tokenizer.decode(input_ids[0].cpu().tolist())
    
    def save(self, path):
        '''Save only weights of the model'''
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses[-1]
        }, f'SaveModel/{path}.pt')
        print(f'Save the Model Weights in SaveModel/{path}.pt')


    def save_model(self, path):
        '''Save the full model'''
        torch.save(self.model, f'SaveModel/{path}.pt')
        print(f'Save the Full Model in SaveModel/{path}.pt')

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
