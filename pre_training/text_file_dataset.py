import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, "r") as f:
            self.sequences = [line.strip() for line in f if line.strip()]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in encoding.items()}

