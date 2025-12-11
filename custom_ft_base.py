import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from models.base_clf_wrapper import BertForBinaryClassification
from models.model_elc_bert_base import Bert
from pre_training.config import BertConfig
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import csv
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
seed = 42
tok_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
config_path = "configs/newbase.json"
model_variant = "base"
seq_length = 1024
num_epochs = 15
pretrained_weights = "checkpoints/elc-bert-base-len_1024/model.bin"
model_name = f"{model_variant}_len-{seq_length}_{seed}_{num_epochs}_epochs"
odir = f"./custom_training_loop/{model_name}"
os.makedirs(odir, exist_ok=True)
logdir = f"./custom_training_loop/{model_name}.txt"

# Set random seed
transformers.set_seed(seed)

# Define custom dataset class
class VarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        if "attention_mask" in item:
            item["attention_mask"] = item["attention_mask"].bool()  # keep 1D per sample
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load tokeniser
tokeniser = AutoTokenizer.from_pretrained(tok_path)

# Load training dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="variant_effect_causal_eqtl",
    sequence_length=seq_length,
    split="train"
)
# Separate data and labels
df = pd.DataFrame.from_dict(dataset)
seqs = df['alt_forward_sequence'].to_list()
labels = df['label'].to_list()
# Split into training and validation
train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size=.2, shuffle=True, random_state=42)
# Encode data
train_encodings = tokeniser(train_seqs, padding=True, truncation=True, max_length=seq_length, return_tensors='pt')
val_encodings = tokeniser(val_seqs, padding=True, truncation=True, max_length=seq_length, return_tensors='pt')
# Convert to correct format
train_dataset = VarDataset(train_encodings, train_labels)
val_dataset = VarDataset(val_encodings, val_labels)

# Load model and pretrained weights
config = BertConfig(config_path)
config.num_labels = 2
model = BertForBinaryClassification(config, num_labels=2)
state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict, strict=False)
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# Write header of log file
with open(logdir, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

best_val_loss = float("inf")

train_start = time.time()
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # -------------------------
    # Training
    # -------------------------
    model.train()
    train_losses = []
    progress_bar = tqdm(range(len(train_loader)), desc="Training")

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        progress_bar.update(1)

    avg_train_loss = sum(train_losses) / len(train_losses)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_losses.append(outputs["loss"].item())

    avg_val_loss = sum(val_losses) / len(val_losses)

    print(f"  Training Loss:   {avg_train_loss:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")

    # ---- Save log to CSV ----
    with open(logdir, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

    # -------------------------
    # Save Best Model
    # -------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(odir, "pytorch_model.bin"))
        print(f"  📌 New best model saved (val loss: {best_val_loss:.4f}) → {odir}")

train_end = time.time()
train_length = train_end - train_start
print('Training time: ', train_length)

# 15 epochs
# Training time:  36828.23901748657
