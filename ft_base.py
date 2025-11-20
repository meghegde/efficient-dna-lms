import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from models.base_clf_wrapper import BertForBinaryClassification
from models.model_elc_bert_base import Bert
from pre_training.config import BertConfig
from transformers import Trainer, TrainingArguments

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
tokeniser = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

# Load training dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="variant_effect_causal_eqtl",
    sequence_length=512,
    split="train"
)
# Separate data and labels
df = pd.DataFrame.from_dict(dataset)
df = df.head(50)
seqs = df['alt_forward_sequence'].to_list()
labels = df['label'].to_list()
# Split into training and validation
train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size=.2, shuffle=True, random_state=42)
# Encode data
train_encodings = tokeniser(train_seqs, padding=True, truncation=True, max_length=512, return_tensors='pt')
val_encodings = tokeniser(val_seqs, padding=True, truncation=True, max_length=512, return_tensors='pt')
# Convert to correct format
train_dataset = VarDataset(train_encodings, train_labels)
val_dataset = VarDataset(val_encodings, val_labels)

config_path = "configs/base.json"
config = BertConfig(config_path)
config.num_labels = 2
model = BertForBinaryClassification(config)
state_dict = torch.load("checkpoints/elc-bert-base_len-512_1000-steps/model.bin", map_location="cpu", weights_only=False)
model.load_state_dict(state_dict, strict=False)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    save_safetensors=False,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

torch.save(model.state_dict(), 'trained_models/base.pt')

