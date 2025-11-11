import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from models.model_elc_bert_base import Bert
from pre_training.config import BertConfig
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
from load_custom_model import load_custom_model
from models.classification_wrapper import BertForBinaryClassification

# Set random seed
seed = 42
transformers.set_seed(seed)

# Constants
tok_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
seq_len = 1024
task_name = "variant_effect_causal_eqtl"
config_path = "configs/base.json"
model_name = f"elc-bert-base_len-1024_100-steps"
model_path = f"checkpoints/{model_name}/model.bin"
num_epochs = 5
odir = f"./trained_models/{model_name}_{num_epochs}_epochs_{seed}"
logdir = f"./logs/{model_name}_{num_epochs}_epochs_{seed}.txt"

# # Load and process data

# Load tokeniser
tokeniser = AutoTokenizer.from_pretrained(tok_path)

# Load training dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=task_name,
    sequence_length=seq_len,
    split="train"
)
# Separate data and labels
df = pd.DataFrame.from_dict(dataset)
seqs = df['alt_forward_sequence'].to_list()
labels = df['label'].to_list()
# Split into training and validation
train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size=.2, shuffle=True, random_state=42)
# Encode data
train_encodings = tokeniser(train_seqs, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
val_encodings = tokeniser(val_seqs, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')

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

train_dataset = VarDataset(train_encodings, train_labels)
val_dataset = VarDataset(val_encodings, val_labels)

# Load pretrained model
config = BertConfig(config_path)
config.num_labels = 2
model = BertForBinaryClassification(config)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # Set up training
# Set training arguments
training_args = TrainingArguments(
    output_dir=odir,
    eval_strategy="epoch",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_safetensors=False
    )

# Load accuracy metric
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
    }

# Create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save trained model
trainer.save_model(odir)

# Save logs
history = trainer.state.log_history
logSave = open(logdir, 'w')
logSave.write(str(history))
logSave.close()
