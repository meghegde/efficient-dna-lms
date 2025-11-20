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

# Constants
seed = 42
tok_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
config_path = "configs/base.json"
model_variant = "base"
seq_length = 512
pretrained_weights = "checkpoints/elc-bert-base_len-512_1000-steps/model.bin"
model_name = f"{model_variant}_len-{seq_length}_{seed}"
odir = f"./trained_models/{model_name}/model.bin"
logdir = f"./logs/{model_name}.txt"
tblogdir = f"./tblogs/{model_name}"

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

# Define metrics
def compute_metrics(eval_pred):
    # eval_pred is (preds, labels) from Trainer
    preds, labels = eval_pred

    # preds can be logits or dict; normalize to ndarray
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.asarray(preds)

    # Case A: two-logit softmax head -> take argmax over classes
    if preds.ndim == 2 and preds.shape[1] == 2:
        pred_ids = preds.argmax(axis=1).astype(np.int32)

    # Case B: single-logit sigmoid head -> threshold at 0.5
    elif preds.ndim == 2 and preds.shape[1] == 1:
        pred_ids = (preds.squeeze(1) > 0.5).astype(np.int32)

    # Case C: already class IDs
    elif preds.ndim == 1:
        pred_ids = preds.astype(np.int32)

    else:
        raise ValueError(f"Unexpected preds shape {preds.shape}")

    # Labels to int32
    labels = np.asarray(labels).astype(np.int32)
    # Flatten if shape is (B,1)
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    # Compute accuracy directly (avoid extra format errors)
    accuracy = (pred_ids == labels).mean().item()

    return {"accuracy": accuracy}

# Define training arguments
# Create Early Stopping callback
es = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=odir,
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_safetensors=False,
    save_total_limit=2,  # save only the two latest checkpoints
    save_strategy="epoch",
    load_best_model_at_end=True,  # This saves also the best performing one
    logging_dir=tblogdir,  # <-- directory for TensorBoard logs
    logging_strategy="steps",  # log every n steps
    logging_steps=50,
    # Arguments for early stopping
    metric_for_best_model='loss',
    greater_is_better=False,
)

# Create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[es]
)

# Train model
trainer.train()

# Save trained model
trainer.save_model(odir)

# Save training logs
history = trainer.state.log_history
logSave = open(logdir, 'w')
logSave.write(str(history))
logSave.close()

