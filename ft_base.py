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
#from torch.optim import SGD

# Constants
seed = 42
#seed = 41
tok_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
config_path = "configs/newbase.json"
model_variant = "base"
num_epochs = 10
# seq_length = 512
seq_length = 768
# seq_length = 1024
# pretrained_weights = "checkpoints/elc-bert-base-len_512/model.bin"
pretrained_weights = "checkpoints/elc-bert-base-len_768/model.bin"
# pretrained_weights = "checkpoints/elc-bert-base-len_1024/model.bin"
# pretrained_weights = "checkpoints/elc-bert-base_tinydnabert/model.bin"
# model_name = f"{model_variant}_len-{seq_length}_{seed}"
model_name = f"{model_variant}_len-{seq_length}_{seed}_{num_epochs}_epochs"
odir = f"./trained_models/{model_name}"
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

    # Compute F1 score
    f1 = f1_score(labels, pred_ids, average="weighted")

    return {"accuracy": accuracy, "f1": f1}

# Define training arguments
# Create Early Stopping callback
# es = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

# Set up training arguments
training_args = TrainingArguments(
    num_train_epochs=num_epochs,
    output_dir=odir,
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-05,
    warmup_ratio=0.15, # Ratio of total steps to warm up from 0 to learning_rate
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

## Change optimiser from default
#optimiser = SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Create trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
#    callbacks=[es],
#    optimizers=(optimiser, None)
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

# Sequence length 512
#{'eval_loss': 0.6932368278503418, 'eval_accuracy': 0.4979227487087357, 'eval_runtime': 57.7483, 'eval_samples_per_second': 308.442, 'eval_steps_per_second': 19.291, 'epoch': 3.0}
#{'train_runtime': 2432.7478, 'train_samples_per_second': 87.861, 'train_steps_per_second': 5.491, 'train_loss': 0.711148133525253, 'epoch': 3.0}

# Sequence length 1024 (pretraining truncated at approx 90% due to GPU OOM)
# {'eval_loss': 0.6932483911514282, 'eval_accuracy': 0.4979227487087357, 'eval_runtime': 57.6915, 'eval_samples_per_second': 308.745, 'eval_steps_per_second': 19.31, 'epoch': 3.0}
# {'train_runtime': 4296.2988, 'train_samples_per_second': 49.751, 'train_steps_per_second': 3.109, 'train_loss': 0.71148204332305, 'epoch': 3.0}

# Sequence length 1024 - increase warmup_steps to 100
#{'eval_loss': 0.6932516694068909, 'eval_accuracy': 0.4979227487087357, 'eval_runtime': 59.4099, 'eval_samples_per_second': 299.815, 'eval_steps_per_second': 18.751, 'epoch': 3.0}
#{'train_runtime': 4290.8268, 'train_samples_per_second': 49.814, 'train_steps_per_second': 3.113, 'train_loss': 0.7118884804131198, 'epoch': 3.0}

## Pretraining on TinyDNABERT dataset - max_length=1024 - once again, pretraining truncated at 90% due to GPU OOM
#{'eval_loss': 0.6932060122489929, 'eval_accuracy': 0.4979227487087357, 'eval_runtime': 59.2382, 'eval_samples_per_second': 300.684, 'eval_steps_per_second': 18.805, 'epoch': 3.0}
#{'train_runtime': 4256.5717, 'train_samples_per_second': 50.215, 'train_steps_per_second': 3.138, 'train_loss': 0.7108661194156053, 'epoch': 3.0}

# Sequence length 1024
# Updated TrainingArguments to:
# Set up training arguments
#training_args = TrainingArguments(
#    output_dir=odir,
#    eval_strategy="epoch",
#    per_device_train_batch_size=16,
#    per_device_eval_batch_size=16,
#    learning_rate=5e-05,
#    warmup_ratio=0.15, # Ratio of total steps to warm up from 0 to learning_rate
#    save_safetensors=False,
#    save_total_limit=2,  # save only the two latest checkpoints
#    save_strategy="epoch",
#    load_best_model_at_end=True,  # This saves also the best performing one
#    logging_dir=tblogdir,  # <-- directory for TensorBoard logs
#    logging_strategy="steps",  # log every n steps
#    logging_steps=50,
#    # Arguments for early stopping
#    metric_for_best_model='loss',
#    greater_is_better=False,
#)
#
#{'eval_loss': 0.6932055950164795, 'eval_accuracy': 0.4979227487087357, 'eval_f1': 0.33102783690999416, 'eval_runtime': 38.734, 'eval_samples_per_second': 459.855, 'eval_steps_per_second': 14.38, 'epoch': 3.0}
#{'train_runtime': 3875.5308, 'train_samples_per_second': 55.152, 'train_steps_per_second': 1.724, 'train_loss': 0.7169729754375566, 'epoch': 3.0}

# Sequence length 1024, change optimiser to SGD
#{'eval_loss': 0.6931596994400024, 'eval_accuracy': 0.4979227487087357, 'eval_f1': 0.33102783690999416, 'eval_runtime': 38.3342, 'eval_samples_per_second': 464.651, 'eval_steps_per_second': 14.53, 'epoch': 3.0}
#{'train_runtime': 3851.4623, 'train_samples_per_second': 55.497, 'train_steps_per_second': 1.735, 'train_loss': 0.7503687098182937, 'epoch': 3.0}

# Sequence length 1024, back to default optimiser, change seed to 41
#{'eval_loss': 0.6932083964347839, 'eval_accuracy': 0.4979227487087357, 'eval_f1': 0.33102783690999416, 'eval_runtime': 38.7528, 'eval_samples_per_second': 459.632, 'eval_steps_per_second': 14.373, 'epoch': 3.0}
#{'train_runtime': 3876.2487, 'train_samples_per_second': 55.142, 'train_steps_per_second': 1.724, 'train_loss': 0.7147506787523219, 'epoch': 3.0}

# Sequence length 1024 - remove early stopping and train for 10 epochs instead
# {'eval_loss': 0.693282961845398, 'eval_accuracy': 0.4979227487087357, 'eval_f1': 0.33102783690999416, 'eval_runtime': 38.692, 'eval_samples_per_second': 460.354, 'eval_steps_per_second': 14.396, 'epoch': 10.0}
# {'train_runtime': 12893.6039, 'train_samples_per_second': 55.258, 'train_steps_per_second': 1.727, 'train_loss': 0.7039725468519508, 'epoch': 10.0}

## Sequence length 1024 - train for 50 epochs
#{'eval_loss': 0.6931859850883484, 'eval_accuracy': 0.4979227487087357, 'eval_f1': 0.33102783690999416, 'eval_runtime': 38.8959, 'eval_samples_per_second': 457.941, 'eval_steps_per_second': 14.32, 'epoch': 50.0}
#{'train_runtime': 64596.4465, 'train_samples_per_second': 55.149, 'train_steps_per_second': 1.724, 'train_loss': 0.696911261720833, 'epoch': 50.0}

# Sequence length 768 - train for 10 epochs (N.B. pretraining truncated at 90% due to GPU OOM)
#{'eval_loss': 0.693278431892395, 'eval_accuracy': 0.4979227487087357, 'eval_f1': 0.33102783690999416, 'eval_runtime': 33.679, 'eval_samples_per_second': 528.875, 'eval_steps_per_second': 16.538, 'epoch': 10.0}
#{'train_runtime': 9961.9863, 'train_samples_per_second': 71.52, 'train_steps_per_second': 2.235, 'train_loss': 0.7040464336581973, 'epoch': 10.0}

