import transformers
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)
from models.model_elc_bert_zero import Bert
from pre_training.config import BertConfig
from models.zero_clf_wrapper import BertForBinaryClassification
import evaluate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix

transformers.set_seed(42)

# Constants
MAX_LEN = 512
TASK_NAME = "variant_effect_causal_eqtl"
TOK_PATH = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
MODEL_NAME = "./trained_models/zero_len-512_42/model.bin/pytorch_model.bin"
CONFIG_PATH = "./configs/base.json"

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split='test'
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
config = BertConfig(CONFIG_PATH)
model = BertForBinaryClassification.from_pretrained(checkpoint_path=MODEL_NAME, config=config, num_labels=2, map_location="cuda")
model.eval()

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(
        examples["alt_forward_sequence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print(type(encoded_dataset))

# TrainingArguments for evaluation only
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=24,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
    report_to="none",
)

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


# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate
eval_results = trainer.evaluate()
print(eval_results)

# Zero-initialised model, pretrained for 1000 steps, then fine-tuned to convergence
# {'eval_loss': 0.6933684945106506, 'eval_accuracy': 0.48589483186639587, 'eval_runtime': 46.8836, 'eval_samples_per_second': 189.021, 'eval_steps_per_second': 3.946}
