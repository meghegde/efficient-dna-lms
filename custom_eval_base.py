import transformers
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)
from models.model_elc_bert_base import Bert
from pre_training.config import BertConfig
from models.base_clf_wrapper import BertForBinaryClassification
import evaluate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score

transformers.set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
#MAX_LEN = 512
MAX_LEN = 1024
TASK_NAME = "variant_effect_causal_eqtl"
TOK_PATH = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
MODEL_NAME = "./custom_training_loop/base_len-1024_42_15_epochs/pytorch_model.bin"
CONFIG_PATH = "./configs/newbase.json"

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split='test'
)
df = pd.DataFrame.from_dict(dataset)
seqs = df['alt_forward_sequence'].to_list()
labels = df['label'].to_list()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
config = BertConfig(CONFIG_PATH)
model = BertForBinaryClassification.from_pretrained(checkpoint_path=MODEL_NAME, config=config, num_labels=2, map_location="cuda")
model.to(device)
model.eval()

def predict_proba(model, tokenizer, texts):
    model.eval()
    all_probs = []

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            out = model(**enc)

        # --- FIX: handle dict OR tuple ---
        if isinstance(out, dict):
            logits = out["logits"]
        else:
            # tuple case: logits is first element
            logits = out[0]

        probs = F.softmax(logits, dim=-1)[0].cpu()
        all_probs.append(probs)

    return torch.stack(all_probs)


def predict_classes(probs):
    return probs.argmax(dim=-1).tolist()

def accuracy(preds, labels):
    return (np.array(preds) == np.array(labels)).mean()

def compute_f1(preds, labels, average="macro"):
    """
    average = "binary", "macro", "micro", or "weighted"
    """
    return f1_score(labels, preds, average=average)

def compute_auroc(probs, labels):
    """
    probs: tensor [N, num_classes]
    labels: list or array of ints
    """
    probs = probs.numpy()
    labels = np.array(labels)

    num_classes = probs.shape[1]

    if num_classes == 2:
        # AUROC for binary classification: use positive class probability
        return roc_auc_score(labels, probs[:, 1])
    else:
        # multiclass: one-vs-rest AUROC
        return roc_auc_score(labels, probs, multi_class="ovr")

probs = predict_proba(model, tokenizer, seqs)
preds = predict_classes(probs)

acc = accuracy(preds, labels)
f1  = compute_f1(preds, labels, average="binary")
auc = compute_auroc(probs, labels)

print("Predicted classes :", preds)
print("Probabilities     :\n", probs)
print("Accuracy:", acc)
print("F1 Score:", f1)
print("AUROC:", auc)

# Base model - 3 epochs
# All predicted classes were 0
#Accuracy: 0.48589483186639587
#F1 Score: 0.0
#AUROC: 0.5

# Base model - 15 epochs
# All predicted classes were 1
#Accuracy: 0.5141051681336042
#F1 Score: 0.6790877925175138
#AUROC: 0.5
