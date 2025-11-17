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
from models.model_elc_bert_base import Bert
from pre_training.config import BertConfig
from models.classification_wrapper import BertForBinaryClassification
import evaluate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix

transformers.set_seed(42)

# Constants
MAX_LEN = 512
TASK_NAME = "variant_effect_causal_eqtl"
TOK_PATH = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
MODEL_NAME = "./trained_models/elc-bert-base_len-512_1000-steps_5_epochs_42/pytorch_model.bin"
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
model = BertForBinaryClassification.from_pretrained(model_bin_path=MODEL_NAME, config=config, map_location="cuda")
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

# Load accuracy metric
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
    }

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
