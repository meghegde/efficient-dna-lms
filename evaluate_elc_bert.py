import os
import argparse
import transformers
import torch
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
from models.classification_wrapper import BertForBinaryClassification
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        help="Path to the finetuned model to evaluate."
    )
    parser.add_argument(
        "--dataset",
        default="eQTL",
        type=str,
        help="The dataset on which to fine-tune the model."
    )
    # Other arguments
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialisation"
    )
    args = parser.parse_args()
    return args

def load_data(args, tokeniser):

    def preprocess_function(seqs):
        return tokeniser(
            seqs["alt_forward_sequence"],
            padding="max_length",
            truncation=True,
            max_length=args.seq_length,
        )

    if args.dataset=="eQTL":
        # Load training dataset
        dataset = load_dataset(
            "InstaDeepAI/genomics-long-range-benchmark",
            task_name="variant_effect_causal_eqtl",
            sequence_length=args.seq_length,
            split="test"
        )
        encoded_dataset = dataset.map(preprocess_function, batched=True)
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return encoded_dataset

def load_model(args):
    config_path = "configs/base.json"
    config = BertConfig(config_path)
    model = BertForBinaryClassification.from_pretrained(model_bin_path=args.finetuned_model_path, config=config, map_location="cuda")
    model.eval()
    return model

def compute_metrics(eval_pred):

    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
    }

def setup_training(tokeniser, model, encoded_dataset):
    # TrainingArguments for evaluation only
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=24,
        do_train=False,
        do_eval=True,
        logging_dir="./logs",
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokeniser)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=encoded_dataset,
        tokenizer=tokeniser,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set random seed
    transformers.set_seed(args.seed)

    # Constants
    tok_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

    # Load tokeniser
    tokeniser = AutoTokenizer.from_pretrained(tok_path)

    # Load data
    dataset = load_data(args, tokeniser)

    # Load model
    model = load_model(args)

    # Set up training
    trainer = setup_training(tokeniser, model, dataset)

    # Evaluate model
    eval_results = trainer.evaluate()
    print(eval_results)
