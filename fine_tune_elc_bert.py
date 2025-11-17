import os
import argparse
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from models.model_elc_bert_base import Bert
from pre_training.config import BertConfig
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
from load_custom_model import load_custom_model
from models.classification_wrapper import BertForBinaryClassification

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_variant",
        default="base",
        type="str",
        help="The model variant to use; base, zero, normalized, or weighted_output."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type="str",
        help="Path to the pretrained pytorch_model.bin."
    )
    parser.add_argument(
        "--dataset",
        default="eQTL",
        type="str",
        help="The dataset on which to fine-tune the model."
    )
    parser.add_argument(
        "--loss_convergence",
        default="True",
        type="str",
        help="Whether to train until loss converges"
    )
    # Other arguments
    parser.add_argument(
        "--seq_length",
        default=512
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Total batch size for training per GPUs and per \
            grad accumulation step.",
    )
    parser.add_argument(
        "--num_epochs",
        default=5
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
 
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

def load_dataset(args, tok_path):
    if args.dataset=="eQTL":
        # Load tokeniser
        tokeniser = AutoTokenizer.from_pretrained(tok_path)

        # Load training dataset
        dataset = load_dataset(
            "InstaDeepAI/genomics-long-range-benchmark",
            task_name="variant_effect_causal_eqtl",
            sequence_length=args.seq_length,
            split="train"
        )
        # Separate data and labels
        df = pd.DataFrame.from_dict(dataset)
        seqs = df['alt_forward_sequence'].to_list()
        labels = df['label'].to_list()
        # Split into training and validation
        train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size=.2, shuffle=True, random_state=42)
        # Encode data
        train_encodings = tokeniser(train_seqs, padding=True, truncation=True, max_length=args.seq_length, return_tensors='pt')
        val_encodings = tokeniser(val_seqs, padding=True, truncation=True, max_length=args.seq_length, return_tensors='pt')
        # Convert to correct format
        train_dataset = VarDataset(train_encodings, train_labels)
        val_dataset = VarDataset(val_encodings, val_labels)

    return train_dataset, val_dataset

def load_model(args):
    config_path = "configs/base.json"
    config = BertConfig(config_path)
    config.num_labels = 2
    model = BertForBinaryClassification(config)
    if os.path.exists(args.pretrained_model_path):
        state_dict = torch.load(args.pretrained_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Checkpoint {args.pretrained_model_path} not found. Initializing model from scratch.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Working on device: ', device)
    model.to(device)
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

def setup_training(args, odir, tblogdir, model, train_dataset, val_dataset):

    # Make directory for tensorboard logs if it doesn't exist
    os.makedirs(tblogdir, exist_ok=True)

    if args.loss_convergence.upper() == "TRUE":
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
    else:
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=odir,
            eval_strategy="epoch",
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            save_safetensors=False,
            save_total_limit=2,  # save only the two latest checkpoints
            save_strategy="epoch",
            load_best_model_at_end=True,  # This saves also the best performing one
            logging_dir=tblogdir,  # <-- directory for TensorBoard logs
            logging_strategy="steps",  # log every n steps
            logging_steps=50
        )

        # Create trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

    return trainer

def save_logs(trainer, logdir):
    history = trainer.state.log_history
    logSave = open(logdir, 'w')
    logSave.write(str(history))
    logSave.close()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Set random seed
    transformers.set_seed(args.seed)

    # Constants
    tok_path = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    model_name = f"{args.model_variant}_len-{args.seq_length}_{args.num_epochs}-epochs_{args.seed}"
    odir = f"./trained_models/{model_name}/model.bin"
    logdir = f"./logs/{model_name}.txt"
    tblogdir = f"./tblogs/{model_name}"

    # Load tokeniser
    tokeniser = AutoTokenizer.from_pretrained(tok_path)

    # Load data
    train_dataset, val_dataset = load_dataset(args)

    # Load model
    model = load_model(args)

    # Set up training
    trainer = setup_training(args, odir=odir, tblogdir=tblogdir, model=model, train_dataset=train_dataset, val_dataset=val_dataset)

    # Train model
    trainer.train()
    # Save trained model
    trainer.save_model(odir)
    # Save logs
    save_logs(trainer, logdir)

