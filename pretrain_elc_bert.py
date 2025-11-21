# coding=utf-8

import os
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from transformers import AutoTokenizer
from pre_training.lamb import Lamb
from pre_training.config import BertConfig

from pre_training.utils import (
    cosine_schedule_with_warmup,
    is_main_process,
    get_rank,
    seed_everything,
    get_world_size,
)
from pre_training.text_file_dataset import Dataset

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_variant",
        default="base",
        type=str,
        help="The model variant to use; base, zero, normalized, or weighted_output."
    )
    parser.add_argument(
        "--input_path",
        default="./data/processed/cached_{sequence_length}.txt",
        type=str,
        help="The input data dir. Should be the cached text file.",
    )
    parser.add_argument(
        "--config_file",
        default="./configs/base.json",
        type=str,
        help="The BERT model config",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints/elc_bert",
        type=str,
        help="The output directory where the model checkpoints \
            will be written.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./tokenizer.json",
        type=str,
        help="The vocabulary the BERT model will train on.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Path to a previous checkpointed training state.",
    )

    # Other parameters
    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="The optimizer to use during pre-training (lamb or adamw).",
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        type=str,
        help="(Not implemented)The learning scheduler to use during training (cosine).",
    )
    parser.add_argument(
        "--seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after \
            WordPiece tokenization. Sequences longer than this \
            will be truncated, and sequences shorter than this \
            will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Total batch size for training per GPUs and per \
            grad accumulation step.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-2,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--max_steps",
        default=31250 // 2,
        type=int,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--long_after",
        default=0.9,
        type=float,
        help="The fraction of steps after which to quadruple the sequence length.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.016,
        type=float,
        help="Proportion of training to perform linear \
            learning rate warmup for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--log_freq", type=int, default=10, help="frequency of logging loss."
    )
    parser.add_argument(
        "--mask_p", default=0.15, type=float, help="Masking probability."
    )
    parser.add_argument(
        "--short_p", default=0.1, type=float, help="Short sequence probability."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="Fraction of weight decay to apply. (Should be between 0 and 1)",
    )
    parser.add_argument(
        "--max_gradient",
        default=2.0,
        type=float,
        help="Max value for gradient clipping.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        default=1,
        type=int,
        help="The number of gradient accumulation steps to do.",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0,
        type=float,
        help="The label smoothing to apply to apply to cross-entropy.",
    )
    args = parser.parse_args()

    return args

@torch.no_grad()

def setup_training(args):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")

    if is_main_process():
        tok_per_batch = args.batch_size * args.seq_length
        print(
            f"Training for {args.max_steps:,} steps with {get_world_size()} \
                GPUs"
        )
        print(
            f"In total, the model will be trained on 'steps'\
        ({args.max_steps:,}) x 'GPUs'({get_world_size()}) x \
        'batch_size'({args.batch_size:,}) x 'seq_len'\
        ({args.seq_length:,}) = \
        {args.max_steps * get_world_size() * tok_per_batch:,} \
        subword instances"
        )

    args.device_max_steps = args.max_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 1

    return device, local_rank

def prepare_model_and_optimizer(args, device, local_rank, checkpoint):

    if args.model_variant == "zero":
        from models.model_elc_bert_zero import Bert
    elif args.model_variant == "normalized":
        from models.model_elc_bert_normalized import Bert
    elif args.model_variant == "weighted_output":
        from models.model_elc_bert_weighted_output import Bert
    else:
        from models.model_elc_bert_base import Bert

    config = BertConfig(args.config_file)
    model = Bert(config, args.activation_checkpointing)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    no_decay = ["bias", "layer_norm", "embedding", "prev_layer_weights"]
    high_no = ["res"]
    decay_params = [
        (n, p)
        for n, p in model.named_parameters()
        if (not any(nd in n for nd in no_decay) and not any(hn in n for hn in high_no))
    ]
    no_decay_params = [
        (n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
    ]
    high_no_decay_params = [
        (n, p) for n, p in model.named_parameters() if any(hn in n for hn in high_no)
    ]
    optimizer_grouped_parameters = [
        {"params": [p for _, p in decay_params], "weight_decay": args.weight_decay},
        {"params": [p for _, p in no_decay_params], "weight_decay": 0.0},
        {
            "params": [p for _, p in high_no_decay_params],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 1,
        },
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print()
        print("Parameters with no weight decay and high learning rate:")
        for n, _ in high_no_decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    scheduler = cosine_schedule_with_warmup(
        optimizer,
        int(args.device_max_steps * args.warmup_proportion),
        args.device_max_steps,
        0.1,
    )

    # Get rank info from torchrun
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",      # for GPU
            init_method="env://"
        )

    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun sets this automatically
    torch.cuda.set_device(local_rank)

    model = model.to(local_rank)

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    return model, config, optimizer, scheduler, grad_scaler


def training_epoch(
    model,
    tokenizer,
    data,
    optimizer,
    scheduler,
    grad_scaler,
    global_step,
    epoch,
    args,
    device,
    max_local_steps,
):
    seed = args.seed + get_rank() + epoch * get_world_size()
    train_dataloader = create_train_dataloader(data, args, global_step, seed)

    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0
    avg_accuracy = 0

    if is_main_process():
        current_step = global_step * args.gradient_accumulation
        max_steps = args.device_max_steps * args.gradient_accumulation
        train_iter = tqdm(
            train_dataloader,
            desc="Train iteration",
            initial=current_step,
            total=max_steps,
        )
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True).bool()
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.cuda.amp.autocast(args.mixed_precision):
            model.train()
            prediction = model(input_ids, attention_mask, target_ids)
            pred_flat = prediction.reshape(-1, prediction.size(-1))
            target_flat = target_ids.reshape(-1)

            # mask out the non-masked tokens (targets == -100)
            mask = target_flat != -100
            if mask.sum().item() == 0:
                # nothing to predict in this batch
                continue

            masked_pred = pred_flat[mask]
            masked_target = target_flat[mask]

#            target_ids = target_ids.flatten()
#            target_ids = target_ids[target_ids != -100]
#            loss = F.cross_entropy(
#                prediction, target_ids, label_smoothing=args.label_smoothing
#            )
            loss = F.cross_entropy(masked_pred, masked_target, label_smoothing=args.label_smoothing)
            loss /= args.gradient_accumulation
            total_loss += loss.item()

        with torch.no_grad():
#            accuracy = (prediction.argmax(-1) == target_ids).float().mean()
#            avg_accuracy += accuracy.item() / args.gradient_accumulation
            # compute correct tokens
            preds = masked_pred.argmax(dim=-1)
            correct = (preds == masked_target).long().sum().item()
            count = masked_target.numel()
            running_correct += correct
            running_total += count

        grad_scaler.scale(loss).backward()
        if (local_step + 1) % args.gradient_accumulation == 0:
            grad_scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

            return_value = grad_scaler.step(optimizer)
            grad_scaler.update()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if return_value is None:
                continue

            scheduler.step()

            if is_main_process():
                train_iter.set_postfix_str(
                    f"loss: {total_loss:.2f}, \
                                    accuracy: {avg_accuracy * 100.0:.2f}, \
                                    grad_norm: {grad_norm:.2f}, \
                                    lr: {optimizer.param_groups[0]['lr']:.5f}"
                )

                # if global_step % 100 == 0:
                #     log_parameter_histograms(model, global_step)

                total_loss = 0
                avg_accuracy = 0

        if (
            global_step == int(args.device_max_steps * args.long_after)
            and (local_step + 1) % args.gradient_accumulation == 0
        ):
            optimizer.zero_grad(set_to_none=True)
            return global_step

        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            optimizer.zero_grad(set_to_none=True)
            return global_step

    optimizer.zero_grad(set_to_none=True)
    return global_step

def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    if args.output_dir == "./checkpoints/elc_bert":
        checkpoint_path = f"{args.output_dir}_{args.model_variant}"
    else:
        checkpoint_path = f"{args.output_dir}/model.bin"
    if is_main_process():
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path,
        )

    return checkpoint_path


def load_dataset(args, tokenizer, device):
    seq_length = (
        args.seq_length * 4
        if global_step >= int(args.device_max_steps * args.long_after)
        else args.seq_length
    )
    # Using new text_file_dataset.py
    train_data = Dataset(args.input_path, tokenizer, max_length=seq_length)

    print(f"Loaded training file {get_rank()}", flush=True)
    print(f"First item: {train_data.__getitem__(0)}")

    batch_size = (
        args.batch_size // 4
        if global_step > args.device_max_steps * args.long_after
        else args.batch_size
    )
    min_length = torch.tensor(
        len(train_data) // batch_size, dtype=torch.long, device=device
    )
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    return train_data, min_length


def create_train_dataloader(data, args, global_step, seed):
#    batch_size = (
#        args.batch_size // 4
#        if global_step >= int(args.device_max_steps * args.long_after)
#        else args.batch_size
#    )
    batch_size = args.batch_size

    def collate_fn(batch):
        """
        batch: list of items returned by Dataset
        Each item is a dict: {'input_ids': tensor, 'attention_mask': tensor}
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        # Add target_ids if needed for MLM, e.g., same as input_ids or with masking
        target_ids = input_ids.clone()  # for MLM, or generate masked labels here
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids
        }

    # DataLoader
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=7-1,
        generator=torch.Generator().manual_seed(seed), drop_last=True, pin_memory=True, collate_fn=collate_fn)

    return train_dataloader

if __name__ == "__main__":
    args = parse_arguments()
    args.mixed_precision = True
    args.activation_checkpointing = False

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint["args"]
        initial_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_path)
    device, local_rank = setup_training(args)
    model, config, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(
        args, device, local_rank, checkpoint
    )
    train_data, min_length = load_dataset(args, tokenizer, device)

    for epoch in count(initial_epoch):
        if global_step == int(args.device_max_steps * args.long_after):
            train_data, min_length = load_dataset(args, tokenizer, device)

        global_step = training_epoch(
            model,
            tokenizer,
            train_data,
            optimizer,
            scheduler,
            grad_scaler,
            global_step,
            epoch,
            args,
            device,
            min_length,
        )
        checkpoint_path = save(
            model, optimizer, grad_scaler, scheduler, global_step, epoch, args
        )

        if global_step >= args.device_max_steps:
            break
