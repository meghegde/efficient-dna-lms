import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from pre_training.config import BertConfig
# from models.base_clf_wrapper import BertForBinaryClassification
# from models.zero_clf_wrapper import BertForBinaryClassification
from models.normalized_clf_wrapper import BertForBinaryClassification

def encoder_connectivity(model, example_input):
    encoder_layers = list(model.bert.transformer.layers)  # 12 BertLayer blocks
    n = len(encoder_layers)

    M = np.zeros((n, n), dtype=int)

    # Will store: tensor → layer index it came from
    tensor_owner = {}

    hooks = []

    for idx, layer in enumerate(encoder_layers):
        def hook(layer, inputs, output, idx=idx):
            # Mark outputs as owned by this encoder layer
            if isinstance(output, torch.Tensor):
                tensor_owner[output] = idx
            elif isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        tensor_owner[o] = idx

            # Record dependencies: this layer uses outputs of other layers
            if isinstance(inputs, tuple):
                for inp in inputs:
                    if isinstance(inp, torch.Tensor) and inp in tensor_owner:
                        M[idx, tensor_owner[inp]] = 1
                    elif isinstance(inp, (tuple, list)):
                        for x in inp:
                            if isinstance(x, torch.Tensor) and x in tensor_owner:
                                M[idx, tensor_owner[x]] = 1

        hooks.append(layer.register_forward_hook(hook))

    # Run one forward pass to gather connectivity
    model(**example_input)

    for h in hooks:
        h.remove()

    return M

# ---- Base model ----

# Constants
MAX_LEN = 512
TASK_NAME = "variant_effect_causal_eqtl"
TOK_PATH = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
# MODEL_NAME = "./trained_models/base_len-512_42/model.bin/pytorch_model.bin"
# MODEL_NAME = "./trained_models/zero_len-512_42/model.bin/pytorch_model.bin"
MODEL_NAME = "./trained_models/normalized_len-512_42/model.bin/pytorch_model.bin"
CONFIG_PATH = "./configs/base.json"

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split='test'
)
df = dataset.to_pandas()
seq = df["alt_forward_sequence"][0]
label = df["label"][0]

## Load tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
#config = BertConfig(CONFIG_PATH)
#model = BertForBinaryClassification.from_pretrained(checkpoint_path=MODEL_NAME, config=config, map_location="cuda")
#model.eval()
#print(model)

#inputs = tokenizer(seq, return_tensors="pt")
#
#M = encoder_connectivity(model, inputs)
#print("Shape:", M.shape)
#print(M)
 
# Load the model state dictionary
state_dict = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
# print(type(state_dict))
print(state_dict.keys())
# print(state_dict)
