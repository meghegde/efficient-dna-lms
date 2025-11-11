import torch
from transformers import BertForSequenceClassification, BertConfig

def load_custom_model(model_bin_path, config_path):
    # 1️⃣ Load config
    config = BertConfig.from_json_file(config_path)
    model = BertForSequenceClassification(config)
    
    # 2️⃣ Load checkpoint
    checkpoint = torch.load(model_bin_path, map_location="cpu")

    # Handle full checkpoint structure
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    # 3️⃣ Filter only matching keys with same shapes
    model_dict = model.state_dict()
    matched_state_dict = {}
    skipped_keys = []

    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_state_dict[k] = v
            else:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    # 4️⃣ Load filtered weights
    model.load_state_dict(matched_state_dict, strict=False)

    # 5️⃣ Print what was skipped
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys due to mismatched shapes or missing params:")
        for k in skipped_keys:
            print("  -", k)
    else:
        print("✅ All compatible weights loaded successfully!")

    return model

