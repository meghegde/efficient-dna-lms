import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_elc_bert_base import Encoder, Embedding

class BertForBinaryClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)  # Binary
  
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1=attend, 0=pad)
        labels: [B]
        """

        if isinstance(input_ids, tuple):
            print(f"input ids before: {input_ids}")
            input_ids = input_ids[0]
            print(f"input ids after: {input_ids}, {input_ids.shape}")

        x = self.embedding(input_ids).transpose(0, 1)
        x = self.encoder(x, attention_mask)
        x = x.mean(dim=0)
        x = torch.tanh(self.pooler(x))
        logits = self.classifier(x).squeeze(-1)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {"logits": logits, "loss": loss}

    @classmethod
    def from_pretrained(cls, model_bin_path, config, map_location="cpu"):
        """
        Load pretrained model from model.bin file.
        """

        # Create a fresh model object
        model = cls(config) # cls = class; self = instance
        # Load state dict
        state_dict = torch.load(model_bin_path, map_location=map_location)
        # Load with strict=False to allow missing/unexpected keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)

        return model