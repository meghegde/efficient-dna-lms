import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_elc_bert_weighted_output import Bert

class BertForBinaryClassification(nn.Module):
    def __init__(self, config, num_labels=2, activation_checkpointing=False):
        super().__init__()
        self.bert = Bert(config, activation_checkpointing)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _to_bool_mask(mask):
        # Convert to bool without allocating new tensor if already bool
        return (mask != 0) if mask.dtype != torch.bool else mask

    @staticmethod
    def _normalize_attention_mask(attention_mask, input_ids):
        # Ensure 2D [B, L] bool mask, removing extra singleton dims
        B, L = input_ids.shape
        if attention_mask is None:
            return torch.ones(B, L, dtype=torch.bool, device=input_ids.device)

        # Move to same device and bool dtype
        attention_mask = attention_mask.to(input_ids.device)
        attention_mask = (attention_mask != 0) if attention_mask.dtype != torch.bool else attention_mask

        # Squeeze all singleton dims safely
        # Example shapes seen: [B,1,1,1,1,L], [B,1,1,L], [B,L]
        while attention_mask.dim() > 2:
            attention_mask = attention_mask.squeeze(1) if attention_mask.size(1) == 1 else attention_mask.squeeze()

        # Coerce to [B, L]
        if attention_mask.dim() == 1:
            # Could be [L] or [B]
            if attention_mask.size(0) == L:
                attention_mask = attention_mask.unsqueeze(0).expand(B, L)
            elif attention_mask.size(0) == B:
                attention_mask = attention_mask.unsqueeze(1).expand(B, L)
            else:
                raise ValueError(f"1D attention_mask shape {attention_mask.shape} incompatible with input_ids {input_ids.shape}")
        elif attention_mask.shape != (B, L):
            attention_mask = attention_mask.reshape(B, L)

        return attention_mask

    def forward(self, input_ids, attention_mask=None, labels=None):

        # Normalize mask to [B, L] (bool) before passing to backbone
        attention_mask = self._normalize_attention_mask(attention_mask, input_ids)

        # contextualized embeddings: [batch, seq_len, hidden]
        last_hidden = self.bert.get_contextualized(input_ids, attention_mask)[-1]

        # use CLS token (first position)
        cls_emb = last_hidden[:, 0, :]  # [batch, hidden]

        logits = self.classifier(self.dropout(cls_emb))  # [batch, num_labels]

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    @classmethod
    def from_pretrained(cls, checkpoint_path, config, num_labels=2, map_location="cpu"):
        # config must be a config object, not a Bert
        model = cls(config, num_labels=num_labels)
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=False)
        return model
