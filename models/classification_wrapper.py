import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

# ----------------------------
# Masked Softmax
# ----------------------------
class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, dim=-1):
        x = x.masked_fill(mask == 0, float("-inf"))
        softmax = torch.softmax(x, dim=dim)
#        softmax = softmax.masked_fill(mask == 0, 0.0)
        ctx.save_for_backward(softmax)
        ctx.dim = dim
        return softmax

    @staticmethod
    def backward(ctx, grad_output):
        (softmax,) = ctx.saved_tensors
        dim = ctx.dim
 #       grad_input = torch._C._nn.softmax_backward(grad_output, softmax, dim)
        grad_input = softmax * (grad_output - (grad_output * softmax).sum(dim=dim, keepdim=True))
        return grad_input, None, None

# ----------------------------
# Attention
# ----------------------------
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_size)

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask):
        L, B, D = x.size()
        x = self.layer_norm(x)

        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.reshape(L, B * self.num_heads, self.head_size).transpose(0, 1)
        k = k.reshape(L, B * self.num_heads, self.head_size).transpose(0, 1)
        v = v.reshape(L, B * self.num_heads, self.head_size).transpose(0, 1)

        # Attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # Mask: [B, L] -> [B*H, 1, L]
        mask = attention_mask.unsqueeze(1).expand(B, self.num_heads, L)
        mask = mask.reshape(B * self.num_heads, 1, L)

        attn_probs = MaskedSoftmax.apply(attn_scores, mask, -1)
        attn_probs = self.dropout(attn_probs)

        # Context
        context = torch.bmm(attn_probs, v)
        context = context.transpose(0, 1).reshape(L, B, D)
        context = self.out_proj(context)
        return context

# ----------------------------
# FeedForward
# ----------------------------
class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate, approximate="tanh")

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, x):
        return self.mlp(x)

# ----------------------------
# Encoder Layer
# ----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, attention_mask):
        x = x + self.attention(x, attention_mask)
        x = x + self.mlp(x)
        return x

# ----------------------------
# Encoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, attention_mask):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

# ----------------------------
# Embeddings
# ----------------------------
class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):
        x = self.word_embedding(input_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

# ----------------------------
# Bert for Binary Classification
# ----------------------------
class BertForBinaryClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)  # Binary

    def forward(self, input_ids, attention_mask, labels=None):
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1=attend, 0=pad)
        labels: [B]
        """
        x = self.embedding(input_ids).transpose(0, 1)  # [L, B, D]
        x = self.encoder(x, attention_mask)
        x = x.mean(dim=0)  # Mean pooling over sequence
        x = torch.tanh(self.pooler(x))
        logits = self.classifier(x).squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {"logits": logits, "loss": loss}

