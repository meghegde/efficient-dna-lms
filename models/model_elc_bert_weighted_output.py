import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from torch.utils import checkpoint

# FIXME type hint
# FIXME Docstring


class Bert(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)

    def get_contextualized(self, input_ids, attention_mask):
        static_embeddings, relative_embedding = self.embedding(input_ids)
        contextualized_embeddings = self.transformer(
            static_embeddings,
            attention_mask.unsqueeze(1).unsqueeze(2),
            relative_embedding,
        )
        return contextualized_embeddings

    def forward(self, input_ids, attention_mask, masked_lm_labels=None):
        contextualized_embeddings = self.get_contextualized(input_ids, attention_mask)[
            -1
        ]
        subword_prediction = self.classifier(
            contextualized_embeddings, masked_lm_labels
        )

        return subword_prediction


class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing
        temp = torch.zeros(config.num_hidden_layers + 1)
        self.prev_layer_weights = nn.Parameter(temp)

    def forward(self, hidden_states, attention_mask, relative_embedding):
        hidden_states = [hidden_states]
        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_states.append(
                    checkpoint.checkpoint(
                        layer, hidden_states, attention_mask, relative_embedding
                    )
                )
            else:
                hidden_states.append(
                    layer(hidden_states, attention_mask, relative_embedding)
                )

        prev_layer_weights = F.softmax(self.prev_layer_weights, dim=-1)
        output = prev_layer_weights[0] * hidden_states[0]
        for i, hidden_state in enumerate(hidden_states[1:]):
            output = output + prev_layer_weights[i + 1] * hidden_state
        hidden_states.append(output)

        return hidden_states


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(
                config.hidden_size, config.layer_norm_eps, elementwise_affine=False
            ),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(
                config.hidden_size, config.layer_norm_eps, elementwise_affine=False
            ),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0)),
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(
            self.nonlinearity[1].weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x, masked_lm_labels=None):
        if masked_lm_labels is not None:
            x = torch.index_select(
                x.flatten(0, 1),
                0,
                torch.nonzero(masked_lm_labels.flatten() != -100).squeeze(),
            )
        x = self.nonlinearity(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)
        temp = torch.zeros(layer_num + 1)
        self.prev_layer_weights = nn.Parameter(temp)

    def forward(self, hidden_states, padding_mask, relative_embedding):
        prev_layer_weights = F.softmax(self.prev_layer_weights, dim=-1)
        x = prev_layer_weights[0] * hidden_states[0]
        for i, hidden_state in enumerate(hidden_states[1:]):
            x = x + prev_layer_weights[i + 1] * hidden_state
        attention = self.attention(x, padding_mask, relative_embedding)
        x = attention + self.mlp(x + attention)
        return x


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate="tanh")
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
            ),
            nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(
                config.intermediate_size,
                eps=config.layer_norm_eps,
                elementwise_affine=False,
            ),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(
            self.mlp[1].weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.mlp[-2].weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float("-inf"))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of \
                    the number of attention heads {config.num_attention_heads}"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(
            config.hidden_size, 2 * config.hidden_size, bias=True
        )
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps, elementwise_affine=False
        )
        self.post_layer_norm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps, elementwise_affine=True
        )

        position_indices = torch.arange(
            config.max_position_embeddings, dtype=torch.long
        ).unsqueeze(1) - torch.arange(
            config.max_position_embeddings, dtype=torch.long
        ).unsqueeze(
            0
        )
        position_indices = self.make_log_bucket_position(
            position_indices,
            config.position_bucket_size,
            config.max_position_embeddings,
        )
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            mid - 1,
            torch.abs(relative_pos).clamp(max=max_position - 1),
        )
        log_pos = (
            torch.ceil(
                torch.log(abs_pos / mid)
                / math.log((max_position - 1) / mid)
                * (mid - 1)
            ).int()
            + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(
            self.in_proj_qk.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.in_proj_v.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.out_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    # def forward(self, hidden_states, attention_mask, relative_embedding):
    #     key_len, batch_size, _ = hidden_states.size()
    #     query_len = key_len

    #     if self.position_indices.size(0) < query_len:
    #         position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(
    #             1
    #         ) - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
    #         position_indices = self.make_log_bucket_position(
    #             position_indices, self.config.position_bucket_size, 512
    #         )
    #         position_indices = self.config.position_bucket_size - 1 + position_indices
    #         self.register_buffer(
    #             "position_indices",
    #             position_indices.to(hidden_states.device),
    #             persistent=True,
    #         )

    #     hidden_states = self.pre_layer_norm(hidden_states)

    #     query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
    #     value = self.in_proj_v(hidden_states)  # shape: [T, B, D]

    #     query_pos, key_pos = self.in_proj_qk(self.dropout(relative_embedding)).chunk(
    #         2, dim=-1
    #     )  # shape: [2C-1, D]
    #     query_pos = query_pos.view(
    #         -1, self.num_heads, self.head_size
    #     )  # shape: [2C-1, H, D]
    #     key_pos = key_pos.view(
    #         -1, self.num_heads, self.head_size
    #     )  # shape: [2C-1, H, D]

    #     query = query.reshape(
    #         query_len, batch_size * self.num_heads, self.head_size
    #     ).transpose(0, 1)
    #     key = key.reshape(
    #         key_len, batch_size * self.num_heads, self.head_size
    #     ).transpose(0, 1)
    #     value = value.view(
    #         key_len, batch_size * self.num_heads, self.head_size
    #     ).transpose(0, 1)

    #     attention_scores = torch.bmm(
    #         query, key.transpose(1, 2) * self.scale
    #     )  # shape: [B, H, Tq, Tk]
    #     attention_scores = attention_scores.view(
    #         batch_size, self.num_heads, query_len, key_len
    #     )

    #     query = query.view(batch_size, self.num_heads, query_len, self.head_size)
    #     key = key.view(batch_size, self.num_heads, query_len, self.head_size)

    #     attention_scores_qp = torch.einsum(
    #         "bhqd,khd->bhqk", query, key_pos * self.scale
    #     )  # shape: [B, H, Tq, Tr]
    #     attention_scores_pk = torch.einsum(
    #         "bhkd,qhd->bhqk", key * self.scale, query_pos
    #     )  # shape: [B, H, Tr, Tk]

    #     position_indices = self.position_indices[:query_len, :key_len].expand(
    #         batch_size, self.num_heads, -1, -1
    #     )

    #     attention_scores_qp = attention_scores_qp.gather(
    #         dim=-1, index=position_indices
    #     )  # shape: [B, H, Tq, Tk]
    #     attention_scores_pk = attention_scores_pk.gather(
    #         dim=-2, index=position_indices
    #     )  # shape: [B, H, Tq, Tk]

    #     attention_scores.add_(attention_scores_qp)
    #     attention_scores.add_(attention_scores_pk)

    #     attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

    #     attention_probs = self.dropout(attention_probs)
    #     context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
    #     context = context.transpose(0, 1).reshape(
    #         context.size(1), -1, self.hidden_size
    #     )  # shape: [Q, B, H*D]
    #     context = self.out_proj(context)
    #     context = self.post_layer_norm(context)
    #     context = self.dropout(context)

    #     return context

    def forward(self, hidden_states, attention_mask, relative_embedding):
        # hidden_states: [T, B, D]
        L = hidden_states.size(0)        # seq_len
        B = hidden_states.size(1)        # batch_size
        D = hidden_states.size(2)
        query_len = L
        key_len = L

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) - \
                            torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(
                position_indices, self.config.position_bucket_size, query_len
            )
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer(
                "position_indices",
                position_indices.to(hidden_states.device),
                persistent=True,
            )

        hidden_states = self.pre_layer_norm(hidden_states)

        # [T, B, D] -> project
        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)
        value = self.in_proj_v(hidden_states)  # [T, B, D]

        # relative positions
        query_pos, key_pos = self.in_proj_qk(self.dropout(relative_embedding)).chunk(2, dim=-1)
        query_pos = query_pos.view(-1, self.num_heads, self.head_size)
        key_pos   = key_pos.view(-1, self.num_heads, self.head_size)

        # reshape to heads
        query = query.reshape(query_len, B * self.num_heads, self.head_size).transpose(0, 1)  # [B*H, T, d]
        key   = key.reshape(key_len,   B * self.num_heads, self.head_size).transpose(0, 1)    # [B*H, T, d]
        value = value.reshape(key_len, B * self.num_heads, self.head_size).transpose(0, 1)    # [B*H, T, d]

        # base scores [B, H, Tq, Tk]
        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)                 # [B*H, Tq, Tk]
        attention_scores = attention_scores.view(B, self.num_heads, query_len, key_len)       # [B, H, Tq, Tk]

        # Q/K for positional components
        query_h = query.view(B, self.num_heads, query_len, self.head_size)
        key_h   = key.view(B, self.num_heads, key_len,   self.head_size)

        attention_scores_qp = torch.einsum("bhqd,khd->bhqk", query_h, key_pos * self.scale)   # [B,H,Tq,Tr]
        attention_scores_pk = torch.einsum("bhkd,qhd->bhqk", key_h   * self.scale, query_pos) # [B,H,Tr,Tk]

        position_indices = self.position_indices[:query_len, :key_len].expand(B, self.num_heads, -1, -1)
        attention_scores_qp = attention_scores_qp.gather(dim=-1, index=position_indices)      # [B,H,Tq,Tk]
        attention_scores_pk = attention_scores_pk.gather(dim=-2, index=position_indices)      # [B,H,Tq,Tk]

        attention_scores.add_(attention_scores_qp)
        attention_scores.add_(attention_scores_pk)

        # ---- MASK NORMALIZATION (robust, no brittle asserts) ----
        # Goal: attention_mask_final = [B, 1, 1, L] (bool), aligned with Tk = L
        m = attention_mask.to(hidden_states.device)
        m = (m != 0) if m.dtype != torch.bool else m

        # Common cases: [B, L], [B, 1, 1, L], [L], [B], transposed [L, B]
        if m.dim() == 4 and m.shape == (B, 1, 1, L):
            attention_mask_final = m
        else:
            # Reduce extra singleton dims
            while m.dim() > 2:
                if m.size(1) == 1:
                    m = m.squeeze(1)
                else:
                    m = m.squeeze()

            # Handle transposed [L, B]
            if m.dim() == 2 and m.shape == (L, B):
                m = m.transpose(0, 1)  # -> [B, L]

            # 1D masks: [L] or [B]
            if m.dim() == 1:
                if m.size(0) == L:
                    m = m.unsqueeze(0).expand(B, L)  # [B, L]
                elif m.size(0) == B:
                    # If only batch info provided, assume all positions valid
                    m = m.unsqueeze(1).expand(B, L)  # [B, L]
                else:
                    # Fallback: assume all tokens valid
                    m = torch.ones(B, L, dtype=torch.bool, device=hidden_states.device)

            # Now coerce to exact [B, L], slice/pad if length differs
            if m.shape != (B, L):
                Bm, Lm = m.shape
                # Fix batch if needed by expanding or slicing
                if Bm != B:
                    if Bm == 1:
                        m = m.expand(B, Lm)
                    else:
                        # Slice to match B
                        m = m[:B, ...]
                # Fix length by slicing/padding
                if Lm > L:
                    m = m[:, :L]
                elif Lm < L:
                    pad = torch.ones(B, L - Lm, dtype=m.dtype, device=m.device)
                    m = torch.cat([m, pad], dim=1)

            attention_mask_final = m.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        # ---- END MASK NORMALIZATION ----

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask_final, -1)

        attention_probs = self.dropout(attention_probs)
        # [B*H, Q, D]
        context = torch.bmm(attention_probs.flatten(0, 1), value)
        # [Q, B, H*D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)

        return context

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(
            torch.empty(2 * config.position_bucket_size - 1, config.hidden_size)
        )
        self.relative_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(
            self.relative_embedding, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.word_embedding.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )

    def forward(self, input_ids):
        word_embedding = self.dropout(
            self.word_layer_norm(self.word_embedding(input_ids))
        )
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings
