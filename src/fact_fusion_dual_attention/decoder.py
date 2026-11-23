"""Post-norm caption decoder with two parallel, gated cross-attentions."""

import math

import torch
from torch import Tensor, nn

from src.shared.decoder import PositionalEncoding


class DualCrossAttentionLayer(nn.TransformerDecoderLayer):
    """Keep standard self-attention/FFN and replace the cross-attention block."""

    def __init__(self, d_model: int, nhead: int, dropout: float, initial_gate: float = 0.1):
        if not math.isfinite(initial_gate) or not 0 < initial_gate < 1:
            raise ValueError("initial_gate must be finite and strictly between 0 and 1.")
        super().__init__(d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.fact_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_img_norm = nn.LayerNorm(d_model)
        self.gate_fact_norm = nn.LayerNorm(d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, math.log(initial_gate / (1 - initial_gate)))

    def forward(self, tgt: Tensor, memory: Tensor, fact_memory: Tensor, fact_padding_mask: Tensor,
                active_indices: Tensor, tgt_mask: Tensor, tgt_key_padding_mask: Tensor | None = None,
                memory_key_padding_mask: Tensor | None = None) -> Tensor:
        attended, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False,
        )
        hidden = self.norm1(tgt + self.dropout1(attended))

        image_context, _ = self.multihead_attn(
            hidden, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False,
        )
        context = image_context
        # Never send an all-padding fact row through attention.
        if active_indices.numel():
            active_hidden = hidden.index_select(0, active_indices)
            active_facts = fact_memory.index_select(0, active_indices)
            active_padding_mask = fact_padding_mask.index_select(0, active_indices)
            fact_context, _ = self.fact_attention(
                active_hidden, active_facts, active_facts, key_padding_mask=active_padding_mask, need_weights=False,
            )
            active_image_context = image_context.index_select(0, active_indices)
            gate_inputs = torch.cat([
                active_hidden, self.gate_img_norm(active_image_context), self.gate_fact_norm(fact_context),
            ], dim=-1)  # [active batch, steps, 3 * d_model]
            gate = self.gate_mlp(gate_inputs)  # [active batch, steps, 1]
            combined_context = (1 - gate) * active_image_context + gate * fact_context
            context = context.index_copy(0, active_indices, combined_context)

        hidden = self.norm2(hidden + self.dropout2(context))
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(hidden))))
        return self.norm3(hidden + self.dropout3(ffn_output))


class DualAttentionCaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
                 dropout: float, max_length: int, pad_idx: int, initial_gate: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList([
            DualCrossAttentionLayer(d_model, nhead, dropout, initial_gate) for _ in range(num_layers)
        ])
        self.out_fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor, memory: Tensor, fact_memory: Tensor, fact_padding_mask: Tensor,
                attention_mask: Tensor | None = None, memory_key_padding_mask: Tensor | None = None,
                return_last_step_only: bool = False) -> Tensor:
        if fact_memory.ndim != 3 or fact_memory.shape[:2] != fact_padding_mask.shape:
            raise ValueError("Fact memory and padding mask must align as [batch, facts, d_model].")
        if input_ids.size(0) != fact_memory.size(0) or memory.size(0) != input_ids.size(0):
            raise ValueError("Caption, visual memory and fact memory batch sizes must match.")

        hidden = self.embedding(input_ids) * math.sqrt(self.d_model)
        hidden = self.pos_encoding(hidden)
        causal_mask = torch.triu(
            torch.ones(input_ids.size(1), input_ids.size(1), dtype=torch.bool, device=input_ids.device), diagonal=1,
        )
        target_padding_mask = None
        if attention_mask is not None:
            target_padding_mask = attention_mask == 0
        fact_padding_mask = fact_padding_mask.bool()
        active_indices = (~fact_padding_mask.all(dim=1)).nonzero(as_tuple=True)[0]

        for layer in self.layers:
            hidden = layer(
                hidden, memory, fact_memory, fact_padding_mask, active_indices, causal_mask,
                tgt_key_padding_mask=target_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
            )
        if return_last_step_only:
            hidden = hidden[:, -1:, :]
        return self.out_fc(hidden)
