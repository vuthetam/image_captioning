import math

import torch
from torch import Tensor, nn

from src.fact_rag.models import FactRAGCaptioner
from src.shared.decoder import PositionalEncoding


class FactTokenRAGCaptioner(FactRAGCaptioner):
    """Append contextualized fact tokens to unchanged image-fact fusion memory."""

    def __init__(self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
                 dropout: float, max_length: int, pad_idx: int,
                 visual_feature_dim: int, fact_feature_dim: int, context_layers: int,
                 max_context_length: int, fusion_layers: int = 1):
        super().__init__(
            vocab_size, d_model, nheads, nlayers, dropout, max_length, pad_idx,
            visual_feature_dim, fact_feature_dim, fusion_layers,
        )
        self.context_pos_encoding = PositionalEncoding(d_model, max_context_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dropout=dropout, batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=context_layers, enable_nested_tensor=False,
        )

    def encode_memory(self, visual_inputs: Tensor, fact_features: Tensor,
                      fact_attention_mask: Tensor, rag_input_ids: Tensor,
                      rag_attention_mask: Tensor, include_cls_token: bool = False
                      ) -> tuple[Tensor, Tensor]:
        if rag_input_ids.ndim != 2 or rag_input_ids.shape != rag_attention_mask.shape:
            raise ValueError("Fact context IDs and attention mask must align.")
        if rag_input_ids.size(0) != visual_inputs.size(0):
            raise ValueError("Visual and fact context batch sizes must match.")
        fused_memory = super().encode_memory(
            visual_inputs, fact_features, fact_attention_mask,
            include_cls_token=include_cls_token,
        )
        visual_padding_mask = torch.zeros(
            fused_memory.shape[:2], dtype=torch.bool, device=fused_memory.device,
        )
        context_padding_mask = rag_attention_mask == 0
        has_context = rag_attention_mask.ne(0).any(dim=1)
        if not has_context.any():
            return fused_memory, visual_padding_mask

        # All-padding rows must not enter Transformer attention. Encode only
        # active rows; the supplementary slots of other rows remain masked.
        active_indices = has_context.nonzero(as_tuple=True)[0]
        active_padding_mask = context_padding_mask[has_context]
        hidden = self.decoder.embedding(rag_input_ids[has_context]) * math.sqrt(self.decoder.d_model)
        hidden = self.context_pos_encoding(hidden)
        active_memory = self.context_encoder(hidden, src_key_padding_mask=active_padding_mask)
        active_memory = active_memory.masked_fill(active_padding_mask.unsqueeze(-1), 0)
        context_memory = active_memory.new_zeros(
            rag_input_ids.size(0), rag_input_ids.size(1), active_memory.size(2),
        )
        context_memory = context_memory.index_copy(0, active_indices, active_memory)
        memory = torch.cat([fused_memory, context_memory], dim=1)
        memory_padding_mask = torch.cat(
            [visual_padding_mask, context_padding_mask], dim=1,
        )
        return memory, memory_padding_mask

    def forward(self, visual_inputs: Tensor, input_ids: Tensor, attention_mask: Tensor,
                fact_features: Tensor, fact_attention_mask: Tensor, rag_input_ids: Tensor,
                rag_attention_mask: Tensor, include_cls_token: bool = False) -> Tensor:
        memory, memory_padding_mask = self.encode_memory(
            visual_inputs, fact_features, fact_attention_mask, rag_input_ids, rag_attention_mask,
            include_cls_token=include_cls_token,
        )
        return self.decoder(
            input_ids=input_ids[:, :-1], memory=memory, attention_mask=attention_mask[:, :-1],
            memory_key_padding_mask=memory_padding_mask,
        )

