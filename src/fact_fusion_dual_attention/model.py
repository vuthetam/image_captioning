"""Original Fact fusion plus adapted attribute/triplet memory for the decoder."""

from torch import Tensor

from src.fact_fusion_dual_attention.adapter import FactAdapter
from src.fact_fusion_dual_attention.decoder import DualAttentionCaptionDecoder
from src.shared.facts.fusion import FactFusionEncoder


class FactFusionDualAttentionCaptioner(FactFusionEncoder):
    def __init__(self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
                 dropout: float, max_length: int, pad_idx: int, visual_feature_dim: int,
                 fact_feature_dim: int, fusion_layers: int = 1, initial_gate: float = 0.1):
        super().__init__(visual_feature_dim, fact_feature_dim, d_model, nheads, dropout, fusion_layers)
        self.fact_adapter = FactAdapter(d_model, dropout)
        self.decoder = DualAttentionCaptionDecoder(
            vocab_size, d_model, nheads, nlayers, dropout, max_length, pad_idx, initial_gate,
        )

    def encode_memory(self, visual_inputs: Tensor, fact_features: Tensor, fact_attention_mask: Tensor,
                      attr_triplet_fact_indices: Tensor, attr_triplet_fact_mask: Tensor,
                      include_cls_token: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        fused_memory, projected_fact_memory, projected_fact_padding_mask = self.encode_fusion(
            visual_inputs, fact_features, fact_attention_mask, include_cls_token=include_cls_token,
        )
        if attr_triplet_fact_indices.ndim != 2 or attr_triplet_fact_indices.shape != attr_triplet_fact_mask.shape:
            raise ValueError("Direct fact indices and mask must have shape [batch, direct facts].")
        if attr_triplet_fact_indices.size(0) != projected_fact_memory.size(0):
            raise ValueError("Direct indices and fusion facts must have the same batch size.")

        valid = attr_triplet_fact_mask.bool()
        indices = attr_triplet_fact_indices.masked_fill(~valid, 0)
        if ((indices < 0) | (indices >= projected_fact_memory.size(1))).any():
            raise ValueError("A direct fact index is outside the fusion fact memory.")
        if (projected_fact_padding_mask.gather(1, indices) & valid).any():
            raise ValueError("A direct fact index refers to fusion padding.")

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, projected_fact_memory.size(2))
        fact_memory = projected_fact_memory.gather(1, gather_indices)
        if valid.any():
            fact_memory = self.fact_adapter(fact_memory)
        fact_padding_mask = ~valid
        fact_memory = fact_memory.masked_fill(fact_padding_mask.unsqueeze(-1), 0)
        return fused_memory, fact_memory, fact_padding_mask

    def forward(self, visual_inputs: Tensor, input_ids: Tensor, attention_mask: Tensor,
                fact_features: Tensor, fact_attention_mask: Tensor, attr_triplet_fact_indices: Tensor,
                attr_triplet_fact_mask: Tensor, include_cls_token: bool = False) -> Tensor:
        fused_memory, fact_memory, fact_padding_mask = self.encode_memory(
            visual_inputs, fact_features, fact_attention_mask,
            attr_triplet_fact_indices, attr_triplet_fact_mask,
            include_cls_token=include_cls_token,
        )
        logits = self.decoder(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], memory=fused_memory,
            fact_memory=fact_memory, fact_padding_mask=fact_padding_mask,
        )
        return logits
