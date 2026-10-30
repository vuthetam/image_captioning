import torch
from torch import Tensor, nn

from src.fact_memory_rag.decoder import FactMemoryRAGDecoder
from src.shared.facts.fusion import FactFusionEncoder


class FactMemoryRAGCaptioner(FactFusionEncoder):
    """Append the exact projected facts used as fusion K/V to the fused patches."""

    def __init__(self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
                 dropout: float, max_length: int, pad_idx: int,
                 visual_feature_dim: int, fact_feature_dim: int, fusion_layers: int = 1):
        super().__init__(
            visual_feature_dim, fact_feature_dim, d_model, nheads, dropout, fusion_layers,
        )
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.decoder = FactMemoryRAGDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nheads, num_layers=nlayers,
            dropout=dropout, max_length=max_length, pad_idx=pad_idx,
            embedding=self.shared_embedding,
        )

    def encode_memory(self, visual_inputs: Tensor, fact_features: Tensor,
                      fact_attention_mask: Tensor, include_cls_token: bool = False
                      ) -> tuple[Tensor, Tensor]:
        fused_memory, fact_memory, fact_padding_mask = self.encode_fusion(
            visual_inputs, fact_features, fact_attention_mask,
            include_cls_token=include_cls_token,
        )
        memory = torch.cat([fused_memory, fact_memory], dim=1)
        visual_padding_mask = torch.zeros(
            fused_memory.shape[:2], dtype=torch.bool, device=fused_memory.device,
        )
        memory_padding_mask = torch.cat(
            [visual_padding_mask, fact_padding_mask], dim=1,
        )
        return memory, memory_padding_mask

    def forward(self, visual_inputs: Tensor, input_ids: Tensor, attention_mask: Tensor,
                fact_features: Tensor, fact_attention_mask: Tensor,
                include_cls_token: bool = False) -> Tensor:
        memory, memory_padding_mask = self.encode_memory(
            visual_inputs, fact_features, fact_attention_mask,
            include_cls_token=include_cls_token,
        )
        logits = self.decoder(
            input_ids=input_ids[:, :-1], memory=memory,
            attention_mask=attention_mask[:, :-1],
            memory_key_padding_mask=memory_padding_mask,
        )
        return logits
