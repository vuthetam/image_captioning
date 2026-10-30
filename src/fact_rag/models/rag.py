from torch import Tensor

from src.shared.decoder import TransformerCaptionDecoder
from src.shared.facts.fusion import FactFusionBlock, FactFusionEncoder


class FactRAGCaptioner(FactFusionEncoder):
    def __init__(self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
                 dropout: float, max_length: int, pad_idx: int,
                 visual_feature_dim: int, fact_feature_dim: int, fusion_layers: int = 1):
        super().__init__(
            visual_feature_dim, fact_feature_dim, d_model, nheads, dropout, fusion_layers,
        )
        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nheads, num_layers=nlayers,
            dropout=dropout, max_length=max_length, pad_idx=pad_idx,
        )

    def encode_memory(self, visual_inputs: Tensor, fact_features: Tensor,
                      fact_attention_mask: Tensor, include_cls_token: bool = False) -> Tensor:
        memory, _, _ = self.encode_fusion(
            visual_inputs, fact_features, fact_attention_mask,
            include_cls_token=include_cls_token,
        )
        return memory

    def forward(self, visual_inputs: Tensor, input_ids: Tensor, attention_mask: Tensor,
                fact_features: Tensor, fact_attention_mask: Tensor,
                include_cls_token: bool = False) -> Tensor:
        memory = self.encode_memory(
            visual_inputs, fact_features, fact_attention_mask,
            include_cls_token=include_cls_token,
        )
        return self.decoder(
            input_ids=input_ids[:, :-1], memory=memory, attention_mask=attention_mask[:, :-1],
        )
