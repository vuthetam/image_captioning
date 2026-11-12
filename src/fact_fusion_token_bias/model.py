"""Keep Fact RAG fusion and adjust logits with gates on fact token candidates."""

from torch import Tensor

from src.fact_fusion_token_bias.decoder import TokenBiasCaptionDecoder
from src.shared.facts.fusion import FactFusionEncoder


class FactFusionTokenBiasCaptioner(FactFusionEncoder):
    def __init__(
        self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
        dropout: float, max_length: int, pad_idx: int,
        visual_feature_dim: int, fact_feature_dim: int, fusion_layers: int = 1,
        initial_alpha: float = 0.1,
    ):
        super().__init__(visual_feature_dim, fact_feature_dim, d_model, nheads, dropout, fusion_layers)
        self.decoder = TokenBiasCaptionDecoder(
            vocab_size, d_model, nheads, nlayers, dropout, max_length, pad_idx,
            initial_alpha=initial_alpha,
        )

    def encode_memory(
        self, visual_inputs: Tensor, fact_features: Tensor, fact_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """Only fused image tokens form decoder memory, as in Fact RAG."""
        memory, _, _ = self.encode_fusion(
            visual_inputs, fact_features, fact_attention_mask, include_cls_token=include_cls_token,
        )
        return memory

    def forward(
        self, visual_inputs: Tensor, input_ids: Tensor, attention_mask: Tensor,
        fact_features: Tensor, fact_attention_mask: Tensor,
        candidate_token_ids: Tensor, candidate_token_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        memory = self.encode_memory(
            visual_inputs, fact_features, fact_attention_mask, include_cls_token=include_cls_token,
        )
        logits = self.decoder(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], memory=memory,
            candidate_token_ids=candidate_token_ids, candidate_token_mask=candidate_token_mask,
        )
        return logits
