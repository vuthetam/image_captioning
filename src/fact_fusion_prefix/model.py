"""Original fact fusion with prefix-conditioned caption decoding."""

from src.fact_fusion_prefix.decoder import PrefixCaptionDecoder
from src.shared.facts.fusion import FactFusionEncoder


class FactFusionPrefixCaptioner(FactFusionEncoder):
    def __init__(self, vocab_size, d_model, nheads, nlayers, dropout, max_length, pad_idx,
                 visual_feature_dim, fact_feature_dim, max_prefix_length, fusion_layers=1):
        super().__init__(visual_feature_dim, fact_feature_dim, d_model, nheads, dropout, fusion_layers)
        self.decoder = PrefixCaptionDecoder(
            vocab_size, d_model, nheads, nlayers, dropout, max_prefix_length + max_length - 1, pad_idx,
        )

    def encode_memory(self, visual_inputs, fact_features, fact_attention_mask, include_cls_token=False):
        fused_memory, _, _ = self.encode_fusion(
            visual_inputs=visual_inputs, fact_features=fact_features,
            fact_attention_mask=fact_attention_mask, include_cls_token=include_cls_token,
        )
        return fused_memory

    def forward(self, visual_inputs, input_ids, attention_mask, prefix_lengths, fact_features,
                fact_attention_mask, include_cls_token=False):
        fused_memory = self.encode_memory(
            visual_inputs=visual_inputs, fact_features=fact_features,
            fact_attention_mask=fact_attention_mask, include_cls_token=include_cls_token,
        )
        logits = self.decoder(
            input_ids=input_ids[:, :-1], memory=fused_memory,
            prefix_lengths=prefix_lengths, attention_mask=attention_mask[:, :-1],
        )

        return logits
