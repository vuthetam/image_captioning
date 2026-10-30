"""Shared image-to-fact fusion; facts remain available as decoder memory."""

import torch
from torch import Tensor, nn


class FactFusionBlock(nn.Module):
    """Image-to-fact cross-attention and position-wise FFN, both post-norm."""

    def __init__(self, d_model: int, nheads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual: Tensor, facts: Tensor, padding_mask: Tensor):
        attended, _ = self.attention(
            visual, facts, facts, key_padding_mask=padding_mask, need_weights=False,
        )
        hidden = self.attention_norm(visual + self.dropout(attended))
        memory = self.ffn_norm(hidden + self.dropout(self.ffn(hidden)))
        return memory


class FactFusionEncoder(nn.Module):
    """Keep projector/fusion parameter names compatible with original Fact RAG."""

    def __init__(self, visual_feature_dim: int, fact_feature_dim: int, d_model: int,
                 nheads: int, dropout: float, fusion_layers: int = 1):
        super().__init__()
        if fusion_layers < 1:
            raise ValueError("fusion_layers must be positive.")
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, d_model), nn.LayerNorm(d_model),
        )
        self.fact_projector = nn.Sequential(
            nn.Linear(fact_feature_dim, d_model), nn.LayerNorm(d_model),
        )
        self.fusion = nn.ModuleList([
            FactFusionBlock(d_model, nheads, dropout) for _ in range(fusion_layers)
        ])

    def encode_fusion(self, visual_inputs: Tensor, fact_features: Tensor,
                      fact_attention_mask: Tensor, include_cls_token: bool = False
                      ) -> tuple[Tensor, Tensor, Tensor]:
        if visual_inputs.ndim != 3:
            raise ValueError("Visual features must have shape [batch, tokens, feature_dim].")
        if not include_cls_token and visual_inputs.size(1) < 2:
            raise ValueError("Visual features must contain CLS followed by patch tokens.")
        if fact_features.ndim != 3 or fact_features.shape[:2] != fact_attention_mask.shape:
            raise ValueError("Fact features and their attention mask must align.")
        if fact_features.size(0) != visual_inputs.size(0):
            raise ValueError("Visual and fact batch sizes must match.")
        if fact_features.size(1) == 0:
            raise ValueError("Fact features must contain at least one slot.")
        valid = fact_attention_mask.to(dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid fact.")

        visual_tokens = visual_inputs if include_cls_token else visual_inputs[:, 1:]
        visual_memory = self.visual_projector(
            visual_tokens.to(dtype=self.visual_projector[0].weight.dtype),
        )
        fact_memory = self.fact_projector(
            fact_features.to(dtype=self.fact_projector[0].weight.dtype),
        )
        fact_padding_mask = ~valid
        fact_memory = fact_memory.masked_fill(fact_padding_mask.unsqueeze(-1), 0)
        fused_memory = visual_memory
        for block in self.fusion:
            fused_memory = block(fused_memory, fact_memory, fact_padding_mask)
        return fused_memory, fact_memory, fact_padding_mask
