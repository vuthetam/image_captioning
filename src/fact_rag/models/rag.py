import torch
from torch import Tensor, nn

from src.shared.decoder import TransformerCaptionDecoder


class FactFusionBlock(nn.Module):
    """Image-to-fact cross-attention followed by a position-wise FFN (post-norm)."""

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
        attended, _ = self.attention(visual, facts, facts, key_padding_mask=padding_mask, need_weights=False)
        hidden = self.attention_norm(visual + self.dropout(attended))
        return self.ffn_norm(hidden + self.dropout(self.ffn(hidden)))


class FactRAGCaptioner(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
                 dropout: float, max_length: int, pad_idx: int,
                 visual_feature_dim: int, fact_feature_dim: int, fusion_layers: int = 1):
        super().__init__()
        if fusion_layers < 1:
            raise ValueError("fusion_layers must be positive.")
        self.visual_projector = nn.Sequential(nn.Linear(visual_feature_dim, d_model), nn.LayerNorm(d_model))
        self.fact_projector = nn.Sequential(nn.Linear(fact_feature_dim, d_model), nn.LayerNorm(d_model))
        self.fusion = nn.ModuleList([FactFusionBlock(d_model, nheads, dropout) for _ in range(fusion_layers)])
        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nheads, num_layers=nlayers,
            dropout=dropout, max_length=max_length, pad_idx=pad_idx,
        )

    def encode_memory(self, visual_inputs: Tensor, fact_features: Tensor,
                      fact_attention_mask: Tensor, include_cls_token: bool = False) -> Tensor:
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
        visual_memory = self.visual_projector(visual_tokens.to(dtype=self.visual_projector[0].weight.dtype))
        facts_memory = self.fact_projector(fact_features.to(dtype=self.fact_projector[0].weight.dtype))
        facts_memory = facts_memory.masked_fill(~valid.unsqueeze(-1), 0)
        padding_mask = ~valid
        memory = visual_memory
        for block in self.fusion:
            memory = block(memory, facts_memory, padding_mask)
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
