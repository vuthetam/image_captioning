import torch
from torch import Tensor, nn

from src.fact_text_rag.decoder import FactTextRAGDecoder
from src.fact_text_rag.encoder import TextContextEncoder
from src.shared.facts.fusion import FactFusionEncoder


class FactTextRAGCaptioner(FactFusionEncoder):
    """Append contextualized retrieved captions to image-fact fusion memory."""

    def __init__(self, vocab_size: int, d_model: int, nheads: int, nlayers: int,
                 dropout: float, max_length: int, pad_idx: int,
                 visual_feature_dim: int, fact_feature_dim: int, context_layers: int,
                 max_context_length: int, fusion_layers: int = 1):
        super().__init__(
            visual_feature_dim, fact_feature_dim, d_model, nheads, dropout, fusion_layers,
        )
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.context_encoder = TextContextEncoder(
            d_model=d_model, nhead=nheads, dropout=dropout,
            max_context_length=max_context_length, embedding=self.shared_embedding,
            num_layers=context_layers,
        )
        self.decoder = FactTextRAGDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nheads, num_layers=nlayers,
            dropout=dropout, max_length=max_length, pad_idx=pad_idx,
            embedding=self.shared_embedding,
        )

    def encode_memory(self, visual_inputs: Tensor, fact_features: Tensor,
                      fact_attention_mask: Tensor, rag_input_ids: Tensor,
                      rag_attention_mask: Tensor, include_cls_token: bool = False
                      ) -> tuple[Tensor, Tensor]:
        if rag_input_ids.ndim != 2 or rag_input_ids.shape != rag_attention_mask.shape:
            raise ValueError("Caption context IDs and attention mask must align.")
        if rag_input_ids.size(0) != visual_inputs.size(0):
            raise ValueError("Visual and caption context batch sizes must match.")
        if not rag_attention_mask.ne(0).any(dim=1).all():
            raise ValueError("Each sample must contain at least one caption context token.")
        fused_memory, _, _ = self.encode_fusion(
            visual_inputs, fact_features, fact_attention_mask,
            include_cls_token=include_cls_token,
        )
        context_memory = self.context_encoder(rag_input_ids, rag_attention_mask)
        context_padding_mask = rag_attention_mask == 0
        memory = torch.cat([fused_memory, context_memory], dim=1)
        visual_padding_mask = torch.zeros(
            fused_memory.shape[:2], dtype=torch.bool, device=fused_memory.device,
        )
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
        logits = self.decoder(
            input_ids=input_ids[:, :-1], memory=memory,
            attention_mask=attention_mask[:, :-1],
            memory_key_padding_mask=memory_padding_mask,
        )
        return logits
