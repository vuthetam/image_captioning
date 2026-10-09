import torch
from torch import Tensor, nn

from src.v7.decoder import TransformerCaptionDecoderV7
from src.v7.encoder import TextContextEncoderV7


class RAGCaptionerV7(nn.Module):
    """Decode from original patches, related-image CLS tokens and text contexts."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nheads: int,
        nlayers: int,
        dropout: float,
        max_length: int,
        pad_idx: int,
        visual_feature_dim: int,
        context_layers: int,
        max_context_length: int,
    ) -> None:
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Both visual inputs come from the same frozen visual encoder.
        self.visual_projector = nn.Linear(visual_feature_dim, d_model)
        self.context_encoder = TextContextEncoderV7(
            d_model=d_model,
            nhead=nheads,
            dropout=dropout,
            max_context_length=max_context_length,
            embedding=self.shared_embedding,
            num_layers=context_layers,
        )
        self.decoder = TransformerCaptionDecoderV7(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nheads,
            num_layers=nlayers,
            dropout=dropout,
            max_length=max_length,
            pad_idx=pad_idx,
            embedding=self.shared_embedding,
        )

    def encode_memory(
        self,
        visual_inputs: Tensor,
        related_cls_tokens: Tensor,
        rag_input_ids: Tensor,
        rag_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if not include_cls_token:
            visual_inputs = visual_inputs[:, 1:, :]

        target_dtype = self.visual_projector.weight.dtype
        visual_memory = self.visual_projector(visual_inputs.to(dtype=target_dtype))
        related_memory = self.visual_projector(related_cls_tokens.to(dtype=target_dtype))
        context_memory = self.context_encoder(rag_input_ids, rag_attention_mask)
        memory = torch.cat([visual_memory, related_memory, context_memory], dim=1)

        visual_padding_mask = torch.zeros(visual_memory.shape[:2], dtype=torch.bool, device=memory.device)
        related_padding_mask = torch.zeros(related_memory.shape[:2], dtype=torch.bool, device=memory.device)
        context_padding_mask = rag_attention_mask == 0
        memory_padding_mask = torch.cat([visual_padding_mask, related_padding_mask, context_padding_mask], dim=1)
        return memory, memory_padding_mask

    def forward(
        self,
        visual_inputs: Tensor,
        related_cls_tokens: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        rag_input_ids: Tensor,
        rag_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        memory, memory_padding_mask = self.encode_memory(
            visual_inputs, related_cls_tokens, rag_input_ids, rag_attention_mask,
            include_cls_token=include_cls_token,
        )
        logits = self.decoder(
            input_ids=input_ids[:, :-1],
            memory=memory,
            attention_mask=attention_mask[:, :-1],
            memory_key_padding_mask=memory_padding_mask,
        )
        return logits
