import torch
from torch import Tensor, nn

from src.v6.decoder import TransformerCaptionDecoderV6
from src.v6.encoder import TextContextEncoderV6


class RAGCaptionerV6(nn.Module):
    """V1-style text RAG with standardized training and current data formats."""

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
        self.shared_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )

        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.context_encoder = TextContextEncoderV6(
            d_model=d_model,
            nhead=nheads,
            dropout=dropout,
            max_context_length=max_context_length,
            embedding=self.shared_embedding,
            num_layers=context_layers,
        )
        self.decoder = TransformerCaptionDecoderV6(
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
        rag_input_ids: Tensor,
        rag_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if not include_cls_token:
            visual_inputs = visual_inputs[:, 1:, :]

        visual_inputs = visual_inputs.to(dtype=self.visual_projector[0].weight.dtype)
        visual_memory = self.visual_projector(visual_inputs)
        context_memory = self.context_encoder(rag_input_ids, rag_attention_mask)
        memory = torch.cat([visual_memory, context_memory], dim=1)

        visual_padding_mask = torch.zeros(
            visual_memory.shape[:2], dtype=torch.bool, device=memory.device
        )
        context_padding_mask = rag_attention_mask == 0
        memory_padding_mask = torch.cat(
            [visual_padding_mask, context_padding_mask], dim=1
        )
        return memory, memory_padding_mask

    def forward(
        self,
        visual_inputs: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        rag_input_ids: Tensor,
        rag_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        memory, memory_padding_mask = self.encode_memory(
            visual_inputs, rag_input_ids, rag_attention_mask,
            include_cls_token=include_cls_token,
        )
        logit = self.decoder(
            input_ids=input_ids[:, :-1],
            memory=memory,
            attention_mask=attention_mask[:, :-1],
            memory_key_padding_mask=memory_padding_mask,
        )
        return logit
