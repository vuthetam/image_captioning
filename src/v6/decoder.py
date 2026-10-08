import math

import torch
from torch import Tensor, nn

from src.shared.decoder import PositionalEncoding


class TransformerCaptionDecoderV6(nn.Module):
    """Transformer decoder with an injectable token embedding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_length: int,
        pad_idx: int,
        embedding: nn.Embedding,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = embedding
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.embedding(input_ids) * math.sqrt(self.d_model)
        hidden = self.pos_encoding(hidden)

        target_padding_mask = None
        if attention_mask is not None:
            target_padding_mask = attention_mask == 0

        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=self._causal_mask(hidden.size(1), hidden.device),
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output(decoded)
