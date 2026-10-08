import math
from torch import Tensor, nn

from src.shared.decoder import PositionalEncoding


class TextContextEncoderV6(nn.Module):
    """Contextualize retrieved caption tokens using the decoder embedding."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        max_context_length: int,
        embedding: nn.Embedding,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.pos_encoding = PositionalEncoding(d_model, max_context_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        padding_mask = attention_mask == 0

        hidden = self.embedding(input_ids) * math.sqrt(self.d_model)
        hidden = self.pos_encoding(hidden)
        ctx_memory = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return ctx_memory
