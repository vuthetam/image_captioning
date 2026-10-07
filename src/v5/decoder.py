import math
import torch
from torch import Tensor, nn

from src.shared.decoder import PositionalEncoding


class TransformerCaptionDecoderV5(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_length: int = 40,
        pad_idx: int = 0,
        embedding_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
            
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_bool_mask(self, size: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        input = self.embedding(input_ids) * math.sqrt(self.d_model)
        input = self.pos_encoding(input)

        # Causal Boolean Mask
        tgt_mask = self._generate_square_subsequent_bool_mask(input.size(1), input.device)
        
        tgt_key_padding_mask = None
        if attention_mask is not None:
            # Boolean mask for padding: True = ignore this token
            tgt_key_padding_mask = (attention_mask == 0)

        out = self.decoder(
            tgt=input,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_fc(out)

