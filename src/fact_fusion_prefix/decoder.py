"""Standard caption decoder with a per-sample bidirectional prefix mask."""

import math

import torch

from src.shared.decoder import TransformerCaptionDecoder


class PrefixCaptionDecoder(TransformerCaptionDecoder):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, max_length, pad_idx):
        super().__init__(vocab_size, d_model, nhead, num_layers, dropout, max_length, pad_idx)
        self.nhead = nhead

    def prefix_mask(self, size, prefix_lengths, device):
        positions = torch.arange(size, device=device)
        queries = positions.view(1, size, 1)
        keys = positions.view(1, 1, size)
        boundary = prefix_lengths.view(-1, 1, 1)
        blocked = torch.where(queries < boundary, keys >= boundary, keys > queries)
        return blocked.repeat_interleave(self.nhead, dim=0)  # [batch * heads, steps, steps]

    def forward(self, input_ids, memory, prefix_lengths, attention_mask=None, memory_key_padding_mask=None):
        if prefix_lengths.shape != (input_ids.size(0),):
            raise ValueError("Each sample requires its own prefix length.")
        if ((prefix_lengths < 1) | (prefix_lengths > input_ids.size(1))).any():
            raise ValueError("Prefix lengths must fit within the decoder input.")
        if input_ids.size(1) > self.pos_encoding.pos_encoding.size(1):
            raise ValueError("Prefix plus caption exceeds positional encoding capacity.")
        hidden = self.embedding(input_ids) * math.sqrt(self.d_model)
        hidden = self.pos_encoding(hidden)
        mask = self.prefix_mask(input_ids.size(1), prefix_lengths, input_ids.device)
        padding_mask = attention_mask == 0 if attention_mask is not None else None
        hidden = self.decoder(
            tgt=hidden, memory=memory, tgt_mask=mask, tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=False,
        )
        return self.out_fc(hidden)
