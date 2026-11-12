"""Caption decoder with a gate for each unique fact token in the vocabulary."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.shared.decoder import TransformerCaptionDecoder


class TokenBiasCaptionDecoder(TransformerCaptionDecoder):
    def __init__(
        self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
        dropout: float, max_length: int, pad_idx: int, initial_alpha: float = 0.1,
    ) -> None:
        if not math.isfinite(initial_alpha) or initial_alpha <= 0:
            raise ValueError("initial_alpha must be finite and positive.")
        super().__init__(vocab_size, d_model, nhead, num_layers, dropout, max_length, pad_idx)
        self.token_gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid(),
        )
        raw_alpha = initial_alpha + math.log(-math.expm1(-initial_alpha))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha)

    def compute_bias(
        self, hidden_states: Tensor, candidate_token_ids: Tensor, candidate_token_mask: Tensor,
    ) -> Tensor:
        """Score deduplicated candidates using the decoder's own token embeddings."""
        if candidate_token_ids.ndim != 2 or candidate_token_ids.shape != candidate_token_mask.shape:
            raise ValueError("Candidate IDs and mask must have shape [batch, candidates].")
        if hidden_states.size(0) != candidate_token_ids.size(0):
            raise ValueError("Decoder states and candidates must have the same batch size.")

        batch_size, num_steps, _ = hidden_states.shape
        # Raw decoder embeddings: [batch, candidates, d_model]. No positions/scaling.
        candidate_embeddings = self.embedding(candidate_token_ids)

        # Equivalent to Linear([h; E(w)]), using the same first Linear's weights.
        first_layer = self.token_gate_mlp[0]
        hidden_weights = first_layer.weight[:, :self.d_model]
        token_weights = first_layer.weight[:, self.d_model:]
        hidden_term = F.linear(hidden_states, hidden_weights, first_layer.bias)
        token_term = F.linear(candidate_embeddings, token_weights)
        combined_hidden = hidden_term.unsqueeze(2) + token_term.unsqueeze(1)  # [batch, steps, candidates, d_model]

        gate_hidden = self.token_gate_mlp[1](combined_hidden)
        gate_logits = self.token_gate_mlp[2](gate_hidden)
        gates = self.token_gate_mlp[3](gate_logits)
        gates = gates.squeeze(-1)  # [batch, steps, candidates]

        valid_tokens = candidate_token_mask.bool()
        token_gate_mask = valid_tokens.unsqueeze(1).to(dtype=gates.dtype)
        token_gates = gates * token_gate_mask

        token_indices = candidate_token_ids.masked_fill(~valid_tokens, self.pad_idx)
        token_indices = token_indices.unsqueeze(1).expand(-1, num_steps, -1)

        # Valid IDs are unique per image. Repeated padding IDs only write zeros.
        bias = gates.new_zeros(batch_size, num_steps, self.vocab_size)
        bias = bias.scatter(dim=2, index=token_indices, src=token_gates)
        return bias

    def forward(
        self, input_ids: Tensor, memory: Tensor,
        candidate_token_ids: Tensor, candidate_token_mask: Tensor,
        attention_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_last_step_only: bool = False,
    ) -> Tensor:
        tgt = self.embedding(input_ids) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask == 0

        hidden_states = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        if return_last_step_only:
            hidden_states = hidden_states[:, -1:, :]

        logits = self.out_fc(hidden_states)
        bias = self.compute_bias(hidden_states, candidate_token_ids, candidate_token_mask)
        alpha = self.alpha.to(dtype=logits.dtype)
        bias = bias.to(dtype=logits.dtype)
        adjusted_logits = logits + alpha * bias
        return adjusted_logits
