"""Caption decoder with a fact gate added to vocabulary logits."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.shared.decoder import TransformerCaptionDecoder


class FactBiasCaptionDecoder(TransformerCaptionDecoder):
    def __init__(
        self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
        dropout: float, max_length: int, pad_idx: int, initial_alpha: float = 0.1,
    ) -> None:
        if not math.isfinite(initial_alpha) or initial_alpha <= 0:
            raise ValueError("initial_alpha must be finite and positive.")
        super().__init__(vocab_size, d_model, nhead, num_layers, dropout, max_length, pad_idx)
        self.fact_gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid(),
        )
        raw_alpha = initial_alpha + math.log(-math.expm1(-initial_alpha))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha)

    def compute_bias(
        self, hidden_states: Tensor, fact_memory: Tensor, fact_padding_mask: Tensor,
        fact_token_ids: Tensor, fact_token_mask: Tensor,
    ) -> Tensor:
        if fact_memory.shape[:2] != fact_padding_mask.shape:
            raise ValueError("Fact memory and padding mask must align.")
        if fact_token_ids.ndim != 3 or fact_token_ids.shape != fact_token_mask.shape:
            raise ValueError("Fact token IDs and mask must have shape [batch, facts, tokens].")
        if fact_token_ids.shape[:2] != fact_memory.shape[:2]:
            raise ValueError("Fact token IDs must preserve fusion fact order and padding.")
        if hidden_states.size(0) != fact_memory.size(0):
            raise ValueError("Decoder states and facts must have the same batch size.")

        # Split the first Linear's weights; equivalent to Linear([h; F]).
        batch_size, num_steps, _ = hidden_states.shape
        first_layer = self.fact_gate_mlp[0]
        hidden_weights = first_layer.weight[:, :self.d_model]
        fact_weights = first_layer.weight[:, self.d_model:]

        # hidden_states: [batch, steps, d_model]
        # Add the first Linear's bias once, in the hidden-state branch.
        hidden_term = F.linear(hidden_states, hidden_weights, first_layer.bias)
        # fact_memory: [batch, facts, d_model]
        fact_term = F.linear(fact_memory, fact_weights)

        combined_hidden = hidden_term.unsqueeze(2) + fact_term.unsqueeze(1)  # [batch, steps, facts, d_model]

        # One sigmoid gate per timestep/fact pair: [batch, steps, facts].
        gate_hidden = self.fact_gate_mlp[1](combined_hidden)
        gate_logits = self.fact_gate_mlp[2](gate_hidden)
        gates = self.fact_gate_mlp[3](gate_logits)
        gates = gates.squeeze(-1)

        # Only real tokens of real facts receive their fact's gate.
        valid_facts = ~fact_padding_mask.bool()
        valid_tokens = fact_token_mask.bool() & valid_facts.unsqueeze(-1)
        token_gate_mask = valid_tokens.unsqueeze(1).to(dtype=gates.dtype)
        token_gates = gates.unsqueeze(-1) * token_gate_mask

        # Flatten fact/token slots for mapping into the vocabulary.
        token_gates = token_gates.flatten(2)

        token_indices = fact_token_ids.flatten(1)
        token_indices = token_indices.unsqueeze(1).expand(-1, num_steps, -1)

        # Ignore the stored ID of every masked slot, including fact padding.
        invalid_tokens = ~valid_tokens.flatten(1)
        invalid_tokens = invalid_tokens.unsqueeze(1)
        token_indices = token_indices.masked_fill(invalid_tokens, self.pad_idx)

        # A token shared by several facts receives the maximum gate.
        bias = gates.new_zeros(batch_size, num_steps, self.vocab_size)
        bias = bias.scatter_reduce(
            dim=2, index=token_indices, src=token_gates,
            reduce="amax", include_self=True,
        )

        return bias

    def forward(
        self, input_ids: Tensor, memory: Tensor, fact_memory: Tensor,
        fact_padding_mask: Tensor, fact_token_ids: Tensor, fact_token_mask: Tensor,
        attention_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        return_last_step_only: bool = False,
    ) -> Tensor:
        # Token embeddings and positional encoding.
        tgt = self.embedding(input_ids) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)

        # Causal mask and caption padding mask.
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)

        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask == 0

        # Decoder hidden states before vocabulary projection.
        hidden_states = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        if return_last_step_only:
            hidden_states = hidden_states[:, -1:, :]

        # Vocabulary logits and fact-level bias use the same hidden states.
        logits = self.out_fc(hidden_states)
        bias = self.compute_bias(
            hidden_states=hidden_states,
            fact_memory=fact_memory,
            fact_padding_mask=fact_padding_mask,
            fact_token_ids=fact_token_ids,
            fact_token_mask=fact_token_mask,
        )

        alpha = self.alpha.to(dtype=logits.dtype)
        bias = bias.to(dtype=logits.dtype)
        adjusted_logits = logits + alpha * bias

        return adjusted_logits
