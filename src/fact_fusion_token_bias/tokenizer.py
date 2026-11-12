"""Collect unique decoder vocabulary candidates from the selected fusion facts."""

from typing import Sequence

import torch
from torch import Tensor

from src.shared.facts.tokenizer import tokenize_fact
from src.shared.vocabulary import Vocabulary


def encode_candidate_token_ids(
    facts_per_sample: Sequence[Sequence[str]], vocab: Vocabulary, device,
) -> tuple[Tensor, Tensor]:
    """Deduplicate token IDs across all facts of each image, in encounter order."""
    if not facts_per_sample or any(not facts for facts in facts_per_sample):
        raise ValueError("Each sample must contain at least one fact.")

    special_ids = {vocab.stoi[token] for token in vocab.special_tokens}
    batch_token_ids = []
    for facts in facts_per_sample:
        token_ids = []
        seen = set()
        for fact in facts:
            for word in tokenize_fact(fact):
                token_id = vocab.stoi.get(word)
                if token_id is None or token_id in special_ids or token_id in seen:
                    continue
                seen.add(token_id)
                token_ids.append(token_id)
        batch_token_ids.append(token_ids)

    # An all-OOV sample keeps masked padding; fusion still uses its original facts.
    max_tokens = max(1, max(len(token_ids) for token_ids in batch_token_ids))
    shape = (len(facts_per_sample), max_tokens)
    candidate_token_ids = torch.full(shape, vocab.pad_idx(), dtype=torch.long)
    candidate_token_mask = torch.zeros(shape, dtype=torch.bool)
    for sample_index, token_ids in enumerate(batch_token_ids):
        num_tokens = len(token_ids)
        candidate_token_ids[sample_index, :num_tokens] = torch.tensor(token_ids, dtype=torch.long)
        candidate_token_mask[sample_index, :num_tokens] = True

    candidate_token_ids = candidate_token_ids.to(device)
    candidate_token_mask = candidate_token_mask.to(device)
    return candidate_token_ids, candidate_token_mask
