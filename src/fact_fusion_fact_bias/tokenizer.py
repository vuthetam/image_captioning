"""Map fusion facts to decoder vocabulary IDs without changing CLIP inputs."""

import re
from typing import Sequence

import torch
from torch import Tensor

from src.shared.vocabulary import Vocabulary


def tokenize_fact(text: str) -> list[str]:
    """Split hyphens/punctuation and remove straight or curly apostrophes."""
    normalized = text.lower().replace("-", " ").replace("'", "").replace("’", "")
    # Preserve special-token strings so the vocabulary filter can reject them.
    return re.findall(r"<[^>\s]+>|\w+", normalized)


def encode_fact_token_ids(
    facts_per_sample: Sequence[Sequence[str]], vocab: Vocabulary, device,
) -> tuple[Tensor, Tensor]:
    """Keep fact order and deduplicate vocabulary IDs within each fact."""
    if not facts_per_sample or any(not facts for facts in facts_per_sample):
        raise ValueError("Each sample must contain at least one fact.")
    special_ids = {vocab.stoi[token] for token in vocab.special_tokens}
    batch_token_ids = []
    max_facts = max(len(facts) for facts in facts_per_sample)
    max_tokens = 1  # Keep a masked slot even when every word is out of vocabulary.
    for facts in facts_per_sample:
        sample_token_ids = []
        for fact in facts:
            token_ids = []
            seen = set()
            for word in tokenize_fact(fact):
                token_id = vocab.stoi.get(word)
                if token_id is None or token_id in special_ids or token_id in seen:
                    continue
                seen.add(token_id)
                token_ids.append(token_id)
            max_tokens = max(max_tokens, len(token_ids))
            sample_token_ids.append(token_ids)
        batch_token_ids.append(sample_token_ids)

    shape = (len(facts_per_sample), max_facts, max_tokens)
    fact_token_ids = torch.full(shape, vocab.pad_idx(), dtype=torch.long)
    fact_token_mask = torch.zeros(shape, dtype=torch.bool)
    for sample_index, sample_token_ids in enumerate(batch_token_ids):
        for fact_index, token_ids in enumerate(sample_token_ids):
            num_tokens = len(token_ids)
            fact_token_ids[sample_index, fact_index, :num_tokens] = torch.tensor(token_ids, dtype=torch.long)
            fact_token_mask[sample_index, fact_index, :num_tokens] = True

    fact_token_ids = fact_token_ids.to(device)
    fact_token_mask = fact_token_mask.to(device)
    return fact_token_ids, fact_token_mask
