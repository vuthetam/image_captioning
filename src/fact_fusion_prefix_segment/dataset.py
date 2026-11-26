"""Compact fact-prefix/caption sequences, padded only at the end of a batch."""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from src.shared.facts.dataset import RAGFeatureDataset as FactFeatureDataset
from src.shared.facts.tokenizer import tokenize_fact


def encode_prefix(facts, vocab, max_prefix_length):
    if max_prefix_length < 1:
        raise ValueError("Maximum prefix length must be positive.")
    token_ids = []
    special_tokens = set(vocab.special_tokens)
    for fact in facts:
        words = [word for word in tokenize_fact(fact) if word not in special_tokens]
        if not words:
            continue
        fact_ids = [vocab.token_to_idx(word) for word in words]
        separator = [vocab.sep_idx()] if token_ids else []
        if len(token_ids) + len(separator) + len(fact_ids) + 1 > max_prefix_length:
            break
        token_ids.extend(separator)
        token_ids.extend(fact_ids)
    return token_ids + [vocab.sos_idx()]


class RAGFeatureDataset(FactFeatureDataset):
    def __init__(self, df, vocab, features_path, rag_contexts_path, top_k, max_prefix_length, caption_sort_by="itm"):
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by, return_indices=True)
        self.vocab = vocab
        self.prefix_lookup = {}
        for imgid, indices_by_column in self.fact_index_lookup.items():
            indices = sorted(set(indices_by_column["attribute_texts"]) | set(indices_by_column["triplet_texts"]))
            facts = [self.fact_lookup[imgid][index] for index in indices]
            self.prefix_lookup[imgid] = encode_prefix(facts, vocab, max_prefix_length)

    def __getitem__(self, index):
        visual_inputs, facts, imgid = super().__getitem__(index)
        prefix_ids = torch.tensor(self.prefix_lookup[imgid], dtype=torch.long)
        return visual_inputs, prefix_ids, facts, imgid


class RAGFeatureCaptionDataset(RAGFeatureDataset):
    def __init__(self, df, vocab, features_path, rag_contexts_path, max_length, top_k,
                 max_prefix_length, caption_sort_by="itm"):
        if max_length < 2:
            raise ValueError("Caption max_length must include SOS and EOS.")
        super().__init__(df, vocab, features_path, rag_contexts_path, top_k, max_prefix_length, caption_sort_by)
        self.max_length = max_length

    def __getitem__(self, index):
        visual_inputs, prefix_ids, facts, _ = super().__getitem__(index)
        caption_ids, caption_mask = self.vocab.encode_from_tokens(self.dataframe.iloc[index]["tokens"], self.max_length)
        caption_ids = torch.tensor(caption_ids[1:sum(caption_mask)], dtype=torch.long)
        input_ids = torch.cat([prefix_ids, caption_ids])
        prefix_length = prefix_ids.numel()

        return visual_inputs, input_ids, prefix_length, facts


def collate_fact_batch(samples, pad_idx=0):
    visual_inputs, sequences, third, fourth = zip(*samples)
    visual_inputs = default_collate(visual_inputs)
    lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
    input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    attention_mask = torch.arange(input_ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    if isinstance(third[0], int):
        prefix_lengths = torch.tensor(third, dtype=torch.long)
        facts = list(fourth)
        return visual_inputs, input_ids, attention_mask, prefix_lengths, facts
    facts = list(third)
    image_ids = default_collate(fourth)
    return visual_inputs, input_ids, lengths, facts, image_ids
