"""Keep fusion facts unchanged and select their attribute/triplet indices."""

from pathlib import Path

import torch

from src.shared.facts.dataset import RAGFeatureDataset as FactFeatureDataset
from src.shared.facts.dataset import collate_fact_batch as collate_fusion_batch
from src.shared.vocabulary import Vocabulary


DIRECT_FACT_COLUMNS = ("attribute_texts", "triplet_texts")


class RAGFeatureDataset(FactFeatureDataset):
    def __init__(self, df, features_path: str | Path, rag_contexts_path: str | Path,
                 top_k: int, caption_sort_by: str = "itm"):
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by, return_indices=True)
        self.attr_triplet_fact_index_lookup = {}
        for imgid, img_fact_indices_by_column in self.fact_index_lookup.items():
            # Merge attribute and triplet indices, deduplicate them, then restore fusion order.
            attr_triplet_fact_indices = {
                index for column in DIRECT_FACT_COLUMNS for index in img_fact_indices_by_column[column]
            }
            self.attr_triplet_fact_index_lookup[imgid] = sorted(attr_triplet_fact_indices)

    def __getitem__(self, index):
        visual_inputs, facts, imgid = super().__getitem__(index)
        attr_triplet_fact_indices = self.attr_triplet_fact_index_lookup[imgid]
        return visual_inputs, facts, attr_triplet_fact_indices, imgid


class RAGFeatureCaptionDataset(RAGFeatureDataset):
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, max_length: int, top_k: int,
                 caption_sort_by: str = "itm"):
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by)
        self.vocab = vocab
        self.max_length = max_length

    def __getitem__(self, index):
        visual_inputs, facts, attr_triplet_fact_indices, _ = super().__getitem__(index)
        row = self.dataframe.iloc[index]
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return visual_inputs, input_ids, attention_mask, facts, attr_triplet_fact_indices


def collate_fact_batch(samples):
    """Reuse fusion collation, padding only the image-specific subset indices."""
    is_inference = len(samples[0]) == 4
    if is_inference:
        fusion_samples = [(visual, facts, imgid) for visual, facts, _, imgid in samples]
        visual_inputs, facts, imgids = collate_fusion_batch(fusion_samples)
        attr_triplet_fact_index_groups = [sample[2] for sample in samples]
    else:
        fusion_samples = [sample[:4] for sample in samples]
        visual_inputs, input_ids, attention_mask, facts = collate_fusion_batch(fusion_samples)
        attr_triplet_fact_index_groups = [sample[4] for sample in samples]

    # A masked slot represents an empty subset, never a real additional fact.
    max_facts = max(1, max(map(len, attr_triplet_fact_index_groups)))
    attr_triplet_fact_indices = torch.zeros(len(samples), max_facts, dtype=torch.long)
    attr_triplet_fact_mask = torch.zeros(len(samples), max_facts, dtype=torch.bool)
    for sample_index, sample_fact_indices in enumerate(attr_triplet_fact_index_groups):
        attr_triplet_fact_indices[sample_index, :len(sample_fact_indices)] = torch.tensor(
            sample_fact_indices, dtype=torch.long,
        )
        attr_triplet_fact_mask[sample_index, :len(sample_fact_indices)] = True

    if is_inference:
        return visual_inputs, facts, attr_triplet_fact_indices, attr_triplet_fact_mask, imgids
    return visual_inputs, input_ids, attention_mask, facts, attr_triplet_fact_indices, attr_triplet_fact_mask
