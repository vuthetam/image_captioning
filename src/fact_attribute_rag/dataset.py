from pathlib import Path

import pyarrow.parquet as pq
import torch

from src.fact_token_rag.dataset import encode_fact_context
from src.shared.dataset import rag_caption_order
from src.shared.facts.dataset import RAGFeatureDataset as FactFeatureDataset, normalize_fact
from src.shared.vocabulary import Vocabulary


def load_attribute_context_lookup(context_path: str | Path, top_k: int,
                                  caption_sort_by: str = "itm"):
    """Select attribute facts per caption using the same ranking as fusion."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if caption_sort_by not in {"clip", "itm"}:
        raise ValueError("caption_sort_by must be 'clip' or 'itm'.")
    columns = ["imgid", "captions", "attribute_texts"]
    if caption_sort_by == "itm":
        columns.append("itm_scores")
    missing = set(columns) - set(pq.read_schema(context_path).names)
    if missing:
        raise ValueError(f"Missing attribute context columns {sorted(missing)}: {context_path}")

    context_lookup = {}
    for row in pq.read_table(context_path, columns=columns).to_pylist():
        imgid = int(row["imgid"])
        if imgid in context_lookup:
            raise ValueError(f"Duplicate attribute context imgid={imgid}: {context_path}")
        captions = row["captions"]
        if not isinstance(captions, list):
            raise ValueError(f"Invalid captions for imgid={imgid}: {context_path}")
        attributes = row["attribute_texts"]
        if not isinstance(attributes, list) or len(attributes) != len(captions):
            raise ValueError(f"attribute_texts must align with captions for imgid={imgid}: {context_path}")
        if any(not isinstance(group, list) for group in attributes):
            raise ValueError(f"attribute_texts must contain fact lists for imgid={imgid}: {context_path}")
        order = rag_caption_order(
            len(captions), caption_sort_by, row.get("itm_scores"),
            f"for imgid={imgid}: {context_path}",
        )
        caption_fact_groups = []
        for caption_index in order[:top_k]:
            facts = []
            for text in attributes[caption_index]:
                fact = normalize_fact(text)
                if fact:
                    facts.append(fact)
            caption_fact_groups.append(facts)
        context_lookup[imgid] = caption_fact_groups
    return context_lookup


class RAGFeatureDataset(FactFeatureDataset):
    """Return all fusion facts and supplementary attribute token IDs."""

    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, tokens_per_caption: int, top_k: int,
                 caption_sort_by: str = "itm"):
        if tokens_per_caption < 1:
            raise ValueError("tokens_per_caption must be positive.")
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by)
        self.vocab = vocab
        self.tokens_per_caption = tokens_per_caption
        self.top_k = top_k
        self.context_lookup = load_attribute_context_lookup(rag_contexts_path, top_k, caption_sort_by)

    def __getitem__(self, index):
        visual_inputs, facts, imgid = super().__getitem__(index)
        rag_input_ids, rag_attention_mask = encode_fact_context(
            self.context_lookup[imgid], self.vocab, self.tokens_per_caption, self.top_k,
        )
        return visual_inputs, facts, rag_input_ids, rag_attention_mask, imgid


class RAGFeatureCaptionDataset(RAGFeatureDataset):
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, max_length: int, tokens_per_caption: int,
                 top_k: int, caption_sort_by: str = "itm"):
        super().__init__(
            df, vocab, features_path, rag_contexts_path, tokens_per_caption, top_k,
            caption_sort_by,
        )
        self.max_length = max_length

    def __getitem__(self, index):
        visual_inputs, facts, rag_input_ids, rag_attention_mask, _ = super().__getitem__(index)
        row = self.dataframe.iloc[index]
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return visual_inputs, input_ids, attention_mask, facts, rag_input_ids, rag_attention_mask
