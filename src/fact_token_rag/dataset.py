from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq
import torch
from torch.utils.data import default_collate

from src.shared.dataset import rag_caption_order
from src.shared.facts.dataset import RAGFeatureDataset as FactFeatureDataset, normalize_fact
from src.shared.facts.dataset import collate_fact_batch as collate_fusion_batch
from src.shared.vocabulary import Vocabulary


TOKEN_FACT_COLUMNS = ("attribute_texts", "triplet_texts")


def tokenize_fact(text: str) -> list[str]:
    """Use decoder vocabulary casing; split hyphens and remove apostrophes."""
    return text.lower().replace("-", " ").replace("'", "").split()


def load_fact_token_context_lookup(context_path: str | Path, top_k: int,
                                   caption_sort_by: str = "itm"):
    """Select attributes/triplets per caption using the fusion caption order."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if caption_sort_by not in {"clip", "itm"}:
        raise ValueError("caption_sort_by must be 'clip' or 'itm'.")
    columns = ["imgid", "captions", *TOKEN_FACT_COLUMNS]
    if caption_sort_by == "itm":
        columns.append("itm_scores")
    missing = set(columns) - set(pq.read_schema(context_path).names)
    if missing:
        raise ValueError(f"Missing fact token context columns {sorted(missing)}: {context_path}")

    context_lookup = {}
    for row in pq.read_table(context_path, columns=columns).to_pylist():
        imgid = int(row["imgid"])
        if imgid in context_lookup:
            raise ValueError(f"Duplicate fact token context imgid={imgid}: {context_path}")
        captions = row["captions"]
        if not isinstance(captions, list):
            raise ValueError(f"Invalid captions for imgid={imgid}: {context_path}")
        for column in TOKEN_FACT_COLUMNS:
            groups = row[column]
            if not isinstance(groups, list) or len(groups) != len(captions):
                raise ValueError(f"{column} must align with captions for imgid={imgid}: {context_path}")
            if any(not isinstance(group, list) for group in groups):
                raise ValueError(f"{column} must contain fact lists for imgid={imgid}: {context_path}")
        order = rag_caption_order(
            len(captions), caption_sort_by, row.get("itm_scores"), f"for imgid={imgid}: {context_path}",
        )
        caption_fact_groups = []
        for caption_index in order[:top_k]:
            facts = []
            for column in TOKEN_FACT_COLUMNS:
                for text in row[column][caption_index]:
                    fact = normalize_fact(text)
                    if fact:
                        facts.append(fact)
            caption_fact_groups.append(facts)
        context_lookup[imgid] = caption_fact_groups
    return context_lookup


def encode_fact_context(
    caption_fact_groups: Sequence[Sequence[str]],
    vocab: Vocabulary,
    tokens_per_caption: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep whole facts within each caption budget, counting each fact's EOS.

    Deduplication uses whitespace-normalized original strings. Only facts actually
    retained in memory are marked seen, so an omitted fact can fit a later group.
    """
    if tokens_per_caption < 1 or top_k < 1:
        raise ValueError("tokens_per_caption and top_k must be positive.")
    token_ids = []
    seen = set()
    for facts in caption_fact_groups[:top_k]:
        caption_token_ids = []
        for text in facts:
            fact = normalize_fact(text)
            if not fact or fact in seen:
                continue
            words = tokenize_fact(fact)
            if not words:
                continue
            fact_token_ids = [vocab.token_to_idx(word) for word in words]
            fact_token_ids.append(vocab.eos_idx())
            if len(caption_token_ids) + len(fact_token_ids) > tokens_per_caption:
                break
            caption_token_ids.extend(fact_token_ids)
            seen.add(fact)
        token_ids.extend(caption_token_ids)

    attention_mask = [1] * len(token_ids)
    max_context_length = top_k * tokens_per_caption
    padding_length = max_context_length - len(token_ids)
    token_ids.extend([vocab.pad_idx()] * padding_length)
    attention_mask.extend([0] * padding_length)
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    return token_ids, attention_mask


class RAGFeatureDataset(FactFeatureDataset):
    """Return patches, unchanged CLIP facts and supplementary fact token IDs."""

    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, tokens_per_caption: int, top_k: int,
                 caption_sort_by: str = "itm"):
        if tokens_per_caption < 1:
            raise ValueError("tokens_per_caption must be positive.")
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by)
        self.vocab = vocab
        self.tokens_per_caption = tokens_per_caption
        self.top_k = top_k
        self.max_context_length = top_k * tokens_per_caption
        self.context_lookup = load_fact_token_context_lookup(rag_contexts_path, top_k, caption_sort_by)

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


def collate_fact_batch(samples):
    """Use Fact RAG collation for fusion inputs and stack supplementary IDs."""
    if len(samples[0]) == 5:
        fusion_samples = [(visual, facts, imgid) for visual, facts, _, _, imgid in samples]
        visual_inputs, facts, imgids = collate_fusion_batch(fusion_samples)
        rag_input_ids = default_collate([sample[2] for sample in samples])
        rag_attention_mask = default_collate([sample[3] for sample in samples])
        return visual_inputs, facts, rag_input_ids, rag_attention_mask, imgids

    fusion_samples = [sample[:4] for sample in samples]
    visual_inputs, input_ids, attention_mask, facts = collate_fusion_batch(fusion_samples)
    rag_input_ids = default_collate([sample[4] for sample in samples])
    rag_attention_mask = default_collate([sample[5] for sample in samples])
    return visual_inputs, input_ids, attention_mask, facts, rag_input_ids, rag_attention_mask


