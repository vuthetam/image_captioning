from pathlib import Path
from typing import Sequence

import torch
import pyarrow.parquet as pq
from torch.utils.data import default_collate

from src.shared.dataset import rag_caption_order
from src.shared.facts.dataset import RAGFeatureDataset as FactFeatureDataset
from src.shared.vocabulary import Vocabulary


def load_caption_context_lookup(context_path: str | Path, top_k: int,
                                caption_sort_by: str = "itm"):
    """Select original caption tokens using the same ordering as fact retrieval."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if caption_sort_by not in {"clip", "itm"}:
        raise ValueError("caption_sort_by must be 'clip' or 'itm'.")
    columns = ["imgid", "captions", "tokens"]
    if caption_sort_by == "itm":
        columns.append("itm_scores")
    missing = set(columns) - set(pq.read_schema(context_path).names)
    if missing:
        raise ValueError(f"Missing caption context columns {sorted(missing)}: {context_path}")

    caption_lookup = {}
    for row in pq.read_table(context_path, columns=columns).to_pylist():
        imgid = int(row["imgid"])
        if imgid in caption_lookup:
            raise ValueError(f"Duplicate caption context imgid={imgid}: {context_path}")
        captions = row["captions"]
        tokens = row["tokens"]
        if not isinstance(captions, list) or not isinstance(tokens, list) or len(tokens) != len(captions):
            raise ValueError(f"Caption tokens must align with captions for imgid={imgid}: {context_path}")
        if any(not isinstance(words, list) or any(not isinstance(word, str) for word in words)
               for words in tokens):
            raise ValueError(f"Invalid caption token lists for imgid={imgid}: {context_path}")
        order = rag_caption_order(
            len(captions), caption_sort_by, row.get("itm_scores"), f"for imgid={imgid}: {context_path}",
        )
        caption_lookup[imgid] = [tokens[index] for index in order[:top_k]]
    return caption_lookup


def encode_rag_context(
    caption_tokens: Sequence[Sequence[str]],
    vocab: Vocabulary,
    max_context_length: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode pre-tokenized retrieved captions separated by ``<eos>``."""
    tokens: list[str] = []
    for words in caption_tokens[:top_k]:
        if tokens:
            tokens.append(vocab.eos_token)
        tokens.extend(words)

    tokens = tokens[:max_context_length]
    token_ids = [vocab.token_to_idx(token) for token in tokens]
    attention_mask = [1] * len(token_ids)

    padding_length = max_context_length - len(token_ids)
    token_ids.extend([vocab.pad_idx()] * padding_length)
    attention_mask.extend([0] * padding_length)
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    return token_ids, attention_mask


class RAGFeatureDataset(FactFeatureDataset):
    """Return patches, facts and caption tokens selected from the same top K."""

    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, max_context_length: int, top_k: int,
                 caption_sort_by: str = "itm"):
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by)
        if max_context_length < 1:
            raise ValueError("max_context_length must be positive.")
        self.vocab = vocab
        self.max_context_length = max_context_length
        self.top_k = top_k
        self.caption_lookup = load_caption_context_lookup(rag_contexts_path, top_k, caption_sort_by)
        for imgid in set(map(int, self.dataframe["imgid"])):
            if not any(self.caption_lookup[imgid]):
                raise ValueError(f"No caption context tokens selected for imgid={imgid}.")

    def __getitem__(self, index):
        visual_inputs, facts, imgid = super().__getitem__(index)
        rag_input_ids, rag_attention_mask = encode_rag_context(
            self.caption_lookup[imgid], self.vocab, self.max_context_length, self.top_k,
        )
        return visual_inputs, facts, rag_input_ids, rag_attention_mask, imgid


class RAGFeatureCaptionDataset(RAGFeatureDataset):
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, max_length: int, max_context_length: int,
                 top_k: int, caption_sort_by: str = "itm"):
        super().__init__(
            df, vocab, features_path, rag_contexts_path, max_context_length, top_k,
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
    """Stack tensors while preserving each sample's variable-length fact list."""
    if len(samples[0]) == 5:
        visual_inputs, facts, rag_input_ids, rag_attention_mask, imgids = zip(*samples)
        visual_inputs = default_collate(visual_inputs)
        facts = list(facts)
        rag_input_ids = default_collate(rag_input_ids)
        rag_attention_mask = default_collate(rag_attention_mask)
        imgids = default_collate(imgids)
        return visual_inputs, facts, rag_input_ids, rag_attention_mask, imgids

    visual_inputs, input_ids, attention_mask, facts, rag_input_ids, rag_attention_mask = zip(*samples)
    visual_inputs = default_collate(visual_inputs)
    input_ids = default_collate(input_ids)
    attention_mask = default_collate(attention_mask)
    facts = list(facts)
    rag_input_ids = default_collate(rag_input_ids)
    rag_attention_mask = default_collate(rag_attention_mask)
    return visual_inputs, input_ids, attention_mask, facts, rag_input_ids, rag_attention_mask
