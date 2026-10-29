from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, default_collate

from src.shared.dataset import H5FeatureStore, rag_caption_order
from src.shared.vocabulary import Vocabulary


FACT_COLUMNS = ("object_texts", "attribute_texts", "triplet_texts")


def normalize_fact(text: str) -> str:
    """Normalize whitespace without changing case or the fact's wording."""
    if not isinstance(text, str):
        raise ValueError("Facts must be strings.")
    return " ".join(text.split())


def load_rag_fact_lookup(context_path: str | Path, top_k: int, caption_sort_by: str = "itm"):
    """Sort captions, select top K, then gather and deduplicate their facts."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if caption_sort_by not in {"clip", "itm"}:
        raise ValueError("caption_sort_by must be 'clip' or 'itm'.")
    columns = ["imgid", "captions", *FACT_COLUMNS]
    if caption_sort_by == "itm":
        columns.append("itm_scores")
    missing = set(columns) - set(pq.read_schema(context_path).names)
    if missing:
        raise ValueError(f"Missing fact context columns {sorted(missing)}: {context_path}")

    lookup = {}
    for row in pq.read_table(context_path, columns=columns).to_pylist():
        imgid = int(row["imgid"])
        if imgid in lookup:
            raise ValueError(f"Duplicate fact context imgid={imgid}: {context_path}")
        captions = row["captions"]
        if not isinstance(captions, list):
            raise ValueError(f"Invalid captions for imgid={imgid}: {context_path}")
        for column in FACT_COLUMNS:
            groups = row[column]
            if not isinstance(groups, list) or len(groups) != len(captions):
                raise ValueError(f"{column} must align with captions for imgid={imgid}: {context_path}")
            if any(not isinstance(group, list) for group in groups):
                raise ValueError(f"{column} must contain fact lists for imgid={imgid}: {context_path}")
        order = rag_caption_order(
            len(captions), caption_sort_by, row.get("itm_scores"), f"for imgid={imgid}: {context_path}"
        )
        facts, seen = [], set()
        for caption_index in order[:top_k]:
            for column in FACT_COLUMNS:
                for text in row[column][caption_index]:
                    fact = normalize_fact(text)
                    if fact and fact not in seen:
                        seen.add(fact)
                        facts.append(fact)
        lookup[imgid] = facts
    return lookup


class RAGFeatureDataset(Dataset):
    """Return (visual_inputs, facts, imgid); CLIP runs in the main process."""

    def __init__(self, df, features_path: str | Path, rag_contexts_path: str | Path,
                 top_k: int, caption_sort_by: str = "itm"):
        self.dataframe = df.reset_index(drop=True)
        self.feature_store = H5FeatureStore(features_path)
        self.fact_lookup = load_rag_fact_lookup(rag_contexts_path, top_k, caption_sort_by)
        imgids = set(map(int, self.dataframe["imgid"]))
        missing = imgids - self.fact_lookup.keys()
        if missing:
            raise ValueError(f"Missing fact contexts for {len(missing)} images, e.g. imgid={min(missing)}")
        for imgid in imgids:
            if not self.fact_lookup[imgid]:
                raise ValueError(f"No facts selected for imgid={imgid}; each image must have at least one fact.")

    @property
    def feature_shape(self):
        return self.feature_store.feature_shape

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        imgid = int(row["imgid"])
        visual_inputs = self.feature_store.get_feature(imgid)
        facts = self.fact_lookup[imgid]

        return visual_inputs, facts, imgid


class RAGFeatureCaptionDataset(RAGFeatureDataset):
    """Return (visual_inputs, input_ids, attention_mask, facts) for training."""

    def __init__(self, df, vocab: Vocabulary, features_path: str | Path,
                 rag_contexts_path: str | Path, max_length: int, top_k: int,
                 caption_sort_by: str = "itm"):
        super().__init__(df, features_path, rag_contexts_path, top_k, caption_sort_by)
        self.vocab = vocab
        self.max_length = max_length

    def __getitem__(self, index):
        visual_inputs, facts, _ = super().__getitem__(index)
        row = self.dataframe.iloc[index]
        input_ids, attn_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)

        return visual_inputs, input_ids, attn_mask, facts


def collate_fact_batch(samples):
    """Preserve tuple order and keep variable-length facts grouped by image."""
    if len(samples[0]) == 3:
        visual_inputs, facts, imgids = zip(*samples)
        visual_inputs = default_collate(visual_inputs)
        facts = list(facts)
        imgids = default_collate(imgids)

        return visual_inputs, facts, imgids

    visual_inputs, input_ids, attention_mask, facts = zip(*samples)
    visual_inputs = default_collate(visual_inputs)
    input_ids = default_collate(input_ids)
    attention_mask = default_collate(attention_mask)
    facts = list(facts)

    return visual_inputs, input_ids, attention_mask, facts
