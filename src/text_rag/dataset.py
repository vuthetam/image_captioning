from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.shared.dataset import H5FeatureStore
from src.shared.vocabulary import Vocabulary


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
    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def _load_rag_lookup(path: str | Path):
    rag_df = pd.read_parquet(path, columns=["imgid", "tokens"])
    if rag_df["imgid"].duplicated().any():
        raise ValueError(f"RAG contexts contain duplicate imgid values: {path}")
    return dict(zip(rag_df["imgid"], rag_df["tokens"]))


class RAGFeatureCaptionDataset(Dataset):
    """Training dataset for V1-style text RAG using precomputed visual tokens."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        features_path: str | Path,
        rag_contexts_path: str | Path,
        max_length: int,
        max_context_length: int,
        top_k: int,
    ) -> None:
        self.feature_store = H5FeatureStore(features_path)
        self.dataframe = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        self.max_context_length = max_context_length
        self.top_k = top_k
        self.rag_lookup = _load_rag_lookup(rag_contexts_path)

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return self.feature_store.feature_shape

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        imgid = int(row["imgid"])
        visual_features = self.feature_store.get_feature(imgid)
        input_ids, attention_mask = self.vocab.encode_from_tokens(
            row["tokens"], self.max_length
        )
        rag_input_ids, rag_attention_mask = encode_rag_context(
            self.rag_lookup.get(imgid, []),
            self.vocab,
            self.max_context_length,
            self.top_k,
        )
        return (
            visual_features,
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            rag_input_ids,
            rag_attention_mask,
        )


class RAGFeatureDataset(Dataset):
    """Inference dataset for V1-style text RAG."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocab: Vocabulary,
        features_path: str | Path,
        rag_contexts_path: str | Path,
        max_context_length: int,
        top_k: int,
    ) -> None:
        self.feature_store = H5FeatureStore(features_path)
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.max_context_length = max_context_length
        self.top_k = top_k
        self.rag_lookup = _load_rag_lookup(rag_contexts_path)

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return self.feature_store.feature_shape

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        imgid = int(self.dataframe.iloc[index]["imgid"])
        visual_features = self.feature_store.get_feature(imgid)
        rag_input_ids, rag_attention_mask = encode_rag_context(
            self.rag_lookup.get(imgid, []),
            self.vocab,
            self.max_context_length,
            self.top_k,
        )
        return visual_features, rag_input_ids, rag_attention_mask, imgid
