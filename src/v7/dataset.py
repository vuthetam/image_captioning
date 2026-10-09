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


class RAGFeatureDataset(Dataset):
    """Load independent image and caption retrieval results for each query image."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        features_path: str | Path,
        related_features_path: str | Path,
        rag_contexts_path: str | Path,
        related_images_path: str | Path,
        max_context_length: int,
        top_k_captions: int,
        top_k_images: int,
    ) -> None:
        self.dataframe = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_context_length = max_context_length
        self.top_k_captions = top_k_captions
        self.top_k_images = top_k_images
        if top_k_captions < 1 or top_k_images < 1 or max_context_length < 1:
            raise ValueError("Retrieval K values and max_context_length must be positive.")

        self.feature_store = H5FeatureStore(features_path)
        if Path(features_path).resolve() == Path(related_features_path).resolve():
            self.related_store = self.feature_store
        else:
            self.related_store = H5FeatureStore(related_features_path)
        if self.feature_store.feature_shape[-1] != self.related_store.feature_shape[-1]:
            raise ValueError("Original and related visual features must have the same dimension.")

        rag_df = pd.read_parquet(rag_contexts_path, columns=["imgid", "tokens"])
        related_df = pd.read_parquet(related_images_path, columns=["imgid", "retrieved_imgids"])
        if rag_df["imgid"].duplicated().any() or related_df["imgid"].duplicated().any():
            raise ValueError("Retrieval files must contain one row per query imgid.")
        self.rag_lookup = dict(zip(rag_df["imgid"], rag_df["tokens"]))
        self.related_lookup = dict(zip(related_df["imgid"], related_df["retrieved_imgids"]))

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return self.feature_store.feature_shape

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        imgid = int(self.dataframe.iloc[index]["imgid"])
        visual_features = self.feature_store.get_feature(imgid)
        related_imgids = self.related_lookup[imgid][:self.top_k_images]
        if len(related_imgids) != self.top_k_images:
            raise ValueError(f"Image {imgid} has fewer than {self.top_k_images} related images.")
        related_cls_tokens = torch.stack([self.related_store.get_cls_token(related_id) for related_id in related_imgids])
        rag_input_ids, rag_attention_mask = encode_rag_context(
            self.rag_lookup[imgid], self.vocab, self.max_context_length, self.top_k_captions
        )
        return visual_features, related_cls_tokens, rag_input_ids, rag_attention_mask, imgid


class RAGFeatureCaptionDataset(RAGFeatureDataset):
    """Add target caption tokens to the same retrieval inputs used at inference."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        features_path: str | Path,
        related_features_path: str | Path,
        rag_contexts_path: str | Path,
        related_images_path: str | Path,
        max_length: int,
        max_context_length: int,
        top_k_captions: int,
        top_k_images: int,
    ) -> None:
        super().__init__(
            df=df,
            vocab=vocab,
            features_path=features_path,
            related_features_path=related_features_path,
            rag_contexts_path=rag_contexts_path,
            related_images_path=related_images_path,
            max_context_length=max_context_length,
            top_k_captions=top_k_captions,
            top_k_images=top_k_images,
        )
        self.max_length = max_length

    def __getitem__(self, index: int):
        visual_features, related_cls_tokens, rag_input_ids, rag_attention_mask, imgid = super().__getitem__(index)
        row = self.dataframe.iloc[index]
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return visual_features, related_cls_tokens, input_ids, attention_mask, rag_input_ids, rag_attention_mask
