import torch
from pathlib import Path
from torch.utils.data import Dataset

from src.shared.dataset import H5FeatureStore
from src.shared.vocabulary import Vocabulary


class FeatureCaptionDatasetV5_RAG(Dataset):
    '''
    Dataset dùng cho lúc Huấn Luyện V5 RAG.
    Trả về: (ảnh gốc, ảnh RAG, scores, text)
    '''
    def __init__(self, df, related_df, vocab: Vocabulary, visual_features_path: str | Path, rag_features_path: str | Path, max_length: int, top_k: int = 4):
        self.df = df.reset_index(drop=True)
        self.related_df = related_df.set_index("imgid")
        self.vocab = vocab
        self.max_length = max_length
        self.top_k = top_k
        self.original_store = H5FeatureStore(visual_features_path)
        self.rag_store = H5FeatureStore(rag_features_path)

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        # 1. Lấy toàn bộ features của ảnh gốc (CLS + Patches) từ tập tương ứng
        visual_feature = self.original_store.get_feature(imgid)
        
        # 2. Lấy CLS features [K, D] và Scores [K] từ RAG
        related_row = self.related_df.loc[imgid]
        retrieved_imgids = related_row["retrieved_imgids"][:self.top_k]
        retrieval_scores = related_row["retrieval_scores"][:self.top_k]
        
        # Lấy token đầu tiên [0] (CLS token) của mỗi ảnh truy hồi
        rag_cls_tokens = [self.rag_store.get_cls_token(r_id) for r_id in retrieved_imgids]
        rag_cls_tokens = torch.stack(rag_cls_tokens) # shape: [K, D]
        rag_score = torch.tensor(retrieval_scores, dtype=torch.float32) # shape: [K]
        
        # 3. Text (Caption)
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return visual_feature, rag_cls_tokens, rag_score, input_ids, attention_mask


class FeatureDatasetV5_RAG(Dataset):
    '''
    Dataset dùng cho lúc Sinh Câu V5 RAG (Inference).
    Trả về: (ảnh gốc, ảnh RAG, scores, imgid)
    '''
    def __init__(self, df, related_df, visual_features_path: str | Path, rag_features_path: str | Path, top_k: int = 4):
        self.df = df.reset_index(drop=True)
        self.related_df = related_df.set_index("imgid")
        self.top_k = top_k
        self.original_store = H5FeatureStore(visual_features_path)
        self.rag_store = H5FeatureStore(rag_features_path)

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store.get_feature(imgid)
        
        related_row = self.related_df.loc[imgid]
        retrieved_imgids = related_row["retrieved_imgids"][:self.top_k]
        retrieval_scores = related_row["retrieval_scores"][:self.top_k]
        
        rag_cls_tokens = [self.rag_store.get_cls_token(r_id) for r_id in retrieved_imgids]
        rag_cls_tokens = torch.stack(rag_cls_tokens)
        rag_score = torch.tensor(retrieval_scores, dtype=torch.float32)
        
        return visual_feature, rag_cls_tokens, rag_score, imgid
