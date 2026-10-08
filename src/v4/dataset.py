from src.shared.dataset import H5FeatureStore
import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from src.shared.vocabulary import Vocabulary

class FeatureCaptionDatasetV4(H5FeatureStore, Dataset):
    '''Dataset dùng cho lúc Huấn Luyện (Training) - Trả về ảnh và text (Không RAG)'''
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path, max_length: int):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        H5FeatureStore.__init__(self, features_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.get_feature(imgid)
        
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return visual_feature, input_ids, attention_mask


class FeatureDatasetV4(H5FeatureStore, Dataset):
    '''Dataset dùng cho lúc Sinh Câu (Inference) - Chỉ trả về ảnh'''
    def __init__(self, df, features_path: str | Path):
        self.df = df.reset_index(drop=True)
        H5FeatureStore.__init__(self, features_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        visual_feature = self.get_feature(imgid)
        
        return visual_feature, imgid

class FeatureCaptionDatasetV4_RAG(Dataset):
    '''Dataset dùng cho lúc Huấn Luyện V4 RAG Online - Trả về (ảnh gốc, rag_patches, scores, text)'''
    def __init__(self, df, related_df, vocab: Vocabulary, features_path: str | Path, rag_features_path: str | Path, max_length: int, top_k: int = 4):
        self.df = df.reset_index(drop=True)
        self.related_df = related_df.set_index("imgid")
        self.vocab = vocab
        self.max_length = max_length
        self.top_k = top_k
        self.original_store = H5FeatureStore(features_path)
        self.rag_store = H5FeatureStore(rag_features_path)
        self.feature_shape = self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store.get_feature(imgid) # [197, D]
        
        # Đọc K ảnh RAG (Lấy toàn bộ patch [197, D] chứ không chỉ lấy CLS)
        related_row = self.related_df.loc[imgid]
        retrieved_imgids = related_row["retrieved_imgids"][:self.top_k]
        retrieval_scores = related_row["retrieval_scores"][:self.top_k]
        
        rag_patches = [self.rag_store.get_feature(r_id) for r_id in retrieved_imgids]
        rag_patches = torch.stack(rag_patches) # shape: [K, 197, D]
        rag_score = torch.tensor(retrieval_scores, dtype=torch.float32) # shape: [K]
        
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Return 5 elements for Training
        return visual_feature, rag_patches, rag_score, input_ids, attention_mask


class FeatureDatasetV4_RAG(Dataset):
    '''Dataset dùng cho lúc Sinh Câu V4 RAG Online - Trả về (ảnh gốc, rag_patches, scores, imgid)'''
    def __init__(self, df, related_df, features_path: str | Path, rag_features_path: str | Path, top_k: int = 4):
        self.df = df.reset_index(drop=True)
        self.related_df = related_df.set_index("imgid")
        self.top_k = top_k
        self.original_store = H5FeatureStore(features_path)
        self.rag_store = H5FeatureStore(rag_features_path)
        self.feature_shape = self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store.get_feature(imgid)
        
        related_row = self.related_df.loc[imgid]
        retrieved_imgids = related_row["retrieved_imgids"][:self.top_k]
        retrieval_scores = related_row["retrieval_scores"][:self.top_k]
        
        rag_patches = [self.rag_store.get_feature(r_id) for r_id in retrieved_imgids]
        rag_patches = torch.stack(rag_patches)
        rag_score = torch.tensor(retrieval_scores, dtype=torch.float32)
        
        # Return 4 elements for Inference
        return visual_feature, rag_patches, rag_score, imgid

