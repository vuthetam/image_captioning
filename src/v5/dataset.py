from src.shared.dataset import H5FeatureStore
import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from src.shared.vocabulary import Vocabulary

class FeatureCaptionDatasetV5_RAG(Dataset):
    '''Dataset dùng cho lúc Huấn Luyện V5 RAG - Trả về (ảnh gốc, ảnh RAG, text)'''
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path, rag_tensors_path: str | Path, max_length: int):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        self.original_store = H5FeatureStore(features_path)
        self.rag_store = H5FeatureStore(rag_tensors_path)
        
        # Verify shapes match
        self.feature_shape = self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store.get_feature(imgid)
        rag_feature = self.rag_store.get_feature(imgid)
        
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Return 4 elements for Training
        return visual_feature, rag_feature, input_ids, attention_mask


class FeatureDatasetV5_RAG(Dataset):
    '''Dataset dùng cho lúc Sinh Câu V5 RAG - Trả về (ảnh gốc, ảnh RAG, imgid)'''
    def __init__(self, df, features_path: str | Path, rag_tensors_path: str | Path):
        self.df = df.reset_index(drop=True)
        self.original_store = H5FeatureStore(features_path)
        self.rag_store = H5FeatureStore(rag_tensors_path)
        self.feature_shape = self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store.get_feature(imgid)
        rag_feature = self.rag_store.get_feature(imgid)
        
        # Return 3 elements for Inference
        return visual_feature, rag_feature, imgid

