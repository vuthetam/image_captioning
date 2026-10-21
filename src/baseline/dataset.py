from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.shared.dataset import H5FeatureStore
from src.shared.vocabulary import Vocabulary

class FeatureCaptionDataset(H5FeatureStore, Dataset):
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


class FeatureDataset(H5FeatureStore, Dataset):
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

