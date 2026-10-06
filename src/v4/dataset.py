import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from src.shared.vocabulary import Vocabulary

class _H5FeatureStoreV4:
    '''Lazily reads CLIP features from HDF5 format to avoid multi-processing issues.'''
    def __init__(self, features_path: str | Path) -> None:
        self.features_path = Path(features_path)
        self._h5_file = None
        with h5py.File(self.features_path, "r") as h5_file:
            stored_imgids = np.asarray(h5_file["imgids"], dtype=np.int64)
            self.feature_shape = tuple(h5_file["features"].shape[1:])

        self._imgid_to_index = {
            int(imgid): index for index, imgid in enumerate(stored_imgids.tolist())
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _feature_for_imgid(self, imgid: int) -> torch.Tensor:
        imgid = int(imgid)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        feature_index = self._imgid_to_index[imgid]
        feature = np.asarray(self._h5_file["features"][feature_index])
        return torch.from_numpy(feature)


class FeatureCaptionDatasetV4(_H5FeatureStoreV4, Dataset):
    '''Dataset dùng cho lúc Huấn Luyện (Training) - Trả về ảnh và text (Không RAG)'''
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path, max_length: int):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        _H5FeatureStoreV4.__init__(self, features_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self._feature_for_imgid(imgid)
        
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return visual_feature, input_ids, attention_mask


class FeatureDatasetV4(_H5FeatureStoreV4, Dataset):
    '''Dataset dùng cho lúc Sinh Câu (Inference) - Chỉ trả về ảnh'''
    def __init__(self, df, features_path: str | Path):
        self.df = df.reset_index(drop=True)
        _H5FeatureStoreV4.__init__(self, features_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        visual_feature = self._feature_for_imgid(imgid)
        
        return visual_feature, imgid

class FeatureCaptionDatasetV4_RAG(Dataset):
    '''Dataset dùng cho lúc Huấn Luyện V4 RAG - Trả về (ảnh gốc, ảnh RAG, text)'''
    def __init__(self, df, vocab: Vocabulary, features_path: str | Path, rag_tensors_path: str | Path, max_length: int):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        self.original_store = _H5FeatureStoreV4(features_path)
        self.rag_store = _H5FeatureStoreV4(rag_tensors_path)
        
        # Verify shapes match
        self.feature_shape = self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store._feature_for_imgid(imgid)
        rag_feature = self.rag_store._feature_for_imgid(imgid)
        
        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Return 4 elements for Training
        return visual_feature, rag_feature, input_ids, attention_mask


class FeatureDatasetV4_RAG(Dataset):
    '''Dataset dùng cho lúc Sinh Câu V4 RAG - Trả về (ảnh gốc, ảnh RAG, imgid)'''
    def __init__(self, df, features_path: str | Path, rag_tensors_path: str | Path):
        self.df = df.reset_index(drop=True)
        self.original_store = _H5FeatureStoreV4(features_path)
        self.rag_store = _H5FeatureStoreV4(rag_tensors_path)
        self.feature_shape = self.original_store.feature_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgid = int(row["imgid"])
        
        visual_feature = self.original_store._feature_for_imgid(imgid)
        rag_feature = self.rag_store._feature_for_imgid(imgid)
        
        # Return 3 elements for Inference
        return visual_feature, rag_feature, imgid

