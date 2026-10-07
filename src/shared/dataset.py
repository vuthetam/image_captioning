import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import h5py
import numpy as np

class RawImageDataset(Dataset):
    """
    Dataset dùng chung để đọc ảnh gốc từ đĩa và áp dụng biến đổi.
    Hỗ trợ cả torchvision transform hoặc HuggingFace processor.
    """
    def __init__(self, df, images_path, transform=None, processor=None):
        self.imgids = df['imgid'].tolist()
        self.filepaths = df['filepath'].tolist()
        self.filenames = df['filename'].tolist()
        self.images_path = Path(images_path)
        self.transform = transform
        self.processor = processor

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        path = self.images_path / self.filepaths[idx] / self.filenames[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            
            if self.transform:
                # Dành cho torchvision transform (VD: CLIPVisualEncoder)
                pixel_values = self.transform(rgb_img)
            elif self.processor:
                # Dành cho HF processor (VD: CLIPModel)
                inputs = self.processor(images=rgb_img, return_tensors="pt")
                pixel_values = inputs["pixel_values"][0]
            else:
                pixel_values = rgb_img
                
            return pixel_values, self.imgids[idx]


class H5FeatureStore:
    """Lazily reads features from the project's row-aligned HDF5 format."""

    def __init__(self, features_path: str | Path) -> None:
        self.features_path = Path(features_path)
        self._h5_file: h5py.File | None = None
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

    def get_feature(self, imgid: int) -> torch.Tensor:
        imgid = int(imgid)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        feature_index = self._imgid_to_index[imgid]
        feature = np.asarray(self._h5_file["features"][feature_index])
        return torch.from_numpy(feature)

    def get_cls_token(self, imgid: int) -> torch.Tensor:
        """Chỉ đọc duy nhất CLS token (token đầu tiên) từ ổ cứng, tối ưu Disk I/O"""
        imgid = int(imgid)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        feature_index = self._imgid_to_index[imgid]
        
        # Slice trực tiếp token [0] (CLS) trên HDF5 trước khi nạp vào RAM
        feature = np.asarray(self._h5_file["features"][feature_index, 0])
        return torch.from_numpy(feature)
