import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

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

