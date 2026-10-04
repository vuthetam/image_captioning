import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import faiss
from PIL import Image

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_DF_PATH,
    VAL_DF_PATH,
    KB_MODEL_ID,
    IMAGES_DIR,
    IMAGE_KB_FAISS_INDEX_PATH,
    IMAGE_KB_METADATA_PATH
)
from src.shared.utils import extract_clip_features

class ImageDataset(Dataset):
    def __init__(self, df, images_dir, processor):
        self.imgids = df['imgid'].tolist()
        self.filepaths = df['filepath'].tolist()
        self.filenames = df['filename'].tolist()
        self.images_dir = Path(images_dir)
        self.processor = processor

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        path = self.images_dir / self.filepaths[idx] / self.filenames[idx]
        with Image.open(path) as img:
            # CLIPProcessor tự lo việc resize, center crop, normalize, và to_tensor
            inputs = self.processor(images=img.convert("RGB"), return_tensors="pt")
            
        # processor trả về dạng batch (shape: [1, C, H, W]), ta lấy phần tử [0]
        return inputs['pixel_values'][0]

class CLIPFeatureExtractorWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng device: {device}")
    num_gpus = torch.cuda.device_count()
    
    # 1. Load CLIP Model
    print(f"Loading CLIP model ({KB_MODEL_ID})...")
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = CLIPModel.from_pretrained(KB_MODEL_ID, **model_kwargs).to(device)
    processor = CLIPProcessor.from_pretrained(KB_MODEL_ID)
    model.eval()

    # Tạo bộ bọc (wrapper) để chạy DataParallel cho hàm get_image_features
    extractor = CLIPFeatureExtractorWrapper(model)
    if num_gpus > 1:
        extractor = torch.nn.DataParallel(extractor)

    # 2. Đọc tập Train và loại bỏ trùng lặp imgid
    print("Đọc tập train_df để xây dựng KB...")
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    
    # Drop duplicate imgid
    print(f"Tổng số hàng trước khi loại trùng lặp: {len(train_df):,}")
    combined_df = train_df.drop_duplicates(subset=['imgid']).reset_index(drop=True)
    print(f"Tổng số ảnh (imgid) độc lập cần mã hóa (Train only): {len(combined_df):,}")
    
    # Chỉ giữ lại metadata tối giản
    metadata_df = combined_df[['imgid', 'filepath', 'filename']].copy()

    # Khởi tạo Dataset và DataLoader
    dataset = ImageDataset(metadata_df, IMAGES_DIR, processor)
    
    # Đặt num_workers=0 hoặc 1 nếu chạy test, có thể nâng lên khi chạy thực tế
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. Khởi tạo FAISS Index
    # CLIP large có dimension = 768, ViT-B/32 có dimension = 512.
    d = model.config.projection_dim
    index = faiss.IndexFlatIP(d)

    # 4. Rút trích Vector theo Batch
    print("Bắt đầu trích xuất Image Embeddings...")
    with torch.no_grad():
        for pixel_values in tqdm(dataloader, desc="Encoding Images"):
            pixel_values = pixel_values.to(device)
            
            # Lấy image features
            if num_gpus > 1:
                raw_output = extractor(pixel_values)
            else:
                raw_output = model.get_image_features(pixel_values=pixel_values)
            
            # Xử lý tương thích mọi phiên bản transformers để rút ra Tensor vector
            image_features = extract_clip_features(raw_output)

            # Chuẩn hóa (Normalize) vector để dùng Inner Product tính ra Cosine Similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Chuyển về numpy float32 và nạp vào FAISS
            embeddings_np = image_features.cpu().numpy().astype('float32')
            index.add(embeddings_np)

    # 5. Lưu file xuống ổ cứng
    print("\nĐang lưu Image Knowledge Base xuống đĩa...")
    
    # Tạo thư mục chứa nếu chưa có
    IMAGE_KB_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(IMAGE_KB_FAISS_INDEX_PATH))
    metadata_df.to_parquet(IMAGE_KB_METADATA_PATH)
    
    print(f"HOÀN TẤT! Đã lưu tại:\n- {IMAGE_KB_FAISS_INDEX_PATH}\n- {IMAGE_KB_METADATA_PATH}")

if __name__ == "__main__":
    main()
