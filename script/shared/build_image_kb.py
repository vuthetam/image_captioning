import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor
import faiss

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_DF_PATH,
    KB_MODEL_ID,
    IMAGES_DIR,
    IMAGE_KB_FAISS_INDEX_PATH,
    IMAGE_KB_METADATA_PATH
)
from src.shared.dataset import RawImageDataset
from src.shared.encoder import CLIPImageEmbeddingEncoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng device: {device}")
    num_gpus = torch.cuda.device_count()
    
    # 1. Load CLIP Model
    print(f"Loading CLIP model ({KB_MODEL_ID})...")
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    extractor = CLIPImageEmbeddingEncoder(**model_kwargs).eval().to(device)
    if num_gpus > 1:
        extractor = torch.nn.DataParallel(extractor)
    
    processor = CLIPProcessor.from_pretrained(KB_MODEL_ID)

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
    dataset = RawImageDataset(metadata_df, IMAGES_DIR, processor=processor)
    
    # Đặt num_workers=0 hoặc 1 nếu chạy test, có thể nâng lên khi chạy thực tế
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. Khởi tạo FAISS Index
    # CLIP large có dimension = 768, ViT-B/32 có dimension = 512.
    base_model = extractor.module.model if isinstance(extractor, torch.nn.DataParallel) else extractor.model
    d = base_model.config.projection_dim
    index = faiss.IndexFlatIP(d)

    # 4. Rút trích Vector theo Batch
    print("Bắt đầu trích xuất Image Embeddings...")
    with torch.no_grad():
        for batch_tensors, _ in tqdm(dataloader, desc="Encoding Images"):
            pixel_values = batch_tensors.to(device)
            
            # Lấy image features
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    image_features = extractor(pixel_values)
            else:
                image_features = extractor(pixel_values)

            # Ép kiểu sang float32 TRƯỚC KHI chuẩn hóa (để tránh lỗi vượt quá giới hạn của float16 gây NaN)
            image_features = image_features.to(torch.float32)
            
            # Chuẩn hóa (Normalize) vector để dùng Inner Product tính ra Cosine Similarity
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
            # Chuyển về numpy và nạp vào FAISS
            embeddings_np = image_features.cpu().numpy()
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
