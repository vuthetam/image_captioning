import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import h5py

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_DF_PATH,
    TRAIN_IMAGE_EMBEDDINGS_PATH,
    IMAGE_KB_FAISS_INDEX_PATH,
    IMAGE_KB_METADATA_PATH
)

def main():
    print("Đọc tập train_df để xây dựng KB...")
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    
    # Drop duplicate imgid
    combined_df = train_df.drop_duplicates(subset=['imgid']).reset_index(drop=True)
    metadata_df = combined_df[['imgid', 'filepath', 'filename']].copy()

    # 1. Nạp Image Embeddings đã được trích xuất sẵn từ HDF5
    print(f"Đang nạp image embeddings từ {TRAIN_IMAGE_EMBEDDINGS_PATH.name}...")
    if not TRAIN_IMAGE_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file {TRAIN_IMAGE_EMBEDDINGS_PATH}. Hãy chạy script extract_image_embeddings.py trước!")
        
    with h5py.File(TRAIN_IMAGE_EMBEDDINGS_PATH, "r") as h5f:
        h5_imgids = np.asarray(h5f["imgids"], dtype=np.int64)
        h5_features = np.asarray(h5f["features"], dtype=np.float32)
        
    if len(h5_imgids) != len(metadata_df):
        raise ValueError(f"Số lượng ảnh trong HDF5 ({len(h5_imgids)}) không khớp với metadata ({len(metadata_df)})")
        
    # Kiểm tra tính toàn vẹn của thứ tự imgid
    if not np.array_equal(h5_imgids, metadata_df['imgid'].to_numpy()):
        print("Cảnh báo: Thứ tự imgid trong HDF5 không khớp với metadata. Đang đồng bộ lại...")
        # Tạo mapping để sắp xếp lại features cho đúng với metadata_df
        idx_map = {imgid: idx for idx, imgid in enumerate(h5_imgids)}
        ordered_indices = [idx_map[imgid] for imgid in metadata_df['imgid']]
        h5_features = h5_features[ordered_indices]
        
    # 2. Khởi tạo FAISS Index
    d = h5_features.shape[1]
    print(f"Khởi tạo FAISS IndexFlatIP với dimension = {d}")
    index = faiss.IndexFlatIP(d)

    # 3. Chuẩn hóa L2 và thêm vào FAISS
    faiss.normalize_L2(h5_features)
    index.add(h5_features)

    # 4. Lưu file xuống ổ cứng
    print("\nĐang lưu Image Knowledge Base xuống đĩa...")
    
    # Tạo thư mục chứa nếu chưa có
    IMAGE_KB_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(IMAGE_KB_FAISS_INDEX_PATH))
    metadata_df.to_parquet(IMAGE_KB_METADATA_PATH)
    
    print(f"HOÀN TẤT! Đã lưu tại:\n- {IMAGE_KB_FAISS_INDEX_PATH}\n- {IMAGE_KB_METADATA_PATH}")

if __name__ == "__main__":
    main()
