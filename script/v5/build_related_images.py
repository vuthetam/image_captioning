import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_IMAGE_EMBEDDINGS_PATH, VAL_IMAGE_EMBEDDINGS_PATH, TEST_IMAGE_EMBEDDINGS_PATH,
    TRAIN_RELATED_IMAGES_PATH, VAL_RELATED_IMAGES_PATH, TEST_RELATED_IMAGES_PATH,
    IMAGE_KB_FAISS_INDEX_PATH, IMAGE_KB_METADATA_PATH,
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
)

TOP_K = 8

def process_and_retrieve(embeddings_path, output_parquet_path, faiss_index, kb_imgids, meta_dict):
    print(f"\n--- Đang xử lý {embeddings_path.name} ---")
    
    # 1. Load query features and imgids from HDF5
    print("Đang nạp mảng image embeddings vào bộ nhớ...")
    with h5py.File(embeddings_path, "r") as h5f:
        query_imgids = np.asarray(h5f["imgids"], dtype=np.int64)
        query_features = np.asarray(h5f["features"], dtype=np.float32)
        
    faiss.normalize_L2(query_features)
        
    print(f"Đã nạp {len(query_imgids)} vector truy vấn. Đang chạy FAISS search...")
    
    # 2. Search FAISS
    # Tìm K+1 phòng trường hợp tự lấy lại ảnh chính nó
    distances, indices = faiss_index.search(query_features, TOP_K + 1)
    
    # 3. Process results
    results = []
    
    for i, imgid in enumerate(query_imgids):
        raw_scores = distances[i]
        raw_retrieved_idx = indices[i]
        
        # Ánh xạ từ faiss index sang imgid thật
        raw_retrieved_imgids = [kb_imgids[idx] for idx in raw_retrieved_idx if idx != -1]
        
        valid_imgids = []
        valid_scores = []
        
        for r_id, score in zip(raw_retrieved_imgids, raw_scores):
            if r_id != imgid: # Loại bỏ ảnh chính nó
                valid_imgids.append(r_id)
                valid_scores.append(score)
        
        # Cắt đúng Top K
        top_k_ids = valid_imgids[:TOP_K]
        top_k_scores = valid_scores[:TOP_K]
        
        orig_meta = meta_dict.get(int(imgid), {"filepath": "", "filename": ""})
        r_filepaths = [meta_dict.get(r_id, {"filepath": ""})["filepath"] for r_id in top_k_ids]
        r_filenames = [meta_dict.get(r_id, {"filename": ""})["filename"] for r_id in top_k_ids]

        results.append({
            "imgid": int(imgid),
            "filepath": orig_meta["filepath"],
            "filename": orig_meta["filename"],
            "retrieved_imgids": top_k_ids,
            "retrieved_filepaths": r_filepaths,
            "retrieved_filenames": r_filenames,
            "retrieval_scores": [float(s) for s in top_k_scores]
        })
        
    # 4. Save to parquet
    df = pd.DataFrame(results)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet_path)
    print(f"Đã lưu kết quả tại {output_parquet_path} (Kích thước: {df.shape})")

def main():
    if not IMAGE_KB_FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Lỗi: Không tìm thấy {IMAGE_KB_FAISS_INDEX_PATH}")
        
    print("Đang tải FAISS Image Index và KB Metadata...")
    faiss_index = faiss.read_index(str(IMAGE_KB_FAISS_INDEX_PATH))
    kb_df = pd.read_parquet(IMAGE_KB_METADATA_PATH)
    kb_imgids = kb_df["imgid"].tolist()
    
    print("Đang tải metadata từ các tập train/val/test...")
    df_train = pd.read_parquet(TRAIN_DF_PATH)
    df_val = pd.read_parquet(VAL_DF_PATH)
    df_test = pd.read_parquet(TEST_DF_PATH)
    df_all = pd.concat([df_train, df_val, df_test]).drop_duplicates(subset=["imgid"])
    meta_dict = df_all.set_index("imgid")[["filepath", "filename"]].to_dict(orient="index")
    
    splits = [
        (TRAIN_IMAGE_EMBEDDINGS_PATH, TRAIN_RELATED_IMAGES_PATH),
        (VAL_IMAGE_EMBEDDINGS_PATH, VAL_RELATED_IMAGES_PATH),
        (TEST_IMAGE_EMBEDDINGS_PATH, TEST_RELATED_IMAGES_PATH)
    ]
    
    for emb_path, out_path in splits:
        if emb_path.exists():
            process_and_retrieve(emb_path, out_path, faiss_index, kb_imgids, meta_dict)
        else:
            raise FileNotFoundError(f"Lỗi: Không tìm thấy {emb_path}")

if __name__ == "__main__":
    main()