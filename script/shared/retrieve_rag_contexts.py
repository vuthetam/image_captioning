"""Retrieve text RAG contexts using precomputed CLIP image embeddings on CPU."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_IMAGE_EMBEDDINGS_PATH, VAL_IMAGE_EMBEDDINGS_PATH, TEST_IMAGE_EMBEDDINGS_PATH,
    TRAIN_RAG_CONTEXTS_PATH, VAL_RAG_CONTEXTS_PATH, TEST_RAG_CONTEXTS_PATH,
    RETRIEVAL_ENCODER_MODEL, TEXT_KB_FAISS_INDEX_PATH, TEXT_KB_METADATA_PATH,
)

TARGET_K = 8
CONTEXT_COLUMNS = ["retrieval_imgids", "captions", "tokens", "objects", "relations"]
METADATA_COLUMNS = ["imgid", "caption", "tokens", "objects", "relations"]


def process_and_retrieve(img_embeddings_path, output_parquet_path, index, kb_metadata, batch_size=256):
    img_embeddings_path = Path(img_embeddings_path)
    output_parquet_path = Path(output_parquet_path)
    print(f"\nĐang xử lý {img_embeddings_path.name}...")

    kb_imgids = kb_metadata["imgid"].to_numpy(dtype=np.int64)
    # Một ảnh có nhiều caption trong text KB; lấy dư để loại hết caption của chính nó.
    max_self_captions = int(kb_metadata["imgid"].value_counts().max())
    search_k = min(index.ntotal, TARGET_K + max_self_captions)
    results = []

    with h5py.File(img_embeddings_path, "r") as h5f:
        stored_model = h5f.attrs.get("model_id")
        if stored_model is not None and stored_model != RETRIEVAL_ENCODER_MODEL:
            raise ValueError(
                f"Model của {img_embeddings_path} ({stored_model}) không khớp "
                f"RETRIEVAL_ENCODER_MODEL ({RETRIEVAL_ENCODER_MODEL})."
            )

        query_imgids = np.asarray(h5f["imgids"], dtype=np.int64)
        features = h5f["features"]
        if query_imgids.ndim != 1 or features.ndim != 2 or features.shape[0] != len(query_imgids):
            raise ValueError("HDF5 phải có imgids [N] và image embeddings features [N, D].")
        if features.shape[1] != index.d:
            raise ValueError(
                f"Image embeddings có {features.shape[1]} chiều nhưng text index có {index.d} chiều. "
                "Hãy dùng image embeddings và text KB từ cùng RETRIEVAL_ENCODER_MODEL."
            )

        # Đọc từng batch để không phải nạp toàn bộ embedding của split vào RAM.
        for start in tqdm(range(0, len(query_imgids), batch_size), desc="Retrieving", leave=False):
            batch_imgids = query_imgids[start:start + batch_size]
            query_features = np.asarray(features[start:start + batch_size], dtype=np.float32)
            faiss.normalize_L2(query_features)
            distances, indices = index.search(query_features, search_k)

            for imgid, retrieved_indices, raw_scores in zip(batch_imgids, indices, distances):
                # FAISS có thể trả -1 nếu không đủ kết quả; giữ scores và indices thẳng hàng.
                valid = (retrieved_indices >= 0) & (retrieved_indices < len(kb_metadata))
                retrieved_indices = retrieved_indices[valid]
                raw_scores = raw_scores[valid]
                not_self = kb_imgids[retrieved_indices] != imgid
                selected_indices = retrieved_indices[not_self][:TARGET_K]
                selected_scores = raw_scores[not_self][:TARGET_K]
                selected_metadata = kb_metadata.iloc[selected_indices]

                result = {"imgid": int(imgid)}
                for output_column, metadata_column in zip(CONTEXT_COLUMNS, METADATA_COLUMNS):
                    result[output_column] = selected_metadata[metadata_column].tolist()
                result["retrieval_scores"] = [float(score) for score in selected_scores]
                results.append(result)

    results_df = pd.DataFrame(
        results, columns=["imgid", *CONTEXT_COLUMNS, "retrieval_scores"]
    )
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_parquet_path, index=False)
    print(f"Đã lưu {len(results_df):,} ảnh tại {output_parquet_path}")


def main():
    splits = [
        (TRAIN_IMAGE_EMBEDDINGS_PATH, TRAIN_RAG_CONTEXTS_PATH),
        (VAL_IMAGE_EMBEDDINGS_PATH, VAL_RAG_CONTEXTS_PATH),
        (TEST_IMAGE_EMBEDDINGS_PATH, TEST_RAG_CONTEXTS_PATH),
    ]
    for path in [TEXT_KB_FAISS_INDEX_PATH, TEXT_KB_METADATA_PATH, *(p for p, _ in splits)]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {path}")

    print(f"Truy hồi từ image embeddings đã lưu ({RETRIEVAL_ENCODER_MODEL}) trên CPU.")
    print("Đang tải Text FAISS Index và Knowledge Base Metadata...")
    index = faiss.read_index(str(TEXT_KB_FAISS_INDEX_PATH))
    kb_metadata = pd.read_parquet(TEXT_KB_METADATA_PATH)
    if index.ntotal == 0 or index.ntotal != len(kb_metadata):
        raise ValueError("Text index phải không rỗng và có số vector bằng số dòng metadata.")
    missing_columns = set(METADATA_COLUMNS) - set(kb_metadata.columns)
    if missing_columns:
        raise ValueError(f"Text KB metadata thiếu các cột: {sorted(missing_columns)}")

    for img_embeddings_path, output_parquet_path in splits:
        process_and_retrieve(img_embeddings_path, output_parquet_path, index, kb_metadata)


if __name__ == "__main__":
    main()
