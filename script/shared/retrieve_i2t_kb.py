"""Retrieve CLIP caption candidates on CPU and save them for separate ITM scoring.

Run with:
    python script/shared/retrieve_i2t_kb.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_IMAGE_EMBEDDINGS_PATH, VAL_IMAGE_EMBEDDINGS_PATH, TEST_IMAGE_EMBEDDINGS_PATH,
    TRAIN_I2T_PATH, VAL_I2T_PATH, TEST_I2T_PATH,
    RETRIEVAL_OPENCLIP_MODEL_ID, I2T_KB_FAISS_INDEX_PATH, I2T_KB_METADATA_PATH,
    I2T_RETRIEVAL_TOP_K,
)

CONTEXT_COLUMNS = ["retrieval_imgids", "captions", "tokens", "objects",
                   "attributes", "triplets", "object_texts", "attribute_texts", "triplet_texts"]
METADATA_COLUMNS = ["imgid", "caption", "tokens", "objects",
                    "attributes", "triplets", "object_texts", "attribute_texts", "triplet_texts"]


def context_schema(kb_schema, column_pairs):
    """Keep nested caption types explicit in the intermediate retrieval file."""
    fields = [pa.field("imgid", pa.int64())]
    fields.extend(pa.field(out, pa.list_(kb_schema.field(source).type))
                  for out, source in column_pairs)
    fields.append(pa.field("clip_scores", pa.list_(pa.float32())))
    return pa.schema(fields, metadata={
        b"retrieval_encoder_model": RETRIEVAL_OPENCLIP_MODEL_ID.encode("utf-8"),
        b"caption_order": b"CLIP similarity descending; self captions excluded",
    })


def retrieve_captions(img_embeddings_path, output_parquet_path, index, kb_metadata,
                      kb_schema, batch_size=128):
    """Stage 1: retrieve all queries in HDF5 order and save CLIP candidates."""
    if batch_size < 1:
        raise ValueError("Retrieval batch size must be positive.")
    img_embeddings_path = Path(img_embeddings_path)
    output_parquet_path = Path(output_parquet_path)
    print(f"\nFAISS: {img_embeddings_path.name}")

    kb_imgids = kb_metadata["imgid"].to_numpy(dtype=np.int64)
    # Một ảnh có nhiều caption trong text KB; lấy dư để loại hết caption của chính nó.
    max_self_captions = int(kb_metadata["imgid"].value_counts().max())
    search_k = min(index.ntotal, I2T_RETRIEVAL_TOP_K + max_self_captions)
    column_pairs = [(out, source) for out, source in zip(CONTEXT_COLUMNS, METADATA_COLUMNS)
                    if source in kb_metadata.columns]
    schema = context_schema(kb_schema, column_pairs)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_parquet_path.with_suffix(".tmp.parquet")

    with h5py.File(img_embeddings_path, "r") as h5f:
        stored_model = h5f.attrs.get("model_id")
        if isinstance(stored_model, bytes):
            stored_model = stored_model.decode("utf-8")
        if stored_model is not None and stored_model != RETRIEVAL_OPENCLIP_MODEL_ID:
            raise ValueError(
                f"Model của {img_embeddings_path} ({stored_model}) không khớp "
                f"RETRIEVAL_OPENCLIP_MODEL_ID ({RETRIEVAL_OPENCLIP_MODEL_ID})."
            )

        query_imgids = np.asarray(h5f["imgids"], dtype=np.int64)
        features = h5f["features"]
        if query_imgids.ndim != 1 or features.ndim != 2 or features.shape[0] != len(query_imgids):
            raise ValueError("HDF5 phải có imgids [N] và image embeddings features [N, D].")
        if features.shape[1] != index.d:
            raise ValueError(
                f"Image embeddings có {features.shape[1]} chiều nhưng text index có {index.d} chiều. "
                "Hãy dùng image embeddings và text KB từ cùng RETRIEVAL_OPENCLIP_MODEL_ID."
            )

        try:
            with pq.ParquetWriter(pending, schema) as writer:
                for start in tqdm(range(0, len(query_imgids), batch_size),
                                  desc="FAISS retrieval", unit="batch", leave=False):
                    stop = min(start + batch_size, len(query_imgids))
                    batch_imgids = query_imgids[start:stop]
                    query_features = np.asarray(features[start:stop], dtype=np.float32)
                    faiss.normalize_L2(query_features)
                    distances, indices = index.search(query_features, search_k)
                    results = []

                    for imgid, retrieved_indices, raw_scores in zip(batch_imgids, indices, distances):
                        # Keep scores aligned with valid FAISS indices and exclude all self captions.
                        valid = (retrieved_indices >= 0) & (retrieved_indices < len(kb_metadata))
                        retrieved_indices = retrieved_indices[valid]
                        raw_scores = raw_scores[valid]
                        not_self = kb_imgids[retrieved_indices] != imgid
                        selected_indices = retrieved_indices[not_self][:I2T_RETRIEVAL_TOP_K]
                        selected_scores = raw_scores[not_self][:I2T_RETRIEVAL_TOP_K]
                        selected_metadata = kb_metadata.iloc[selected_indices]

                        result = {"imgid": int(imgid)}
                        for output_column, metadata_column in column_pairs:
                            result[output_column] = selected_metadata[metadata_column].tolist()
                        result["clip_scores"] = [float(score) for score in selected_scores]
                        results.append(result)
                    writer.write_table(pa.Table.from_pylist(results, schema=schema))
            pending.replace(output_parquet_path)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    print(f"Đã lưu ứng viên CLIP: {output_parquet_path}")


def main():
    splits = [
        (TRAIN_IMAGE_EMBEDDINGS_PATH, TRAIN_I2T_PATH),
        (VAL_IMAGE_EMBEDDINGS_PATH, VAL_I2T_PATH),
        (TEST_IMAGE_EMBEDDINGS_PATH, TEST_I2T_PATH),
    ]
    input_paths = [I2T_KB_FAISS_INDEX_PATH, I2T_KB_METADATA_PATH]
    input_paths.extend(embeddings_path for embeddings_path, _ in splits)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {path}")

    print(f"FAISS trên CPU; model={RETRIEVAL_OPENCLIP_MODEL_ID}")
    index = faiss.read_index(str(I2T_KB_FAISS_INDEX_PATH))
    kb_metadata = pd.read_parquet(I2T_KB_METADATA_PATH)
    if index.ntotal == 0 or index.ntotal != len(kb_metadata):
        raise ValueError("Text index phải không rỗng và có số vector bằng số dòng metadata.")
    missing_columns = {"imgid", "caption", "tokens", "objects"} - set(kb_metadata.columns)
    if missing_columns:
        raise ValueError(f"Text KB metadata thiếu các cột: {sorted(missing_columns)}")
    kb_schema = pq.read_schema(I2T_KB_METADATA_PATH)
    for img_embeddings_path, output_parquet_path in splits:
        retrieve_captions(
            img_embeddings_path, output_parquet_path.with_suffix(".clip.parquet"),
            index, kb_metadata, kb_schema,
        )


if __name__ == "__main__":
    main()
