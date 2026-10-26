"""Retrieve captions with CLIP/FAISS on CPU, then score them with BLIP ITM.

Run on multi GPUs with:
    accelerate launch script/shared/retrieve_i2t_kb.py
Each process owns a BLIP model and a disjoint range of query images.
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
from accelerate import PartialState
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_IMAGE_EMBEDDINGS_PATH, VAL_IMAGE_EMBEDDINGS_PATH, TEST_IMAGE_EMBEDDINGS_PATH,
    TRAIN_I2T_PATH, VAL_I2T_PATH, TEST_I2T_PATH,
    RETRIEVAL_ENCODER_MODEL, I2T_KB_FAISS_INDEX_PATH, I2T_KB_METADATA_PATH,
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH, IMAGES_DIR,
    BLIP_ITM_MODEL_ID, I2T_RETRIEVAL_TOP_K, BLIP_ITM_IMAGE_BATCH_SIZE,
)
from src.shared.blip_itm import BlipITMScorer

CONTEXT_COLUMNS = ["retrieval_imgids", "captions", "tokens", "objects",
                   "attributes", "triplets", "object_texts", "attribute_texts", "triplet_texts"]
METADATA_COLUMNS = ["imgid", "caption", "tokens", "objects",
                    "attributes", "triplets", "object_texts", "attribute_texts", "triplet_texts"]


def load_image_paths(split_path):
    """Resolve query images using the same path convention as RawImageDataset."""
    images = pd.read_parquet(split_path, columns=["imgid", "filepath", "filename"])
    images = images.drop_duplicates()
    if images["imgid"].duplicated().any():
        raise ValueError(f"Conflicting image paths for the same imgid in {split_path}.")
    paths = {}
    for row in images.itertuples(index=False):
        if not isinstance(row.filepath, str) or not isinstance(row.filename, str) or not row.filename:
            raise ValueError(f"Invalid image path for imgid={row.imgid} in {split_path}.")
        paths[int(row.imgid)] = IMAGES_DIR / row.filepath / row.filename
    return paths


def context_schema(kb_schema, column_pairs, scorer):
    """Keep nested context types explicit, including for ranks with no images."""
    fields = [pa.field("imgid", pa.int64())]
    fields.extend(pa.field(out, pa.list_(kb_schema.field(source).type))
                  for out, source in column_pairs)
    fields.extend(pa.field(name, pa.list_(pa.float32()))
                  for name in ("clip_scores", "itm_scores"))
    return pa.schema(fields, metadata={
        b"retrieval_encoder_model": RETRIEVAL_ENCODER_MODEL.encode("utf-8"),
        b"blip_itm_model_id": scorer.model_id.encode("utf-8"),
        b"blip_itm_revision": (scorer.revision or "unknown").encode("utf-8"),
        b"itm_score_definition": b"softmax([non_match, match])[1]",
        b"caption_order": b"CLIP similarity descending; self captions excluded",
    })


def merge_result_shards(output_path, num_shards, expected_rows, schema):
    """Stream shards in rank order, preserving the original HDF5 query order."""
    shard_paths = [output_path.with_suffix(f".rank{rank}.tmp.parquet")
                   for rank in range(num_shards)]
    pending = output_path.with_suffix(".tmp.parquet")
    try:
        written = 0
        with pq.ParquetWriter(pending, schema) as writer:
            for rank, shard_path in enumerate(shard_paths):
                expected_shard_rows = (
                    expected_rows * (rank + 1) // num_shards
                    - expected_rows * rank // num_shards
                )
                with pq.ParquetFile(shard_path) as parquet:
                    if parquet.metadata.num_rows != expected_shard_rows:
                        raise ValueError(f"Unexpected row count in {shard_path}.")
                    for batch in parquet.iter_batches(batch_size=256):
                        writer.write_batch(batch)
                        written += batch.num_rows
        if written != expected_rows:
            raise ValueError(f"Expected {expected_rows} query rows, wrote {written}.")
        pending.replace(output_path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    for shard_path in shard_paths:
        shard_path.unlink()


def process_and_retrieve(img_embeddings_path, output_parquet_path, index, kb_metadata,
                         image_paths, scorer, state, kb_schema, batch_size=256,
                         itm_image_batch_size=32):
    if batch_size < 1:
        raise ValueError("Retrieval batch size must be positive.")
    if itm_image_batch_size < 1:
        raise ValueError("BLIP ITM image batch size must be positive.")
    img_embeddings_path = Path(img_embeddings_path)
    output_parquet_path = Path(output_parquet_path)
    print(f"\n[rank {state.process_index}] Đang xử lý {img_embeddings_path.name}...")

    kb_imgids = kb_metadata["imgid"].to_numpy(dtype=np.int64)
    # Một ảnh có nhiều caption trong text KB; lấy dư để loại hết caption của chính nó.
    max_self_captions = int(kb_metadata["imgid"].value_counts().max())
    search_k = min(index.ntotal, I2T_RETRIEVAL_TOP_K + max_self_captions)
    column_pairs = [(out, source) for out, source in zip(CONTEXT_COLUMNS, METADATA_COLUMNS)
                    if source in kb_metadata.columns]
    schema = context_schema(kb_schema, column_pairs, scorer)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = output_parquet_path.with_suffix(f".rank{state.process_index}.tmp.parquet")

    with h5py.File(img_embeddings_path, "r") as h5f:
        stored_model = h5f.attrs.get("model_id")
        if isinstance(stored_model, bytes):
            stored_model = stored_model.decode("utf-8")
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

        # Contiguous, disjoint shards also keep HDF5 reads sequential on each rank.
        split_start = len(query_imgids) * state.process_index // state.num_processes
        split_stop = len(query_imgids) * (state.process_index + 1) // state.num_processes
        try:
            with pq.ParquetWriter(shard_path, schema) as writer:
                for start in tqdm(range(split_start, split_stop, batch_size),
                                  desc=f"Retrieving + ITM [rank {state.process_index}]", leave=False):
                    stop = min(start + batch_size, split_stop)
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
                        if int(imgid) not in image_paths:
                            raise ValueError(f"No image path for query imgid={imgid}.")
                        results.append(result)

                    for itm_start in range(0, len(results), itm_image_batch_size):
                        image_batch = results[itm_start:itm_start + itm_image_batch_size]
                        itm_scores = scorer.score_batch(
                            [image_paths[result["imgid"]] for result in image_batch],
                            [result["captions"] for result in image_batch],
                        )
                        if len(itm_scores) != len(image_batch):
                            raise ValueError("ITM score group count differs from image batch size.")
                        for result, scores in zip(image_batch, itm_scores):
                            if len(scores) != len(result["captions"]):
                                raise ValueError(
                                    f"ITM score count differs from caption count for imgid={result['imgid']}."
                                )
                            result["itm_scores"] = scores
                    writer.write_table(pa.Table.from_pylist(results, schema=schema))
        except BaseException:
            shard_path.unlink(missing_ok=True)
            raise

    state.wait_for_everyone()
    if state.is_main_process:
        merge_result_shards(output_parquet_path, state.num_processes, len(query_imgids), schema)
        print(f"Đã lưu {len(query_imgids):,} ảnh tại {output_parquet_path}")
    state.wait_for_everyone()


def main():
    state = PartialState()
    splits = [
        (TRAIN_IMAGE_EMBEDDINGS_PATH, TRAIN_DF_PATH, TRAIN_I2T_PATH),
        (VAL_IMAGE_EMBEDDINGS_PATH, VAL_DF_PATH, VAL_I2T_PATH),
        (TEST_IMAGE_EMBEDDINGS_PATH, TEST_DF_PATH, TEST_I2T_PATH),
    ]
    input_paths = [I2T_KB_FAISS_INDEX_PATH, I2T_KB_METADATA_PATH]
    for embeddings_path, split_path, _ in splits:
        input_paths.extend([embeddings_path, split_path])
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {path}")

    print(f"[rank {state.process_index}] CLIP/FAISS={RETRIEVAL_ENCODER_MODEL} (CPU); "
          f"BLIP ITM={BLIP_ITM_MODEL_ID}; device={state.device}")
    index = faiss.read_index(str(I2T_KB_FAISS_INDEX_PATH))
    kb_metadata = pd.read_parquet(I2T_KB_METADATA_PATH)
    if index.ntotal == 0 or index.ntotal != len(kb_metadata):
        raise ValueError("Text index phải không rỗng và có số vector bằng số dòng metadata.")
    missing_columns = {"imgid", "caption", "tokens", "objects"} - set(kb_metadata.columns)
    if missing_columns:
        raise ValueError(f"Text KB metadata thiếu các cột: {sorted(missing_columns)}")
    kb_schema = pq.read_schema(I2T_KB_METADATA_PATH)

    # Populate the model cache on rank 0 first; each rank then owns its model copy.
    with state.main_process_first():
        scorer = BlipITMScorer(BLIP_ITM_MODEL_ID, state.device)

    for img_embeddings_path, split_path, output_parquet_path in splits:
        image_paths = load_image_paths(split_path)
        process_and_retrieve(
            img_embeddings_path, output_parquet_path, index, kb_metadata,
            image_paths, scorer, state, kb_schema,
            itm_image_batch_size=BLIP_ITM_IMAGE_BATCH_SIZE,
        )


if __name__ == "__main__":
    main()
