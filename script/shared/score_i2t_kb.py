"""Score saved CLIP candidates with BLIP ITM without rerunning FAISS retrieval.

Run on two GPUs with:
    accelerate launch --multi_gpu --num_processes 2 script/shared/score_i2t_kb.py
Each process scores its contiguous query shard in RAM, then writes it once.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from accelerate import PartialState
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    TRAIN_I2T_PATH, VAL_I2T_PATH, TEST_I2T_PATH,
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH, IMAGES_DIR,
    BLIP_ITM_MODEL_ID, BLIP_ITM_IMAGE_BATCH_SIZE, RETRIEVAL_OPENCLIP_MODEL_ID,
)
from src.shared.blip_itm import BlipITMScorer


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


def merge_result_shards(output_path, num_shards, expected_rows, schema):
    """Merge whole shards in rank order, preserving the original HDF5 query order."""
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
                    table = parquet.read()
                    writer.write_table(table)
                    written += table.num_rows
                    del table
        if written != expected_rows:
            raise ValueError(f"Expected {expected_rows} query rows, wrote {written}.")
        pending.replace(output_path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    for shard_path in shard_paths:
        shard_path.unlink()


def iter_context_batches(parquet, start, stop, batch_size):
    """Read only row groups overlapping a rank's contiguous query range."""
    if start == stop:
        return
    row_groups = []
    group_start = 0
    batch_start = 0
    for group in range(parquet.num_row_groups):
        group_stop = group_start + parquet.metadata.row_group(group).num_rows
        if group_start >= stop:
            break
        if group_stop > start:
            if not row_groups:
                batch_start = group_start
            row_groups.append(group)
        group_start = group_stop

    for batch in parquet.iter_batches(batch_size=batch_size, row_groups=row_groups):
        local_start = max(start - batch_start, 0)
        local_stop = min(stop - batch_start, batch.num_rows)
        if local_stop > local_start:
            yield batch.slice(local_start, local_stop - local_start)
        batch_start += batch.num_rows
        if batch_start >= stop:
            break


def score_retrieved_captions(input_parquet_path, output_parquet_path, image_paths,
                             scorer, state, itm_image_batch_size=32):
    """Stage 2: score a query shard in RAM, then write it once in original order."""
    if itm_image_batch_size < 1:
        raise ValueError("BLIP ITM image batch size must be positive.")
    output_parquet_path = Path(output_parquet_path)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = output_parquet_path.with_suffix(f".rank{state.process_index}.tmp.parquet")
    with pq.ParquetFile(input_parquet_path) as parquet:
        num_rows = parquet.metadata.num_rows
        schema = parquet.schema_arrow.append(pa.field("itm_scores", pa.list_(pa.float32())))
        metadata = dict(schema.metadata or {})
        metadata.update({
            b"blip_itm_model_id": scorer.model_id.encode("utf-8"),
            b"blip_itm_revision": (scorer.revision or "unknown").encode("utf-8"),
            b"itm_score_definition": b"softmax([non_match, match])[1]",
        })
        schema = schema.with_metadata(metadata)
        split_start = num_rows * state.process_index // state.num_processes
        split_stop = num_rows * (state.process_index + 1) // state.num_processes
        # Load only this rank's rows before scoring; GPU batches remain independent.
        results = [row for batch in iter_context_batches(parquet, split_start, split_stop, 256)
                   for row in batch.to_pylist()]

    try:
        with tqdm(total=len(results), desc="BLIP ITM [rank 0]",
                  unit="image", leave=False, disable=not state.is_main_process) as progress:
            for start in range(0, len(results), itm_image_batch_size):
                image_batch = results[start:start + itm_image_batch_size]
                for result in image_batch:
                    if result["imgid"] not in image_paths:
                        raise ValueError(f"No image path for query imgid={result['imgid']}.")
                if len(image_batch) == 1:
                    result = image_batch[0]
                    itm_scores = [scorer.score(image_paths[result["imgid"]], result["captions"])]
                else:
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
                progress.update(len(image_batch))
        pq.write_table(pa.Table.from_pylist(results, schema=schema), shard_path)
    except BaseException:
        shard_path.unlink(missing_ok=True)
        raise
    del results

    state.wait_for_everyone()
    if state.is_main_process:
        merge_result_shards(output_parquet_path, state.num_processes, num_rows, schema)
        print(f"Đã lưu {num_rows:,} ảnh đã chấm ITM tại {output_parquet_path}")
    state.wait_for_everyone()


def main():
    state = PartialState()
    splits = [
        (TRAIN_DF_PATH, TRAIN_I2T_PATH),
        (VAL_DF_PATH, VAL_I2T_PATH),
        (TEST_DF_PATH, TEST_I2T_PATH),
    ]
    for split_path, output_parquet_path in splits:
        candidates_path = output_parquet_path.with_suffix(".clip.parquet")
        for path in (split_path, candidates_path):
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy {path}; chạy retrieve_i2t_kb.py trước.")
        with pq.ParquetFile(candidates_path) as parquet:
            stored_model = (parquet.schema_arrow.metadata or {}).get(b"retrieval_encoder_model")
            if stored_model is not None and stored_model.decode("utf-8") != RETRIEVAL_OPENCLIP_MODEL_ID:
                raise ValueError(f"Model của {candidates_path} không khớp RETRIEVAL_OPENCLIP_MODEL_ID.")

    if state.is_main_process:
        print(f"BLIP ITM={BLIP_ITM_MODEL_ID}; processes={state.num_processes}; "
              f"image_batch_size={BLIP_ITM_IMAGE_BATCH_SIZE}")
    # Populate the model cache on rank 0 first; each rank then owns its model copy.
    with state.main_process_first():
        scorer = BlipITMScorer(BLIP_ITM_MODEL_ID, state.device)

    for split_path, output_parquet_path in splits:
        image_paths = load_image_paths(split_path)
        score_retrieved_captions(
            output_parquet_path.with_suffix(".clip.parquet"), output_parquet_path,
            image_paths, scorer, state,
            itm_image_batch_size=BLIP_ITM_IMAGE_BATCH_SIZE,
        )


if __name__ == "__main__":
    main()
