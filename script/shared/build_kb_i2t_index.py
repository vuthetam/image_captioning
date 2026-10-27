"""Build a caption FAISS index from existing I2T metadata in stored row order."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import pyarrow.parquet as pq
import torch
from accelerate import PartialState
from tqdm.auto import tqdm

from src.shared.config import (
    RETRIEVAL_OPENCLIP_MODEL_ID,
    I2T_KB_FAISS_INDEX_PATH,
    I2T_KB_METADATA_PATH,
    KB_I2T_INDEX_BATCH_SIZE,
)
from src.shared.openclip_encoder import load_openclip_retrieval_components


def iter_caption_batches(parquet, start, stop):
    """Read a contiguous row range, skipping Parquet row groups outside it."""
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

    for batch in parquet.iter_batches(batch_size=KB_I2T_INDEX_BATCH_SIZE,
                                     row_groups=row_groups, columns=['caption']):
        local_start = max(start - batch_start, 0)
        local_stop = min(stop - batch_start, batch.num_rows)
        if local_stop > local_start:
            yield batch.slice(local_start, local_stop - local_start)
        batch_start += batch.num_rows
        if batch_start >= stop:
            break


def build_caption_index(metadata_path, index_path, device, shard_index=0, num_shards=1):
    """Encode a disjoint row range; rank-order concatenation restores metadata order."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid caption shard index or count.')
    if KB_I2T_INDEX_BATCH_SIZE < 1:
        raise ValueError('Caption embedding batch size must be positive.')
    with pq.ParquetFile(metadata_path) as parquet:
        num_rows = parquet.metadata.num_rows
        if num_rows == 0:
            raise ValueError('Caption metadata must not be empty.')
        start = num_rows * shard_index // num_shards
        stop = num_rows * (shard_index + 1) // num_shards
        model, _, tokenizer = load_openclip_retrieval_components(
            RETRIEVAL_OPENCLIP_MODEL_ID, device=device,
        )
        index = faiss.IndexFlatIP(model.visual.output_dim)
        with tqdm(total=stop - start, desc=f'Encoding captions [rank {shard_index}]',
                  unit='caption', position=shard_index, leave=False) as progress:
            for batch in iter_caption_batches(parquet, start, stop):
                tokens = tokenizer(batch.column(0).to_pylist()).to(device)
                with torch.inference_mode():
                    features = model.encode_text(tokens).float()
                    features = torch.nn.functional.normalize(features, dim=-1)
                if tuple(features.shape) != (batch.num_rows, index.d):
                    raise ValueError('Caption embedding shape differs from the input batch.')
                index.add(features.cpu().numpy())
                progress.update(batch.num_rows)
        if index.ntotal != stop - start:
            raise ValueError('Index shard and metadata range row counts differ.')
    suffix = f'.rank{shard_index}.tmp.faiss' if num_shards > 1 else '.tmp.faiss'
    pending = index_path.with_suffix(suffix)
    faiss.write_index(index, str(pending))
    return pending


def merge_index_shards(index_path, num_shards, expected_rows):
    """Append contiguous rank shards in metadata order and validate their sizes."""
    shard_paths = [index_path.with_suffix(f'.rank{rank}.tmp.faiss')
                   for rank in range(num_shards)]
    index = None
    for rank, shard_path in enumerate(shard_paths):
        shard = faiss.read_index(str(shard_path))
        expected_shard_rows = (expected_rows * (rank + 1) // num_shards
                               - expected_rows * rank // num_shards)
        if shard.ntotal != expected_shard_rows:
            raise ValueError(f'Unexpected caption index row count in {shard_path}.')
        if index is None:
            index = shard
        else:
            index.merge_from(shard)
    if index is None or index.ntotal != expected_rows:
        raise ValueError('Merged index and metadata row counts differ.')
    pending = index_path.with_suffix('.tmp.faiss')
    faiss.write_index(index, str(pending))
    for shard_path in shard_paths:
        shard_path.unlink()
    return pending


def main():
    state = PartialState()
    if not I2T_KB_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f'Metadata not found: {I2T_KB_METADATA_PATH}. '
            'Run build_kb_i2t_metadata.py first or copy existing metadata to this path.'
        )
    if state.is_main_process:
        I2T_KB_FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    state.wait_for_everyone()
    if state.is_main_process:
        print(f'[rank {state.process_index}] CLIP={RETRIEVAL_OPENCLIP_MODEL_ID}; '
              f'device={state.device}; batch size={KB_I2T_INDEX_BATCH_SIZE}')
    pending = build_caption_index(
        I2T_KB_METADATA_PATH, I2T_KB_FAISS_INDEX_PATH, str(state.device),
        shard_index=state.process_index, num_shards=state.num_processes,
    )
    state.wait_for_everyone()
    if state.is_main_process:
        if state.num_processes > 1:
            with pq.ParquetFile(I2T_KB_METADATA_PATH) as parquet:
                expected_rows = parquet.metadata.num_rows
            pending = merge_index_shards(
                I2T_KB_FAISS_INDEX_PATH, state.num_processes, expected_rows,
            )
        pending.replace(I2T_KB_FAISS_INDEX_PATH)
        print(f'Saved caption index: {I2T_KB_FAISS_INDEX_PATH}')
    state.wait_for_everyone()

if __name__ == '__main__':
    main()
