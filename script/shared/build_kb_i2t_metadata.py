"""Build train-only caption metadata with FACTUAL graphs and SimpleNLG phrases."""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import model_info
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from accelerate import PartialState
from tqdm.auto import tqdm
from src.shared.config import (
    I2T_KB_MANIFEST_PATH,
    I2T_KB_METADATA_PATH,
    SCENE_GRAPH_BATCH_SIZE,
    TRAIN_DF_PATH,
)
from src.shared.knowledge_graph import CheckedParser, GraphLinearizer

SCENE_GRAPH_MODEL = 'lizhuang144/flan-t5-base-VG-factual-sg-id'

# Fixed extraction protocol, independent of the caption decoder's beam settings.
PARSING_OPTIONS = dict(beam_size=2, max_input_len=128, max_output_len=256)


def metadata_schema():
    strings = pa.list_(pa.string())
    attribute = pa.struct([('object', pa.string()), ('attribute', pa.string())])
    triplet = pa.struct([(k, pa.string()) for k in ('subject', 'predicate', 'object', 'voice')])
    return pa.schema([
        ('imgid', pa.int64()),
        ('filepath', pa.string()), ('filename', pa.string()),
        ('caption', pa.string()), ('tokens', strings), ('objects', strings),
        ('attributes', pa.list_(attribute)), ('triplets', pa.list_(triplet)),
        ('object_texts', strings),
        ('attribute_texts', strings), ('triplet_texts', strings),
    ])


def resolve_model_revision(manifest_path):
    """Reuse the recorded model commit across builds, or pin current HEAD."""
    manifest_path = Path(manifest_path)
    revision = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('model') == SCENE_GRAPH_MODEL:
            revision = manifest.get('revision')
    if not revision:
        try:
            revision = model_info(SCENE_GRAPH_MODEL).sha
        except Exception:
            revision = None
    manifest = {'model': SCENE_GRAPH_MODEL, 'revision': revision}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return revision


def build_metadata(train_df_path, metadata_path, revision, device,
                   shard_index=0, num_shards=1, batch_size=256):
    """Extract and linearize captions for one data shard, streaming to a Parquet file.

    When num_shards > 1 (multi-GPU), each rank writes its own temporary shard file
    which is later merged by merge_shards() on rank 0.
    """
    df = pd.read_parquet(train_df_path).reset_index(drop=True)
    cap_col = 'caption' if 'caption' in df else 'raw'
    required = {'imgid', 'tokens', cap_col}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing training columns: {sorted(required - set(df.columns))}')
    if df.empty or any(not isinstance(s, str) or not s.strip() for s in df[cap_col]):
        raise ValueError('Training captions must be nonempty strings.')
    df_shard = df.iloc[shard_index::num_shards].reset_index(drop=True)
    parser = CheckedParser(SCENE_GRAPH_MODEL, device, revision)
    linearizer = GraphLinearizer()
    schema = metadata_schema()
    suffix = f'.shard{shard_index}.tmp.parquet' if num_shards > 1 else '.tmp.parquet'
    pending = metadata_path.with_suffix(suffix)
    try:
        with pq.ParquetWriter(pending, schema) as writer:
            for start in tqdm(range(0, len(df_shard), batch_size),
                              desc=f'Building scene graphs [shard {shard_index}/{num_shards}]',
                              leave=False, disable=shard_index != 0):
                stop = min(start + batch_size, len(df_shard))
                captions = df_shard.iloc[start:stop][cap_col].tolist()
                graphs = parser.parse(captions, return_text=False, filter_factual_chars=False,
                                      batch_size=SCENE_GRAPH_BATCH_SIZE, **PARSING_OPTIONS)
                if len(graphs) != len(captions):
                    raise ValueError('FACTUAL did not return one graph per caption.')
                rows = []
                for position, graph in enumerate(graphs, start):
                    original = df_shard.iloc[position]
                    try:
                        normalized = linearizer.normalize(graph)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f'Invalid graph at row {position}: {error}') from error
                    if not normalized['objects']:
                        raise ValueError(f'Empty graph at row {position}; review caption.')
                    rows.append({
                        'imgid': int(original['imgid']),
                        'filepath': str(original.get('filepath', '')),
                        'filename': str(original.get('filename', '')),
                        'caption': original[cap_col], 'tokens': list(original['tokens']),
                        **normalized,
                    })
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return pending


def merge_shards(num_shards, metadata_path, output_path):
    """Concatenate rank shards into output_path, then remove the shards."""
    shard_paths = [
        metadata_path.with_suffix(f'.shard{i}.tmp.parquet') for i in range(num_shards)
    ]
    tables = [pq.read_table(p) for p in shard_paths]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, output_path)
    for p in shard_paths:
        p.unlink(missing_ok=True)


def main():
    state = PartialState()
    device = str(state.device)

    I2T_KB_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Only rank 0 resolves (and writes) the manifest to avoid race conditions.
    if state.is_main_process:
        revision = resolve_model_revision(I2T_KB_MANIFEST_PATH)
    state.wait_for_everyone()
    revision = json.loads(I2T_KB_MANIFEST_PATH.read_text(encoding='utf-8'))['revision']

    print(f'[rank {state.process_index}] FACTUAL={SCENE_GRAPH_MODEL}@{revision}; device={device}')

    build_metadata(
        TRAIN_DF_PATH, I2T_KB_METADATA_PATH, revision, device,
        shard_index=state.process_index, num_shards=state.num_processes,
    )

    state.wait_for_everyone()

    if state.is_main_process:
        pending_meta = I2T_KB_METADATA_PATH.with_suffix('.tmp.parquet')
        if state.num_processes > 1:
            merge_shards(state.num_processes, I2T_KB_METADATA_PATH, pending_meta)

        pending_meta.replace(I2T_KB_METADATA_PATH)
        print(f'Saved metadata: {I2T_KB_METADATA_PATH}')


if __name__ == '__main__':
    main()
