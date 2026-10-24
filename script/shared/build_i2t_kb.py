"""Build a train-only caption KB with FACTUAL graphs and SimpleNLG phrases."""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
from huggingface_hub import model_info
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from src.shared.config import (
    RETRIEVAL_ENCODER_MODEL,
    I2T_KB_FAISS_INDEX_PATH,
    I2T_KB_MANIFEST_PATH,
    I2T_KB_METADATA_PATH,
    TRAIN_DF_PATH,
)
from src.shared.knowledge_graph import CheckedParser, GraphLinearizer
from src.shared.utils import extract_global_embedding

# Use separate batch sizes for scene graph parsing and CLIP embedding.
SCENE_GRAPH_BATCH_SIZE = 32
KB_EMBEDDING_BATCH_SIZE = 64
SCENE_GRAPH_MODEL = 'lizhuang144/flan-t5-base-VG-factual-sg-id'

# Fixed extraction protocol, independent of the caption decoder's beam settings.
PARSING_OPTIONS = dict(beam_size=5, max_input_len=128, max_output_len=256)


def metadata_schema():
    strings = pa.list_(pa.string())
    attribute = pa.struct([('object', pa.string()), ('attribute', pa.string())])
    triplet = pa.struct([(k, pa.string()) for k in ('subject', 'predicate', 'object', 'voice')])
    return pa.schema([
        ('caption_id', pa.int64()), ('imgid', pa.int64()),
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
        revision = model_info(SCENE_GRAPH_MODEL).sha
    manifest = {'model': SCENE_GRAPH_MODEL, 'revision': revision}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return revision


def build_metadata(train_df_path, metadata_path, revision, device, batch_size=256):
    """Extract and linearize all captions, streaming batches to one Parquet file."""
    df = pd.read_parquet(train_df_path).reset_index(drop=True)
    cap_col = 'caption' if 'caption' in df else 'raw'
    required = {'imgid', 'tokens', cap_col}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing training columns: {sorted(required - set(df.columns))}')
    if df.empty or any(not isinstance(s, str) or not s.strip() for s in df[cap_col]):
        raise ValueError('Training captions must be nonempty strings.')
    parser = CheckedParser(SCENE_GRAPH_MODEL, device, revision)
    linearizer = GraphLinearizer()
    schema = metadata_schema()
    pending = metadata_path.with_suffix('.tmp.parquet')
    try:
        with pq.ParquetWriter(pending, schema) as writer:
            for start in tqdm(range(0, len(df), batch_size), desc='Building scene graphs'):
                stop = min(start + batch_size, len(df))
                captions = df.iloc[start:stop][cap_col].tolist()
                graphs = parser.parse(captions, return_text=False, filter_factual_chars=False,
                                      batch_size=SCENE_GRAPH_BATCH_SIZE, **PARSING_OPTIONS)
                if len(graphs) != len(captions):
                    raise ValueError('FACTUAL did not return one graph per caption.')
                rows = []
                for position, graph in enumerate(graphs, start):
                    original = df.iloc[position]
                    try:
                        normalized = linearizer.normalize(graph)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f'Invalid graph at caption_id={position}: {error}') from error
                    if not normalized['objects']:
                        raise ValueError(f'Empty graph at caption_id={position}; review caption.')
                    rows.append({
                        'caption_id': position, 'imgid': int(original['imgid']),
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


def build_caption_index(metadata_path, index_path, device):
    """Index full captions; semantic units are stored for subsequent experiments."""
    model = CLIPModel.from_pretrained(RETRIEVAL_ENCODER_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(RETRIEVAL_ENCODER_MODEL)
    index = faiss.IndexFlatIP(model.config.projection_dim)
    parquet = pq.ParquetFile(metadata_path)
    for batch in tqdm(parquet.iter_batches(batch_size=KB_EMBEDDING_BATCH_SIZE,
                                         columns=['caption']), desc='Encoding captions'):
        inputs = processor(text=batch.column(0).to_pylist(), padding=True, truncation=True,
                           max_length=model.config.text_config.max_position_embeddings,
                           return_tensors='pt')
        with torch.inference_mode():
            features = extract_global_embedding(model.get_text_features(
                **{k: v.to(device) for k, v in inputs.items()}
            )).float()
            features = torch.nn.functional.normalize(features, dim=-1)
        index.add(features.cpu().numpy())
    if index.ntotal != parquet.metadata.num_rows:
        raise ValueError('Index and metadata row counts differ.')
    pending = index_path.with_suffix('.tmp.faiss')
    faiss.write_index(index, str(pending))
    return pending


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    I2T_KB_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = I2T_KB_MANIFEST_PATH
    revision = resolve_model_revision(manifest_path)
    print(f'FACTUAL={SCENE_GRAPH_MODEL}@{revision}; device={device}')
    metadata_pending = build_metadata(
        TRAIN_DF_PATH, I2T_KB_METADATA_PATH, revision, device
    )
    index_pending = build_caption_index(metadata_pending, I2T_KB_FAISS_INDEX_PATH, device)
    metadata_pending.replace(I2T_KB_METADATA_PATH)
    index_pending.replace(I2T_KB_FAISS_INDEX_PATH)
    print(f'Saved KB: {I2T_KB_METADATA_PATH}\nCaption index: {I2T_KB_FAISS_INDEX_PATH}')


if __name__ == '__main__':
    main()
