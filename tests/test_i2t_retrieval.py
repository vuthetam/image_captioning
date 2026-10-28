"""Exercise the two retrieval stages with real FAISS, HDF5 and Parquet files."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import faiss
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from script.shared import retrieve_i2t_kb as retrieval
from script.shared import score_i2t_kb as scoring


class Scorer:
    model_id = 'test-itm'
    revision = 'fixed'

    def score(self, image_path, captions):
        return [(int(image_path.stem) % 10) / 10 + int(caption.split('-')[1]) / 100
                for caption in captions]

    def score_batch(self, image_paths, captions_per_image):
        return [self.score(path, captions) for path, captions in zip(image_paths, captions_per_image)]


def state_for(rank, count):
    return SimpleNamespace(
        process_index=rank, num_processes=count, is_main_process=rank == 0, device='cpu',
        wait_for_everyone=lambda: None, main_process_first=nullcontext,
    )


class RetrievalStagesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.kb = pd.DataFrame([
            dict(imgid=imgid, caption=f'caption-{i}', tokens=[f'caption-{i}'],
                 objects=['dog'], attributes=[{'object': 'dog', 'attribute': 'red'}])
            for i, imgid in enumerate([1, 1, 2, 3, 4])
        ])
        self.kb_schema = pa.Table.from_pandas(self.kb, preserve_index=False).schema
        self.vectors = np.array([[1, 0, 0], [.9, .1, 0], [0, 1, 0],
                                 [0, 0, 1], [0, .2, .8]], dtype=np.float32)
        faiss.normalize_L2(self.vectors)
        self.index = faiss.IndexFlatIP(3)
        self.index.add(self.vectors)
        self.query_ids = [99, 2, 1, 88, 77]
        self.paths = {imgid: self.root / f'{imgid}.jpg' for imgid in self.query_ids}
        self.source = self.root / 'images.h5'
        with h5py.File(self.source, 'w') as stored:
            stored['imgids'] = self.query_ids
            stored['features'] = self.vectors
            stored.attrs['model_id'] = retrieval.RETRIEVAL_OPENCLIP_MODEL_ID
        self.candidates = self.root / 'candidates.parquet'
        with patch.object(retrieval, 'I2T_RETRIEVAL_TOP_K', 2):
            retrieval.retrieve_captions(self.source, self.candidates, self.index, self.kb,
                                        self.kb_schema, batch_size=2)

    def test_stages_preserve_queries_captions_scores_and_semantic_fields(self):
        original_candidates = self.candidates.read_bytes()
        candidates = pq.read_table(self.candidates)
        self.assertNotIn('itm_scores', candidates.column_names)
        self.assertNotIn(b'blip_itm_model_id', candidates.schema.metadata)
        distances, indices = self.index.search(self.vectors, len(self.kb))
        for row, imgid, ids, scores in zip(candidates.to_pylist(), self.query_ids, indices, distances):
            selected = [(int(i), float(score)) for i, score in zip(ids, scores)
                        if self.kb.iloc[int(i)]['imgid'] != imgid][:2]
            self.assertEqual(row['captions'], [self.kb.iloc[i]['caption'] for i, _ in selected])
            np.testing.assert_allclose(row['clip_scores'], [s for _, s in selected], atol=1e-6)

        for processes in (1, 3, 7):
            with self.subTest(processes=processes):
                target = self.root / f'result-{processes}.parquet'
                # Main runs last here so that all simulated worker shards exist before merge.
                for rank in reversed(range(processes)):
                    scoring.score_retrieved_captions(
                        self.candidates, target, self.paths, Scorer(), state_for(rank, processes),
                        itm_image_batch_size=2,
                    )
                table = pq.read_table(target)
                self.assertEqual(table['imgid'].to_pylist(), self.query_ids)
                self.assertEqual(table.schema.metadata[b'blip_itm_model_id'], b'test-itm')
                for expected, row in zip(candidates.to_pylist(), table.to_pylist()):
                    self.assertEqual({k: v for k, v in row.items() if k != 'itm_scores'}, expected)
                    np.testing.assert_allclose(
                        row['itm_scores'], Scorer().score(self.paths[row['imgid']], row['captions']),
                        atol=1e-6,
                    )
                self.assertFalse(list(self.root.glob(f'result-{processes}*.tmp.parquet')))
        self.assertEqual(self.candidates.read_bytes(), original_candidates)

    def test_shard_is_written_only_after_all_its_images_are_scored(self):
        target = self.root / 'buffered.parquet'
        shard = target.with_suffix('.rank1.tmp.parquet')
        scored_ids = []
        scorer = Scorer()
        original_score = scorer.score

        def score(image_path, captions):
            self.assertFalse(shard.exists())
            scored_ids.append(int(image_path.stem))
            return original_score(image_path, captions)

        with patch.object(scorer, 'score', side_effect=score):
            scoring.score_retrieved_captions(
                self.candidates, target, self.paths, scorer, state_for(1, 2),
                itm_image_batch_size=1,
            )
        expected_ids = self.query_ids[len(self.query_ids) // 2:]
        self.assertEqual(scored_ids, expected_ids)
        with pq.ParquetFile(shard) as parquet:
            self.assertEqual(parquet.metadata.num_row_groups, 1)
            self.assertEqual(parquet.read()['imgid'].to_pylist(), expected_ids)

    def test_bad_itm_scores_preserve_candidates_and_previous_final_file(self):
        original_candidates = self.candidates.read_bytes()
        target = self.root / 'result.parquet'
        target.write_bytes(b'previous result')
        scorer = Scorer()
        with patch.object(scorer, 'score_batch', return_value=[[], []]):
            with self.assertRaisesRegex(ValueError, 'ITM score count'):
                scoring.score_retrieved_captions(
                    self.candidates, target, self.paths, scorer, state_for(0, 1),
                    itm_image_batch_size=2,
                )
        self.assertEqual(target.read_bytes(), b'previous result')
        self.assertEqual(self.candidates.read_bytes(), original_candidates)
        self.assertFalse(target.with_suffix('.rank0.tmp.parquet').exists())

    def test_scoring_can_rerun_without_faiss_inputs_and_preserves_candidates(self):
        index_path = self.root / 'kb.faiss'
        faiss.write_index(self.index, str(index_path))
        metadata_path = self.root / 'kb.parquet'
        self.kb.to_parquet(metadata_path, index=False)
        split_path = self.root / 'split.parquet'
        pd.DataFrame([dict(imgid=i, filepath='', filename=f'{i}.jpg')
                      for i in self.query_ids]).to_parquet(split_path, index=False)
        targets = {split: self.root / f'{split}_i2t.parquet' for split in ('TRAIN', 'VAL', 'TEST')}

        replacements = dict(I2T_KB_FAISS_INDEX_PATH=index_path, I2T_KB_METADATA_PATH=metadata_path,
                            I2T_RETRIEVAL_TOP_K=2)
        score_replacements = dict(IMAGES_DIR=self.root)
        for split, target in targets.items():
            replacements[f'{split}_IMAGE_EMBEDDINGS_PATH'] = self.source
            replacements[f'{split}_I2T_PATH'] = target
            score_replacements[f'{split}_DF_PATH'] = split_path
            score_replacements[f'{split}_I2T_PATH'] = target
        with patch.multiple(retrieval, **replacements):
            retrieval.main()

        candidates = {target: target.with_suffix('.clip.parquet').read_bytes()
                      for target in targets.values()}
        for target in targets.values():
            self.assertFalse(target.exists())
        # Scoring only needs candidates, split image paths and original images.
        index_path.unlink()
        metadata_path.unlink()
        self.source.unlink()
        with (
            patch.multiple(scoring, **score_replacements),
            patch.object(scoring, 'PartialState', return_value=state_for(0, 1)),
            patch.object(scoring, 'BlipITMScorer', side_effect=lambda *args: Scorer()),
        ):
            previous = None
            for batch_size in (1, 3):
                with patch.object(scoring, 'BLIP_ITM_IMAGE_BATCH_SIZE', batch_size):
                    scoring.main()
                current = [pq.read_table(target).to_pylist() for target in targets.values()]
                if previous is not None:
                    self.assertEqual(current, previous)
                previous = current
                for target in targets.values():
                    self.assertEqual(pq.read_table(target)['imgid'].to_pylist(), self.query_ids)
                    self.assertEqual(target.with_suffix('.clip.parquet').read_bytes(), candidates[target])


if __name__ == '__main__':
    unittest.main()
