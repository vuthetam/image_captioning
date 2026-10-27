"""Check caption row order across Parquet batches and contiguous FAISS shards."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from script.shared import build_kb_i2t_index as builder


class CaptionEncoder:
    visual = SimpleNamespace(output_dim=3)

    def encode_text(self, tokens):
        values = tokens[:, 0].float() + 1
        return torch.stack((torch.ones_like(values), values, values.square()), dim=1)


def tokenize(captions):
    return torch.tensor([[int(caption)] for caption in captions], dtype=torch.long)


class CaptionIndexTests(unittest.TestCase):
    def test_shards_preserve_every_row_across_groups_batches_and_empty_ranges(self):
        for num_rows, num_shards in ((17, 3), (2, 4)):
            with self.subTest(rows=num_rows, shards=num_shards), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                metadata = root / 'metadata.parquet'
                pq.write_table(pa.table({'caption': [str(i) for i in range(num_rows)]}),
                               metadata, row_group_size=4)
                original_metadata = metadata.read_bytes()
                target = root / 'captions.faiss'
                with (
                    patch.object(builder, 'KB_I2T_INDEX_BATCH_SIZE', 3),
                    patch.object(builder, 'load_openclip_retrieval_components',
                                 return_value=(CaptionEncoder(), None, tokenize)),
                ):
                    single_path = builder.build_caption_index(metadata, root / 'single.faiss', 'cpu')
                    # Completion order differs from metadata order.
                    for rank in reversed(range(num_shards)):
                        shard_path = builder.build_caption_index(
                            metadata, target, 'cpu', shard_index=rank, num_shards=num_shards,
                        )
                        shard = faiss.read_index(str(shard_path))
                        start = num_rows * rank // num_shards
                        stop = num_rows * (rank + 1) // num_shards
                        self.assertEqual(shard.ntotal, stop - start)
                    merged_path = builder.merge_index_shards(target, num_shards, num_rows)

                single = faiss.read_index(str(single_path))
                merged = faiss.read_index(str(merged_path))
                expected = CaptionEncoder().encode_text(tokenize([str(i) for i in range(num_rows)]))
                expected = torch.nn.functional.normalize(expected, dim=-1).numpy()
                self.assertEqual(merged.ntotal, num_rows)
                np.testing.assert_allclose(merged.reconstruct_n(0, num_rows), expected, atol=1e-7)
                np.testing.assert_array_equal(merged.reconstruct_n(0, num_rows),
                                              single.reconstruct_n(0, num_rows))
                self.assertEqual(metadata.read_bytes(), original_metadata)
                self.assertFalse(list(root.glob('*.rank*.tmp.faiss')))

    def test_incomplete_shard_does_not_replace_existing_index(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'captions.faiss'
            target.write_bytes(b'previous index')
            for rank, count in enumerate((2, 1)):
                shard = faiss.IndexFlatIP(3)
                shard.add(np.ones((count, 3), dtype=np.float32))
                faiss.write_index(shard, str(target.with_suffix(f'.rank{rank}.tmp.faiss')))
            with self.assertRaisesRegex(ValueError, 'row count'):
                builder.merge_index_shards(target, 2, expected_rows=4)
            self.assertEqual(target.read_bytes(), b'previous index')
            self.assertEqual(len(list(Path(folder).glob('*.rank*.tmp.faiss'))), 2)

    def test_mismatched_batch_is_rejected_before_writing_index(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata = Path(folder) / 'metadata.parquet'
            target = Path(folder) / 'captions.faiss'
            pq.write_table(pa.table({'caption': ['0', '1']}), metadata)
            encoder = CaptionEncoder()
            with (
                patch.object(builder, 'load_openclip_retrieval_components',
                             return_value=(encoder, None, tokenize)),
                patch.object(encoder, 'encode_text', return_value=torch.ones(1, 3)),
            ):
                with self.assertRaisesRegex(ValueError, 'embedding shape'):
                    builder.build_caption_index(metadata, target, 'cpu')
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.tmp.faiss').exists())


if __name__ == '__main__':
    unittest.main()
