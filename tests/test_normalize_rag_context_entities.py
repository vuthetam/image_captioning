"""Check the context-only entrypoint with small offline Parquet fixtures."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import simplenlg
    import factual_scene_graph
    KB_DEPS = True
except ImportError:
    KB_DEPS = False

if KB_DEPS:
    from script.shared.normalize_rag_context_entities import TEXT_COLUMNS, main, normalize_context_file
    from src.shared.knowledge_graph import GraphLinearizer


@unittest.skipUnless(KB_DEPS, "Install requirements-kb.txt to run context normalization checks")
class NormalizeRAGContextEntitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linearizer = GraphLinearizer()
        cls.aliases = {"motor bike": "motorcycle"}

    def context_table(self):
        context = {
            "imgid": 8, "retrieval_imgids": [7, 9],
            "captions": ["red motor bike", "person riding motor bike"],
            "tokens": [["red", "motor", "bike"], ["person", "riding", "motor", "bike"]],
            "objects": [["motor bike"], ["person", "motor bike"]],
            "attributes": [[{"object": "motor bike", "attribute": "red"}], []],
            "triplets": [[], [{"subject": "person", "predicate": "ride", "object": "motor bike", "voice": "active"}]],
            "object_texts": [["motor bike"], ["person", "motor bike"]],
            "attribute_texts": [["red motor bike"], []],
            "triplet_texts": [[], ["person riding motor bike"]],
            "clip_scores": [0.5, 0.4], "itm_scores": [0.2, 0.9],
        }
        return pa.Table.from_pylist([context]).replace_schema_metadata(
            {b"caption_order": b"CLIP descending", b"blip_itm_model_id": b"unchanged-model"},
        )

    def test_context_keeps_caption_groups_and_every_other_column(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.parquet", Path(folder) / "output.parquet"
            pq.write_table(self.context_table(), source)
            before = pq.read_table(source)
            original_bytes = source.read_bytes()
            normalize_context_file(source, output, self.linearizer, self.aliases, batch_size=1)
            after = pq.read_table(output)
            self.assertTrue(after.schema.equals(before.schema, check_metadata=True))
            for column in before.column_names:
                if column not in TEXT_COLUMNS:
                    self.assertTrue(after[column].equals(before[column]), column)
            row = after.to_pylist()[0]
            self.assertEqual(row["object_texts"], [["motorcycle"], ["person", "motorcycle"]])
            self.assertEqual(row["attribute_texts"], [["red motorcycle"], []])
            self.assertEqual(row["triplet_texts"], [[], ["person riding motorcycle"]])
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_cli_accepts_arbitrary_filename_and_optional_output(self):
        for explicit_output in (False, True):
            with self.subTest(explicit_output=explicit_output), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                source = folder / "custom_caption_candidates.v2.parquet"
                pq.write_table(self.context_table(), source)
                original_bytes = source.read_bytes()
                arguments = ["normalize_rag_context_entities.py", "--context", str(source), "--batch-size", "1"]
                if explicit_output:
                    output = folder / "results" / "normalized.parquet"
                    arguments.extend(["--output", str(output)])
                else:
                    output = folder / "custom_caption_candidates.v2_aliases.parquet"
                with patch("sys.argv", arguments):
                    main()
                self.assertEqual(pq.read_table(output)["object_texts"][0].as_py(),
                                 [["motorcycle"], ["person", "motorcycle"]])
                self.assertEqual(source.read_bytes(), original_bytes)
                self.assertEqual({path for path in folder.rglob("*.parquet")}, {source, output})

    def test_empty_caption_group_and_misalignment(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.parquet", Path(folder) / "output.parquet"
            table = self.context_table()
            empty = {column: ([] if isinstance(value, list) else 10)
                     for column, value in table.to_pylist()[0].items()}
            pq.write_table(pa.Table.from_pylist([empty], schema=table.schema), source)
            normalize_context_file(source, output, self.linearizer, self.aliases, batch_size=1)
            for column in TEXT_COLUMNS:
                self.assertEqual(pq.read_table(output)[column][0].as_py(), [])

            bad = table.set_column(table.schema.get_field_index("attributes"), "attributes",
                                   pa.array([[]], type=table["attributes"].type))
            pq.write_table(bad, source)
            output = Path(folder) / "bad_output.parquet"
            with self.assertRaisesRegex(ValueError, "not aligned"):
                normalize_context_file(source, output, self.linearizer, self.aliases, batch_size=1)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".tmp.parquet").exists())


if __name__ == "__main__":
    unittest.main()
