"""Offline alias normalization with real SimpleNLG and small Parquet fixtures."""

from copy import deepcopy
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
    from script.shared.normalize_kb_entities import TEXT_COLUMNS, main, normalize_metadata
    from src.shared.knowledge_graph import GraphLinearizer


@unittest.skipUnless(KB_DEPS, "Install requirements-kb.txt to run alias normalization checks")
class NormalizeKBEntitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linearizer = GraphLinearizer()
        cls.aliases = {"motor bike": "motorcycle", "fridge": "refrigerator", "grey": "gray"}

    def facts(self):
        return {
            "objects": ["motor bike", "fridge", "small fridge", "motor bike"],
            "attributes": [
                {"object": "motor bike", "attribute": "grey"},
                {"object": "fridge", "attribute": "made of metal"},
                {"object": "motor bike", "attribute": "pv:not park"},
            ],
            "triplets": [
                {"subject": "person", "predicate": "ride", "object": "motor bike", "voice": "active"},
                {"subject": "motor bike", "predicate": "not park near", "object": "fridge", "voice": "passive"},
                {"subject": "motor bike", "predicate": "next to", "object": "small fridge", "voice": "unspecified"},
            ],
        }

    def metadata_row(self):
        facts = self.facts()
        return {
            "imgid": 7, "caption": "Original motor bike caption.",
            "tokens": ["original", "motor", "bike", "caption"],
            **facts, **self.linearizer.linearize_facts(**facts),
        }

    def test_aliases_apply_to_whole_entities_before_realization(self):
        facts = self.facts()
        original = deepcopy(facts)
        texts = self.linearizer.linearize_facts(**facts, entity_aliases=self.aliases)
        self.assertEqual(texts["object_texts"],
                         ["motorcycle", "refrigerator", "small fridge", "motorcycle"])
        self.assertEqual(texts["attribute_texts"],
                         ["grey motorcycle", "refrigerator made of metal", "motorcycle not parked"])
        self.assertEqual(texts["triplet_texts"], [
            "person riding motorcycle", "motorcycle not parked near refrigerator",
            "motorcycle next to small fridge",
        ])
        self.assertEqual(facts, original)

    def test_metadata_keeps_every_other_column_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.parquet", Path(folder) / "output.parquet"
            table = pa.Table.from_pylist([self.metadata_row()]).replace_schema_metadata({b"source": b"FACTUAL"})
            pq.write_table(table, source)
            original_table = pq.read_table(source)
            source_bytes = source.read_bytes()
            normalize_metadata(source, output, self.linearizer, self.aliases, batch_size=1)
            result = pq.read_table(output)
            self.assertTrue(result.schema.equals(original_table.schema, check_metadata=True))
            for column in original_table.column_names:
                if column not in TEXT_COLUMNS:
                    self.assertEqual(result[column].to_pylist(), original_table[column].to_pylist())
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(result["object_texts"][0].as_py()[0], "motorcycle")
            with self.assertRaises(FileExistsError):
                normalize_metadata(source, output, self.linearizer, self.aliases)

    def test_cli_processes_kb_without_context_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata = folder / "metadata.parquet"
            pq.write_table(pa.Table.from_pylist([self.metadata_row()]), metadata)
            original_bytes = metadata.read_bytes()
            with patch("sys.argv", ["normalize_kb_entities.py", "--metadata", str(metadata),
                                     "--batch-size", "1"]):
                main()
            self.assertTrue((folder / "metadata_aliases.parquet").is_file())
            self.assertEqual(metadata.read_bytes(), original_bytes)
            self.assertEqual({path.name for path in folder.iterdir()},
                             {"metadata.parquet", "metadata_aliases.parquet"})


if __name__ == "__main__":
    unittest.main()
