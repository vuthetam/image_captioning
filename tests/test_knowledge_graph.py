"""Offline checks: real FACTUAL graph converter and SimpleNLG, no model download."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

try:
    import simplenlg
    import factual_scene_graph
    KB_DEPS = True
except ImportError:
    KB_DEPS = False

if KB_DEPS:
    from src.shared.knowledge_graph import CheckedParser, GraphLinearizer
    from script.shared.build_i2t_kb import build_metadata, metadata_schema, resolve_model_revision


def entity(head, quantity='', attributes=()):
    return dict(head=head, quantity=quantity, attributes=set(attributes))


def graph(subject='man', predicate='v:ride', obj='bicycle', quantity='', attrs=('red',)):
    return {'entities': [entity(subject, quantity), entity(obj, quantity, attrs)],
            'relations': [{'subject': 0, 'relation': predicate, 'object': 1}]}


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0
    text = '( man , v:ride , bicycle ) , ( bicycle , is , red )'
    def __call__(self, prompts, **kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids': torch.ones(len(prompts), 9, dtype=torch.long),
                'attention_mask': torch.ones(len(prompts), 9, dtype=torch.long)}
    def batch_decode(self, generated, **kwargs):
        return [self.text] * len(generated)


class Model:
    eos = 1
    def to(self, device):
        return self
    def eval(self):
        return self
    def generate(self, input_ids, **kwargs):
        return torch.tensor([[0, 7, self.eos]] * len(input_ids))


@unittest.skipUnless(KB_DEPS, 'Install requirements-kb.txt to run KB integration checks')
class KnowledgeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linearizer = GraphLinearizer()

    def test_active_attribute_and_empty_collections(self):
        row = self.linearizer.normalize(graph())
        self.assertEqual(set(row), {'objects', 'attributes', 'triplets',
                                    'object_texts', 'attribute_texts', 'triplet_texts'})
        self.assertEqual(row['objects'], ['man', 'bicycle'])
        self.assertEqual(row['attribute_texts'], ['red bicycle'])
        self.assertEqual(row['triplet_texts'], ['man riding bicycle'])
        bare = self.linearizer.normalize({'entities': [entity('man')], 'relations': []})
        self.assertEqual(bare['triplets'], [])
        schema = metadata_schema().remove(0).remove(0).remove(0).remove(0).remove(0)
        self.assertEqual(set(schema.names), set(row))
        pa.Table.from_pylist([row, bare], schema=schema)

    def test_passive_keeps_graph_subject(self):
        row = self.linearizer.normalize(graph('car', 'pv:park on', 'street', attrs=()))
        self.assertEqual(row['triplet_texts'], ['car parked on street'])
        self.assertEqual(row['triplets'][0]['voice'], 'passive')

    def test_negation_is_preserved_without_do_support(self):
        row = self.linearizer.normalize(graph(predicate='v:not ride'))
        self.assertEqual(row['triplet_texts'], ['man not riding bicycle'])

    def test_prepositions_quantities_mass_and_compounds(self):
        self.assertEqual(self.linearizer.normalize(graph(predicate='next to'))['triplet_texts'],
                         ['man next to bicycle'])
        row = self.linearizer.normalize(graph('men', 'v:ride', 'bicycles', '2', ()))
        self.assertEqual(row['triplet_texts'], ['men riding bicycles'])
        row = self.linearizer.normalize(graph('men', 'v:stand on', 'grass', attrs=()))
        self.assertEqual(row['triplet_texts'], ['men standing on grass'])
        row = self.linearizer.normalize({'entities': [entity('traffic light')], 'relations': []})
        self.assertEqual(row['objects'], ['traffic light'])
        row = self.linearizer.normalize({'entities': [entity('apple')], 'relations': []})
        self.assertEqual(row['object_texts'], ['apple'])

    def test_noun_forms_are_preserved_without_articles(self):
        labels = ['people', 'glasses', 'jeans', 'men', 'apple', 'water', 'tennis players']
        source = {'entities': [entity(label) for label in labels], 'relations': []}
        row = self.linearizer.normalize(source)
        self.assertEqual(row['objects'], labels)
        self.assertEqual(row['object_texts'], labels)
        row = self.linearizer.normalize(graph('people', 'v:lie next to', 'glasses',
                                              attrs=('blue',)))
        self.assertEqual(row['attribute_texts'], ['blue glasses'])
        self.assertEqual(row['triplet_texts'], ['people lying next to glasses'])

    def test_states_and_passive_attributes_use_bare_labels(self):
        source = {'entities': [entity('tables', attributes=('made of wood',)),
                               entity('jeans', attributes=('pv:not wear',))],
                  'relations': []}
        row = self.linearizer.normalize(source)
        self.assertEqual(row['attribute_texts'], ['tables made of wood', 'jeans not worn'])
        row = self.linearizer.normalize(graph('glasses', 'pv:not pick up by', 'people',
                                              attrs=()))
        self.assertEqual(row['triplet_texts'], ['glasses not picked up by people'])

    def test_quantities_are_omitted_from_normalized_kb(self):
        source = graph('men', 'v:ride', 'bicycles', quantity='2')
        without_counts = graph('men', 'v:ride', 'bicycles')
        self.assertEqual(self.linearizer.normalize(source),
                         self.linearizer.normalize(without_counts))
        self.assertEqual(source['entities'][0]['quantity'], '2')
        source = {'entities': [entity('men', quantity='2')], 'relations': []}
        row = self.linearizer.normalize(source)
        self.assertEqual(row['objects'], ['men'])
        self.assertEqual(row['object_texts'], ['men'])
        self.assertEqual(row['attributes'], [])
        self.assertEqual(row['attribute_texts'], [])

    def test_entity_ids_are_removed_without_changing_input_graph(self):
        source = graph('man', 'next to', 'man:1', attrs=('old',))
        row = self.linearizer.normalize(source)
        self.assertNotIn(':1', row['triplet_texts'][0])
        self.assertEqual(source['entities'][1]['head'], 'man:1')
        self.assertEqual(source['relations'][0]['object'], 1)

    def test_invalid_references_rejected(self):
        broken = graph()
        broken['relations'][0]['subject'] = 4
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.linearizer.normalize(broken)

    def test_retrieval_carries_units_and_excludes_self(self):
        import faiss
        import h5py
        import numpy as np
        from script.shared.retrieve_i2t_kb import process_and_retrieve
        rows = []
        for imgid in [7, 8]:
            rows.append(dict(imgid=imgid, caption='caption',
                             tokens=['caption'], **self.linearizer.normalize(graph())))
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1., 0.], [0.9, 0.1]], dtype='float32'))
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'images.h5', Path(directory) / 'i2t.parquet'
            with h5py.File(source, 'w') as handle:
                handle['imgids'] = [7]
                handle['features'] = np.array([[1., 0.]], dtype='float32')
            process_and_retrieve(source, target, index, pd.DataFrame(rows))
            result = pq.read_table(target).to_pylist()[0]
            self.assertEqual(result['retrieval_imgids'], [8])
            self.assertEqual(result['triplet_texts'], [['man riding bicycle']])
            self.assertEqual(result['attributes'][0][0]['attribute'], 'red')
            # Existing KBs remain readable without the newly added columns.
            legacy = pd.DataFrame(rows)[['imgid', 'caption', 'tokens', 'objects']]
            legacy['relations'] = [['ride'], ['ride']]
            process_and_retrieve(source, target, index, legacy)
            result = pq.read_table(target).to_pylist()[0]
            self.assertEqual(result['retrieval_imgids'], [8])
            self.assertNotIn('triplet_texts', result)
            self.assertNotIn('relations', result)

    def make_parser(self, tokenizer=None, model=None):
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer or Tokenizer()), \
             patch('transformers.AutoModelForSeq2SeqLM.from_pretrained', return_value=model or Model()):
            return CheckedParser('test-id', 'cpu')

    def test_real_converter_and_return_text_false(self):
        parser = self.make_parser()
        result = parser.parse(['A man riding a red bicycle.'], return_text=False)
        self.assertEqual(self.linearizer.normalize(result[0])['attribute_texts'], ['red bicycle'])
        t = Tokenizer()
        t.text = '( bird ) , ( man , v:ride , bicycle )'
        parser = self.make_parser(tokenizer=t)
        output = parser.parse(['caption'], return_text=False)[0]
        self.assertIn('bird', [e['head'] for e in output['entities']])

    def test_decoding_settings_match_original_parser(self):
        from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

        # Support the original tokenizer API without downloading a model.
        class OriginalTokenizer(Tokenizer):
            def as_target_tokenizer(self):
                return nullcontext()
            def encode(self, text, **kwargs):
                return [7] * 9
            def __call__(self, prompts, **kwargs):
                return super().__call__(prompts, truncation=False)

        for beams in (1, 5):
            with self.subTest(beams=beams):
                original = SceneGraphParser.__new__(SceneGraphParser)
                original.device = 'cpu'
                original.tokenizer = OriginalTokenizer()
                original.model = Model()
                checked = self.make_parser()
                results = []
                for parser in (original, checked):
                    parser.model.generate = Mock(wraps=parser.model.generate)
                    parser.tokenizer.batch_decode = Mock(wraps=parser.tokenizer.batch_decode)
                    results.append(parser._parse_batch(['caption'], 128, 256, beams, False, 32))

                expected = original.model.generate.call_args
                actual = checked.model.generate.call_args
                self.assertTrue(torch.equal(expected.args[0], actual.args[0]))
                expected_options, actual_options = dict(expected.kwargs), dict(actual.kwargs)
                self.assertTrue(torch.equal(expected_options.pop('attention_mask'),
                                            actual_options.pop('attention_mask')))
                self.assertEqual(expected_options, actual_options)
                self.assertEqual(original.tokenizer.batch_decode.call_args.kwargs,
                                 checked.tokenizer.batch_decode.call_args.kwargs)
                self.assertEqual(results[0], results[1])

    def test_truncation_and_malformed_output_rejected(self):
        parser = self.make_parser()
        with self.assertRaisesRegex(ValueError, 'input'):
            parser.parse(['caption'], max_input_len=2)
        bad_model = Model()
        bad_model.eos = 7
        with self.assertRaisesRegex(ValueError, 'output limit'):
            self.make_parser(model=bad_model).parse(['caption'])
        t = Tokenizer()
        t.text = '( man , ride , bicycle'
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            self.make_parser(tokenizer=t).parse(['caption'])

    def test_parquet_roundtrip_and_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            train = folder / 'train.parquet'
            pd.DataFrame({'imgid': [7, 8], 'caption': ['first', 'second'],
                          'tokens': [['first'], ['second']]}).to_parquet(train)
            parser = SimpleNamespace(parse=lambda captions, **kw: [graph() for _ in captions])
            output = folder / 'kb.parquet'
            manifest = folder / 'kb_manifest.json'
            with patch('script.shared.build_i2t_kb.model_info', return_value=SimpleNamespace(sha='fixed')) as lookup:
                self.assertEqual(resolve_model_revision(manifest), 'fixed')
                lookup.assert_called_once()
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(set(saved), {'model', 'revision'})
            saved['train_sha256'] = 'unused'
            manifest.write_text(json.dumps(saved), encoding='utf-8')
            with patch('script.shared.build_i2t_kb.CheckedParser', return_value=parser) as parser_class:
                pending = build_metadata(train, output, 'fixed', 'cpu', batch_size=1)
                self.assertEqual(parser_class.call_count, 1)
            data = pq.read_table(pending).to_pylist()
            self.assertEqual(data[0]['tokens'], ['first'])
            self.assertEqual(data[0]['triplets'][0]['subject'], 'man')
            self.assertFalse(output.with_suffix('.parts').exists())
            with patch('script.shared.build_i2t_kb.model_info') as lookup:
                self.assertEqual(resolve_model_revision(manifest), 'fixed')
                lookup.assert_not_called()
            self.assertEqual(set(json.loads(manifest.read_text(encoding='utf-8'))), {'model', 'revision'})
            # Every build processes all captions again, without cached partitions.
            with patch('script.shared.build_i2t_kb.CheckedParser', return_value=parser) as parser_class:
                pending = build_metadata(train, output, 'fixed', 'cpu', batch_size=1)
                self.assertEqual(parser_class.call_count, 1)
            pending.replace(output)
            previous = output.read_bytes()
            failing = SimpleNamespace(parse=Mock(side_effect=[[graph()], ValueError('parse failed')]))
            with patch('script.shared.build_i2t_kb.CheckedParser', return_value=failing):
                with self.assertRaisesRegex(ValueError, 'parse failed'):
                    build_metadata(train, output, 'fixed', 'cpu', batch_size=1)
            self.assertFalse(pending.exists())
            self.assertEqual(output.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
