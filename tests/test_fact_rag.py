"""Offline fact selection, lazy cache, fusion and one-batch script integration."""
import copy
import importlib
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from accelerate import Accelerator
import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection

from src.fact_rag.dataset import load_rag_fact_lookup, RAGFeatureDataset, collate_fact_batch
from src.fact_rag.encoder import CachedCLIPFactEncoder
from src.fact_rag.models import FactRAGCaptioner
from src.shared.vocabulary import Vocabulary
from src.shared.inference import beam_search
from src.shared.utils import setup_reproducibility


def context_rows():
    return [
        dict(imgid=7, captions=["first", "best", "tied"], itm_scores=[.1, .9, .9],
             object_texts=[["outside"], ["  dog ", "red dog", "Dog"], ["dog", "tree"]],
             attribute_texts=[[], ["red   dog"], []],
             triplet_texts=[[], ["dog on grass"], ["dog on grass"]]),
        dict(imgid=8, captions=[], itm_scores=[], object_texts=[], attribute_texts=[], triplet_texts=[]),
    ]


def write_context(path, rows):
    strings = pa.list_(pa.string())
    schema = pa.schema([
        ("imgid", pa.int64()), ("captions", strings), ("itm_scores", pa.list_(pa.float32())),
        *[(column, pa.list_(strings)) for column in ("object_texts", "attribute_texts", "triplet_texts")],
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


class TinyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        ids = [[0, *[3 + sum(map(ord, word)) % 25 for word in text.split()], 2] for text in texts]
        width = max(map(len, ids))
        return {
            "input_ids": torch.tensor([row + [1] * (width - len(row)) for row in ids]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids]),
        }


def tiny_clip():
    return CLIPTextModelWithProjection(CLIPTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, projection_dim=8,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
        bos_token_id=0, eos_token_id=2, pad_token_id=1,
    ))


def captioner(vocab_size=12, pad_idx=0, fusion_layers=1):
    return FactRAGCaptioner(
        vocab_size=vocab_size, d_model=16, nheads=2, nlayers=1, dropout=0,
        max_length=6, pad_idx=pad_idx, visual_feature_dim=6, fact_feature_dim=8,
        fusion_layers=fusion_layers,
    )


def close_loader_stores(accelerator):
    # Accelerate retains prepared loaders; close their HDF5 handles before Windows cleanup.
    for loader in accelerator._dataloaders:
        store = getattr(loader.dataset, "feature_store", None)
        if store is not None and store._h5_file is not None:
            store._h5_file.close()
            store._h5_file = None
    accelerator.free_memory()


class FactSelectionTests(unittest.TestCase):
    def test_sort_select_then_deduplicate_all_three_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.parquet"
            write_context(path, context_rows())
            before = path.read_bytes()
            self.assertEqual(load_rag_fact_lookup(path, 2, "itm")[7],
                             ["dog", "red dog", "Dog", "dog on grass", "tree"])
            self.assertEqual(load_rag_fact_lookup(path, 1, "clip")[7], ["outside"])
            self.assertEqual(load_rag_fact_lookup(path, 2, "itm")[8], [])
            self.assertEqual(path.read_bytes(), before)

    def test_missing_context_schema_alignment_and_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.parquet"
            for field, value in (("itm_scores", [.5]), ("itm_scores", [.1, float("nan"), .2]),
                                 ("object_texts", [[]]), ("triplet_texts", [None, [], []])):
                with self.subTest(field=field, value=value):
                    rows = context_rows()
                    rows[0][field] = value
                    write_context(path, rows)
                    with self.assertRaises(ValueError):
                        load_rag_fact_lookup(path, 2, "itm")
            write_context(path, context_rows() * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_rag_fact_lookup(path, 2)
            write_context(path, context_rows())
            table = pq.read_table(path).drop(["attribute_texts"])
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "Missing fact context columns"):
                load_rag_fact_lookup(path, 2)
            write_context(path, context_rows())
            features = Path(directory) / "features.h5"
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [99, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            with self.assertRaisesRegex(ValueError, "Missing fact contexts"):
                RAGFeatureDataset(pd.DataFrame({"imgid": [99]}), features, path, 2)
            with self.assertRaisesRegex(ValueError, "No facts selected for imgid=8"):
                RAGFeatureDataset(pd.DataFrame({"imgid": [8]}), features, path, 2)


class FactCacheTests(unittest.TestCase):
    def test_cache_accepts_tensor_and_tuple_clip_outputs(self):
        for output_format in ("tensor", "tuple"):
            with self.subTest(output_format=output_format):
                model, tokenizer = tiny_clip(), TinyTokenizer()
                with patch("src.fact_rag.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                     patch("src.fact_rag.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=model):
                    encoder = CachedCLIPFactEncoder("test/clip", "cpu")
                output = model(**TinyTokenizer()(["dog", "tree"]))
                expected = output.text_embeds.detach()
                compatible_output = expected if output_format == "tensor" else (
                    output.last_hidden_state, expected,
                )
                with patch.object(model, "forward", return_value=compatible_output) as forward:
                    features, mask = encoder.encode([["dog", "tree"]])
                    cached_features, _ = encoder.encode([["dog", "tree"]])
                self.assertEqual(forward.call_count, 1)
                torch.testing.assert_close(features[0], expected)
                torch.testing.assert_close(cached_features, features)
                self.assertEqual(mask.tolist(), [[True, True]])
                self.assertTrue(all(value.device.type == "cpu" for value in encoder.cache.values()))

    def test_real_clip_projected_output_is_cached_on_cpu_without_normalization(self):
        torch.manual_seed(7)
        model, tokenizer = tiny_clip(), TinyTokenizer()
        with patch("src.fact_rag.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("src.fact_rag.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=model):
            encoder = CachedCLIPFactEncoder("test/clip", "cpu", batch_size=2)
        expected = model(**TinyTokenizer()(["dog", "red dog"])).text_embeds.detach()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            features, mask = encoder.encode([[" dog ", "red dog"], ["dog"], ["dog"]])
        torch.testing.assert_close(features[0], expected)
        self.assertEqual(mask.tolist(), [[True, True], [True, False], [True, False]])
        self.assertEqual(tokenizer.calls, [["dog", "red dog"]])
        encoder.encode([["dog"], ["Dog"]])
        self.assertEqual(tokenizer.calls, [["dog", "red dog"], ["Dog"]])
        for empty_batch in ([], [[]], [["dog"], []]):
            with self.subTest(empty_batch=empty_batch):
                with self.assertRaisesRegex(ValueError, "at least one fact"):
                    encoder.encode(empty_batch)
        self.assertEqual(len(tokenizer.calls), 2)
        self.assertEqual(len(encoder.cache), 3)
        self.assertTrue(all(value.device.type == "cpu" and not value.requires_grad
                            and value.dtype == torch.float32 for value in encoder.cache.values()))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(torch.allclose(features[0].norm(dim=-1), torch.ones(2)))
        with self.assertRaisesRegex(ValueError, "context length"):
            encoder.encode([["word " * 20]])


class FusionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_one_batch_with_frozen_clip_and_mixed_precision(self):
        setup_reproducibility(42)
        model, tokenizer = tiny_clip(), TinyTokenizer()
        with patch("src.fact_rag.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("src.fact_rag.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=model):
            encoder = CachedCLIPFactEncoder("test/clip", "cuda", batch_size=2)
        facts, valid = encoder.encode([["dog", "tree"], ["tree"], ["dog"]])
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary([["dog", "tree"]])
        network = captioner(len(vocab), vocab.pad_idx()).cuda()
        visual = torch.randn(3, 4, 6, device="cuda")
        ids, mask = vocab.encode_from_tokens(["dog"], 6)
        ids = torch.tensor([ids] * 3, device="cuda")
        mask = torch.tensor([mask] * 3, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = network(visual, ids, mask, facts, valid)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(),
                                                     ignore_index=vocab.pad_idx())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in network.parameters()))
        network.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            memory = network.encode_memory(visual, facts, valid)
            output = beam_search(network.decoder, memory, vocab, beam_size=2, max_length=6)
        self.assertEqual(tuple(output.shape), (3, 6))
        self.assertTrue(all(value.device.type == "cpu" for value in encoder.cache.values()))

    def test_patch_memory_ignores_cls_and_padding(self):
        torch.manual_seed(1)
        model = captioner(fusion_layers=2).eval()
        visual, facts = torch.randn(3, 4, 6), torch.randn(3, 2, 8)
        valid = torch.tensor([[True, False], [True, False], [True, True]])
        with torch.no_grad():
            memory = model.encode_memory(visual, facts, valid)
            self.assertEqual(tuple(memory.shape), (3, 3, 16))
            memory_with_cls = model.encode_memory(visual, facts, valid, include_cls_token=True)
            self.assertEqual(tuple(memory_with_cls.shape), (3, 4, 16))
            other_visual, other_facts = visual.clone(), facts.clone()
            other_visual[:, 0] += 100
            other_facts[~valid] += 100
            torch.testing.assert_close(model.encode_memory(other_visual, other_facts, valid), memory)
            other_facts[0, 0] *= -1
            self.assertFalse(torch.allclose(model.encode_memory(visual, other_facts, valid)[0], memory[0]))

    def test_rejects_zero_fact_slots_and_samples_without_valid_facts(self):
        model = captioner()
        visual = torch.randn(2, 4, 6)
        for valid in (torch.zeros(2, 0, dtype=torch.bool),
                      torch.zeros(2, 1, dtype=torch.bool),
                      torch.tensor([[True, False], [False, False]])):
            with self.subTest(mask_shape=valid.shape, mask=valid.tolist()):
                facts = torch.zeros(*valid.shape, 8)
                with self.assertRaisesRegex(ValueError, "at least one"):
                    model.encode_memory(visual, facts, valid)


class FactScriptTests(unittest.TestCase):
    def test_train_val_and_generate_each_use_one_batch_with_zero_workers(self):
        train = importlib.import_module("script.fact_rag.train_rag")
        generate = importlib.import_module("script.fact_rag.generate_rag_captions")
        accelerator = Accelerator(cpu=True)
        text_model, tokenizer = tiny_clip(), TinyTokenizer()
        initial_text = copy.deepcopy(text_model.state_dict())
        encoders = []
        batch_widths = []

        def collate_tuples(samples):
            self.assertTrue(all(isinstance(sample, tuple) for sample in samples))
            batch = collate_fact_batch(samples)
            self.assertIsInstance(batch, tuple)
            fact_index = 1 if len(batch) == 3 else 3
            self.assertEqual(batch[fact_index], [sample[fact_index] for sample in samples])
            batch_widths.append(len(batch))
            return batch

        def make_encoder(*args):
            encoder = CachedCLIPFactEncoder(*args)
            encoders.append(encoder)
            return encoder

        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            cleanup.callback(close_loader_stores, accelerator)
            root = Path(directory)
            context, features = root / "contexts.parquet", root / "features.h5"
            rows = context_rows()
            rows[1] = dict(imgid=8, captions=["dog"], itm_scores=[.5],
                           object_texts=[["dog"]], attribute_texts=[[]], triplet_texts=[[]])
            write_context(context, rows)
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            frame = pd.DataFrame({"imgid": [7, 7, 8], "tokens": [["dog"], ["tree"], ["dog"]]})
            split = root / "split.parquet"
            frame.to_parquet(split, index=False)
            test_split = root / "test.parquet"
            pd.DataFrame({"imgid": [7, 8]}).to_parquet(test_split, index=False)
            vocab = Vocabulary(freq_threshold=1)
            vocab.build_vocabulary(frame["tokens"])
            vocab_path = vocab.save(root / "vocab.json")
            common = dict(
                BATCH_SIZE=3, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=1,
                DROPOUT=0., MAX_LENGTH=6, FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2,
                VISUAL_ENCODER_MODEL="test/clip", TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=2,
                VOCAB_PATH=vocab_path, BEST_CHECKPOINT_PATH=root / "best.pth", RUN_MODE="fact_test",
            )
            with patch("src.fact_rag.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch("src.fact_rag.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=text_model), \
                 patch.object(train, "Accelerator", return_value=accelerator), \
                 patch.object(generate, "Accelerator", return_value=accelerator), \
                 patch.object(train, "CachedCLIPFactEncoder", side_effect=make_encoder), \
                 patch.object(generate, "CachedCLIPFactEncoder", side_effect=make_encoder), \
                 patch.object(train, "collate_fact_batch", side_effect=collate_tuples), \
                 patch.object(generate, "collate_fact_batch", side_effect=collate_tuples), \
                 patch("os._exit"):
                with patch.multiple(train, **common, NUM_EPOCHS=1,
                                    TRAIN_DF_PATH=split, VAL_DF_PATH=split,
                                    TRAIN_I2T_PATH=context, VAL_I2T_PATH=context,
                                    TRAIN_VISUAL_FEATURES_PATH=features, VAL_VISUAL_FEATURES_PATH=features,
                                    LAST_CHECKPOINT_PATH=root / "last.pth"):
                    train.main()
                checkpoint = torch.load(common["BEST_CHECKPOINT_PATH"], weights_only=True)
                self.assertEqual(checkpoint["epoch"], 1)
                self.assertEqual(checkpoint["global_step"], 1)
                self.assertTrue(torch.isfinite(torch.tensor(checkpoint["train_loss"])))
                self.assertTrue(all(not key.startswith("fact_encoder") for key in checkpoint["model_state_dict"]))
                self.assertEqual(len(encoders[0].cache), 5)
                calls_after_train = len(tokenizer.calls)
                with patch.multiple(generate, **common, BEAM_SIZE=2,
                                    TEST_DF_PATH=test_split, TEST_I2T_PATH=context,
                                    TEST_VISUAL_FEATURES_PATH=features, PREDICTIONS_PATH=root / "predictions.json"):
                    generate.main()
                import json
                predictions = json.loads((root / "predictions.json").read_text(encoding="utf-8"))
                self.assertEqual([item["imgid"] for item in predictions], [7, 8])
                self.assertTrue(all(isinstance(item["caption"], str) for item in predictions))
                self.assertEqual(len(encoders[1].cache), 5)
                self.assertGreater(len(tokenizer.calls), calls_after_train)
                self.assertEqual(batch_widths, [4, 4, 3])
            for key, value in text_model.state_dict().items():
                torch.testing.assert_close(value, initial_text[key], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in text_model.parameters()))


if __name__ == "__main__":
    unittest.main()
