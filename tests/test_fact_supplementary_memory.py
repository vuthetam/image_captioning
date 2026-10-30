"""Supplementary memory, masking, gradients and one-batch script integration."""

import copy
import importlib
import json
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

from src.fact_memory_rag.models import FactMemoryRAGCaptioner
from src.fact_text_rag.dataset import RAGFeatureDataset, load_caption_context_lookup
from src.fact_text_rag.models import FactTextRAGCaptioner
from src.shared.facts.dataset import load_rag_fact_lookup
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.shared.inference import beam_search
from src.shared.vocabulary import Vocabulary
from test_fact_rag import TinyTokenizer, close_loader_stores, tiny_clip


def make_model(caption_context=False, vocab_size=12, pad_idx=0):
    kwargs = dict(
        vocab_size=vocab_size, d_model=16, nheads=2, nlayers=1, dropout=0.,
        max_length=6, pad_idx=pad_idx, visual_feature_dim=6, fact_feature_dim=8,
    )
    if caption_context:
        return FactTextRAGCaptioner(**kwargs, context_layers=1, max_context_length=8)
    return FactMemoryRAGCaptioner(**kwargs)


def inputs(caption_context=False, device="cpu"):
    values = dict(
        visual_inputs=torch.randn(2, 4, 6, device=device),
        fact_features=torch.randn(2, 3, 8, device=device),
        fact_attention_mask=torch.tensor([[True, False, False], [True, True, True]], device=device),
    )
    if caption_context:
        values["rag_input_ids"] = torch.tensor([[4, 5, 0, 0], [5, 6, 7, 0]], device=device)
        values["rag_attention_mask"] = values["rag_input_ids"].ne(0).long()
    return values


def write_context(path):
    rows = [
        dict(imgid=7, captions=["dog tree", "red dog"], tokens=[["dog", "tree"], ["red", "dog"]],
             itm_scores=[.1, .9], object_texts=[["outside"], [" dog ", "dog"]],
             attribute_texts=[[], ["red dog"]], triplet_texts=[[], ["dog on grass"]]),
        dict(imgid=8, captions=["tree"], tokens=[["tree"]], itm_scores=[.5],
             object_texts=[["tree"]], attribute_texts=[[]], triplet_texts=[[]]),
    ]
    strings = pa.list_(pa.string())
    schema = pa.schema([
        ("imgid", pa.int64()), ("captions", strings), ("tokens", pa.list_(strings)),
        ("itm_scores", pa.list_(pa.float32())),
        *[(name, pa.list_(strings)) for name in ("object_texts", "attribute_texts", "triplet_texts")],
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


class SupplementaryMemoryTests(unittest.TestCase):
    def test_direct_fact_memory_reuses_the_fusion_keys_and_values(self):
        model = make_model().eval()
        values = inputs()
        projected = []
        keys = []
        with ExitStack() as cleanup:
            cleanup.callback(model.fact_projector.register_forward_hook(
                lambda module, args, output: projected.append(output),
            ).remove)
            cleanup.callback(model.fusion[0].attention.register_forward_pre_hook(
                lambda module, args: keys.append(args[1]),
            ).remove)
            memory, padding = model.encode_memory(**values)
        self.assertEqual(len(projected), 1)
        self.assertEqual(len(keys), 1)
        torch.testing.assert_close(memory[:, 3:], keys[0], rtol=0, atol=0)
        self.assertTrue(keys[0].requires_grad)
        self.assertEqual(tuple(memory.shape), (2, 6, 16))
        self.assertEqual(padding.tolist(), [
            [False, False, False, False, True, True], [False] * 6,
        ])
        self.assertEqual(memory[0, 4:].count_nonzero().item(), 0)
        with_cls, _ = model.encode_memory(**values, include_cls_token=True)
        self.assertEqual(tuple(with_cls.shape), (2, 7, 16))

    def test_caption_memory_is_appended_and_shares_decoder_embedding(self):
        model = make_model(caption_context=True).eval()
        values = inputs(caption_context=True)
        memory, padding = model.encode_memory(**values)
        fused, _, _ = model.encode_fusion(
            values["visual_inputs"], values["fact_features"], values["fact_attention_mask"],
        )
        context = model.context_encoder(values["rag_input_ids"], values["rag_attention_mask"])
        torch.testing.assert_close(memory[:, :3], fused)
        torch.testing.assert_close(memory[:, 3:], context)
        self.assertIs(model.context_encoder.embedding, model.decoder.embedding)
        self.assertIs(model.shared_embedding, model.decoder.embedding)
        self.assertEqual(tuple(memory.shape), (2, 7, 16))
        self.assertTrue(torch.equal(padding[:, 3:], values["rag_attention_mask"].eq(0)))
        self.assertFalse(padding[:, :3].any())
        empty = dict(values, rag_attention_mask=torch.zeros_like(values["rag_attention_mask"]))
        with self.assertRaisesRegex(ValueError, "caption context token"):
            model.encode_memory(**empty)

    def test_logits_ignore_cls_fact_padding_and_caption_padding(self):
        ids = torch.tensor([[1, 4, 2, 0, 0, 0], [1, 5, 6, 2, 0, 0]])
        for caption_context in (False, True):
            with self.subTest(caption_context=caption_context), torch.no_grad():
                model = make_model(caption_context).eval()
                values = inputs(caption_context)
                expected = model(**values, input_ids=ids, attention_mask=ids.ne(0))
                changed = {key: value.clone() for key, value in values.items()}
                changed["visual_inputs"][:, 0] += 1000
                changed["fact_features"][~values["fact_attention_mask"]] += 1000
                if caption_context:
                    changed["rag_input_ids"][values["rag_attention_mask"].eq(0)] = 9
                actual = model(**changed, input_ids=ids, attention_mask=ids.ne(0))
                torch.testing.assert_close(actual, expected)
                changed["fact_features"][0, 0] *= -1
                affected = model(**changed, input_ids=ids, attention_mask=ids.ne(0))
                self.assertFalse(torch.allclose(affected[0], expected[0]))

    def test_direct_fact_memory_accepts_different_slot_counts(self):
        model = make_model().eval()
        values = inputs()
        ids = torch.tensor([[1, 4, 2], [1, 5, 2]])
        with torch.no_grad():
            expected = model(**values, input_ids=ids, attention_mask=ids.ne(0))
            values["fact_features"] = torch.cat([values["fact_features"], torch.randn(2, 4, 8)], dim=1)
            values["fact_attention_mask"] = torch.cat([
                values["fact_attention_mask"], torch.zeros(2, 4, dtype=torch.bool),
            ], dim=1)
            actual = model(**values, input_ids=ids, attention_mask=ids.ne(0))
        torch.testing.assert_close(actual, expected)

    def test_all_trainable_branches_receive_gradients(self):
        ids = torch.tensor([[1, 4, 2, 0, 0, 0], [1, 5, 6, 2, 0, 0]])
        for caption_context in (False, True):
            with self.subTest(caption_context=caption_context):
                model = make_model(caption_context)
                logits = model(**inputs(caption_context), input_ids=ids, attention_mask=ids.ne(0))
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), ids[:, 1:].flatten(), ignore_index=0,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(model.fact_projector[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.fusion[0].attention.in_proj_weight.grad.abs().sum().item(), 0)
                if caption_context:
                    self.assertGreater(model.context_encoder.encoder.layers[0].linear1.weight.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_one_batch_forward_backward_and_beam_search(self):
        for caption_context in (False, True):
            with self.subTest(caption_context=caption_context):
                model = make_model(caption_context).cuda()
                values = inputs(caption_context, device="cuda")
                ids = torch.tensor([[1, 4, 2, 0, 0, 0], [1, 5, 6, 2, 0, 0]], device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**values, input_ids=ids, attention_mask=ids.ne(0))
                    loss = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1), ids[:, 1:].flatten(), ignore_index=0,
                    )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))
                vocab = Vocabulary(freq_threshold=1)
                vocab.build_vocabulary([["dog", "tree", "red", "grass", "on", "a", "ball", "runs"]])
                self.assertEqual(len(vocab), 12)
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    memory, padding = model.encode_memory(**values)
                    output = beam_search(model.decoder, memory, vocab, 2, 6,
                                         memory_key_padding_mask=padding)
                self.assertEqual(tuple(output.shape), (2, 6))


class SupplementaryDatasetTests(unittest.TestCase):
    def test_caption_tokens_and_facts_follow_the_same_selected_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Path(directory) / "contexts.parquet"
            write_context(context)
            facts = load_rag_fact_lookup(context, 1, "itm")
            captions = load_caption_context_lookup(context, 1, "itm")
            self.assertEqual(facts[7], ["dog", "red dog", "dog on grass"])
            self.assertEqual(captions[7], [["red", "dog"]])
            facts = load_rag_fact_lookup(context, 1, "clip")
            captions = load_caption_context_lookup(context, 1, "clip")
            self.assertEqual(facts[7], ["outside"])
            self.assertEqual(captions[7], [["dog", "tree"]])
            table = pq.read_table(context)
            rows = table.to_pylist()
            rows[0]["tokens"] = [["dog"]]
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), context)
            with self.assertRaisesRegex(ValueError, "tokens must align"):
                load_caption_context_lookup(context, 1, "itm")

    def test_missing_or_empty_caption_tokens_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Path(directory) / "contexts.parquet"
            features = Path(directory) / "features.h5"
            write_context(context)
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            vocab = Vocabulary(freq_threshold=1)
            vocab.build_vocabulary([["dog"]])
            table = pq.read_table(context)
            pq.write_table(table.drop(["tokens"]), context)
            with self.assertRaisesRegex(ValueError, "Missing caption context columns"):
                RAGFeatureDataset(pd.DataFrame({"imgid": [7]}), vocab, features, context, 8, 1)
            rows = table.to_pylist()
            rows[0]["tokens"] = [[], []]
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), context)
            with self.assertRaisesRegex(ValueError, "No caption context tokens"):
                RAGFeatureDataset(pd.DataFrame({"imgid": [7]}), vocab, features, context, 8, 1)


class SupplementaryScriptTests(unittest.TestCase):
    def test_both_pipelines_train_validate_and_generate_one_batch_with_zero_workers(self):
        for pipeline in ("fact_text_rag", "fact_memory_rag"):
            with self.subTest(pipeline=pipeline):
                self.run_pipeline(pipeline)

    def run_pipeline(self, pipeline):
        train = importlib.import_module(f"script.{pipeline}.train_rag")
        generate = importlib.import_module(f"script.{pipeline}.generate_rag_captions")
        accelerator = Accelerator(cpu=True)
        text_model = tiny_clip()
        initial_text = copy.deepcopy(text_model.state_dict())
        tokenizer = TinyTokenizer()
        encoders = []
        batch_widths = []
        original_collate = train.collate_fact_batch

        def collate_tuples(samples):
            self.assertTrue(all(isinstance(sample, tuple) for sample in samples))
            batch = original_collate(samples)
            self.assertIsInstance(batch, tuple)
            batch_widths.append(len(batch))
            return batch

        def make_encoder(*args):
            encoder = CachedCLIPFactEncoder(*args)
            encoders.append(encoder)
            return encoder

        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            cleanup.callback(close_loader_stores, accelerator)
            root = Path(directory)
            context = root / "contexts.parquet"
            features = root / "features.h5"
            write_context(context)
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            frame = pd.DataFrame({"imgid": [7, 7, 8], "tokens": [["dog"], ["tree"], ["dog"]]})
            split = root / "split.parquet"
            frame.to_parquet(split, index=False)
            test_split = root / "test.parquet"
            pd.DataFrame({"imgid": [7, 8]}).to_parquet(test_split, index=False)
            vocab = Vocabulary(freq_threshold=1)
            vocab.build_vocabulary([["dog", "tree", "red", "on", "grass"]])
            vocab_path = vocab.save(root / "vocab.json")
            settings = dict(
                BATCH_SIZE=3, NUM_WORKERS=0, NUM_EPOCHS=1, DMODEL=16, NHEADS=2, NLAYERS=1,
                DROPOUT=0., MAX_LENGTH=6, CTX_NLAYERS=1, MAX_CTX_LENGTH=8,
                FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2,
                VISUAL_ENCODER_MODEL="test/clip", TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                VOCAB_PATH=vocab_path, BEST_CHECKPOINT_PATH=root / "best.pth",
                LAST_CHECKPOINT_PATH=root / "last.pth", PREDICTIONS_PATH=root / "predictions.json",
                RUN_MODE=pipeline, BEAM_SIZE=2,
                TRAIN_DF_PATH=split, VAL_DF_PATH=split, TEST_DF_PATH=test_split,
                TRAIN_I2T_PATH=context, VAL_I2T_PATH=context, TEST_I2T_PATH=context,
                TRAIN_VISUAL_FEATURES_PATH=features, VAL_VISUAL_FEATURES_PATH=features,
                TEST_VISUAL_FEATURES_PATH=features,
            )
            training_settings = dict(settings)
            for name in ("TEST_DF_PATH", "TEST_I2T_PATH", "TEST_VISUAL_FEATURES_PATH",
                         "PREDICTIONS_PATH", "BEAM_SIZE"):
                training_settings.pop(name)
            generation_settings = {
                name: settings[name] for name in (
                    "BATCH_SIZE", "NUM_WORKERS", "DMODEL", "NHEADS", "NLAYERS",
                    "DROPOUT", "MAX_LENGTH", "CTX_NLAYERS", "MAX_CTX_LENGTH",
                    "FACT_FUSION_NLAYERS", "FACT_TEXT_BATCH_SIZE", "VISUAL_ENCODER_MODEL",
                    "TEXT_RAG_CAPTION_SORT_BY", "TOP_K_CAPTIONS", "VOCAB_PATH",
                    "BEST_CHECKPOINT_PATH", "PREDICTIONS_PATH", "RUN_MODE", "BEAM_SIZE",
                    "TEST_DF_PATH", "TEST_I2T_PATH", "TEST_VISUAL_FEATURES_PATH",
                )
            }
            if pipeline == "fact_memory_rag":
                training_settings.pop("CTX_NLAYERS")
                training_settings.pop("MAX_CTX_LENGTH")
                generation_settings.pop("CTX_NLAYERS")
                generation_settings.pop("MAX_CTX_LENGTH")
            cleanup.enter_context(patch.multiple(train, **training_settings))
            cleanup.enter_context(patch.multiple(generate, **generation_settings))
            cleanup.enter_context(patch("src.shared.facts.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer))
            cleanup.enter_context(patch("src.shared.facts.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=text_model))
            cleanup.enter_context(patch.object(train, "Accelerator", return_value=accelerator))
            cleanup.enter_context(patch.object(generate, "Accelerator", return_value=accelerator))
            cleanup.enter_context(patch.object(train, "CachedCLIPFactEncoder", side_effect=make_encoder))
            cleanup.enter_context(patch.object(generate, "CachedCLIPFactEncoder", side_effect=make_encoder))
            cleanup.enter_context(patch.object(train, "collate_fact_batch", side_effect=collate_tuples))
            cleanup.enter_context(patch.object(generate, "collate_fact_batch", side_effect=collate_tuples))
            train.main()
            checkpoint = torch.load(settings["BEST_CHECKPOINT_PATH"], weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(checkpoint["train_loss"])))
            self.assertTrue(all(not key.startswith("fact_encoder") for key in checkpoint["model_state_dict"]))
            self.assertEqual(len(encoders[0].cache), 4)
            self.assertEqual(len(tokenizer.calls), 2)  # Validation reuses the training cache.
            with patch(f"src.{pipeline}.inference_rag.beam_search", wraps=beam_search) as search:
                generate.main()
            self.assertEqual(search.call_count, 1)
            padding = search.call_args.kwargs["memory_key_padding_mask"]
            self.assertEqual(padding.shape[0], 2)
            self.assertTrue(padding.any())
            predictions = json.loads(settings["PREDICTIONS_PATH"].read_text(encoding="utf-8"))
            self.assertEqual([row["imgid"] for row in predictions], [7, 8])
            self.assertTrue(all(isinstance(row["caption"], str) for row in predictions))
            self.assertEqual(len(encoders[1].cache), 4)
            self.assertEqual(len(tokenizer.calls), 4)  # Generation owns a fresh cache.
            self.assertEqual(batch_widths, [6, 6, 5] if pipeline == "fact_text_rag" else [4, 4, 3])
            for key, value in text_model.state_dict().items():
                torch.testing.assert_close(value, initial_text[key], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in text_model.parameters()))


if __name__ == "__main__":
    unittest.main()
