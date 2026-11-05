"""Fact-token selection, empty memory and one-batch offline script checks."""

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
import pyarrow.parquet as pq
import torch

from src.fact_rag.models import FactRAGCaptioner
from src.fact_token_rag.dataset import RAGFeatureDataset, encode_fact_context, load_fact_token_context_lookup, tokenize_fact
from src.fact_token_rag.models import FactTokenRAGCaptioner
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.shared.inference import beam_search
from src.shared.utils import setup_reproducibility
from src.shared.vocabulary import Vocabulary
from test_fact_rag import TinyTokenizer, close_loader_stores, tiny_clip, write_context


def context_rows(empty=False):
    return [
        dict(imgid=7, captions=["first", "best"], itm_scores=[.1, .9],
             object_texts=[["outside"], [" dog ", "dog"]],
             attribute_texts=[[], [] if empty else ["red dog"]],
             triplet_texts=[[], [] if empty else ["dog on grass"]]),
        dict(imgid=8, captions=["tree"], itm_scores=[.5], object_texts=[["tree"]],
             attribute_texts=[[]], triplet_texts=[[]]),
    ]


def vocabulary():
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary([["dog", "tree", "red", "on", "grass", "mans", "t", "shirt"]])
    return vocab


def make_model():
    return FactTokenRAGCaptioner(
        vocab_size=12, d_model=16, nheads=2, nlayers=1, dropout=0., max_length=6,
        pad_idx=0, visual_feature_dim=6, fact_feature_dim=8,
        context_layers=1, max_context_length=12,
    )


def inputs():
    return dict(
        visual_inputs=torch.randn(2, 4, 6), fact_features=torch.randn(2, 3, 8),
        fact_attention_mask=torch.tensor([[True, False, False], [True, True, True]]),
        rag_input_ids=torch.tensor([[4, 5, 2, 0], [0, 0, 0, 0]]),
        rag_attention_mask=torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]]),
    )


class FactTokenDatasetTests(unittest.TestCase):
    def test_tokenizer_deduplication_and_whole_fact_budgets(self):
        self.assertEqual(tokenize_fact(" Man's T-shirt "), ["mans", "t", "shirt"])
        vocab = vocabulary()
        ids, mask = encode_fact_context(
            [["red dog", "dog on grass"], [" red   dog ", "dog on grass"]], vocab, 6, 2,
        )
        # A fact omitted from caption 1 can still be included from caption 2.
        self.assertEqual(vocab.decode(ids[mask.bool()].tolist()),
                         ["red", "dog", "<eos>", "dog", "on", "grass", "<eos>"])
        self.assertEqual(tuple(ids.shape), (12,))
        ids, mask = encode_fact_context([["red dog", "red   dog"]], vocab, 6, 1)
        self.assertEqual(mask.sum().item(), 3)
        self.assertEqual(vocab.decode(ids[mask.bool()].tolist()), ["red", "dog", "<eos>"])

    def test_selection_and_empty_supplement_keep_fusion_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            context, features = Path(directory) / "contexts.parquet", Path(directory) / "features.h5"
            write_context(context, context_rows())
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            self.assertEqual(load_fact_token_context_lookup(context, 1, "itm")[7], [["red dog", "dog on grass"]])
            self.assertEqual(load_fact_token_context_lookup(context, 1, "clip")[7], [[]])
            dataset = RAGFeatureDataset(pd.DataFrame({"imgid": [7, 8]}), vocabulary(), features, context, 8, 1)
            self.assertEqual(dataset.fact_lookup[7], ["dog", "red dog", "dog on grass"])
            self.assertEqual(dataset[0][3].sum().item(), 7)
            self.assertEqual(dataset[1][1], ["tree"])
            self.assertEqual(dataset[1][3].sum().item(), 0)
            dataset.feature_store._h5_file.close()
            dataset.feature_store._h5_file = None
            self.assertNotIn("tokens", pq.read_schema(context).names)
            values = context_rows()
            values[0]["attribute_texts"][1] = ["red motorbike"]
            write_context(context, values)
            self.assertEqual(load_fact_token_context_lookup(context, 1)[7][0][0], "red motorbike")
            values[1]["object_texts"] = [[]]
            write_context(context, values)
            with self.assertRaisesRegex(ValueError, "No facts selected"):
                RAGFeatureDataset(pd.DataFrame({"imgid": [8]}), vocabulary(), features, context, 8, 1)


class FactTokenMemoryTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_forward_backward_and_beam_search_one_batch(self):
        setup_reproducibility(42)
        model = make_model().cuda()
        values = {name: value.cuda() for name, value in inputs().items()}
        ids = torch.tensor([[1, 4, 2, 0], [1, 5, 6, 2]], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**values, input_ids=ids, attention_mask=ids.ne(0))
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            memory, padding = model.encode_memory(**values)
            sequences = beam_search(model.decoder, memory, vocabulary(), 2, 6,
                                    memory_key_padding_mask=padding)
        self.assertEqual(tuple(sequences.shape), (2, 6))

    def test_mixed_batch_preserves_base_fusion_and_masks_empty_row(self):
        model, values = make_model().eval(), inputs()
        sizes = []
        hook = model.context_encoder.register_forward_pre_hook(lambda module, args: sizes.append(args[0].size(0)))
        with torch.no_grad():
            memory, padding = model.encode_memory(**values)
            fused = FactRAGCaptioner.encode_memory(model, values["visual_inputs"], values["fact_features"], values["fact_attention_mask"])
        hook.remove()
        self.assertEqual(sizes, [1])
        torch.testing.assert_close(memory[:, :3], fused, rtol=0, atol=0)
        self.assertTrue(padding[1, 3:].all())
        self.assertEqual(memory[1, 3:].count_nonzero().item(), 0)
        self.assertEqual(sum(isinstance(module, torch.nn.Embedding) for module in model.modules()), 1)
        ids = torch.tensor([[1, 4, 2], [1, 5, 2]])
        with torch.no_grad():
            mixed = model.decoder(ids, memory, memory_key_padding_mask=padding)
            fusion_only = model.decoder(ids[1:], fused[1:])
        torch.testing.assert_close(mixed[1:], fusion_only)
        with_cls, _ = model.encode_memory(**values, include_cls_token=True)
        self.assertEqual(with_cls.size(1), memory.size(1) + 1)

    def test_all_empty_batch_skips_text_encoder(self):
        model, values = make_model().eval(), inputs()
        values["rag_attention_mask"].zero_()
        with patch.object(model.context_encoder, "forward", side_effect=AssertionError("Empty encoder input")):
            memory, padding = model.encode_memory(**values)
        self.assertEqual(tuple(memory.shape), (2, 3, 16))
        self.assertFalse(padding.any())
        ids = torch.tensor([[1, 4, 2], [1, 5, 2]])
        loss = model(**values, input_ids=ids, attention_mask=ids.ne(0)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.fact_projector[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.context_encoder.parameters()))

    def test_padding_invariance_and_branch_gradients(self):
        model, values = make_model().eval(), inputs()
        ids = torch.tensor([[1, 4, 2, 0], [1, 5, 6, 2]])
        expected = model(**values, input_ids=ids, attention_mask=ids.ne(0))
        changed = {name: value.clone() for name, value in values.items()}
        changed["visual_inputs"][:, 0] += 1000
        changed["fact_features"][~values["fact_attention_mask"]] += 1000
        changed["rag_input_ids"][values["rag_attention_mask"].eq(0)] = 9
        actual = model(**changed, input_ids=ids, attention_mask=ids.ne(0))
        torch.testing.assert_close(actual, expected)
        loss = torch.nn.functional.cross_entropy(expected.flatten(0, 1), ids[:, 1:].flatten(), ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.context_encoder.layers[0].linear1.weight.grad.abs().sum().item(), 0)


class FactTokenScriptTests(unittest.TestCase):
    def test_one_batch_train_validate_generate_mixed_and_empty_context(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                self.run_pipeline(empty)

    def run_pipeline(self, empty):
        train = importlib.import_module("script.fact_token_rag.train_rag")
        generate = importlib.import_module("script.fact_token_rag.generate_rag_captions")
        accelerator, text_model, tokenizer = Accelerator(cpu=True), tiny_clip(), TinyTokenizer()
        initial_text, encoders = copy.deepcopy(text_model.state_dict()), []

        def make_encoder(*args):
            encoder = CachedCLIPFactEncoder(*args)
            encoders.append(encoder)
            return encoder

        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            cleanup.callback(close_loader_stores, accelerator)
            root = Path(directory)
            context, features = root / "contexts.parquet", root / "features.h5"
            write_context(context, context_rows(empty))
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            split, test_split = root / "split.parquet", root / "test.parquet"
            pd.DataFrame({"imgid": [7, 7, 8], "tokens": [["dog"], ["tree"], ["dog"]]}).to_parquet(split, index=False)
            pd.DataFrame({"imgid": [7, 8], "cocoid": [101, 202]}).to_parquet(test_split, index=False)
            common = dict(
                BATCH_SIZE=3, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=1,
                DROPOUT=0., MAX_LENGTH=6, CTX_NLAYERS=1,
                FACT_CTX_TOKENS_PER_CAPTION=8, MAX_FACT_CTX_LENGTH=8,
                FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2, VISUAL_ENCODER_MODEL="test/clip",
                TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                VOCAB_PATH=vocabulary().save(root / "vocab.json"),
                BEST_CHECKPOINT_PATH=root / "best.pth", RUN_MODE="fact_token_test",
            )
            training = dict(common, NUM_EPOCHS=1, LAST_CHECKPOINT_PATH=root / "last.pth",
                            TRAIN_DF_PATH=split, VAL_DF_PATH=split, TRAIN_I2T_PATH=context, VAL_I2T_PATH=context,
                            TRAIN_VISUAL_FEATURES_PATH=features, VAL_VISUAL_FEATURES_PATH=features)
            generation = dict(common, BEAM_SIZE=2, PREDICTIONS_PATH=root / "predictions.json",
                              TEST_DF_PATH=test_split, TEST_I2T_PATH=context, TEST_VISUAL_FEATURES_PATH=features)
            cleanup.enter_context(patch.multiple(train, **training))
            cleanup.enter_context(patch.multiple(generate, **generation))
            cleanup.enter_context(patch("src.shared.facts.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer))
            cleanup.enter_context(patch("src.shared.facts.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=text_model))
            accelerator_factory = cleanup.enter_context(patch.object(train, "Accelerator", return_value=accelerator))
            cleanup.enter_context(patch.object(generate, "Accelerator", return_value=accelerator))
            cleanup.enter_context(patch.object(train, "CachedCLIPFactEncoder", side_effect=make_encoder))
            cleanup.enter_context(patch.object(generate, "CachedCLIPFactEncoder", side_effect=make_encoder))
            train.main()
            self.assertTrue(accelerator_factory.call_args.kwargs["kwargs_handlers"][0].find_unused_parameters)
            checkpoint = torch.load(common["BEST_CHECKPOINT_PATH"], weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(checkpoint["train_loss"])))
            expected_facts, expected_calls = (2, 1) if empty else (4, 2)
            self.assertEqual(len(encoders[0].cache), expected_facts)
            self.assertEqual(len(tokenizer.calls), expected_calls)
            with patch("src.fact_token_rag.inference_rag.beam_search", wraps=beam_search) as search:
                generate.main()
            self.assertEqual(search.call_count, 1)
            padding = search.call_args.kwargs["memory_key_padding_mask"]
            self.assertEqual(tuple(padding.shape), (2, 3 if empty else 11))
            if not empty:
                self.assertTrue(padding[1, 3:].all())
            predictions = json.loads(generation["PREDICTIONS_PATH"].read_text(encoding="utf-8"))
            self.assertEqual([row["imgid"] for row in predictions], [7, 8])
            self.assertEqual([row["cocoid"] for row in predictions], [101, 202])
            self.assertEqual(len(encoders[1].cache), expected_facts)
            self.assertEqual(len(tokenizer.calls), expected_calls * 2)
            for name, value in text_model.state_dict().items():
                torch.testing.assert_close(value, initial_text[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in text_model.parameters()))


if __name__ == "__main__":
    unittest.main()
