"""Offline checks for fact gates, gradients, beam routing and one-batch scripts."""

import copy
import importlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from accelerate import Accelerator
import h5py
import pandas as pd
import torch

from src.fact_fusion_fact_bias.inference_rag import _FactBiasBeamDecoder
from src.fact_fusion_fact_bias.model import FactFusionFactBiasCaptioner
from src.fact_fusion_fact_bias.tokenizer import encode_fact_token_ids
from src.fact_fusion.models import FactRAGCaptioner
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.shared.facts.tokenizer import tokenize_fact
from src.shared.inference import beam_search
from src.shared.utils import setup_reproducibility
from src.shared.vocabulary import Vocabulary
from test_fact_rag import TinyTokenizer, close_loader_stores, tiny_clip, write_context


def vocabulary():
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary([["a", "mans", "blue", "green", "shirt", "dog", "tree", "red", "on", "grass"]])
    return vocab


def make_model():
    return FactFusionFactBiasCaptioner(
        vocab_size=len(vocabulary()), d_model=16, nheads=2, nlayers=1, dropout=0.,
        max_length=6, pad_idx=0, visual_feature_dim=6, fact_feature_dim=8,
    )


def model_inputs(device="cpu", all_oov=False):
    facts = [["notinvocabulary"] if all_oov else ["dog", "red dog", "dog on grass"], ["tree"]]
    fact_token_ids, fact_token_mask = encode_fact_token_ids(facts, vocabulary(), device)
    num_facts = fact_token_ids.size(1)
    fact_mask = torch.zeros(2, num_facts, dtype=torch.bool, device=device)
    for index, group in enumerate(facts):
        fact_mask[index, :len(group)] = True
    input_ids = torch.tensor([[1, 9, 2, 0], [1, 10, 11, 2]], device=device)
    return dict(
        visual_inputs=torch.randn(2, 4, 6, device=device),
        input_ids=input_ids, attention_mask=input_ids.ne(0),
        fact_features=torch.randn(2, num_facts, 8, device=device),
        fact_attention_mask=fact_mask, fact_token_ids=fact_token_ids, fact_token_mask=fact_token_mask,
    )


class FactBiasTests(unittest.TestCase):
    def test_tokenization_per_fact_dedup_oov_and_padding(self):
        self.assertEqual(tokenize_fact("A man's blue-green shirt, man's!"),
                         ["a", "mans", "blue", "green", "shirt", "mans"])
        self.assertEqual(tokenize_fact("MAN’S shirt"), ["mans", "shirt"])
        vocab = vocabulary()
        facts = [["A man's blue-green shirt, man's!", "<sos> <eos> <pad> <unk> mystery"], ["dog dog"]]
        ids, mask = encode_fact_token_ids(facts, vocab, "cpu")
        self.assertEqual(vocab.decode(ids[0, 0][mask[0, 0]].tolist()), ["a", "mans", "blue", "green", "shirt"])
        self.assertFalse(mask[0, 1].any())
        self.assertEqual(vocab.decode(ids[1, 0][mask[1, 0]].tolist()), ["dog"])
        self.assertFalse(mask[1, 1].any())
        with self.assertRaisesRegex(ValueError, "at least one"):
            encode_fact_token_ids([[]], vocab, "cpu")

    def test_bias_matches_concat_mlp_and_max_reference(self):
        setup_reproducibility(42)
        decoder = make_model().decoder
        hidden = torch.randn(2, 3, 16)
        facts = torch.randn(2, 3, 16)
        ids, mask = encode_fact_token_ids([["dog", "red dog", "dog on grass"], ["tree"]], vocabulary(), "cpu")
        padding = torch.tensor([[False, False, False], [False, True, True]])
        bias = decoder.compute_bias(hidden, facts, padding, ids, mask)
        expected = torch.zeros_like(bias)
        for batch_index in range(2):
            for step in range(3):
                for fact_index in range(3):
                    if padding[batch_index, fact_index]:
                        continue
                    gate = decoder.fact_gate_mlp(torch.cat([hidden[batch_index, step], facts[batch_index, fact_index]])).squeeze()
                    for token_id in ids[batch_index, fact_index][mask[batch_index, fact_index]]:
                        expected[batch_index, step, token_id] = torch.maximum(expected[batch_index, step, token_id], gate)
        torch.testing.assert_close(bias, expected)
        self.assertAlmostEqual(decoder.alpha.item(), 0.1, places=6)
        facts[1, 1:] = 1000
        torch.testing.assert_close(decoder.compute_bias(hidden, facts, padding, ids, mask), bias)
        self.assertEqual(bias[:, :, :4].count_nonzero().item(), 0)

    def test_fusion_and_zero_alpha_match_fact_rag_then_bias_trains(self):
        setup_reproducibility(42)
        model = make_model().eval()
        base = FactRAGCaptioner(
            len(vocabulary()), 16, 2, 1, 0., 6, 0, 6, 8,
        ).eval()
        base.load_state_dict({name: value for name, value in model.state_dict().items()
                              if name in base.state_dict()})
        values = model_inputs()
        base_values = {name: value for name, value in values.items() if not name.startswith("fact_token")}
        with torch.no_grad():
            fused, _, _ = model.encode_memory(values["visual_inputs"], values["fact_features"], values["fact_attention_mask"])
            original = base.encode_memory(values["visual_inputs"], values["fact_features"], values["fact_attention_mask"])
            torch.testing.assert_close(fused, original, rtol=0, atol=0)
            model.decoder.raw_alpha.fill_(-1000)
            torch.testing.assert_close(model(**values), base(**base_values), rtol=0, atol=0)
            with_cls, _, _ = model.encode_memory(values["visual_inputs"], values["fact_features"], values["fact_attention_mask"], True)
            self.assertEqual(with_cls.size(1), fused.size(1) + 1)
            model.decoder.raw_alpha.fill_(math.log(math.expm1(0.1)))
        model.train()
        logits = model(**values)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), values["input_ids"][:, 1:].flatten(), ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertGreater(model.decoder.raw_alpha.grad.abs().item(), 0)
        self.assertGreater(model.decoder.fact_gate_mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.fact_projector[0].weight.grad.abs().sum().item(), 0)

    def test_all_oov_keeps_gate_in_autograd_with_zero_bias(self):
        model = make_model().train()
        values = model_inputs(all_oov=True)
        values["fact_token_mask"].zero_()
        logits = model(**values)
        logits.sum().backward()
        for parameter in model.decoder.fact_gate_mlp.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.count_nonzero().item(), 0)
        self.assertIsNotNone(model.decoder.raw_alpha.grad)
        self.assertEqual(model.decoder.raw_alpha.grad.item(), 0)

    def test_beam_search_uses_bias_and_keeps_images_separate(self):
        setup_reproducibility(42)
        model, vocab = make_model().eval(), vocabulary()
        with torch.no_grad():
            model.decoder.out_fc.weight.zero_()
            model.decoder.out_fc.bias.fill_(-4)
            model.decoder.out_fc.bias[vocab.eos_idx()] = 0
            for parameter in model.decoder.fact_gate_mlp.parameters():
                parameter.zero_()
            model.decoder.raw_alpha.fill_(20)
            ids, mask = encode_fact_token_ids([["dog"], ["tree"], ["unknownword"]], vocab, "cpu")
            memory, facts, padding = model.encode_memory(torch.randn(3, 4, 6), torch.randn(3, 1, 8), torch.ones(3, 1))
            adapter = _FactBiasBeamDecoder(model.decoder, facts, padding, ids, mask, 3)
            sequences = beam_search(adapter, memory, vocab, 3, 6)
            self.assertTrue(sequences[0, 1:].eq(vocab.stoi["dog"]).all())
            self.assertTrue(sequences[1, 1:].eq(vocab.stoi["tree"]).all())
            self.assertEqual(sequences[2, 1].item(), vocab.eos_idx())
            self.assertTrue(sequences[2, 2:].eq(vocab.pad_idx()).all())
            # Last-token generation takes the same path as teacher-forced decoding.
            prefix = torch.tensor([[1, 9], [1, 10], [1, 2]])
            kwargs = dict(memory=memory, fact_memory=facts, fact_padding_mask=padding,
                          fact_token_ids=ids, fact_token_mask=mask)
            full = model.decoder(prefix, **kwargs)
            last = model.decoder(prefix, return_last_step_only=True, **kwargs)
            torch.testing.assert_close(last, full[:, -1:])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_one_batch_forward_backward_and_search(self):
        setup_reproducibility(42)
        model, values = make_model().cuda(), model_inputs("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**values)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), values["input_ids"][:, 1:].flatten(), ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model.eval()
            memory, facts, padding = model.encode_memory(values["visual_inputs"], values["fact_features"], values["fact_attention_mask"])
            adapter = _FactBiasBeamDecoder(model.decoder, facts, padding, values["fact_token_ids"], values["fact_token_mask"], 2)
            sequences = beam_search(adapter, memory, vocabulary(), 2, 6)
        self.assertEqual(tuple(sequences.shape), (2, 6))


class FactBiasScriptTests(unittest.TestCase):
    def test_one_batch_train_val_generate_and_checkpoint(self):
        setup_reproducibility(42)
        train = importlib.import_module("script.fact_fusion_fact_bias.train")
        generate = importlib.import_module("script.fact_fusion_fact_bias.generate_captions")
        accelerator = Accelerator(cpu=True)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            root = Path(directory)
            context, features, split, test_split = [root / name for name in ("contexts.parquet", "features.h5", "train.parquet", "test.parquet")]
            rows = [
                dict(imgid=7, captions=["outside", "best"], itm_scores=[.1, .9],
                     object_texts=[["outside"], ["dog"]], attribute_texts=[[], ["red dog"]],
                     triplet_texts=[[], ["dog on grass"]]),
                dict(imgid=8, captions=["tree"], itm_scores=[.5], object_texts=[["tree"]],
                     attribute_texts=[[]], triplet_texts=[[]]),
            ]
            write_context(context, rows)
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            pd.DataFrame({"imgid": [7, 8], "tokens": [["red", "dog"], ["tree"]]}).to_parquet(split, index=False)
            pd.DataFrame({"imgid": [7, 8], "cocoid": [101, 202]}).to_parquet(test_split, index=False)
            common = dict(
                SEED=42, BATCH_SIZE=2, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=1,
                DROPOUT=0., MAX_LENGTH=6, FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2,
                VISUAL_ENCODER_MODEL="test/clip", TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                VOCAB_PATH=vocabulary().save(root / "vocab.json"), BEST_CHECKPOINT_PATH=root / "best.pth",
                RUN_MODE="fact_fusion_fact_bias_test",
            )
            training = dict(common, NUM_EPOCHS=1, LAST_CHECKPOINT_PATH=root / "last.pth",
                            TRAIN_DF_PATH=split, VAL_DF_PATH=split, TRAIN_I2T_PATH=context, VAL_I2T_PATH=context,
                            TRAIN_VISUAL_FEATURES_PATH=features, VAL_VISUAL_FEATURES_PATH=features)
            generation = dict(common, BEAM_SIZE=2, PREDICTIONS_PATH=root / "predictions.json",
                              TEST_DF_PATH=test_split, TEST_I2T_PATH=context, TEST_VISUAL_FEATURES_PATH=features)
            encoders = []
            tokenizer, text_model = TinyTokenizer(), tiny_clip()
            initial_text = copy.deepcopy(text_model.state_dict())

            def make_encoder(*args):
                encoder = CachedCLIPFactEncoder(*args)
                encoders.append(encoder)
                return encoder

            cleanup.enter_context(patch.multiple(train, **training))
            cleanup.enter_context(patch.multiple(generate, **generation))
            cleanup.enter_context(patch.object(train, "Accelerator", return_value=accelerator))
            cleanup.enter_context(patch.object(generate, "Accelerator", return_value=accelerator))
            cleanup.enter_context(patch("src.shared.facts.encoder.AutoTokenizer.from_pretrained", return_value=tokenizer))
            cleanup.enter_context(patch("src.shared.facts.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=text_model))
            cleanup.enter_context(patch.object(train, "CachedCLIPFactEncoder", side_effect=make_encoder))
            cleanup.enter_context(patch.object(generate, "CachedCLIPFactEncoder", side_effect=make_encoder))
            cleanup.enter_context(patch("os._exit"))
            # Run this callback before TemporaryDirectory cleanup on Windows.
            cleanup.callback(close_loader_stores, accelerator)
            train.main()
            checkpoint = torch.load(common["BEST_CHECKPOINT_PATH"], weights_only=True)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertIn("decoder.raw_alpha", checkpoint["model_state_dict"])
            self.assertIn("decoder.fact_gate_mlp.0.weight", checkpoint["model_state_dict"])
            self.assertEqual(len(encoders[0].cache), 4)
            self.assertEqual(len(tokenizer.calls), 2)  # Validation reuses the train cache.
            with patch("src.fact_fusion_fact_bias.inference_rag.beam_search", wraps=beam_search) as search:
                generate.main()
            self.assertEqual(search.call_count, 1)
            predictions = json.loads(generation["PREDICTIONS_PATH"].read_text(encoding="utf-8"))
            self.assertEqual([row["imgid"] for row in predictions], [7, 8])
            self.assertEqual([row["cocoid"] for row in predictions], [101, 202])
            self.assertEqual(len(encoders[1].cache), 4)
            self.assertEqual(len(tokenizer.calls), 4)
            for name, value in text_model.state_dict().items():
                torch.testing.assert_close(value, initial_text[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in text_model.parameters()))


if __name__ == "__main__":
    unittest.main()
