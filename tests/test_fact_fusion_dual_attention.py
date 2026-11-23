"""Offline checks for fact selection, gated decoding and one-batch scripts."""

import copy
import importlib
import json
import math
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from accelerate import Accelerator
import h5py
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from src.fact_fusion.models import FactRAGCaptioner
from src.fact_fusion_dual_attention.dataset import (
    RAGFeatureCaptionDataset, RAGFeatureDataset, collate_fact_batch,
)
from src.fact_fusion_dual_attention.inference_rag import _DualAttentionBeamDecoder
from src.fact_fusion_dual_attention.model import FactFusionDualAttentionCaptioner
from src.shared.facts.dataset import load_rag_fact_lookup
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.shared.inference import beam_search
from src.shared.utils import setup_reproducibility
from src.shared.vocabulary import Vocabulary
from test_fact_rag import TinyTokenizer, close_loader_stores, tiny_clip, write_context


def vocabulary():
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary([["a", "red", "dog", "on", "grass", "tree", "woman", "bicycle"]])
    return vocab


def make_model(device="cpu", initial_gate=0.1):
    return FactFusionDualAttentionCaptioner(
        len(vocabulary()), 16, 2, 2, 0., 6, 0, 6, 8, initial_gate=initial_gate,
    ).to(device)


def model_inputs(device="cpu"):
    return dict(
        visual_inputs=torch.randn(2, 4, 6, device=device),
        input_ids=torch.tensor([[1, 4, 5, 2, 0], [1, 7, 6, 5, 2]], device=device),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], device=device),
        fact_features=torch.randn(2, 3, 8, device=device),
        fact_attention_mask=torch.tensor([[True, True, True], [True, False, False]], device=device),
        attr_triplet_fact_indices=torch.tensor([[1, 2], [0, 0]], device=device),
        attr_triplet_fact_mask=torch.tensor([[True, True], [False, False]], device=device),
    )


def encode_inputs(model, values):
    return model.encode_memory(**{key: value for key, value in values.items()
                                 if key not in {"input_ids", "attention_mask"}})


def copy_base_decoder(model):
    base = FactRAGCaptioner(len(vocabulary()), 16, 2, 2, 0., 6, 0, 6, 8).eval()
    for name in ["visual_projector", "fact_projector", "fusion"]:
        getattr(base, name).load_state_dict(getattr(model, name).state_dict())
    for name in ["embedding", "pos_encoding", "out_fc"]:
        getattr(base.decoder, name).load_state_dict(getattr(model.decoder, name).state_dict())
    for base_layer, layer in zip(base.decoder.decoder.layers, model.decoder.layers):
        state = {name: value for name, value in layer.state_dict().items() if name in base_layer.state_dict()}
        base_layer.load_state_dict(state)
    return base


def distributed_worker(rank, rendezvous):
    """Compare DDP updates against a combined-batch reference, including unused branches."""
    setup_reproducibility(42)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = make_model()
        reference = copy.deepcopy(model)
        wrapped = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        torch.manual_seed(100 + rank)
        values = model_inputs()
        # Reuse one tiny synthetic batch per rank; vary only branch availability.
        for phase in range(3):
            batch = {**values, "attr_triplet_fact_mask": values["attr_triplet_fact_mask"].clone()}
            if (phase == 0 and rank == 0) or phase == 2:
                batch["attr_triplet_fact_mask"].zero_()
            rank_batches = [None, None]
            dist.all_gather_object(rank_batches, batch)
            combined = {key: torch.cat([item[key] for item in rank_batches]) for key in batch}
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            logits = wrapped(**batch)
            targets = batch["input_ids"][:, 1:]
            token_count = targets.ne(0).sum()
            dist.all_reduce(token_count)
            loss_sum = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0, reduction="sum")
            (loss_sum / token_count * 2).backward()
            expected = reference(**combined)
            reference_loss = nn.functional.cross_entropy(
                expected.flatten(0, 1), combined["input_ids"][:, 1:].flatten(), ignore_index=0,
            )
            reference_loss.backward()
            for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
                if expected_parameter.grad is None:
                    assert actual_parameter.grad is None
                else:
                    torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=2e-6, rtol=2e-4)
            optimizer.step()
            reference_optimizer.step()
            for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual_parameter, expected_parameter, atol=2e-6, rtol=2e-4)
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_two_cpu_ranks_with_different_empty_fact_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=True)


class DirectFactTests(unittest.TestCase):
    def test_ranking_dedup_provenance_indices_and_tuple_collation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, features = root / "context.parquet", root / "features.h5"
            rows = [
                dict(imgid=7, captions=["first", "best", "tied"], itm_scores=[.1, .9, .9],
                     object_texts=[["outside"], ["dog", "red dog"], ["tree"]],
                     attribute_texts=[[], [" red   dog ", "dog"], ["red dog"]],
                     triplet_texts=[[], ["dog on grass"], ["dog on grass", "tree near dog"]]),
                dict(imgid=8, captions=["tree"], itm_scores=[.5], object_texts=[["tree"]],
                     attribute_texts=[[]], triplet_texts=[[]]),
            ]
            write_context(context, rows)
            before = context.read_bytes()
            fusion, indices = load_rag_fact_lookup(context, 2, "itm", return_indices=True)
            self.assertEqual(fusion[7], ["dog", "red dog", "dog on grass", "tree", "tree near dog"])
            self.assertEqual(fusion, load_rag_fact_lookup(context, 2, "itm"))
            self.assertEqual(indices, {
                7: {"object_texts": [0, 1, 3], "attribute_texts": [1, 0], "triplet_texts": [2, 4]},
                8: {"object_texts": [0], "attribute_texts": [], "triplet_texts": []},
            })
            clip_fusion, clip_indices = load_rag_fact_lookup(context, 1, "clip", return_indices=True)
            self.assertEqual(clip_fusion, load_rag_fact_lookup(context, 1, "clip"))
            self.assertEqual(clip_indices[7], {
                "object_texts": [0], "attribute_texts": [], "triplet_texts": [],
            })
            self.assertEqual(context.read_bytes(), before)

            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            frame = pd.DataFrame({"imgid": [7, 8], "tokens": [["red", "dog"], ["tree"]]})
            with patch("src.shared.facts.dataset.pq.read_table", wraps=pq.read_table) as read_table:
                dataset = RAGFeatureCaptionDataset(frame, vocabulary(), features, context, 6, 2)
            self.assertEqual(read_table.call_count, 1)
            inference = RAGFeatureDataset(frame, features, context, 2)
            try:
                batch = collate_fact_batch([dataset[0], dataset[1]])
                self.assertEqual(len(batch), 6)
                self.assertEqual(batch[3], [fusion[7], fusion[8]])
                self.assertEqual(batch[4][0].tolist(), [0, 1, 2, 4])
                self.assertFalse(batch[5][1].any())
                empty_batch = collate_fact_batch([inference[1]])
                self.assertEqual(len(empty_batch), 5)
                self.assertEqual(tuple(empty_batch[2].shape), (1, 1))
                self.assertFalse(empty_batch[3].any())
            finally:
                for item in [dataset, inference]:
                    if item.feature_store._h5_file is not None:
                        item.feature_store._h5_file.close()


class DualAttentionTests(unittest.TestCase):
    def setUp(self):
        setup_reproducibility(42)

    def test_fusion_reuses_projected_vectors_adapter_once_and_cls_flag(self):
        model, values = make_model().eval(), model_inputs()
        base = copy_base_decoder(model)
        with patch.object(model.fact_adapter, "forward", wraps=model.fact_adapter.forward) as adapter:
            memory, facts, padding = encode_inputs(model, values)
        self.assertEqual(adapter.call_count, 1)
        expected_memory, projected, _ = base.encode_fusion(
            values["visual_inputs"], values["fact_features"], values["fact_attention_mask"],
        )
        torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
        torch.testing.assert_close(facts[0], model.fact_adapter(projected[0, [1, 2]]))
        self.assertEqual(padding.tolist(), [[False, False], [True, True]])
        self.assertEqual(facts[1].count_nonzero().item(), 0)
        direct_grad = torch.autograd.grad(facts[0, :, 0].sum(), model.fact_projector[0].weight, retain_graph=True)[0]
        fusion_grad = torch.autograd.grad(memory[:, :, 0].sum(), model.fact_projector[0].weight)[0]
        self.assertGreater(direct_grad.abs().sum().item(), 0)
        self.assertGreater(fusion_grad.abs().sum().item(), 0)
        cls_values = {**values, "include_cls_token": True}
        self.assertEqual(encode_inputs(model, cls_values)[0].size(1), memory.size(1) + 1)
        invalid = {**values, "attr_triplet_fact_indices": torch.tensor([[1, 2], [2, 0]]),
                   "attr_triplet_fact_mask": torch.tensor([[True, True], [True, False]])}
        with self.assertRaisesRegex(ValueError, "fusion padding"):
            encode_inputs(model, invalid)
        with self.assertRaisesRegex(ValueError, "at least one valid fact"):
            encode_inputs(model, {**values, "fact_attention_mask": torch.zeros_like(values["fact_attention_mask"])})

    def test_gate_norms_parallel_queries_raw_mix_and_independent_layers(self):
        model, values = make_model().eval(), model_inputs()
        layer, other = model.decoder.layers
        self.assertIsNot(layer.gate_mlp[0].weight, other.gate_mlp[0].weight)
        self.assertIsNot(layer.gate_img_norm.weight, other.gate_img_norm.weight)
        self.assertIsNot(layer.fact_attention.in_proj_weight, layer.multihead_attn.in_proj_weight)
        captured = {}
        with ExitStack() as stack:
            for name in ["multihead_attn", "fact_attention", "gate_img_norm", "gate_fact_norm", "gate_mlp", "norm2"]:
                def capture(module, args, output, name=name):
                    captured[name] = (args, output)
                stack.enter_context(getattr(layer, name).register_forward_hook(capture))
            model(**values)
        image_context = captured["multihead_attn"][1][0]
        fact_context = captured["fact_attention"][1][0]
        hidden = captured["multihead_attn"][0][0]
        torch.testing.assert_close(captured["fact_attention"][0][0], hidden[:1])
        gate_inputs, gate = captured["gate_mlp"][0][0], captured["gate_mlp"][1]
        self.assertEqual(tuple(gate.shape), (1, 4, 1))
        torch.testing.assert_close(gate, torch.full_like(gate, .1))
        torch.testing.assert_close(gate_inputs[..., :16], hidden[:1])
        torch.testing.assert_close(gate_inputs[..., 16:32], layer.gate_img_norm(image_context[:1]))
        torch.testing.assert_close(gate_inputs[..., 32:], layer.gate_fact_norm(fact_context))
        expected = hidden + image_context
        expected[0] = hidden[0] + .9 * image_context[0] + .1 * fact_context[0]
        torch.testing.assert_close(captured["norm2"][0][0], expected)

    def test_zero_gate_and_no_fact_rows_match_original_decoder(self):
        model, values = make_model().eval(), model_inputs()
        base = copy_base_decoder(model)
        base_values = {k: v for k, v in values.items() if k not in {
            "attr_triplet_fact_indices", "attr_triplet_fact_mask",
        }}
        original = base(**base_values)
        actual = model(**values)
        torch.testing.assert_close(actual[1], original[1], atol=1e-6, rtol=1e-5)
        with ExitStack() as stack:
            for layer in model.decoder.layers:
                stack.enter_context(layer.gate_mlp.register_forward_hook(lambda module, args, output: torch.zeros_like(output)))
            torch.testing.assert_close(model(**values), original, atol=1e-6, rtol=1e-5)
        values["attr_triplet_fact_mask"].zero_()
        with patch.object(model.fact_adapter, "forward", wraps=model.fact_adapter.forward) as adapter:
            torch.testing.assert_close(model(**values), original, atol=1e-6, rtol=1e-5)
        self.assertEqual(adapter.call_count, 0)
        model.train()
        model(**values).sum().backward()
        self.assertIsNone(model.fact_adapter.ffn[0].weight.grad)
        self.assertIsNone(model.decoder.layers[0].fact_attention.in_proj_weight.grad)
        self.assertIsNotNone(model.decoder.layers[0].self_attn.in_proj_weight.grad)

    def test_padding_permutation_and_masked_index_do_not_change_output(self):
        model, values = make_model().eval(), model_inputs()
        expected = model(**values)
        altered = copy.deepcopy(values)
        altered["fact_features"][1, 1:] = 1000
        altered["attr_triplet_fact_indices"][1] = -999
        torch.testing.assert_close(model(**altered), expected)
        altered = copy.deepcopy(values)
        altered["attr_triplet_fact_indices"][0] = torch.tensor([2, 1])
        torch.testing.assert_close(model(**altered), expected, atol=1e-6, rtol=1e-5)
        memory, facts, padding = encode_inputs(model, values)
        prefix = values["input_ids"][:, :-1]
        original = model.decoder(prefix, memory, facts, padding)
        extra_facts = torch.cat([facts, torch.randn(2, 3, 16) * 1000], dim=1)
        extra_mask = torch.cat([padding, torch.ones(2, 3, dtype=torch.bool)], dim=1)
        torch.testing.assert_close(model.decoder(prefix, memory, extra_facts, extra_mask), original)

    def test_causal_mask_and_last_step_only_with_dynamic_gate(self):
        model, values = make_model().eval(), model_inputs()
        with torch.no_grad():
            for layer in model.decoder.layers:
                layer.gate_mlp[2].weight.normal_(0, .2)
        memory, facts, padding = encode_inputs(model, values)
        prefix = values["input_ids"][:, :-1]
        altered = prefix.clone()
        altered[:, 2:] = torch.tensor([[8, 9], [4, 10]])
        original = model.decoder(prefix, memory, facts, padding)
        changed = model.decoder(altered, memory, facts, padding)
        torch.testing.assert_close(original[:, :2], changed[:, :2], atol=1e-6, rtol=1e-5)
        self.assertGreater((original[:, 2:] - changed[:, 2:]).abs().sum().item(), 0)
        last = model.decoder(prefix, memory, facts, padding, return_last_step_only=True)
        torch.testing.assert_close(last, original[:, -1:])

    def test_gate_and_adapter_gradients_after_two_updates_on_one_batch(self):
        model, values = make_model(), model_inputs()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**values)
            self.assertEqual(tuple(logits.shape), (2, 4, len(vocabulary())))
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), values["input_ids"][:, 1:].flatten(), ignore_index=0)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            for layer in model.decoder.layers:
                self.assertGreater(layer.gate_mlp[2].weight.grad.abs().sum().item(), 0)
                for p in [layer.gate_mlp[0].weight, layer.gate_img_norm.weight, layer.gate_fact_norm.weight]:
                    if update == 0:
                        self.assertEqual(p.grad.count_nonzero().item(), 0)
                    else:
                        self.assertGreater(p.grad.abs().sum().item(), 0)
            self.assertGreater(model.fact_adapter.ffn[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        with self.assertRaisesRegex(ValueError, "strictly between"):
            make_model(initial_gate=0.)
        self.assertAlmostEqual(make_model(initial_gate=.25).decoder.layers[0].gate_mlp[2].bias.item(), math.log(.25/.75), places=6)

    def test_beam_routing_matches_per_image_and_adapter_is_not_recomputed(self):
        model, values = make_model().eval(), model_inputs()
        with torch.no_grad(), patch.object(model.fact_adapter, "forward", wraps=model.fact_adapter.forward) as adapter:
            memory, facts, padding = encode_inputs(model, values)
            wrapper = _DualAttentionBeamDecoder(model.decoder, facts, padding, 3)
            self.assertEqual(tuple(wrapper.fact_memory.shape), (6, 2, 16))
            batched = beam_search(wrapper, memory, vocabulary(), 3, 6)
            individual = []
            for index in range(2):
                wrapper = _DualAttentionBeamDecoder(model.decoder, facts[index:index+1], padding[index:index+1], 3)
                individual.append(beam_search(wrapper, memory[index:index+1], vocabulary(), 3, 6))
            torch.testing.assert_close(batched, torch.cat(individual))
        self.assertEqual(adapter.call_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_one_batch_backward_and_search(self):
        model, values = make_model("cuda"), model_inputs("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**values)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), values["input_ids"][:, 1:].flatten(), ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            model.eval()
            memory, facts, padding = encode_inputs(model, values)
            wrapper = _DualAttentionBeamDecoder(model.decoder, facts, padding, 2)
            self.assertEqual(tuple(beam_search(wrapper, memory, vocabulary(), 2, 6).shape), (2, 6))


class DualAttentionScriptTests(unittest.TestCase):
    def test_one_batch_train_val_generate_checkpoint_and_resume(self):
        setup_reproducibility(42)
        train = importlib.import_module("script.fact_fusion_dual_attention.train")
        generate = importlib.import_module("script.fact_fusion_dual_attention.generate_captions")
        accelerator = Accelerator(cpu=True)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            root = Path(directory)
            context, features, split, test_split = [root / name for name in ("context.parquet", "features.h5", "train.parquet", "test.parquet")]
            rows = [
                dict(imgid=7, captions=["outside", "best"], itm_scores=[.1, .9],
                     object_texts=[["outside"], ["dog"]], attribute_texts=[[], ["red dog"]], triplet_texts=[[], ["dog on grass"]]),
                dict(imgid=8, captions=["tree"], itm_scores=[.5], object_texts=[["tree"]], attribute_texts=[[]], triplet_texts=[[]]),
            ]
            write_context(context, rows)
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            pd.DataFrame({"imgid": [7, 8], "tokens": [["red", "dog"], ["tree"]]}).to_parquet(split, index=False)
            pd.DataFrame({"imgid": [7, 8], "cocoid": [101, 202]}).to_parquet(test_split, index=False)
            common = dict(
                SEED=42, BATCH_SIZE=2, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=2,
                DROPOUT=0., MAX_LENGTH=6, FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2,
                VISUAL_ENCODER_MODEL="test/clip", TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                VOCAB_PATH=vocabulary().save(root / "vocab.json"), BEST_CHECKPOINT_PATH=root / "best.pth",
                RUN_MODE="fact_fusion_dual_attention_test",
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
            cleanup.callback(close_loader_stores, accelerator)
            train.main()
            checkpoint = torch.load(common["BEST_CHECKPOINT_PATH"], weights_only=True)
            self.assertEqual(checkpoint["global_step"], 1)
            state = checkpoint["model_state_dict"]
            self.assertIn("fact_adapter.ffn.0.weight", state)
            self.assertIn("decoder.layers.1.gate_fact_norm.weight", state)
            self.assertIn("decoder.layers.0.fact_attention.in_proj_weight", state)
            self.assertEqual(len(encoders[0].cache), 4)
            self.assertEqual(len(tokenizer.calls), 2)
            with patch("src.fact_fusion_dual_attention.inference_rag.beam_search", wraps=beam_search) as search:
                generate.main()
            self.assertEqual(search.call_count, 1)
            self.assertEqual(tuple(search.call_args.args[1].shape), (2, 3, 16))
            predictions = json.loads(generation["PREDICTIONS_PATH"].read_text(encoding="utf-8"))
            self.assertEqual([row["imgid"] for row in predictions], [7, 8])
            self.assertEqual([row["cocoid"] for row in predictions], [101, 202])
            self.assertEqual(len(encoders[1].cache), 4)
            self.assertEqual(len(tokenizer.calls), 4)
            with patch.object(train, "train_one_epoch") as training_step:
                train.main()  # Resume at the completed epoch: no second training batch.
            training_step.assert_not_called()
            for name, value in text_model.state_dict().items():
                torch.testing.assert_close(value, initial_text[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in text_model.parameters()))


if __name__ == "__main__":
    unittest.main()
