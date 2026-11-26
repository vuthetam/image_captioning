"""Offline CIDEr selection and one-batch integration for fact captioning pipelines."""

import copy
import importlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import h5py
import pandas as pd
import torch
from accelerate import Accelerator
from pycocoevalcap.cider.cider import Cider

from src.shared.checkpoint import load_checkpoint
from src.shared.checkpoint_selection import ValidationCider, load_epoch_weights, select_best_checkpoint
from test_fact_fusion_dual_attention import vocabulary
from test_fact_rag import TinyTokenizer, tiny_clip, write_context, close_loader_stores


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.accelerator = Accelerator(cpu=True)

    def test_cider_uses_all_references_and_rejects_missing_images(self):
        dataframe = pd.DataFrame({
            "imgid": [7, 8],
            "all_raws": [["A red dog.", "A dog on grass."], ["A green tree."]],
        })
        scorer = ValidationCider(dataframe)
        captions = {7: ["a", "red", "dog"], 8: ["a", "green", "tree"]}
        predictions = {imgid: [{"caption": " ".join(tokens)}] for imgid, tokens in captions.items()}
        expected, _ = Cider().compute_score(scorer.references, scorer.tokenizer.tokenize(predictions))
        self.assertAlmostEqual(scorer.compute(captions), expected)
        self.assertGreater(expected, 0)
        self.assertEqual(len(scorer.references[7]), 2)
        with self.assertRaisesRegex(ValueError, "every validation image"):
            scorer.compute({7: ["dog"]})
        with self.assertRaisesRegex(ValueError, "one row per image"):
            ValidationCider(pd.concat([dataframe, dataframe]))

    def test_selection_numeric_order_ties_and_incremental_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"epoch_{epoch:02d}.pth" for epoch in [10, 2, 1]]
            for path in paths:
                path.write_bytes(path.name.encode())
            scores_path, best_path = root / "checkpoint_scores.json", root / "best_checkpoint.pth"
            visited = []

            def evaluate(path):
                if visited:
                    saved = json.loads(scores_path.read_text())
                    self.assertEqual(len(saved["epochs"]), len(visited))
                epoch = int(path.stem.removeprefix("epoch_"))
                visited.append(epoch)
                return epoch, {1: .5, 2: .8, 10: .8}[epoch]

            report = select_best_checkpoint(paths, evaluate, best_path, scores_path,
                                            {"selection_metric": "CIDEr"}, self.accelerator)
            self.assertEqual(visited, [1, 2, 10])
            self.assertEqual(report["best_epoch"], 2)
            self.assertEqual(best_path.read_bytes(), (root / "epoch_02.pth").read_bytes())
            self.assertEqual(len(list(root.glob("*.json"))), 1)
            self.assertEqual(json.loads(scores_path.read_text()), report)
            with self.assertRaisesRegex(FileNotFoundError, "No epoch"):
                select_best_checkpoint([], evaluate, best_path, scores_path, {}, self.accelerator)

    def test_selection_retains_completed_scores_when_next_evaluation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "epoch_01.pth", root / "epoch_02.pth"]
            paths[0].write_bytes(b"first")
            paths[1].write_bytes(b"second")

            def evaluate(path):
                if path == paths[1]:
                    raise RuntimeError("interrupted evaluation")
                return 1, .9

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                select_best_checkpoint(paths, evaluate, root / "best.pth", root / "scores.json", {}, self.accelerator)
            saved = json.loads((root / "scores.json").read_text())
            self.assertEqual(saved["best_epoch"], 1)
            self.assertEqual(len(saved["epochs"]), 1)
            self.assertEqual((root / "best.pth").read_bytes(), b"first")

    def test_one_batch_train_select_and_existing_generate(self):
        model_names = {
            "fact_fusion": "FactRAGCaptioner",
            "fact_fusion_prefix": "FactFusionPrefixCaptioner",
            "fact_fusion_token_bias": "FactFusionTokenBiasCaptioner",
        }
        for pipeline in model_names:
            with self.subTest(pipeline=pipeline), tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
                root = Path(directory)
                train = importlib.import_module(f"script.{pipeline}.train_save_epochs")
                select = importlib.import_module(f"script.{pipeline}.select_checkpoint")
                generate = importlib.import_module(f"script.{pipeline}.generate_captions")
                context, features, split, grouped_path = [root / name for name in
                                                         ["context.parquet", "features.h5", "train.parquet", "grouped.parquet"]]
                write_context(context, [
                    dict(imgid=7, captions=["dog"], itm_scores=[.9], object_texts=[["dog"]],
                         attribute_texts=[["red dog"]], triplet_texts=[["dog on grass"]]),
                    dict(imgid=8, captions=["tree"], itm_scores=[.8], object_texts=[["tree"]],
                         attribute_texts=[[]], triplet_texts=[[]]),
                ])
                with h5py.File(features, "w") as handle:
                    handle["imgids"] = [7, 8]
                    handle["features"] = torch.randn(2, 4, 6).numpy()
                pd.DataFrame({"imgid": [7, 8], "tokens": [["red", "dog"], ["tree"]]}).to_parquet(split)
                pd.DataFrame({"imgid": [7, 8], "cocoid": [101, 202],
                              "all_raws": [["A red dog.", "A dog on grass."], ["A tree."]]}).to_parquet(grouped_path)
                vocab = vocabulary()
                if pipeline in {"fact_fusion_prefix", "fact_fusion_token_bias"}:
                    vocab._add_token("<sep>")
                    vocab.special_tokens.append("<sep>")
                vocab_path = vocab.save(root / "vocab.json")
                common = dict(
                    SEED=42, BATCH_SIZE=2, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=1,
                    DROPOUT=0., MAX_LENGTH=6, FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2,
                    VISUAL_ENCODER_MODEL="test/clip", TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                    VOCAB_PATH=vocab_path,
                )
                if pipeline == "fact_fusion_prefix":
                    common["MAX_FACT_PREFIX_LENGTH"] = 12
                training = dict(common, NUM_EPOCHS=1, RUN_CHECKPOINT_DIR=root / "checkpoints",
                                TRAIN_DF_PATH=split, TRAIN_I2T_PATH=context, TRAIN_VISUAL_FEATURES_PATH=features)
                selection = {name: value for name, value in common.items() if hasattr(select, name)}
                selection.update(BEAM_SIZE=2, RUN_CHECKPOINT_DIR=training["RUN_CHECKPOINT_DIR"],
                                 BEST_CHECKPOINT_PATH=root / "best.pth", VAL_GROUPED_DF_PATH=grouped_path,
                                 VAL_I2T_PATH=context, VAL_VISUAL_FEATURES_PATH=features)
                generation = dict(common, BEAM_SIZE=2, BEST_CHECKPOINT_PATH=root / "best.pth",
                                  PREDICTIONS_PATH=root / "predictions.json", TEST_DF_PATH=grouped_path,
                                  TEST_I2T_PATH=context, TEST_VISUAL_FEATURES_PATH=features)
                for module, settings in [(train, training), (select, selection), (generate, generation)]:
                    cleanup.enter_context(patch.multiple(module, **settings))
                    cleanup.enter_context(patch.object(module, "Accelerator", return_value=self.accelerator))
                text_model = tiny_clip()
                initial_text = copy.deepcopy(text_model.state_dict())
                cleanup.enter_context(patch("src.shared.facts.encoder.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()))
                cleanup.enter_context(patch("src.shared.facts.encoder.CLIPTextModelWithProjection.from_pretrained", return_value=text_model))
                cleanup.enter_context(patch("os._exit"))
                cleanup.callback(close_loader_stores, self.accelerator)
                with patch.object(train, "train_one_epoch", wraps=train.train_one_epoch) as train_epoch:
                    train.main()
                self.assertEqual(train_epoch.call_count, 1)
                self.assertEqual(len(train_epoch.call_args.args[1]), 1)
                path = training["RUN_CHECKPOINT_DIR"] / "epoch_01.pth"
                checkpoint = torch.load(path, weights_only=True)
                self.assertEqual(checkpoint["seed"], 42)
                self.assertEqual(checkpoint["vocab_state"], vocab.to_state())
                self.assertNotIn("optimizer_state_dict", checkpoint)
                self.assertNotIn("scheduler_state_dict", checkpoint)
                self.assertNotIn("best_val_loss", checkpoint)
                self.assertEqual(list(training["RUN_CHECKPOINT_DIR"].glob("last*")), [])
                if pipeline == "fact_fusion_token_bias":
                    self.assertEqual(checkpoint["model_config"]["model_kwargs"]["initial_alpha"], .3)
                    self.assertIn("decoder.raw_alpha", checkpoint["model_state_dict"])
                    self.assertIn("decoder.token_gate_mlp.0.weight", checkpoint["model_state_dict"])
                    trained_model = self.accelerator.unwrap_model(train_epoch.call_args.args[0])
                    self.assertIsNotNone(trained_model.decoder.raw_alpha.grad)
                    self.assertTrue(torch.isfinite(trained_model.decoder.raw_alpha.grad).all())
                with self.assertRaisesRegex(FileExistsError, "already exist"):
                    train.main()

                checkpoint["epoch"] = 2
                torch.save(checkpoint, path.with_name("epoch_02.pth"))
                with patch.object(select.ValidationCider, "compute", side_effect=[.7, .9]) as score:
                    select.main()
                self.assertEqual(score.call_count, 2)
                for call in score.call_args_list:
                    self.assertEqual(set(call.args[0]), {7, 8})
                report = json.loads((training["RUN_CHECKPOINT_DIR"] / "checkpoint_scores.json").read_text())
                self.assertEqual(report["best_epoch"], 2)
                self.assertEqual([row["CIDEr"] for row in report["epochs"]], [.7, .9])
                self.assertEqual(torch.load(root / "best.pth", weights_only=True)["epoch"], 2)
                # The existing checkpoint loader and generation entry point accept the selected file.
                model_type = getattr(generate, model_names[pipeline])
                model = model_type(**checkpoint["model_config"]["model_kwargs"])
                self.assertEqual(load_checkpoint(root / "best.pth", model)[0], 2)
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    load_epoch_weights(path, model, vocab, {})
                with self.assertRaisesRegex(ValueError, "Vocabulary differs"):
                    different_vocab = type(vocab)(freq_threshold=999)
                    load_epoch_weights(path, model, different_vocab, checkpoint["model_config"])
                generate.main()
                predictions = json.loads((root / "predictions.json").read_text())
                self.assertEqual([row["imgid"] for row in predictions], [7, 8])
                self.assertEqual([row["cocoid"] for row in predictions], [101, 202])
                for name, value in text_model.state_dict().items():
                    torch.testing.assert_close(value, initial_text[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
