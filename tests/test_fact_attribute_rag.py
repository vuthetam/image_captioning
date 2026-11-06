"""Attribute-only supplementary context and one-batch script checks."""

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
import torch

from src.fact_attribute_rag.dataset import RAGFeatureDataset, load_attribute_context_lookup
from src.fact_token_rag.dataset import RAGFeatureDataset as FactTokenDataset
from src.shared.facts.encoder import CachedCLIPFactEncoder
from test_fact_rag import TinyTokenizer, close_loader_stores, tiny_clip, write_context
from test_fact_token_rag import vocabulary


def context_rows():
    return [
        dict(imgid=7, captions=["first", "best"], itm_scores=[.1, .9],
             object_texts=[["bus"], ["dog"]],
             attribute_texts=[["blue bus"], ["red dog", " red  dog "]],
             triplet_texts=[["bus on road"], ["dog on grass"]]),
        dict(imgid=8, captions=["tree"], itm_scores=[.5], object_texts=[["tree"]],
             attribute_texts=[[]], triplet_texts=[["tree on grass"]]),
    ]


class FactAttributeTests(unittest.TestCase):
    def test_sort_and_attribute_only_context_preserve_fusion_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, features = root / "contexts.parquet", root / "features.h5"
            write_context(context, context_rows())
            with h5py.File(features, "w") as handle:
                handle["imgids"] = [7, 8]
                handle["features"] = torch.randn(2, 4, 6).numpy()
            self.assertEqual(load_attribute_context_lookup(context, 1, "itm")[7], [["red dog", "red dog"]])
            self.assertEqual(load_attribute_context_lookup(context, 1, "clip")[7], [["blue bus"]])
            frame = pd.DataFrame({"imgid": [7, 8]})
            dataset = RAGFeatureDataset(frame, vocabulary(), features, context, 8, 1, "itm")
            token_dataset = FactTokenDataset(frame, vocabulary(), features, context, 8, 1, "itm")
            try:
                self.assertEqual(dataset.fact_lookup, token_dataset.fact_lookup)
                visual, facts, ids, mask, imgid = dataset[0]
                self.assertEqual(imgid, 7)
                self.assertEqual(facts, ["dog", "red dog", "dog on grass"])
                self.assertEqual(vocabulary().decode(ids[mask.bool()].tolist()), ["red", "dog", "<eos>"])
                _, facts, ids, mask, imgid = dataset[1]
                self.assertEqual(facts, ["tree", "tree on grass"])
                self.assertEqual(mask.sum().item(), 0)
            finally:
                for instance in (dataset, token_dataset):
                    if instance.feature_store._h5_file is not None:
                        instance.feature_store._h5_file.close()
                        instance.feature_store._h5_file = None

    def test_one_batch_train_val_generate_with_mixed_and_empty_attributes(self):
        train = importlib.import_module("script.fact_attribute_rag.train_rag")
        generate = importlib.import_module("script.fact_attribute_rag.generate_rag_captions")
        for empty in (False, True):
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
                accelerator = Accelerator(cpu=True)
                cleanup.callback(close_loader_stores, accelerator)
                root = Path(directory)
                context, features = root / "contexts.parquet", root / "features.h5"
                rows = context_rows()
                if empty:
                    rows[0]["attribute_texts"] = [[], []]
                write_context(context, rows)
                with h5py.File(features, "w") as handle:
                    handle["imgids"] = [7, 8]
                    handle["features"] = torch.randn(2, 4, 6).numpy()
                split, test_split = root / "split.parquet", root / "test.parquet"
                pd.DataFrame({"imgid": [7, 7, 8], "tokens": [["dog"], ["tree"], ["dog"]]}).to_parquet(split)
                pd.DataFrame({"imgid": [7, 8], "cocoid": [101, 202]}).to_parquet(test_split)
                common = dict(
                    BATCH_SIZE=3, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=1,
                    DROPOUT=0., MAX_LENGTH=6, CTX_NLAYERS=1,
                    FACT_CTX_TOKENS_PER_CAPTION=8, MAX_FACT_CTX_LENGTH=8,
                    FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2, VISUAL_ENCODER_MODEL="test/clip",
                    TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                    VOCAB_PATH=vocabulary().save(root / "vocab.json"),
                    BEST_CHECKPOINT_PATH=root / "best.pth", RUN_MODE="fact_attribute_test",
                )
                training = dict(common, NUM_EPOCHS=1, LAST_CHECKPOINT_PATH=root / "last.pth",
                                TRAIN_DF_PATH=split, VAL_DF_PATH=split, TRAIN_I2T_PATH=context, VAL_I2T_PATH=context,
                                TRAIN_VISUAL_FEATURES_PATH=features, VAL_VISUAL_FEATURES_PATH=features)
                generation = dict(common, BEAM_SIZE=2, PREDICTIONS_PATH=root / "predictions.json",
                                  TEST_DF_PATH=test_split, TEST_I2T_PATH=context, TEST_VISUAL_FEATURES_PATH=features)
                cleanup.enter_context(patch.multiple(train, **training))
                cleanup.enter_context(patch.multiple(generate, **generation))
                cleanup.enter_context(patch("src.shared.facts.encoder.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()))
                cleanup.enter_context(patch("src.shared.facts.encoder.CLIPTextModelWithProjection.from_pretrained", side_effect=lambda _: tiny_clip()))
                cleanup.enter_context(patch.object(train, "Accelerator", return_value=accelerator))
                cleanup.enter_context(patch.object(generate, "Accelerator", return_value=accelerator))
                train.main()
                checkpoint = torch.load(common["BEST_CHECKPOINT_PATH"], weights_only=True)
                self.assertEqual(checkpoint["global_step"], 1)
                self.assertTrue(torch.isfinite(torch.tensor(checkpoint["train_loss"])))
                generate.main()
                predictions = json.loads(generation["PREDICTIONS_PATH"].read_text(encoding="utf-8"))
                self.assertEqual([(row["imgid"], row["cocoid"]) for row in predictions], [(7, 101), (8, 202)])


if __name__ == "__main__":
    unittest.main()
