"""Caption selection checks using small local Parquet and HDF5 fixtures."""

from pathlib import Path
import tempfile
import unittest

import h5py
import pandas as pd

from src.shared.vocabulary import Vocabulary
from src.shared.dataset import load_rag_caption_lookup
from src.multimodal_rag.dataset import RAGFeatureDataset as MultimodalFeatureDataset
from src.multimodal_rag.dataset import RAGFeatureCaptionDataset as MultimodalTrainingDataset
from src.text_rag.dataset import (
    RAGFeatureCaptionDataset,
    RAGFeatureDataset,
    encode_rag_context,
)


class CaptionSortingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "contexts.parquet"

    def write_context(self, scores=None, tokens=None):
        row = {"imgid": 7, "tokens": tokens if tokens is not None else [["clip"], ["best"], ["tied"], ["last"]]}
        if scores is not None:
            row["itm_scores"] = scores
        pd.DataFrame([row]).to_parquet(self.path, index=False)

    def test_clip_option_retains_order_with_and_without_itm_scores(self):
        for scores in (None, [0.1, 0.9, 0.9, 0.8]):
            with self.subTest(scores=scores):
                self.write_context(scores)
                result = load_rag_caption_lookup(self.path, "clip")
                self.assertEqual([list(words) for words in result[7]],
                                 [["clip"], ["best"], ["tied"], ["last"]])

    def test_default_selects_itm_top_k_with_stable_ties_in_train_and_inference(self):
        self.write_context([0.1, 0.9, 0.9, 0.8])
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary([["clip", "best", "tied", "last", "target"]])
        features_path = self.path.with_suffix(".h5")
        with h5py.File(features_path, "w") as handle:
            handle["imgids"] = [7]
            handle["features"] = [[[1.0, 2.0], [3.0, 4.0]]]
        frame = pd.DataFrame([{"imgid": 7, "tokens": ["target"]}])
        common = dict(vocab=vocab, features_path=features_path,
                      rag_contexts_path=self.path, max_context_length=6,
                      top_k=2)
        training = RAGFeatureCaptionDataset(frame, max_length=5, **common)
        inference = RAGFeatureDataset(frame, **common)
        expected = [vocab.token_to_idx("best"), vocab.eos_idx(),
                    vocab.token_to_idx("tied"), *([vocab.pad_idx()] * 3)]
        try:
            train_sample, inference_sample = training[0], inference[0]
            self.assertEqual(train_sample[3].tolist(), expected)
            self.assertEqual(inference_sample[1].tolist(), expected)
            self.assertEqual(train_sample[4].tolist(), [1, 1, 1, 0, 0, 0])
            self.assertEqual(inference_sample[2].tolist(), [1, 1, 1, 0, 0, 0])
        finally:
            for dataset in (training, inference):
                if dataset.feature_store._h5_file is not None:
                    dataset.feature_store._h5_file.close()
        # Loading with ITM does not rewrite the stored retrieval order.
        self.assertEqual(list(load_rag_caption_lookup(self.path, "clip")[7][0]), ["clip"])

    def test_itm_requires_available_aligned_finite_scores(self):
        self.write_context()
        with self.assertRaisesRegex(ValueError, "itm_scores column"):
            load_rag_caption_lookup(self.path)
        invalid = ([0.5], [0.1, 0.2, float("nan"), 0.4],
                   [0.1, 0.2, float("inf"), 0.4], [0.1, None, 0.3, 0.4])
        for scores in invalid:
            with self.subTest(scores=scores):
                self.write_context(scores)
                with self.assertRaisesRegex(ValueError, "ITM scores"):
                    load_rag_caption_lookup(self.path, "itm")

    def test_empty_candidates_remain_a_padded_context(self):
        self.write_context(scores=[], tokens=[])
        context = load_rag_caption_lookup(self.path, "itm")[7]
        vocab = Vocabulary()
        ids, mask = encode_rag_context(context, vocab, max_context_length=4, top_k=2)
        self.assertEqual(ids.tolist(), [vocab.pad_idx()] * 4)
        self.assertEqual(mask.tolist(), [0] * 4)

    def test_unknown_sort_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "caption_sort_by"):
            load_rag_caption_lookup(self.path, "unknown")

    def test_multimodal_variants_preserve_clip_order_without_itm_scores(self):
        self.write_context()
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary([["clip", "best", "tied", "last"]])
        features_path = self.path.with_suffix(".h5")
        with h5py.File(features_path, "w") as handle:
            handle["imgids"] = [7, 8]
            handle["features"] = [[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]]
        related_path = self.path.parent / "related.parquet"
        pd.DataFrame([{"imgid": 7, "retrieved_imgids": [8]}]).to_parquet(related_path)
        for original_features in (features_path, None):
            with self.subTest(include_input_image=original_features is not None):
                dataset = MultimodalFeatureDataset(
                    df=pd.DataFrame([{"imgid": 7}]), vocab=vocab,
                    features_path=original_features, related_features_path=features_path,
                    rag_contexts_path=self.path, related_images_path=related_path,
                    max_context_length=6, top_k_captions=2, top_k_images=1,
                )
                try:
                    self.assertEqual(dataset[0][2].tolist(),
                                     [vocab.token_to_idx("clip"), vocab.eos_idx(),
                                      vocab.token_to_idx("best"), *([vocab.pad_idx()] * 3)])
                finally:
                    for store in (dataset.feature_store, dataset.related_store):
                        if store is not None and store._h5_file is not None:
                            store._h5_file.close()

    def test_multimodal_training_forwards_configured_itm_sort(self):
        self.write_context([0.1, 0.9, 0.9, 0.8])
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary([["clip", "best", "tied", "last"]])
        features_path = self.path.with_suffix(".h5")
        with h5py.File(features_path, "w") as handle:
            handle["imgids"] = [7, 8]
            handle["features"] = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
        related_path = self.path.parent / "related.parquet"
        pd.DataFrame([{"imgid": 7, "retrieved_imgids": [8]}]).to_parquet(related_path)
        for input_features in (features_path, None):
            dataset = MultimodalTrainingDataset(
                df=pd.DataFrame([{"imgid": 7, "tokens": ["clip"]}]), vocab=vocab,
                features_path=input_features, related_features_path=features_path,
                rag_contexts_path=self.path, related_images_path=related_path,
                max_length=5, max_context_length=6, top_k_captions=2, top_k_images=1,
                caption_sort_by="itm",
            )
            try:
                self.assertEqual(dataset[0][4].tolist(),
                                 [vocab.token_to_idx("best"), vocab.eos_idx(),
                                  vocab.token_to_idx("tied"), *([vocab.pad_idx()] * 3)])
            finally:
                for store in (dataset.feature_store, dataset.related_store):
                    if store is not None and store._h5_file is not None:
                        store._h5_file.close()


if __name__ == "__main__":
    unittest.main()
