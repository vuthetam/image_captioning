"""Exercise image/text retrieval artifacts with a tiny real OpenCLIP, offline."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import faiss
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch


@unittest.skipUnless(importlib.util.find_spec("open_clip"), "Install requirements-retrieval.txt")
class RetrievalEncoderTests(unittest.TestCase):
    def test_checkpoint_choices_produce_matching_image_and_text_artifacts(self):
        import open_clip
        from open_clip.transform import image_transform

        from script.shared import build_kb_i2t_index as builder
        from script.shared import extract_image_embeddings as extractor
        from src.shared import encoder as encoders

        get_tokenizer = open_clip.get_tokenizer
        cases = [
            ("hf-hub:timm/vit_base_patch32_clip_224.openai", 16),
            ("hf-hub:timm/vit_large_patch14_clip_224.openai", 24),
            ("hf-hub:apple/DFN2B-CLIP-ViT-L-14", 24),
            ("hf-hub:apple/DFN5B-CLIP-ViT-H-14", 32),
        ]
        for model_id, output_dim in cases:
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                Image.new("RGB", (48, 32), "red").save(root / "red.png")
                Image.new("RGB", (32, 48), "green").save(root / "green.png")
                captions = ["a red square", "a green square", "a red square on a wall " * 20]
                frame = pd.DataFrame({
                    "imgid": [10, 20, 10], "filepath": ["", "", ""],
                    "filename": ["red.png", "green.png", "red.png"], "caption": captions,
                })
                split_path = root / "split.parquet"
                frame.to_parquet(split_path, index=False)

                # Only checkpoint loading is replaced. The towers, transforms,
                # tokenizer, HDF5 writing and FAISS indexing are real.
                torch.manual_seed(42)
                model = open_clip.CLIP(
                    embed_dim=output_dim,
                    vision_cfg=dict(layers=1, width=64, head_width=32,
                                    patch_size=16, image_size=32),
                    text_cfg=dict(context_length=12, vocab_size=49408,
                                  width=64, heads=2, layers=1),
                    quick_gelu=True,
                )
                transform = image_transform(32, is_train=False)
                tokenizer = get_tokenizer("ViT-L-14", context_length=model.context_length)
                with (
                    patch.object(open_clip, "create_model_from_pretrained",
                                 return_value=(model, transform)) as load,
                    patch.object(open_clip, "get_tokenizer", return_value=tokenizer) as tokenize,
                    patch.object(encoders, "RETRIEVAL_OPENCLIP_MODEL_ID", model_id),
                    patch.object(extractor, "RETRIEVAL_OPENCLIP_MODEL_ID", model_id),
                    patch.object(extractor, "IMAGES_DIR", root),
                    patch.object(extractor, "NUM_WORKERS", 0),
                    patch.object(builder, "RETRIEVAL_OPENCLIP_MODEL_ID", model_id),
                    patch.object(builder, "KB_I2T_INDEX_BATCH_SIZE", 2),
                ):
                    encoder = encoders.CLIPImageEmbeddingEncoder(device="cpu")
                    self.assertEqual(encoder.output_dim, output_dim)
                    self.assertFalse(model.training)
                    self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                    image_path = root / "images.h5"
                    extractor.process_and_save(
                        split_path, image_path, encoder,
                        encoder.preprocess, torch.device("cpu"), batch_size=2,
                    )
                    index_path = builder.build_caption_index(
                        split_path, root / "captions.faiss", "cpu",
                    )
                    load.assert_called_with(
                        model_id, device="cpu", precision="fp32",
                    )
                    self.assertEqual(load.call_count, 2)
                    tokenize.assert_called_with(model_id, context_length=12)

                with h5py.File(image_path, "r") as stored:
                    self.assertEqual(stored.attrs["model_id"], model_id)
                    np.testing.assert_array_equal(stored["imgids"][:], [10, 20])
                    self.assertEqual(stored["features"].shape, (2, output_dim))
                    self.assertEqual(stored["features"].dtype, np.dtype("float16"))
                    image_features = stored["features"][:].astype(np.float32)

                with torch.inference_mode():
                    text_features = model.encode_text(tokenizer(captions)).float()
                    text_features = torch.nn.functional.normalize(text_features, dim=-1).numpy()
                index = faiss.read_index(str(index_path))
                self.assertEqual((index.ntotal, index.d), (3, output_dim))
                np.testing.assert_allclose(index.reconstruct_n(0, 3), text_features, atol=1e-6)
                faiss.normalize_L2(image_features)
                scores, indices = index.search(image_features, 3)
                expected = image_features @ text_features.T
                np.testing.assert_array_equal(indices, np.argsort(-expected, axis=1))
                np.testing.assert_allclose(scores, np.take_along_axis(expected, indices, axis=1),
                                           atol=1e-6)


if __name__ == "__main__":
    unittest.main()
