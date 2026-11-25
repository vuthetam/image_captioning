"""Offline prefix-mask, loss and beam checks with one-batch script integration."""

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
from torch import nn
from accelerate import Accelerator

from src.fact_fusion_prefix.dataset import encode_prefix, RAGFeatureCaptionDataset, RAGFeatureDataset, collate_fact_batch
from src.fact_fusion_prefix.decoder import PrefixCaptionDecoder
from src.fact_fusion_prefix.engine_rag import _step
from src.fact_fusion_prefix.beam_search import beam_search, pack_prefix_and_caption
from src.fact_fusion_prefix.model import FactFusionPrefixCaptioner
from script.fact_fusion_prefix.build_vocab import build_prefix_vocabulary
from src.shared.decoder import TransformerCaptionDecoder
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.shared.utils import setup_reproducibility
from src.shared.vocabulary import Vocabulary
from test_fact_fusion_dual_attention import vocabulary
from test_fact_rag import TinyTokenizer, tiny_clip, write_context, close_loader_stores


def prefix_vocab():
    vocab = vocabulary()
    vocab.stoi['<sep>'] = len(vocab.itos)
    vocab.itos.append('<sep>')
    vocab.special_tokens.append('<sep>')
    return vocab


def decoder(vocab):
    return PrefixCaptionDecoder(len(vocab), 16, 2, 2, 0., 30, vocab.pad_idx()).eval()


class PrefixTests(unittest.TestCase):
    def setUp(self):
        setup_reproducibility(42)
        torch.set_num_threads(1)
        self.vocab = prefix_vocab()

    def test_vocab_preserves_ids_persistence_and_base_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base, target = Path(directory)/'base.json', Path(directory)/'prefix.json'
            vocabulary().save(base)
            original = base.read_bytes()
            extended = build_prefix_vocabulary(base, target)
            self.assertEqual(extended.itos[:-1], vocabulary().itos)
            self.assertEqual(extended.itos[-1], '<sep>')
            self.assertEqual(base.read_bytes(), original)
            self.assertEqual(build_prefix_vocabulary(base, target).to_state(), extended.to_state())
            with self.assertRaises(ValueError):
                build_prefix_vocabulary(base, base)

    def test_prefix_budget_whole_facts_tokenization_and_empty(self):
        v = self.vocab
        ids = encode_prefix(['red-dog', 'woman on bicycle', 'tree'], v, 6)
        self.assertEqual(ids, [v.stoi['red'], v.stoi['dog'], v.sos_idx()])
        self.assertEqual(encode_prefix([], v, 1), [v.sos_idx()])
        self.assertEqual(encode_prefix(['impossible long first fact', 'dog'], v, 3), [v.sos_idx()])
        self.assertEqual(encode_prefix(['unknown'], v, 3), [v.unk_idx(), v.sos_idx()])
        self.assertNotIn(v.pad_idx(), ids)

    def test_ranking_dedup_subset_and_compact_batch(self):
        from src.shared.facts.dataset import load_rag_fact_lookup
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context, features = root/'context.parquet', root/'features.h5'
            write_context(context, [
                dict(imgid=7,captions=['first','best'],itm_scores=[.1,.9],
                     object_texts=[['tree'],['dog','red dog']],
                     attribute_texts=[[],[' red   dog ']],triplet_texts=[[],['dog on grass','red dog']]),
                dict(imgid=8,captions=['tree'],itm_scores=[.5],object_texts=[['tree']],
                     attribute_texts=[[]],triplet_texts=[[]]),
            ])
            with h5py.File(features,'w') as handle:
                handle['imgids'] = [7,8]
                handle['features'] = torch.randn(2,4,6).numpy()
            frame = pd.DataFrame(dict(imgid=[7,8],tokens=[['red','dog'],['tree']]))
            dataset = RAGFeatureCaptionDataset(frame,self.vocab,features,context,6,1,30,'itm')
            clip = RAGFeatureDataset(frame,self.vocab,features,context,1,30,'clip')
            try:
                self.assertEqual(dataset.fact_lookup,load_rag_fact_lookup(context,1,'itm'))
                self.assertEqual(self.vocab.decode(dataset.prefix_lookup[7]),
                                 ['red','dog','<sep>','dog','on','grass','<sos>'])
                self.assertEqual(clip.prefix_lookup[7],[self.vocab.sos_idx()])
                batch = collate_fact_batch([dataset[0],dataset[1]])
                self.assertEqual(batch[3].tolist(),[7,1])
                self.assertEqual(batch[1][0,6].item(),self.vocab.sos_idx())
                self.assertTrue(batch[2][0].all())
                self.assertFalse(batch[2][1,3:].any())
                self.assertNotIn(self.vocab.pad_idx(),batch[1][0].tolist())
            finally:
                for item in (dataset,clip):
                    if item.feature_store._h5_file is not None:
                        item.feature_store._h5_file.close()

    def test_mask_causality_bidirectional_prefix_and_batch_padding(self):
        v, model = self.vocab, decoder(self.vocab)
        inputs = torch.tensor([[4,5,1,4,5,6], [1,5,6,0,0,0]])
        lengths = torch.tensor([3,1])
        valid = torch.tensor([[1,1,1,1,1,1], [1,1,1,0,0,0]])
        memory = torch.randn(2,3,16)
        mask = model.prefix_mask(6, lengths, inputs.device).view(2,2,6,6)
        self.assertFalse(mask[0,0,:3,:3].any())
        self.assertTrue(mask[0,0,:3,3:].all())
        with torch.no_grad():
            expected = model(inputs,memory,lengths,valid)
            future = inputs.clone()
            future[0,4:] = torch.tensor([8,9])
            changed = model(future,memory,lengths,valid)
            torch.testing.assert_close(expected[0,:4],changed[0,:4])
            later_fact = inputs.clone()
            later_fact[0,1] = 9
            changed = model(later_fact,memory,lengths,valid)
            self.assertGreater((expected[0,0]-changed[0,0]).abs().sum().item(),0)
            single = model(inputs[1:2,:3],memory[1:2],lengths[1:2],valid[1:2,:3])
            torch.testing.assert_close(expected[1,:3],single[0],atol=1e-6,rtol=1e-5)
            self.assertTrue(torch.isfinite(expected).all())

    def test_sos_only_matches_standard_decoder(self):
        v, model = self.vocab, decoder(self.vocab)
        base = TransformerCaptionDecoder(len(v),16,2,2,0.,30,v.pad_idx()).eval()
        base.load_state_dict(model.state_dict())
        inputs, memory = torch.tensor([[1,4,5],[1,6,0]]), torch.randn(2,3,16)
        attention = inputs.ne(0)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs,memory,torch.ones(2,dtype=torch.long),attention),
                                       base(inputs,memory,attention),atol=1e-6,rtol=1e-5)

    def test_loss_alignment_and_real_target_count(self):
        v = self.vocab
        ids = torch.tensor([[4,5,1,4,5,2], [1,5,2,0,0,0]])
        valid = torch.tensor([[1,1,1,1,1,1],[1,1,1,0,0,0]])
        logits = torch.randn(2,5,len(v),requires_grad=True)
        model = unittest.mock.Mock(return_value=logits)
        encoder = unittest.mock.Mock()
        encoder.encode.return_value = (torch.zeros(2,1,8), torch.ones(2,1,dtype=torch.bool))
        loss,count = _step(model,(torch.zeros(2,3,6),ids,valid,torch.tensor([3,1]),[['dog'],['tree']]),encoder,0,'cpu')
        targets = torch.tensor([[0,0,4,5,2],[5,2,0,0,0]])
        expected = nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=0,reduction='sum')
        torch.testing.assert_close(loss,expected)
        self.assertEqual(count.item(),5)
        loss.backward()
        self.assertEqual(logits.grad[0,:2].count_nonzero().item(),0)
        self.assertGreater(logits.grad[0,2:].abs().sum().item(),0)

    def test_beam_variable_prefixes_match_individual_and_no_sep(self):
        v, model = self.vocab, decoder(self.vocab)
        memory = torch.randn(2,3,16)
        prefixes = torch.tensor([[4,5,1],[1,0,0]])
        lengths = torch.tensor([3,1])
        packed,mask,total = pack_prefix_and_caption(prefixes,lengths,torch.tensor([[1,6],[1,7]]),0)
        self.assertEqual(packed.tolist(),[[4,5,1,6],[1,7,0,0]])
        self.assertEqual(total.tolist(),[4,2])
        with torch.no_grad():
            # SEP would dominate without the explicit beam constraint.
            model.out_fc.bias[v.stoi['<sep>']] = 30
        actual = beam_search(model,memory,v,prefixes,lengths,3,6)
        expected = torch.cat([beam_search(model,memory[i:i+1],v,prefixes[i:i+1,:lengths[i]],lengths[i:i+1],3,6)
                              for i in range(2)])
        self.assertEqual(actual.tolist(),expected.tolist())
        self.assertFalse(actual.eq(v.stoi['<sep>']).any())
        self.assertEqual(tuple(actual.shape),(2,6))

    def test_beam_length_penalty_and_eos_match_original(self):
        from src.shared.inference import beam_search as original_search
        v = self.vocab

        class ScriptedDecoder(nn.Module):
            def forward(self, input_ids, memory, prefix_lengths=None, **kwargs):
                # Same next-token distribution regardless of the supplied prefix.
                logits = torch.full((*input_ids.shape,len(v)),-20.,device=input_ids.device)
                logits[...,v.stoi['dog']] = 0.
                logits[...,v.eos_idx()] = -.2
                logits[...,v.stoi['<sep>']] = float('-inf')
                return logits

        model, memory = ScriptedDecoder(), torch.zeros(2,1,16)
        prefixes, lengths = torch.tensor([[4,5,1],[1,0,0]]),torch.tensor([3,1])
        for penalty in (0.,1.,2.):
            actual = beam_search(model,memory,v,prefixes,lengths,3,6,penalty)
            expected = original_search(model,memory,v,3,6,penalty)
            self.assertEqual(actual.tolist(),expected.tolist())

    def test_fusion_equivalence_and_one_batch_backward(self):
        from src.fact_fusion.models import FactRAGCaptioner
        v = self.vocab
        model = FactFusionPrefixCaptioner(len(v),16,2,2,0.,6,0,6,8,10)
        base = FactRAGCaptioner(len(v),16,2,2,0.,6,0,6,8).eval()
        for name in ('visual_projector','fact_projector','fusion'):
            getattr(base,name).load_state_dict(getattr(model,name).state_dict())
        visual,facts,mask = torch.randn(2,4,6),torch.randn(2,2,8),torch.ones(2,2,dtype=torch.bool)
        torch.testing.assert_close(model.encode_memory(visual,facts,mask),base.encode_memory(visual,facts,mask))
        inputs = torch.tensor([[4,5,1,4,5,2],[1,6,2,0,0,0]])
        output = model(visual,inputs,inputs.ne(0),torch.tensor([3,1]),facts,mask)
        loss = output[:,2:,4].sum()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_fp16_one_batch_backward_and_beam(self):
        v = self.vocab
        model = FactFusionPrefixCaptioner(len(v),16,2,2,0.,6,0,6,8,10).cuda()
        visual, facts = torch.randn(2,4,6,device='cuda'), torch.randn(2,2,8,device='cuda')
        mask = torch.ones(2,2,dtype=torch.bool,device='cuda')
        ids = torch.tensor([[4,5,1,4,5,2],[1,6,2,0,0,0]],device='cuda')
        lengths = torch.tensor([3,1],device='cuda')
        targets = torch.tensor([[0,0,4,5,2],[6,2,0,0,0]],device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            output = model(visual,ids,ids.ne(0),lengths,facts,mask)
            loss = nn.functional.cross_entropy(output.flatten(0,1),targets.flatten(),ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            memory = model.encode_memory(visual,facts,mask)
            prefixes = torch.tensor([[4,5,1],[1,0,0]],device='cuda')
            generated = beam_search(model.decoder,memory,v,prefixes,lengths,2,6)
        self.assertEqual(tuple(generated.shape),(2,6))

class PrefixScriptTests(unittest.TestCase):
    def test_one_batch_train_val_generate_checkpoint_and_resume(self):
        setup_reproducibility(42)
        train = importlib.import_module("script.fact_fusion_prefix.train")
        generate = importlib.import_module("script.fact_fusion_prefix.generate_captions")
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
            base_vocab_path = vocabulary().save(root / "vocab.json")
            common = dict(
                SEED=42, BATCH_SIZE=2, NUM_WORKERS=0, DMODEL=16, NHEADS=2, NLAYERS=2,
                DROPOUT=0., MAX_LENGTH=6, FACT_FUSION_NLAYERS=1, FACT_TEXT_BATCH_SIZE=2,
                VISUAL_ENCODER_MODEL="test/clip", TEXT_RAG_CAPTION_SORT_BY="itm", TOP_K_CAPTIONS=1,
                VOCAB_PATH=root / "prefix_vocab.json", MAX_FACT_PREFIX_LENGTH=12, BEST_CHECKPOINT_PATH=root / "best.pth",
                RUN_MODE="fact_fusion_prefix_test",
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
            with self.assertRaisesRegex(FileNotFoundError, "build_vocab.py"):
                train.main()
            builder = importlib.import_module("script.fact_fusion_prefix.build_vocab")
            builder.main(["--vocab-path", str(base_vocab_path),
                          "--output-path", str(common["VOCAB_PATH"])])
            train.main()
            checkpoint = torch.load(common["BEST_CHECKPOINT_PATH"], weights_only=True)
            self.assertEqual(checkpoint["global_step"], 1)
            state = checkpoint["model_state_dict"]
            self.assertIn("decoder.decoder.layers.0.multihead_attn.in_proj_weight", state)
            self.assertEqual(state["decoder.embedding.weight"].shape[0], len(vocabulary()) + 1)
            self.assertEqual(len(encoders[0].cache), 4)
            self.assertEqual(len(tokenizer.calls), 2)
            with patch("src.fact_fusion_prefix.inference_rag.beam_search", wraps=beam_search) as search:
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
