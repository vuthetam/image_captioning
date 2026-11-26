"""Segment assignment, zero-init equivalence and one-batch learning checks."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.fact_fusion_prefix.model import FactFusionPrefixCaptioner
from src.fact_fusion_prefix.beam_search import beam_search as prefix_beam_search
from src.fact_fusion_prefix_segment.beam_search import beam_search
from src.fact_fusion_prefix_segment.decoder import SegmentPrefixCaptionDecoder
from src.fact_fusion_prefix_segment.engine_rag import _step
from src.fact_fusion_prefix_segment.model import FactFusionPrefixSegmentCaptioner
from src.shared.utils import setup_reproducibility
from test_fact_fusion_prefix import prefix_vocab


class PrefixSegmentTests(unittest.TestCase):
    def setUp(self):
        setup_reproducibility(42)
        torch.set_num_threads(1)
        self.vocab = prefix_vocab()

    def make_models(self):
        args = (len(self.vocab), 16, 2, 2, 0., 6, self.vocab.pad_idx(), 6, 8, 12)
        original = FactFusionPrefixCaptioner(*args).eval()
        segment = FactFusionPrefixSegmentCaptioner(*args).eval()
        loaded = segment.load_state_dict(original.state_dict(), strict=False)
        self.assertEqual(loaded.missing_keys, ["decoder.segment_embedding.weight"])
        self.assertEqual(loaded.unexpected_keys, [])
        return original, segment

    def batch(self, device="cpu"):
        # First sample has facts; second has only SOS before its caption.
        ids = torch.tensor([[4, 5, 1, 4, 5, 2], [1, 6, 2, 0, 0, 0]], device=device)
        lengths = torch.tensor([3, 1], device=device)
        visual = torch.randn(2, 4, 6, device=device)
        facts = torch.randn(2, 2, 8, device=device)
        fact_mask = torch.tensor([[True, True], [True, False]], device=device)
        return ids, lengths, visual, facts, fact_mask

    def test_zero_initialization_parameter_count_and_sos_boundary(self):
        original, model = self.make_models()
        self.assertEqual(model.decoder.segment_embedding.weight.shape, (2, 16))
        self.assertEqual(model.decoder.segment_embedding.weight.count_nonzero().item(), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in original.parameters()), 32)
        ids, lengths, visual, facts, fact_mask = self.batch()
        with patch.object(model.decoder.segment_embedding, "forward", wraps=model.decoder.segment_embedding.forward) as embed:
            with torch.no_grad():
                model(visual, ids, ids.ne(0), lengths, facts, fact_mask)
        # SOS at index 2/0 belongs to caption segment, including SOS-only prefixes.
        self.assertEqual(embed.call_args.args[0].tolist(), [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])

    def test_segment_vectors_are_added_to_the_correct_positions(self):
        model = SegmentPrefixCaptionDecoder(len(self.vocab), 16, 2, 1, 0., 12, 0).eval()
        ids = torch.tensor([[4, self.vocab.sep_idx(), 5, 1, 6]])
        lengths = torch.tensor([4])
        captured = []
        handle = model.decoder.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs["tgt"].detach().clone()), with_kwargs=True)
        try:
            with torch.no_grad():
                model(ids, torch.zeros(1, 2, 16), lengths)
                model.segment_embedding.weight[0].fill_(.2)
                model.segment_embedding.weight[1].fill_(.7)
                model(ids, torch.zeros(1, 2, 16), lengths)
        finally:
            handle.remove()
        expected = torch.tensor([.2, .2, .2, .7, .7]).view(1, 5, 1).expand(1, 5, 16)
        torch.testing.assert_close(captured[1] - captured[0], expected)

    def test_zero_init_matches_original_fusion_logits_and_beam(self):
        original, model = self.make_models()
        ids, lengths, visual, facts, fact_mask = self.batch()
        with torch.no_grad():
            memory = model.encode_memory(visual, facts, fact_mask)
            original_memory = original.encode_memory(visual, facts, fact_mask)
            torch.testing.assert_close(memory, original_memory, rtol=0, atol=0)
            torch.testing.assert_close(model(visual, ids, ids.ne(0), lengths, facts, fact_mask),
                                       original(visual, ids, ids.ne(0), lengths, facts, fact_mask), rtol=0, atol=0)
            prefixes = torch.tensor([[4, 5, 1], [1, 0, 0]])
            for beam_size in (1, 3):
                actual = beam_search(model.decoder, memory, self.vocab, prefixes, lengths, beam_size, 6)
                expected = prefix_beam_search(original.decoder, original_memory, self.vocab, prefixes, lengths, beam_size, 6)
                self.assertEqual(actual.tolist(), expected.tolist())

    def test_caption_loss_updates_both_segment_vectors(self):
        _, model = self.make_models()
        model.train()
        ids, lengths, visual, facts, fact_mask = self.batch()
        encoder = unittest.mock.Mock()
        encoder.encode.return_value = (facts, fact_mask)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, count = _step(model, (visual, ids, ids.ne(0), lengths, [["red dog"], ["tree"]]), encoder, 0, "cpu")
        self.assertEqual(count.item(), 5)
        (loss / count).backward()
        gradient = model.decoder.segment_embedding.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(gradient.abs().sum(dim=1).gt(0).all())
        optimizer.step()
        self.assertTrue(model.decoder.segment_embedding.weight.abs().sum(dim=1).gt(0).all())

    def test_mask_remains_bidirectional_for_prefix_and_causal_for_caption(self):
        _, model = self.make_models()
        ids, lengths, visual, facts, fact_mask = self.batch()
        with torch.no_grad():
            model.decoder.segment_embedding.weight.normal_(std=.02)
            expected = model(visual, ids, ids.ne(0), lengths, facts, fact_mask)
            future = ids.clone()
            future[0, 4] = 7
            actual = model(visual, future, ids.ne(0), lengths, facts, fact_mask)
            torch.testing.assert_close(expected[0, :4], actual[0, :4], rtol=0, atol=0)
            later_fact = ids.clone()
            later_fact[0, 1] = 7
            changed = model(visual, later_fact, ids.ne(0), lengths, facts, fact_mask)
            self.assertGreater((expected[0, 0] - changed[0, 0]).abs().sum().item(), 0)
            mask = model.decoder.prefix_mask(5, lengths, ids.device).view(2, 2, 5, 5)
            self.assertFalse(mask[0, 0, :3, :3].any())
            self.assertTrue(mask[0, 0, :3, 3:].all())
            single = model(visual[1:], ids[1:, :3], ids[1:, :3].ne(0), lengths[1:], facts[1:], fact_mask[1:])
            torch.testing.assert_close(expected[1, :2], single[0], atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_one_batch_backward_and_beam(self):
        _, model = self.make_models()
        model = model.cuda().train()
        ids, lengths, visual, facts, fact_mask = self.batch("cuda")
        targets = torch.tensor([[0, 0, 4, 5, 2], [6, 2, 0, 0, 0]], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(visual, ids, ids.ne(0), lengths, facts, fact_mask)
            loss = nn.functional.cross_entropy(output.flatten(0, 1), targets.flatten(), ignore_index=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(model.decoder.segment_embedding.weight.grad.abs().sum(dim=1).gt(0).all())
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            memory = model.encode_memory(visual, facts, fact_mask)
            prefixes = torch.tensor([[4, 5, 1], [1, 0, 0]], device="cuda")
            generated = beam_search(model.decoder, memory, self.vocab, prefixes, lengths, 2, 6)
        self.assertEqual(tuple(generated.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
