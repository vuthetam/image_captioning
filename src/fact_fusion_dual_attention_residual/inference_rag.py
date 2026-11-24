"""Beam search with static adapted fact memory aligned to each image."""

from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.shared.inference import beam_search
from src.shared.vocabulary import Vocabulary


class _DualAttentionBeamDecoder(nn.Module):
    def __init__(self, decoder, fact_memory, fact_padding_mask, beam_size):
        super().__init__()
        self.decoder = decoder
        self.fact_memory = fact_memory.repeat_interleave(beam_size, dim=0)
        self.fact_padding_mask = fact_padding_mask.repeat_interleave(beam_size, dim=0)

    def forward(self, input_ids, memory, memory_key_padding_mask=None):
        # Search only reorders beams within an image; their fact memories are identical.
        return self.decoder(
            input_ids=input_ids, memory=memory, memory_key_padding_mask=memory_key_padding_mask,
            fact_memory=self.fact_memory, fact_padding_mask=self.fact_padding_mask,
            return_last_step_only=True,
        )


@torch.no_grad()
def generate_captions(model: nn.Module, dataloader: Iterable, vocab: Vocabulary, beam_size: int,
                      max_length: int, accelerator: Accelerator, fact_encoder: CachedCLIPFactEncoder,
                      length_penalty: float = 1.0, show_progress: bool = False,
                      include_cls_token: bool = False) -> dict[int, list[str]]:
    model.eval()
    base_model = accelerator.unwrap_model(model)
    captions = {}
    iterator = tqdm(
        dataloader, disable=not (show_progress and accelerator.is_local_main_process),
        leave=False, desc="Generating Fact Fusion Dual Attention Residual",
    )
    for visual_inputs, facts, attr_triplet_fact_indices, attr_triplet_fact_mask, image_ids in iterator:
        visual_inputs = visual_inputs.to(accelerator.device)
        image_ids = image_ids.to(accelerator.device)
        attr_triplet_fact_indices = attr_triplet_fact_indices.to(accelerator.device)
        attr_triplet_fact_mask = attr_triplet_fact_mask.to(accelerator.device)
        fact_features, fact_mask = fact_encoder.encode(facts)
        with accelerator.autocast():
            memory, fact_memory, fact_padding_mask = base_model.encode_memory(
                visual_inputs, fact_features, fact_mask, attr_triplet_fact_indices, attr_triplet_fact_mask,
                include_cls_token=include_cls_token,
            )
            beam_decoder = _DualAttentionBeamDecoder(base_model.decoder, fact_memory, fact_padding_mask, beam_size)
            sequences = beam_search(beam_decoder, memory, vocab, beam_size, max_length, length_penalty)
        gathered_sequences = accelerator.gather_for_metrics(sequences)
        gathered_ids = accelerator.gather_for_metrics(image_ids)
        if accelerator.is_main_process:
            for image_id, token_ids in zip(gathered_ids.tolist(), gathered_sequences.tolist()):
                captions[int(image_id)] = vocab.decode(token_ids, skip_special_tokens=True)
    return captions
