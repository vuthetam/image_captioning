from __future__ import annotations

from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.shared.vocabulary import Vocabulary
from src.shared.inference import beam_search

@torch.no_grad()
def generate_captions_rag_v5(
    model: nn.Module,
    dataloader: Iterable,
    vocab: Vocabulary,
    beam_size: int,
    max_length: int,
    accelerator: Accelerator,
    length_penalty: float = 1.0,
    show_progress: bool = False,
    include_cls_token: bool = False,
) -> dict[int, list[str]]:
    """Generate captions for every image in *dataloader* using batched beam search with RAG features.

    Args:
        dataloader: prepared DataLoader. Must yield (visual_inputs, rag_inputs, image_ids).
    """
    model.eval()
    base_model = accelerator.unwrap_model(model)

    all_captions: dict[int, list[str]] = {}
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Generating V5 RAG")

    for batch in iterator:
        visual_inputs, rag_inputs, image_ids = batch
        
        visual_inputs = visual_inputs.to(accelerator.device)
        rag_inputs = rag_inputs.to(accelerator.device)
        image_ids = image_ids.to(accelerator.device)

        with accelerator.autocast():
            memory = base_model.encode_memory(visual_inputs, rag_inputs, include_cls_token)
            
            mem_mask = None

            sequences = beam_search(
                base_model.decoder, memory, vocab, beam_size, max_length, length_penalty, memory_key_padding_mask=mem_mask
            )

        # Gather both generated sequences and their corresponding IDs
        gathered_seqs = accelerator.gather_for_metrics(sequences)
        gathered_ids = accelerator.gather_for_metrics(image_ids)

        if accelerator.is_main_process:
            for imgid_tensor, token_ids in zip(gathered_ids, gathered_seqs.tolist()):
                imgid = int(imgid_tensor.item())
                tokens = vocab.decode(token_ids, skip_special_tokens=True)
                tokens = [t for t in tokens if t != vocab.pad_token]
                all_captions[imgid] = tokens

    return all_captions

