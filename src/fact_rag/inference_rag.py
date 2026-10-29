from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.fact_rag.encoder import CachedCLIPFactEncoder
from src.shared.inference import beam_search
from src.shared.vocabulary import Vocabulary


@torch.no_grad()
def generate_captions(
    model: nn.Module, dataloader: Iterable, vocab: Vocabulary, beam_size: int,
    max_length: int, accelerator: Accelerator, fact_encoder: CachedCLIPFactEncoder,
    length_penalty: float = 1.0, show_progress: bool = False,
    include_cls_token: bool = False,
) -> dict[int, list[str]]:
    model.eval()
    base_model = accelerator.unwrap_model(model)
    captions = {}
    iterator = tqdm(dataloader, disable=not (show_progress and accelerator.is_local_main_process),
                    leave=False, desc="Generating Fact RAG")
    for visual_inputs, facts, image_ids in iterator:
        visual_inputs = visual_inputs.to(accelerator.device)
        image_ids = image_ids.to(accelerator.device)
        fact_features, fact_mask = fact_encoder.encode(facts)
        with accelerator.autocast():
            memory = base_model.encode_memory(
                visual_inputs, fact_features, fact_mask,
                include_cls_token=include_cls_token,
            )
            sequences = beam_search(
                base_model.decoder, memory, vocab, beam_size, max_length, length_penalty,
            )
        gathered_sequences = accelerator.gather_for_metrics(sequences)
        gathered_ids = accelerator.gather_for_metrics(image_ids)
        if accelerator.is_main_process:
            for image_id, token_ids in zip(gathered_ids.tolist(), gathered_sequences.tolist()):
                captions[int(image_id)] = vocab.decode(token_ids, skip_special_tokens=True)
    return captions
