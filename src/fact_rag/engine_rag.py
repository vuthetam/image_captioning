from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.fact_rag.encoder import CachedCLIPFactEncoder
from src.shared.utils import trainable_parameters


def _step(model, batch, fact_encoder, pad_idx, device, include_cls_token=False):
    visual_inputs, input_ids, attention_mask, facts = batch
    visual_inputs = visual_inputs.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    fact_features, fact_mask = fact_encoder.encode(facts)
    logits = model(
        visual_inputs=visual_inputs, input_ids=input_ids, attention_mask=attention_mask,
        fact_features=fact_features, fact_attention_mask=fact_mask,
        include_cls_token=include_cls_token,
    )
    targets = input_ids[:, 1:]
    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=pad_idx, reduction="sum",
    )
    return loss_sum, (targets != pad_idx).sum()


def train_one_epoch(
    model: nn.Module, dataloader: Iterable, optimizer, pad_idx: int,
    accelerator: Accelerator, fact_encoder: CachedCLIPFactEncoder,
    max_grad_norm: float = 1.0, show_progress: bool = False, scheduler=None,
    include_cls_token: bool = False,
) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0
    iterator = tqdm(dataloader, disable=not (show_progress and accelerator.is_local_main_process),
                    leave=False, desc="Training Fact RAG", miniters=5)
    for batch in iterator:
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss_sum, num_tokens = _step(
                model, batch, fact_encoder, pad_idx, accelerator.device, include_cls_token,
            )
        global_tokens = accelerator.reduce(num_tokens.detach().clone(), reduction="sum")
        normalized_loss = loss_sum / global_tokens.clamp(min=1) * accelerator.num_processes
        accelerator.backward(normalized_loss)
        accelerator.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += accelerator.reduce(loss_sum.detach(), reduction="sum").item()
        total_tokens += global_tokens.item()
        iterator.set_postfix(
            loss=total_loss / max(total_tokens, 1),
            cached_facts=len(fact_encoder.cache),
            refresh=False,
        )
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module, dataloader: Iterable, pad_idx: int, accelerator: Accelerator,
    fact_encoder: CachedCLIPFactEncoder, show_progress: bool = False,
    include_cls_token: bool = False,
) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    iterator = tqdm(dataloader, disable=not (show_progress and accelerator.is_local_main_process),
                    leave=False, desc="Evaluating Fact RAG", miniters=5)
    for batch in iterator:
        with accelerator.autocast():
            loss_sum, num_tokens = _step(
                model, batch, fact_encoder, pad_idx, accelerator.device, include_cls_token,
            )
        total_loss += accelerator.reduce(loss_sum.detach(), reduction="sum").item()
        total_tokens += accelerator.reduce(num_tokens.detach(), reduction="sum").item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1), refresh=False)
    return total_loss / max(total_tokens, 1)
