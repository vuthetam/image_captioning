from __future__ import annotations

from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.shared.utils import trainable_parameters


def _step(
    model: nn.Module,
    visual_inputs: torch.Tensor,
    related_cls_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rag_input_ids: torch.Tensor,
    rag_attention_mask: torch.Tensor,
    pad_idx: int,
    include_cls_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_ids = input_ids[:, 1:]
    logits = model(
        visual_inputs=visual_inputs,
        related_cls_tokens=related_cls_tokens,
        input_ids=input_ids,
        attention_mask=attention_mask,
        rag_input_ids=rag_input_ids,
        rag_attention_mask=rag_attention_mask,
        include_cls_token=include_cls_token,
    )
    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    num_tokens = (target_ids != pad_idx).sum()
    return loss_sum, num_tokens


def _move_batch(batch, device: torch.device):
    return tuple(tensor.to(device) for tensor in batch)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    pad_idx: int,
    accelerator: Accelerator,
    max_grad_norm: float = 1.0,
    show_progress: bool = False,
    include_cls_token: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(
        dataloader,
        disable=not (show_progress and accelerator.is_local_main_process),
        leave=False,
        desc="Training V7 RAG",
    )

    for batch in iterator:
        visual_inputs, related_cls_tokens, input_ids, attention_mask, rag_input_ids, rag_attention_mask = _move_batch(batch, accelerator.device)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss_sum, num_tokens = _step(
                model=model,
                visual_inputs=visual_inputs,
                related_cls_tokens=related_cls_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                rag_input_ids=rag_input_ids,
                rag_attention_mask=rag_attention_mask,
                pad_idx=pad_idx,
                include_cls_token=include_cls_token,
            )

        global_tokens = accelerator.reduce(num_tokens.detach().clone(), reduction="sum")
        normalized_loss = (loss_sum / global_tokens.clamp(min=1)) * accelerator.num_processes
        accelerator.backward(normalized_loss)
        accelerator.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        optimizer.step()

        reduced_loss = accelerator.reduce(loss_sum.detach(), reduction="sum")
        total_loss += reduced_loss.item()
        total_tokens += global_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    pad_idx: int,
    accelerator: Accelerator,
    show_progress: bool = False,
    include_cls_token: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(
        dataloader,
        disable=not (show_progress and accelerator.is_local_main_process),
        leave=False,
        desc="Evaluating V7 RAG",
    )

    for batch in iterator:
        visual_inputs, related_cls_tokens, input_ids, attention_mask, rag_input_ids, rag_attention_mask = _move_batch(batch, accelerator.device)
        with accelerator.autocast():
            loss_sum, num_tokens = _step(
                model=model,
                visual_inputs=visual_inputs,
                related_cls_tokens=related_cls_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                rag_input_ids=rag_input_ids,
                rag_attention_mask=rag_attention_mask,
                pad_idx=pad_idx,
                include_cls_token=include_cls_token,
            )

        reduced_loss = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")
        total_loss += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)
