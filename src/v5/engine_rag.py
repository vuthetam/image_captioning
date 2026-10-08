from typing import Iterable
import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm
from src.shared.utils import trainable_parameters

def _step_rag_v5(
    model: nn.Module,
    visual_inputs: torch.Tensor,
    rag_cls_tokens: torch.Tensor,
    rag_score: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_idx: int,
    include_cls_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Target shift left by 1
    target_ids = input_ids[:, 1:]
    
    logits = model(
        visual_inputs=visual_inputs,
        rag_inputs=rag_cls_tokens,
        rag_scores=rag_score,
        input_ids=input_ids,
        attention_mask=attention_mask,
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


def train_one_epoch_rag_v5(
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
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Training V5 RAG")

    for batch in iterator:
        visual_inputs, rag_cls_tokens, rag_score, input_ids, attention_mask = batch

        visual_inputs = visual_inputs.to(accelerator.device)
        rag_cls_tokens = rag_cls_tokens.to(accelerator.device)
        rag_score = rag_score.to(accelerator.device)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss_sum, num_tokens = _step_rag_v5(
                model, visual_inputs, rag_cls_tokens, rag_score, input_ids, attention_mask, pad_idx, include_cls_token
            )
     
        global_tokens = accelerator.reduce(num_tokens.detach().clone(), reduction="sum")
        normalized_loss = (loss_sum / global_tokens.clamp(min=1)) * accelerator.num_processes
        
        accelerator.backward(normalized_loss)
        
        accelerator.clip_grad_norm_(
            trainable_parameters(model),
            max_grad_norm,
        )

        optimizer.step()

        reduced_loss = accelerator.reduce(loss_sum.detach(), reduction="sum")
        
        total_loss += reduced_loss.item()
        total_tokens += global_tokens.item()
        
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_one_epoch_rag_v5(
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
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Evaluating V5 RAG")

    for batch in iterator:
        visual_inputs, rag_cls_tokens, rag_score, input_ids, attention_mask = batch

        visual_inputs = visual_inputs.to(accelerator.device)
        rag_cls_tokens = rag_cls_tokens.to(accelerator.device)
        rag_score = rag_score.to(accelerator.device)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        with accelerator.autocast():
            loss_sum, num_tokens = _step_rag_v5(
                model, visual_inputs, rag_cls_tokens, rag_score, input_ids, attention_mask, pad_idx, include_cls_token
            )

        reduced_loss = accelerator.reduce(loss_sum.detach(), reduction="sum")
        global_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss += reduced_loss.item()
        total_tokens += global_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)
