import torch
from torch import Tensor

from src.v7.models.rag import RAGCaptionerV7


class RAGCaptionerV7NoInputImage(RAGCaptionerV7):
    """Decode using only related-image CLS tokens and retrieved caption contexts."""

    def encode_memory(
        self,
        visual_inputs: Tensor,
        related_cls_tokens: Tensor,
        rag_input_ids: Tensor,
        rag_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> tuple[Tensor, Tensor]:
        # Keep the V7 call signature so training and inference can share their code.
        # Neither visual_inputs nor include_cls_token participates in this memory.
        target_dtype = self.visual_projector[0].weight.dtype
        related_memory = self.visual_projector(related_cls_tokens.to(dtype=target_dtype))
        context_memory = self.context_encoder(rag_input_ids, rag_attention_mask)
        memory = torch.cat([related_memory, context_memory], dim=1)

        related_padding_mask = torch.zeros(related_memory.shape[:2], dtype=torch.bool, device=memory.device)
        context_padding_mask = rag_attention_mask == 0
        memory_padding_mask = torch.cat([related_padding_mask, context_padding_mask], dim=1)
        return memory, memory_padding_mask
