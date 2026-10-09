import torch
from torch import Tensor, nn

from src.v5.decoder import TransformerCaptionDecoderV5


class RagCaptionerV5(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nheads: int,
        nlayers: int,
        dropout: float,
        max_length: int,
        pad_idx: int,
        visual_feature_dim: int = 768,
    ):
        super().__init__()
        
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Projector dùng chung cho cả Ảnh gốc và Ảnh RAG (do cùng không gian vector B32)
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.decoder = TransformerCaptionDecoderV5(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nheads,
            num_layers=nlayers,
            dropout=dropout,
            max_length=max_length,
            pad_idx=pad_idx,
            embedding_layer=self.shared_embedding
        )
        
    def encode_memory(
        self, 
        visual_inputs: Tensor, 
        rag_inputs: Tensor, 
        include_cls_token: bool = False
    ) -> Tensor:
        """
        Tạo memory cho Decoder bằng cách nối (Concat) ảnh gốc và ảnh RAG.
        visual_inputs: [B, 50, D] hoặc [B, 49, D]
        rag_inputs: [B, K, D] (với K là các CLS tokens)
        """
        if not include_cls_token:
            # Bỏ đi CLS token của ảnh gốc (chỉ lấy patches)
            visual_inputs = visual_inputs[:, 1:, :]
            
        target_dtype = self.visual_projector[0].weight.dtype
        visual_inputs = visual_inputs.to(dtype=target_dtype)
        rag_inputs = rag_inputs.to(dtype=target_dtype)
        
        # 1. Đưa cả hai về không gian d_model 
        visual_features = self.visual_projector(visual_inputs) 
        rag_features = self.visual_projector(rag_inputs)      
        
        # 2. Nối chuỗi! (Early Concatenation / Visual Prompting)
        # Kết quả: [B, N + K, d_model]
        memory = torch.cat([visual_features, rag_features], dim=1)
        
        return memory

    def forward(
        self,
        visual_inputs: Tensor,
        rag_inputs: Tensor,
        rag_scores: Tensor, 
        input_ids: Tensor,
        attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """
        Luồng Training
        """
        memory = self.encode_memory(visual_inputs, rag_inputs, include_cls_token)
        
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=memory,
            attention_mask=decoder_attention_mask,
        )
        return logits

