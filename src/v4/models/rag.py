import torch
from torch import Tensor, nn

from src.v4.decoder import TransformerCaptionDecoderV4


class RagFusionBlock(nn.Module):
    def __init__(self, d_model: int, nheads: int, dropout: float = 0.1):
        super().__init__()
        
        # --- 1. RAG Self-Attention Block ---
        # Giúp các mảnh RAG patches giao tiếp, sắp xếp lại thông tin nội bộ
        self.rag_context_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True
        )

        # --- 2. V & RAG Cross-Attention Block ---
        # Dung hợp thông tin: Ảnh gốc (Query) đi tìm thông tin bổ sung từ RAG (Key/Value)
        self.fusion_cross_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        self.fusion_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # --- 3. Outer Residual ---
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, visual_features: Tensor, rag_features: Tensor) -> Tensor:
        """
        visual_features (V): Đặc trưng ảnh gốc [Batch, N, d_model]
        rag_features (RAG): RAG Tensor [Batch, N, d_model]
        """
        # GIAI ĐOẠN 1: RAG Self-Attention (S-Attn)
        encoded_rag = self.rag_context_encoder(rag_features)
        
        # GIAI ĐOẠN 2: Cross-Attention (C-Attn)
        fusion_attn_out, _ = self.fusion_cross_attn(
            query=visual_features, 
            key=encoded_rag, 
            value=encoded_rag
        )
        attn_residual_out = self.cross_attn_norm(visual_features + self.dropout(fusion_attn_out))
        
        ffn_out = self.fusion_ffn(attn_residual_out)
        fused_features = self.ffn_norm(attn_residual_out + self.dropout(ffn_out))
        
        # GIAI ĐOẠN 3: Outer Residual
        # Cộng feature đã dung hợp RAG trở lại làm phần bổ trợ cho visual feature gốc
        enhanced_visual_features = self.output_norm(visual_features + fused_features)
        
        return enhanced_visual_features


class RagCaptionerV4(nn.Module):
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
        
        # Projector dùng chung cho cả Ảnh gốc và Ảnh RAG
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Khối lai ghép RAG
        self.rag_fusion = RagFusionBlock(d_model=d_model, nheads=nheads, dropout=dropout)
        
        self.decoder = TransformerCaptionDecoderV4(
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
        Luồng Sinh Từ (Inference/Evaluate):
        Chỉ trích xuất đặc trưng memory một lần duy nhất.
        
        visual_inputs: [B, N, D_vis]
        rag_inputs: [B, K, N, D_vis]
        """
        if not include_cls_token:
            visual_inputs = visual_inputs[:, 1:, :] # [B, 196, D_vis]
            rag_inputs = rag_inputs[:, :, 1:, :]    # [B, K, 196, D_vis]
            
        B, K, N, D_vis = rag_inputs.shape
        
        # --- 1. ONLINE PATCH RETRIEVAL TRÊN GPU ---
        # Flatten K và N để dùng cho Cross-Attention (Tích vô hướng)
        rag_flat = rag_inputs.reshape(B, K * N, D_vis) # [B, K*N, D_vis]
        
        # Chuẩn hoá L2 để nhân ma trận tương đương với Cosine Similarity
        q_norm = torch.nn.functional.normalize(visual_inputs, p=2, dim=-1) # [B, N, D_vis]
        k_norm = torch.nn.functional.normalize(rag_flat, p=2, dim=-1)      # [B, K*N, D_vis]
        
        # Tính độ tương đồng giữa các patch của ảnh gốc và TẤT CẢ các patch của K ảnh truy hồi
        similarity = torch.bmm(q_norm, k_norm.transpose(1, 2)) # [B, N, K*N]
        
        # Chọn ra patch có độ tương đồng cao nhất
        best_patch_indices = torch.argmax(similarity, dim=-1, keepdim=True) # [B, N, 1]
        
        # Expand index để gather feature (D_vis)
        best_patch_indices_expanded = best_patch_indices.expand(-1, -1, D_vis) # [B, N, D_vis]
        
        # Trích xuất các patch tốt nhất từ rag_flat để tạo thành RAG Tensor
        retrieved_rag_features = torch.gather(rag_flat, 1, best_patch_indices_expanded) # [B, N, D_vis]
        
        # --- 2. DUNG HỢP BỞI RAG FUSION BLOCK ---
        target_dtype = self.visual_projector[0].weight.dtype
        visual_inputs = visual_inputs.to(dtype=target_dtype)
        retrieved_rag_features = retrieved_rag_features.to(dtype=target_dtype)
        
        visual_features = self.visual_projector(visual_inputs)
        rag_features = self.visual_projector(retrieved_rag_features)
        
        fused_memory = self.rag_fusion(visual_features, rag_features)
        return fused_memory

    def forward(
        self,
        visual_inputs: Tensor,
        rag_inputs: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """
        Luồng Training (Teacher Forcing).
        visual_inputs: Đặc trưng ảnh gốc [B, N, D_vis]
        rag_inputs: Đặc trưng ảnh truy hồi [B, K, N, D_vis]
        """
        fused_memory = self.encode_memory(visual_inputs, rag_inputs, include_cls_token)
        
        # 3. Đưa vào Decoder
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=fused_memory,
            attention_mask=decoder_attention_mask,
        )
        return logits

