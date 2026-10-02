import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tái sử dụng PositionalEncoding từ shared
from src.shared.decoder import PositionalEncoding

class DualCrossAttnDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 1. Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Dual Cross Attention
        self.img_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rag_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Dynamic Gate (Fusion)
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 0.0)
        
        # Add & Norm 1 lần duy nhất cho dòng chảy chính (Bảo toàn phương sai 1.0)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 3. Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, image_memory, rag_memory,
                tgt_mask=None, tgt_key_padding_mask=None, rag_key_padding_mask=None):
        # Sub-layer 1: Self-Attention (Causal) - Sử dụng is_causal=True để kích hoạt FlashAttention
        attn_out, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=True
        )
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        
        # Sub-layer 2A: Image Cross-Attention
        attn_img, _ = self.img_cross_attn(query=tgt, key=image_memory, value=image_memory)
        
        # Sub-layer 2B: RAG Cross-Attention
        attn_rag, _ = self.rag_cross_attn(
            query=tgt, key=rag_memory, value=rag_memory,
            key_padding_mask=rag_key_padding_mask
        )
        
        # Sub-layer 2C: Dynamic Gate Fusion (Sử dụng dữ liệu gốc để không khuếch đại nhiễu)
        gate = torch.sigmoid(self.gate_linear(torch.cat([attn_img, attn_rag], dim=-1)))
        
        # Sub-layer 2D: Trộn dữ liệu gốc và Add & Norm 1 lần duy nhất cho dòng chảy chính
        fused_attn = gate * attn_img + (1 - gate) * attn_rag
        tgt = self.norm2(tgt + self.dropout2(fused_attn))
        
        # Sub-layer 3: FFN
        ffn_out = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


class DualCrossAttnDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.1, max_length=100,
                 embedding_layer=None, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        
        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)
        self.pos_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DualCrossAttnDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.out_linear = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz, device):
        # Causal Mask (Boolean)
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, tgt_ids, image_memory, rag_memory,
                tgt_attention_mask=None, rag_attention_mask=None):
        seq_len = tgt_ids.size(1)
        
        # 1. Causal Mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len, tgt_ids.device)
        
        # 2. Padding Masks
        tgt_key_padding_mask = None
        if tgt_attention_mask is not None:
            tgt_key_padding_mask = (tgt_attention_mask == 0)
            
        rag_key_padding_mask = None
        if rag_attention_mask is not None:
            rag_key_padding_mask = (rag_attention_mask == 0)
        
        # 3. Embedding + Positional Encoding
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.pos_dropout(x)
        
        # 4. N Decoder Layers
        for layer in self.layers:
            x = layer(
                tgt=x,
                image_memory=image_memory,
                rag_memory=rag_memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                rag_key_padding_mask=rag_key_padding_mask
            )
            
        # 5. Project to Vocab
        return self.out_linear(x)
