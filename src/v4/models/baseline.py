from torch import Tensor, nn
from src.v4.decoder import TransformerCaptionDecoderV4

class BaselineCaptionerV4(nn.Module):
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
        
        # Shared Embedding (Quyết định 1)
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Quyết định 3: Tích hợp Visual Projector (Dùng Sequential cho gọn)
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
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
        
    def forward(
        self,
        visual_inputs: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """
        Luồng Training (Teacher Forcing).
        visual_inputs: Đặc trưng ảnh đã trích xuất sẵn từ HDF5 [B, N, D_vis]
        """
        # Bỏ CLS token nếu được yêu cầu
        if not include_cls_token:
            visual_inputs = visual_inputs[:, 1:, :]
            
        # Ép kiểu cho an toàn khi dùng HDF5 float16 hoặc Mixed Precision
        visual_inputs = visual_inputs.to(dtype=self.visual_projector[0].weight.dtype)
        
        memory = self.visual_projector(visual_inputs)
        
        # Cắt token cuối của input_ids làm đầu vào cho decoder
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=memory,
            attention_mask=decoder_attention_mask,
        )
        return logits

