# V4 Baseline Architecture Design & Decision Log

## 1. Understanding Summary
- **Mục tiêu:** Xây dựng V4 Model làm Baseline (Thuần túy Ảnh -> Chữ, Không RAG) để phục vụ việc chuyển đổi sang đánh giá `clip-vit-base-patch32`.
- **Định hướng:** V4 sẽ kế thừa sự đơn giản của V1, nhưng được trang bị các kỹ thuật tối ưu hiện đại nhất rút ra từ quá trình phát triển V3.

## 2. Decision Log (Các Quyết Định Kiến Trúc)

### Quyết định 1: Dependency Injection cho Text Embedding (Future-Proofing)
- **Thiết kế:** `nn.Embedding` sẽ được khởi tạo ở cấp độ Model (`BaselineCaptionerV4`) dưới dạng `self.shared_embedding`. Sau đó nó được truyền (inject) vào Decoder qua tham số.
- **Lý do:** Tăng tính linh hoạt và khả năng mở rộng. Nếu trong tương lai (V5) ta cắm thêm một `TransformerEncoder` khác để đọc Text, ta có thể dễ dàng cho chúng dùng chung (Share) cái Embedding này mà không cần sửa code của Decoder.

### Quyết định 2: Tối ưu Causal Mask (Kích hoạt FlashAttention)
- **Thiết kế:** Thay vì dùng Float Mask (`-inf`), ta dùng Boolean Mask (`True/False`) dạng tam giác trên (upper triangular).
- **Lý do:** Kích hoạt tính năng ngầm định của PyTorch 2.x. Khi PyTorch nhận diện được hình dáng Causal Boolean Mask, nó sẽ tự động bypass luồng tính toán Python chậm chạp và gọi trực tiếp thư viện C++ (FlashAttention / Memory Efficient Attention) thông qua cờ `is_causal=True` ở backend, giúp tăng tốc độ train và giảm RAM đáng kể.

### Quyết định 3: Tích hợp Visual Projector
- **Thiết kế:** Không dùng class `VisualProjector` rời rạc. Mã nguồn của `nn.Linear` và `nn.LayerNorm` sẽ được nhúng thẳng vào trong class Model chính.
- **Lý do:** Tránh phân mảnh mã nguồn, dễ dàng kiểm soát quá trình phóng chiều (project) đặc trưng ảnh.

### Quyết định 4: Chuẩn hóa DDP Loss (Chống lệch Gradient)
- **Thiết kế:** Trong `engine.py`, Loss sẽ được tính tổng (`reduction='sum'`), sau đó chia cho TỔNG SỐ LƯỢNG TOKEN thật (đã loại bỏ Padding) của TẤT CẢ các GPU (`global_tokens`), và cuối cùng nhân lại cho `num_processes`.
- **Lý do:** Khắc phục lỗi sai số toán học cực kỳ nguy hiểm của cơ chế DDP (Distributed Data Parallel). DDP tự động chia trung bình Gradient theo số lượng GPU. Nếu ta chia Loss cục bộ ở từng GPU theo số lượng token cục bộ, trọng số sẽ bị lệch. Công thức trên triệt tiêu hoàn toàn sai số này.
- **Fix Memory Leak:** Hàm `_step_v4` tuyệt đối KHÔNG trả về `logits` để chống phình to RAM qua mỗi mẻ batch.

---
## 3. Implementation Handoff (Kế Hoạch Viết Code)
1. **`src/v4/decoder.py`**: Viết lại `TransformerCaptionDecoder` hỗ trợ nhận `embedding_layer` và tạo Boolean Mask.
2. **`src/v4/models/baseline.py`**: Viết class `BaselineCaptionerV4` tích hợp Shared Embedding và Visual Projector.
3. **`src/v4/engine.py`**: Triển khai công thức DDP Loss bá đạo trên.
4. **`script/v4/train_baseline.py`**: Viết script train, tích hợp vá lỗi `os._exit(0)` cho Kaggle và truyền `visual_feature_dim` động từ file `.h5`.

