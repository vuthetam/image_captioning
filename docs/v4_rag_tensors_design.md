# Design Document: Pre-computing V4 RAG Tensors

## 1. Understanding Summary
- **Mục tiêu:** Tạo một script tiền xử lý (`build_rag_tensors.py`) để sinh ra các tensor RAG (Kích thước `[49 x 768]`) cho các tập Train, Val và Test.
- **Tại sao cần nó:** Để giải phóng vòng lặp huấn luyện (Training loop) khỏi việc truy cập ổ cứng ngẫu nhiên (Random I/O) và tìm kiếm FAISS. Giúp quá trình train V4 nhanh tương đương Baseline.
- **Đối tượng sử dụng:** Kiến trúc mô hình V4 (Image-to-Image RAG ở cấp độ Patch).
- **Ràng buộc chính:** Không làm tắc nghẽn ổ đĩa, đảm bảo không rò rỉ dữ liệu tập Validation vào Knowledge Base, tránh lỗi tràn số `NaN` của `float16`.

## 2. Assumptions
- **Dung lượng hệ thống:** Quá trình chạy script sẽ tiêu tốn khoảng ~9 GB RAM hệ thống (System RAM) do nạp toàn bộ đặc trưng gốc (`train_visual_features.h5` và `val/test`) lên RAM cùng lúc.
- **Thiết kế Offline:** Cố định giá trị `TOP_K_RAG_IMAGES = 4`. Việc thay đổi `TOP_K` yêu cầu chạy lại script này.
- **Kiểu dữ liệu gốc:** Các ma trận lưu trong file `.h5` gốc có định dạng `float16`.

## 3. Decision Log
| Vấn đề | Quyết định (Được chọn) | Lý do chọn | Các lựa chọn đã loại bỏ |
| :--- | :--- | :--- | :--- |
| **Quản lý RAM** | Nạp toàn bộ `train_features.h5` (8.5GB) và `val/test` lên RAM. | Tốc độ truy xuất mảng ngẫu nhiên (Random Access) trên RAM cực nhanh. Tối ưu hơn hẳn đọc từ ổ cứng. | Đọc trực tiếp từ file H5 (Quá chậm do Bottleneck IO). Đổi sang `.npy + mmap` (Cần đổi định dạng toàn project, phức tạp). |
| **Truy hồi FAISS** | Dùng vector 512-dim (từ `model.get_image_features()`) để tra cứu. | Đảm bảo tính nhất quán với `image_kb.faiss` đã build. Lấy được ảnh có ngữ nghĩa tương đồng nhất. | Dùng CLS token 768-dim (Đã build KB bằng 512-dim nên không đổi). |
| **So khớp Patch** | Khớp trên không gian 768-dim. | Bảo tồn chi tiết thị giác nguyên bản của ViT. | Ép qua lớp Projection 512-dim (Trọng số random sẽ làm hỏng kết quả). |
| **Tràn số `float16`** | Ép sang `float32` ngay khi khởi tạo Tensor, tính toán xong lưu H5 ở `float16`. | Hàm L2 Normalize sẽ nổ `NaN` nếu bình phương các giá trị của 768 chiều trên `float16`. Ép sang 32-bit đảm bảo an toàn toán học tuyệt đối. | Giữ nguyên `float16` (Sẽ crash toàn bộ script). |

## 4. Final Architecture (Luồng chạy)
1. **Khởi tạo:**
   - Nạp CLIP Model (`float16`).
   - Mở FAISS Index (Chỉ chứa ảnh Train).
   - Đọc các biến config từ `config.py`.
2. **Nạp RAM:** 
   - Đọc toàn bộ `train_visual_features.h5` (Vừa làm Query cho tập Train, vừa làm Ngân hàng Truy hồi cho cả 3 tập).
   - Đọc `val_visual_features.h5` và `test_visual_features.h5` (Chỉ dùng làm Query).
3. **Vòng lặp Xử lý (Cho mỗi tập Train/Val/Test):**
   - Đẩy ảnh gốc qua mô hình CLIP -> Vector `512-dim`.
   - Tìm FAISS ra `K` `imgid`. Loại bỏ ảnh trùng với chính nó.
   - Trích xuất patch `[49 x 768]` của Query và `[K*49 x 768]` của Retrieved (Ép thành `float32`).
   - Tính Cosine Similarity, dùng `argmax` nhặt 49 patch tốt nhất.
   - Đẩy RAG Tensor `[49 x 768]` (Dạng `float16`) vào file HDF5 mới.

---
**Trạng thái Script:** Đã hoàn thiện và lưu tại `script/v4/build_rag_tensors.py`. Sẵn sàng chạy.

