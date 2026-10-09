# Thứ tự chạy script

Chạy các lệnh từ thư mục gốc của project. Cần có dataset MS COCO và cấu hình đường dẫn trong `.env` hoặc dùng giá trị mặc định trong `src/config.py`.

## 1. Chuẩn bị dữ liệu

```bash
python script/preprocess_dfs.py
```

Script đọc `dataset_coco.json`, tách dữ liệu thành train/validation/test và tạo:

- `artifacts/splits/train_df.parquet`
- `artifacts/splits/val_df.parquet`
- `artifacts/splits/test_df.parquet`

## 2. Tạo vocabulary

```bash
python script/build_vocab.py
```

Script đọc `train_df.parquet`, tạo vocabulary từ cột `tokens` với ngưỡng tần suất mặc định là `5`, rồi ghi:

```text
artifacts/vocab.json
```

## 3. Xây dựng Knowledge Base cho RAG

```bash
python script/shared/build_kb.py
```

Script dùng caption của tập train để tạo text embedding bằng `RETRIEVAL_ENCODER_MODEL`, chuẩn hóa vector và lưu:

- `artifacts/<retrieval-model-id>/kb/kb_text_index.faiss`
- `artifacts/<retrieval-model-id>/kb/kb_text_metadata.parquet`

## 4. Retrieve context RAG

```bash
python script/shared/retrieve_rag_contexts.py
```

Chạy `python script/shared/extract_image_embeddings.py` trước để tạo image embeddings
cho train/val/test bằng `RETRIEVAL_ENCODER_MODEL`. Script truy hồi đọc các embedding
đã lưu từ HDF5, chuẩn hóa L2 và tìm caption tương tự trong text FAISS index trên CPU.
Script không cần ảnh gốc, visual features, tải CLIP hay chạy Accelerate, và lưu tối đa
`8` context cho mỗi ảnh vào:

- `artifacts/<retrieval-model-id>/rag_contexts/train_rag_contexts.parquet`
- `artifacts/<retrieval-model-id>/rag_contexts/val_rag_contexts.parquet`
- `artifacts/<retrieval-model-id>/rag_contexts/test_rag_contexts.parquet`

Đầu vào gồm `train_image_embeddings.h5`, `val_image_embeddings.h5`,
`test_image_embeddings.h5` trong `IMAGE_EMBEDDINGS_DIR`, cùng `kb_text_index.faiss`
và `kb_text_metadata.parquet` trong thư mục KB của cùng model. Mỗi file HDF5 có
`imgids [N]` và `features [N, D]` đã qua CLIP projection. Caption thuộc chính ảnh
truy vấn được loại khỏi kết quả. Trong tên thư mục model, `/` được thay bằng `-`
(ví dụ `openai-clip-vit-large-patch14`).

Parquet đầu ra có `imgid` của ảnh truy vấn và `retrieval_imgids` là danh sách ID ảnh
của từng caption truy hồi, cùng thứ tự với `captions`, `tokens`, `objects`, `relations`
và `retrieval_scores`. Một ID có thể xuất hiện nhiều lần nếu lấy nhiều caption của ảnh đó.

## 5. Train và generate V6 text RAG

V6 giữ kiến trúc text RAG của V1: các caption truy hồi được encode bằng embedding
dùng chung với decoder và một Transformer encoder, sau đó nối với visual memory.
Visual projector của V6 dùng `Linear → LayerNorm` để chuẩn hóa token ảnh trước khi nối.
V6 đọc cột `tokens` đã tách sẵn từ các file RAG context hiện tại, nối các caption
bằng `<eos>` mà không tokenize lại từ `captions`, và chuẩn hóa training loss
theo tổng số target token hợp lệ trên tất cả process trước khi backward.

Đặt một `RUN_MODE` riêng trong `.env` (ví dụ `v6_rag_l14`) để checkpoint và kết quả
không dùng chung với các phiên bản trước, rồi chạy:

```bash
accelerate launch script/v6/train_rag.py
accelerate launch script/v6/generate_rag_captions.py
python script/shared/evaluate.py
```

V6 cần visual features HDF5 của ảnh gốc và các file `*_rag_contexts.parquet` được
tạo ở bước 4. Các tham số chính là `TOP_K_CAPTIONS`, `CTX_TOKENS_PER_CAPTION`
và `CTX_NLAYERS`. Mặc định `MAX_CTX_LENGTH` được tính bằng
`TOP_K_CAPTIONS * CTX_TOKENS_PER_CAPTION + TOP_K_CAPTIONS` (hiện tại `4 * 22 + 4 = 92`).
Context chèn `<eos>` giữa các caption; đây là giới hạn tổng chiều dài, không bắt buộc
mỗi caption phải đủ 22 token. Vẫn có thể đặt
trực tiếp `MAX_CTX_LENGTH` trong `.env` nếu muốn dùng một giới hạn cố định.
Caption mục tiêu và câu sinh vẫn dùng `MAX_LENGTH=40`, độc lập với chiều dài context.

## 6. Train và generate V7 kết hợp ảnh và caption

V7 chọn độc lập các ảnh liên quan tốt nhất và các caption tốt nhất từ hai file
`*_related_images.parquet` và `*_rag_contexts.parquet` hiện có. Không yêu cầu caption
phải thuộc các ảnh liên quan đã chọn, và không cần gộp hai knowledge base.
Nếu chưa có kết quả truy hồi ảnh, tạo image KB bằng
`python script/shared/build_image_kb.py`, rồi chạy `python script/v5/build_related_images.py`.

Memory đưa vào decoder có thứ tự:

```text
[patch tokens ảnh gốc, CLS tokens ảnh liên quan, caption tokens qua text encoder]
```

Hai nhánh ảnh dùng chung projector `Linear → LayerNorm`. V7 có text encoder, decoder và hàm
encode context riêng trong `src/v7`, không import các phiên bản trước;
word embedding được dùng chung giữa context và decoder.
Padding mask chỉ che các vị trí padding của caption context. Mặc định bỏ CLS của
ảnh gốc; có thể bật qua `include_cls_token` khi gọi model, engine hoặc inference.

Đặt `RUN_MODE=v7_rag_l14` trong `.env`, rồi chạy:

```bash
accelerate launch script/v7/train_rag.py
accelerate launch script/v7/generate_rag_captions.py
python script/shared/evaluate.py
```

Script yêu cầu `RUN_MODE` bắt đầu bằng `v7` để tách checkpoint khỏi các phiên bản cũ.
`TOP_K_RAG_IMAGES` điều khiển số CLS ảnh liên quan, còn `TOP_K_CAPTIONS` điều khiển
số caption. Hai giá trị độc lập và không được vượt số kết quả có trong parquet.
Với visual features B/32, K ảnh = 4 và context length = 92, memory có
`49 + 4 + 92 = 145` vị trí, tính cả padding context.

Cần visual features của ảnh gốc theo từng split và `train_visual_features.h5`
để đọc CLS ảnh liên quan, kể cả khi validation hoặc generate trên tập test.
Các visual features phải được trích bằng cùng `VISUAL_ENCODER_MODEL`;
image embeddings dùng cho retrieval không thay thế được visual features này.
Training loss được chuẩn hóa theo tổng số target token hợp lệ trên tất cả process
trước khi backward như V6.

## 7. Train, generate va evaluate bang visual features

Sau khi da co ca ba file H5, train va sinh caption khong can doc anh hay tai
CLIP vision encoder nua:

```bash
RUN_MODE=baseline_precomputed /home/tam/Link\ to\ workspace/ML/.venv/bin/accelerate launch script/train_precomputed_features.py
RUN_MODE=baseline_precomputed /home/tam/Link\ to\ workspace/ML/.venv/bin/accelerate launch script/generate_precomputed_captions.py
```

Hai lenh phai dung cung `RUN_MODE` de dung chung checkpoint. `RUN_MODE` moi
tranh resume nham checkpoint cu duoc train theo duong online-encoder. Feature
files can co format `imgids` va `features` nhu mo ta o `ARTIFACTS.md`; code se
kiem tra moi `imgid` trong split deu co feature truoc khi train.

Cuoi cung chay `evaluate.ipynb` de tinh BLEU, METEOR, ROUGE-L, CIDEr va SPICE
tu prediction JSON. Notebook nay khong encode anh; phan hien thi mau van can
`IMAGES_PATH` neu muon xem anh goc.

## Cấu hình đường dẫn chính

Các biến được định nghĩa trong `src/config.py`:

| Biến | Mặc định |
| --- | --- |
| `DATASET_COCO_PATH` | `dataset/mscoco/dataset_coco.json` |
| `IMAGES_PATH` | `dataset/mscoco/images` |
| `ARTIFACTS_DIR` | `artifacts` |
| `VISUAL_FEATURES_DIR` | `artifacts/visual_features` |
| `CHECKPOINTS_DIR` | `checkpoints` |
| `RUN_MODE` | `baseline` |

Có thể ghi đè bằng biến môi trường hoặc file `.env`.
