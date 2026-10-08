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

Parquet đầu ra có `imgid` của ảnh truy vấn và `imgids` là danh sách ID ảnh
của từng caption truy hồi, cùng thứ tự với `captions`, `tokens`, `objects`, `relations`
và `retrieval_scores`. Một ID có thể xuất hiện nhiều lần nếu lấy nhiều caption của ảnh đó.

## 5. Train, generate va evaluate bang visual features

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
