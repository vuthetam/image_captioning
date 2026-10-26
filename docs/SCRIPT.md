# Thứ tự chạy script

Chạy các lệnh từ thư mục gốc của project. Cần có dataset MS COCO và cấu hình đường dẫn trong `.env` hoặc dùng giá trị mặc định trong `src/shared/config.py`.

Tên script sinh caption thống nhất theo dạng `generate_<loại>_captions.py`:

| Phiên bản | Script |
| --- | --- |
| Baseline | `script/baseline/generate_baseline_captions.py` |
| Image RAG | `script/image_rag/generate_rag_captions.py` |
| Text RAG | `script/text_rag/generate_rag_captions.py` |
| Multimodal RAG | `script/multimodal_rag/generate_rag_captions.py` |

Mã nguồn và script V1–V4 đã được loại bỏ. Baseline được tách riêng trong
`src/baseline` và `script/baseline`. Baseline, image RAG, text RAG và multimodal RAG vẫn dùng
`src/shared` và vocabulary `artifacts/vocab_v3.json`; giữ nguyên vocabulary hiện có
để tương thích với checkpoint.

V5 được đổi tên thành `image_rag`, V6 thành `text_rag`, V7 thành `multimodal_rag`
trong cả `src` và `script`. Thư mục artifacts/checkpoint và `RUN_MODE` hiện có được
giữ nguyên. Khi tiếp tục một lần chạy cũ, dùng đúng `RUN_MODE` cũ; tên mới trong
các ví dụ bên dưới dành cho lần chạy mới.

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
python script/shared/build_vocab.py
```

Script đọc `train_df.parquet`, tạo vocabulary từ cột `tokens` với ngưỡng tần suất mặc định là `5`, rồi ghi:

```text
artifacts/vocab_v3.json
```

## 3. Xây dựng Knowledge Base cho RAG

```bash
python script/shared/build_kb.py
```

Chuẩn bị thêm các dependency cho bước xây KB:

```bash
pip install -r requirements-kb.txt
```

Script chỉ dùng caption của tập **train**. FACTUAL Base bản `-id`
(`lizhuang144/flan-t5-base-VG-factual-sg-id`, hằng số trong `build_kb.py`)
chạy inference với `return_text=False`, giữ định danh entity và marker `v:`/`pv:`.
pySimpleNLG chạy thuần Python, không cần Java, chỉ hiện thực dạng phân từ của
động từ; danh từ được ghép trực tiếp để giữ nguyên dạng FACTUAL trả về.
FACTUAL được giải phóng khỏi GPU trước khi tải CLIP.

Metadata giữ `caption_id` (vị trí dòng trong train split), `imgid`, `filepath`,
`filename`, `caption` và `tokens` gốc, rồi bổ sung:

- `objects`: danh sách tên object, không có hậu tố ID.
- `attributes`: các struct `{object, attribute}`, bỏ qua số lượng trong graph.
- `triplets`: các struct `{subject, predicate, object, voice}`.
- `object_texts`, `attribute_texts`, `triplet_texts`: cụm linearized tương ứng.

Ví dụ graph `(man, v:ride, bicycle), (bicycle, is, red)` tạo các cụm
`man`, `bicycle`, `red bicycle`, `man riding bicycle`.
Quan hệ `pv:park on` được dựng thành `parked on`, không đổi vai subject/object.
Không tự thêm mạo từ, suy ra thì hay đổi số ít/số nhiều của danh từ.
Không lưu trường số lượng của graph vào metadata hay ghép vào keyword. Từ điển SimpleNLG
giúp chọn thuộc tính tính từ đứng trước danh từ; trạng thái như `made of wood`
đứng sau danh từ, không tự thêm động từ `be`.
Đây là linearization theo luật/lexicon, không đảm bảo ngữ pháp hay parsing chính
xác tuyệt đối. Cần kiểm tra mẫu thật trước khi dùng các cụm cho thí nghiệm truy hồi.

Cấu hình trong `src/shared/config.py`: đường dẫn train/KB
và `RETRIEVAL_ENCODER_MODEL`. Batch size được đặt cố định trong `build_kb.py`:
FACTUAL dùng 32, CLIP dùng 64.
Script tự chọn CUDA nếu có, xử lý toàn bộ train split và ghi vào đường dẫn KB.
Giao thức trích xuất cố định trong `PARSING_OPTIONS` ở `build_kb.py`:
5 beams, giới hạn input/output 128/256 token. Các giá trị này độc lập với
tham số sinh caption của mô hình chính.

Đầu vào quá dài, đầu ra chạm giới hạn, hoặc graph sai định dạng sẽ dừng để kiểm tra,
không âm thầm cắt hoặc bỏ đơn vị. Graph chỉ có entity riêng lẻ vẫn được giữ.

Metadata được ghi lần lượt theo batch vào một file Parquet tạm, không lưu file
chunk hay tiếp tục từ batch đã xử lý. Nếu bị ngắt, lần chạy sau xử lý lại từ đầu.
`kb_i2t_manifest.json` nằm cùng thư mục với metadata, giữ model commit đã chọn
để các lần chạy dùng cùng phiên bản. File chỉ lưu hai trường `model` và `revision`.
Trên Kaggle cần Internet khi tải checkpoint/package lần đầu.

Sau khi xây metadata, script tạo text embedding **của caption gốc** bằng
`RETRIEVAL_ENCODER_MODEL`, chuẩn hóa vector và lưu:

- `artifacts/<retrieval-model-id>/kb/kb_i2t_index.faiss`
- `artifacts/<retrieval-model-id>/kb/kb_i2t_metadata.parquet`

Index vẫn theo caption để tương thích với text RAG hiện tại. Các cụm được lưu
để kiểm tra và phục vụ bước xây index theo đơn vị sau này; model train vẫn đọc
tokens caption. File đích chỉ được thay khi hoàn thành cả metadata và index.
Không chạy nhiều builder cùng một thư mục output đồng thời.

Kiểm tra offline (không tải trọng số FACTUAL) bằng:

```bash
python -m unittest discover -s tests -v
```

## 4. Retrieve context RAG

```bash
python script/shared/retrieve_i2t_kb.py
```

Trên Kaggle T4 ×2, chia các ảnh cho hai process bằng Accelerate:

```bash
accelerate launch --multi_gpu --num_processes 2 script/shared/retrieve_i2t_kb.py
```

Chạy `python script/shared/extract_image_embeddings.py` trước để tạo image embeddings
cho train/val/test bằng `RETRIEVAL_ENCODER_MODEL`. Script truy hồi đọc các embedding
đã lưu từ HDF5, chuẩn hóa L2 và tìm caption tương tự trong text FAISS index trên CPU.
Sau khi loại caption thuộc chính ảnh truy vấn, script giữ tối đa `I2T_RETRIEVAL_TOP_K=10`
caption theo điểm CLIP và dùng BLIP ITM chấm từng cặp ảnh–caption. Kết quả lưu vào:

- `artifacts/<retrieval-model-id>/rag_contexts/train_i2t.parquet`
- `artifacts/<retrieval-model-id>/rag_contexts/val_i2t.parquet`
- `artifacts/<retrieval-model-id>/rag_contexts/test_i2t.parquet`

Đầu vào gồm `train_image_embeddings.h5`, `val_image_embeddings.h5`,
`test_image_embeddings.h5` trong `IMAGE_EMBEDDINGS_DIR`, cùng `kb_i2t_index.faiss`
và `kb_i2t_metadata.parquet` trong thư mục KB của cùng model. Mỗi file HDF5 có
`imgids [N]` và `features [N, D]` đã qua CLIP projection. Caption thuộc chính ảnh
truy vấn được loại khỏi kết quả. Trong tên thư mục model, `/` được thay bằng `-`
(ví dụ `openai-clip-vit-large-patch14`).

Parquet đầu ra có `imgid` của ảnh truy vấn và `retrieval_imgids` là danh sách ID ảnh
của từng caption truy hồi, cùng thứ tự với `captions`, `tokens`, `objects`
và `clip_scores`. Cột `itm_scores` chứa xác suất match của BLIP,
`softmax([non_match, match])[1]`, cùng thứ tự với các caption. Bước này bổ sung
điểm ITM; thứ tự caption vẫn theo CLIP. Một ID có thể xuất hiện nhiều lần nếu lấy nhiều caption của ảnh đó.
Với KB mới, kết quả còn giữ `attributes`, `triplets`,
`object_texts`, `attribute_texts`, `triplet_texts`, cùng thứ tự caption truy hồi.
Script vẫn đọc được KB cũ chưa có các cột bổ sung.

BLIP cần ảnh gốc tại `IMAGES_PATH / filepath / filename`. Đường dẫn được tra từ
`imgid`, `filepath`, `filename` trong `train_df.parquet`, `val_df.parquet` và
`test_df.parquet`; không dùng caption tham chiếu của validation/test để chấm điểm.
Các dependency cần cho bước này: `torch`, `transformers`, `accelerate`, `Pillow`,
`faiss-cpu`, `h5py`, `numpy`, `pandas`, `pyarrow`, `tqdm` và `python-dotenv`.

Cấu hình trong `src/shared/config.py`, có thể ghi đè bằng môi trường hoặc `.env`:

```dotenv
BLIP_ITM_MODEL_ID=Salesforce/blip-itm-large-coco
I2T_RETRIEVAL_TOP_K=10
BLIP_ITM_IMAGE_BATCH_SIZE=2
```

Có thể đổi sang `Salesforce/blip-itm-base-coco` mà không sửa code chấm điểm.
`BLIP_ITM_IMAGE_BATCH_SIZE` là số ảnh query chấm cùng lúc trên mỗi GPU, độc lập
với batch tìm kiếm FAISS. Tổng số cặp trong một batch ITM bằng tổng số caption
truy hồi thực tế của nhóm ảnh đó; ví dụ 2 ảnh, mỗi ảnh 10 caption thì có 20 cặp.
Batch cuối có thể ít ảnh hơn; ảnh không có caption nhận danh sách điểm rỗng.
Script dùng `PartialState()` và dtype mặc định của model, không bật autocast hay ép FP16.

Mỗi process load một BLIP trên `state.device`, encode nhóm ảnh có caption trong
một lượt và giữ toàn bộ CLS/patch features trên GPU. `repeat_interleave()` tạo
tensor đặc trưng theo đúng các cặp ảnh–caption để text encoder chấm điểm.
Điểm được tách lại theo từng ảnh và giữ thứ tự CLIP. Features được giải phóng
khi chấm xong nhóm ảnh; không lưu cache xuống đĩa. VRAM tăng theo tổng số cặp,
nên bắt đầu với 2 ảnh/GPU rồi đo tốc độ và VRAM trước khi tăng batch.

Các process xử lý những đoạn ảnh liên tiếp, ghi từng batch vào Parquet shard
tạm riêng. Rank 0 ghép shard theo thứ tự ảnh gốc trong HDF5 và chỉ thay file đích
khi ghi xong. Metadata của Parquet lưu ID checkpoint BLIP, revision nếu có,
encoder truy hồi và định nghĩa điểm ITM. Không chạy đồng thời nhiều lệnh truy hồi
ghi vào cùng các file output.

## 5. Train và generate text RAG

text RAG giữ kiến trúc text RAG của V1: các caption truy hồi được encode bằng embedding
dùng chung với decoder và một Transformer encoder, sau đó nối với visual memory.
Visual projector của text RAG dùng `Linear → LayerNorm` để chuẩn hóa token ảnh trước khi nối.
text RAG đọc cột `tokens` đã tách sẵn từ các file RAG context hiện tại, nối các caption
bằng `<eos>` mà không tokenize lại từ `captions`, và chuẩn hóa training loss
theo tổng số target token hợp lệ trên tất cả process trước khi backward.

Đặt một `RUN_MODE` riêng trong `.env` (ví dụ `text_rag_l14`) để checkpoint và kết quả
không dùng chung với các phiên bản trước, rồi chạy:

```bash
accelerate launch script/text_rag/train_rag.py
accelerate launch script/text_rag/generate_rag_captions.py
python script/shared/evaluate.py
```

Train và generate baseline, image RAG, text RAG và multimodal RAG (cả biến thể `--no-input-image`)
đều gọi `setup_reproducibility(42)` trong `src/shared/utils.py`.
Hàm đặt seed bằng `set_seed(42)` và bật chế độ tính toán xác định với
`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark=False`
và `torch.backends.cudnn.deterministic=True`.
Hàm tự đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi đặt seed. Các script gọi
hàm trước khi tạo `Accelerator` hoặc thực hiện phép tính CUDA, nên không cần đặt
biến này thủ công trong notebook Kaggle. Khi khởi động, log ghi
seed, các cờ xác định, phiên bản PyTorch/CUDA, số process và precision để đối chiếu.
DataLoader train của các phiên bản này dùng `seed_worker` và generator riêng từ
`create_seeded_generator(42)`;
cả hai helper cũng nằm trong `src/shared/utils.py`.

Để kiểm tra tính lặp lại, dùng cùng code, dữ liệu, hyperparameter, phiên bản thư viện,
số GPU/process và precision; mỗi lần train từ đầu dùng một `RUN_MODE` mới để tránh
resume checkpoint. Nếu gặp phép toán không hỗ trợ chế độ xác định, script sẽ báo
`RuntimeError`; cần xử lý phép toán đó thay vì bỏ qua lỗi hoặc chỉ bật cảnh báo.
Chế độ này có thể làm train chậm hơn và không bảo đảm kết quả giống hệt giữa các
phiên bản PyTorch hay môi trường khác nhau.

text RAG cần visual features HDF5 của ảnh gốc và các file `*_i2t.parquet` được
tạo ở bước 4. Các tham số chính là `TOP_K_CAPTIONS`, `CTX_TOKENS_PER_CAPTION`
và `CTX_NLAYERS`. Mặc định `MAX_CTX_LENGTH` được tính bằng
`TOP_K_CAPTIONS * CTX_TOKENS_PER_CAPTION + TOP_K_CAPTIONS` (hiện tại `4 * 22 + 4 = 92`).
Context chèn `<eos>` giữa các caption; đây là giới hạn tổng chiều dài, không bắt buộc
mỗi caption phải đủ 22 token. Vẫn có thể đặt
trực tiếp `MAX_CTX_LENGTH` trong `.env` nếu muốn dùng một giới hạn cố định.
Caption mục tiêu và câu sinh vẫn dùng `MAX_LENGTH=40`, độc lập với chiều dài context.

## 6. Train và generate multimodal RAG kết hợp ảnh và caption

multimodal RAG chọn độc lập các ảnh liên quan tốt nhất và các caption tốt nhất từ hai file
`*_i2i.parquet` và `*_i2t.parquet` hiện có. Không yêu cầu caption
phải thuộc các ảnh liên quan đã chọn, và không cần gộp hai knowledge base.
Nếu chưa có kết quả truy hồi ảnh, tạo image KB bằng
`python script/shared/build_image_kb.py`, rồi chạy `python script/image_rag/build_related_images.py`.

Memory đưa vào decoder có thứ tự:

```text
[patch tokens ảnh gốc, CLS tokens ảnh liên quan, caption tokens qua text encoder]
```

Hai nhánh ảnh dùng chung projector `Linear → LayerNorm`. multimodal RAG có text encoder, decoder và hàm
encode context riêng trong `src/multimodal_rag`, không import các phiên bản trước;
word embedding được dùng chung giữa context và decoder.
Padding mask chỉ che các vị trí padding của caption context. Mặc định bỏ CLS của
ảnh gốc; có thể bật qua `include_cls_token` khi gọi model, engine hoặc inference.

Đặt `RUN_MODE=multimodal_rag_l14` trong `.env`, rồi chạy:

```bash
accelerate launch script/multimodal_rag/train_rag.py
accelerate launch script/multimodal_rag/generate_rag_captions.py
python script/shared/evaluate.py
```

Dùng `RUN_MODE` riêng cho từng thí nghiệm để tách checkpoint và kết quả.
`TOP_K_RAG_IMAGES` điều khiển số CLS ảnh liên quan, còn `TOP_K_CAPTIONS` điều khiển
số caption. Hai giá trị độc lập và không được vượt số kết quả có trong parquet.
Với visual features B/32, K ảnh = 4 và context length = 92, memory có
`49 + 4 + 92 = 145` vị trí, tính cả padding context.

Cần visual features của ảnh gốc theo từng split và `train_visual_features.h5`
để đọc CLS ảnh liên quan, kể cả khi validation hoặc generate trên tập test.
Các visual features phải được trích bằng cùng `VISUAL_ENCODER_MODEL`;
image embeddings dùng cho retrieval không thay thế được visual features này.
Training loss được chuẩn hóa theo tổng số target token hợp lệ trên tất cả process
trước khi backward như text RAG.

### multimodal RAG không dùng đặc trưng ảnh đầu vào

Biến thể `MultimodalRAGCaptionerNoInputImage` trong `src/multimodal_rag/models/rag_no_input_image.py`
chỉ đưa CLS token của ảnh liên quan và token caption đã qua Transformer encoder
vào memory của decoder:

```text
[CLS tokens ảnh liên quan, caption tokens qua text encoder]
```

Đặt `RUN_MODE=multimodal_rag_no_input_image` trong `.env` để huấn luyện từ đầu với
checkpoint và kết quả riêng, rồi dùng cùng tùy chọn cho train và generate:

```bash
accelerate launch script/multimodal_rag/train_rag.py --no-input-image
accelerate launch script/multimodal_rag/generate_rag_captions.py --no-input-image
python script/shared/evaluate.py
```

Biến thể này không đọc visual features của ảnh đầu vào. Vẫn cần
`train_visual_features.h5` để lấy CLS của các ảnh liên quan và các file kết quả
truy hồi ảnh/caption của từng split. Ảnh đầu vào vẫn quyết định kết quả truy hồi;
đặc trưng của ảnh đó không được đưa trực tiếp vào model sinh caption.
Với K ảnh = 4 và context length = 92, memory có `4 + 92 = 96` vị trí,
tính cả padding context. Projector, text encoder và decoder giữ cùng cấu trúc
với multimodal RAG để so sánh ảnh hưởng của việc bỏ nhánh ảnh đầu vào.

## 7. Train, generate va evaluate bang visual features

Sau khi da co ca ba file H5, train va sinh caption khong can doc anh hay tai
CLIP vision encoder nua:

Ví dụ baseline, đặt `RUN_MODE=baseline` trong `.env`, rồi chạy:

```bash
accelerate launch script/baseline/train.py
accelerate launch script/baseline/generate_baseline_captions.py
```

Baseline giữ nguyên kiến trúc và định dạng state dict của V4 baseline cũ. Để tiếp tục
dùng checkpoint hoặc kết quả cũ, giữ nguyên `RUN_MODE` của lần chạy đó
(ví dụ `v4_baseline_precomputed`); không cần di chuyển hay đổi tên checkpoint.

Hai lenh phai dung cung `RUN_MODE` de dung chung checkpoint. `RUN_MODE` moi
tranh resume nham checkpoint cu duoc train theo duong online-encoder. Feature
files can co format `imgids` va `features` nhu mo ta o `ARTIFACTS.md`; code se
kiem tra moi `imgid` trong split deu co feature truoc khi train.

Cuoi cung chay `evaluate.ipynb` de tinh BLEU, METEOR, ROUGE-L, CIDEr va SPICE
tu prediction JSON. Notebook nay khong encode anh; phan hien thi mau van can
`IMAGES_PATH` neu muon xem anh goc.

## 8. Learning rate: linear warmup và cosine decay

Baseline, image RAG, text RAG và multimodal RAG (kể cả `--no-input-image`) dùng
AdamW với scheduler có sẵn của Transformers:
`get_cosine_with_min_lr_schedule_with_warmup_lr_rate`. Cấu hình mặc định:

```dotenv
LEARNING_RATE=1e-4
WARMUP_RATIO=0.05
WARMUP_START_FACTOR=0.01
MIN_LR_RATIO=0.05
```

`LEARNING_RATE` là LR đỉnh. Scheduler tăng tuyến tính từ 1% LR đỉnh trong 5% đầu
quá trình, sau đó giảm theo cosine đến 5% LR đỉnh. Ví dụ:
`1e-6 → 1e-4 → 5e-6`.

Accelerate mặc định dùng `split_batches=False`, nên `BATCH_SIZE` áp dụng trên mỗi
GPU. `total_updates` lấy từ số batch mỗi process sau khi Accelerate chuẩn bị
DataLoader, nhân với số epoch. Accelerate bọc scheduler trong
`AcceleratedScheduler`; mỗi optimizer update thành công, wrapper tiến scheduler
bên trong theo số process. Helper trong `src/shared/scheduler.py` quy đổi số bước
cho wrapper và chuyển cấu hình sang API Transformers; thư viện tính đường cong LR.

Engine gọi `scheduler.step()` sau `optimizer.step()`. Wrapper xử lý các bước chưa
đồng bộ và trường hợp optimizer bỏ qua update do mixed precision. Log hiển thị LR
và số optimizer update; validation không tiến scheduler.

Checkpoint lưu state và cấu hình scheduler để kiểm tra khi resume. Giữ nguyên LR
đỉnh, các tỉ lệ, tổng số epoch và số process để tiếp tục cùng lịch. Checkpoint cũ
chưa có state scheduler vẫn dùng được để nạp weights và inference; để train theo
lịch mới, bắt đầu `RUN_MODE` mới.

Chạy kiểm tra scheduler và checkpoint bằng:

```bash
python -m unittest discover -s tests -v
```

## Cấu hình đường dẫn chính

Các biến được định nghĩa trong `src/shared/config.py`:

| Biến | Mặc định |
| --- | --- |
| `DATASET_COCO_PATH` | `dataset/mscoco/dataset_coco.json` |
| `IMAGES_PATH` | `dataset/mscoco/images` |
| `ARTIFACTS_DIR` | `artifacts` |
| `VISUAL_FEATURES_DIR` | `artifacts/visual_features` |
| `CHECKPOINTS_DIR` | `checkpoints` |
| `RUN_MODE` | `baseline` |

Có thể ghi đè bằng biến môi trường hoặc file `.env`.
