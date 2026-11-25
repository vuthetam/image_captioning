# Kế hoạch Fact Fusion Prefix

Ngày chốt: 25/11/2026.

Trạng thái: đã triển khai pipeline riêng; hướng dẫn chạy và kiểm tra tại mục 9.

## 1. Phạm vi

- Tạo pipeline riêng `fact_fusion_prefix`, phát triển từ `fact_fusion` và train lại từ đầu.
- Giữ nguyên logic fusion, CLIP cache, loss normalization, optimizer, scheduler, seed và các cấu hình huấn luyện hiện có.
- Dùng lại các thành phần shared phù hợp. Decoder và beam search cho prefix nằm trong pipeline mới, không sửa hành vi code cũ.
- Không thêm adapter, dual cross-attention, gate hoặc logit bias.

## 2. Facts và cấu trúc prefix

- Nhánh fusion giữ nguyên toàn bộ object/attribute/triplet facts, projector, fusion blocks và flag CLS của Fact Fusion.
- Prefix chỉ lấy `attribute_texts` và `triplet_texts` từ đúng top-k captions được chọn cho fusion. Sort theo `TEXT_RAG_CAPTION_SORT_BY`, sau đó lấy `TOP_K_CAPTIONS`.
- Trim/gộp whitespace, bỏ chuỗi rỗng, dedup chuỗi trong toàn bộ tập facts bổ sung và giữ thứ tự gặp đầu tiên.
- Dùng `load_rag_fact_lookup(..., return_indices=True)` một lần; hợp các indices thuộc attributes/triplets và lấy theo thứ tự danh sách facts fusion đã dedup. Một fact thuộc nhiều loại chỉ xuất hiện một lần trong prefix.
- Dùng `tokenize_fact()` hiện tại: lowercase, dấu `-` thành khoảng trắng, xoá dấu nháy đơn thẳng/cong, tách từ theo tokenizer. Không dùng CLIP tokenizer cho prefix.
- Tra token IDs bằng vocabulary decoder; từ ngoài vocab dùng `<unk>` theo cơ chế hiện tại.
- Thêm `<sep>` trực tiếp bằng token ID giữa các facts, không qua tokenizer. Không thêm separator sau fact cuối.

```text
fact₁ <sep> fact₂ <sep> ... <sos> caption <eos>
└──────────── prefix ───────────┘
```

`<sos>` là tên token bắt đầu caption đang có trong dự án, tương ứng `<bos>` đã thảo luận. Không thêm `<caption>`.

Nếu không có attributes/triplets hoặc không giữ lại được fact nào trong ngân sách, prefix chỉ có `<sos>`. Nhánh fusion vẫn báo lỗi nếu ảnh không có fact nào, như code cũ.

## 3. Ngân sách prefix

Thêm cấu hình riêng, không dùng chung budget của các pipeline text/fact token:

```python
FACT_PREFIX_TOKENS_PER_CAPTION = int(os.getenv("FACT_PREFIX_TOKENS_PER_CAPTION", "30"))
MAX_FACT_PREFIX_LENGTH = TOP_K_CAPTIONS * FACT_PREFIX_TOKENS_PER_CAPTION + 1
```

- `TOP_K_CAPTIONS * FACT_PREFIX_TOKENS_PER_CAPTION` (mặc định `TOP_K_CAPTIONS * 30`) là ngân sách chung sau dedup cho nội dung facts và `<sep>`, không phải giới hạn riêng từng caption.
- `+1` dành cho `<sos>`. Không tính caption và `<eos>` vào ngân sách prefix.
- Thêm lần lượt từng fact nguyên vẹn; fact tiếp theo làm vượt ngân sách thì dừng. Không cắt giữa fact, không tìm fact ngắn hơn ở phía sau để lấp chỗ trống.
- Ví dụ top-k=4: prefix tối đa 121 token.
- Độ dài thực tế linh hoạt; ghép chuỗi prefix + caption liên tiếp, chỉ padding cuối chuỗi theo batch, không chèn padding giữa prefix và caption.
- Budget chỉ giới hạn prefix văn bản, không giảm tập facts đưa vào fusion.
- Kiểm tra top-k và budget mỗi caption là số nguyên dương.

Tham khảo phép đo 5.000 ảnh validation, ITM sort: prefix top-k=4 dài nhất 98 token trước giới hạn. Đây là số liệu định hướng, không bảo đảm mọi split đều dưới 121 token.

## 4. Vocabulary và embedding

- Tạo bản vocabulary riêng từ vocab cũ; giữ nguyên toàn bộ token IDs và thêm đúng một special token `<sep>` ở cuối nếu chưa có.
- Đăng ký `<sep>` trong `special_tokens` để decode có thể loại bỏ đúng cách. Không ghi đè vocab gốc.
- Chạy script `script/fact_fusion_prefix/build_vocab.py` trước train. Train và generate chỉ nạp/kiểm tra vocab đã lưu, báo lỗi nếu file chưa có.
- Train và generate phải dùng cùng bản vocab mở rộng, cùng token IDs.
- Khởi tạo embedding và vocabulary projection theo kích thước vocab mới; train lại từ đầu.
- Prefix và caption dùng cùng một embedding của decoder. Không thêm text encoder riêng, không buộc weight tying giữa embedding và output projection.

## 5. Decoder và attention mask

Giữ kiến trúc Transformer Decoder của Fact Fusion, thay causal mask thông thường bằng prefix mask ở mọi self-attention layer:

| Query / Key | Prefix đến hết `<sos>` | Caption |
|---|---|---|
| Prefix đến hết `<sos>` | Đọc hai chiều toàn bộ prefix | Không đọc |
| Caption | Đọc toàn bộ prefix | Đọc đến vị trí hiện tại, không đọc tương lai |

- Xác định ranh giới prefix riêng cho từng mẫu, kết hợp với padding mask. Không dùng một ranh giới chung cho cả batch.
- Dùng boolean mask nhất quán với decoder hiện tại; `True` là vị trí bị chặn.
- Cross-attention của cả prefix và caption tiếp tục đọc fused visual memory.
- Giữ embedding scaling và positional encoding như decoder hiện tại, vị trí liên tục từ đầu prefix sang caption.
- `MAX_LENGTH` giữ quy ước caption cũ: bao gồm `<sos>` và `<eos>` khi chuẩn bị target. `<sos>` ở cuối prefix đồng thời là đầu caption, chỉ xuất hiện một lần tại ranh giới.
- Độ dài tối đa toàn chuỗi là `MAX_FACT_PREFIX_LENGTH + MAX_LENGTH - 1`. Cấp dung lượng positional encoding theo giới hạn này; teacher forcing dùng input đã bỏ token cuối.
- Ví dụ prefix tối đa 121 và `MAX_LENGTH=40`: toàn chuỗi tối đa 160 token, input teacher forcing tối đa 159 token.

## 6. Teacher forcing và loss

- Nối prefix và caption rồi shift input/target đúng một lần như cơ chế teacher forcing hiện tại.
- Tạo loss mask theo vị trí target và ranh giới prefix của từng mẫu. Chỉ tính loss cho target là caption tokens và `<eos>`; bỏ các vị trí target thuộc phần facts, `<sep>`, `<sos>` và padding. Không loại từ caption chỉ vì từ đó cũng xuất hiện trong prefix.
- Hidden state tại `<sos>` dự đoán từ caption đầu tiên, nên logit ở vị trí này phải được tính loss.
- Không bỏ loss theo toàn bộ vị trí input thuộc prefix vì như vậy sẽ bỏ nhầm dự đoán từ caption đầu tiên.
- Các vị trí target bị bỏ được gán `ignore_index`; số token dùng chuẩn hoá phải đếm từ cùng loss mask. Không dùng attention mask để đếm vì prefix vẫn tham gia attention.
- Chuẩn hoá theo số target caption tokens thực tế, không tính prefix. Giữ cơ chế chuẩn hoá nhiều GPU của Fact Fusion.
- Giữ quy ước giới hạn `MAX_LENGTH` của caption hiện tại; prefix không chiếm ngân sách caption.

Ví dụ căn chỉnh (dấu `×` là target bỏ qua, không phải một token trong vocab):

```text
Chuỗi:  red  dog  <sos>  a    dog    <eos>
Input:  red  dog  <sos>  a    dog
Target:  ×    ×     a    dog  <eos>
```

Với độ dài prefix `P` tính cả `<sos>`, vị trí logit đầu tiên được tính loss là `P - 1` theo chỉ số bắt đầu từ 0. Các vị trí tiếp theo chỉ được tính nếu target caption chưa phải padding.

## 7. Inference và beam search riêng

- Tạo beam search trong pipeline mới; không sửa trực tiếp `src/shared/inference.py`.
- Khởi đầu bằng toàn bộ prefix kết thúc ở `<sos>`, sinh caption tự hồi quy với cùng mask như khi train.
- Prefix khác độ dài vẫn phải hoạt động đúng theo từng ảnh; dùng logits tại vị trí token thật cuối cùng, không lấy nhầm vị trí padding.
- Giữ prefix, fused memory và ranh giới prefix đúng với ảnh khi mở rộng hoặc sắp xếp beams.
- Score chỉ cộng log-probability của các token sinh sau `<sos>`, gồm `<eos>` khi kết thúc; không cộng likelihood của prefix đã cung cấp.
- Giữ quy ước length penalty của beam search cũ trên chuỗi caption bắt đầu bằng `<sos>`: bộ đếm khởi tạo bằng 1, tăng ở mỗi bước beam chưa kết thúc sau khi cập nhật trạng thái EOS, không tăng ở bước sinh `<eos>` và các bước padding sau đó. Không cộng số token facts hoặc `<sep>` vào bộ đếm.
- Sinh tối đa `MAX_LENGTH - 1` bước sau `<sos>` như code cũ. Không trừ độ dài facts khỏi số bước này và không tự thêm một bước để sinh EOS khi hết giới hạn.
- Chặn `<sep>` khi sinh caption: sau `log_softmax`, gán log-probability của `<sep>` bằng `-inf` trước khi chọn beams. Giữ score của các token hợp lệ theo phân phối mô hình; không chuẩn hoá lại sau khi chặn. Loss train vẫn dùng toàn bộ vocabulary projection.
- Beam đã kết thúc chỉ nối padding với score không đổi, như beam search cũ.
- Dừng ở `<eos>`; cắt bỏ prefix bằng ranh giới đã lưu trước khi decode caption.
- Tính fused memory một lần mỗi batch ảnh và dùng lại trong beam search.
- Giữ định dạng predictions hiện tại, gồm `imgid`, `cocoid`, `caption`; in epoch checkpoint đã nạp.

## 8. Tổ chức và kiểm tra

- Mã nguồn trong `src/fact_fusion_prefix`; scripts `script/fact_fusion_prefix/train.py` và `generate_captions.py`.
- Vocab, checkpoint và kết quả dùng path/run riêng. Không resume checkpoint từ Fact Fusion hoặc các biến thể khác.
- Kiểm tra lựa chọn facts, sort/top-k, dedup, ngân sách và separator.
- Kiểm tra vocab giữ nguyên IDs cũ, thêm `<sep>` đúng một lần và train/generate dùng chung vocab.
- Kiểm tra prefix đọc hai chiều nhưng không đọc caption; token caption tương lai không ảnh hưởng logits ở vị trí trước, ở mọi decoder layer.
- Kiểm tra loss alignment tại `<sos>`, bỏ prefix/padding khỏi cả loss và số token chuẩn hoá.
- Kiểm tra từ xuất hiện ở cả facts và caption vẫn được tính loss tại vị trí target caption.
- Kiểm tra prefix rỗng, độ dài khác nhau, padding và logits hữu hạn.
- Khi prefix chỉ gồm `<sos>`, prefix mask phải trùng causal mask. Với cùng weights, cùng vocab mở rộng, cùng input/memory và ở chế độ eval, logits phải tương đương decoder Fact Fusion trong sai số số học cho phép.
- Kiểm tra beam routing, caption length penalty độc lập độ dài prefix và checkpoint round-trip.
- Kiểm tra budget toàn chuỗi không đếm `<sos>` hai lần, số bước sinh không đổi theo độ dài prefix, và `<sep>` không được sinh trong caption.
- Chạy thử train/val/generate với một batch nhỏ, `num_workers=0`; dùng scratch/test code, không chèn giới hạn test vào production.

## 9. Triển khai và cách chạy

- `dataset.py`: dùng lại loader Fact, tạo prefix, nối caption trước khi padding cuối batch.
- `decoder.py`: kế thừa decoder shared, đổi mask theo ranh giới prefix của từng mẫu; giữ cross-attention/FFN chuẩn.
- `model.py`: dùng nguyên `FactFusionEncoder`, cung cấp fused memory cho decoder mới.
- `engine_rag.py`: loss mask theo vị trí; chỉ đếm target caption thực tế, gồm EOS.
- `beam_search.py`: ghép lại prefix thật và caption mỗi bước, chọn logits tại token thật cuối; giữ score/length penalty của beam cũ và chặn SEP.
- `script/fact_fusion_prefix/build_vocab.py` chứa toàn bộ logic tạo/kiểm tra vocab mở rộng; không có module vocabulary riêng trong `src/fact_fusion_prefix`. `--vocab-path` chỉ vocab gốc (mặc định lấy `VOCAB_PATH`), `--output-path` là tham số bắt buộc chỉ file mới. Sau khi tạo, cập nhật `VOCAB_PATH` trong env sang file mới. Train/generate chỉ dùng `Vocabulary.load(VOCAB_PATH)` và kiểm tra có special token SEP.
- Khi collate inference, prefix có thể được padding tạm để chứa trong tensor. Beam search bỏ các slot đó khi ghép caption, nên input decoder không có padding nằm giữa prefix và caption.

Ví dụ `.env`:

```dotenv
RUN_MODE=fact_fusion_prefix_itm_42
SEED=42
TOP_K_CAPTIONS=4
TEXT_RAG_CAPTION_SORT_BY=itm
FACT_PREFIX_TOKENS_PER_CAPTION=30
```

Các path context và feature dùng cấu hình hiện có. Tạo vocab từ file gốc rồi đổi `VOCAB_PATH` trong env sang file đầu ra trước train. Ví dụ:

```bash
python script/fact_fusion_prefix/build_vocab.py --vocab-path artifacts/vocab_v3.json --output-path artifacts/vocab_prefix.json
```

```dotenv
VOCAB_PATH=artifacts/vocab_prefix.json
```

Train từ đầu bằng RUN_MODE mới; giữ nguyên RUN_MODE và cấu hình khi generate hoặc resume.

```bash
accelerate launch script/fact_fusion_prefix/train.py
accelerate launch script/fact_fusion_prefix/generate_captions.py
python script/shared/evaluate.py
```

Kiểm tra nằm trong `tests/test_fact_fusion_prefix.py`, gồm prefix mask, loss alignment, budget/vocab, chọn facts CLIP/ITM, fusion tương đương, beam theo batch/từng ảnh, length penalty, backward, CUDA FP16 (nếu có) và train/val/generate/checkpoint một batch với `num_workers=0`.
