# Kế hoạch bổ sung logit bias từ facts cho Fact RAG

Ngày ghi nhận: 12/11/2026.

Trạng thái: người dùng đã yêu cầu triển khai biến thể theo fact trước. Mã nguồn hiện nằm trong `src/fact_fusion_fact_bias`, scripts trong `script/fact_fusion_fact_bias`. Biến thể theo token chưa được triển khai. Kết quả kiểm tra xem mục 12.

## 1. Mục tiêu và phạm vi

Giữ kiến trúc fusion và memory của Fact RAG hiện tại, bổ sung một nhánh điều chỉnh logits dựa trên các facts truy hồi. Thử riêng hai biến thể:

1. **Fact bias**: tính một gate cho mỗi fact, rồi ánh xạ gate sang các token thuộc fact.
2. **Token bias**: tính một gate cho mỗi token duy nhất xuất hiện trong các facts.

Hai biến thể đều chỉ đưa fused memory vào decoder. Không concatenate captions, fact embeddings hoặc fact token embeddings thành memory bổ sung như Fact Text/Fact Memory/Fact Token RAG.

Tạo pipeline mới theo cách tổ chức của dự án; không làm thay đổi hành vi của các pipeline cũ. Ưu tiên import các thành phần Fact RAG/shared đã có thể dùng trực tiếp. Không tạo file chỉ để import trung gian, không đưa thành phần riêng của thử nghiệm vào shared khi chưa cần thiết.

Tên thư mục đã được chốt: `fact_fusion_fact_bias` và `fact_fusion_token_bias`. Thực hiện `fact_fusion_fact_bias` trước. Tên script theo quy ước hiện tại: `train.py`, `generate_captions.py`.

## 2. Nguồn facts và nhánh fusion giữ nguyên

- Đọc contexts đã truy hồi theo các path/config hiện có; dùng dữ liệu chưa chuẩn hóa entity aliases cho thử nghiệm này.
- Sort captions theo `TEXT_RAG_CAPTION_SORT_BY` (`clip` hoặc `itm`), chọn `TOP_K_CAPTIONS`, rồi mới lấy facts.
- Sử dụng đủ ba cột: `object_texts`, `attribute_texts`, `triplet_texts` cho cả hai biến thể.
- Dedup facts giống Fact RAG: trim/gộp whitespace, giữ thứ tự gặp đầu tiên; không đổi casing hoặc cách diễn đạt của chuỗi đưa vào CLIP.
- CLIP Text dùng `VISUAL_ENCODER_MODEL`, đóng băng; cache lười vector CLIP trong RAM CPU, riêng từng process/rank khi chạy Accelerate.
- Cache chỉ chứa projected text embedding gốc của CLIP, không L2 normalize. Không cache vector sau lớp chiếu đang train hoặc gate phụ thuộc timestep.
- Visual features được xử lý theo flag CLS hiện có; mặc định dùng patch tokens.
- Visual features và fact features qua các projector `Linear → LayerNorm` riêng.
- Giữ số block fusion, residual, LayerNorm và FFN như Fact RAG hiện tại. Patch tokens làm Q, facts làm K/V; fused memory giữ số token ảnh.
- Không có fact hợp lệ cho một ảnh thì báo lỗi như hiện tại.

Việc giữ nguyên nhánh fusion nói đến kiến trúc và cách xử lý dữ liệu. Khi train end-to-end với nhánh bias mới, gradient bổ sung có thể làm các trọng số projector/fusion/decoder học ra giá trị khác Fact RAG.

## 3. Công thức chung

Tại timestep `t`, với token vocabulary `w`:

```text
h_t = hidden state sau decoder cuối, ngay trước vocabulary projection
z_t = vocabulary_projection(h_t)
alpha = softplus(raw_alpha)
adjusted_logits_t(w) = z_t(w) + alpha * b_t(w)
```

- `alpha` là một scalar trainable chung cho toàn mô hình, không phải scalar riêng theo ảnh hoặc timestep.
- Tham số `initial_alpha` ở captioner/decoder có mặc định `0.1`; scripts train/generate không cần truyền khi dùng mặc định. Dùng cùng giá trị khởi tạo cho cả hai biến thể khi so sánh.
- Khởi tạo `raw_alpha = log(expm1(0.1))`, xấp xỉ `-2.252168`.
- Gate dùng sigmoid, thuộc `(0, 1)`; bias sau masking thuộc `[0, 1]`.
- Token ngoài tập ứng viên có bias `0`.
- Chưa dùng bias âm, không thêm copy distribution hoặc gate trộn hai phân phối xác suất.
- Train bằng caption cross-entropy trên adjusted logits; không cần nhãn gate hoặc auxiliary loss ở phiên bản đầu.
- Giữ teacher forcing, mask padding, chuẩn hóa loss và cách chọn checkpoint như pipeline hiện có.

Với batch size `B`, chiều dài decoder `T`, vocab size `V`, kích thước hidden `D`:

```text
hidden_states: [B, T, D]
base_logits:   [B, T, V]
bias:          [B, T, V]   # shape logic; có thể scatter trực tiếp để giảm cấp phát
output_logits: [B, T, V]
```

Không đưa softmax vào trước hàm cross-entropy nhận logits. Khi generate, cộng bias trước `log_softmax` của beam search.

## 4. Biến thể theo fact

```text
F_j = fact embedding sau fact_projector (Linear → LayerNorm)
g_tj = sigmoid(MLP_fact(concat(h_t, F_j)))

b_t(w) = max(g_tj với các fact j chứa token ID w)
         0 nếu không có fact nào chứa token ID w
```

- Dùng chính `F_j` đã có ở nhánh fusion, trước khi được dùng làm K/V. Không CLIP encode lần nữa, không tạo projector fact mới.
- Fusion hiện tại chưa tính gate bias; MLP theo `[h_t; F_j]` là thành phần mới, chạy ở từng timestep.
- Một MLP dùng chung cho tất cả facts và timesteps trong mô hình này.
- Token IDs dedup trong từng fact. Giữ ánh xạ từng fact tới các token IDs của chính nó.
- Một token xuất hiện trong nhiều facts nhận `max` gate, không cộng dồn theo số lần xuất hiện.
- Fact padding không được tham gia `max` hoặc ghi bias vào vocab.
- Fact có chuỗi hợp lệ nhưng không còn token ứng viên sau lọc vocabulary vẫn tham gia fusion; nó không đóng góp bias từ vựng.

Shape logic: `F: [B, N, D]`, gate `[B, T, N]`, với `N` là số fact tối đa trong batch sau padding. Thứ tự `F_j`, fact mask và danh sách token IDs theo fact phải luôn khớp nhau.

## 5. Biến thể theo token

```text
candidate_ids = dedup token IDs từ toàn bộ facts đã chọn của từng ảnh
E(w) = decoder.embedding(w)
g_t(w) = sigmoid(MLP_token(concat(h_t, E(w))))
b_t(w) = g_t(w) nếu w là ứng viên; ngược lại bằng 0
```

- Embedding để tính gate lấy từ bảng embedding đang dùng của decoder, không dùng CLIP Text và không tạo bảng embedding token riêng.
- Dùng vector tra trực tiếp từ `decoder.embedding`, trước positional encoding và bước nhân `sqrt(d_model)` dành cho đầu vào decoder.
- Một MLP dùng chung cho tất cả token ứng viên và timesteps trong mô hình này.
- Dedup token IDs trên toàn bộ facts của một ảnh, giữ thứ tự gặp đầu tiên. Mỗi token ứng viên chỉ tính gate một lần mỗi timestep.
- Nhánh fusion vẫn dùng CLIP Text và cache như Fact RAG. Chỉ nhánh tính token bias dùng embedding decoder.

Shape logic: candidate IDs `[B, U]`, candidate embeddings `[B, U, D]`, gate `[B, T, U]`, với `U` là số ứng viên tối đa trong batch. Mask các slot padding trước khi ghi vào bias.

## 6. Quy tắc tokenize và ánh xạ vocabulary

Tokenize phục vụ bias tách biệt với tokenization của CLIP. Không sửa chuỗi facts dùng cho CLIP/cache chỉ để xây token IDs.

Quy tắc đã thống nhất:

1. Chuyển chữ thường.
2. Thay dấu `-` bằng khoảng trắng: `blue-green` → `blue green`, không ghép thành `bluegreen`.
3. Xóa dấu nháy đơn ASCII `'` và dấu nháy cong `’`, không thêm khoảng trắng: `man's` → `mans`.
4. Xử lý dấu câu/tách từ phù hợp vocabulary decoder; không dùng tokenizer CLIP cho bước này.
5. Ánh xạ từ sang token IDs của đúng vocabulary decoder đang load.
6. Bỏ từ ngoài vocabulary và mọi special token, gồm `<pad>`, `<sos>`, `<eos>`, `<unk>` theo vocabulary thực tế.
7. Giữ từ chức năng như `a`, `is`, `on`; không stemming hoặc lemmatize.

Ví dụ trước khi lọc vocabulary:

```text
A man's blue-green shirt
→ a mans blue green shirt
→ [a, mans, blue, green, shirt]
```

Không thêm SOS/EOS giữa các facts. Không áp dụng ngân sách token của nhánh concatenate Fact Token RAG: ở đây lấy các token ứng viên từ toàn bộ facts đã chọn sau top-k. Số ứng viên linh hoạt và pad theo batch khi cần.

Cả hai biến thể cần token IDs để ghi bias vào đúng vị trí vocabulary. Chỉ biến thể token bias lấy thêm embedding decoder của những IDs này.

Trường hợp các facts hợp lệ nhưng toàn bộ token bị loại sau lọc vocabulary: bias của ảnh đó bằng `0`, tiếp tục dùng fused memory. Không nhầm với trường hợp không có fact nào, vốn phải báo lỗi.

## 7. Huấn luyện và inference

- MLP gate và `raw_alpha` thuộc trainable captioner, được optimizer cập nhật và lưu trong checkpoint.
- Không detach `h_t`, `F_j` hoặc embedding decoder ở nhánh bias. CLIP vẫn đóng băng như cũ.
- Projected fact vectors và embedding decoder đang train không được lưu vào cache dùng lại qua các optimizer steps.
- Không tính teacher forcing shift thêm lần nữa ngoài cách hiện tại của captioner/engine.
- Tính bias cho mọi timestep có loss khi train; khi generate chỉ cần bias cho timestep dự đoán tiếp theo.
- Compute fused memory và dữ liệu fact/candidate độc lập timestep một lần mỗi batch generate, tái sử dụng qua các bước beam search.
- Expand dữ liệu facts/candidate theo đúng thứ tự ảnh và beam; bảo đảm vẫn khớp khi chọn lại beam. Không để dữ liệu ảnh này tác động logits ảnh khác.
- Mask fact/token padding trước phép reduce/scatter. Mask padding của caption tiếp tục được xử lý trong loss như cũ.
- Khi ghi nhiều gates vào cùng token ID ở fact bias, phép reduce `max` phải giữ autograd.
- Theo yêu cầu khôi phục cách tối ưu, tách các cột trọng số của Linear đầu thành phần hidden và phần fact. Chiếu hidden một lần mỗi timestep, chiếu fact một lần mỗi fact, rồi cộng bằng broadcasting thành `[B, T, N, D]` trước ReLU. Bias của Linear chỉ cộng một lần. Không tạo tensor concatenate `[B, T, N, 2D]`; chưa benchmark tốc độ thực tế.

MLP mặc định đã dùng cho biến thể theo fact: `Linear(2D, D) → ReLU → Linear(D, 1) → Sigmoid`. Lớp đầu được tính thành tổng hai phép chiếu dùng hai phần trọng số của cùng lớp Linear, tương đương Linear trên vector concatenate. Sau khi cộng mới chạy ReLU, Linear cuối và Sigmoid; giữ nguyên các tham số học được và cấu trúc state dict.

## 8. Những điểm đã kiểm tra trong mã nguồn hiện tại

- `src/shared/facts/dataset.py`: `load_rag_fact_lookup()` đã sort captions, chọn top-k, gom đủ ba loại facts và dedup; dùng làm nguồn facts chung cho fusion và bias.
- `src/shared/facts/encoder.py`: `CachedCLIPFactEncoder` đã đóng băng CLIP, cache lười CPU theo process và trả fact features cùng mask.
- `src/shared/facts/fusion.py`: `encode_fusion()` đã trả `(fused_memory, fact_memory, fact_padding_mask)`; `fact_memory` chính là `F` cần cho fact bias.
- `src/fact_rag/models/rag.py`: captioner hiện chỉ lấy fused memory rồi gọi decoder.
- `src/shared/decoder.py`: `TransformerCaptionDecoder.forward()` hiện trả thẳng `out_fc(out)`, chưa cung cấp hidden states cho nhánh bias. Cần cách truy cập `out` trong pipeline mới mà giữ hành vi decoder cũ.
- `src/shared/vocabulary.py`: `Vocabulary` ánh xạ các tokens có sẵn sang IDs; không có tokenizer raw text riêng. OOV hiện ánh xạ về `<unk>`, vì vậy nhánh bias phải lọc OOV/special IDs rõ ràng.
- `src/fact_token_rag/dataset.py`: có helper xử lý chữ thường, `-` và nháy ASCII; helper hiện chưa xử lý nháy cong/dấu câu. Đây chỉ là đối chiếu, không cần phụ thuộc pipeline Fact Token RAG.
- `src/shared/inference.py`: `beam_search()` hiện chỉ truyền memory/memory mask vào decoder. Cần cơ chế truyền dữ liệu bias hoặc beam search riêng cho pipeline mới; không được chỉ thêm bias lúc train rồi generate theo đường cũ.
- `script/shared/preprocess_dfs.py` dùng cột `tokens` có sẵn từ dữ liệu. Khi triển khai tokenizer facts, kiểm tra cách biểu diễn dấu câu trong vocabulary đang dùng; không giả định `Vocabulary` đã cung cấp tokenizer.

## 9. Các bước triển khai ở phiên sau

1. Đọc `CODEX.md` và tài liệu này; không dùng `GEMINI.md`. Nhận xác nhận thực hiện code theo quy định dự án trước khi sửa source.
2. Tạo hai pipeline mới, tái sử dụng nguồn facts, frozen encoder và fusion hiện có.
3. Tạo ánh xạ fact → token IDs và tập token IDs duy nhất theo ảnh, với các quy tắc ở mục 6.
4. Cho pipeline mới truy cập hidden states decoder trước `out_fc`, thêm hai kiểu gate và `raw_alpha`; bảo toàn hành vi các pipeline cũ.
5. Nối adjusted logits vào train/validation và generation/beam search. Giữ config sort, top-k, CLS, seed, loss normalization và các tham số train/generate hiện có.
6. Bổ sung script/config cần thiết ở mức tối thiểu, cập nhật hướng dẫn chạy. Không gộp thêm cơ chế khác như negative bias, copy distribution, stopword filtering hoặc concatenate memory vào thử nghiệm đầu.
7. Khi được yêu cầu test: smoke test đúng 1 batch, `num_workers=0` (hoặc 1), dùng scratch script/command; không chèn logic test vào production source.

## 10. Các điểm cần kiểm chứng khi triển khai

- Hyphen/apostrophe, OOV, special tokens và dedup theo đúng quy tắc.
- Fact/token masks khớp features; padding không tạo bias, các token ngoài facts giữ nguyên logits trước softmax.
- Fact bias dùng `max` đúng khi nhiều facts có cùng token; token bias không tính lặp ứng viên.
- Bias có shape đúng; logits/loss hữu hạn; gradient đến MLP và `raw_alpha`.
- Ảnh không có candidate token sau lọc vẫn chạy với bias `0`.
- Generate có dùng bias, beam size > 1 và batch > 1 không trộn dữ liệu giữa ảnh.
- Đặt `alpha=0` trong kiểm tra scratch thì nhánh mới trả đúng base logits với cùng trọng số/đầu vào và chế độ eval.
- Chạy Accelerate nhiều process đúng cách; chú ý nhánh rỗng ứng viên có thể khiến tham số gate không được dùng trong một lượt forward.

Đây là danh sách kiểm chứng; kết quả thực tế của biến thể đã triển khai được cập nhật ở mục 12.

## 11. Giới hạn và đánh giá thực nghiệm

- Bias dương chỉ khuyến khích từ trong facts; không trực tiếp hạ logit bằng bias âm. Xác suất token ngoài facts vẫn có thể giảm do mẫu số softmax.
- Fact bias dùng chung gate cho các từ thuộc cùng fact; không chọn riêng vai trò/từ trong fact. `Max` cho gradient chủ yếu qua fact đạt cực đại đối với mỗi token.
- Token bias điều chỉnh từng từ, nhưng không nhận trực tiếp toàn bộ ngữ nghĩa fact chứa từ đó; thông tin ngữ cảnh facts/ảnh đến qua `h_t`.
- Không thể kết luận trước biến thể nào tốt hơn hoặc chắc chắn cải thiện Fact RAG.
- So sánh hai biến thể và Fact RAG dưới cùng dữ liệu, sort, top-k, seed, train và decoding config. Khi báo cáo nhiều seed, dùng mean ± sample SD và nêu số seed.
- Theo dõi BLEU-4, CIDEr, SPICE, CHAIRs/CHAIRi; xem thêm độ dài captions, lặp từ, giá trị alpha và phân bố gate hợp lệ để hiểu nhánh bias đang làm gì.

## 12. Trạng thái triển khai biến thể theo fact

- Đã thêm captioner, decoder/gate, tokenizer ánh xạ vocabulary, engine train/validation và inference cho `fact_fusion_fact_bias`.
- Captioner nằm trong `src/fact_fusion_fact_bias/model.py`. `encode_memory()` trả `(memory, fact_memory, fact_padding_mask)` và được dùng chung bởi `forward()` và inference; generate tính bộ ba này một lần mỗi batch, trước beam search.
- Tái sử dụng trực tiếp dataset/collate, CLIP encoder/cache, fusion, scheduler/checkpoint và beam search chung. Theo yêu cầu chuẩn hóa sau đó, causal mask và target padding mask trong `src/shared/decoder.py` được đổi từ float sang boolean, đồng thời cập nhật target padding mask của decoder fact bias. Giữ nguyên tên và kích thước trọng số.
- Lớp chuyển tiếp cho beam search expand static facts theo từng ảnh/beam. Các beam cùng ảnh dùng cùng facts, nên việc đổi thứ tự beam trong ảnh không làm lệch facts. Bias chỉ tính cho timestep cuối khi generate.
- `tokenizer.py` dùng dấu câu làm ranh giới từ, giữ nguyên chuỗi special token để lọc bằng vocabulary. Không thay chuỗi CLIP đầu vào.
- Đã chạy `python -m unittest discover -s tests -p test_fact_fusion_fact_bias.py -v`: cả 7 kiểm tra đều qua. Bao gồm tokenizer/dedup/OOV, đối chiếu MLP + max với công thức tham chiếu, fusion và logits khi tắt bias khớp Fact RAG, gradient (kể cả toàn OOV), beam search không trộn facts giữa ảnh, CUDA FP16 forward/backward/search và một batch train/validation/generate bằng scripts với `num_workers=0`. CLIP trong kiểm tra được khởi tạo nhỏ tại chỗ, không tải model pretrained.
- Chưa train/generate trên toàn bộ dữ liệu và chưa chạy nhiều GPU thực tế. Hướng dẫn sử dụng nằm trong `docs/SCRIPT.md`.
- Kiểm tra riêng trước/sau chuẩn hóa mask trên cùng trọng số: shared decoder và fact bias decoder load state dict cũ với `strict=True`, beam search cho cùng chuỗi token, backward hữu hạn trên CPU FP32, CUDA FP32 và CUDA FP16. Sai khác logits tối đa khoảng `2.4e-7` ở FP32 và `0` ở CUDA FP16 trong batch thử; đây không phải đánh giá lại metrics trên toàn bộ test set.
- Sau chuẩn hóa, 24 tests của Fact RAG, Fact Token, Fact Attribute và Fact Fusion Fact Bias đều qua, gồm smoke tests train/validation/generate một batch với `num_workers=0` và các kiểm tra CUDA FP16. Đã sửa import tên script cũ trong ba bộ tests cho khớp `train.py`/`generate_captions.py`, cùng assertion checkpoint cho tên `fact_gate_mlp` hiện tại.
