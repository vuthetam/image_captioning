# Kế hoạch Fact Fusion Dual Attention

Ngày ghi nhận: 23/11/2026.

Trạng thái: đã triển khai pipeline riêng theo thiết kế dưới đây; kết quả kiểm tra tại mục 12.

## 1. Mục tiêu và các quyết định đã chốt

Giữ nhánh fusion của Fact, bổ sung khả năng đọc trực tiếp các attribute/triplet vectors trong mỗi decoder layer. Hai kết quả cross-attention được kết hợp bằng một gate học theo ảnh, vị trí sinh và decoder layer.

- Tên pipeline: `fact_fusion_dual_attention`.
- Train captioner từ đầu; CLIP Text vẫn dùng pretrained weights và đóng băng như Fact hiện tại.
- Gate khởi tạo chính xác bằng `0.1`: hệ số trộn ban đầu của fused visual là 0.9, fact trực tiếp là 0.1. Đây là hệ số, không bảo đảm tỷ lệ độ lớn hoặc lượng thông tin thực tế là 90/10.
- Một residual FFN adapter cho fact memory, tính một lần trước decoder và dùng chung đầu ra cho mọi decoder layer.
- Hai cross-attention chạy song song, cùng query; mỗi nhánh trong mỗi decoder layer có tham số riêng.
- Mỗi decoder layer có MLP gate riêng, xuất một scalar cho mỗi vị trí sinh.
- Đầu vào gate gồm H đã qua Add & Norm và hai contexts được chuẩn hóa bằng LayerNorm riêng. Phép trộn vẫn dùng C_img và C_fact gốc.
- Huấn luyện bằng caption cross-entropy; không thêm auxiliary loss hoặc logit bias.
- Tạo pipeline riêng; giữ nguyên hành vi và source của các pipeline cũ.

Giả thuyết cần kiểm chứng: attention riêng cho facts và gate tại từng decoder layer có thể kiểm soát việc đọc lại tri thức tốt hơn các biến thể đã thử. Chưa khẳng định kiến trúc này sẽ tăng metrics.

## 2. Nguồn facts và cách ánh xạ hai nhánh

### Nhánh fusion

Dùng nguyên quy tắc của `src/shared/facts/dataset.py` và kiến trúc trong `src/shared/facts/fusion.py`:

1. Đọc contexts từ các path hiện có, sử dụng dữ liệu chưa chuẩn hóa entity aliases.
2. Sort captions theo `TEXT_RAG_CAPTION_SORT_BY`: `clip` giữ thứ tự lưu sẵn, `itm` sắp xếp giảm dần ổn định theo ITM score.
3. Chọn `TOP_K_CAPTIONS`, sau đó gom đủ `object_texts`, `attribute_texts`, `triplet_texts`.
4. Trim/gộp whitespace, bỏ chuỗi rỗng, deduplicate theo chuỗi và giữ thứ tự xuất hiện. Không thay casing hoặc tách từ trong chuỗi đưa vào CLIP.
5. Mỗi fact được CLIP Text encode thành một vector riêng, dùng `VISUAL_ENCODER_MODEL`, không L2 normalize.
6. Dùng lại `CachedCLIPFactEncoder`: cache lười CPU RAM FP32, riêng từng process; train và val trong cùng process dùng chung cache.
7. Projector ảnh và fact vẫn là `Linear → LayerNorm`. Patch ảnh làm Q, toàn bộ facts làm K/V; giữ nguyên residual, norms, FFN và `FACT_FUSION_NLAYERS`.
8. Mặc định bỏ CLS; giữ API `include_cls_token=False` như code cũ.

### Nhánh fact trực tiếp

- Chỉ lấy `attribute_texts` và `triplet_texts` từ đúng các captions đã chọn cho fusion.
- Gọi `load_rag_fact_lookup(..., return_indices=True)` để tạo danh sách facts đã dedup và chỉ số theo từng cột trong cùng lượt đọc context. Các pipeline khác giữ mặc định `return_indices=False` để chỉ nhận danh sách facts như trước.
- Chọn các index thuộc attribute/triplet, mỗi index một lần theo thứ tự danh sách fusion đã dedup. Lấy vector từ `fact_memory` mà `encode_fusion()` đã trả về, sau fact projector; không cần loader đọc lại context riêng.
- Một chuỗi xuất hiện ở cả object và attribute/triplet vẫn được chọn cho nhánh trực tiếp nếu nó xuất hiện trong attribute/triplet. Không dựa vào các khoảng index cố định để xác định loại fact vì fusion đã dedup giữa các cột.
- Index và padding mask của tập con phải luôn đồng bộ với thứ tự facts dùng cho CLIP/fusion. Không tự sort lại vectors sau khi tạo index.
- Không gọi CLIP lần hai, không tạo fact projector thứ hai; gradient của cả hai nhánh cùng đi về fact projector hiện có.
- Không tokenize sang vocabulary decoder, không thêm SOS/EOS, không giới hạn theo token context. Số vectors là số attribute/triplet facts duy nhất thực tế, có padding khi ghép batch.
- Một fact quá dài đối với CLIP vẫn báo lỗi theo encoder hiện tại; không âm thầm cắt câu.

## 3. Fact adapter

Adapter là một residual FFN với Pre-LayerNorm. Nó xử lý độc lập từng fact bằng cùng trọng số, không có self-attention giữa các facts.

```text
F: attribute/triplet vectors sau fact projector [B, N, D]
│
├──────────────────────────────────────────────┐
↓                                              │
LayerNorm(D)                                   │
↓                                              │
Linear(D → 4D)                                 │
↓                                              │
ReLU → Dropout                                 │
↓                                              │
Linear(4D → D)                                 │
↓                                              │
Dropout                                        │
└────────────────── cộng F ─────────────────────┘
                        ↓
             F_adapted: [B, N, D]
```

```text
F_adapted = F + Dropout(W2 · Dropout(ReLU(W1 · LayerNorm(F) + b1)) + b2)
```

- LayerNorm của adapter có tham số riêng với LayerNorm của fact projector.
- Dropout dùng `DROPOUT` hiện có. Chiều ẩn adapter là `4D`.
- Không thêm LayerNorm sau phép cộng residual trong adapter này.
- Mask các slot padding về 0 sau adapter và truyền padding mask sang fact attention.
- Adapter là một module duy nhất của captioner, không tạo bản riêng cho từng decoder layer.
- Tính adapter một lần trong `encode_memory()`. Trong inference, dùng lại memory này suốt các timestep và beams; không cache lâu dài vector sau adapter đang train.

Đây là FFN mở rộng chiều, không phải bottleneck adapter `D → r → D` với `r < D` thường dùng để tiết kiệm tham số.

## 4. Một decoder layer

Ký hiệu `X` là đầu vào layer, `M_img` là fused visual memory, `M_fact` là fact memory sau adapter:

```text
H = LayerNorm(X + Dropout(MaskedSelfAttention(X, X, X)))

C_img  = ImageCrossAttention(Q=H, K=M_img,  V=M_img)
C_fact = FactCrossAttention(Q=H,  K=M_fact, V=M_fact)

C_img_gate  = LayerNorm_img(C_img)
C_fact_gate = LayerNorm_fact(C_fact)
G = sigmoid(GateMLP(concat(H, C_img_gate, C_fact_gate)))
C = (1 - G) * C_img + G * C_fact

Z = LayerNorm(H + Dropout(C))
Y = LayerNorm(Z + Dropout(DecoderFFN(Z)))
```

- Hai cross-attention đều đọc cùng `H`, không chạy tuần tự nhánh này rồi dùng kết quả làm query nhánh kia.
- `H` là đầu ra sau residual và LayerNorm của masked self-attention, không phải đầu ra attention thô. Hai LayerNorm dành cho contexts chỉ nằm trên đường tính gate; không thay đổi contexts dùng trong phép trộn.
- Mỗi nhánh có Q/K/V và output projection riêng. Các decoder layers cũng có attention weights riêng.
- Attention ảnh và attention fact có softmax riêng trên hai tập keys riêng, không concatenate hai memories vào một attention.
- Chỉ cộng residual `H` một lần sau khi kết hợp hai contexts. Giữ masked self-attention, Add & Norm và FFN theo kiểu post-norm của decoder hiện tại.
- FFN bên trong decoder giữ cấu hình Fact hiện có; hiện `nn.TransformerDecoderLayer` dùng mặc định `dim_feedforward=2048`, ReLU. Không tự thay chiều ẩn decoder thành `4D` khi `D` khác 512; `4D` ở mục 3 chỉ áp dụng cho adapter mới.
- `M_img` đã chứa thông tin facts từ fusion. Gate lựa chọn giữa thông tin gắn với các patch và facts được truy cập trực tiếp.
- Nhánh fact chỉ cần key padding mask, không dùng causal mask. Causal mask và caption padding mask áp dụng cho masked self-attention như cũ.
- Sau layer cuối: vocabulary Linear → logits. Train dùng cross-entropy nhận logits; generate dùng log-softmax trong beam search.

## 5. Dynamic gate

Mỗi decoder layer có hai LayerNorm dành riêng cho đầu vào context của gate và một MLP riêng:

```text
H:                              [B, T, D]  # Đã qua Add & Norm
C_img_gate  = LayerNorm_img(C_img):  [B, T, D]
C_fact_gate = LayerNorm_fact(C_fact): [B, T, D]
    ↓
concat(H, C_img_gate, C_fact_gate): [B, T, 3D]
    → Linear(3D → D)
    → ReLU
    → Linear(D → 1)
    → Sigmoid
G: [B, T, 1]
```

- Giữ nguyên `H = LayerNorm(X + Dropout(MaskedSelfAttention(X, X, X)))` làm đầu vào gate; không thêm LayerNorm thứ hai cho H ở đây.
- `LayerNorm_img(D)` và `LayerNorm_fact(D)` có tham số học riêng với nhau, với các norms khác và với các decoder layers khác. Chúng chuẩn hóa theo chiều D tại từng vị trí để giảm ảnh hưởng của chênh lệch độ lớn giữa hai contexts lên MLP gate.
- Chỉ dùng `C_img_gate`, `C_fact_gate` để tính G. Phép kết hợp luôn là `C = (1 - G) * C_img + G * C_fact` với contexts gốc; sau đó giữ `Z = LayerNorm(H + Dropout(C))` và FFN như mục 4. Không thêm LayerNorm cho toàn bộ vector concat trong thiết kế này.
- Cùng MLP áp dụng cho mọi ảnh và vị trí trong một layer; đầu vào khác nhau tạo gate khác nhau.
- Gate là một scalar cho toàn bộ vector context của một vị trí, không phải gate cho từng fact, từng head hoặc từng chiều D.
- Khởi tạo weight của Linear cuối bằng 0 và bias bằng `log(0.1 / 0.9) ≈ -2.197224577`. Khi đó gate ban đầu bằng 0.1 ở mọi vị trí có facts, trong sai số số học.
- Có thể nhận `initial_gate: float = 0.1` tại constructor, kiểm tra `0 < initial_gate < 1`; scripts dùng mặc định, chưa cần thêm env mới.
- MLP gate, fact attention, adapter và captioner đều học qua caption loss. Gate không bị giữ cố định ở 0.1.
- Do Linear cuối khởi tạo weight bằng 0, gradient đến Linear đầu của gate và hai LayerNorm đầu vào gate bằng 0 ở lượt backward đầu là bình thường; cần kiểm tra chúng nhận gradient sau khi Linear cuối được cập nhật. Adapter và fact attention vẫn nhận gradient qua phép trộn context ngay từ đầu.
- Công thức trộn có tính cạnh tranh: tăng G làm giảm hệ số của C_img. Không đổi thành cộng `C_img + G * C_fact` trong lần triển khai này.

## 6. Shape và trường hợp thiếu facts

Ký hiệu `B`: số ảnh, `P`: số patch, `N_all`: số facts fusion, `N`: số attribute/triplet facts, `T`: số vị trí caption, `D`: d_model, `V`: vocab size. Các chiều fact là kích thước lớn nhất trong batch sau padding.

| Thành phần | Shape |
| --- | --- |
| CLIP fact features | `[B, N_all, clip_dim]` |
| Fact memory sau projector, trước chọn tập con | `[B, N_all, D]` |
| Fused visual memory | `[B, P, D]` |
| Attribute/triplet memory trước và sau adapter | `[B, N, D]` |
| Fact key padding mask | `[B, N]`, True = padding |
| H, C_img, C_fact, C_img_gate, C_fact_gate, C | `[B, T, D]` |
| Đầu vào MLP gate | `[B, T, 3D]` |
| Gate | `[B, T, 1]` |
| Logits | `[B, T, V]` |

- Không có bất kỳ fusion fact hợp lệ nào cho một ảnh: báo lỗi như Fact cũ.
- Có fusion facts nhưng không có attributes/triplets: ảnh đó dùng `C = C_img` trong mọi decoder layer, tương đương gate hiệu dụng bằng 0.
- Batch trộn mẫu có/không có facts bổ sung: chỉ đưa các hàng có facts vào fact attention; điền context còn lại bằng 0 và ép gate hiệu dụng bằng 0 ở các hàng thiếu facts.
- Không chạy attention trên hàng toàn padding rồi mới nhân gate 0. Không thêm một fact giả có thể được attention tới.
- Nếu cả batch không có facts bổ sung: bỏ qua adapter, fact attention và gate; dùng đường ảnh cho batch đó.
- Padding trong một mẫu không được ảnh hưởng đến kết quả trên các vị trí thật.
- Fact memory là tập các fact vectors; không thêm positional encoding cho nhánh này. Caption input vẫn dùng token embedding nhân sqrt(D) và positional encoding sin/cos như decoder cũ.

## 7. Train, validation, DDP và checkpoint

- Captioner train từ đầu; không tự nạp checkpoint của pipeline Fact cũ. Chỉ resume checkpoint thuộc chính run mới nếu chạy tiếp run đó.
- Giữ teacher forcing: model đưa `input_ids[:, :-1]` vào decoder, engine dùng `input_ids[:, 1:]` làm targets. Không shift thêm lần thứ hai.
- CE dùng `ignore_index=pad_idx`, cộng loss trên token thật và chuẩn hóa theo tổng token thật; DDP giữ cách scale hiện có: `local_loss_sum / global_tokens * num_processes`.
- Giữ AdamW, gradient clipping, warmup/cosine scheduler, `SEED`, loader shuffle và `drop_last=True` cho train như Fact. Validation vẫn đánh giá không gradient và chọn best theo val loss nhỏ hơn nghiêm ngặt.
- Một batch toàn thiếu facts bổ sung có thể bỏ qua tham số của nhánh mới. Dùng cấu hình DDP `find_unused_parameters=True` tại script của pipeline mới để xử lý cả trường hợp các rank có trạng thái facts khác nhau, như cách pipeline attributes hiện làm; không thay cấu hình DDP của các pipeline cũ.
- Cache CLIP vẫn riêng mỗi process, không đồng bộ cache giữa GPU và không đưa cache/CLIP frozen vào optimizer hoặc checkpoint captioner.
- Kiểm tra forward/backward mixed precision; masks dùng boolean nhất quán, không sửa global mixed precision để phục vụ riêng pipeline.
- Checkpoint mới chứa adapter, hai cross-attention và gate của từng decoder layer, cùng optimizer/scheduler như quy ước hiện có.
- Lưu ý sẵn có: validation nhiều GPU với Accelerate even_batches có thể tính cả mẫu lặp để làm đầy batch cuối. Giữ cách tính hiện tại để đối chiếu các pipeline, không khẳng định val loss luôn tương đương đánh giá mỗi mẫu gốc đúng một lần.

## 8. Inference

- `encode_memory()` trả fused visual memory, fact memory đã qua adapter và fact padding mask; đặt tên từng biến rõ ràng.
- Encode CLIP, fusion và adapter đúng một lần cho mỗi batch ảnh; decoder đọc lại hai memories ở mọi layer và timestep.
- Dùng lại shared `beam_search` bằng một beam decoder wrapper của pipeline mới; wrapper gắn fact memory và mask, lặp theo beam bằng đúng thứ tự ảnh.
- Giữ memories của hai nhánh và fact masks đồng bộ với ảnh. Các beams của cùng ảnh dùng chung facts; shared beam search chỉ chọn lại beams trong từng ảnh.
- Chỉ chọn hidden state cuối để vocabulary projection khi generate nếu có tối ưu này; mọi decoder layer vẫn xử lý prefix đầy đủ. Không chỉ truyền một token mới khi chưa có KV cache.
- Predictions vẫn lưu `imgid`, `cocoid`, `caption`; in epoch checkpoint đã load. Evaluate dùng script hiện có, bao gồm CHAIR theo setup hiện tại.

## 9. Tổ chức mã nguồn và cấu hình dự kiến

```text
src/fact_fusion_dual_attention/
    __init__.py
    dataset.py        # Chọn attributes/triplets và ánh xạ vào facts fusion
    adapter.py        # Residual FFN adapter
    decoder.py        # Dual-attention layers, gate và caption decoder
    model.py          # Captioner, encode_memory và forward train
    engine_rag.py     # Train/val với dữ liệu và model mới
    inference_rag.py  # Beam wrapper và generate

script/fact_fusion_dual_attention/
    train.py
    generate_captions.py

tests/test_fact_fusion_dual_attention.py
```

- Dùng trực tiếp shared facts encoder, fusion, datasets khi phù hợp, positional encoding, vocabulary, checkpoint, scheduler, utils và beam search. Dataset mới chỉ bổ sung logic chọn tập con/index/mask cần thiết, giữ dữ liệu batch dạng tuple với các biến tường minh.
- Không tạo file import trung gian; không sao chép nguyên module chỉ để đổi tên. Thành phần riêng của pipeline để trong thư mục pipeline, không đẩy thêm vào shared.
- Không sửa `src/shared/decoder.py`, shared fusion hoặc pipeline Fact cũ để phục vụ decoder mới.
- Hướng dẫn chạy đã bổ sung trong `docs/SCRIPT.md`, mục Fact Fusion Dual Attention.
- Giữ các config hiện có: `VISUAL_ENCODER_MODEL`, `RAG_CONTEXTS_SUBDIR`, `TEXT_RAG_CAPTION_SORT_BY`, `TOP_K_CAPTIONS`, `FACT_TEXT_BATCH_SIZE`, `FACT_FUSION_NLAYERS`, kiến trúc và train config.
- Chưa cần `CTX_NLAYERS`, `FACT_CTX_TOKENS_PER_CAPTION`, `MAX_FACT_CTX_LENGTH`, candidate token IDs hoặc tham số alpha của logit bias.
- Dùng RUN_MODE mới và khác nhau giữa các seed, vì path checkpoint hiện theo RUN_MODE, không tự chia thư mục theo SEED.

Ví dụ env cho một run mới:

```bash
RUN_MODE=fact_fusion_dual_attention_itm_42
SEED=42
TEXT_RAG_CAPTION_SORT_BY=itm
TOP_K_CAPTIONS=4
FACT_FUSION_NLAYERS=1
FACT_TEXT_BATCH_SIZE=128
```

RAG_CONTEXTS_SUBDIR và các config khác giữ cùng giá trị với run Fact dùng làm mốc so sánh.

## 10. Kiểm tra trước khi chạy thực nghiệm

1. **Chọn facts:** so sánh fusion facts với loader Fact hiện có cho cả clip/itm và top-k; kiểm tra dedup, trùng chuỗi giữa các loại fact, thứ tự index và mask. Không sửa parquet gốc.
2. **Fusion tương đương:** với cùng weights, inputs và eval mode, fused visual memory phải giống Fact. Test bỏ/giữ CLS và ảnh không có fusion fact báo lỗi.
3. **Adapter:** giữ đúng shape, không làm lẫn batch/fact, chỉ được gọi một lần trong encode_memory. Padding được mask; gradient đến fact projector qua cả hai nhánh.
4. **Dual attention/gate:** hai nhánh nhận cùng H sau Add & Norm và có tham số riêng; contexts qua hai LayerNorm riêng khi tính gate, nhưng phép trộn dùng contexts gốc. Gate khởi tạo 0.1, shape `[B,T,1]`, trộn đúng công thức. Các decoder layers không dùng chung cùng một module gate/attention hoặc LayerNorm đầu vào gate.
5. **Đối chiếu decoder cũ:** ép gate bằng 0 trong test, chuyển weights phần chung từ decoder Fact, bật eval và so logits với đường ảnh cũ. Đây là kiểm tra toán học với cùng weights, không yêu cầu hai mô hình train từ đầu cùng seed phải có weights/metrics giống nhau.
6. **Trường hợp rỗng/padding:** test có facts, chỉ một ảnh thiếu facts, cả batch thiếu facts và padding bị thay giá trị; không NaN, ảnh thiếu facts đi đúng đường ảnh.
7. **Train:** một batch forward/backward/update, CE bỏ padding và chuẩn hóa token đúng; gradient hữu hạn. Kiểm tra gradient MLP gate qua ít nhất hai optimizer updates để tính đến khởi tạo Linear cuối bằng 0.
8. **Inference:** full logits và last-step logits khớp nhau ở eval; beams không dùng nhầm facts của ảnh khác; encoder/adapter không chạy lại ở mỗi timestep.
9. **Scripts/checkpoint:** actual train, val và generate mỗi giai đoạn chỉ một batch nhỏ, `NUM_WORKERS=0`; CLIP dùng tiny local fixture không tải model thật. Kiểm tra save/load/resume scheduler và JSON có cocoid.
10. **CUDA/DDP:** test mixed precision một batch nếu có GPU. Kiểm tra trường hợp một rank thiếu toàn bộ facts bổ sung khi có môi trường nhiều process; báo rõ nếu chưa chạy được nhiều GPU thực tế, không suy diễn từ test một GPU.
11. **Hồi quy:** chạy kiểm tra liên quan của Fact/shared; xác nhận diff không thay source các pipeline cũ. Không hardcode limit/head/break/test flags vào production source.
12. **Causal mask:** chạy decoder ở `eval` với cùng weights, hai memories và padding masks; giữ nguyên prefix trước vị trí k, chỉ thay token thật tại k và/hoặc phía sau. Logits tại các vị trí trước k (`logits[:, :k, :]`) phải giữ nguyên trong sai số số học. Kiểm tra cả mẫu có attributes/triplets và mẫu chỉ dùng nhánh ảnh để bảo đảm decoder không đọc token tương lai qua self-attention hoặc đường tính gate.

Chỉ báo các kiểm tra thực tế đã chạy. Không chạy full train hoặc full generation trong quá trình triển khai/kiểm tra nhẹ.

Sau khi kiểm tra code đạt yêu cầu, đánh giá các seed 42/43/44 với cùng cấu hình dữ liệu, training và beam search như Fact; báo mean ± sample standard deviation và chênh lệch theo từng seed cho BLEU-4, CIDEr, SPICE, CHAIRs, CHAIRi. Có thể log gate trung bình theo decoder layer trên vị trí caption thật để kiểm tra mức sử dụng nhánh mới; không dùng tập test để chọn checkpoint hoặc điều chỉnh siêu tham số.

## 11. Cơ sở kiến trúc

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf): cơ sở cho position-wise FFN, masked self-attention, cross-attention, residual và LayerNorm. Transformer gốc dùng post-norm; adapter mới dùng pre-norm như lựa chọn riêng ở mục 3.
- [On Layer Normalization in the Transformer Architecture, ICML 2020](https://proceedings.mlr.press/v119/xiong20b.html): phân tích tổ chức Pre-LN/Post-LN; không phải bằng chứng trực tiếp cho hiệu quả của fact adapter này.
- [Meshed-Memory Transformer for Image Captioning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cornia_Meshed-Memory_Transformer_for_Image_Captioning_CVPR_2020_paper.pdf): tham khảo nguyên lý điều tiết nhiều kết quả cross-attention bằng gate. Thiết kế ở đây dùng hai nguồn fused visual/fact memory và một scalar gate trộn bổ sung, không sao chép nguyên kiến trúc của paper.

## 12. Triển khai và kiểm tra thực tế

Đã tạo `src/fact_fusion_dual_attention`, `script/fact_fusion_dual_attention/train.py`, `script/fact_fusion_dual_attention/generate_captions.py` và `tests/test_fact_fusion_dual_attention.py`. Không thay source các pipeline cũ hoặc shared.

- Dataset kế thừa shared Fact dataset; chỉ bổ sung index tập con và collate index/mask. `encode_memory()` dùng lại `encode_fusion()` và lấy đúng vectors sau projector bằng gather có gradient, sau đó chạy một adapter dùng chung.
- Decoder layer kế thừa `nn.TransformerDecoderLayer` để giữ self-attention, các norms và FFN cũ; thay khối cross-attention bằng hai nhánh có gate. LayerNorm của hai contexts chỉ dùng trong đầu vào MLP gate.
- Nhánh fact attention/gate chỉ chạy trên ảnh có facts bổ sung. Script train dùng `find_unused_parameters=True` như mục 7.
- Đã pass 11 test mới: chọn facts clip/itm và dedup provenance, tuple collation, fusion/CLS tương đương Fact, tập con projected vectors, adapter một lần, gate/norm/mix đúng thiết kế, đường ảnh khi thiếu facts, padding/permutation, causal mask, last-step logits, gradients qua hai updates trên cùng batch, beam routing, checkpoint/resume và script integration.
- Test script train/val/generate mỗi giai đoạn chỉ một batch hai ảnh, `NUM_WORKERS=0`, dùng tiny CLIP local. Kiểm tra cache tái sử dụng, CLIP đóng băng, JSON có imgid/cocoid/caption, scheduler global step và strict checkpoint loading.
- CUDA FP16 forward/backward và beam search trên một batch đã pass; gradients hữu hạn.
- DDP Gloo hai process CPU đã pass: gradient và SGD updates khớp reference batch gộp khi một rank thiếu facts, cả hai rank có facts và cả hai rank thiếu facts. Test dùng lại một batch tổng hợp nhỏ mỗi rank, không chạy full training.
- Đã chạy 24 test hồi quy của Fact/shared, fact bias và token bias. Một test script Fact cũ ban đầu import đường dẫn đã đổi tên `script.fact_rag`; đã cập nhật đúng hai import trong test thành `script.fact_fusion` và chạy lại test đó thành công. Không sửa scripts/models cũ.
- Đã kiểm tra parse/import các module mới và whitespace diff. Chưa chạy full dataset hoặc nhiều GPU vật lý; test DDP hai process CPU không thay thế xác nhận NCCL/multi-GPU thực tế.
