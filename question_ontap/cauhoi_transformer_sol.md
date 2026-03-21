# Giải Đáp Câu Hỏi Transformer

## 1. Câu nền tảng

**Transformer là gì?**
Kiến trúc mạng nơ-ron dùng self-attention thay cho recurrence, được giới thiệu trong paper "Attention Is All You Need" (2017). Xử lý toàn bộ chuỗi song song, học phụ thuộc xa tốt, trở thành nền tảng của hầu hết LLM và nhiều task NLP/vision hiện đại.

**Vì sao Transformer khác RNN/CNN truyền thống?**
RNN: xử lý tuần tự, phụ thuộc xa khó do vanishing gradient. CNN: xử lý cục bộ, receptive field tăng dần theo layer. Transformer: xử lý song song, mỗi token attend trực tiếp mọi token khác trong một bước — phụ thuộc xa ngay lập tức.

**Transformer mạnh ở điểm nào so với RNN?**
Song song hóa khi train (GPU-efficient), phụ thuộc xa không bị vanishing gradient, mỗi cặp vị trí chỉ cách nhau 1 bước attention.

**Encoder và decoder trong Transformer gốc làm gì?**
Encoder: xử lý input sequence, tạo contextual representations. Decoder: nhận encoder output, sinh output sequence autoregressive, có thêm cross-attention sang encoder.

**Một layer encoder gồm những khối nào?**
Multi-head self-attention → Add & LayerNorm → Feed-forward network (2-layer MLP) → Add & LayerNorm.

**Một layer decoder gồm những khối nào?**
Masked self-attention → Add & LayerNorm → Cross-attention (attend encoder) → Add & LayerNorm → Feed-forward → Add & LayerNorm.

**Vì sao residual connection và layer norm quan trọng?**
Residual: gradient highway, cho phép train mạng sâu. Layer norm: ổn định activation, tránh internal covariate shift, giúp training nhanh và ổn định hơn.

**Khi nào dùng encoder-only, decoder-only, encoder-decoder?**
Encoder-only (BERT): hiểu ngôn ngữ, classification, NER. Decoder-only (GPT): generation, language modeling. Encoder-decoder (T5, BART): translation, summarization — input và output khác nhau rõ ràng.

---

## 2. Self-attention

**Self-attention là gì?**
Cơ chế mỗi token trong chuỗi attend đến tất cả token khác (kể cả chính nó) để tạo contextual representation. Output mỗi token là weighted sum của Values của tất cả token, weights từ similarity với các token khác.

**Query, Key, Value là gì về trực giác?**
Query: "Tôi đang tìm kiếm gì?" Key: "Tôi có thể cung cấp thông tin gì?" Value: "Thông tin thực sự tôi cung cấp." Attention score = similarity(Q, K) → dùng để weighted-sum V.

**Scaled dot-product attention tính như thế nào?**
`Attention(Q, K, V) = softmax(QK^T / √d_k) V`. Nhân Q và K^T → scores → chia √d_k → softmax → weights → nhân với V → output.

**Vì sao phải chia cho √d_k?**
Khi `d_k` lớn, dot product Q·K có variance lớn → softmax bão hòa ở một vài giá trị rất cao → gradient gần 0. Chia √d_k → giữ variance ổn định → softmax cho phân phối mềm hơn → gradient tốt hơn.

**Self-attention giúp mô hình học phụ thuộc xa như thế nào?**
Mỗi token có thể attend trực tiếp đến bất kỳ token nào khác trong chuỗi trong một bước — không cần truyền thông tin qua nhiều bước như RNN.

**Vì sao nói self-attention cho "global dependencies"?**
Mọi cặp token (i, j) đều có thể tương tác trực tiếp qua attention, bất kể khoảng cách. Path length = O(1) trong khi RNN là O(n).

---

## 3. Multi-head attention

**Multi-head attention là gì?**
Chạy `h` attention heads song song, mỗi head có tập W_Q, W_K, W_V riêng, rồi concat kết quả: `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O`.

**Vì sao không chỉ dùng một attention head?**
Single head phải attend vào nhiều loại quan hệ khác nhau cùng lúc → bị "averaging". Multi-head: mỗi head chuyên chú ý một loại phụ thuộc (cú pháp, ngữ nghĩa, coreference...).

**Mỗi head học được điều gì khác nhau?**
Thực nghiệm cho thấy các head khác nhau học attend theo vị trí tương đối, theo quan hệ cú pháp, theo coreference... — mỗi head bắt một loại thông tin khác nhau.

**Vì sao nhiều head vẫn có chi phí gần tương đương một head full-dim?**
Mỗi head dùng dimension `d_k = d_model/h` thay vì `d_model` → chi phí mỗi head giảm `h` lần → tổng chi phí `h` heads tương đương 1 head full-dim.

---

## 4. Positional encoding

**Vì sao Transformer cần positional encoding?**
Self-attention là permutation-invariant — không quan tâm thứ tự token. Nếu không có positional info, mô hình không biết "từ thứ 1, 2, 3...". Phải inject thông tin vị trí vào.

**Nếu bỏ positional encoding thì mô hình mất gì?**
Mất thứ tự — câu "Mèo ăn cá" và "Cá ăn mèo" sẽ có representation giống nhau. Model không phân biệt được thứ tự token.

**Positional encoding được cộng vào embedding hay nối?**
Cộng vào (add): `input = token_embedding + positional_encoding`. Giữ nguyên chiều, đơn giản hơn concat.

**Sinusoidal positional encoding là gì?**
`PE(pos, 2i) = sin(pos/10000^{2i/d_model})`, `PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})`. Mỗi vị trí có pattern sin/cos ở nhiều tần số khác nhau — unique và có thể tổng quát sang chuỗi dài hơn lúc train.

**Learned vs fixed positional encoding khác nhau thế nào?**
Fixed (sinusoidal): không cần train thêm, có thể extrapolate sang sequence dài hơn. Learned: train như embedding thông thường — linh hoạt hơn nhưng không extrapolate tốt ngoài training length.

---

## 5. Encoder, decoder, và masking

**Self-attention trong encoder và decoder khác nhau ở đâu?**
Encoder: full self-attention — mỗi token attend đến tất cả token (cả trái và phải). Decoder: causal (masked) self-attention — mỗi token chỉ attend đến các token trước đó.

**Decoder vì sao phải dùng causal mask?**
Decoder sinh token autoregressive — khi sinh token `t`, chưa có token `t+1,..., T`. Nếu attend vào tương lai → "nhìn trước đáp án" khi train → model không học được gì.

**Encoder-decoder attention dùng query từ đâu, key/value từ đâu?**
Query từ decoder (vị trí đang sinh), Key và Value từ encoder output (toàn bộ câu nguồn). Decoder "hỏi" encoder "phần nào câu nguồn liên quan đến tôi đang sinh?".

**Causal masking bảo vệ tính autoregressive như thế nào?**
Mask bằng `-∞` các vị trí tương lai trước softmax → softmax output = 0 cho vị trí tương lai → không attend vào tương lai.

**Padding mask khác causal mask ở đâu?**
Padding mask: ẩn các token là padding (pad token không mang nghĩa — không nên attend vào). Causal mask: ẩn tương lai trong decoder để bảo toàn autoregressive. Cả hai có thể kết hợp.

---

## 6. Ưu điểm so với RNN/CNN

**Vì sao Transformer huấn luyện song song tốt hơn RNN?**
RNN phải xử lý tuần tự — step `t+1` đợi step `t`. Transformer xử lý toàn bộ chuỗi trong một bước attention — song song hóa trên GPU.

**Vì sao self-attention giúp đường truyền giữa hai token xa ngắn hơn?**
Hai token cách xa nhau vẫn interact trực tiếp qua attention trong 1 bước. RNN phải truyền qua O(n) steps. CNN với kernel `k` cần O(n/k) layers.

**Nhược điểm của self-attention trên chuỗi rất dài là gì?**
Chi phí tính toán O(n²) và O(n²) bộ nhớ theo độ dài chuỗi `n` — prohibitive với n = hàng chục nghìn tokens.

---

## 7. BERT, GPT, T5

**BERT thuộc họ encoder-only hay decoder-only?**
Encoder-only. Xử lý chuỗi với attention hai chiều (bidirectional) — mỗi token thấy tất cả token khác.

**Vì sao BERT được gọi là bidirectional Transformer?**
Self-attention trong BERT không masking — mỗi token attend cả trái lẫn phải → biểu diễn mỗi token phụ thuộc ngữ cảnh đầy đủ hai chiều.

**MLM của BERT là gì?**
Masked Language Modeling: che ngẫu nhiên 15% tokens, yêu cầu model dự đoán token bị che từ ngữ cảnh hai chiều. Học biểu diễn phong phú mà không cần nhãn.

**GPT-style model là decoder-only theo nghĩa nào?**
Chỉ dùng decoder với causal masking — mỗi token chỉ thấy các token trước. Sinh văn bản autoregressive từ trái sang phải.

**T5 khác BERT/GPT ở điểm nào?**
T5: encoder-decoder đầy đủ, format mọi task thành "text-to-text" (input text → output text). Linh hoạt với cả comprehension và generation tasks.

**"Text-to-text" trong T5 nghĩa là gì?**
Mọi task NLP (classification, translation, QA, summarization) đều được format thành: một text string làm input → một text string làm output. Dùng chung một kiến trúc và loss (cross-entropy).

**Khi nào chọn BERT, GPT, T5?**
BERT: hiệu ngôn ngữ, classification, NER, NLI — cần bidirectional context. GPT: generation, chatbot, code. T5: dịch thuật, summarization, QA — có cả input và output rõ ràng.

---

## 8. ViT

**Vision Transformer biến ảnh thành input sequence như thế nào?**
Chia ảnh thành các patch vuông nhỏ (ví dụ 16×16 pixel), flatten mỗi patch thành vector, áp linear projection → sequence of patch embeddings + positional encoding → đưa vào Transformer encoder chuẩn.

**Vì sao ViT chia ảnh thành patches?**
Ảnh có H×W pixel — quá dài để xử lý từng pixel (attention O(n²)). Patch giảm sequence length xuống còn `(H/P)×(W/P)` — manageable.

**ViT khác CNN ở inductive bias như thế nào?**
CNN: strong inductive bias về locality và translation equivariance — phù hợp ảnh với ít data. ViT: không có inductive bias locality — cần nhiều data hơn để học, nhưng linh hoạt hơn với data lớn.

**Khi nào ViT cần nhiều dữ liệu hơn CNN?**
Với dataset nhỏ-vừa, CNN thường tốt hơn nhờ inductive bias. ViT vượt CNN khi pretrain trên JFT-300M hay ImageNet-21k — cần rất nhiều data để học được spatial structure từ đầu.

---

## 9. Huấn luyện và thực hành

**Vì sao warmup learning rate hay đi cùng Transformer?**
Đầu training weights ngẫu nhiên → gradient lớn → LR lớn gây explode. Warmup: tăng LR từ 0 lên giá trị target từ từ → ổn định early training. Sau warmup giảm dần theo lịch trình.

**Vì sao context length dài làm chi phí attention tăng mạnh?**
Self-attention tính Q×K^T: `(n×d_k) × (d_k×n) = n²×d_k` operations. Context dài 2x → 4x compute cho attention.

**Vì sao inference của decoder-only model có thể chậm hơn training?**
Training: toàn bộ sequence song song (teacher forcing). Inference: autoregressive — sinh từng token một, phải đợi token trước xong. KV cache giúp nhưng vẫn tuần tự theo token.

**Vì sao Transformer rất hợp với pretraining rồi fine-tuning?**
Pretrain trên data khổng lồ → học rich representation. Fine-tune trên ít data task-specific → adapt nhanh. Cơ chế attention linh hoạt → transfer tốt giữa các task.

---

## 10. Hạn chế

**Vì sao attention chuẩn gặp khó với chuỗi rất dài?**
O(n²) compute và memory. Với n = 100K tokens: 10^10 operations — không khả thi. Cần Sparse Attention, Linear Attention, hoặc State Space Models.

**Vì sao Transformer ít inductive bias hơn CNN?**
CNN: local receptive field, weight sharing, translation equivariance — baked in. Transformer: không có bias về locality hay translation — phải học mọi thứ từ data.

**Khi dữ liệu ít, Transformer có thể bất lợi gì?**
Không có inductive bias → cần nhiều data để học cấu trúc mà CNN đã có sẵn. Với dataset nhỏ, ViT thường kém CNN. CNN strong prior giúp tổng quát hóa tốt hơn với ít data.

**Attention map có thật sự giải thích được mô hình không?**
Chỉ một phần. Attention weights cho biết token nào được "chú ý", nhưng không phải là gradient/attribution chính xác. Interpretability của attention vẫn là chủ đề tranh luận.

---

## 11. Phân biệt

**Self-attention vs cross-attention:** Self: Q, K, V cùng từ một chuỗi. Cross: Q từ decoder, K/V từ encoder — attend sang sequence khác.

**Single-head vs multi-head:** Single: một attention function. Multi: `h` attention heads song song → học nhiều loại phụ thuộc → representation phong phú hơn.

**Encoder-only vs decoder-only vs encoder-decoder:** Encoder: hiểu ngôn ngữ (BERT). Decoder: sinh text (GPT). Encoder-decoder: seq2seq có input và output rõ ràng (T5, BART).

**BERT vs GPT:** BERT: bidirectional, masked LM, classification tasks. GPT: causal (left-to-right), next-token prediction, generation.

**BERT vs T5:** BERT encoder-only, classification head trên top. T5 encoder-decoder, mọi task đều text-to-text.

**Causal mask vs padding mask:** Causal: ẩn tương lai trong decoder. Padding: ẩn pad tokens trong cả encoder lẫn decoder.

**Transformer vs RNN:** Transformer song song, O(1) path length giữa token xa, O(n²) memory. RNN tuần tự, O(n) path, O(1) memory per step.

**ViT vs CNN:** ViT: global attention, ít inductive bias, cần nhiều data. CNN: local receptive field, strong inductive bias, giỏi với ít data.

---

## 12. Câu tự luận

**Trình bày kiến trúc Transformer gốc:**
Encoder: N×(Multi-head Self-Attention + Add&Norm + FFN + Add&Norm). Decoder: N×(Masked Self-Attention + Add&Norm + Cross-Attention + Add&Norm + FFN + Add&Norm). Input: token embeddings + positional encoding. Output: linear + softmax trên vocabulary.

**Giải thích scaled dot-product attention và vì sao chia √d_k:**
`Attention = softmax(QK^T/√d_k)V`. Q×K^T tính similarity giữa mỗi cặp query-key. Chia √d_k vì dot product có std ∝ √d_k với random initialization → không chia thì variance lớn → softmax bão hòa → gradient vanish → không học được gì.

**Vì sao multi-head attention tốt hơn single-head:**
Single head trung bình hóa nhiều loại attention → mất thông tin tinh tế. Multi-head: mỗi head học loại attention riêng (cú pháp, ngữ nghĩa, coreference...) trong không gian con chiều thấp → representation giàu thông tin hơn với cùng compute.

**Vai trò của positional encoding:**
Transformer permutation-invariant về thứ tự token. Positional encoding thêm thông tin vị trí vào embedding: `input = embed(token) + pos_enc(position)`. Sinusoidal: `sin/cos` ở nhiều tần số → unique mỗi vị trí, có thể extrapolate. Learned: linh hoạt hơn nhưng không extrapolate tốt.

**So sánh BERT, GPT, và T5:**
BERT: encoder-only, bidirectional attention, pretrain với MLM+NSP — strong cho classification/NLU. GPT: decoder-only, causal attention, pretrain với LM — thiết kế cho generation và in-context learning. T5: encoder-decoder, text-to-text framework — unify mọi NLP task, mạnh cho translation/summarization/QA.

**Vì sao ViT có thể thay CNN trong một số bài toán ảnh:**
ViT bỏ inductive bias locality — với đủ data (ImageNet-21k, JFT-300M), học được representation tốt hơn CNN. Pretraining lớn → fine-tune tốt trên downstream tasks. Global attention từ layer 1 → nắm bắt long-range spatial dependencies ngay lập tức, trong khi CNN phải chồng nhiều lớp.
