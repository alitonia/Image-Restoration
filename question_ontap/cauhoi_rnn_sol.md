# Giải Đáp Câu Hỏi Recurrent Neural Networks

## 1. Câu nền tảng

**RNN là gì?**
Mạng nơ-ron có vòng lặp hồi quy: hidden state từ bước trước truyền lại làm input cho bước tiếp theo. Giúp xử lý dữ liệu tuần tự có độ dài thay đổi.

**Vì sao RNN phù hợp với dữ liệu tuần tự hơn MLP thường?**
MLP xử lý input kích thước cố định, không có bộ nhớ. RNN duy trì hidden state tích lũy thông tin qua thời gian → xử lý chuỗi có phụ thuộc thứ tự tự nhiên.

**Hidden state trong RNN là gì?**
Vector biểu diễn "bộ nhớ" tại thời điểm `t`. Được cập nhật tại mỗi bước: `h_t = f(h_{t-1}, x_t)`. Mang thông tin tích lũy từ các bước trước.

**"Chia sẻ tham số qua thời gian" nghĩa là gì?**
Cùng bộ tham số `W_h, W_x, b` dùng tại mọi time step — giảm số tham số và cho phép generalize qua vị trí.

**Nhiệm vụ many-to-one, one-to-many, many-to-many khác nhau thế nào?**
Many-to-one: chuỗi input → một output (sentiment). One-to-many: một input → chuỗi output (captioning). Many-to-many: chuỗi → chuỗi (translation), hoặc output tại mỗi bước (NER).

**RNN khác feedforward network ở điểm nào?**
Feedforward: không có memory, xử lý input độc lập. RNN: có hidden state hồi quy — thông tin từ bước trước ảnh hưởng bước sau.

---

## 2. Vanilla RNN

**Một bước cập nhật của vanilla RNN làm gì?**
`h_t = tanh(W_h h_{t-1} + W_x x_t + b)`. Kết hợp hidden state trước với input qua linear transformation rồi phi tuyến.

**"Unroll qua thời gian" nghĩa là gì?**
Mở rộng RNN thành đồ thị tính toán tuyến tính: mỗi time step là một bản copy của cell. Cho phép apply backpropagation qua các bước.

**RNN có thật sự "nhớ" toàn bộ quá khứ không?**
Về lý thuyết có, nhưng thực tế không. Thông tin bước xa bị "pha loãng" qua nhiều phép nhân → vanishing gradient → mất thông tin dài hạn.

**Vanilla RNN mạnh ở điểm nào?**
Đơn giản, ít tham số, hoạt động tốt với chuỗi ngắn và phụ thuộc cục bộ.

**Vanilla RNN yếu ở điểm nào?**
Vanishing gradient → không học được phụ thuộc dài hạn. Xử lý tuần tự → chậm.

---

## 3. BPTT, vanishing/exploding gradients

**BPTT là gì?**
Backpropagation Through Time: unroll RNN rồi áp backprop tiêu chuẩn. Gradient truyền ngược từ loss cuối qua tất cả time steps.

**Vì sao RNN dễ gặp vanishing gradient?**
Gradient nhân qua ma trận `W_h` tại mỗi bước ngược. Nếu singular values < 1 → gradient → 0 sau nhiều bước → không học được phụ thuộc xa.

**Vì sao RNN cũng có thể gặp exploding gradient?**
Nếu singular values > 1 → gradient blow up → training không ổn định, loss NaN.

**Gradient clipping dùng để làm gì?**
Giới hạn norm gradient: nếu `||g|| > threshold`, scale về threshold. Xử lý exploding gradient mà không cần thay đổi kiến trúc.

**Truncated BPTT là gì?**
Chỉ backpropagate qua `k` bước gần nhất. Giảm compute nhưng mất khả năng học phụ thuộc dài hơn `k`.

---

## 4. LSTM

**LSTM ra đời để giải quyết vấn đề gì?**
Vanishing gradient khiến vanilla RNN không học được phụ thuộc dài hạn. LSTM dùng cell state và gates để kiểm soát luồng thông tin, gradient chạy tốt hơn qua nhiều bước.

**Cell state trong LSTM là gì?**
Luồng thông tin "highway" chạy dọc chuỗi: `C_t`. Cập nhật bằng phép cộng → gradient chạy mà không bị nhân nhiều lần với `W_h`.

**Vì sao cell state giúp giữ thông tin dài hạn tốt hơn?**
Cập nhật: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t`. Gradient channel qua phép cộng → tránh vanishing.

**Forget gate làm gì?**
`f_t = σ(W_f [h_{t-1}, x_t] + b_f)`. Quyết định bao nhiêu `C_{t-1}` được giữ lại. `f_t ≈ 0`: quên, `f_t ≈ 1`: giữ nguyên.

**Input gate làm gì?**
`i_t = σ(...)` chọn vị trí cập nhật, `g_t = tanh(...)` tính giá trị mới → `C_t += i_t ⊙ g_t`. Thêm thông tin mới vào cell state.

**Output gate làm gì?**
`o_t = σ(...)`, `h_t = o_t ⊙ tanh(C_t)`. Lọc cell state ra hidden state — quyết định thông tin nào expose ra ngoài.

**Hidden state và cell state khác nhau thế nào?**
`C_t`: bộ nhớ dài hạn, highway gradient, không ra ngoài trực tiếp. `h_t`: output của bước, bộ nhớ ngắn hạn, được dùng làm input cho bước kế.

**Vì sao LSTM tốt hơn vanilla RNN trên chuỗi dài?**
Gradient chạy qua cell state highway, forget gate giữ thông tin quan trọng qua nhiều bước → không bị vanishing như vanilla RNN.

**Nhược điểm của LSTM là gì?**
Nhiều tham số hơn (4x so với vanilla RNN), chậm hơn. Vẫn tuần tự → không song song hóa theo time như Transformer.

---

## 5. GRU

**GRU là gì?**
Gated Recurrent Unit: biến thể đơn giản hơn LSTM với 2 gates (update + reset), không có cell state riêng. Ít tham số hơn nhưng thường hiệu quả tương đương LSTM.

**GRU khác LSTM ở điểm nào?**
LSTM: 3 gates + cell state riêng. GRU: 2 gates, merge cell và hidden state thành một. Đơn giản hơn, ít tham số hơn.

**Update gate có vai trò gì?**
Kết hợp chức năng forget + input gate: `z_t = σ(...)`. `h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t`. Quyết định giữ bao nhiêu quá khứ vs thêm bao nhiêu mới.

**Reset gate có vai trò gì?**
`r_t = σ(...)`. Kiểm soát bao nhiêu hidden state trước ảnh hưởng candidate: `h̃_t = tanh(W[r_t ⊙ h_{t-1}, x_t])`. `r_t ≈ 0`: bỏ qua quá khứ.

**Khi nào nên thử GRU trước LSTM?**
Khi muốn model nhẹ hơn, data ít, hoặc LSTM overfit. GRU thường cho kết quả tương đương với ít compute hơn.

---

## 6. Seq2seq

**Seq2seq là gì?**
Encoder RNN nén chuỗi nguồn thành context vector; decoder RNN giải mã ra chuỗi đích. Input và output có thể khác độ dài.

**Encoder làm gì, decoder làm gì?**
Encoder: đọc toàn chuỗi nguồn → context vector (last hidden state). Decoder: nhận context vector làm h_0, sinh từng token autoregressive.

**Bottleneck của context vector cố định là gì?**
Phải nén toàn bộ câu nguồn (dù dài bao nhiêu) vào một vector kích thước cố định → thông tin bị mất → câu dài dịch kém.

**Teacher forcing là gì?**
Khi train decoder, dùng token đúng thật sự của bước trước (không phải token mô hình sinh) làm input → training nhanh, ổn định hơn.

**Exposure bias là gì?**
Train dùng token thật → test dùng token tự sinh → phân phối khác → lỗi tích lũy khi sinh chuỗi dài.

---

## 7. Attention trong RNN

**Attention ra đời để giải quyết vấn đề gì?**
Bottleneck context vector cố định. Attention cho decoder "nhìn lại" tất cả encoder hidden states — weighted sum dynamic tại mỗi decode step.

**Context vector động khác context vector cố định thế nào?**
Cố định: one vector cho mọi decode step. Động: `c_t = Σ α_{t,i} h_i^{enc}` — khác nhau ở mỗi bước, focus vào phần liên quan.

**Vì sao attention giúp dịch câu dài tốt hơn?**
Decoder focus vào đúng phần của câu nguồn khi sinh mỗi token đích → không bị giới hạn bởi vector nén.

**Attention từng là bước ngoặt lớn của mô hình chuỗi vì sao?**
Giải quyết bottleneck cơ bản của seq2seq, cải thiện dịch câu dài, và cơ chế attention sau đó là nền tảng của Transformer.

---

## 8. Regularization và thực hành

**Dropout trong RNN khó hơn feedforward network ở điểm nào?**
Áp dropout vào recurrent connections mỗi bước khác mask → phá gradient flow. Cần dùng cùng mask xuyên suốt sequence (variational dropout), hoặc chỉ apply vào non-recurrent connections.

**Padding và masking để làm gì?**
Chuỗi trong batch khác độ dài → pad về cùng chiều dài. Mask để model biết đâu là padding, không tính loss trên token pad.

**Vì sao inference của RNN chậm hơn Transformer?**
RNN phải xử lý tuần tự — bước `t+1` đợi bước `t` xong. Transformer có thể song song hóa toàn bộ sequence khi train.

**Perplexity là gì trong language modeling?**
`PPL = exp(average cross-entropy)`. Perplexity thấp = mô hình dự đoán token tiếp theo tốt và tự tin hơn.

---

## 9. Phân biệt

**Vanilla RNN vs LSTM:** Vanilla đơn giản nhưng vanishing gradient. LSTM có gates + cell state → học phụ thuộc dài hạn tốt hơn.

**LSTM vs GRU:** LSTM 3 gates + cell state riêng. GRU 2 gates, gọn hơn. Hiệu quả thường tương đương — GRU dùng khi muốn nhẹ hơn.

**Hidden state vs cell state:** `h_t` output mỗi bước. `C_t` (LSTM) bộ nhớ dài hạn không output trực tiếp.

**Vanishing vs exploding gradient:** Vanishing: singular values W < 1 → gradient → 0. Exploding: > 1 → gradient → ∞. Xử lý: LSTM/GRU (vanishing), gradient clipping (exploding).

**Seq2seq cơ bản vs +attention:** Cơ bản: context vector cố định → bottleneck. +Attention: context vector động → không bottleneck, câu dài tốt hơn.

**Teacher forcing vs free running:** Teacher forcing: feed token thật → stable training. Free running: feed token sinh → sát inference nhưng khó train.

---

## 10. Câu tự luận

**Cơ chế hoạt động của vanilla RNN:**
Tại mỗi step `t`: `h_t = tanh(W_h h_{t-1} + W_x x_t + b)`. Output: `y_t = W_o h_t`. Cùng tham số mọi step. Thông tin tích lũy trong `h_t`. Hạn chế: vanishing gradient khiến thông tin xa không học được.

**Vì sao RNN gặp vanishing/exploding gradient:**
BPTT nhân gradient qua `W_h` nhiều lần liên tiếp → `W_h^k`. Singular values < 1: vanish; > 1: explode. Chuỗi càng dài vấn đề càng nặng.

**Vì sao LSTM khắc phục phụ thuộc dài hạn tốt hơn:**
Cell state highway: cập nhật bằng phép cộng thay vì nhân → gradient không vanish qua nhiều bước. Forget gate có thể ≈ 1 để gradient chảy nguyên vẹn. Input gate chọn lọc thông tin.

**So sánh LSTM và GRU:**
LSTM: 3 gates, cell state riêng, nhiều tham số, mạnh hơn với chuỗi rất dài. GRU: 2 gates, không cell state riêng, gọn hơn, thường đủ tốt. Dùng GRU trước khi cần LSTM.

**Seq2seq cho dịch máy:**
Encoder LSTM đọc câu nguồn → last hidden state = context vector. Decoder LSTM nhận context vector làm h_0, sinh câu đích token by token với teacher forcing khi train. Hạn chế bottleneck → thêm attention.

**Vì sao attention cải thiện encoder-decoder RNN:**
`c_t = Σ α_{t,i} h_i^{enc}` với attention weights `α` học được. Decoder focus vào đúng phần câu nguồn khi sinh mỗi token đích → không bị bottleneck → câu dài chính xác hơn → alignment interpretable.
